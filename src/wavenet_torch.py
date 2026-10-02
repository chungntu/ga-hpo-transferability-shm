"""
PyTorch port of WaveNet.py (Keras) — the original TF build has no GPU support
on Windows, see HANDOFF_REVISION.md section 5.3.

The port is layer-for-layer faithful to build_wavenet_model_residual_blocks:

    causal Conv1d(1 -> F, k=2, dil=1) + relu
    for i in range(blocks * res):
        d = 2 ** (i % res)
        tanh(causal conv) * sigmoid(causal conv)      # gated activation
        conv 1x1                                       # -> skip
        + input                                        # -> residual
    relu(sum of skips) -> conv1x1+relu -> conv1x1+relu -> dense(F)+relu
    -> flatten -> dense(n_classes)

`head='gap'` swaps the Flatten for GlobalAveragePooling1D. That is an
architectural change (it removes the tens-of-millions-parameter final Dense
that almost certainly drives the train-accuracy-1.000 behaviour), so it is
NOT the default; switching it on means re-running everything.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__(in_ch, out_ch, kernel_size, dilation=dilation)
        self._pad = (kernel_size - 1) * dilation

    def forward(self, x):
        if self._pad:
            x = F.pad(x, (self._pad, 0))
        return super().forward(x)


class WaveNetResidual(nn.Module):
    def __init__(self, filters, kernel_size, dilation):
        super().__init__()
        self.tanh_conv = CausalConv1d(filters, filters, kernel_size, dilation)
        self.sigm_conv = CausalConv1d(filters, filters, kernel_size, dilation)
        self.mix = nn.Conv1d(filters, filters, 1)

    def forward(self, x):
        z = torch.tanh(self.tanh_conv(x)) * torch.sigmoid(self.sigm_conv(x))
        skip = self.mix(z)
        return skip + x, skip


class WaveNet(nn.Module):
    def __init__(self, n_classes, filters, n_blocks, res_per_block,
                 width, kernel_size=2, head="flatten", in_channels=1,
                 init="keras"):
        super().__init__()
        self.head_kind = head
        self.first = CausalConv1d(in_channels, filters, kernel_size, 1)
        self.blocks = nn.ModuleList([
            WaveNetResidual(filters, kernel_size, 2 ** (i % res_per_block))
            for i in range(n_blocks * res_per_block)
        ])
        self.post1 = nn.Conv1d(filters, filters, 1)
        self.post2 = nn.Conv1d(filters, filters, 1)
        self.dense = nn.Conv1d(filters, filters, 1)     # Keras Dense over channels
        if head == "flatten":
            self.out = nn.Linear(filters * width, n_classes)
        elif head == "gap":
            self.out = nn.Linear(filters, n_classes)
        else:
            raise ValueError(f"head must be 'flatten' or 'gap', got {head!r}")
        if init == "keras":
            keras_init_(self)
        elif init != "torch":
            raise ValueError(f"init must be 'keras' or 'torch', got {init!r}")

    def forward(self, x):                # x: (N, 1, T)
        x = F.relu(self.first(x))
        skips = None
        for blk in self.blocks:
            x, skip = blk(x)
            skips = skip if skips is None else skips + skip
        h = F.relu(skips)
        h = F.relu(self.post1(h))
        h = F.relu(self.post2(h))
        h = F.relu(self.dense(h))
        if self.head_kind == "gap":
            h = h.mean(dim=2)
        else:
            h = h.flatten(1)
        return self.out(h)               # logits


def keras_init_(module):
    """Match Keras defaults: glorot_uniform weights, zero biases.

    PyTorch initialises Conv1d and Linear with kaiming_uniform(a=sqrt(5)), which
    is scaled for ReLU. This network stacks 16-20 gated tanh*sigmoid layers, where
    that scaling is wrong enough to stop it learning at full window length, while
    the Keras original -- glorot_uniform, i.e. Xavier -- trains. Porting the
    architecture without porting the initialiser is not porting the model.
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return module


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------
# training
# ----------------------------------------------------------------------

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _evaluate(model, X, y, device, batch_size, amp):
    model.eval()
    n = X.shape[0]
    correct, loss_sum = 0, 0.0
    preds = np.empty(n, dtype=np.int64)
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size].to(device, non_blocking=True)
        yb = y[i:i + batch_size].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(xb)
            loss = F.cross_entropy(logits.float(), yb)
        p = logits.argmax(1)
        preds[i:i + xb.shape[0]] = p.cpu().numpy()
        correct += (p == yb).sum().item()
        loss_sum += loss.item() * xb.shape[0]
    return correct / n, loss_sum / n, preds


def train_eval(data, lr, filters=None, res_per_block=None, n_blocks=None,
               epochs=15, batch_size=16, seed=0, device="cuda",
               head="flatten", amp=True, verbose=1,
               early_stop_flag=1, early_stop_threshold=0.2, early_stop_min_epochs=3,
               model_fn=None, init="keras"):
    """
    Train one configuration on a SplitData and return a metrics dict.

    Fitness convention kept from the paper: val accuracy at the LAST epoch.
    `val_acc_best` is reported alongside so both are on record.

    `model_fn` swaps in another base model: a callable taking
    (n_classes, width) and returning an nn.Module. The WaveNet port is used
    when it is None.
    """
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    def as_tensor(X, y):
        # (N, T, 1) -> (N, 1, T)
        return (torch.from_numpy(np.ascontiguousarray(X.transpose(0, 2, 1))),
                torch.from_numpy(y))

    Xtr, ytr = as_tensor(data.X_train, data.y_train)
    Xva, yva = as_tensor(data.X_val, data.y_val)
    Xte, yte = as_tensor(data.X_test, data.y_test)

    if model_fn is None:
        model = WaveNet(data.n_classes, filters, n_blocks, res_per_block,
                        data.width, head=head, init=init).to(device)
    else:
        model = model_fn(data.n_classes, data.width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = Xtr.shape[0]
    g = torch.Generator().manual_seed(seed)
    hist = []
    best_val = -math.inf
    t0 = time.perf_counter()
    stopped_early = False

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        run_loss, run_correct = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xtr[idx].to(device, non_blocking=True)
            yb = ytr[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(xb)
                loss = F.cross_entropy(logits.float(), yb)
            loss.backward()
            opt.step()
            run_loss += loss.item() * xb.shape[0]
            run_correct += (logits.argmax(1) == yb).sum().item()

        va_acc, va_loss, _ = _evaluate(model, Xva, yva, device, batch_size, amp)
        best_val = max(best_val, va_acc)
        hist.append(dict(epoch=ep, loss=run_loss / n, acc=run_correct / n,
                         val_loss=va_loss, val_acc=va_acc))
        if verbose:
            print(f"  ep {ep:3d}/{epochs}  loss {run_loss/n:.4f}  acc {run_correct/n:.4f}"
                  f"  val_loss {va_loss:.4f}  val_acc {va_acc:.4f}", flush=True)

        if early_stop_flag and ep >= early_stop_min_epochs and best_val < early_stop_threshold:
            if verbose:
                print(f"  [early stop] best val_acc {best_val:.4f} < {early_stop_threshold}",
                      flush=True)
            stopped_early = True
            break

    train_time = time.perf_counter() - t0

    va_acc, _, va_pred = _evaluate(model, Xva, yva, device, batch_size, amp)
    te_acc, _, te_pred = _evaluate(model, Xte, yte, device, batch_size, amp)

    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                                 f1_score, recall_score)
    y_true = data.y_test
    metrics = dict(
        lr=lr, filters=filters, res_per_block=res_per_block, n_blocks=n_blocks,
        epochs=epochs, epochs_run=len(hist), batch_size=batch_size, seed=seed,
        head=head, n_params=count_params(model),
        val_acc=va_acc, val_acc_best=best_val,
        test_acc=te_acc,
        test_macro_f1=float(f1_score(y_true, te_pred, average="macro", zero_division=0)),
        test_balanced_acc=float(balanced_accuracy_score(y_true, te_pred)),
        train_time_sec=train_time,
        stopped_early=stopped_early,
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    )
    extras = dict(
        history=hist,
        confusion_matrix=confusion_matrix(y_true, te_pred).tolist(),
        per_class_recall=recall_score(y_true, te_pred, average=None,
                                      zero_division=0).tolist(),
        test_pred=te_pred.tolist(), val_pred=va_pred.tolist(),
    )
    del model
    torch.cuda.empty_cache()
    return metrics, extras
