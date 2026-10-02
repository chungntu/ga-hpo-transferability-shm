"""
Diagnostic: is the record-level Z24 small-scale task learnable by WaveNet at
all, and if so what does it need?

After the leakage fix, every configuration tried so far sits at chance (0.20
on 5 classes) while a PSD + logistic-regression baseline on the SAME split
reaches ~0.71 test accuracy. So the signal is there and the network is not
using it. Two suspects, both named in HANDOFF_REVISION.md:

  * the 15-epoch budget (section 2.5) was chosen while validation was
    contaminated, so it has no basis once validation is independent;
  * the Flatten -> Dense head (section 5.3) carries tens of millions of
    parameters and memorises the training set.

This crosses them: {15, 100} epochs x {flatten, gap}, one configuration
(the paper's Table 3 Small-scale winner), one seed. Early stopping is off:
the <0.2-after-3-epochs rule would kill every run at chance before it had a
chance to move.

Results go to results/diag_budget_head.json and into runs.csv like any other
run, so nothing here is off the record.
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  _os.pardir, _os.pardir, "src"))

import sys
# Windows consoles default to cp1252; project paths contain Vietnamese text.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json
import os

from runlog import log_run, new_run_id
from shm_data import load_dataset, provenance_digest
from wavenet_torch import train_eval

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root

# paper Table 3, Small-scale winner
CONFIG = dict(lr=7e-5, filters=4, res_per_block=8, n_blocks=2)
DATASET = "z24_small"
SEED = 0
GRID = [(15, "flatten"), (100, "flatten"), (15, "gap"), (100, "gap")]


def main():
    data = load_dataset(DATASET, seed=SEED)
    digest = provenance_digest(data)
    n_sensors = len({m["sensor"] for m in data.meta["train"]})
    rows = []

    for epochs, head in GRID:
        print(f"\n=== epochs={epochs} head={head} ===", flush=True)
        metrics, extras = train_eval(
            data, epochs=epochs, head=head, seed=SEED, verbose=0,
            early_stop_flag=0, **CONFIG)

        log_run(dict(
            run_id=new_run_id("diag"),
            experiment="diag_budget_head",
            dataset_train=DATASET, dataset_eval=DATASET,
            n_classes=data.n_classes, n_sensors=n_sensors,
            window_length=data.width,
            split_scheme=data.meta["split_scheme"], split_digest=digest,
            config_source="paper_table3_small",
            notes=f"diagnostic: budget x head, early stopping disabled",
            **{k: metrics[k] for k in (
                "lr", "filters", "res_per_block", "n_blocks", "epochs",
                "epochs_run", "batch_size", "head", "n_params", "seed",
                "val_acc", "val_acc_best", "test_acc", "test_macro_f1",
                "test_balanced_acc", "stopped_early", "train_time_sec",
                "gpu_name")},
        ), extras)

        hist = extras["history"]
        final_train_acc = hist[-1]["acc"]
        print(f"  params {metrics['n_params']:,}"
              f"  train_acc {final_train_acc:.3f}"
              f"  val {metrics['val_acc']:.3f} (best {metrics['val_acc_best']:.3f})"
              f"  test {metrics['test_acc']:.3f}"
              f"  macroF1 {metrics['test_macro_f1']:.3f}"
              f"  {metrics['train_time_sec']:.0f}s", flush=True)
        rows.append(dict(epochs=epochs, head=head,
                         n_params=metrics["n_params"],
                         final_train_acc=final_train_acc,
                         val_acc=metrics["val_acc"],
                         val_acc_best=metrics["val_acc_best"],
                         test_acc=metrics["test_acc"],
                         test_macro_f1=metrics["test_macro_f1"],
                         train_time_sec=metrics["train_time_sec"],
                         val_curve=[h["val_acc"] for h in hist],
                         train_curve=[h["acc"] for h in hist]))

    out = os.path.join(HERE, "results", "diag_budget_head.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(dataset=DATASET, seed=SEED, config=CONFIG,
                       split_digest=digest, rows=rows), fh, indent=2)

    print("\n" + "=" * 68)
    print(f"{'epochs':>7} {'head':>8} {'params':>12} {'train':>7} "
          f"{'val':>7} {'best':>7} {'test':>7} {'macroF1':>8}")
    for r in rows:
        print(f"{r['epochs']:>7} {r['head']:>8} {r['n_params']:>12,} "
              f"{r['final_train_acc']:>7.3f} {r['val_acc']:>7.3f} "
              f"{r['val_acc_best']:>7.3f} {r['test_acc']:>7.3f} "
              f"{r['test_macro_f1']:>8.3f}")
    print(f"\nchance level = {1/data.n_classes:.3f};  "
          f"PSD+logreg on this split = 0.711 test")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
