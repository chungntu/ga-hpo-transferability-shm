"""
Fourth diagnostic: does a conventional 1D-CNN on short windows generalise
across records, where the WaveNet port does not?

State of the evidence on the leakage-free Z24 small-scale split:
    grid, 72 configs, 15 ep    47/72 exactly at chance, corr(val,test) = -0.50
    WaveNet 200 ep, lr 1e-3    train 1.000  test 0.222
    WaveNet window 8192        train 0.910  test 0.174   (8x more samples)
    PSD + logistic regression  test 0.667 - 0.720
So: information present, WaveNet memorises records and transfers nothing.

Two candidate causes are crossed here.

  model    WaveNet (flatten at full length) vs 1D-CNN (conv+pool, then GAP)
  scaler   'global'     keeps per-setup amplitude differences, which the
                        network can use to identify the SETUP rather than
                        the damage class -- useless on an unseen setup;
           'per_window' removes amplitude entirely (instance normalisation).

Both scalers are leakage-free. If per_window alone closes the gap, the
finding is about normalisation; if the CNN alone does, it is about
architecture; if neither, raw-waveform models are the wrong tool here and
that itself is the answer.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  _os.pardir, _os.pardir, "src"))
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json, os
from functools import partial

from cnn1d_torch import CNN1D
from runlog import log_run, new_run_id
from shm_data import load_records, build_splits, Z24_CLASSES_5, provenance_digest
from wavenet_torch import train_eval

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
WINDOW = 2048
EPOCHS = 40
LR = 1e-3
SCALERS = ["global", "per_window"]


def cnn_factory(n_classes, width):
    return CNN1D(n_classes, filters=16, n_blocks=4, kernel_size=9, width=width)


def main():
    records = load_records("z24", Z24_CLASSES_5, "5")
    rows = []
    for scaler in SCALERS:
        data = build_splits(records, seed=0, window_length=WINDOW,
                            split_scheme="record", scaler=scaler, verbose=False)
        digest = provenance_digest(data)
        n_sensors = len({m["sensor"] for m in data.meta["train"]})
        for model_name in ("cnn1d", "wavenet"):
            print(f"\n=== {model_name} | scaler={scaler} | window={WINDOW} "
                  f"({data.X_train.shape[0]} train) ===", flush=True)
            kw = dict(model_fn=cnn_factory) if model_name == "cnn1d" else dict(
                filters=8, res_per_block=8, n_blocks=2, head="flatten")
            m, ex = train_eval(data, lr=LR, epochs=EPOCHS, seed=0, verbose=0,
                               batch_size=64, early_stop_flag=0, **kw)
            log_run(dict(
                run_id=new_run_id("diag"), experiment="diag_cnn1d",
                dataset_train="z24_small", dataset_eval="z24_small",
                n_classes=data.n_classes, n_sensors=n_sensors,
                window_length=WINDOW, split_scheme="record", split_digest=digest,
                config_source=f"{model_name}_{scaler}",
                notes=f"model/scaler probe; {model_name}; scaler={scaler}",
                **{k: m[k] for k in ("lr","filters","res_per_block","n_blocks",
                   "epochs","epochs_run","batch_size","head","n_params","seed",
                   "val_acc","val_acc_best","test_acc","test_macro_f1",
                   "test_balanced_acc","stopped_early","train_time_sec","gpu_name")},
            ), ex)
            h = ex["history"]
            rows.append(dict(model=model_name, scaler=scaler,
                             n_params=m["n_params"], final_train_acc=h[-1]["acc"],
                             val_acc=m["val_acc"], val_acc_best=m["val_acc_best"],
                             test_acc=m["test_acc"], test_macro_f1=m["test_macro_f1"],
                             test_balanced_acc=m["test_balanced_acc"],
                             train_time_sec=m["train_time_sec"]))
            r = rows[-1]
            print(f"  params {r['n_params']:,}  train {r['final_train_acc']:.3f}"
                  f"  val {r['val_acc']:.3f}  test {r['test_acc']:.3f}"
                  f"  macroF1 {r['test_macro_f1']:.3f}  {r['train_time_sec']:.0f}s",
                  flush=True)

    with open(os.path.join(HERE,"results","diag_cnn1d.json"),"w",encoding="utf-8") as f:
        json.dump(dict(window=WINDOW, epochs=EPOCHS, lr=LR, rows=rows), f, indent=2)

    print("\n" + "="*78)
    print(f"{'model':>9} {'scaler':>12} {'params':>10} {'train':>7} {'val':>7} "
          f"{'test':>7} {'macroF1':>8}")
    for r in rows:
        print(f"{r['model']:>9} {r['scaler']:>12} {r['n_params']:>10,} "
              f"{r['final_train_acc']:>7.3f} {r['val_acc']:>7.3f} "
              f"{r['test_acc']:>7.3f} {r['test_macro_f1']:>8.3f}")
    print("chance 0.200 | PSD+logreg at window 2048: 0.720 test")


if __name__ == "__main__":
    main()
