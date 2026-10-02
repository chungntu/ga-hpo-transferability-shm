"""
Stability check for the repaired pipeline before committing to it.

Best combination found so far (Z24 small-scale, leakage-free record split):
    window 2048, per_window normalisation, 1D-CNN  ->  test 0.753
against PSD+logreg 0.720 and chance 0.200.

That is a single seed. Here the SPLIT seed is varied, so each seed reassigns
which measurement setups land in train/val/test -- the quantity the paper's
Table 4/6 claimed to average over but never did. Five seeds, mean and 95% CI
from the t-distribution, exactly the formula printed as equation (3) in the
paper.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  _os.pardir, _os.pardir, "src"))
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json, os
import numpy as np
from cnn1d_torch import CNN1D
from runlog import log_run, new_run_id
from shm_data import load_records, build_splits, Z24_CLASSES_5, provenance_digest
from wavenet_torch import train_eval

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
SEEDS = [0, 1, 2, 3, 4]
WINDOW, EPOCHS, LR = 2048, 40, 1e-3


def cnn_factory(n_classes, width):
    return CNN1D(n_classes, filters=16, n_blocks=4, kernel_size=9, width=width)


def ci95(v):
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return 0.0
    from scipy import stats
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def main():
    records = load_records("z24", Z24_CLASSES_5, "5")
    rows = []
    for seed in SEEDS:
        data = build_splits(records, seed=seed, window_length=WINDOW,
                            split_scheme="record", scaler="per_window", verbose=False)
        digest = provenance_digest(data)
        n_sensors = len({m["sensor"] for m in data.meta["train"]})
        m, ex = train_eval(data, lr=LR, epochs=EPOCHS, seed=seed, verbose=0,
                           batch_size=64, early_stop_flag=0, model_fn=cnn_factory)
        log_run(dict(
            run_id=new_run_id("diag"), experiment="diag_seeds",
            dataset_train="z24_small", dataset_eval="z24_small",
            n_classes=data.n_classes, n_sensors=n_sensors, window_length=WINDOW,
            split_scheme="record", split_digest=digest,
            config_source="cnn1d_per_window_w2048",
            notes="split-seed stability of the repaired pipeline",
            **{k: m[k] for k in ("lr","filters","res_per_block","n_blocks",
               "epochs","epochs_run","batch_size","head","n_params","seed",
               "val_acc","val_acc_best","test_acc","test_macro_f1",
               "test_balanced_acc","stopped_early","train_time_sec","gpu_name")},
        ), ex)
        rows.append(dict(seed=seed, split_digest=digest,
                         val_acc=m["val_acc"], test_acc=m["test_acc"],
                         test_macro_f1=m["test_macro_f1"],
                         test_balanced_acc=m["test_balanced_acc"]))
        print(f"seed {seed}: val {m['val_acc']:.3f}  test {m['test_acc']:.3f}  "
              f"macroF1 {m['test_macro_f1']:.3f}  (split {digest})", flush=True)

    with open(os.path.join(HERE,"results","diag_seeds.json"),"w",encoding="utf-8") as f:
        json.dump(dict(seeds=SEEDS, window=WINDOW, epochs=EPOCHS, lr=LR, rows=rows), f, indent=2)

    print("\n" + "="*60)
    for key in ("test_acc", "test_macro_f1", "test_balanced_acc"):
        v = [r[key] for r in rows]
        print(f"{key:>20}: {np.mean(v):.3f} +/- {ci95(v):.3f}  "
              f"(min {min(v):.3f}, max {max(v):.3f})")


if __name__ == "__main__":
    main()
