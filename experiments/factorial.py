"""
Controlled factorial experiment: number of classes vs number of sensors.

The submitted paper attributes transfer failure to "input complexity", defined
as the number of damage classes AND sensors AND the signal length, all varied
together with the dataset identity. Nothing can be attributed to any one of
them: the Small-scale/Full-scale contrast changes every factor at once, so
"complexity" as used in the paper is a bundle, not a variable.

This varies them ONE AT A TIME from a common baseline, on a single dataset, so
each factor gets its own effect:

    baseline           5 classes,  5 channels
    classes arm        5 / 10 / 15 classes at 5 channels
    channels arm       5 / 15 / 33 channels at 5 classes

Signal length is held at 2048 everywhere, and the dataset is held at Z24, so
neither can confound the comparison. Five cells in total (the baseline is
shared by both arms), each a full 72-configuration grid, which gives a complete
response surface per cell and therefore rank correlations and regrets between
cells -- the same analysis as the cross-dataset case but with a single factor
moving.

Channels are taken as the first N of the 33 in each setup file, for ALL cells
including the 5-channel one. That is deliberate: the z24_small preset uses the
five sensors that were fixed across setups (a different set of five), and
mixing the two definitions inside a factorial would reintroduce exactly the
confounding this experiment exists to remove. The difference is noted with the
results.

Usage:
    python factorial.py --model cnn1d --seed 0
    python factorial.py --model cnn1d --seed 0 --arm classes
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  _os.pardir, "src"))
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import argparse
import json
import os
import time

from search import SEARCH_SPACES, Evaluator, all_configs
from shm_data import Z24_CLASSES_15, build_splits, load_records

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
WINDOW, SCALER, EPOCHS = 2048, "per_window", 40

# (label, n_classes, n_channels)
CELLS = [
    ("c05_s05", 5, 5),      # baseline, shared by both arms
    ("c10_s05", 10, 5),     # classes arm
    ("c15_s05", 15, 5),
    ("c05_s15", 5, 15),     # channels arm
    ("c05_s33", 5, 33),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn1d", choices=list(SEARCH_SPACES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="both",
                    choices=["both", "classes", "channels"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args()

    cells = CELLS
    if a.arm == "classes":
        cells = [c for c in CELLS if c[2] == 5]
    elif a.arm == "channels":
        cells = [c for c in CELLS if c[1] == 5]

    for label, n_classes, n_channels in cells:
        out = os.path.join(HERE, "results",
                           f"factorial_{a.model}_{label}_seed{a.seed}.json")
        if os.path.exists(out):
            print(f"[skip] {os.path.basename(out)}", flush=True)
            continue

        records = load_records("z24", Z24_CLASSES_15[:n_classes], "all",
                               max_sensors=n_channels)
        data = build_splits(records, seed=a.seed, window_length=WINDOW,
                            split_scheme="record", scaler=SCALER,
                            stride=WINDOW * 4, verbose=False)
        print(f"\n=== {label}: {n_classes} classes x {n_channels} channels  "
              f"train {data.X_train.shape[0]:,} / test {data.X_test.shape[0]:,} ===",
              flush=True)

        ev = Evaluator(data, f"z24_{label}", f"factorial_{a.model}_{label}",
                       a.seed, model=a.model, epochs=a.epochs, scaler=SCALER)
        t0 = time.perf_counter()
        for cfg in all_configs(a.model):
            ev(cfg, config_source="factorial")
        elapsed = time.perf_counter() - t0

        best_key, best_m = max(ev.cache.items(), key=lambda kv: kv[1]["val_acc"])
        payload = dict(strategy="grid", model=a.model, dataset=f"z24_{label}",
                       seed=a.seed, epochs=a.epochs, window=WINDOW,
                       scaler=SCALER, n_classes=n_classes,
                       n_channels=n_channels, factorial_label=label,
                       unique_evaluations=ev.n_evals, wall_time_sec=elapsed,
                       split_digest=ev.digest,
                       best_config=dict(zip(list(SEARCH_SPACES[a.model]), best_key)),
                       best_val_acc=best_m["val_acc"],
                       best_test_acc=best_m["test_acc"],
                       best_test_macro_f1=best_m["test_macro_f1"],
                       trace=ev.trace, ga_history=None)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  best {payload['best_config']}  val {best_m['val_acc']:.3f} "
              f"test {best_m['test_acc']:.3f}  ({elapsed/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
