"""
Leave-one-setup-out cross-validation on Z24.

A single record-level split of Z24 small-scale puts exactly ONE measurement
setup in validation and two in test, out of nine. The 5-seed check in
REVISION_NOTES.md section 5.2 gave test macro-F1 0.668 +/- 0.101 -- the
interval is wide because each seed evaluates on a different pair of setups,
not because the model is unstable. No confidence interval computed from a
single split can fix that; the estimator itself is noisy.

LOSO CV uses every setup exactly once as the test set, so the reported mean is
over all nine held-out setups instead of a random two, and the spread has a
clear meaning: how much performance depends on WHICH setup is unseen. That is
the quantity an engineer deploying to a new sensor configuration actually
cares about.

The configuration used is the one selected by the campaign grid on the same
scenario; pass --config to override.

Usage:
    python loso_cv.py --model cnn1d --classes 5
    python loso_cv.py --model wavenet --classes 15
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
import glob
import json
import os

import numpy as np
from scipy import stats

from runlog import log_run, new_run_id
from search import SEARCH_SPACES, build_kwargs, genes
from shm_data import (Z24_CLASSES_5, Z24_CLASSES_15, build_splits,
                      load_records, provenance_digest)
from wavenet_torch import train_eval

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
EPOCHS, BATCH, WINDOW, SCALER = 40, 64, 2048, "per_window"


def best_config_from_grid(model, dataset):
    """Configuration selected by validation accuracy in the campaign grids."""
    paths = sorted(glob.glob(os.path.join(
        HERE, "results", f"search_grid_{model}_{dataset}_seed*.json")))
    if not paths:
        return None
    # pick the configuration with the highest mean validation accuracy across seeds
    tally = {}
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        for t in d["trace"]:
            key = tuple(t["config"][g] for g in genes(model))
            tally.setdefault(key, []).append(t["fitness"])
    best = max(tally, key=lambda k: np.mean(tally[k]))
    return dict(zip(genes(model), best))


def ci95(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn1d", choices=list(SEARCH_SPACES))
    ap.add_argument("--classes", type=int, default=5, choices=[5, 15])
    ap.add_argument("--sensors", default=None,
                    help="z24 sensor mode: '5' (fixed five) or 'all'")
    ap.add_argument("--folds", type=int, default=9)
    ap.add_argument("--config", default=None, help="JSON dict to override")
    ap.add_argument("--multichannel", action="store_true",
                    help="one sample per (record, window) carrying every channel")
    ap.add_argument("--stride-mult", type=int, default=1,
                    help="window stride as a multiple of the window length; "
                         "use >1 on z24_full, which otherwise yields 92k windows")
    a = ap.parse_args()

    dataset = "z24_small" if a.classes == 5 else "z24_full"
    sensors = a.sensors or ("5" if a.classes == 5 else "all")
    classes = Z24_CLASSES_5 if a.classes == 5 else Z24_CLASSES_15

    grid_ds = dataset + ("_mc" if a.multichannel else "")
    cfg = (json.loads(a.config) if a.config
           else best_config_from_grid(a.model, grid_ds)
           or best_config_from_grid(a.model, dataset))
    if cfg is None:
        raise SystemExit(f"no grid results for {a.model}/{dataset}; pass --config")
    print(f"[loso] {a.model} on {dataset} ({sensors} sensors), config {cfg}", flush=True)

    records = load_records("z24", classes, sensors)
    rows = []
    for fold in range(a.folds):
        data = build_splits(records, window_length=WINDOW, split_scheme="loso",
                            fold=fold, scaler=SCALER,
                            stride=WINDOW * a.stride_mult,
                            multichannel=a.multichannel, verbose=False)
        digest = provenance_digest(data)
        m, ex = train_eval(data, epochs=EPOCHS, batch_size=BATCH, seed=0,
                           verbose=0, early_stop_flag=0,
                           **build_kwargs(a.model, cfg,
                                          int(data.X_train.shape[2])))
        test_setups = sorted({mm["record_id"] for mm in data.meta["test"]})
        log_run(dict(
            run_id=new_run_id("loso"), experiment=f"loso_{a.model}_{dataset}",
            dataset_train=dataset, dataset_eval=dataset,
            n_classes=data.n_classes,
            n_sensors=len({mm["sensor"] for mm in data.meta["train"]}),
            window_length=WINDOW, split_scheme="loso", split_digest=digest,
            config_source="grid_selected", model=a.model, scaler=SCALER,
            notes=f"LOSO fold {fold}/{a.folds}; held-out setups: "
                  + ",".join(s.split('/')[-1] for s in test_setups),
            kernel_size=cfg.get("kernel_size", ""), filters=cfg.get("filters", ""),
            res_per_block=cfg.get("res_per_block", ""),
            n_blocks=cfg.get("n_blocks", ""),
            **{k: m[k] for k in ("lr", "epochs", "epochs_run", "batch_size",
               "head", "n_params", "seed", "val_acc", "val_acc_best", "test_acc",
               "test_macro_f1", "test_balanced_acc", "stopped_early",
               "train_time_sec", "gpu_name")},
        ), ex)
        rows.append(dict(fold=fold, test_setups=test_setups,
                         val_acc=m["val_acc"], test_acc=m["test_acc"],
                         test_macro_f1=m["test_macro_f1"],
                         test_balanced_acc=m["test_balanced_acc"]))
        print(f"  fold {fold}: val {m['val_acc']:.3f}  test {m['test_acc']:.3f}  "
              f"macroF1 {m['test_macro_f1']:.3f}", flush=True)

    suffix = "_mc" if a.multichannel else ""
    out = os.path.join(HERE, "results", f"loso_{a.model}_{dataset}{suffix}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(model=a.model, dataset=dataset, config=cfg,
                       folds=a.folds, rows=rows), fh, indent=2)

    print("\n" + "=" * 62)
    for key in ("test_acc", "test_macro_f1", "test_balanced_acc"):
        v = [r[key] for r in rows]
        print(f"{key:>20}: {np.mean(v):.3f} +/- {ci95(v):.3f}  "
              f"(min {min(v):.3f}, max {max(v):.3f}, n={len(v)} folds)")
    print("\nNote: class 03 has 8 setups, the others 9, so with 9 folds its "
          "first setup\nis held out twice. Reported rather than hidden.")


if __name__ == "__main__":
    main()
