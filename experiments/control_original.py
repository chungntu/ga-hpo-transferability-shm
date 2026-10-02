"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  _os.pardir, "src"))
Faithful reproduction of the submitted paper's protocol, then one leak removed
at a time.

control_leaky.py showed that the random-window split alone accounts for up to
+0.32 macro-F1 but does NOT reproduce the submitted 0.97-0.98. That control was
not faithful enough: it kept the repaired pipeline's 2048-sample window, its
train-only scaler and its 40-epoch budget, and it reported test metrics. The
submitted numbers were produced with

    window 65536          one sample per channel per record, no windowing
    random sample split   channels of one setup in train AND test   (leak L1)
    scaler fit on ALL     StandardScaler().fit_transform() before splitting (L3)
    15 epochs
    VALIDATION accuracy   reported as the result; the test set was never used

so this reproduces exactly that, and then removes one leak at a time to
attribute the inflation. Configurations are Table 3 of the paper.

The point is not to re-obtain a high number but to show which choice produces
it, so the response letter can state the cause instead of asserting it.

Usage:  python control_original.py [--seeds 0 1 2]
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import argparse
import json
import os

import numpy as np
from scipy import stats

from runlog import log_run, new_run_id
from shm_data import (Z24_CLASSES_5, Z24_CLASSES_15, build_splits,
                      load_records, provenance_digest)
from wavenet_torch import train_eval

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root

# paper Table 3
CFG = {"z24_small": dict(lr=7e-5, filters=4, res_per_block=8, n_blocks=2),
       "z24_full":  dict(lr=5e-5, filters=16, res_per_block=8, n_blocks=2)}

# EPOCHS: the paper states 15 epochs in section 3.2.1, but that is the GA's
# budget (Step1/Step2, whose logs are all tagged ep15). The runs that produced
# Table 4 and Table 6 are Step3-Step6, which declare epochs = 50 and whose logs
# are tagged ep50. So the protocol being reproduced here is 50 epochs; the
# paper's "15" is a documentation error. An earlier version of this ladder used
# 15 and reached chance everywhere, which said nothing about the paper.
ORIG_EPOCHS = 50

# name                window  split            scaler            epochs
LADDER = [
    ("original",        65536, "random_window", "per_timestep_all", ORIG_EPOCHS),
    ("minus_L3_scaler", 65536, "random_window", "per_timestep",     ORIG_EPOCHS),
    ("minus_L1_split",  65536, "record",        "per_timestep",     ORIG_EPOCHS),
    ("plus_window",      2048, "record",        "per_timestep",     ORIG_EPOCHS),
    ("repaired",         2048, "record",        "per_window",       40),
]


def ci95(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--scenarios", nargs="+", default=["z24_small", "z24_full"])
    a = ap.parse_args()

    out_path = os.path.join(HERE, "results",
                            f"control_original_ep{ORIG_EPOCHS}.json")
    done, rows = {}, []
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            rows = json.load(fh)["rows"]
        done = {(r["scenario"], r["step"], r["seed"]): r for r in rows}
        print(f"[resume] {len(done)} run(s) already done")

    for scenario in a.scenarios:
        classes = Z24_CLASSES_5 if scenario == "z24_small" else Z24_CLASSES_15
        sensors = "5" if scenario == "z24_small" else "all"
        recs = load_records("z24", classes, sensors)
        cfg = CFG[scenario]
        print(f"\n{'='*84}\n{scenario}  (paper Table 3 config: {cfg})", flush=True)

        for step, window, scheme, scaler, epochs in LADDER:
            for seed in a.seeds:
                if (scenario, step, seed) in done:
                    print(f"  [skip] {step} seed{seed}")
                    continue
                # z24_full at window 65536 gives one sample per channel; at 2048
                # it gives 32, so thin it there to keep the runs comparable in cost
                stride = window * (8 if (window == 2048 and scenario == "z24_full") else 1)
                data = build_splits(recs, seed=seed, window_length=window,
                                    split_scheme=scheme, scaler=scaler,
                                    stride=stride, verbose=False)
                m, ex = train_eval(data, epochs=epochs, batch_size=16, seed=seed,
                                   verbose=0, early_stop_flag=0, head="flatten",
                                   **cfg)
                log_run(dict(
                    run_id=new_run_id("ctrlorig"), experiment="control_original",
                    dataset_train=f"{scenario}::{step}",
                    dataset_eval=f"{scenario}::{step}",
                    n_classes=data.n_classes,
                    n_sensors=len({mm["sensor"] for mm in data.meta["train"]}),
                    window_length=window, split_scheme=scheme,
                    split_digest=provenance_digest(data),
                    config_source="paper_table3", model="wavenet", scaler=scaler,
                    notes=f"protocol ladder step '{step}'",
                    kernel_size="", filters=cfg["filters"],
                    res_per_block=cfg["res_per_block"], n_blocks=cfg["n_blocks"],
                    **{k: m[k] for k in (
                        "lr", "epochs", "epochs_run", "batch_size", "head",
                        "n_params", "seed", "val_acc", "val_acc_best", "test_acc",
                        "test_macro_f1", "test_balanced_acc", "stopped_early",
                        "train_time_sec", "gpu_name")},
                ), ex)
                r = dict(scenario=scenario, step=step, seed=seed, window=window,
                         scheme=scheme, scaler=scaler, epochs=epochs,
                         n_train=int(data.X_train.shape[0]),
                         n_val=int(data.X_val.shape[0]),
                         val_acc=m["val_acc"], test_acc=m["test_acc"],
                         test_macro_f1=m["test_macro_f1"])
                rows.append(r)
                print(f"  {step:16s} seed{seed}  n_val {r['n_val']:>5,}  "
                      f"VAL acc {r['val_acc']:.3f}   test acc {r['test_acc']:.3f}  "
                      f"macroF1 {r['test_macro_f1']:.3f}", flush=True)
                with open(out_path, "w", encoding="utf-8") as fh:
                    json.dump(dict(configs=CFG, ladder=LADDER, rows=rows), fh, indent=2)

    print("\n" + "=" * 84)
    print("VALIDATION accuracy -- the quantity the submitted paper reports")
    print(f"{'scenario':>11} {'step':>17} {'val acc':>18} {'test acc':>18}")
    for scenario in a.scenarios:
        for step, *_ in LADDER:
            v = [r["val_acc"] for r in rows
                 if r["scenario"] == scenario and r["step"] == step]
            t = [r["test_acc"] for r in rows
                 if r["scenario"] == scenario and r["step"] == step]
            if not v:
                continue
            print(f"{scenario:>11} {step:>17} {np.mean(v):>10.3f} +/-{ci95(v):<7.3f}"
                  f"{np.mean(t):>10.3f} +/-{ci95(t):<7.3f}")
    print("\nSubmitted paper: 0.75 (Small-scale) and 0.98 (Full-scale), "
          "validation accuracy.")


if __name__ == "__main__":
    main()
