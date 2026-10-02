"""
Why does z24_full barely work?

Selected-configuration macro-F1 on test, from the campaign grids:
    cnn1d   z24_full   0.213 +/- 0.168   (chance ~0.067 for 15 classes)
    wavenet z24_full   0.068 +/- 0.019   (= chance)
while the same pipeline reaches 0.73 on z24_small and 0.96 on qugs_small. Every
low rank-agreement in the transfer analysis involves z24_full, so its numbers
cannot be used until this is understood.

Hypothesis under test. Z24's 15 states include GRADED SEVERITIES of the same
damage type -- pier settlement at 20/40/80/95 mm, 2/4/6 ruptured tendons. Those
classes plausibly differ mainly in vibration AMPLITUDE, and `per_window`
normalisation removes amplitude entirely. The normalisation that rescued
z24_small (5 classes, +0.18) may be destroying z24_full (15 classes).

Confound to keep in mind: z24_small uses the five sensors held fixed across
setups, z24_full uses all 33 channels of each setup as independent samples. So
this crosses normalisation with sampling density, and reports the channel count
separately rather than pretending it is controlled.

    scaler   per_window (amplitude removed) vs global (amplitude kept)
    stride   8x (campaign setting, 11.6k train) vs 4x (23.2k train)
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
import numpy as np
from cnn1d_torch import CNN1D
from runlog import log_run, new_run_id
from shm_data import Z24_CLASSES_15, build_splits, load_records, provenance_digest
from wavenet_torch import train_eval

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
WINDOW, EPOCHS = 2048, 40
# configuration selected by the campaign grid for cnn1d on z24_full
CFG = dict(filters=32, n_blocks=4, kernel_size=9)
LR = 3e-4


def factory(n_classes, width):
    return CNN1D(n_classes, width=width, **CFG)


def main():
    records = load_records("z24", Z24_CLASSES_15, "all")
    rows = []
    for scaler in ("per_window", "global"):
        for mult in (8, 4):
            data = build_splits(records, seed=0, window_length=WINDOW,
                                split_scheme="record", scaler=scaler,
                                stride=WINDOW * mult, verbose=False)
            m, ex = train_eval(data, lr=LR, epochs=EPOCHS, batch_size=64,
                               seed=0, verbose=0, early_stop_flag=0,
                               model_fn=factory)
            log_run(dict(
                run_id=new_run_id("diag"), experiment="diag_z24full",
                dataset_train="z24_full", dataset_eval="z24_full",
                n_classes=data.n_classes,
                n_sensors=len({mm["sensor"] for mm in data.meta["train"]}),
                window_length=WINDOW, split_scheme="record",
                split_digest=provenance_digest(data),
                config_source="grid_selected", model="cnn1d", scaler=scaler,
                notes=f"z24_full probe; scaler={scaler}; stride={mult}x",
                kernel_size=CFG["kernel_size"], filters=CFG["filters"],
                res_per_block="", n_blocks=CFG["n_blocks"],
                **{k: m[k] for k in ("lr","epochs","epochs_run","batch_size",
                   "head","n_params","seed","val_acc","val_acc_best","test_acc",
                   "test_macro_f1","test_balanced_acc","stopped_early",
                   "train_time_sec","gpu_name")},
            ), ex)
            pcr = ex["per_class_recall"]
            rows.append(dict(scaler=scaler, stride_mult=mult,
                             n_train=int(data.X_train.shape[0]),
                             train_acc=ex["history"][-1]["acc"],
                             val_acc=m["val_acc"], test_acc=m["test_acc"],
                             test_macro_f1=m["test_macro_f1"],
                             per_class_recall=pcr))
            r = rows[-1]
            print(f"scaler={scaler:<11} stride={mult}x  train {r['n_train']:>6,}  "
                  f"trainacc {r['train_acc']:.3f}  val {r['val_acc']:.3f}  "
                  f"test {r['test_acc']:.3f}  macroF1 {r['test_macro_f1']:.3f}",
                  flush=True)
            print("    per-class recall: " +
                  " ".join(f"{v:.2f}" for v in pcr), flush=True)

    with open(os.path.join(HERE, "results", "diag_z24full.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(config=CFG, lr=LR, epochs=EPOCHS, rows=rows), f, indent=2)
    print(f"\nchance macro-F1 for 15 classes ~ {1/15:.3f}")
    print("campaign grid gave cnn1d/z24_full 0.213 +/- 0.168 with per_window, stride 8x")


if __name__ == "__main__":
    main()
