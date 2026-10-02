"""
Is z24_full limited by the amount of data, or is it simply not learnable from a
single channel with 15 classes?

diag_z24full.py ruled out normalisation (global made it worse, 0.077 vs 0.113)
and showed heavy overfitting at every setting (train acc 0.98, test macro-F1
0.11-0.20 against chance 0.067). Doubling the data helped, 0.113 -> 0.171, so
the data axis is worth pushing to its limit before declaring the scenario
intractable: stride 1 uses every window, 92k training samples, 8x the campaign
setting.

If stride 1 still lands near 0.2, the amount of data is not the constraint and
the scenario is genuinely hard, which is then what gets reported.
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
from cnn1d_torch import CNN1D
from runlog import log_run, new_run_id
from shm_data import Z24_CLASSES_15, build_splits, load_records, provenance_digest
from wavenet_torch import train_eval

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
WINDOW = 2048
CFG = dict(filters=32, n_blocks=4, kernel_size=9)
LR = 3e-4


def factory(n_classes, width):
    return CNN1D(n_classes, width=width, **CFG)


def main():
    records = load_records("z24", Z24_CLASSES_15, "all")
    rows = []
    for mult, epochs in ((2, 40), (1, 40)):
        data = build_splits(records, seed=0, window_length=WINDOW,
                            split_scheme="record", scaler="per_window",
                            stride=WINDOW * mult, verbose=False)
        m, ex = train_eval(data, lr=LR, epochs=epochs, batch_size=64, seed=0,
                           verbose=0, early_stop_flag=0, model_fn=factory)
        log_run(dict(
            run_id=new_run_id("diag"), experiment="diag_z24full_data",
            dataset_train="z24_full", dataset_eval="z24_full",
            n_classes=data.n_classes,
            n_sensors=len({mm["sensor"] for mm in data.meta["train"]}),
            window_length=WINDOW, split_scheme="record",
            split_digest=provenance_digest(data),
            config_source="grid_selected", model="cnn1d", scaler="per_window",
            notes=f"z24_full data-limit probe; stride={mult}x",
            kernel_size=CFG["kernel_size"], filters=CFG["filters"],
            res_per_block="", n_blocks=CFG["n_blocks"],
            **{k: m[k] for k in ("lr","epochs","epochs_run","batch_size","head",
               "n_params","seed","val_acc","val_acc_best","test_acc",
               "test_macro_f1","test_balanced_acc","stopped_early",
               "train_time_sec","gpu_name")},
        ), ex)
        rows.append(dict(stride_mult=mult, n_train=int(data.X_train.shape[0]),
                         epochs=epochs, train_acc=ex["history"][-1]["acc"],
                         val_acc=m["val_acc"], test_acc=m["test_acc"],
                         test_macro_f1=m["test_macro_f1"],
                         train_time_sec=m["train_time_sec"]))
        r = rows[-1]
        print(f"stride={mult}x  train {r['n_train']:>6,}  trainacc {r['train_acc']:.3f}  "
              f"val {r['val_acc']:.3f}  test {r['test_acc']:.3f}  "
              f"macroF1 {r['test_macro_f1']:.3f}  ({r['train_time_sec']/60:.0f} min)",
              flush=True)

    with open(os.path.join(HERE, "results", "diag_z24full_data.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(config=CFG, lr=LR, rows=rows), f, indent=2)
    print("\nfor reference, same config: stride 8x -> 0.113, stride 4x -> 0.171")
    print("chance macro-F1 for 15 classes ~ 0.067")


if __name__ == "__main__":
    main()
