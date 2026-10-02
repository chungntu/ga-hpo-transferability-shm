"""
Does the Z24 Full-scale label definition explain the low score?

The submitted paper treats states 03/04/05/06 -- pier settlement at 20, 40, 80
and 95 mm -- as four separate classes. The established benchmark convention
(Abdrabo et al. 2024, and used by the authors' own PIAE paper on Z24) treats
states 1-8 as a single HEALTHY condition, "since they induce only minor and
largely reversible changes in the dynamic response", and states 9-17 as damage.

If that convention is right, the 15-class task asks the model to separate
classes that the benchmark literature considers dynamically near-identical, and
the low score is a property of the label definition, not of the model.

Three formulations, same split protocol, same architecture, same configuration:

  A  15-class   the paper's definition (01,03..07,09..17)
  B  10-class   convention: states 1-8 merged into one healthy class,
                states 9-17 kept separate  -> 1 + 9 = 10 classes
  C   9-class   damage states 9-17 only (damage TYPE given that damage exists)

A PSD + logistic-regression ceiling is computed for each, so the numbers can be
read as "what the task contains" rather than "what this model achieves".
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from baseline_psd import psd_features
from cnn1d_torch import CNN1D
from runlog import log_run, new_run_id
from shm_data import build_splits, load_records, provenance_digest
from wavenet_torch import train_eval

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
WINDOW, EPOCHS, STRIDE_MULT = 2048, 40, 8
CFG = dict(filters=16, n_blocks=3, kernel_size=9)      # grid-selected for z24_full
LR = 3e-4

ALL17 = [f"{i:02d}" for i in range(1, 18)]
HEALTHY = ALL17[:8]            # S1-S8
DAMAGE = ALL17[8:]             # S9-S17

FORMULATIONS = {
    "A_15class_paper": dict(
        classes=["01", "03", "04", "05", "06", "07"] + DAMAGE,
        remap=None),
    "B_10class_convention": dict(
        classes=ALL17,
        remap=lambda c: 0 if c in HEALTHY else 1 + DAMAGE.index(c)),
    "C_9class_damage_only": dict(
        classes=DAMAGE,
        remap=None),
}


def factory(n_classes, width):
    return CNN1D(n_classes, width=width, **CFG)


def main():
    out_rows = []
    for name, spec in FORMULATIONS.items():
        recs = load_records("z24", spec["classes"], "all")
        if spec["remap"]:
            for r in recs:
                r.class_idx = spec["remap"](r.class_name)
        else:
            order = {c: i for i, c in enumerate(spec["classes"])}
            for r in recs:
                r.class_idx = order[r.class_name]

        data = build_splits(recs, seed=0, window_length=WINDOW,
                            split_scheme="record", scaler="per_window",
                            stride=WINDOW * STRIDE_MULT, verbose=False)
        counts = np.bincount(data.y_train, minlength=data.n_classes)
        print(f"\n=== {name}: {data.n_classes} classes, chance {1/data.n_classes:.3f} ===")
        print(f"    train {data.X_train.shape[0]:,} / val {data.X_val.shape[0]:,} "
              f"/ test {data.X_test.shape[0]:,}")
        print(f"    train per class: min {counts.min():,} max {counts.max():,} "
              f"(imbalance {counts.max()/counts.min():.1f}x)")

        # ceiling: PSD + logistic regression
        Xtr, Xte = psd_features(data.X_train), psd_features(data.X_test)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), data.y_train)
        p = clf.predict(sc.transform(Xte))
        psd = dict(acc=accuracy_score(data.y_test, p),
                   f1=f1_score(data.y_test, p, average="macro", zero_division=0),
                   bal=balanced_accuracy_score(data.y_test, p))
        print(f"    PSD+logreg   acc {psd['acc']:.3f}  macroF1 {psd['f1']:.3f}  "
              f"balanced {psd['bal']:.3f}", flush=True)

        m, ex = train_eval(data, lr=LR, epochs=EPOCHS, batch_size=64, seed=0,
                           verbose=0, early_stop_flag=0, model_fn=factory)
        log_run(dict(
            run_id=new_run_id("diag"), experiment="diag_z24_labels",
            dataset_train=f"z24_{name}", dataset_eval=f"z24_{name}",
            n_classes=data.n_classes,
            n_sensors=len({mm["sensor"] for mm in data.meta["train"]}),
            window_length=WINDOW, split_scheme="record",
            split_digest=provenance_digest(data),
            config_source="grid_selected_z24_full", model="cnn1d",
            scaler="per_window", notes=f"label-definition probe: {name}",
            kernel_size=CFG["kernel_size"], filters=CFG["filters"],
            res_per_block="", n_blocks=CFG["n_blocks"],
            **{k: m[k] for k in ("lr", "epochs", "epochs_run", "batch_size",
               "head", "n_params", "seed", "val_acc", "val_acc_best", "test_acc",
               "test_macro_f1", "test_balanced_acc", "stopped_early",
               "train_time_sec", "gpu_name")},
        ), ex)
        print(f"    1D-CNN       acc {m['test_acc']:.3f}  macroF1 {m['test_macro_f1']:.3f}"
              f"  balanced {m['test_balanced_acc']:.3f}  (train acc "
              f"{ex['history'][-1]['acc']:.3f})", flush=True)
        out_rows.append(dict(name=name, n_classes=data.n_classes,
                             chance=1/data.n_classes,
                             n_train=int(data.X_train.shape[0]),
                             imbalance=float(counts.max()/counts.min()),
                             psd=psd,
                             cnn=dict(acc=m["test_acc"], f1=m["test_macro_f1"],
                                      bal=m["test_balanced_acc"],
                                      train_acc=ex["history"][-1]["acc"]),
                             per_class_recall=ex["per_class_recall"]))

    with open(os.path.join(HERE, "results", "diag_z24_labels.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(config=CFG, lr=LR, epochs=EPOCHS, rows=out_rows), f, indent=2)

    print("\n" + "=" * 88)
    print(f"{'formulation':24s} {'cls':>4} {'chance':>7} {'PSD F1':>8} "
          f"{'CNN F1':>8} {'CNN bal':>8} {'lift over chance':>17}")
    for r in out_rows:
        print(f"{r['name']:24s} {r['n_classes']:>4} {r['chance']:>7.3f} "
              f"{r['psd']['f1']:>8.3f} {r['cnn']['f1']:>8.3f} {r['cnn']['bal']:>8.3f} "
              f"{r['cnn']['f1']/r['chance']:>16.1f}x")


if __name__ == "__main__":
    main()
