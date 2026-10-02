"""
Does multi-channel input lift the Z24 scenarios?

Where this comes from. The single-channel formulation caps at about 0.27 macro-F1
on Z24 15-class no matter how the labels are grouped (diag_z24_labels.py) and no
matter how much data is used (diag_z24full_data.py), and a linear model on
spectral features reaches that same 0.27. So the limit is the INFORMATION in one
20 s trace from one accelerometer, not the model. Channels were recorded
simultaneously; phase and amplitude relations between measurement points are
exactly what carries damage-location information, and the paper's formulation
throws them away by treating each channel as an independent sample.

The trade-off is real and is the thing being measured here: multi-channel input
multiplies information per sample but divides the sample count by C.

Controlled comparison -- identical split, scaler, window, epochs, configuration;
only the input formulation changes. Channels used are ones whose index means the
same physical location in every record: the five fixed Z24 sensors, and the QUGS
joints. A PSD + logistic-regression ceiling is computed for both formulations.
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

from cnn1d_torch import CNN1D
from runlog import log_run, new_run_id
from shm_data import (QUGS_5, QUGS_15, Z24_CLASSES_5, Z24_CLASSES_15,
                      build_splits, load_records, provenance_digest)
from wavenet_torch import train_eval

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
WINDOW, EPOCHS, LR = 2048, 40, 1e-3
CFG = dict(filters=32, n_blocks=4, kernel_size=9)

CASES = [
    # label            dataset  classes          sensors  max_s  scheme      stride
    ("z24_5cls",      "z24",    Z24_CLASSES_5,   "5",     None,  "record",   1),
    ("z24_15cls",     "z24",    Z24_CLASSES_15,  "5",     None,  "record",   1),
    ("qugs_5cls",     "qugs-A", QUGS_5,          "all",   5,     "campaign", 1),
    ("qugs_15cls",    "qugs-A", QUGS_15,         "all",   15,    "campaign", 4),
]


def psd_multi(X, n_bands=128):
    """log-PSD per channel, concatenated."""
    feats = []
    for c in range(X.shape[2]):
        x = X[:, :, c]
        P = np.abs(np.fft.rfft(x * np.hanning(x.shape[1]), axis=1)) ** 2
        e = np.linspace(0, P.shape[1], n_bands + 1).astype(int)
        B = np.stack([P[:, e[i]:e[i+1]].mean(1) for i in range(n_bands)], 1)
        feats.append(np.log10(B + 1e-20))
    return np.concatenate(feats, axis=1)


def main():
    rows = []
    for label, ds, classes, sensors, max_s, scheme, stride_mult in CASES:
        recs = load_records(ds, classes, sensors, max_sensors=max_s)
        test_recs = None
        if scheme == "campaign":
            test_recs = load_records("qugs-B", classes, sensors, max_sensors=max_s)
        fracs = (0.75, 0.25, 0.0) if scheme == "campaign" else (0.7, 0.1, 0.2)

        print(f"\n{'='*78}\n{label}")
        for mc in (False, True):
            data = build_splits(recs, seed=0, window_length=WINDOW,
                                split_scheme=scheme, fracs=fracs,
                                stride=WINDOW * stride_mult, scaler="per_window",
                                test_records=test_recs, multichannel=mc,
                                verbose=False)
            C = data.X_train.shape[2]
            tag = f"multi({C}ch)" if mc else "single(1ch)"

            Xtr, Xte = psd_multi(data.X_train), psd_multi(data.X_test)
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), data.y_train)
            p = clf.predict(sc.transform(Xte))
            psd_f1 = f1_score(data.y_test, p, average="macro", zero_division=0)

            m, ex = train_eval(
                data, lr=LR, epochs=EPOCHS, batch_size=64, seed=0, verbose=0,
                early_stop_flag=0,
                model_fn=lambda nc, w, _c=C: CNN1D(nc, width=w, in_channels=_c, **CFG))

            log_run(dict(
                run_id=new_run_id("diag"), experiment="diag_multichannel",
                dataset_train=f"{label}_{tag}", dataset_eval=f"{label}_{tag}",
                n_classes=data.n_classes, n_sensors=C, window_length=WINDOW,
                split_scheme=scheme, split_digest=provenance_digest(data),
                config_source="multichannel_probe", model="cnn1d",
                scaler="per_window",
                notes=f"input formulation probe: {tag}; {data.X_train.shape[0]} train samples",
                kernel_size=CFG["kernel_size"], filters=CFG["filters"],
                res_per_block="", n_blocks=CFG["n_blocks"],
                **{k: m[k] for k in ("lr","epochs","epochs_run","batch_size","head",
                   "n_params","seed","val_acc","val_acc_best","test_acc",
                   "test_macro_f1","test_balanced_acc","stopped_early",
                   "train_time_sec","gpu_name")},
            ), ex)

            print(f"  {tag:12s} train {data.X_train.shape[0]:>6,}  "
                  f"trainacc {ex['history'][-1]['acc']:.3f}  "
                  f"CNN F1 {m['test_macro_f1']:.3f}  bal {m['test_balanced_acc']:.3f}  "
                  f"|  PSD F1 {psd_f1:.3f}", flush=True)
            rows.append(dict(case=label, multichannel=mc, n_channels=C,
                             n_classes=data.n_classes,
                             n_train=int(data.X_train.shape[0]),
                             train_acc=ex["history"][-1]["acc"],
                             cnn_f1=m["test_macro_f1"],
                             cnn_bal=m["test_balanced_acc"],
                             cnn_acc=m["test_acc"], psd_f1=float(psd_f1)))

    with open(os.path.join(HERE, "results", "diag_multichannel.json"), "w",
              encoding="utf-8") as f:
        json.dump(dict(config=CFG, lr=LR, epochs=EPOCHS, rows=rows), f, indent=2)

    print("\n" + "=" * 78)
    print(f"{'case':12s} {'cls':>4} | {'single F1':>10} {'multi F1':>9} {'delta':>7} "
          f"| {'PSD single':>11} {'PSD multi':>10}")
    for i in range(0, len(rows), 2):
        s_, m_ = rows[i], rows[i+1]
        print(f"{s_['case']:12s} {s_['n_classes']:>4} | {s_['cnn_f1']:>10.3f} "
              f"{m_['cnn_f1']:>9.3f} {m_['cnn_f1']-s_['cnn_f1']:>+7.3f} "
              f"| {s_['psd_f1']:>11.3f} {m_['psd_f1']:>10.3f}")


if __name__ == "__main__":
    main()
