"""
Where does the QUGS localisation task stop being saturated? (paper Table 9)

Sweeps the number of simultaneously recorded channels, all 31 states, trained
on campaign A and tested on campaign B, for a 1D-CNN with a fixed setting and
for logistic regression on log-PSD features.

This is the script that produced results/qugs_channel_curve.json; it was first
run inline and is saved here unchanged apart from the path set-up.

    python experiments/diagnostics/diag_qugs_channels.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  _os.pardir, _os.pardir, "src"))
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from cnn1d_torch import CNN1D
from shm_data import build_splits, load_records
from wavenet_torch import train_eval

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
ALL31 = [f"state{i:02d}" for i in range(1, 32)]
CFG = dict(filters=32, n_blocks=4, kernel_size=9)


def psd(X, nb=64):
    out = []
    for c in range(X.shape[2]):
        P = np.abs(np.fft.rfft(X[:, :, c] * np.hanning(X.shape[1]), axis=1)) ** 2
        e = np.linspace(0, P.shape[1], nb + 1).astype(int)
        out.append(np.log10(np.stack([P[:, e[i]:e[i + 1]].mean(1) for i in range(nb)], 1) + 1e-20))
    return np.concatenate(out, 1)


def main():
    rows = []
    print(f"{'channels':>9} {'train':>7} {'CNN F1':>8} {'PSD F1':>8}")
    for n_sens in (1, 2, 3, 4, 6, 10, 30):
        A = load_records("qugs-A", ALL31, "all", max_sensors=n_sens)
        B = load_records("qugs-B", ALL31, "all", max_sensors=n_sens)
        d = build_splits(A, seed=0, window_length=2048, split_scheme="campaign",
                         fracs=(0.75, 0.25, 0.0), stride=2048, scaler="per_window",
                         test_records=B, multichannel=True, verbose=False)
        Xtr, Xte = psd(d.X_train), psd(d.X_test)
        sc = StandardScaler().fit(Xtr)
        p = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), d.y_train).predict(sc.transform(Xte))
        pf1 = f1_score(d.y_test, p, average="macro", zero_division=0)
        m, _ = train_eval(d, lr=1e-3, epochs=40, batch_size=64, seed=0, verbose=0,
                          early_stop_flag=0,
                          model_fn=lambda nc, w, _c=n_sens: CNN1D(nc, width=w, in_channels=_c, **CFG))
        rows.append(dict(n_channels=n_sens, n_train=int(d.X_train.shape[0]),
                         cnn_f1=m["test_macro_f1"], psd_f1=float(pf1)))
        print(f"{n_sens:>9} {d.X_train.shape[0]:>7,} {m['test_macro_f1']:>8.3f} {pf1:>8.3f}", flush=True)
    with open(os.path.join(HERE, "results", "qugs_channel_curve.json"), "w", encoding="utf-8") as f:
        json.dump(dict(states=31, chance=1 / 31, rows=rows), f, indent=2)
    print(f"\nchance = {1 / 31:.3f}")


if __name__ == "__main__":
    main()
