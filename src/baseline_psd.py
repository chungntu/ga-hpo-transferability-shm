"""
Reference baseline: log-PSD features + multinomial logistic regression, on
exactly the same leakage-free splits used for WaveNet.

Purpose is not to propose a better model. It is to establish how much
class-discriminative information survives the record-level split, so that a
WaveNet result at chance can be read correctly: as a statement about the
model, not about the data. Reported for every window length so it can be
compared against diag_window.py run for run.
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from shm_data import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
N_BANDS = 128

def psd_features(X, n_bands=N_BANDS):
    x = X[..., 0]
    P = np.abs(np.fft.rfft(x * np.hanning(x.shape[1]), axis=1)) ** 2
    edges = np.linspace(0, P.shape[1], n_bands + 1).astype(int)
    B = np.stack([P[:, edges[i]:edges[i+1]].mean(1) for i in range(n_bands)], 1)
    return np.log10(B + 1e-20)

def main(preset="z24_small", windows=(65536, 8192, 2048), seed=0):
    rows = []
    for w in windows:
        d = load_dataset(preset, seed=seed, window_length=w, verbose=False)
        nb = min(N_BANDS, w // 4)
        Xtr, Xva, Xte = (psd_features(d.X_train, nb), psd_features(d.X_val, nb),
                         psd_features(d.X_test, nb))
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), d.y_train)
        out = dict(window=w, n_bands=nb, n_train=int(len(d.y_train)))
        for name, X, y in (("train", Xtr, d.y_train), ("val", Xva, d.y_val),
                           ("test", Xte, d.y_test)):
            p = clf.predict(sc.transform(X))
            out[f"{name}_acc"] = float(accuracy_score(y, p))
            out[f"{name}_macro_f1"] = float(f1_score(y, p, average="macro", zero_division=0))
            out[f"{name}_balanced_acc"] = float(balanced_accuracy_score(y, p))
        rows.append(out)
        print(f"window {w:>6}  bands {nb:>3}  ntrain {out['n_train']:>5}  "
              f"train {out['train_acc']:.3f}  val {out['val_acc']:.3f}  "
              f"test {out['test_acc']:.3f}  macroF1 {out['test_macro_f1']:.3f}", flush=True)
    path = os.path.join(HERE, "results", f"baseline_psd_{preset}_seed{seed}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(preset=preset, seed=seed, rows=rows), f, indent=2)

if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
