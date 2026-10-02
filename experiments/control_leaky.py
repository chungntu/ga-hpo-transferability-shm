"""
Control experiment: how much of the published accuracy is the split protocol?

This is the single most important piece of evidence for the response letter. The
submitted paper reports 0.97-0.98 on Z24 Full-scale; the leakage-free protocol
gives 0.26. Rather than asserting that the difference is the split, this measures
it: the SAME model, the SAME configuration, the SAME data, the SAME window,
scaler and epoch budget, run under two protocols that differ in one respect.

    record          whole measurement setups are held out (correct)
    random_window   windows are pooled and split at random, so windows and
                    channels of one setup appear in train AND test (the
                    submitted code's protocol, and that of much of the Z24
                    literature -- e.g. the open-access Sensors 2023 study that
                    reports 97.5% on five Z24 classes with a random sample-level
                    split over all channels)

If the leaky protocol reproduces ~0.97 while the record protocol gives ~0.26,
then the gap is attributable to the protocol and to nothing else, and the
published numbers -- ours and others' -- can be interpreted accordingly without
having to accuse anyone of anything.

Usage:  python control_leaky.py [--seeds 0 1 2]
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
from shm_data import load_dataset, provenance_digest
from wavenet_torch import train_eval

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
EPOCHS, BATCH, WINDOW, SCALER = 40, 64, 2048, "per_window"

PAIRS = [("z24_small", "z24_small_leaky"), ("z24_full", "z24_full_leaky")]


def grid_config(model, dataset):
    """Configuration with the highest mean validation accuracy in the campaign grids."""
    tally = {}
    for p in glob.glob(os.path.join(HERE, "results",
                                    f"search_grid_{model}_{dataset}_seed*.json")):
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        for t in d["trace"]:
            tally.setdefault(tuple(t["config"][g] for g in genes(model)),
                             []).append(t["fitness"])
    if not tally:
        return None
    return dict(zip(genes(model), max(tally, key=lambda k: np.mean(tally[k]))))


def ci95(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", default=["cnn1d", "wavenet"],
                    choices=list(SEARCH_SPACES))
    a = ap.parse_args()

    out_path = os.path.join(HERE, "results", "control_leaky.json")
    done = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            for r in json.load(fh)["rows"]:
                done[(r["model"], r["preset"], r["seed"])] = r
        print(f"[resume] {len(done)} run(s) already done")

    rows = list(done.values())
    for model in a.models:
        for correct, leaky in PAIRS:
            cfg = grid_config(model, correct)
            if cfg is None:
                print(f"[skip] no grid for {model}/{correct}")
                continue
            print(f"\n=== {model} | {correct} vs {leaky} | config {cfg} ===", flush=True)
            for preset in (correct, leaky):
                for seed in a.seeds:
                    if (model, preset, seed) in done:
                        print(f"  [skip] {preset} seed{seed}")
                        continue
                    data = load_dataset(preset, seed=seed, window_length=WINDOW,
                                        scaler=SCALER, verbose=False)
                    C = int(data.X_train.shape[2])
                    m, ex = train_eval(data, epochs=EPOCHS, batch_size=BATCH,
                                       seed=seed, verbose=0, early_stop_flag=0,
                                       **build_kwargs(model, cfg, C))
                    log_run(dict(
                        run_id=new_run_id("control"), experiment="control_leaky",
                        dataset_train=preset, dataset_eval=preset,
                        n_classes=data.n_classes, n_sensors=C,
                        window_length=WINDOW,
                        split_scheme=data.meta["split_scheme"],
                        split_digest=provenance_digest(data),
                        config_source=f"grid_selected_{correct}", model=model,
                        scaler=SCALER,
                        notes=f"protocol control: {data.meta['split_scheme']}",
                        kernel_size=cfg.get("kernel_size", ""),
                        filters=cfg.get("filters", ""),
                        res_per_block=cfg.get("res_per_block", ""),
                        n_blocks=cfg.get("n_blocks", ""),
                        **{k: m[k] for k in (
                            "lr", "epochs", "epochs_run", "batch_size", "head",
                            "n_params", "seed", "val_acc", "val_acc_best",
                            "test_acc", "test_macro_f1", "test_balanced_acc",
                            "stopped_early", "train_time_sec", "gpu_name")},
                    ), ex)
                    r = dict(model=model, preset=preset, seed=seed,
                             pair=correct, scheme=data.meta["split_scheme"],
                             n_train=int(data.X_train.shape[0]),
                             test_acc=m["test_acc"],
                             test_macro_f1=m["test_macro_f1"],
                             test_balanced_acc=m["test_balanced_acc"])
                    rows.append(r)
                    print(f"  {preset:18s} seed{seed}  acc {r['test_acc']:.3f}  "
                          f"macroF1 {r['test_macro_f1']:.3f}", flush=True)
                    with open(out_path, "w", encoding="utf-8") as fh:
                        json.dump(dict(epochs=EPOCHS, window=WINDOW,
                                       scaler=SCALER, rows=rows), fh, indent=2)

    print("\n" + "=" * 84)
    print(f"{'model':>8} {'scenario':>11} {'record split':>22} {'random-window split':>22} "
          f"{'gap':>7}")
    for model in a.models:
        for correct, leaky in PAIRS:
            c = [r["test_macro_f1"] for r in rows
                 if r["model"] == model and r["preset"] == correct]
            l = [r["test_macro_f1"] for r in rows
                 if r["model"] == model and r["preset"] == leaky]
            if not c or not l:
                continue
            print(f"{model:>8} {correct:>11} {np.mean(c):>13.3f} +/-{ci95(c):<7.3f}"
                  f"{np.mean(l):>13.3f} +/-{ci95(l):<7.3f}{np.mean(l)-np.mean(c):>+7.3f}")
    print("\nmacro-F1 on the independent test set. The only difference between the "
          "two\ncolumns is which windows are allowed into the training set.")


if __name__ == "__main__":
    main()
