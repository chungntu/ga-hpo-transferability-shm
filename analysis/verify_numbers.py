"""
Recompute every number quoted in paper/PAPER_COMPACT.md from results/.

    python analysis/verify_numbers.py

Prints one block per table/claim; compare by eye with the paper. Nothing is
trained here.
"""

import collections
import glob
import itertools
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def grids(model, ds):
    """{seed: {config_key: trace_row}}"""
    out = {}
    for p in sorted(glob.glob(os.path.join(RES, f"search_grid_{model}_{ds}_seed*.json"))):
        d = load(os.path.basename(p))
        out[d["seed"]] = {json.dumps(t["config"], sort_keys=True): t for t in d["trace"]}
    return out


def best_by_val(tab):
    return max(tab.values(), key=lambda t: t["fitness"])


def ci95(v):
    v = np.asarray(v, float)
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def head(t):
    print(f"\n=== {t} " + "=" * max(0, 70 - len(t)))


# ------------------------------------------------------------ Table 3
head("Table 3 / Fig 3: input formulation (diag_multichannel.json)")
d = load("diag_multichannel.json")
print("fixed config:", d["config"], "lr", d["lr"], "epochs", d["epochs"])
for r in d["rows"]:
    print(f"{r['case']:<11} mc={str(r['multichannel']):<5} C={r['n_channels']:<3} n_train={r['n_train']:<6} "
          f"cnn_f1={r['cnn_f1']:.4f} psd_f1={r['psd_f1']:.4f}")

# ------------------------------------------------------------ Table 4
head("Table 4 / Fig 4: classes vs sensors, best-by-validation test macro-F1")
cells = ["c05_s05", "c10_s05", "c15_s05", "c05_s15", "c05_s33"]
for model in ("cnn1d", "wavenet"):
    f = {c: np.array([load(f"factorial_{model}_{c}_seed{s}.json")["best_test_macro_f1"]
                      for s in range(3)]) for c in cells}
    for c in cells[1:]:
        dd = f[c] - f["c05_s05"]
        print(f"{model:<8} {c}: mean {dd.mean():+.3f}  p={stats.ttest_1samp(dd, 0).pvalue:.3f}  "
              f"per-seed {np.round(dd, 3).tolist()}")

# ------------------------------------------------------------ Table 5
head("Table 5 / Fig 5: best (by validation) vs median, test macro-F1, paired by seed")
for model in ("cnn1d", "wavenet"):
    for ds in ("z24_small", "z24_full", "qugs_small", "qugs_full"):
        g = grids(model, ds)
        b = np.array([best_by_val(t)["test_macro_f1"] for t in g.values()])
        m = np.array([np.median([x["test_macro_f1"] for x in t.values()]) for t in g.values()])
        p = stats.ttest_rel(b, m).pvalue
        print(f"{model:<8} {ds:<11} n_seed={len(b)} n_cfg={len(next(iter(g.values())))} "
              f"best {b.mean():.3f} median {m.mean():.3f} gain {b.mean() - m.mean():+.3f} p={p:.4f}")

# ------------------------------------------------------------ transfer loss
head("Transfer loss: target F1 of source-selected config vs target-selected config")
pairs = {"small": ("z24_small", "qugs_small"), "full": ("z24_full", "qugs_full")}
for model in ("cnn1d", "wavenet"):
    G = {ds: grids(model, ds) for pr in pairs.values() for ds in pr}
    for tgt_name in ("z24", "qugs"):
        for pool in (("small",), ("full",), ("small", "full")):
            loss = []
            for lvl in pool:
                z, q = pairs[lvl]
                src, tgt = (q, z) if tgt_name == "z24" else (z, q)
                for s in sorted(set(G[src]) & set(G[tgt])):
                    a, b = G[src][s], G[tgt][s]
                    ks = sorted(set(a) & set(b))
                    i_src = max(ks, key=lambda k: a[k]["fitness"])
                    i_tgt = max(ks, key=lambda k: b[k]["fitness"])
                    loss.append(b[i_tgt]["test_macro_f1"] - b[i_src]["test_macro_f1"])
            loss = np.array(loss)
            p = stats.ttest_1samp(loss, 0).pvalue if len(loss) > 1 and loss.std() > 0 else float("nan")
            print(f"{model:<8} target={tgt_name:<5} levels={'+'.join(pool):<11} n={len(loss)} "
                  f"mean loss {loss.mean():+.3f}  p={p:.3f}")

# ------------------------------------------------------------ Table 6
head("Table 6 / Fig 6: rank agreement (Spearman, test macro-F1)")


def rho(a, b):
    k = sorted(set(a) & set(b))
    return stats.spearmanr([a[x]["test_macro_f1"] for x in k], [b[x]["test_macro_f1"] for x in k]).correlation


for model in ("cnn1d", "wavenet"):
    for lvl, (z, q) in pairs.items():
        Z, Q = grids(model, z), grids(model, q)
        wz = np.mean([rho(Z[i], Z[j]) for i, j in itertools.combinations(sorted(Z), 2)])
        wq = np.mean([rho(Q[i], Q[j]) for i, j in itertools.combinations(sorted(Q), 2)])
        ac = np.mean([rho(Z[s], Q[s]) for s in sorted(set(Z) & set(Q))])
        print(f"{model:<8} {lvl:<5} within Z24 {wz:.3f}  within QUGS {wq:.3f}  across {ac:.3f}")

# ------------------------------------------------------------ WaveNet multi-channel
head("WaveNet single vs multi-channel (grid, best by validation, test macro-F1)")
for ds in ("z24_full", "z24_small"):
    for suffix in ("", "_mc"):
        g = grids("wavenet", ds + suffix)
        print(f"wavenet {ds + suffix:<13} {np.mean([best_by_val(t)['test_macro_f1'] for t in g.values()]):.3f}"
              f"  (n_seed={len(g)})")
for ds in ("z24_full", "z24_small"):
    for suffix in ("", "_mc"):
        g = grids("cnn1d", ds + suffix)
        print(f"cnn1d   {ds + suffix:<13} {np.mean([best_by_val(t)['test_macro_f1'] for t in g.values()]):.3f}"
              f"  (n_seed={len(g)})")

# ------------------------------------------------------------ QUGS multi-channel ties
head("QUGS multi-channel: settings at the ceiling (mean test macro-F1 >= 0.999)")
for model in ("cnn1d", "wavenet"):
    for ds in ("qugs_small_mc", "qugs_full_mc"):
        g = grids(model, ds)
        if not g:
            continue
        keys = set.intersection(*[set(t) for t in g.values()])
        mean = {k: np.mean([g[s][k]["test_macro_f1"] for s in g]) for k in keys}
        n1 = sum(1 for v in mean.values() if v >= 0.999)
        print(f"{model:<8} {ds:<14} {n1}/{len(mean)} settings at >= 0.999")

# ------------------------------------------------------------ Table 7
head("Table 7 / Fig 7: offline search replay (hpo_offline.json)")
h = load("hpo_offline.json")
agg = collections.defaultdict(list)
for r in h["rows"]:
    agg[(r["algorithm"], r["budget"])].append(r)
algs = ["random", "ga", "ga_long", "tpe", "bo_gp", "succ_halving", "hill_climb"]
n_tab = len(agg[("random", h["budgets"][0])])
print("tables:", n_tab, " repeats:", h["repeats"], " budgets:", h["budgets"])
nconf = collections.Counter(r["n_configs"] for r in agg[("random", 6.0)])
print("tables by space size:", dict(nconf))
print("budget " + " ".join(f"{a:>12}" for a in algs))
for b in h["budgets"]:
    print(f"{b:>6.0f} " + " ".join(f"{np.mean([r['mean_regret'] for r in agg[(a, b)]]):>12.4f}" for a in algs))
print("hit rate at budget 6:", {a: round(float(np.mean([r['hit_rate'] for r in agg[(a, 6.0)]])), 3) for a in algs})
print("GA mean cost by budget:", {b: round(float(np.mean([r['mean_cost'] for r in agg[('ga', b)]])), 1)
                                  for b in h["budgets"]})

# ------------------------------------------------------------ Table 8
head("Table 8: leave-one-setup-out (mean, 95% CI half-width over folds; also SD)")
for name in ("cnn1d_z24_small_mc", "cnn1d_z24_small", "wavenet_z24_small_mc", "wavenet_z24_small",
             "cnn1d_z24_full_mc", "cnn1d_z24_full", "wavenet_z24_full_mc", "wavenet_z24_full"):
    d = load(f"loso_{name}.json")
    a = [r["test_acc"] for r in d["rows"]]
    m = [r["test_macro_f1"] for r in d["rows"]]
    print(f"{name:<22} folds={len(a)} acc {np.mean(a):.3f} ± {ci95(a):.3f} (sd {np.std(a, ddof=1):.3f})  "
          f"f1 {np.mean(m):.3f} ± {ci95(m):.3f} (sd {np.std(m, ddof=1):.3f})")

# ------------------------------------------------------------ Table 9
head("Table 9: QUGS 31 states vs number of sensors (qugs_channel_curve.json)")
q = load("qugs_channel_curve.json")
print("states", q["states"], "chance", round(q["chance"], 3))
for r in q["rows"]:
    print(f"C={r['n_channels']:<3} n_train={r['n_train']} cnn_f1={r['cnn_f1']:.4f} psd_f1={r['psd_f1']:.4f}")

# ------------------------------------------------------------ misc
head("Misc")
print("window 2048 samples: Z24 %.2f s, QUGS %.2f s" % (2048 / 100, 2048 / 1024))
print("budget 6 as share of space: 72 -> %.1f%%, 36 -> %.1f%%" % (600 / 72, 600 / 36))
psd = load("baseline_psd_z24_small_seed0.json")["rows"]
print("spectral classifier, Z24 5 classes, single split, window 2048:",
      [(r["window"], round(r["test_acc"], 3), round(r["test_macro_f1"], 3)) for r in psd if r["window"] == 2048])
zf = load("diag_z24full.json")["rows"]
print("Z24 15 classes single-channel, fixed config, training windows by stride:",
      [(r["stride_mult"], r["scaler"], r["n_train"], round(r["test_macro_f1"], 3)) for r in zf])
