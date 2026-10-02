"""
The reframed transferability analysis.

The submitted paper measures transferability by taking the single best
configuration found on a source scenario, applying it to a target scenario,
and reading off one accuracy. That is n = 1 per cell: no distribution, no
test, and no way to separate a real effect from split noise.

Transferability is really a property of the RESPONSE SURFACE of the objective
over the hyperparameter space, so with a complete 72-configuration grid per
scenario it can be measured directly:

  rank agreement   Spearman rho and Kendall tau between the source and target
                   surfaces, n = 72 per pair. A high rho means a configuration
                   that is good on the source tends to be good on the target,
                   which is exactly what "the hyperparameters transfer" means.

  transfer regret  target_best - target_score(argmax source), absolute and
                   relative to the target's achievable range. It answers the
                   practical question: what does reusing a configuration cost you?

  top-k overlap    how many of the source's best k configurations are among
                   the target's best k. Robust to monotone rescaling.

Search strategies are scored against the same grids as
ground truth: whether the GA optimum is the true optimum, its regret, and how
it compares with random search under an identical evaluation budget.

Selection always uses validation accuracy (the GA's own fitness, as in the
paper); every reported outcome is an independent TEST metric.

Usage:  python analyze.py [--metric test_macro_f1]
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  _os.pardir, "src"))
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import argparse
import glob
import itertools
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
RESULTS = os.path.join(HERE, "results")
GENES = {"wavenet": ["lr", "filters", "res_per_block", "n_blocks"],
         "cnn1d": ["lr", "filters", "n_blocks", "kernel_size"]}


def load_grids():
    """{(model, dataset, seed): {config_key: trace_row}}"""
    grids = {}
    for path in glob.glob(os.path.join(RESULTS, "search_grid_*.json")):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("model") not in GENES:
            print(f"[skip] {os.path.basename(path)}: no recognised model field")
            continue
        table = {}
        for t in d["trace"]:
            key = tuple(t["config"][g] for g in GENES[d["model"]])
            table[key] = t
        grids[(d["model"], d["dataset"], d["seed"])] = table
    return grids


def ci95(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def score(row, metric):
    return row["fitness"] if metric == "val" else row[metric]


def surface_pair(src, tgt, metric, topk=5):
    """Transfer statistics from one response surface to another."""
    keys = sorted(set(src) & set(tgt))
    if len(keys) < 3:
        return None
    a = np.array([score(src[k], metric) for k in keys])
    b = np.array([score(tgt[k], metric) for k in keys])
    rho, p_rho = stats.spearmanr(a, b)
    tau, p_tau = stats.kendalltau(a, b)

    # selection uses validation accuracy, exactly as the GA does
    sel_src = np.array([src[k]["fitness"] for k in keys])
    sel_tgt = np.array([tgt[k]["fitness"] for k in keys])
    i_src, i_tgt = int(np.argmax(sel_src)), int(np.argmax(sel_tgt))

    transferred, best = b[i_src], b[i_tgt]
    spread = float(b.max() - b.min())
    top_s = np.argsort(-sel_src)[:topk]
    top_t = set(np.argsort(-sel_tgt)[:topk].tolist())

    return dict(
        n=len(keys), spearman=float(rho), spearman_p=float(p_rho),
        kendall=float(tau), kendall_p=float(p_tau),
        transferred_score=float(transferred), target_best=float(best),
        regret_abs=float(best - transferred),
        regret_rel=float((best - transferred) / spread) if spread > 0 else 0.0,
        target_spread=spread,
        topk_overlap=int(sum(1 for i in top_s if i in top_t)), topk=topk,
        source_argmax=list(keys[i_src]), target_argmax=list(keys[i_tgt]),
    )


def report_scenarios(grids, metric):
    print(f"\n{'=' * 78}\nSCENARIO SUMMARY  |  metric={metric}")
    print(f"{'model':>8} {'dataset':>12} {'selected':>17} {'median':>8} "
          f"{'worst':>8} {'spread':>8}")
    agg = defaultdict(list)
    for (model, dataset, _seed), tab in sorted(grids.items()):
        v = np.array([score(t, metric) for t in tab.values()])
        sel = np.array([t["fitness"] for t in tab.values()])
        agg[(model, dataset)].append((v[int(np.argmax(sel))], np.median(v),
                                      v.min(), v.max() - v.min()))
    for (model, dataset), rows in sorted(agg.items()):
        r = np.array(rows)
        print(f"{model:>8} {dataset:>12} {r[:, 0].mean():>8.3f}+/-{ci95(r[:, 0]):<8.3f}"
              f"{r[:, 1].mean():>8.3f} {r[:, 2].mean():>8.3f} {r[:, 3].mean():>8.3f}"
              f"   (n_seed={len(rows)})")


def report_transfer(grids, metric):
    for model in sorted({k[0] for k in grids}):
        datasets = sorted({k[1] for k in grids if k[0] == model})
        seeds = sorted({k[2] for k in grids if k[0] == model})
        if len(datasets) < 2:
            continue
        print(f"\n{'=' * 78}\nTRANSFER between scenarios  |  model={model}  "
              f"metric={metric}")
        print(f"{'source -> target':<30} {'rho':>7} {'tau':>7} {'regret':>8} "
              f"{'rel':>7} {'top5':>6}")
        for src_d, tgt_d in itertools.permutations(datasets, 2):
            rows = []
            for s in seeds:
                a, b = grids.get((model, src_d, s)), grids.get((model, tgt_d, s))
                if a and b:
                    r = surface_pair(a, b, metric)
                    if r:
                        rows.append(r)
            if not rows:
                continue
            m = lambda k: float(np.mean([r[k] for r in rows]))
            print(f"{src_d + ' -> ' + tgt_d:<30} {m('spearman'):>7.3f} "
                  f"{m('kendall'):>7.3f} {m('regret_abs'):>8.3f} "
                  f"{m('regret_rel'):>7.3f} {m('topk_overlap'):>6.1f}"
                  f"   (n_seed={len(rows)})")


def report_search(grids, metric):
    """GA and random search scored against the exhaustive grid."""
    rows = defaultdict(list)
    for path in glob.glob(os.path.join(RESULTS, "search_*.json")):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if d["strategy"] == "grid":
            continue
        grid = grids.get((d["model"], d["dataset"], d["seed"]))
        if not grid:
            continue
        truth_key = max(grid, key=lambda k: grid[k]["fitness"])
        found_key = tuple(d["best_config"][g] for g in GENES[d["model"]])
        if found_key not in grid:
            continue
        rows[(d["model"], d["dataset"], d["strategy"])].append(dict(
            found_optimum=found_key == truth_key,
            regret=score(grid[truth_key], metric) - score(grid[found_key], metric),
            evals=d["unique_evaluations"]))
    if not rows:
        return
    print(f"\n{'=' * 78}\nSEARCH STRATEGY vs EXHAUSTIVE GRID  |  metric={metric}")
    print(f"{'model':>8} {'dataset':>12} {'strategy':>8} {'hit rate':>9} "
          f"{'regret':>18} {'evals':>6}")
    for (model, dataset, strat), rs in sorted(rows.items()):
        reg = [r["regret"] for r in rs]
        print(f"{model:>8} {dataset:>12} {strat:>8} "
              f"{np.mean([r['found_optimum'] for r in rs]):>8.0%} "
              f"{np.mean(reg):>9.3f}+/-{ci95(reg):<8.3f}"
              f"{np.mean([r['evals'] for r in rs]):>6.1f}   (n={len(rs)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="test_macro_f1",
                    choices=["test_acc", "test_macro_f1", "val"])
    a = ap.parse_args()
    grids = load_grids()
    if not grids:
        print("no grid results yet")
        return
    print(f"loaded {len(grids)} grid(s): "
          + ", ".join(f"{m}/{d}/seed{s}" for (m, d, s) in sorted(grids)))
    report_scenarios(grids, a.metric)
    report_transfer(grids, a.metric)
    report_search(grids, a.metric)


if __name__ == "__main__":
    main()
