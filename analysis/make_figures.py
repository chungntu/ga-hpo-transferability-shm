"""
Figures for paper/PAPER_COMPACT.md.

Every figure is regenerated from the versioned JSON in results/; nothing here
trains a model. Output: paper/figures/fig<N>_<name>.png (300 dpi) and .pdf.

    python analysis/make_figures.py
"""

import collections
import glob
import itertools
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

# Reference categorical palette (light mode), fixed order.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def grid_surface(model, dataset, metric="test_macro_f1"):
    """Mean over seeds of `metric` per configuration, for one grid scenario."""
    acc = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(RES, f"search_grid_{model}_{dataset}_seed*.json"))):
        for t in load(os.path.basename(path))["trace"]:
            acc[json.dumps(t["config"], sort_keys=True)].append(t[metric])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def grid_per_seed(model, dataset):
    """List over seeds of (test macro-F1 of the validation-best config, median over configs)."""
    out = []
    for path in sorted(glob.glob(os.path.join(RES, f"search_grid_{model}_{dataset}_seed*.json"))):
        tr = load(os.path.basename(path))["trace"]
        best = max(tr, key=lambda t: t["fitness"])
        out.append((best["test_macro_f1"], float(np.median([t["test_macro_f1"] for t in tr]))))
    return out


# ---------------------------------------------------------------- Figure 1
def fig1_input_formulation():
    rng = np.random.default_rng(3)
    t = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(-0.6, 6.0); ax.axis("off")
    cols = [BLUE, ORANGE, AQUA, VIOLET, MAGENTA]
    ys = [4.4, 3.6, 2.8, 2.0, 1.2]

    def trace(x0, y0, w, h, phase, color, lw=1.0):
        s = (np.sin(2 * np.pi * 6 * t + phase) + 0.5 * np.sin(2 * np.pi * 15 * t + 2 * phase)
             + 0.25 * rng.standard_normal(t.size))
        ax.plot(x0 + w * t, y0 + h * s / 3.2, color=color, lw=lw)

    ax.text(1.25, 5.85, "Sensors recorded\nat the same time", ha="center", va="top", fontsize=8.5, color=INK2)
    for i, (y, c) in enumerate(zip(ys, cols)):
        trace(0.2, y, 2.1, 0.55, i * 0.7, c)
        ax.text(0.05, y, f"S{i + 1}", ha="right", va="center", fontsize=8, color=INK2)

    # single-channel: five separate (T,1) samples
    ax.text(5.0, 5.85, "Single-channel input\n5 samples of shape (T, 1)", ha="center", va="top", fontsize=8.5)
    for i, (y, c) in enumerate(zip(ys, cols)):
        ax.add_patch(Rectangle((3.9, y - 0.3), 2.2, 0.6, fill=False, ec=GRID, lw=0.8))
        trace(4.0, y, 2.0, 0.45, i * 0.7, c, lw=0.8)
    # multi-channel: one (T,5) sample
    ax.text(8.6, 5.85, "Multi-channel input\n1 sample of shape (T, 5)", ha="center", va="top", fontsize=8.5)
    ax.add_patch(Rectangle((7.45, 0.85), 2.3, 3.9, fill=False, ec=INK2, lw=1.0))
    for i, (y, c) in enumerate(zip(ys, cols)):
        trace(7.55, y, 2.1, 0.45, i * 0.7, c, lw=0.8)

    ax.add_patch(FancyArrowPatch((2.45, 2.8), (3.8, 2.8), arrowstyle="-|>", mutation_scale=10,
                                 color=MUTED, lw=1.0))
    ax.plot([6.75, 6.75], [0.9, 4.7], color=GRID, lw=1.0)
    ax.text(6.75, 2.8, "or", ha="center", va="center", fontsize=9, color=INK2,
            bbox=dict(fc="white", ec="none", pad=2))
    ax.text(5.0, 0.15, "each sensor seen alone:\nfrequency content only",
            ha="center", va="center", fontsize=8, color=INK2)
    ax.text(8.6, 0.15, "all sensors seen together:\nfrequency + relative motion",
            ha="center", va="center", fontsize=8, color=INK2)
    save(fig, "fig1_input_formulation")


# ---------------------------------------------------------------- Figure 2
def fig2_split():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6),
                             gridspec_kw={"width_ratios": [1.4, 1], "wspace": 0.25})
    # Z24: leave-one-setup-out
    ax = axes[0]
    ax.set_title("Z24: leave-one-setup-out", loc="left")
    n = 9
    for fold in range(n):
        for s in range(n):
            if s == fold:
                c, lab = ORANGE, "test"
            elif s == (fold + 1) % n:
                c, lab = YELLOW, "validation"
            else:
                c, lab = BLUE, "training"
            ax.add_patch(Rectangle((s + 0.06, n - 1 - fold + 0.08), 0.88, 0.84, color=c, lw=0))
    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_xticks(np.arange(n) + 0.5, [str(i + 1) for i in range(n)])
    ax.set_yticks(np.arange(n) + 0.5, [str(n - i) for i in range(n)])
    ax.set_xlabel("sensor setup"); ax.set_ylabel("fold")
    ax.grid(False); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # QUGS: campaign A -> B
    ax = axes[1]
    ax.set_title("QUGS: separate campaigns", loc="left")
    ax.add_patch(Rectangle((0.0, 0.55), 0.78, 0.35, color=BLUE, lw=0))
    ax.add_patch(Rectangle((0.80, 0.55), 0.20, 0.35, color=YELLOW, lw=0))
    ax.add_patch(Rectangle((0.0, 0.08), 1.0, 0.35, color=ORANGE, lw=0))
    ax.text(0.39, 0.725, "A: training", ha="center", va="center", color="white", fontsize=8.5)
    ax.text(0.90, 0.725, "val.", ha="center", va="center", color=INK, fontsize=8)
    ax.text(0.5, 0.255, "B: test", ha="center", va="center", color="white", fontsize=8.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, YELLOW, ORANGE)]
    fig.legend(handles, ["training", "validation", "test"], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.16))
    save(fig, "fig2_split")


# ---------------------------------------------------------------- Figure 3
def fig3_formulation_bars():
    rows = {(r["case"], r["multichannel"]): r for r in load("diag_multichannel.json")["rows"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, (clf, key) in zip(axes, [("1D-CNN (fixed setting)", "cnn_f1"),
                                     ("Spectral classifier", "psd_f1")]):
        cases = [("z24_5cls", "Z24, 5"), ("z24_15cls", "Z24, 15"),
                 ("qugs_5cls", "QUGS, 5"), ("qugs_15cls", "QUGS, 15")]
        x = np.arange(len(cases)); w = 0.36
        for j, (mc, c, lab) in enumerate([(False, BLUE, "single-channel"), (True, ORANGE, "multi-channel")]):
            v = [rows[(k, mc)][key] for k, _ in cases]
            ax.bar(x + (j - 0.5) * (w + 0.02), v, w, color=c, label=lab)
            for xi, vi in zip(x, v):
                ax.text(xi + (j - 0.5) * (w + 0.02), vi + 0.015, f"{vi:.3f}", ha="center",
                        va="bottom", fontsize=6.5, color=INK2, rotation=90)
        ax.set_xticks(x, [l for _, l in cases]); ax.set_title(clf, loc="left")
        ax.set_xlabel("benchmark, classes")
        ax.set_ylim(0, 1.2); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("test macro-F1")
    axes[0].legend(loc="upper center", ncol=2, bbox_to_anchor=(1.05, 1.22))
    save(fig, "fig3_formulation")


# ---------------------------------------------------------------- Figure 4
def fig4_factorial():
    cells = ["c05_s05", "c10_s05", "c15_s05", "c05_s15", "c05_s33"]
    labels = {"c10_s05": "10 classes", "c15_s05": "15 classes",
              "c05_s15": "15 sensors", "c05_s33": "33 sensors"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, (model, title) in zip(axes, [("cnn1d", "1D-CNN"), ("wavenet", "WaveNet")]):
        f1 = {c: [load(f"factorial_{model}_{c}_seed{s}.json")["best_test_macro_f1"] for s in range(3)]
              for c in cells}
        base = np.array(f1["c05_s05"])
        for i, c in enumerate(cells[1:]):
            d = np.array(f1[c]) - base
            col = RED if c.startswith("c1") else AQUA
            ax.scatter(np.full(3, i), d, s=22, color=col, alpha=0.55, lw=0, zorder=3)
            ax.scatter([i], [d.mean()], s=70, marker="_", color=INK, lw=2, zorder=4)
            p = stats.ttest_1samp(d, 0).pvalue
            ax.text(i, 0.47, f"p={p:.2f}" if p >= 0.01 else f"p={p:.3f}", ha="center",
                    fontsize=7.5, color=INK2)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(range(4), [labels[c] for c in cells[1:]])
        ax.set_title(title, loc="left"); ax.grid(axis="x", visible=False)
        ax.set_ylim(-0.32, 0.52)
    axes[0].set_ylabel("change in test macro-F1\nfrom 5 classes, 5 sensors")
    save(fig, "fig4_classes_sensors")


# ---------------------------------------------------------------- Figure 5
def fig5_tuning():
    scen = [("z24_small", "Z24, 5"), ("z24_full", "Z24, 15"),
            ("qugs_small", "QUGS, 5"), ("qugs_full", "QUGS, 15")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (model, title) in zip(axes, [("cnn1d", "1D-CNN"), ("wavenet", "WaveNet")]):
        for i, (ds, lab) in enumerate(scen):
            surf = np.array(list(grid_surface(model, ds).values()))
            ax.scatter(i + rng.uniform(-0.18, 0.18, surf.size), surf, s=9, color=BLUE,
                       alpha=0.45, lw=0, zorder=2)
            ps = grid_per_seed(model, ds)
            best = np.mean([b for b, _ in ps]); med = np.mean([m for _, m in ps])
            ax.plot([i - 0.28, i + 0.28], [med, med], color=MUTED, lw=2, zorder=3)
            ax.plot([i - 0.28, i + 0.28], [best, best], color=ORANGE, lw=2, zorder=3)
        ax.set_xticks(range(4), [l for _, l in scen]); ax.set_xlabel("benchmark, classes")
        ax.set_title(title, loc="left"); ax.grid(axis="x", visible=False); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("test macro-F1")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=BLUE, alpha=0.6, ms=4),
         Line2D([], [], color=ORANGE, lw=2), Line2D([], [], color=MUTED, lw=2)]
    fig.legend(h, ["one setting (mean over splits)", "best setting (chosen on validation)",
                   "median setting"], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    save(fig, "fig5_tuning")


# ---------------------------------------------------------------- Figure 6
def _grids(model, dataset):
    out = {}
    for path in sorted(glob.glob(os.path.join(RES, f"search_grid_{model}_{dataset}_seed*.json"))):
        d = load(os.path.basename(path))
        out[d["seed"]] = {json.dumps(t["config"], sort_keys=True): t["test_macro_f1"] for t in d["trace"]}
    return out


def _rho(a, b):
    k = sorted(set(a) & set(b))
    return stats.spearmanr([a[x] for x in k], [b[x] for x in k]).correlation


def transfer_agreement(model, classes):
    """Spearman rho of test macro-F1 over all settings, per pair of splits:
    within Z24, within QUGS, and Z24 against QUGS (same split seed)."""
    tag = "small" if classes == 5 else "full"
    z, q = _grids(model, f"z24_{tag}"), _grids(model, f"qugs_{tag}")
    within_z = [_rho(z[i], z[j]) for i, j in itertools.combinations(sorted(z), 2)]
    within_q = [_rho(q[i], q[j]) for i, j in itertools.combinations(sorted(q), 2)]
    across = [_rho(z[s], q[s]) for s in sorted(set(z) & set(q))]
    return within_z, within_q, across


def fig6_transfer():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    rng = np.random.default_rng(1)
    groups = ["within Z24\n(two splits)", "within QUGS\n(two splits)", "Z24 against\nQUGS"]
    for ax, (model, title) in zip(axes, [("cnn1d", "1D-CNN"), ("wavenet", "WaveNet")]):
        for i, (vals, col) in enumerate(zip(transfer_agreement(model, 5), (BLUE, BLUE, ORANGE))):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, s=22, color=col,
                       alpha=0.6, lw=0, zorder=3)
            m = float(np.mean(vals))
            ax.plot([i - 0.25, i + 0.25], [m, m], color=INK, lw=2, zorder=4)
            ax.text(i + 0.3, m, f"{m:.2f}", va="center", fontsize=8, color=INK2)
        ax.set_xticks(range(3), groups); ax.set_title(f"{title}, 5 classes", loc="left")
        ax.set_xlim(-0.5, 2.7); ax.set_ylim(0, 1); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("rank agreement of all settings\n(Spearman ρ)")
    save(fig, "fig6_transfer")


# ---------------------------------------------------------------- Figure 7
def fig7_search():
    h = load("hpo_offline.json")
    budgets = h["budgets"]
    agg = collections.defaultdict(list)
    for r in h["rows"]:
        agg[(r["algorithm"], r["budget"])].append(r["mean_regret"])
    algs = [("random", "Random", INK2, "-"), ("ga", "Genetic", RED, "-"),
            ("ga_long", "Genetic, no generation limit", RED, "--"),
            ("tpe", "Parzen estimator", BLUE, "-"), ("bo_gp", "Gaussian process", VIOLET, "-"),
            ("succ_halving", "Successive halving", AQUA, "-"),
            ("hill_climb", "Hill climbing", YELLOW, "-")]
    n_tables = len(agg[("random", budgets[0])])
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for key, lab, col, ls in algs:
        y = [np.mean(agg[(key, b)]) for b in budgets]
        ax.plot(budgets, y, color=col, ls=ls, lw=2, marker="o", ms=4.5, label=lab)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(budgets); ax.set_xlabel("budget (full trainings)")
    ax.set_ylabel("regret (test macro-F1 below the best setting)")
    ax.legend(loc="upper right", ncol=1)
    ax.set_title(f"Mean over {n_tables} result tables, 100 replays each", loc="left")
    save(fig, "fig7_search")
    return n_tables


if __name__ == "__main__":
    fig1_input_formulation()
    fig2_split()
    fig3_formulation_bars()
    fig4_factorial()
    fig5_tuning()
    fig6_transfer()
    print("tables in replay:", fig7_search())
