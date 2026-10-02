"""
Offline benchmark of hyperparameter-optimisation algorithms on the tabulated
response surfaces.

Every configuration in the search space has already been trained, and every run
kept its per-epoch validation curve. That makes the whole space a lookup table,
so any search algorithm can be replayed on it exactly, hundreds of times, at no
GPU cost: the grid is the ground truth, and the per-epoch curves let
budget-allocating methods such as successive halving be simulated faithfully
rather than approximated.

This is what makes a fair comparison possible at all. Comparing search algorithms
by running each once is dominated by which random start it happened to get; here
each algorithm is replayed 200 times per scenario.

Algorithms
    random        sampling without replacement
    ga            the submitted paper's GA, settings verbatim from Step1
                  (population 8, 4 generations, elitism 2, tournament 3,
                  mutation 0.30, evaluation cache on)
    tpe           tree-structured Parzen estimator over the categorical space
    bo_gp         Gaussian process on ordinal-encoded genes, expected improvement
    succ_halving  successive halving on the per-epoch curves, eta = 3
    hill_climb    random restart, single-gene moves

Accounting. Cost is measured in *equivalent full trainings*: one for each unique
configuration trained to completion, and epochs_used / epochs_total for a
partially trained one. The GA's cache is honoured, so its repeats are free, which
is how it was actually run.

Protocol. Every algorithm selects by validation accuracy, exactly as the GA does.
What is reported is the independent test metric of whatever it selected, plus
whether that configuration is the true validation optimum of the table.

Usage:
    python hpo_offline.py                     # all models, all scenarios
    python hpo_offline.py --budgets 8 16 24   # cost points to report
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  _os.pardir, "src"))

import argparse
import csv
import glob
import json
import os
import random
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
RESULTS = os.path.join(HERE, "results")
GENES = {"wavenet": ["lr", "filters", "res_per_block", "n_blocks"],
         "cnn1d": ["lr", "filters", "n_blocks", "kernel_size"]}

# GA settings, verbatim from Step1_WaveNetRun_5sensors_5cases_GA.py
GA_POP, GA_GENS, GA_ELITE, GA_TOURN, GA_MUT = 8, 4, 2, 3, 0.30


# ----------------------------------------------------------------------
# building the lookup tables
# ----------------------------------------------------------------------

def _runs_index():
    """(experiment, seed, config tuple) -> run_id, for joining the curves."""
    path = os.path.join(RESULTS, "runs.csv")
    idx = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            model = r.get("model")
            if model not in GENES:
                continue
            try:
                key = tuple(float(r[g]) if g == "lr" else int(r[g])
                            for g in GENES[model])
            except (ValueError, KeyError):
                continue
            idx[(r["experiment"], r["seed"], key)] = r["run_id"]
    return idx


def _load_curves():
    """Curves from results/curves.json when present, else from results/details/.

    curves.json is the versioned extract; details/ is 94 MB and is not in the
    repository, so the benchmark has to be reproducible from the extract alone.
    Returns {(experiment, seed, config tuple): [val_acc per epoch]}.
    """
    path = os.path.join(RESULTS, "curves.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        runs = json.load(fh)["runs"]
    out = {}
    for r in runs.values():
        model = r["model"]
        if model not in GENES:
            continue
        key = tuple(r["config"][g] for g in GENES[model])
        out[(r["experiment"], str(r["seed"]), key)] = r["val_acc"]
    return out


def load_tables(verbose=True):
    """{(model, dataset, seed): {config: {val, test_acc, test_f1, curve}}}"""
    curves = _load_curves()
    if verbose:
        print("  curves source:",
              f"results/curves.json ({len(curves)} runs)" if curves
              else "results/details/ (curves.json absent)")
    idx = None if curves else _runs_index()
    tables = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "search_grid_*.json"))):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        model = d.get("model")
        if model not in GENES:
            continue
        exp = f"grid_{model}_{d['dataset']}"
        table, with_curve = {}, 0
        for t in d["trace"]:
            key = tuple(t["config"][g] for g in GENES[model])
            entry = dict(val=t["fitness"], test_acc=t["test_acc"],
                         test_f1=t.get("test_macro_f1", t["test_acc"]),
                         curve=None)
            if curves is not None:
                c = curves.get((exp, str(d["seed"]), key))
                if c:
                    entry["curve"] = c
                    with_curve += 1
            else:
                rid = idx.get((exp, str(d["seed"]), key))
                if rid:
                    dp = os.path.join(RESULTS, "details", f"{rid}.json")
                    if os.path.exists(dp):
                        with open(dp, encoding="utf-8") as fh:
                            entry["curve"] = [h["val_acc"]
                                              for h in json.load(fh)["history"]]
                        with_curve += 1
            table[key] = entry
        tables[(model, d["dataset"], d["seed"])] = table
        if verbose:
            print(f"  {model:>8} {d['dataset']:>14} seed{d['seed']}  "
                  f"{len(table):3d} configs, {with_curve:3d} with curves")
    return tables


# ----------------------------------------------------------------------
# a budgeted evaluator over the table
# ----------------------------------------------------------------------

class Oracle:
    """Answers queries from the table and bills them in full-training units."""

    def __init__(self, table, max_epochs=None):
        self.table = table
        self.keys = list(table)
        self.cost = 0.0
        self.cache = {}
        self.trace = []          # (cost_after, key, val) for full evaluations
        curves = [v["curve"] for v in table.values() if v["curve"]]
        self.max_epochs = max_epochs or (len(curves[0]) if curves else 40)

    def full(self, key):
        """Train to completion. Repeats are free, as the GA's cache made them."""
        if key in self.cache:
            return self.cache[key]
        self.cost += 1.0
        v = self.table[key]["val"]
        self.cache[key] = v
        self.trace.append((self.cost, key, v))
        return v

    def partial(self, key, epochs):
        """Validation accuracy after `epochs`, billed pro rata."""
        c = self.table[key]["curve"]
        if not c:
            return self.full(key)
        e = min(epochs, len(c))
        self.cost += e / self.max_epochs
        return c[e - 1]

    def best_so_far(self):
        if not self.cache:
            return None
        return max(self.cache, key=lambda k: self.cache[k])


# ----------------------------------------------------------------------
# algorithms: each consumes an Oracle until its budget is spent
# ----------------------------------------------------------------------

def alg_random(o, budget, rng):
    keys = o.keys[:]
    rng.shuffle(keys)
    for k in keys:
        if o.cost >= budget:
            break
        o.full(k)


def _neighbours(o, model, max_dist=2):
    """Configurations in the table within `max_dist` gene changes of each other.

    Distance <= 2 rather than 1, for two reasons.

    A local search that re-queries configurations already in the cache spends
    nothing -- caching is honoured here because that is how the GA actually ran --
    so once the reachable neighbourhood is exhausted the budget stops advancing
    and the search cannot terminate. Candidates already evaluated are therefore
    excluded and the walk restarts elsewhere when it runs dry.

    The half-fraction makes that happen sooner. Its parity rule does not remove
    every single-gene move -- a gene with three or four levels can be stepped by
    two and keep the parity -- but it thins them heavily: 84 distance-1 pairs among
    36 points, against 576 among the full 72. Genes with only two levels lose all
    of theirs. So the neighbourhood is genuinely sparser on the fractional design,
    which is a property of the design rather than of the algorithm and is worth
    stating when the two are compared.
    """
    keys = o.keys
    nb = {k: [] for k in keys}
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            d = sum(1 for x, y in zip(keys[a], keys[b]) if x != y)
            if d <= max_dist:
                nb[keys[a]].append(keys[b])
                nb[keys[b]].append(keys[a])
    return nb


def alg_hill_climb(o, budget, rng, model, nb=None):
    nb = nb if nb is not None else _neighbours(o, model)
    cur = rng.choice(o.keys)
    best = o.full(cur)
    stalls = 0
    while o.cost < budget and stalls < 50:
        cands = [k for k in nb[cur] if k not in o.cache]
        if not cands:                      # local optimum: restart elsewhere
            fresh = [k for k in o.keys if k not in o.cache]
            if not fresh:
                return
            cur = rng.choice(fresh)
            v = o.full(cur)
            if v > best:
                best = v
            stalls += 1
            continue
        k = rng.choice(cands)
        v = o.full(k)
        if v > best:
            best, cur, stalls = v, k, 0
        else:
            stalls += 1


def alg_ga(o, budget, rng, model, generations=GA_GENS, pop_size=GA_POP):
    g = GENES[model]
    levels = [sorted({k[i] for k in o.keys}) for i in range(len(g))]
    allowed = set(o.keys)

    def rand_ind():
        return list(rng.choice(o.keys))

    def repair(ind):
        if tuple(ind) in allowed:
            return ind
        for _ in range(200):
            c = list(ind)
            i = rng.randrange(len(g))
            c[i] = rng.choice(levels[i])
            if tuple(c) in allowed:
                return c
        return list(rng.choice(o.keys))

    pop = [rand_ind() for _ in range(pop_size)]
    for _ in range(generations):
        if o.cost >= budget:
            break
        fits = []
        for ind in pop:
            if o.cost >= budget and tuple(ind) not in o.cache:
                fits.append(-1.0)
            else:
                fits.append(o.full(tuple(ind)))
        order = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
        new = [list(pop[i]) for i in order[:GA_ELITE]]
        while len(new) < pop_size:
            def tourn():
                c = rng.sample(range(len(pop)), min(GA_TOURN, len(pop)))
                return pop[max(c, key=lambda i: fits[i])]
            p1, p2 = tourn(), tourn()
            pt = rng.randrange(1, len(g))
            child = [p1[i] if i < pt else p2[i] for i in range(len(g))]
            for i in range(len(g)):
                if rng.random() < GA_MUT:
                    child[i] = rng.choice(levels[i])
            new.append(repair(child))
        pop = new


def alg_tpe(o, budget, rng, model, n_startup=6, gamma=0.25, n_cand=40):
    g = GENES[model]
    levels = [sorted({k[i] for k in o.keys}) for i in range(len(g))]
    obs = []
    keys = o.keys[:]
    rng.shuffle(keys)
    for k in keys[:n_startup]:
        if o.cost >= budget:
            return
        obs.append((k, o.full(k)))
    while o.cost < budget:
        obs.sort(key=lambda t: -t[1])
        n_good = max(1, int(round(gamma * len(obs))))
        good, bad = obs[:n_good], obs[n_good:] or obs
        # per-gene categorical densities with Laplace smoothing
        def dens(pool, i):
            c = defaultdict(lambda: 1.0)
            for k, _ in pool:
                c[k[i]] += 1.0
            tot = sum(c[v] for v in levels[i])
            return {v: c[v] / tot for v in levels[i]}
        lg = [dens(good, i) for i in range(len(g))]
        bg = [dens(bad, i) for i in range(len(g))]
        seen = {k for k, _ in obs}
        pool = [k for k in o.keys if k not in seen]
        if not pool:
            return
        cands = [pool[rng.randrange(len(pool))]
                 for _ in range(min(n_cand, len(pool)))]
        scored = [(sum(np.log(lg[i][k[i]]) - np.log(bg[i][k[i]])
                       for i in range(len(g))), k) for k in cands]
        best = max(scored)[1]
        obs.append((best, o.full(best)))


def alg_bo_gp(o, budget, rng, model, n_startup=6):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    g = GENES[model]
    levels = [sorted({k[i] for k in o.keys}) for i in range(len(g))]
    enc = lambda k: [levels[i].index(k[i]) / max(1, len(levels[i]) - 1)
                     for i in range(len(g))]
    obs = []
    keys = o.keys[:]
    rng.shuffle(keys)
    for k in keys[:n_startup]:
        if o.cost >= budget:
            return
        obs.append((k, o.full(k)))
    while o.cost < budget:
        seen = {k for k, _ in obs}
        pool = [k for k in o.keys if k not in seen]
        if not pool:
            return
        X = np.array([enc(k) for k, _ in obs])
        y = np.array([v for _, v in obs])
        gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(length_scale=0.5, nu=2.5),
            normalize_y=True, alpha=1e-6, random_state=rng.randrange(10 ** 6))
        gp.fit(X, y)
        mu, sd = gp.predict(np.array([enc(k) for k in pool]), return_std=True)
        best = y.max()
        sd = np.maximum(sd, 1e-9)
        z = (mu - best) / sd
        ei = (mu - best) * stats.norm.cdf(z) + sd * stats.norm.pdf(z)
        obs.append((pool[int(np.argmax(ei))], o.full(pool[int(np.argmax(ei))])))


def alg_succ_halving(o, budget, rng, eta=3, min_epochs=None):
    """Successive halving on the real per-epoch validation curves."""
    min_e = min_epochs or max(1, o.max_epochs // (eta ** 2))
    keys = o.keys[:]
    rng.shuffle(keys)
    # start with as many configs as the budget allows at the smallest rung
    n = max(eta, int(budget * o.max_epochs / min_e / 3))
    alive = keys[:min(n, len(keys))]
    e = min_e
    rungs = 0
    while alive and o.cost < budget and rungs < 20:
        rungs += 1
        scored = []
        for k in alive:
            if o.cost >= budget:
                break
            scored.append((o.partial(k, e), k))
        if not scored:
            break
        scored.sort(reverse=True)
        keep = max(1, len(scored) // eta)
        alive = [k for _, k in scored[:keep]]
        if e >= o.max_epochs or len(alive) == 1:
            break
        e = min(o.max_epochs, e * eta)
    # bill the survivor at full length so the selection is comparable
    for k in alive[:1]:
        o.full(k)


ALGS = {
    "random": lambda o, b, r, m, nb=None: alg_random(o, b, r),
    # the schedule used in the earlier work: population 8, 4 generations. With the
    # evaluation cache this spends about 16 distinct evaluations and then stops, so it
    # cannot use a budget larger than that -- which is a property of the schedule, not
    # of evolutionary search. ga_long removes that ceiling by evolving until the budget
    # is spent, so the comparison at large budgets is fair.
    "ga": lambda o, b, r, m, nb=None: alg_ga(o, b, r, m),
    "ga_long": lambda o, b, r, m, nb=None: alg_ga(o, b, r, m, generations=99),
    "tpe": lambda o, b, r, m, nb=None: alg_tpe(o, b, r, m),
    "bo_gp": lambda o, b, r, m, nb=None: alg_bo_gp(o, b, r, m),
    "succ_halving": lambda o, b, r, m, nb=None: alg_succ_halving(o, b, r),
    "hill_climb": lambda o, b, r, m, nb=None: alg_hill_climb(o, b, r, m, nb),
}


# ----------------------------------------------------------------------

def ci95(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=float, nargs="+", default=[8, 14, 20])
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--metric", default="test_f1", choices=["test_f1", "test_acc"])
    a = ap.parse_args()

    print("loading tables")
    tables = load_tables()
    print(f"{len(tables)} tables\n")

    rows = []
    for (model, dataset, seed), table in sorted(tables.items()):
        nb = _neighbours(Oracle(table), model)     # O(n^2), computed once
        truth_key = max(table, key=lambda k: table[k]["val"])
        truth_metric = table[truth_key][a.metric]
        for budget in a.budgets:
            for name, fn in ALGS.items():
                found, hits, costs = [], [], []
                for rep in range(a.repeats):
                    rng = random.Random(hash((model, dataset, seed, name,
                                              budget, rep)) & 0xffffffff)
                    o = Oracle(table)
                    if name == "hill_climb":
                        fn(o, budget, rng, model, nb)
                    else:
                        fn(o, budget, rng, model)
                    k = o.best_so_far()
                    if k is None:
                        continue
                    found.append(table[k][a.metric])
                    hits.append(k == truth_key)
                    costs.append(o.cost)
                if not found:
                    continue
                rows.append(dict(model=model, dataset=dataset, seed=seed,
                                 budget=budget, algorithm=name,
                                 n_configs=len(table),
                                 truth_metric=truth_metric,
                                 mean_metric=float(np.mean(found)),
                                 ci_metric=ci95(found),
                                 mean_regret=float(truth_metric - np.mean(found)),
                                 hit_rate=float(np.mean(hits)),
                                 mean_cost=float(np.mean(costs)),
                                 repeats=len(found)))

    out = os.path.join(RESULTS, "hpo_offline.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(metric=a.metric, budgets=a.budgets,
                       repeats=a.repeats, rows=rows), fh, indent=2)

    # aggregate over scenarios and seeds
    print(f"{'=' * 86}\nPooled over all scenarios and seeds  |  metric={a.metric}"
          f"  |  {a.repeats} replays each")
    print(f"{'budget':>7} {'algorithm':>13} {'regret':>18} {'hit rate':>10} "
          f"{'cost':>7}")
    for budget in a.budgets:
        for name in ALGS:
            sel = [r for r in rows if r["budget"] == budget
                   and r["algorithm"] == name]
            if not sel:
                continue
            reg = [r["mean_regret"] for r in sel]
            print(f"{budget:>7.0f} {name:>13} {np.mean(reg):>9.4f}"
                  f"+/-{ci95(reg):<8.4f}{np.mean([r['hit_rate'] for r in sel]):>9.1%} "
                  f"{np.mean([r['mean_cost'] for r in sel]):>7.1f}")
        print()
    print(f"-> {os.path.basename(out)}")


if __name__ == "__main__":
    main()
