"""
Hyperparameter search for the revision: exhaustive grid (ground truth), the
paper's GA, and a random-search baseline under the same budget.

Two base models are supported, and every scenario is run on both, so the
transferability conclusions can be shown not to depend on the architecture.

  wavenet   the PyTorch port of the submitted paper's model. Genes are paper
            Table 2 (lr x filters x residual dilations x blocks, 72 configs).
            Of the three spaces on record (Table 2, the repo's Step1/Step2
            scripts, the Results logs) Table 2 is the only one containing both
            winners reported in Table 3, so it is the space the submitted
            results must have come from -- see HANDOFF section 3.2. Its
            learning-rate ceiling is raised here; the reason is at the space
            definition below.

  cnn1d     conv + pooling blocks into global average pooling, the standard
            architecture for these benchmarks. Four genes matched one-to-one
            to the paper's, with 'residual dilations per block' replaced by
            'kernel size', its counterpart in a non-dilated network. Also
            4 x 3 x 3 x 2 = 72 configurations, so search budgets and grid
            costs stay directly comparable between the two families.

GA settings are verbatim from Step1_WaveNetRun_5sensors_5cases_GA.py:
population 8, 4 generations, elitism 2, tournament k=3, mutation rate 0.30,
fitness = validation accuracy at the last epoch, evaluation cache on.

Usage
-----
python search.py grid   --dataset z24_small --model cnn1d --seed 0
python search.py ga     --dataset z24_small --model wavenet --seed 0
python search.py random --dataset qugs_small --model cnn1d --budget 14
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  _os.pardir, "src"))

import sys
# Windows consoles default to cp1252; project paths contain Vietnamese text.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import argparse
import itertools
import json
import os
import random
import time

from cnn1d_torch import CNN1D
from runlog import log_run, new_run_id
from shm_data import load_dataset, provenance_digest
from wavenet_torch import WaveNet, train_eval

SEARCH_SPACES = {
    "wavenet": {
        # Table 2 caps lr at 1e-4. Under the repaired pipeline WaveNet reaches
        # test 0.63 at lr 1e-3 but only 0.26 at 1e-4, so its useful region lies
        # OUTSIDE the published space: keeping the cap would make the
        # architecture comparison a comparison of learning-rate ceilings. The
        # range is widened to the CNN's and the change is declared in the paper
        # (a search algorithm cannot find what the search space excludes).
        "lr": [3e-5, 1e-4, 3e-4, 1e-3],
        "filters": [4, 8, 16],
        "res_per_block": [6, 8, 10],
        "n_blocks": [1, 2],
    },
    "cnn1d": {
        "lr": [3e-5, 1e-4, 3e-4, 1e-3],
        "filters": [8, 16, 32],
        "n_blocks": [3, 4, 5],
        "kernel_size": [5, 9],
    },
}

GA_POP_SIZE = 8
GA_GENERATIONS = 4
GA_ELITISM = 2
GA_TOURNAMENT_K = 3
GA_MUTATION_RATE = 0.30

# Defaults of the repaired pipeline, established by the diag_*.py probes.
EPOCHS = 40
BATCH_SIZE = 64
WINDOW = 2048
SCALER = "per_window"

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root


def genes(model):
    return list(SEARCH_SPACES[model])


def all_configs(model, fraction="full"):
    """The 72-point space, or a balanced half of it.

    `fraction="half"` keeps the 36 points whose level indices sum to an even
    number. That is a regular half-fraction: every level of every gene appears in
    exactly half of the retained points, so all four main effects stay estimable
    and a rank correlation still has n = 36. Used for the WaveNet re-run, where a
    full grid on all scenarios and seeds costs about 45 GPU-hours.
    """
    g = genes(model)
    space = SEARCH_SPACES[model]
    out = []
    for idx in itertools.product(*(range(len(space[k])) for k in g)):
        if fraction == "half" and sum(idx) % 2:
            continue
        out.append({k: space[k][i] for k, i in zip(g, idx)})
    if fraction not in ("full", "half"):
        raise ValueError(f"fraction must be 'full' or 'half', got {fraction!r}")
    return out


def cfg_key(model, cfg):
    return tuple(cfg[g] for g in genes(model))


def build_kwargs(model, cfg, in_channels=1):
    """Translate a configuration into train_eval keyword arguments."""
    if model == "wavenet":
        if in_channels == 1:
            return dict(lr=cfg["lr"], filters=cfg["filters"],
                        res_per_block=cfg["res_per_block"],
                        n_blocks=cfg["n_blocks"], head="flatten")

        def wn_factory(n_classes, w, _cfg=cfg, _c=in_channels):
            return WaveNet(n_classes, _cfg["filters"], _cfg["n_blocks"],
                           _cfg["res_per_block"], w, head="flatten",
                           in_channels=_c)
        return dict(lr=cfg["lr"], model_fn=wn_factory)
    if model == "cnn1d":
        def factory(n_classes, w, _cfg=cfg, _c=in_channels):
            return CNN1D(n_classes, filters=_cfg["filters"],
                         n_blocks=_cfg["n_blocks"],
                         kernel_size=_cfg["kernel_size"], width=w,
                         in_channels=_c)
        return dict(lr=cfg["lr"], model_fn=factory)
    raise ValueError(model)


class Evaluator:
    """Trains a config on `data`, caches by config, logs every real run."""

    def __init__(self, data, dataset_name, experiment, seed, model="cnn1d",
                 epochs=EPOCHS, batch_size=BATCH_SIZE, scaler=SCALER, verbose=0):
        self.data = data
        self.dataset_name = dataset_name
        self.experiment = experiment
        self.seed = seed
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = scaler
        self.verbose = verbose
        self.cache = {}
        self.n_evals = 0
        self.digest = provenance_digest(data)
        self.n_sensors = len({m["sensor"] for m in data.meta["train"]})
        self.in_channels = int(data.X_train.shape[2])
        self.trace = []

    def __call__(self, cfg, config_source=None, notes=""):
        k = cfg_key(self.model, cfg)
        if k in self.cache:
            return self.cache[k]

        t0 = time.perf_counter()
        metrics, extras = train_eval(
            self.data, epochs=self.epochs, batch_size=self.batch_size,
            seed=self.seed, verbose=self.verbose, early_stop_flag=0,
            **build_kwargs(self.model, cfg, self.in_channels))
        self.n_evals += 1

        row = dict(
            run_id=new_run_id(self.experiment),
            experiment=self.experiment,
            dataset_train=self.dataset_name, dataset_eval=self.dataset_name,
            n_classes=self.data.n_classes, n_sensors=self.n_sensors,
            window_length=self.data.width,
            split_scheme=self.data.meta["split_scheme"], split_digest=self.digest,
            config_source=config_source or self.experiment,
            model=self.model, scaler=self.scaler, notes=notes,
            kernel_size=cfg.get("kernel_size", ""),
            filters=cfg.get("filters", ""),
            res_per_block=cfg.get("res_per_block", ""),
            n_blocks=cfg.get("n_blocks", ""),
            **{k2: metrics[k2] for k2 in (
                "lr", "epochs", "epochs_run", "batch_size", "head", "n_params",
                "seed", "val_acc", "val_acc_best", "test_acc", "test_macro_f1",
                "test_balanced_acc", "stopped_early", "train_time_sec",
                "gpu_name")},
        )
        log_run(row, extras)

        self.cache[k] = metrics
        self.trace.append(dict(index=self.n_evals, config=dict(cfg),
                               fitness=metrics["val_acc"],
                               test_acc=metrics["test_acc"],
                               test_macro_f1=metrics["test_macro_f1"]))
        print(f"  [{self.n_evals:3d}] {cfg}  val {metrics['val_acc']:.4f}  "
              f"test {metrics['test_acc']:.4f}  ({time.perf_counter()-t0:.1f}s)",
              flush=True)
        return metrics


# ----------------------------------------------------------------------

def run_grid(ev, fraction="full"):
    cfgs = all_configs(ev.model, fraction)
    print(f"[grid] {len(cfgs)} configurations", flush=True)
    for cfg in cfgs:
        ev(cfg, config_source="grid")
    return best_of(ev)


def run_random(ev, budget, seed, fraction="full"):
    rng = random.Random(seed)
    pool = all_configs(ev.model, fraction)
    print(f"[random] budget {budget} evaluations from {len(pool)} candidates",
          flush=True)
    order = list(range(len(pool)))
    rng.shuffle(order)
    for i in order:
        if ev.n_evals >= budget:
            break
        ev(pool[i], config_source="random")
    return best_of(ev)


def run_ga(ev, seed, fraction="full"):
    rng = random.Random(seed)
    space = SEARCH_SPACES[ev.model]
    G = genes(ev.model)
    # When the grid covers only a fractional design, the GA has to stay inside it
    # or its result cannot be scored against that ground truth. Offspring landing
    # outside are snapped to the nearest admissible point by resampling.
    allowed = {cfg_key(ev.model, c) for c in all_configs(ev.model, fraction)}

    def admissible(ind):
        return cfg_key(ev.model, ind) in allowed

    def repair(ind):
        if admissible(ind):
            return ind
        for _ in range(200):
            cand = dict(ind)
            g = rng.choice(G)
            cand[g] = rng.choice(space[g])
            if admissible(cand):
                return cand
        return rng.choice(all_configs(ev.model, fraction))

    def random_individual():
        return dict(rng.choice(all_configs(ev.model, fraction)))

    def tournament(pop, fits):
        cand = rng.sample(range(len(pop)), min(GA_TOURNAMENT_K, len(pop)))
        return dict(pop[max(cand, key=lambda i: fits[i])])

    def crossover(p1, p2):
        pt = rng.randint(1, len(G) - 1)
        return {g: (p1[g] if i < pt else p2[g]) for i, g in enumerate(G)}

    def mutate(ind):
        ind = dict(ind)
        for g in G:
            if rng.random() < GA_MUTATION_RATE:
                ind[g] = rng.choice(space[g])
        return ind

    pop = [random_individual() for _ in range(GA_POP_SIZE)]
    history = []
    for gen in range(1, GA_GENERATIONS + 1):
        print(f"[ga] generation {gen}/{GA_GENERATIONS}", flush=True)
        fits = [ev(ind, config_source="ga")["val_acc"] for ind in pop]
        order = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
        history.append(dict(generation=gen, best_fitness=fits[order[0]],
                            best_config=dict(pop[order[0]]),
                            population=[dict(p) for p in pop],
                            fitnesses=list(fits),
                            unique_evals_so_far=ev.n_evals))
        new_pop = [dict(pop[i]) for i in order[:GA_ELITISM]]
        while len(new_pop) < GA_POP_SIZE:
            new_pop.append(repair(mutate(crossover(tournament(pop, fits),
                                                   tournament(pop, fits)))))
        pop = new_pop
    return best_of(ev), history


def best_of(ev):
    best_k, best_m = max(ev.cache.items(), key=lambda kv: kv[1]["val_acc"])
    return dict(zip(genes(ev.model), best_k)), best_m


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy", choices=["grid", "ga", "random"])
    ap.add_argument("--dataset", default="z24_small")
    ap.add_argument("--model", default="cnn1d", choices=list(SEARCH_SPACES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=14)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--scaler", default=SCALER)
    ap.add_argument("--fraction", default="full", choices=["full", "half"],
                    help="'half' runs a balanced 36-point half-fraction of the grid")
    ap.add_argument("--verbose", type=int, default=0)
    a = ap.parse_args()

    data = load_dataset(a.dataset, seed=a.seed, window_length=a.window,
                        scaler=a.scaler)
    exp = f"{a.strategy}_{a.model}_{a.dataset}"
    ev = Evaluator(data, a.dataset, exp, a.seed, model=a.model,
                   epochs=a.epochs, scaler=a.scaler, verbose=a.verbose)

    t0 = time.perf_counter()
    ga_hist = None
    if a.strategy == "grid":
        best_cfg, best_m = run_grid(ev, a.fraction)
    elif a.strategy == "random":
        best_cfg, best_m = run_random(ev, a.budget, a.seed, a.fraction)
    else:
        (best_cfg, best_m), ga_hist = run_ga(ev, a.seed, a.fraction)
    elapsed = time.perf_counter() - t0

    out = dict(strategy=a.strategy, model=a.model, dataset=a.dataset,
               seed=a.seed, epochs=a.epochs, window=a.window, scaler=a.scaler,
               fraction=a.fraction,
               unique_evaluations=ev.n_evals, wall_time_sec=elapsed,
               split_digest=ev.digest, best_config=best_cfg,
               best_val_acc=best_m["val_acc"], best_test_acc=best_m["test_acc"],
               best_test_macro_f1=best_m["test_macro_f1"],
               trace=ev.trace, ga_history=ga_hist)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results",
                        f"search_{a.strategy}_{a.model}_{a.dataset}_seed{a.seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(f"\n[{a.strategy}/{a.model}] best {best_cfg}")
    print(f"  val_acc {best_m['val_acc']:.4f}  test_acc {best_m['test_acc']:.4f}"
          f"  macro-F1 {best_m['test_macro_f1']:.4f}")
    print(f"  {ev.n_evals} unique evaluations in {elapsed/60:.1f} min")
    print(f"  -> {os.path.basename(path)}")


if __name__ == "__main__":
    main()
