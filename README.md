# Relative importance of input formulation and hyperparameter optimisation in vibration-based SHM

Code, results and manuscript for the paper of the same name. It compares three decisions in
deep-learning damage detection on the **Z24** bridge and **QUGS** grandstand
benchmarks, with training and test data always taken from different measurements:

| decision | effect on test macro-F1 |
|---|---|
| multi-channel instead of single-channel input | +0.48 (Z24, 15 classes: 0.30 → 0.78) |
| best instead of typical hyperparameters | +0.12 to +0.32 (6 of 8 cases) |
| Bayesian instead of random search, same cost | about +0.02 |
| genetic instead of random search, same cost | within ±0.01 |

The manuscript is in [`paper/tex/`](paper/tex/) (LaTeX source and compiled PDF);
[`paper/PAPER_COMPACT.md`](paper/PAPER_COMPACT.md) is the same text in Markdown.

## Reproducing the paper

```bash
pip install -r requirements.txt
bash reproduce.sh tables   # every table, figure and quoted number, from results/ (minutes, no GPU, no data)
bash reproduce.sh all      # re-run every experiment first (GPU and raw data; several days)
```

`reproduce.sh` lists the exact command behind every file in `results/`, skips any
step whose output already exists, and finishes by running
`analysis/verify_numbers.py`, which recomputes every number quoted in the paper.

| paper item | produced by | from |
|---|---|---|
| Table 3, Figure 3 | `experiments/diagnostics/diag_multichannel.py` | `results/diag_multichannel.json` |
| Table 4, Figure 4 | `experiments/factorial.py` | `results/factorial_*.json` |
| Table 5, Figure 5, transfer loss | `experiments/search.py grid` | `results/search_grid_*.json` |
| Table 6, Figure 6 | `experiments/search.py grid` | `results/search_grid_*.json` |
| Table 7, Figure 7 | `analysis/hpo_offline.py` | `results/curves.json`, `results/search_grid_*.json` |
| Table 8 | `experiments/loso_cv.py` | `results/loso_*.json` |
| Table 9 | `experiments/diagnostics/diag_qugs_channels.py` | `results/qugs_channel_curve.json` |
| Figures 1–7 | `analysis/make_figures.py` | all of the above |
| all quoted numbers | `analysis/verify_numbers.py` | all of the above |

Set `PYTHONIOENCODING=utf-8` on Windows (`reproduce.sh` does this): some paths
contain non-ASCII characters.

### Raw data (only for `reproduce.sh all`)

- **Z24** — https://bwk.kuleuven.be/bwm/z24 — unpack to `data/DatasetPDT/<state>/avt/`.
- **QUGS** — https://onur-avci.com/benchmark — download the archives, set `RAW_DIR`
  in `experiments/prepare_qugs.py`, and run it; it writes `data/qugs/{A,B}/stateNN.npy`.

## Layout

```
src/            data pipeline and models
  shm_data.py       records, measurement-level splits, windowing, leakage assertions
  wavenet_torch.py  WaveNet (PyTorch port of the earlier Keras model, Keras initialisation)
  cnn1d_torch.py    1D-CNN
  baseline_psd.py   log-PSD + logistic regression reference classifier
  runlog.py         append-only run log (results/runs.csv)
experiments/    one script per experiment, all resumable
  search.py         grid / genetic / random search over the 72-setting space
  factorial.py      number of classes vs number of sensors
  loso_cv.py        leave-one-setup-out cross-validation on Z24
  extract_curves.py per-epoch curves for the offline search replay
  prepare_qugs.py   QUGS archives -> .npy
  control_*.py      split-protocol controls (measurement-level vs random-window split)
  diagnostics/      fixed-setting probes (Tables 3 and 9 and the pipeline choices)
analysis/       hpo_offline.py, make_figures.py, verify_numbers.py, analyze.py
results/        one JSON per experiment, runs.csv, curves.json
paper/          manuscript (tex/), figures/, Markdown text
```

Not in the repository: `data/` (3.5 GB, re-downloadable), `results/details/`
(94 MB of per-run dumps; only `curves.json` is extracted from it), trained weights,
and `un_use/` (superseded drafts, results and scripts, kept locally).

## Deliberately leaky code paths

`src/shm_data.py` has two options that leak on purpose, used only by
`experiments/control_original.py` and `experiments/control_leaky.py` to measure how
much accuracy a leaky protocol adds: `split_scheme="random_window"` and
`scaler="global_all"` / `"per_timestep_all"`. Both announce themselves on stdout and
are excluded from the leakage assertion. No reported result uses them.
