#!/bin/bash
# Reproduce every result in the paper.
#
#   bash reproduce.sh tables     # tables, figures and number check from results/ (no GPU, no data)
#   bash reproduce.sh all        # re-run every experiment first (GPU + raw data; several days)
#
# Every experiment step is skipped when its output JSON already exists, so the
# script can be stopped and restarted at any point. Delete a JSON to re-run it.
# The commands below are the ones that produced the files in results/.

set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
mkdir -p logs
MODE="${1:-tables}"

run() { # $1 = expected output (or "-"), rest = command
  local out="$1"; shift
  if [ "$out" != "-" ] && [ -f "$out" ]; then echo "[skip] $out"; return; fi
  local log="logs/$(echo "$*" | tr ' /' '__' | cut -c1-120).log"
  echo "[run ] $*"
  python -u "$@" > "$log" 2>&1 && echo "[ok  ] $*" || echo "[FAIL] $* (see $log)"
}

experiments() {
  # 0. QUGS raw archives -> data/qugs/{A,B}/stateNN.npy (set RAW_DIR inside the script first)
  run data/qugs/B/state31.npy experiments/prepare_qugs.py

  # 1. Full grids, single-channel input (Tables 5-7, Figures 5-7)
  for s in 0 1 2 3 4; do for d in z24_small qugs_small; do
    run results/search_grid_cnn1d_${d}_seed$s.json experiments/search.py grid --dataset $d --model cnn1d --seed $s; done; done
  for s in 0 1 2; do for d in z24_full qugs_full; do
    run results/search_grid_cnn1d_${d}_seed$s.json experiments/search.py grid --dataset $d --model cnn1d --seed $s; done; done
  for s in 0 1 2; do for d in z24_small qugs_small z24_full qugs_full; do
    run results/search_grid_wavenet_${d}_seed$s.json experiments/search.py grid --dataset $d --model wavenet --seed $s --fraction half; done; done

  # 2. Full grids, multi-channel input (Section 4.1, Table 7)
  for s in 0 1 2; do for d in z24_small_mc z24_full_mc qugs_small_mc qugs_full_mc; do
    run results/search_grid_cnn1d_${d}_seed$s.json experiments/search.py grid --dataset $d --model cnn1d --seed $s
    run results/search_grid_wavenet_${d}_seed$s.json experiments/search.py grid --dataset $d --model wavenet --seed $s --fraction half; done; done

  # 3. Genetic and random search run live (cross-check of the offline replay)
  for s in 0 1 2 3 4; do for d in z24_small qugs_small; do for k in ga random; do
    run results/search_${k}_cnn1d_${d}_seed$s.json experiments/search.py $k --dataset $d --model cnn1d --seed $s; done; done; done
  for s in 0 1 2; do for d in z24_small qugs_small; do for k in ga random; do
    run results/search_${k}_wavenet_${d}_seed$s.json experiments/search.py $k --dataset $d --model wavenet --seed $s --fraction half; done; done; done

  # 4. Classes vs sensors (Table 4, Figure 4)
  for s in 0 1 2; do for m in cnn1d wavenet; do
    run results/factorial_${m}_c05_s33_seed$s.json experiments/factorial.py --model $m --seed $s; done; done

  # 5. Leave-one-setup-out cross-validation on Z24 (Table 8); settings are taken from the grids
  run results/loso_cnn1d_z24_small.json      experiments/loso_cv.py --model cnn1d --classes 5
  run results/loso_cnn1d_z24_full.json       experiments/loso_cv.py --model cnn1d --classes 15 --stride-mult 8
  run results/loso_cnn1d_z24_small_mc.json   experiments/loso_cv.py --model cnn1d --classes 5 --sensors 5 --multichannel
  run results/loso_cnn1d_z24_full_mc.json    experiments/loso_cv.py --model cnn1d --classes 15 --sensors 5 --multichannel
  run results/loso_wavenet_z24_small.json    experiments/loso_cv.py --model wavenet --classes 5 --sensors 5
  run results/loso_wavenet_z24_full.json     experiments/loso_cv.py --model wavenet --classes 15 --sensors 5 --stride-mult 8
  run results/loso_wavenet_z24_small_mc.json experiments/loso_cv.py --model wavenet --classes 5 --sensors 5 --multichannel
  run results/loso_wavenet_z24_full_mc.json  experiments/loso_cv.py --model wavenet --classes 15 --sensors 5 --multichannel

  # 6. Input formulation with a fixed setting (Table 3, Figure 3) and QUGS sensor sweep (Table 9)
  run results/diag_multichannel.json   experiments/diagnostics/diag_multichannel.py
  run results/qugs_channel_curve.json  experiments/diagnostics/diag_qugs_channels.py
  run results/diag_z24full.json        experiments/diagnostics/diag_z24full.py
  run results/baseline_psd_z24_small_seed0.json src/baseline_psd.py z24_small

  # 7. Per-epoch curves for the offline replay (needs results/details/, written by the runs above)
  run results/curves.json experiments/extract_curves.py
}

tables() {
  # 8. Offline replay of the seven search algorithms (Table 7, Figure 7)
  run results/hpo_offline.json analysis/hpo_offline.py --repeats 100 --budgets 6 10 14 20 28
  # 9. Figures 1-7 -> paper/figures/
  run - analysis/make_figures.py
  # 10. Every number quoted in the paper, recomputed -> logs/
  run - analysis/verify_numbers.py
  echo "Number check written to logs/analysis_verify_numbers.py.log"
}

case "$MODE" in
  all)    experiments; tables ;;
  tables) tables ;;
  *) echo "usage: bash reproduce.sh [tables|all]"; exit 1 ;;
esac
