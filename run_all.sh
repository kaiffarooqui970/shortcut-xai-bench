#!/usr/bin/env bash
# Reproduce everything from a clean clone. Safe to rerun: cached stages are skipped.
set -euo pipefail
cd "$(dirname "$0")"

# The venv and the data/cache live outside the repo (default: ~/.venvs and
# ~/.cache) so a checkout inside a synced folder (iCloud Desktop, Dropbox)
# cannot evict them mid-run. Override with VENV / SHORTCUT_BENCH_HOME.
PY_BIN="${PYTHON:-python3.12}"
VENV="${VENV:-$HOME/.venvs/shortcut-xai-bench}"
export SHORTCUT_BENCH_HOME="${SHORTCUT_BENCH_HOME:-$HOME/.cache/shortcut-xai-bench}"
if [ ! -x "$VENV/bin/python" ]; then
  echo "[setup] creating venv at $VENV with $PY_BIN"
  "$PY_BIN" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet -r requirements.txt
fi
PY="$VENV/bin/python"

echo "[test] unit tests"
$PY -m pytest -q tests

echo "[stage 1] build splits + dump injection samples"
$PY scripts/dump_samples.py

# Pass --wandb through to log to Weights & Biases:  bash run_all.sh --wandb
WANDB_FLAG=""
for a in "$@"; do [ "$a" = "--wandb" ] && WANDB_FLAG="--wandb"; done

echo "[stage 2a] learnability pre-check (rate 1.0, seed 0, every model x patch type)"
$PY scripts/learnability_check.py $WANDB_FLAG

echo "[stage 3a] attribution timing on the pre-check checkpoints"
$PY scripts/time_attribution.py

echo "[stage 5a] sweep: train + attribute every cell (resumable; cached cells are skipped)"
$PY scripts/run_sweep.py $WANDB_FLAG

echo "[stage 5b] reliance comparison CNN vs ViT"
$PY scripts/reliance_report.py

echo "[stage 5c] sanity control: clean-trained models on patched images (trains rate-0 models; cached)."
echo "           Runs before scoring: results.csv takes its ctrl_*/delta_* columns from these models."
$PY scripts/sanity_control.py

echo "[stage 5d] score heatmaps against the ground-truth masks"
$PY scripts/score_results.py

echo "[stage 5e] figures"
$PY scripts/plot_results.py
[ -n "$WANDB_FLAG" ] && $PY scripts/score_results.py --wandb   # attach scores + figures to W&B

echo "[stage 6] channel-reduction sensitivity (rate 1.0, seed 0 slice)"
$PY scripts/reduction_sensitivity.py

echo "[extra] patch-region content proxy"
$PY scripts/patch_content_proxy.py

echo "done: results/results.csv, results/runs.csv, results/figures/, results/runtime.json,"
echo "      results/sanity_control.md, results/reduction_sensitivity.md"
