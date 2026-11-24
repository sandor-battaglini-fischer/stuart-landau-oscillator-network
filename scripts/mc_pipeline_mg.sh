#!/usr/bin/env bash
# Train Mackey-Glass (best SL ablation setup) then measure memory capacity.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
OUT_DIR="${ROOT}/results/memory_capacity/ckpts/mg_sl_best"

$PYTHON training/train_mackey_glass.py \
  --dynamics sl \
  --num-hidden 50 \
  --alpha 0.04 \
  --omega 0.15 \
  --lambda-param -0.1 \
  --gamma-real -0.1 \
  --gamma-imag 0.0 \
  --horizon 25 \
  --input-length 100 \
  --mg-tau 17 \
  --epochs 80 \
  --early-stop-patience 20 \
  --skip-epoch-plots \
  --output-dir "$OUT_DIR"

$PYTHON analysis/memory_capacity.py \
  --task mackey_glass \
  --run-dir "$OUT_DIR" \
  --probes noise,task \
  --compare-init \
  --seq-length 4000 \
  --batch-size 8 \
  --max-delay 100 \
  --washout 100
