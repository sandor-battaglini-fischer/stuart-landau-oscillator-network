#!/usr/bin/env bash
# Train sMNIST (limit-cycle SL) then measure memory capacity.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
OUT_DIR="${ROOT}/results/memory_capacity/ckpts/smnist_sl_lc"

$PYTHON training/train_smnist.py \
  --dynamics sl \
  --num-hidden 50 \
  --alpha 0.04 \
  --omega 0.224 \
  --lambda-param 0.1 \
  --gamma-real -0.18 \
  --gamma-imag 0.0 \
  --epochs 40 \
  --early-stop-patience 8 \
  --skip-epoch-plots \
  --output-dir "$OUT_DIR"

$PYTHON analysis/memory_capacity.py \
  --task smnist \
  --run-dir "$OUT_DIR" \
  --probes noise,task \
  --compare-init \
  --seq-length 3000 \
  --batch-size 8 \
  --max-delay 100 \
  --washout 100 \
  --noise-std 0.5
