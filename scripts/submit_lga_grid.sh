#!/bin/bash
# Submit 3 split LGA grid jobs (one per task) with a shared live heatmap dir.
# Usage: ./scripts/submit_lga_grid.sh
# Resume later: re-run the same script (completed jobs are skipped).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${ROOT}/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"

HEATMAP_DIR="${HEATMAP_DIR:-${ROOT}/results/grid_search/lga_live/heatmaps}"
WORKERS="${WORKERS:-16}"
CORES="${CORES:-16}"
MEM="${MEM:-64}"
TIME_LIMIT="${TIME_LIMIT:-100:00}"

COMMON=(
  local
  --lambda-min -1 --lambda-max 1 --lambda-step 0.1
  --gamma-real-min -0.2 --gamma-real-max 0.2 --gamma-real-step 0.02
  --alpha-values 0.04 0.5 1 5
  --num-seeds 5
  --num-hidden 25
  --epochs 20
  --early-stop-patience 5
  --workers "$WORKERS"
  --heatmap-dir "$HEATMAP_DIR"
)

submit_task() {
  local task="$1"
  local job="lga-${task}"
  local run_dir="${ROOT}/results/grid_search/lga_${task}"
  echo "Submitting ${job} -> ${run_dir}"
  run -t "$TIME_LIMIT" -c "$CORES" -m "$MEM" -j "$job" -o "${job}.out" -e "${job}.err" -- \
    "$PYTHON" experiments/grid_search/lga_grid.py \
      "${COMMON[@]}" \
      --tasks "$task" \
      --run-dir "$run_dir"
}

submit_task imdb
submit_task smnist
submit_task mackey_glass

echo
echo "Live heatmaps (all tasks): ${HEATMAP_DIR}/{imdb,smnist,mackey_glass}/"
echo "Optional watcher on login node:"
echo "  $PYTHON experiments/grid_search/lga_grid.py watch \\"
echo "    --run-dirs results/grid_search/lga_imdb results/grid_search/lga_smnist results/grid_search/lga_mackey_glass \\"
echo "    --heatmap-dir results/grid_search/lga_live/heatmaps --interval 30"
