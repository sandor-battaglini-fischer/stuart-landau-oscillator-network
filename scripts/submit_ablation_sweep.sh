#!/bin/bash
# Submit nonlinearity ablation as 3 task-split jobs (CPU workers = cores).
# Usage:
#   ./scripts/submit_ablation_sweep.sh
#   EPOCHS=200 NUM_RUNS=10 WORKERS=16 ./scripts/submit_ablation_sweep.sh
#   TASKS_TO_RUN=imdb,smnist EPOCHS=200 ./scripts/submit_ablation_sweep.sh
# Resume: re-run the same script (--skip-existing).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${ROOT}/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"

EPOCHS="${EPOCHS:-100}"
NUM_RUNS="${NUM_RUNS:-10}"
WORKERS="${WORKERS:-16}"
IMDB_WORKERS="${IMDB_WORKERS:-8}"
MEM="${MEM:-64}"
TIME_LIMIT="${TIME_LIMIT:-100:00}"
SWEEP_ROOT="${SWEEP_ROOT:-${ROOT}/results/sweeps/ablation_run_${EPOCHS}}"

submit_task() {
  local task="$1"
  local short="$2"
  local workers="$3"
  local job="ablation-${short}"
  local sweep_dir="${SWEEP_ROOT}_${task}"
  echo "Submitting ${job} -> ${sweep_dir}  (-c ${workers}, --workers ${workers})"
  run -t "$TIME_LIMIT" -c "$workers" -m "$MEM" -j "$job" \
    -o "${job}.out" -e "${job}.err" -- \
    "$PYTHON" experiments/sweeps/nonlinearity_ablation_sweep.py \
      --epochs "$EPOCHS" \
      --num-runs "$NUM_RUNS" \
      --workers "$workers" \
      --skip-existing \
      --tasks "$task" \
      --sweep-dir "$sweep_dir"
}

TASKS_TO_RUN="${TASKS_TO_RUN:-imdb,smnist,mackey_glass}"

IFS=',' read -r -a TASK_LIST <<< "$TASKS_TO_RUN"
for task in "${TASK_LIST[@]}"; do
  case "$task" in
    imdb) submit_task imdb imdb "$IMDB_WORKERS" ;;
    smnist) submit_task smnist smnist "$WORKERS" ;;
    mackey_glass|mg) submit_task mackey_glass mg "$WORKERS" ;;
    *) echo "Unknown task: $task" >&2; exit 1 ;;
  esac
done

echo
echo "Per-task dirs under: ${SWEEP_ROOT}_<task>"
echo "Live plots update after each completed seed:"
echo "  ${SWEEP_ROOT}_<task>/nonlinearity_ablation_performance.png"
echo "  ${SWEEP_ROOT}_<task>/ablation_epochs.png"
echo "After all jobs finish, merge + plot:"
echo "  $PYTHON experiments/sweeps/nonlinearity_ablation_sweep.py \\"
echo "    --sweep-dir ${SWEEP_ROOT} \\"
echo "    --combine-from \\"
echo "      imdb:${SWEEP_ROOT}_imdb \\"
echo "      smnist:${SWEEP_ROOT}_smnist \\"
echo "      mackey_glass:${SWEEP_ROOT}_mackey_glass \\"
echo "    --plot-only"
