#!/bin/bash
# Submit the nonlinearity ablation as one Slurm job per task. Rerunning resumes (--skip-existing).
#
#   SBATCH_ACCOUNT=<account> EPOCHS=200 NUM_RUNS=10 ./scripts/submit_ablation_sweep.sh
#   TASKS_TO_RUN=imdb,smnist ./scripts/submit_ablation_sweep.sh
#
# When all jobs are done, merge and plot:
#   python experiments/sweeps/nonlinearity_ablation_sweep.py --sweep-dir <SWEEP_ROOT> --plot-only \
#       --combine-from imdb:<SWEEP_ROOT>_imdb smnist:<SWEEP_ROOT>_smnist mackey_glass:<SWEEP_ROOT>_mackey_glass
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-100}"
NUM_RUNS="${NUM_RUNS:-10}"
WORKERS="${WORKERS:-16}"
IMDB_WORKERS="${IMDB_WORKERS:-8}"
MEM="${MEM:-64G}"
TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
SWEEP_ROOT="${SWEEP_ROOT:-$ROOT/results/sweeps/ablation_run_${EPOCHS}}"
TASKS_TO_RUN="${TASKS_TO_RUN:-imdb,smnist,mackey_glass}"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a TASK_LIST <<< "$TASKS_TO_RUN"
for task in "${TASK_LIST[@]}"; do
  workers="$WORKERS"
  [[ "$task" == imdb ]] && workers="$IMDB_WORKERS"
  sbatch --job-name "ablation-$task" --cpus-per-task "$workers" --mem "$MEM" --time "$TIME_LIMIT" \
    --chdir "$ROOT" --output "$LOG_DIR/ablation-$task-%j.out" \
    --wrap "$PYTHON experiments/sweeps/nonlinearity_ablation_sweep.py --epochs $EPOCHS --num-runs $NUM_RUNS \
      --workers $workers --skip-existing --tasks $task --sweep-dir ${SWEEP_ROOT}_$task"
done
