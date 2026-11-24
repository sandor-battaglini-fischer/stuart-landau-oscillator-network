#!/bin/bash
# Submit the lambda / gamma_real grids as one Slurm job per task. Rerunning resumes (finished points are skipped).
#
#   PRESET=regime SBATCH_ACCOUNT=<account> ./scripts/submit_lga_grid.sh   # Fig. 4
#   PRESET=alpha  SBATCH_ACCOUNT=<account> ./scripts/submit_lga_grid.sh   # Figs. S2-S5
#   PYTHON="apptainer exec /path/to/pytorch.sif python" WORKERS=32 ./scripts/submit_lga_grid.sh
#
# Afterwards: python experiments/grid_search/lga_grid.py summary --run-dirs results/grid_search/<preset>_*
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PRESET="${PRESET:-alpha}"
PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-16}"
MEM="${MEM:-64G}"
TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"

case "$PRESET" in
  regime)  # Fig. 4: alpha = 0.04, averaged over 9 values of gamma_imag
    COMMON="--lambda-min -0.5 --lambda-max 0.5 --lambda-step 0.1
      --gamma-real-min -0.1 --gamma-real-max 0.1 --gamma-real-step 0.02
      --gamma-imag-values -0.2 -0.15 -0.1 -0.05 0 0.05 0.1 0.15 0.2
      --alpha-values 0.04 --num-seeds 1 --epochs 25 --early-stop-patience 10"
    declare -A NUM_HIDDEN=([imdb]=9 [smnist]=50 [mg_h1]=50 [mg_h25]=50) ;;
  alpha)   # Figs. S2-S5: gamma_imag = 0, four input gains
    COMMON="--lambda-min -1 --lambda-max 1 --lambda-step 0.1
      --gamma-real-min -0.2 --gamma-real-max 0.2 --gamma-real-step 0.02
      --gamma-imag-values 0 --alpha-values 0.04 0.5 1 5
      --num-seeds 10 --epochs 25 --early-stop-patience 10"
    declare -A NUM_HIDDEN=([imdb]=25 [smnist]=25 [mg_h1]=25 [mg_h25]=25) ;;
  *) echo "unknown PRESET=$PRESET (regime or alpha)" >&2; exit 1 ;;
esac

declare -A TASK_ARGS=(
  [imdb]="--tasks imdb"
  [smnist]="--tasks smnist"
  [mg_h1]="--tasks mackey_glass --mg-horizon 1 --mg-tau 17 --mg-input-length 100"
  [mg_h25]="--tasks mackey_glass --mg-horizon 25 --mg-tau 17 --mg-input-length 100"
)

for job in imdb smnist mg_h1 mg_h25; do
  sbatch --job-name "lga-$PRESET-$job" --cpus-per-task "$WORKERS" --mem "$MEM" --time "$TIME_LIMIT" \
    --chdir "$ROOT" --output "$LOG_DIR/lga-$PRESET-$job-%j.out" \
    --wrap "$PYTHON experiments/grid_search/lga_grid.py local $(echo $COMMON) ${TASK_ARGS[$job]} \
      --num-hidden ${NUM_HIDDEN[$job]} --workers $WORKERS --run-dir results/grid_search/${PRESET}_$job"
done
