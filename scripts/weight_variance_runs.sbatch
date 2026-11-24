#!/bin/bash
#SBATCH --job-name=weight-variance
#SBATCH --array=0-39%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/weight-variance-%A_%a.out
# Runs for Fig. S6: 10 seeds of each task at its best setting (Table I) with N=128.
# Each run keeps initial_model.pt and last_model.pt.
#
#   mkdir -p logs && sbatch --account=<account> --export=ALL,PYTHON="python" scripts/weight_variance_runs.sbatch
#   python analysis/network_utilisation.py --variance-runs "IMDb=results/weight_variance/imdb_*" \
#       "sMNIST=results/weight_variance/smnist_*" "MG \$H=1\$=results/weight_variance/mg_h1_*" \
#       "MG \$H=25\$=results/weight_variance/mg_h25_*"
set -euo pipefail
PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-40}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" TORCH_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

configs=(imdb smnist mg_h1 mg_h25)
config=${configs[$((SLURM_ARRAY_TASK_ID / 10))]}
seed=$((SLURM_ARRAY_TASK_ID % 10 + 1))
out=results/weight_variance/${config}_seed${seed}
common="--num-hidden 128 --epochs $EPOCHS --early-stop-patience 10 --skip-epoch-plots --seed $seed --output-dir $out"

case "$config" in
  imdb)   $PYTHON training/train_imdb.py $common --lambda-param -0.08 --gamma-real -0.1 --gamma-imag 0.08 ;;
  smnist) $PYTHON training/train_smnist.py $common --lambda-param 0.0 --gamma-real -0.1 --gamma-imag 0.0 ;;
  mg_h1)  $PYTHON training/train_mackey_glass.py $common --horizon 1 --input-length 100 --omega 0.15 --lambda-param -0.2 \
            --gamma-real -0.1 --gamma-imag 0.0 ;;
  mg_h25) $PYTHON training/train_mackey_glass.py $common --horizon 25 --input-length 100 --omega 0.15 --lambda-param -0.1 \
            --gamma-real -0.1 --gamma-imag 0.0 ;;
esac
