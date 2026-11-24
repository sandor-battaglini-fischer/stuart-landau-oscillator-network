# Stuart-Landau Oscillator Network (SLON)

A recurrent network of coupled Stuart-Landau oscillators, trained end to end on sequential MNIST,
IMDb sentiment classification and Mackey-Glass forecasting. Code for the paper *TBD* (IFISC).

Each hidden unit is a complex oscillator

    dz/dt = (λ + iω) z + (γ_r + iγ_i) |z|² z + α tanh(W_in x + W_rec [Re z, Im z] / √N)

integrated with explicit Euler steps of size `h`. Only the input, recurrent and readout weights are
trained; `λ, ω, γ_r, γ_i, α, h` are fixed hyperparameters. `--dynamics` selects the full model
(`sl`), the same model without the cubic term (`lo`), or a damped harmonic oscillator with matching
linear poles (`dho`). `--no-tanh` removes the input nonlinearity.

## Setup

Python 3.10+.

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

MNIST, IMDb and GloVe are downloaded into `data/` on first use. Mackey-Glass is generated. Run
everything from the repository root; outputs go to `results/`.

## Training

```bash
python training/train_smnist.py --epochs 50 --num-hidden 50 --lambda-param -0.04
python training/train_imdb.py --epochs 20 --num-hidden 50
python training/train_mackey_glass.py --epochs 100 --num-hidden 50 --horizon 25 --input-length 100
python training/train_smnist_pixel_values.py --epochs 20     # reconstruct the whole image from the final state
```

Each run writes `log.txt`, `metrics.json`, checkpoints and per-epoch figures (state space, PCA and
correlation dimension, Lyapunov exponents) to `results/<task>/<timestamp>/`. `--sweep-mode` keeps only
`log.txt` and `metrics.json`; `--output-dir` fixes the run directory. See `--help` for all options.

## Experiments

| Script | What it runs |
| --- | --- |
| `experiments/grid_search/lga_grid.py local` | λ × γ_r × γ_i × α grid with several seeds, plus heatmaps |
| `experiments/sweeps/scaling_sweep.py` | hidden size N |
| `experiments/sweeps/nonlinearity_ablation_sweep.py` | `sl` / `lo` / `dho`, with and without tanh |
| `experiments/sweeps/param_sweep.py --param {omega,alpha}` | natural frequency or input gain |
| `experiments/mg_comparison.py` | Mackey-Glass over horizons, τ and removed Fourier modes |

The sweeps call the training scripts, keep their results in `--sweep-dir`, resume with
`--skip-existing` and redraw figures with `--plot-only`. `lga_grid.py` resumes from its `--run-dir`
and redraws with `lga_grid.py aggregate`. `scripts/` has Slurm wrappers for the grid and the ablation.

## Analysis

| Script | Input |
| --- | --- |
| `analysis/readout_state_analysis.py` | a trained run: node trajectories, final state, t-SNE, decision boundary |
| `analysis/memory_capacity.py` | a trained run or a fresh model: Jaeger memory capacity |
| `analysis/autonomous_dynamics.py` | a trained run or a fresh model: driven vs input-free dynamics |
| `analysis/input_signal_analysis.py` | pre-nonlinearity input statistics per task |
| `analysis/network_utilisation.py` | full runs per task and a scaling sweep: dimension and weight usage |
| `analysis/mg_horizon_comparison.py` | `mg_comparison.py` output |
| `analysis/dynamics/` | bifurcation and regime diagrams of a single oscillator |
| `scripts/plot_parameters_retrospective.py` | a trained run: redraw the weight and parameter plots |

`scripts/mc_pipeline_*.sh` train a model and compute its memory capacity in one go.

## Reproducing the figures

Settings follow Tables IV and V of the paper. Grid and sweep scripts write their figures into their run directory.

| Figure | Command |
| --- | --- |
| 1 | `python analysis/dynamics/stuart_landau_bifurcation.py --analytic-only` |
| 3 | `python analysis/input_signal_analysis.py compare` and `... mg` |
| 4 | `PRESET=regime ./scripts/submit_lga_grid.sh`, then `lga_grid.py summary --run-dirs results/grid_search/regime_*` |
| 5 | `python experiments/sweeps/scaling_sweep.py --epochs 40 --horizons 1 25` |
| 6 | `python experiments/sweeps/nonlinearity_ablation_sweep.py --epochs 200 --horizons 1 25` |
| S8, S9 | the same two sweeps with `--tasks mackey_glass --horizons 1 5 10 15 20 25 30 40 50` |
| 7, 9, 10 | `python analysis/readout_state_analysis.py --task {imdb,smnist,mackey_glass} --run-dir <run>` |
| 8 | `python analysis/dynamics/analytical_stuart_landau_regimes.py` |
| S1 | `python experiments/sweeps/param_sweep.py --param omega` |
| S2–S5 | `PRESET=alpha ./scripts/submit_lga_grid.sh`, then `lga_grid.py summary --run-dirs results/grid_search/alpha_*` |
| S6 | `sbatch scripts/weight_variance_runs.sbatch`, then `analysis/network_utilisation.py --variance-runs ...` |
| S7 | `python training/train_smnist_pixel_values.py` |
| S7 (frozen embeddings) | `python training/train_imdb.py --num-hidden 9 --lambda-param -0.08 --gamma-imag 0.08 --early-stop-patience 10 --freeze-embedding` |

## Layout

```
models/       SLON (Stuart-Landau) and HORN (damped harmonic oscillator)
training/     one script per task
experiments/  grid search and sweeps built on the training scripts
analysis/     figures from trained runs or from the oscillator equations
utils/        model factory, run directories, manifold analysis, plotting
examples/     response of a small untrained network to a sine pulse
```

## Citation

```bibtex
@article{slon2026,
  title   = {TBD},
  author  = {TBD},
  journal = {TBD},
  year    = {2026}
}
```
