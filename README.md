# Stuart-Landau Oscillator Network (SLON)

A low-dimensional recurrent neural network built from coupled Stuart-Landau oscillators, inspired by neuromorphic photonic architectures. The model is trained end-to-end on sequence tasks while retaining a clear dynamical-systems interpretation of its internal states.

This repository is the code companion to the paper *TBD* (IFISC). It supports training, hyperparameter search, and post-hoc analysis across three benchmark tasks: sequential MNIST, IMDb sentiment classification, and Mackey-Glass time-series prediction.

## Model

The core module is `SLON` (`models/stuart_landau.py`): a complex-valued oscillator network with input, recurrent, and readout layers. Each hidden unit evolves according to Stuart-Landau dynamics driven by a tanh-gated input force.


| Symbol                     | Role                                                 |
| -------------------------- | ---------------------------------------------------- |
| `h`                        | Discrete time step (microscopic time constant)       |
| `alpha`                    | Input excitability                                   |
| `lambda_param`             | Real part of the linear coefficient (growth/damping) |
| `omega`                    | Natural frequency                                    |
| `gamma_real`, `gamma_imag` | Nonlinear saturation coefficients                    |


For IMDb, a `SLONWithEmbedding` wrapper (`training/train_imdb.py`) adds a word-embedding layer in front of the oscillator core.

Natural frequency is typically set from the stimulus period: `omega ≈ 2π / (sequence_length × h)`.

## Tasks


| Task             | Script                           | Input                                        | Output           |
| ---------------- | -------------------------------- | -------------------------------------------- | ---------------- |
| Sequential MNIST | `training/train_smnist.py`       | 784-pixel scanline (1-D time series)         | 10 digit classes |
| IMDb sentiment   | `training/train_imdb.py`         | Token sequences (GloVe or random embeddings) | Binary sentiment |
| Mackey-Glass     | `training/train_mackey_glass.py` | Past time-series window                      | Future value     |


## Setup

**Requirements:** Python 3.10+. Dependencies are listed in `requirements.txt` (PyTorch, torchvision, NumPy, SciPy, Matplotlib, pandas, tqdm, scikit-learn, imageio, pillow).

Install [uv](https://docs.astral.sh/uv/) if you don't have it yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv
```

Create a virtual environment and install dependencies:

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Run scripts with the activated environment:

```bash
python training/train_smnist.py --epochs 50 --num-hidden 50
```

Or without activating the venv:

```bash
uv run --with-requirements requirements.txt python training/train_smnist.py --epochs 50 --num-hidden 50
```

**Data** (stored under `data/`, gitignored):

- **MNIST** — downloaded automatically on first run (or place under `data/MNIST/`).
- **IMDb** — downloaded automatically from Stanford ACL archive; GloVe embeddings fetched on demand into `data/glove/`.
- **Mackey-Glass** — generated synthetically inside the training script.

Run all commands from the repository root so that `models/` and `utils/` resolve correctly.

## Quick start

```bash
# Sequential MNIST
python training/train_smnist.py --epochs 50 --num-hidden 50

# IMDb sentiment
python training/train_imdb.py --epochs 20 --num-hidden 50

# Mackey-Glass prediction
python training/train_mackey_glass.py --epochs 100 --num-hidden 50
```

Each training script accepts hyperparameters for dynamics (`--h`, `--alpha`, `--omega`, `--lambda-param`, `--gamma-real`, `--gamma-imag`) and supports single-parameter sweeps (e.g. `--sweep-omega`). Use `--help` on any script for the full option list.

**Dynamics demo** — visualise oscillator responses to a sinusoidal pulse:

```bash
python examples/slon_dynamics_demo.py
```

## Training outputs

Runs are saved under `results/<task>/<timestamp>/` (gitignored). A typical run directory contains:

```
results/smnist/20260619_160133/
├── log.txt                          # epoch-by-epoch metrics
├── metrics.json                     # structured training history
├── parameters.json                  # per-epoch weight and dynamics snapshots
├── best_model.pt / last_model.pt    # model checkpoints
├── loss_and_accuracy_over_time.png
├── parameter_evolution_*.png        # weight and dynamics trajectories
├── weight_heatmap_*.png
└── epochs/
    └── epoch00/
        ├── state_space_2d_epoch00.png
        ├── correlation_dimension_epoch00.png
        ├── pca_analysis_epoch00.png
        └── ...
```

Manifold-dimension analysis (PCA, correlation dimension, Lyapunov exponents) runs every 10 epochs and at the end of training when `--analyze-manifold` is enabled (default).

Regenerate parameter plots from a saved run:

```bash
python scripts/plot_parameters_retrospective.py results/smnist/20260619_160133/
```

## Repository structure

```
stuart-landau-oscillator-network/
├── models/
│   └── stuart_landau.py             # SLON core module
├── training/
│   ├── train_smnist.py              # Sequential MNIST
│   ├── train_imdb.py                # IMDb (+ SLONWithEmbedding)
│   └── train_mackey_glass.py        # Mackey-Glass regression
├── utils/
│   ├── run_dirs.py                  # Timestamped run directories, checkpoints
│   ├── slon_analysis.py             # Parameter extraction and statistics
│   ├── manifold_dimension_analysis.py  # State-space / dimension analysis
│   ├── paths.py                     # Project root helper
│   └── plotting_utils/              # Shared plotting (classification, regression, GIFs)
├── experiments/                     # Larger-scale studies
│   ├── grid_search/                 # Grid search over dynamics parameters
│   │   ├── smnist.py
│   │   ├── imdb.py
│   │   └── mackey_glass.py
│   ├── sweeps/                      # Cross-task parameter sweeps
│   │   ├── omega_sweep.py
│   │   └── alpha_sweep.py
│   ├── mackey_glass/
│   │   └── comparison.py            # Multi-configuration Mackey-Glass runs
│   └── multi_task/
│       └── scaling.py               # Hidden-size scaling across all tasks
├── analysis/                        # Post-training figure generation
│   ├── smnist/                      # Log and sweep analysis
│   ├── imdb/
│   ├── mackey_glass/                # Horizon comparison, interactive explorer
│   ├── dynamics/                    # Bifurcation diagrams, memory kernel, regimes
│   ├── multi_task/                  # Cross-task heatmaps, ablations, scaling plots
│   └── input_signal_analysis.py     # Pre-nonlinearity input statistics
├── examples/
│   └── slon_dynamics_demo.py        # Single-unit dynamics visualisation
├── scripts/
│   └── plot_parameters_retrospective.py
├── scripts-to-incorporate/          # Legacy scripts (superseded by analysis/ and experiments/)
├── data/                            # Datasets and caches (gitignored)
└── results/                         # Training and experiment outputs (gitignored)
```

### `experiments/` vs `analysis/`

- `**experiments/**` — launch training runs or sweeps (often calling into `training/` scripts via subprocess). Use these for systematic hyperparameter search and scaling studies.
- `**analysis/**` — read existing `results/` or pre-computed JSON logs and produce publication figures. Most scripts have their own CLI; run with `--help` where available.

Notable analysis entry points:


| Script                                           | Purpose                                  |
| ------------------------------------------------ | ---------------------------------------- |
| `analysis/smnist/log_analysis.py`                | Parse and plot sMNIST training logs      |
| `analysis/imdb/sweep_analysis.py`                | Visualise IMDb omega/lambda sweeps       |
| `analysis/mackey_glass/horizon_comparison.py`    | Compare prediction horizons              |
| `analysis/mackey_glass/interactive.py`           | Interactive Mackey-Glass explorer        |
| `analysis/multi_task/heatmaps.py`                | Cross-task performance heatmaps          |
| `analysis/multi_task/ablation_curves.py`         | Ablation study curves                    |
| `analysis/multi_task/network_utilisation.py`     | Hidden-unit utilisation comparison       |
| `analysis/dynamics/stuart_landau_bifurcation.py` | Bifurcation analysis                     |
| `analysis/dynamics/memory_kernel.py`             | Memory-kernel characterisation           |
| `analysis/input_signal_analysis.py`              | Input statistics before the nonlinearity |


## Citation

```bibtex
@article{slon2026,
  title   = {TBD},
  author  = {TBD},
  journal = {TBD},
  year    = {2026}
}
```

