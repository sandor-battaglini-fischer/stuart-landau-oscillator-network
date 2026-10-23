# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code (PhD project, IFISC) for SLON — a recurrent neural network of coupled Stuart-Landau
oscillators, trained end-to-end on sequence tasks. There is no test suite, linter config, or
package manifest (`pyproject.toml`/`setup.py`); this is experiment/analysis code, not a library.

## Setup and running

Python 3.11, dependencies in `requirements.txt` (PyTorch, NumPy, SciPy, matplotlib, pandas,
scikit-learn, imageio, pillow). Use `uv`:

```bash
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt
```

**Always run scripts from the repository root** — `training/`, `experiments/`, and `analysis/`
scripts each hand-append `PROJECT_ROOT` (computed from `__file__`) to `sys.path` at the top rather
than relying on the package being installed, so absolute imports like `from utils.model_factory
import build_oscillator` only resolve when invoked as `python training/train_smnist.py ...` from
root.

```bash
python training/train_smnist.py --epochs 50 --num-hidden 50
python training/train_imdb.py --epochs 20 --num-hidden 50
python training/train_mackey_glass.py --epochs 100 --num-hidden 50
```

Data downloads/generates into `data/` on first run; training writes to `results/<task>/<timestamp>/`.
Both directories are gitignored. There's no automated test command — verification is "run a short
training job and inspect `results/.../log.txt` and the generated plots," or use
`examples/slon_dynamics_demo.py` for a quick single-unit sanity check.

## Architecture

**Three oscillator dynamics share one interface**, selected via `--dynamics {sl,lo,dho}` and built
through `utils/model_factory.build_oscillator(...)` — never instantiate `SLON`/`HORN` directly from
a training script; go through the factory so the `dho` parameter remapping happens:

- `sl` — full nonlinear Stuart-Landau (`models/stuart_landau.py::SLON`): complex-valued state
  `z = x + iy` per unit, cubic saturation term `gamma * |z|^2 * z`.
- `lo` — same `SLON` class with `linear_dynamics=True`, i.e. the Stuart-Landau equation with the
  nonlinear term dropped (linear oscillator baseline).
- `dho` — a differently-parameterized damped harmonic oscillator (`models/horn.py::HORN`, real
  `(x, y)` state, no complex arithmetic). `build_oscillator` maps Stuart-Landau's `(lambda, omega)`
  onto HORN's `(gamma, Omega)` via `dho_params_matching_linear_sl` so poles match the *linear* SL
  case — this is the only place that conversion happens.

All three follow the same explicit-Euler step pattern: `pre_act = i2h(input) + gain_rec * h2h(state)`,
squash through `tanh` (unless `--no-tanh`) and scale by `alpha` to get the input forcing, then
integrate the ODE with step size `h`. `gain_rec = 1/sqrt(num_nodes)` is a fixed (non-learned)
recurrent scaling. Readout `h2o` is linear over the concatenated state (`[real, imag]` for SL, `x`
only for HORN).

Dynamics parameters (`h`, `alpha`, `omega`, `lambda_param`, `gamma_real`, `gamma_imag`) are plain
tensors set at construction time, not `nn.Parameter`s — only `i2h`, `h2h`, `h2o` are learned.
`lambda_param` must stay negative (decaying linear part) or the SL dynamics diverge over long
sequences (784 steps for sMNIST) and logits go to NaN; training scripts print a warning if you pass
a positive value.

**`experiments/` vs `analysis/`** (both read the same `results/` tree, but for different purposes):
- `experiments/` launches training — grid searches and sweeps that shell out to the `training/*.py`
  scripts via `subprocess`, then aggregate the resulting `results/<task>/<timestamp>*/log.txt` and
  `metrics.json` files. `experiments/sweeps/sweep_common.py` holds shared log-parsing helpers
  (`parse_run_log_header`, metric extraction regexes) reused across sweep scripts — extend it rather
  than re-parsing logs per-script.
- `analysis/` never trains; it reads existing `results/` output or precomputed JSON and produces
  figures. `analysis/dynamics/` is the exception — it's analytical/numerical bifurcation work on the
  oscillator equations themselves, independent of any trained model.

**Manifold/dimension analysis** (`utils/manifold_dimension_analysis.py`: PCA, correlation dimension,
Lyapunov exponents) runs automatically during training every 10 epochs and at the end
(`--analyze-manifold`, default on) — disable with `--skip-epoch-plots` for faster sweep runs
(`--sweep-mode` sets this implicitly and reduces output to just `log.txt`/`metrics.json`).

**Plotting**: `utils/plotting.py` is a thin re-export shim (`from utils.plotting_utils import *`) —
the actual per-task plotting code lives in `utils/plotting_utils/*.py`, split by task
(`smnist_pixel_reconstruction.py`, `mackey_glass.py`, `sentiment_encoding.py`, ...) plus shared
concerns (`style.py` for the shared matplotlib theme/colors, `gifs.py`, `training.py`). Import from
`utils.plotting_utils` (or the `utils.plotting` shim); don't reach into the submodules directly
unless the symbol isn't re-exported.

`scripts-to-incorporate/` is legacy/pre-refactor code superseded by `training/`, `experiments/`, and
`analysis/` — don't build on it, and don't be surprised if it duplicates or conflicts with the
current code.
