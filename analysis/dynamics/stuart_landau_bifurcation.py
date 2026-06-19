import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.plotting_utils.style import apply_style, thesis_red, thesis_blue, ifisc_green, mycmap
apply_style()


try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(x, *args, **kwargs):
        return x


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["newpx"],
    "text.latex.preamble": r"\usepackage{newpxtext,newpxmath}",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "figure.dpi": 144,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
    }
)


thesis_blue = (0, 0.38, 0.68)
ifisc_green = (0.73, 0.83, 0.01)

mycmap = mcolors.LinearSegmentedColormap.from_list(
    "sl_cmap", [thesis_blue, ifisc_green, thesis_red]
)


def simulate_stuart_landau(
    lambda_param,
    omega,
    gamma_real=-1.0,
    gamma_imag=0.0,
    z0=0.1 + 0.0j,
    dt=0.01,
    t_max=400.0,
):
    n_steps = int(t_max / dt)
    t = np.linspace(0.0, t_max, n_steps, endpoint=False)
    z = np.empty(n_steps, dtype=np.complex128)
    z[0] = z0

    lambda_omega = lambda_param + 1j * omega
    gamma = gamma_real + 1j * gamma_imag

    for i in range(n_steps - 1):
        dz = lambda_omega * z[i] + gamma * (np.abs(z[i]) ** 2) * z[i]
        z[i + 1] = z[i] + dt * dz

    return t, z


def compute_bifurcation(
    lambda_values,
    omega,
    gamma_real=-1.0,
    gamma_imag=0.0,
    dt=0.001,
    t_max=400.0,
    transient_fraction=0.5,
):
    steady_amplitudes = []
    regime_labels = []

    for lam in tqdm(lambda_values, desc="1D bifurcation", leave=False):
        _, z = simulate_stuart_landau(
            lambda_param=lam,
            omega=omega,
            gamma_real=gamma_real,
            gamma_imag=gamma_imag,
            z0=0.1 + 0.0j,
            dt=dt,
            t_max=t_max,
        )

        start_idx = int(len(z) * transient_fraction)
        r = np.abs(z[start_idx:])
        r_mean = np.mean(r)

        steady_amplitudes.append(r_mean)

        if r_mean < 1e-3:
            regime_labels.append("fixed_point")
        else:
            regime_labels.append("limit_cycle")

    return np.array(steady_amplitudes), np.array(regime_labels)


def compute_parameter_hypercube(
    lambda_values,
    gamma_real_values,
    gamma_imag_values,
    omega_values,
    dt=0.03,
    t_max=150.0,
    transient_fraction=0.5,
    diverging_threshold=5.0,
):
    shape = (
        len(omega_values),
        len(gamma_imag_values),
        len(gamma_real_values),
        len(lambda_values),
    )
    regime_codes = np.zeros(shape, dtype=int)
    mean_amplitudes = np.zeros(shape, dtype=float)

    total = (
        len(omega_values)
        * len(gamma_imag_values)
        * len(gamma_real_values)
        * len(lambda_values)
    )

    with tqdm(total=total, desc="4D parameter scan") as pbar:
        for io, omega in enumerate(omega_values):
            for ii, gi in enumerate(gamma_imag_values):
                for ir, gr in enumerate(gamma_real_values):
                    for jl, lam in enumerate(lambda_values):
                        _, z = simulate_stuart_landau(
                            lambda_param=lam,
                            omega=omega,
                            gamma_real=gr,
                            gamma_imag=gi,
                            z0=0.1 + 0.0j,
                            dt=dt,
                            t_max=t_max,
                        )

                        r = np.abs(z)

                        if np.any(r > diverging_threshold) or not np.all(
                            np.isfinite(r)
                        ):
                            regime_codes[io, ii, ir, jl] = 2
                        else:
                            start_idx = int(len(r) * transient_fraction)
                            r_ss = r[start_idx:]
                            r_mean = float(np.mean(r_ss))
                            mean_amplitudes[io, ii, ir, jl] = r_mean

                            if r_mean < 1e-3:
                                regime_codes[io, ii, ir, jl] = 0
                            else:
                                regime_codes[io, ii, ir, jl] = 1

                        pbar.update(1)

    return regime_codes, mean_amplitudes


def compute_parameter_plane_2d(
    lambda_values,
    gamma_real_values,
    omega,
    gamma_imag=0.0,
    dt=0.05,
    t_max=150.0,
    transient_fraction=0.5,
    diverging_threshold=5.0,
):
    regime_codes = np.zeros((len(gamma_real_values), len(lambda_values)), dtype=int)
    mean_amplitudes = np.zeros_like(regime_codes, dtype=float)

    total = len(gamma_real_values) * len(lambda_values)
    with tqdm(total=total, desc="2D lambda-gamma_r plane") as pbar:
        for i, gr in enumerate(gamma_real_values):
            for j, lam in enumerate(lambda_values):
                _, z = simulate_stuart_landau(
                    lambda_param=lam,
                    omega=omega,
                    gamma_real=gr,
                    gamma_imag=gamma_imag,
                    z0=0.1 + 0.0j,
                    dt=dt,
                    t_max=t_max,
                )

                r = np.abs(z)

                if np.any(r > diverging_threshold) or not np.all(np.isfinite(r)):
                    regime_codes[i, j] = 2
                else:
                    start_idx = int(len(r) * transient_fraction)
                    r_ss = r[start_idx:]
                    r_mean = float(np.mean(r_ss))
                    mean_amplitudes[i, j] = r_mean

                    if r_mean < 1e-3:
                        regime_codes[i, j] = 0
                    else:
                        regime_codes[i, j] = 1

                pbar.update(1)

    return regime_codes, mean_amplitudes


def compute_parameter_plane_2d_analytic(
    lambda_values,
    gamma_real_values,
    z0_abs=0.1,
    eps=1e-12,
):
    regime_codes = np.zeros((len(gamma_real_values), len(lambda_values)), dtype=int)
    mean_amplitudes = np.zeros_like(regime_codes, dtype=float)

    for i, gr in enumerate(gamma_real_values):
        for j, lam in enumerate(lambda_values):
            if lam > eps and gr < -eps:
                regime_codes[i, j] = 1
                mean_amplitudes[i, j] = np.sqrt(-lam / gr)
            elif lam > eps and gr >= -eps:
                regime_codes[i, j] = 2
            elif abs(lam) <= eps and gr < -eps:
                regime_codes[i, j] = 0
            elif abs(lam) <= eps and abs(gr) <= eps:
                regime_codes[i, j] = 1
                mean_amplitudes[i, j] = z0_abs
            elif abs(lam) <= eps and gr > eps:
                regime_codes[i, j] = 2
            elif lam < -eps and gr > eps:
                r_unstable = np.sqrt(-lam / gr)
                if z0_abs < r_unstable:
                    regime_codes[i, j] = 0
                else:
                    regime_codes[i, j] = 2
            else:
                regime_codes[i, j] = 0

    return regime_codes, mean_amplitudes


def main():
    omega = 1.0
    gamma_imag = 0.0

    lambda_values = np.linspace(-2.0, 2.0, 201)
    gamma_real_values = np.linspace(-2.0, 2.0, 201)

    regime_plane, mean_amp_plane = compute_parameter_plane_2d(
        lambda_values,
        gamma_real_values,
        omega=omega,
        gamma_imag=gamma_imag,
        dt=0.05,
        t_max=150.0,
        transient_fraction=0.5,
        diverging_threshold=5.0,
    )
    regime_plane_analytic, mean_amp_plane_analytic = compute_parameter_plane_2d_analytic(
        lambda_values,
        gamma_real_values,
        z0_abs=0.1,
    )

    scan_data = {
        "lambda_values": lambda_values.tolist(),
        "gamma_real_values": gamma_real_values.tolist(),
        "regime_plane": regime_plane.tolist(),
        "mean_amplitude_plane": mean_amp_plane.tolist(),
        "regime_plane_analytic": regime_plane_analytic.tolist(),
        "mean_amplitude_plane_analytic": mean_amp_plane_analytic.tolist(),
        "omega": float(omega),
        "gamma_imag": float(gamma_imag),
    }

    with open("stuart_landau_lambda_gamma_r_plane_highres.json", "w") as f:
        json.dump(scan_data, f, indent=2)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    regime_cmap = mcolors.ListedColormap([thesis_blue, ifisc_green, thesis_red])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, regime_cmap.N)

    im = ax.imshow(
        regime_plane,
        origin="lower",
        aspect="equal",
        extent=[
            lambda_values[0],
            lambda_values[-1],
            gamma_real_values[0],
            gamma_real_values[-1],
        ],
        cmap=regime_cmap,
        norm=norm,
    )

    ax.set_xlabel(r"$\lambda$", fontsize=18)
    ax.set_ylabel(r"$\gamma_\mathrm{r}$", fontsize=18)
    ax.tick_params(labelsize=14)
    ax.set_title(
        rf"Stuart--Landau regimes in $(\lambda,\gamma_\mathrm{{r}})$"
        + rf", $\omega={omega:.2f}$, $\gamma_i={gamma_imag:.2f}$",
        fontsize=20,
        pad=10,
    )

    legend_handles = [
        Patch(facecolor=thesis_blue, label="fixed point"),
        Patch(facecolor=ifisc_green, label="limit cycle"),
        Patch(facecolor=thesis_red, label="divergent"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=14,
    )

    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout(rect=[0.0, 0.0, 0.8, 1.0])

    base_name = "stuart_landau_lambda_gamma_r_regimes"
    fig.savefig(f"{base_name}.png", transparent=True)

    fig_a, ax_a = plt.subplots(figsize=(8.0, 6.0))
    ax_a.imshow(
        regime_plane_analytic,
        origin="lower",
        aspect="equal",
        extent=[
            lambda_values[0],
            lambda_values[-1],
            gamma_real_values[0],
            gamma_real_values[-1],
        ],
        cmap=regime_cmap,
        norm=norm,
    )
    ax_a.set_xlabel(r"$\lambda$", fontsize=18)
    ax_a.set_ylabel(r"$\gamma_\mathrm{r}$", fontsize=18)
    ax_a.tick_params(labelsize=14)
    ax_a.set_title(
        r"Stuart--Landau analytical regimes in $(\lambda,\gamma_\mathrm{r})$",
        fontsize=20,
        pad=10,
    )
    ax_a.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=14,
    )
    ax_a.set_aspect("equal", adjustable="box")
    fig_a.tight_layout(rect=[0.0, 0.0, 0.8, 1.0])
    analytic_base = "stuart_landau_lambda_gamma_r_regimes_analytic"
    fig_a.savefig(f"{analytic_base}.png", transparent=True)

    gamma_real_fixed = -1.0
    lambda_bif = np.linspace(-2.0, 2.0, 801)
    r_zero = np.zeros_like(lambda_bif)
    r_lc = np.zeros_like(lambda_bif)

    mask_lc = (gamma_real_fixed < 0.0) & (lambda_bif > 0.0)
    r_lc[mask_lc] = np.sqrt(-lambda_bif[mask_lc] / gamma_real_fixed)

    fig_bif, ax_bif = plt.subplots(figsize=(8.0, 5.0))

    # ax_bif.axhline(0.0, color="black", linewidth=1.5)
    ax_bif.plot(
        lambda_bif[lambda_bif < 0.0],
        r_zero[lambda_bif < 0.0],
        color=thesis_blue,
        linestyle="-",
        linewidth=2.5,
        label="fixed point (stable)",
    )
    ax_bif.plot(
        lambda_bif[lambda_bif >= 0.0],
        r_zero[lambda_bif >= 0.0],
        color=thesis_blue,
        linestyle="--",
        linewidth=2.5,
        label="fixed point (unstable)",
    )
    ax_bif.plot(
        lambda_bif[mask_lc],
        r_lc[mask_lc],
        color=ifisc_green,
        linestyle="-",
        linewidth=2.5,
        label="limit cycle (stable)",
    )

    ax_bif.set_xlabel(r"$\lambda$", fontsize=18)
    ax_bif.set_ylabel(r"$r$", fontsize=18)
    ax_bif.tick_params(labelsize=14)
    ax_bif.set_title(
        rf"Stuart--Landau bifurcation, $\gamma_\mathrm{{r}}={gamma_real_fixed:.2f}$"
        + rf", $\omega={omega:.2f}$, $\gamma_i={gamma_imag:.2f}$",
        fontsize=20,
        pad=10,
    )
    ax_bif.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=14,
    )

    ax_bif.set_aspect("equal", adjustable="box")

    fig_bif.tight_layout(rect=[0.0, 0.0, 0.8, 1.0])

    bif_base = "stuart_landau_bifurcation_branches"
    fig_bif.savefig(f"{bif_base}.png", transparent=True)

    plt.show()


if __name__ == "__main__":
    main()

