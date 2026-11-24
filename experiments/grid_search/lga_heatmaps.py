"""Progressive heatmaps for lambda / gamma_real / alpha grid searches."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from utils.plotting_utils.style import ifisc_green, thesis_blue, thesis_red

LIVE_RC = {
    "text.usetex": False,
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": False,
}

TICK_FONTSIZE = 34
AXIS_LABEL_FONTSIZE = 36
TITLE_FONTSIZE = 38
CBAR_LABEL_FONTSIZE = 36

SUMMARY_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["newpx"],
    "text.latex.preamble": r"\usepackage{newpxtext,newpxmath}",
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 22,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "legend.fontsize": 16,
    "figure.titlesize": 22,
    "figure.dpi": 300,
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

HEATMAP_XLABEL_PAD = 65
HEATMAP_XLABEL_PAD_LAMBDA = 55

plt.rcParams.update(LIVE_RC)

PARAM_LABELS_PLAIN = {
    "lambda": r"Lambda ($\lambda$)",
    "gamma_real": r"Gamma Real ($\gamma_r$)",
    "alpha": r"Alpha ($\alpha$)",
}

TASK_LABELS = {
    "mackey_glass": "Mackey-Glass",
    "mg": "Mackey-Glass",
    "imdb": "IMDB",
    "smnist": "sMNIST",
}


def _format_val(v: float, param_name: str) -> str:
    if param_name == "omega":
        return f"{v:.4f}"
    if abs(v) < 1e-9:
        return "0"
    rounded = round(v, 1)
    if abs(v - rounded) < 1e-6:
        return f"{rounded:.1f}"
    return f"{v:.3f}"


def _find_closest_idx(val: float, vals: list[float], tol: float | None = None) -> int | None:
    if tol is None:
        unique_vals = sorted(set(vals))
        if len(unique_vals) <= 1:
            tol = 1e-6
        else:
            diffs = [
                abs(unique_vals[i + 1] - unique_vals[i])
                for i in range(len(unique_vals) - 1)
                if abs(unique_vals[i + 1] - unique_vals[i]) > 1e-9
            ]
            tol = max(1e-6, 0.51 * min(diffs)) if diffs else 1e-6

    best_idx = None
    best_diff = None
    for idx, v in enumerate(vals):
        d = abs(v - val)
        if d <= tol and (best_diff is None or d < best_diff):
            best_idx = idx
            best_diff = d
    return best_idx


THRESHOLDS = {"mg": 90.0, "imdb": 50.0, "smnist": 10.0}


def _cmap_task_key(task_name: str | None, scale: float = 1.0) -> str:
    key = (task_name or "").lower()
    if key in ("mackey_glass", "mg") or scale != 1.0:
        return "mg"
    if key == "smnist":
        return "smnist"
    return "imdb"


def _metric_from_result(result: dict, metric_key: str) -> float:
    if result.get("numerical_error") or result.get("status") not in (None, "ok"):
        if result.get("status") and result.get("status") != "ok":
            return 0.0
    if result.get("numerical_error"):
        return 0.0
    value = result.get(metric_key)
    if value is None:
        value = result.get("metric")
    if value is None:
        return 0.0
    return float(value)


def _metric_to_percent(result: dict, metric_key: str, task_name: str | None, scale: float) -> float:
    raw = _metric_from_result(result, metric_key)
    if _cmap_task_key(task_name, scale) == "mg":
        return max(0.0, raw) * 100.0
    if raw > 1.0:
        return raw
    return raw * 100.0


def make_task_cmap(task_name: str, vmax: float = 100.0):
    t = THRESHOLDS[task_name]
    cdict = {
        "red": [
            (0.0, 0.0, 0.0),
            (t / (2.0 * vmax), thesis_blue[0], thesis_blue[0]),
            (t / vmax, ifisc_green[0], ifisc_green[0]),
            (1.0, thesis_red[0], thesis_red[0]),
        ],
        "green": [
            (0.0, 0.0, 0.0),
            (t / (2.0 * vmax), thesis_blue[1], thesis_blue[1]),
            (t / vmax, ifisc_green[1], ifisc_green[1]),
            (1.0, thesis_red[1], thesis_red[1]),
        ],
        "blue": [
            (0.0, 0.0, 0.0),
            (t / (2.0 * vmax), thesis_blue[2], thesis_blue[2]),
            (t / vmax, ifisc_green[2], ifisc_green[2]),
            (1.0, thesis_red[2], thesis_red[2]),
        ],
    }
    cmap = mcolors.LinearSegmentedColormap("custom_heatmap", cdict, N=256)
    cmap.set_bad((1.0, 1.0, 1.0, 1.0))
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    return cmap, norm


def _build_colormap(vmin_norm: float, vmax_norm: float, blue_at: float, green_at: float):
    def normalize_value(val):
        return (val - vmin_norm) / (vmax_norm - vmin_norm)

    missing_pos = normalize_value(-2)
    zero_pos = normalize_value(-1)
    blue_pos = normalize_value(blue_at)
    green_pos = normalize_value(green_at)
    cdict = {
        "red": [
            (missing_pos, 1.0, 1.0),
            (zero_pos, 0.0, 0.0),
            (blue_pos, thesis_blue[0], thesis_blue[0]),
            (green_pos, ifisc_green[0], ifisc_green[0]),
            (1.0, thesis_red[0], thesis_red[0]),
        ],
        "green": [
            (missing_pos, 1.0, 1.0),
            (zero_pos, 0.0, 0.0),
            (blue_pos, thesis_blue[1], thesis_blue[1]),
            (green_pos, ifisc_green[1], ifisc_green[1]),
            (1.0, thesis_red[1], thesis_red[1]),
        ],
        "blue": [
            (missing_pos, 1.0, 1.0),
            (zero_pos, 0.0, 0.0),
            (blue_pos, thesis_blue[2], thesis_blue[2]),
            (green_pos, ifisc_green[2], ifisc_green[2]),
            (1.0, thesis_red[2], thesis_red[2]),
        ],
    }
    return mcolors.LinearSegmentedColormap("custom_heatmap", cdict, N=256)


def _summary_metric_label(task_name: str | None, scale: float, fallback: str) -> str:
    key = (task_name or "").lower()
    if key in ("mackey_glass", "mg") or scale != 1.0:
        return r"Test R² (\%)"
    if fallback:
        return fallback.replace("%", r"\%") if "\\" not in fallback else fallback
    return r"Test Accuracy (\%)"


def _param_label_tex(param_name: str) -> str:
    if param_name == "lambda":
        return r"$\lambda$"
    if param_name == "omega":
        return r"$\omega$"
    if param_name == "gamma_real":
        return r"$\gamma_{\mathrm{real}}$"
    if param_name == "gamma_imag":
        return r"$\gamma_{\mathrm{imag}}$"
    if param_name == "alpha":
        return r"$\alpha$"
    return param_name.replace("_", r"\_")


def _param_grid_edges(vals: list[float]) -> np.ndarray:
    vals_arr = np.asarray(vals, dtype=float)
    n = len(vals_arr)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([vals_arr[0] - 0.5, vals_arr[0] + 0.5])
    midpoints = (vals_arr[:-1] + vals_arr[1:]) / 2.0
    left_edge = vals_arr[0] - (midpoints[0] - vals_arr[0])
    right_edge = vals_arr[-1] + (vals_arr[-1] - midpoints[-1])
    return np.concatenate([[left_edge], midpoints, [right_edge]])


def _zero_param_index(vals: list[float]) -> int | None:
    x_array = np.asarray(vals, dtype=float)
    idx = np.where(np.isclose(x_array, 0.0, atol=1e-9))[0]
    return int(idx[0]) if len(idx) > 0 else None


def _select_tick_indices(vals: list[float], param_name: str, max_ticks: int = 16) -> list[int]:
    vals = list(vals)
    n = len(vals)
    if n == 0:
        return []
    if param_name in ("omega", "alpha"):
        if n <= max_ticks:
            return list(range(n))
        return list(np.linspace(0, n - 1, max_ticks, dtype=int))

    step = 0.5 if param_name == "lambda" else 0.1
    vmin, vmax = float(min(vals)), float(max(vals))
    start = np.ceil(vmin / step - 1e-9) * step
    nice_vals = []
    v = start
    while v <= vmax + step * 0.5:
        nice_vals.append(round(v, 10))
        v += step

    indices: list[int] = []
    for nv in nice_vals:
        idx = _find_closest_idx(nv, vals)
        if idx is not None and (not indices or indices[-1] != idx):
            indices.append(idx)

    if param_name == "lambda" and vmin <= 0.0 <= vmax:
        zero_idx = _zero_param_index(vals)
        if zero_idx is None:
            zero_idx = int(np.argmin(np.abs(np.asarray(vals, dtype=float))))
        if zero_idx not in indices:
            indices.append(zero_idx)
        indices.sort()

    if len(indices) > max_ticks:
        pick = np.linspace(0, len(indices) - 1, max_ticks, dtype=int)
        indices = [indices[i] for i in pick]

    if not indices:
        if n <= max_ticks:
            return list(range(n))
        return list(np.linspace(0, n - 1, max_ticks, dtype=int))
    return indices


def _draw_zero_parameter_lines(ax, x_vals, y_vals, color="white", linewidth=2.5):
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    if len(x_vals) > 0 and x_vals.min() <= 0.0 <= x_vals.max():
        ax.axvline(0.0, color=color, linewidth=linewidth, alpha=0.95)
    if len(y_vals) > 0 and y_vals.min() <= 0.0 <= y_vals.max():
        ax.axhline(0.0, color=color, linewidth=linewidth, alpha=0.95)


def discrete_hopf_lambda(omega: float, h: float = 1.0) -> float:
    """Hopf boundary of the Euler-discretised node: |1 + h(lambda + i omega)| = 1."""
    return (-1.0 + np.sqrt(max(0.0, 1.0 - (h * omega) ** 2))) / h


def _draw_discrete_hopf_line(ax, lambda_vals, omega, linewidth=2.5):
    lam_c = discrete_hopf_lambda(omega)
    if min(lambda_vals) <= lam_c <= max(lambda_vals):
        ax.axvline(lam_c, color=thesis_red, linestyle=":", linewidth=linewidth)


def _heatmap_xlabel_pad(param1_name: str) -> float:
    if param1_name == "lambda":
        return HEATMAP_XLABEL_PAD_LAMBDA
    return HEATMAP_XLABEL_PAD


def _set_summary_ticks(ax, param1_vals, param2_vals, param1_name, param2_name):
    labelpad_shift = -40
    x_tick_offset_y = -12

    x_indices = _select_tick_indices(param1_vals, param1_name)
    x_tick_positions = [param1_vals[i] for i in x_indices]
    xtick_labels = [_format_val(param1_vals[i], param1_name) for i in x_indices]

    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(["" for _ in xtick_labels])

    for tick_val, label in zip(x_tick_positions, xtick_labels):
        x_shift = 0 if abs(tick_val) < 1e-9 else labelpad_shift
        ax.annotate(
            label,
            (tick_val, 0),
            xytext=(x_shift, x_tick_offset_y),
            textcoords="offset points",
            rotation=60,
            ha="center" if abs(tick_val) < 1e-9 else "left",
            va="top",
            fontsize=TICK_FONTSIZE,
            annotation_clip=False,
            xycoords=("data", "axes fraction"),
        )

    y_indices = _select_tick_indices(param2_vals, param2_name)
    y_tick_positions = [param2_vals[i] for i in y_indices]
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(
        [_format_val(param2_vals[i], param2_name) for i in y_indices],
        fontsize=TICK_FONTSIZE,
    )


def _save_summary_heatmap(
    heatmap_avg: np.ndarray,
    param1_name: str,
    param2_name: str,
    param1_vals: list[float],
    param2_vals: list[float],
    output_path: Path,
    *,
    task_name: str | None,
    chance_level: float,
    scale: float,
    metric_label: str,
    title_suffix: str = "",
    omega: float | None = None,
) -> None:
    missing_mask = np.isnan(heatmap_avg)
    valid_data = heatmap_avg[~missing_mask]
    vmax = max(100.0, float(np.max(valid_data))) if len(valid_data) else 100.0
    data_for_plot = np.ma.array(heatmap_avg, mask=missing_mask)
    custom_cmap, norm = make_task_cmap(_cmap_task_key(task_name, scale), vmax)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.pcolormesh(
        _param_grid_edges(param1_vals),
        _param_grid_edges(param2_vals),
        data_for_plot,
        cmap=custom_cmap,
        norm=norm,
        shading="flat",
    )
    ax.set_aspect("auto")
    ax.grid(False)

    _set_summary_ticks(ax, param1_vals, param2_vals, param1_name, param2_name)
    _draw_zero_parameter_lines(ax, param1_vals, param2_vals)
    if omega is not None and param1_name == "lambda":
        _draw_discrete_hopf_line(ax, param1_vals, omega)

    ax.set_xlabel(
        _param_label_tex(param1_name),
        fontsize=AXIS_LABEL_FONTSIZE,
        labelpad=_heatmap_xlabel_pad(param1_name),
    )
    ax.set_ylabel(_param_label_tex(param2_name), fontsize=AXIS_LABEL_FONTSIZE)

    task_label = TASK_LABELS.get((task_name or "").lower(), task_name or "")
    if title_suffix:
        ax.set_title(f"{task_label}: {title_suffix}" if task_label else title_suffix, fontsize=TITLE_FONTSIZE)
    elif task_label:
        ax.set_title(task_label, fontsize=TITLE_FONTSIZE)

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    cbar.set_label(_summary_metric_label(task_name, scale, metric_label), fontsize=CBAR_LABEL_FONTSIZE)

    key = _cmap_task_key(task_name, scale)
    if key == "imdb":
        ticks = cbar.get_ticks()
        if len(ticks) > 0:
            span = vmax - 50.0
            labels = [50.0 for _ in ticks] if span <= 0 else [50.0 + (t / vmax) * span for t in ticks]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{lbl:.0f}" for lbl in labels])

    plt.tight_layout()
    fig.savefig(output_path, transparent=True)
    plt.close(fig)


def generate_lga_heatmaps(
    results: list[dict],
    output_dir: Path | str,
    config: dict,
    metric_key: str = "best_test_acc",
    metric_label: str = r"Test Accuracy (\%)",
    chance_level: float = 50.0,
    scale: float = 1.0,
    timestamp: str = "live",
    *,
    clip_nonpositive: bool = False,
    annotate: bool = True,
    task_name: str | None = None,
    omega: float | None = None,
) -> list[Path]:
    """Write progressive heatmaps; missing cells stay blank/white.

    clip_nonpositive: treat metric <= 0 (e.g. negative R^2) as zero/grey.
    annotate: live overlays; False uses thesis multi-task visual style.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def values_from_range(key: str, values_key: str) -> list[float]:
        if config.get(values_key):
            return [float(v) for v in config[values_key]]
        rng = config.get(key)
        if rng and len(rng) >= 3:
            return np.linspace(float(rng[0]), float(rng[1]), int(rng[2])).tolist()
        short = values_key.replace("_values", "")
        return sorted({float(r[short]) for r in results if short in r})

    lambda_vals = values_from_range("lambda_range", "lambda_values")
    gamma_real_vals = values_from_range("gamma_real_range", "gamma_real_values")
    alpha_vals = values_from_range("alpha_range", "alpha_values")

    param_pairs = [
        ("lambda", "gamma_real", lambda_vals, gamma_real_vals, "alpha", alpha_vals),
        ("lambda", "alpha", lambda_vals, alpha_vals, "gamma_real", gamma_real_vals),
        ("gamma_real", "alpha", gamma_real_vals, alpha_vals, "lambda", lambda_vals),
    ]

    written: list[Path] = []
    ok_results = [r for r in results if r.get("status", "ok") == "ok"]
    style_ctx = plt.rc_context(SUMMARY_RC) if not annotate else plt.rc_context(LIVE_RC)

    with style_ctx:
        for param1_name, param2_name, param1_vals, param2_vals, slice_name, slice_vals in param_pairs:
            if len(param1_vals) <= 1 or len(param2_vals) <= 1:
                continue

            def build_and_save(current_results: list[dict], suffix: str = "", title_suffix: str = ""):
                heatmap_data = np.full((len(param2_vals), len(param1_vals)), np.nan)
                count_data = np.zeros((len(param2_vals), len(param1_vals)), dtype=int)
                max_data = np.full((len(param2_vals), len(param1_vals)), np.nan)

                for r in current_results:
                    if param1_name not in r or param2_name not in r:
                        continue
                    idx1 = _find_closest_idx(float(r[param1_name]), param1_vals)
                    idx2 = _find_closest_idx(float(r[param2_name]), param2_vals)
                    if idx1 is None or idx2 is None:
                        continue
                    acc = _metric_to_percent(r, metric_key, task_name, scale)
                    if clip_nonpositive and not np.isnan(acc):
                        acc = max(0.0, acc)
                    if np.isnan(heatmap_data[idx2, idx1]):
                        heatmap_data[idx2, idx1] = acc
                        max_data[idx2, idx1] = acc
                        count_data[idx2, idx1] = 1
                    else:
                        heatmap_data[idx2, idx1] += acc
                        max_data[idx2, idx1] = max(max_data[idx2, idx1], acc)
                        count_data[idx2, idx1] += 1

                heatmap_avg = np.divide(
                    heatmap_data,
                    count_data,
                    out=np.full_like(heatmap_data, np.nan),
                    where=count_data != 0,
                )
                if clip_nonpositive:
                    neg = (~np.isnan(heatmap_avg)) & (heatmap_avg < 0)
                    heatmap_avg = heatmap_avg.copy()
                    heatmap_avg[neg] = 0.0
                    max_data = max_data.copy()
                    max_data[neg] = np.maximum(max_data[neg], 0.0)

                base_name = f"heatmap_{param1_name}_vs_{param2_name}"
                if suffix:
                    base_name += f"_{suffix}"
                output_path = output_dir / f"{base_name}_{timestamp}.png"

                if not annotate:
                    tex_title = title_suffix
                    if title_suffix and "=" in title_suffix:
                        # replace plain slice labels with tex when possible
                        for pname, plabel in (
                            ("alpha", r"$\alpha$"),
                            ("lambda", r"$\lambda$"),
                            ("gamma_real", r"$\gamma_{\mathrm{real}}$"),
                        ):
                            plain = PARAM_LABELS_PLAIN[pname]
                            if title_suffix.startswith(plain):
                                tex_title = title_suffix.replace(plain, plabel, 1)
                                break
                    _save_summary_heatmap(
                        heatmap_avg,
                        param1_name,
                        param2_name,
                        param1_vals,
                        param2_vals,
                        output_path,
                        task_name=task_name,
                        chance_level=chance_level,
                        scale=scale,
                        metric_label=metric_label,
                        title_suffix=tex_title,
                        omega=omega,
                    )
                    written.append(output_path)
                    return

                missing_mask = np.isnan(heatmap_avg)
                valid_data = heatmap_avg[~missing_mask]
                vmax = max(100.0, float(np.max(valid_data))) if len(valid_data) else 100.0
                cmap_key = _cmap_task_key(task_name, scale)

                if cmap_key == "mg":
                    heatmap_norm = np.ma.array(heatmap_avg, mask=missing_mask)
                    custom_cmap, norm = make_task_cmap(cmap_key, vmax)
                else:
                    zero_mask = (heatmap_avg == 0) & (~missing_mask)
                    below_mask = (heatmap_avg > 0) & (heatmap_avg < chance_level) & (~missing_mask)
                    heatmap_norm = heatmap_avg.copy()
                    heatmap_norm[below_mask] = chance_level - 0.1
                    heatmap_norm[zero_mask] = -1
                    heatmap_norm[missing_mask] = -2
                    custom_cmap = _build_colormap(-2, vmax, chance_level - 0.1, chance_level)
                    norm = mcolors.Normalize(vmin=-2, vmax=vmax)

                fig, ax = plt.subplots(figsize=(12, 10))
                ax.imshow(
                    heatmap_norm,
                    aspect="auto",
                    origin="lower",
                    cmap=custom_cmap,
                    norm=norm,
                    interpolation="nearest",
                )

                num_ticks_x = min(16, len(param1_vals))
                num_ticks_y = min(16, len(param2_vals))
                if len(param1_vals) > 16:
                    x_indices = np.linspace(0, len(param1_vals) - 1, num_ticks_x, dtype=int)
                    ax.set_xticks(x_indices)
                    ax.set_xticklabels(
                        [_format_val(param1_vals[i], param1_name) for i in x_indices],
                        rotation=45,
                        ha="right",
                        fontsize=28,
                    )
                else:
                    ax.set_xticks(np.arange(len(param1_vals)))
                    ax.set_xticklabels(
                        [_format_val(v, param1_name) for v in param1_vals],
                        rotation=45,
                        ha="right",
                        fontsize=28,
                    )

                if len(param2_vals) > 16:
                    y_indices = np.linspace(0, len(param2_vals) - 1, num_ticks_y, dtype=int)
                    ax.set_yticks(y_indices)
                    ax.set_yticklabels(
                        [_format_val(param2_vals[i], param2_name) for i in y_indices],
                        fontsize=28,
                    )
                else:
                    ax.set_yticks(np.arange(len(param2_vals)))
                    ax.set_yticklabels([_format_val(v, param2_name) for v in param2_vals], fontsize=28)

                ax.set_xlabel(PARAM_LABELS_PLAIN.get(param1_name, param1_name), fontsize=30)
                ax.set_ylabel(PARAM_LABELS_PLAIN.get(param2_name, param2_name), fontsize=30)

                for j in range(len(param2_vals)):
                    for i in range(len(param1_vals)):
                        if missing_mask[j, i]:
                            continue
                        avg_val = heatmap_avg[j, i]
                        max_val = max_data[j, i]
                        if np.isnan(avg_val) or np.isnan(max_val):
                            continue
                        ax.text(i, j, f"{avg_val:.1f}\n{max_val:.1f}", ha="center", va="center", color="white", fontsize=6)

                cbar = plt.colorbar(ax.images[0], ax=ax)
                cbar.set_label(metric_label, fontsize=30)
                cbar.ax.tick_params(labelsize=28)

                if current_results:
                    current_ok = [
                        r for r in current_results if r.get("status", "ok") == "ok" and not r.get("numerical_error")
                    ]
                    if current_ok:
                        best = max(current_ok, key=lambda x: _metric_from_result(x, metric_key))
                        best_idx1 = _find_closest_idx(float(best[param1_name]), param1_vals)
                        best_idx2 = _find_closest_idx(float(best[param2_name]), param2_vals)
                        if best_idx1 is not None and best_idx2 is not None:
                            ax.add_patch(
                                mpatches.Rectangle(
                                    (best_idx1 - 0.5, best_idx2 - 0.5),
                                    1.0,
                                    1.0,
                                    linewidth=2,
                                    edgecolor="white",
                                    facecolor="none",
                                    zorder=10,
                                )
                            )

                n_done = int(np.sum(~missing_mask))
                n_total = len(param1_vals) * len(param2_vals)
                ax.text(
                    0.01,
                    1.02,
                    f"{n_done}/{n_total} cells" + (f" | {title_suffix}" if title_suffix else ""),
                    transform=ax.transAxes,
                    fontsize=14,
                    va="bottom",
                )

                plt.tight_layout()
                fig.savefig(output_path)
                plt.close(fig)
                written.append(output_path)

            build_and_save(ok_results)

            if slice_vals and len(slice_vals) > 1:
                for slice_val in slice_vals:
                    filtered = [
                        r
                        for r in ok_results
                        if slice_name in r and abs(float(r[slice_name]) - float(slice_val)) < 1e-6
                    ]
                    if not filtered:
                        continue
                    label_val = _format_val(slice_val, slice_name)
                    build_and_save(
                        filtered,
                        suffix=f"{slice_name}_{label_val.replace('.', 'p')}",
                        title_suffix=f"{PARAM_LABELS_PLAIN.get(slice_name, slice_name)} = {label_val}",
                    )

    return written
