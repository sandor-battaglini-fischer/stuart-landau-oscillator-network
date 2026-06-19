import os
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatterMathtext, LogLocator
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.plotting_utils.style import apply_style, thesis_red, thesis_blue, ifisc_green, mycmap
apply_style()



thesis_blue = (0, 0.38, 0.68)
ifisc_green = (0.73, 0.83, 0.01)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORK_ANALYSIS_DIR = os.path.join(BASE_DIR, "Network analysis")


def infer_task_label(run_name: str) -> str:
    name = run_name.lower()
    if "imdb" in name:
        return "IMDB"
    if "smnist" in name:
        return "sMNIST"
    if "mg" in name or "mackey" in name:
        return "Mackey-Glass"
    return run_name


def load_manifold_dimensions(run_dir: str):
    path = os.path.join(run_dir, "manifold_dimension_results.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    corr_dim = data.get("correlation_dim", None)
    eff_dim = data.get("effective_dim_pca", None)
    return corr_dim, eff_dim


def parse_network_statistics_file(path: str):
    if not os.path.exists(path):
        return None
    epochs = []
    edges = []
    densities = []
    epochs = []
    mean_weights = []
    max_weights = []
    avg_clusts = []
    mean_degrees = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Network Statistics") or line.startswith("PER-EPOCH") or line.startswith("-"):
                continue
            if line.startswith("SUMMARY"):
                break
            parts = line.split()
            if len(parts) < 7 or not parts[0][0].isdigit():
                continue
            try:
                epoch_idx = int(parts[0])
                edge_val = float(parts[2])
                density_val = float(parts[3])
                mean_w = float(parts[4])
                max_w = float(parts[5])
                avg_clust = float(parts[9])
                mean_deg = float(parts[11])
            except ValueError:
                continue
            epochs.append(epoch_idx)
            edges.append(edge_val)
            densities.append(density_val)
            mean_weights.append(mean_w)
            max_weights.append(max_w)
            avg_clusts.append(avg_clust)
            mean_degrees.append(mean_deg)
    if not mean_weights:
        return None
    return {
        "epochs": np.array(epochs),
        "edges": np.array(edges),
        "density": np.array(densities),
        "mean_weight": np.array(mean_weights),
        "max_weight": np.array(max_weights),
        "avg_clust": np.array(avg_clusts),
        "mean_degree": np.array(mean_degrees),
    }


def compute_layer_series(run_dir: str):
    layer_dirs = {
        "h2h": os.path.join(run_dir, "recurrent_networks"),
        "i2h": os.path.join(run_dir, "i2h_networks"),
        "h2o": os.path.join(run_dir, "h2o_networks"),
    }
    series = {}
    for layer, ldir in layer_dirs.items():
        if not os.path.isdir(ldir):
            continue
        stats_files = [f for f in os.listdir(ldir) if f.startswith("network_statistics_") and f.endswith(".txt")]
        if not stats_files:
            continue
        stats_path = os.path.join(ldir, stats_files[0])
        parsed = parse_network_statistics_file(stats_path)
        if parsed is None:
            continue
        series[layer] = parsed
    return series


def collect_runs():
    runs = []
    if not os.path.isdir(NETWORK_ANALYSIS_DIR):
        return runs
    for name in sorted(os.listdir(NETWORK_ANALYSIS_DIR)):
        run_path = os.path.join(NETWORK_ANALYSIS_DIR, name)
        if not os.path.isdir(run_path):
            continue
        label = infer_task_label(name)
        mdims = load_manifold_dimensions(run_path)
        if mdims is None:
            continue
        corr_dim, eff_dim = mdims
        layer_series = compute_layer_series(run_path)
        runs.append(
            {
                "name": name,
                "label": label,
                "corr_dim": corr_dim,
                "eff_dim": eff_dim,
                "layer_series": layer_series,
            }
        )
    return runs


def compute_weight_usage_for_run(run_name: str):
    run_dir = os.path.join(NETWORK_ANALYSIS_DIR, run_name)
    params_path = os.path.join(run_dir, "parameters.json")
    if not os.path.exists(params_path):
        return {}

    with open(params_path, "r") as f:
        data = json.load(f)

    layer_map = {
        "i2h_weight": "i2h",
        "h2h_weight": "h2h",
        "h2o_weight": "h2o",
    }
    usage = {layer: {"epochs": [], "frac_active": [], "frac_pos": [], "frac_neg": [], "frac_strong_pos": [], "frac_strong_neg": []} for layer in ["i2h", "h2h", "h2o"]}

    for entry in data:
        epoch = entry.get("epoch")
        params = entry.get("params", {})
        for key, layer_name in layer_map.items():
            w = params.get(key)
            if w is None:
                continue
            w_arr = np.asarray(w, dtype=float)
            if w_arr.size == 0:
                continue
            flat = w_arr.ravel()
            abs_flat = np.abs(flat)
            max_abs = np.max(abs_flat)
            if max_abs == 0:
                continue
            thresh = 0.2 * max_abs
            frac_active = float(np.mean(abs_flat > thresh))
            frac_pos = float(np.mean(flat > thresh))
            frac_neg = float(np.mean(flat < -thresh))
            frac_strong_pos = frac_pos
            frac_strong_neg = frac_neg

            u = usage[layer_name]
            u["epochs"].append(epoch)
            u["frac_active"].append(frac_active)
            u["frac_pos"].append(frac_pos)
            u["frac_neg"].append(frac_neg)
            u["frac_strong_pos"].append(frac_strong_pos)
            u["frac_strong_neg"].append(frac_strong_neg)

    for layer_name, u in usage.items():
        if not u["epochs"]:
            continue
        order = np.argsort(np.asarray(u["epochs"]))
        for key in ["epochs", "frac_active", "frac_pos", "frac_neg", "frac_strong_pos", "frac_strong_neg"]:
            arr = np.asarray(u[key], dtype=float)
            u[key] = arr[order]

    return usage


def compute_weight_moments_for_run(run_name: str):
    run_dir = os.path.join(NETWORK_ANALYSIS_DIR, run_name)
    params_path = os.path.join(run_dir, "parameters.json")
    if not os.path.exists(params_path):
        return {}

    with open(params_path, "r") as f:
        data = json.load(f)

    layer_map = {
        "i2h_weight": "i2h",
        "h2h_weight": "h2h",
        "h2o_weight": "h2o",
    }
    moments = {
        layer: {"epochs": [], "mean": [], "std": [], "min": [], "max": []}
        for layer in ["i2h", "h2h", "h2o"]
    }

    for entry in data:
        epoch = entry.get("epoch")
        params = entry.get("params", {})
        for key, layer_name in layer_map.items():
            w = params.get(key)
            if w is None:
                continue
            w_arr = np.asarray(w, dtype=float)
            if w_arr.size == 0:
                continue
            flat = w_arr.ravel()
            m = float(np.mean(flat))
            s = float(np.std(flat))
            mn = float(np.min(flat))
            mx = float(np.max(flat))

            u = moments[layer_name]
            u["epochs"].append(epoch)
            u["mean"].append(m)
            u["std"].append(s)
            u["min"].append(mn)
            u["max"].append(mx)

    for layer_name, u in moments.items():
        if not u["epochs"]:
            continue
        order = np.argsort(np.asarray(u["epochs"]))
        for key in ["epochs", "mean", "std", "min", "max"]:
            arr = np.asarray(u[key], dtype=float)
            u[key] = arr[order]

    return moments


def get_layer_shapes_for_run(run_name: str):
    run_dir = os.path.join(NETWORK_ANALYSIS_DIR, run_name)
    params_path = os.path.join(run_dir, "parameters.json")
    if not os.path.exists(params_path):
        return {}

    with open(params_path, "r") as f:
        data = json.load(f)
    if not data:
        return {}

    first_params = data[0].get("params", {})
    shapes = {}
    for key in ["i2h_weight", "h2h_weight", "h2o_weight"]:
        w = first_params.get(key)
        if w is None:
            continue
        arr = np.asarray(w, dtype=float)
        shapes[key] = arr.shape
    return shapes


def sample_initial_weight_variance_for_run(run_name: str, layer_name: str, seed: int = 0):
    layer_key = {
        "i2h": "i2h_weight",
        "h2h": "h2h_weight",
        "h2o": "h2o_weight",
    }.get(layer_name)
    if layer_key is None:
        return None

    shapes = get_layer_shapes_for_run(run_name)
    shape = shapes.get(layer_key)
    if shape is None:
        return None

    rng = np.random.default_rng(seed)
    w_init = rng.standard_normal(shape)
    return float(np.var(w_init))


def parse_imdb_log(log_path: Path):
    if not log_path.exists():
        return None

    epochs = []
    test_accuracies = []

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("epoch"):
                continue

            parts = line.split(":", 1)
            if len(parts) != 2:
                continue

            epoch_parts = parts[0].split()
            if len(epoch_parts) < 2:
                continue

            try:
                epoch = int(epoch_parts[1])
            except (ValueError, IndexError):
                continue

            rest = parts[1].strip()
            rest_parts = rest.split(",")

            for part in rest_parts:
                if "test:" in part:
                    test_str = part.strip()
                    test_value_str = test_str.split(":", 1)[1].strip().split()[0]
                    try:
                        test = float(test_value_str)
                    except ValueError:
                        continue
                    epochs.append(epoch)
                    test_accuracies.append(test)
                    break

    if len(test_accuracies) == 0 or len(epochs) != len(test_accuracies):
        return None

    sorted_data = sorted(zip(epochs, test_accuracies))
    _, sorted_test = zip(*sorted_data)
    return np.asarray(sorted_test, dtype=float)


def parse_smnist_log(log_path: Path):
    return parse_imdb_log(log_path)


def load_mg_results(results_path: Path, num_hidden_list):
    if not results_path.exists():
        return {}

    with open(results_path, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    by_n = {}

    for r in results:
        params = r.get("parameters", {})
        n = params.get("num_hidden")
        if n is None or n not in num_hidden_list:
            continue

        test_r2 = r.get("test_r2_scores", [])
        if not test_r2:
            continue

        arr = np.asarray(test_r2, dtype=float)
        if n not in by_n:
            by_n[n] = arr
        else:
            existing = by_n[n]
            if len(arr) > len(existing) or np.max(arr) > np.max(existing):
                by_n[n] = arr

    return by_n


def find_min_nodes_within_threshold(best_by_n, threshold_factor=0.98):
    if not best_by_n:
        return None
    global_best = max(best_by_n.values())
    threshold = global_best * threshold_factor
    candidates = [n for n, v in best_by_n.items() if v >= threshold]
    if not candidates:
        return None
    return int(min(candidates))


def compute_effective_system_dimensions():
    base_path = Path(BASE_DIR)
    runs_old_dir = base_path / "runs-old"

    imdb_dirs = {
        1: "results/imdb/20251207_231522",
        2: "results/imdb/20251207_231503",
        4: "results/imdb/20251208_104915",
        9: "results/imdb/20251207_221754",
        50: "results/imdb/20251207_221628",
        128: "results/imdb/20251210_172243",
    }

    smnist_dirs = {
        1: "results/smnist/20251218_114219",
        2: "results/smnist/20251218_114204",
        4: "results/smnist/20251218_114148",
        9: "results/smnist/20251218_114028",
        16: "results/smnist/20260311_123312",
        25: "results/smnist/20260311_123333",
        50: "results/smnist/20251218_115016",
        128: "results/smnist/20251218_120601",
    }

    mg_results_file = base_path / "results/mackey_glass/comparison_20260203_184809" / "full_results.json"
    mg_node_counts = [1, 2, 4, 9, 50, 128]

    imdb_best_by_n = {}
    for n, dirname in imdb_dirs.items():
        log_path = runs_old_dir / dirname / "log.txt"
        test_acc = parse_imdb_log(log_path)
        if test_acc is not None:
            imdb_best_by_n[n] = float(np.max(test_acc))

    smnist_best_by_n = {}
    for n, dirname in smnist_dirs.items():
        log_path = runs_old_dir / dirname / "log.txt"
        test_acc = parse_smnist_log(log_path)
        if test_acc is not None:
            smnist_best_by_n[n] = float(np.max(test_acc))

    mg_data = load_mg_results(mg_results_file, mg_node_counts)
    mg_best_by_n = {}
    for n, values in mg_data.items():
        mg_best_by_n[n] = float(np.max(values))

    eff_dims = {}

    imdb_n = find_min_nodes_within_threshold(imdb_best_by_n, threshold_factor=0.90)
    if imdb_n is not None:
        eff_dims["IMDB"] = 2 * imdb_n

    smnist_n = find_min_nodes_within_threshold(smnist_best_by_n, threshold_factor=0.90)
    if smnist_n is not None:
        eff_dims["sMNIST"] = 2 * smnist_n

    mg_n = find_min_nodes_within_threshold(mg_best_by_n, threshold_factor=0.90)
    if mg_n is not None:
        eff_dims["Mackey-Glass"] = 2 * mg_n

    return eff_dims


def plot_dimension_comparison(runs):
    labels = [r["label"] for r in runs]
    x = np.arange(len(labels))
    corr_dims = [r["corr_dim"] for r in runs]
    eff_dims = [r["eff_dim"] for r in runs]
    system_eff_dims_map = compute_effective_system_dimensions()
    system_eff_dims = [system_eff_dims_map.get(label, np.nan) for label in labels]

    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.bar(x - width, corr_dims, width, label="Correlation dim", color=thesis_blue)
    ax.bar(x, eff_dims, width, label="Effective PCA dim", color=thesis_red)
    ax.bar(x + width, system_eff_dims, width, label="Effective system dim (2N)", color=ifisc_green)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Dimension")
    ax.set_title("Manifold dimensions across tasks")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(NETWORK_ANALYSIS_DIR, "task_dimension_comparison.png")
    fig.savefig(out_path, dpi=600, transparent=True)
    plt.close(fig)


def plot_topological_restructuring(runs):
    layers = ["i2h", "h2h", "h2o"]
    layer_titles = {"i2h": "Input weights", "h2h": "Hidden weights", "h2o": "Output weights"}
    task_colors = {
        "IMDB": thesis_red,
        "sMNIST": thesis_blue,
        "Mackey-Glass": ifisc_green,
    }

    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 8), sharey=False)
    if len(layers) == 1:
        axes = [axes]

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        for run in runs:
            series = run["layer_series"].get(layer)
            if not series:
                continue
            epochs = series["epochs"]
            mean_w = series["mean_weight"]
            color = task_colors.get(run["label"], None)
            ax.plot(epochs, mean_w, label=run["label"], color=color, linewidth=1)

        ax.set_xlabel("Epoch")
        if idx == 0:
            ax.set_ylabel("Mean weight")
        ax.set_title(layer_titles.get(layer, layer))
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=len(handles))

    fig.tight_layout(rect=[0, 0, 1, 0.85])
    out_path = os.path.join(NETWORK_ANALYSIS_DIR, "task_topological_restructuring.png")
    fig.savefig(out_path, dpi=600, transparent=True)
    plt.close(fig)

    metrics = [
        ("max_weight", "Weight (mean ± std, max)"),
        ("edges", "Number of edges"),
        ("density", "Density"),
        ("avg_clust", "Average clustering"),
        ("mean_degree", "Mean degree"),
    ]

    for metric_key, ylabel in metrics:
        fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 8), sharey=False)
        if len(layers) == 1:
            axes = [axes]

        moments_cache = {}

        for idx, layer in enumerate(layers):
            ax = axes[idx]
            for run in runs:
                series = run["layer_series"].get(layer)
                if not series or metric_key not in series:
                    continue
                color = task_colors.get(run["label"], None)

                if metric_key == "max_weight":
                    if run["name"] not in moments_cache:
                        moments_cache[run["name"]] = compute_weight_moments_for_run(run["name"])
                    moments = moments_cache[run["name"]].get(layer)
                    if not moments:
                        continue
                    epochs = moments["epochs"]
                    mean_vals = moments["mean"]
                    std_vals = moments["std"]
                    max_vals = np.maximum(np.abs(moments["max"]), np.abs(moments["min"]))

                    ax.fill_between(
                        epochs,
                        mean_vals - std_vals,
                        mean_vals + std_vals,
                        color=color,
                        alpha=0.20,
                    )
                    ax.plot(epochs, max_vals, color=color, linewidth=1.5, alpha=0.6)
                    ax.plot(epochs, -max_vals, color=color, linewidth=1.5, alpha=0.6)
                    ax.plot(epochs, mean_vals, label=run["label"], color=color, linewidth=2.5)
                else:
                    epochs = series["epochs"]
                    values = series[metric_key]
                    ax.plot(epochs, values, label=run["label"], color=color, linewidth=2)

            ax.set_xlabel("Epoch")
            if idx == 0:
                ax.set_ylabel(ylabel)
            ax.set_title(layer_titles.get(layer, layer))
            ax.grid(True, alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=len(handles))

        fig.tight_layout(rect=[0, 0, 1, 0.85])
        out_path = os.path.join(
            NETWORK_ANALYSIS_DIR,
            f"task_topological_restructuring_{metric_key}.png",
        )
        fig.savefig(out_path, dpi=600, transparent=True)
        plt.close(fig)


def plot_weight_usage(runs):
    layers = ["i2h", "h2h", "h2o"]
    layer_titles = {"i2h": "Input weights", "h2h": "Hidden weights", "h2o": "Output weights"}
    task_colors = {
        "IMDB": thesis_red,
        "sMNIST": thesis_blue,
        "Mackey-Glass": ifisc_green,
    }

    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 8), sharey=False)
    if len(layers) == 1:
        axes = [axes]

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        for run in runs:
            usage = compute_weight_usage_for_run(run["name"])
            if layer not in usage or not usage[layer]["epochs"].size:
                continue
            epochs = usage[layer]["epochs"]
            frac_active = usage[layer]["frac_active"]
            color = task_colors.get(run["label"], None)
            ax.plot(epochs, frac_active, label=run["label"], color=color, linewidth=2)

        ax.set_xlabel("Epoch")
        if idx == 0:
            ax.set_ylabel("Fraction of active weights")
        ax.set_title(layer_titles.get(layer, layer))
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=len(handles))

    fig.tight_layout(rect=[0, 0, 1, 0.85])
    out_path = os.path.join(NETWORK_ANALYSIS_DIR, "task_weight_usage_active.png")
    fig.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 8), sharey=False)
    if len(layers) == 1:
        axes = [axes]

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        for run in runs:
            usage = compute_weight_usage_for_run(run["name"])
            if layer not in usage or not usage[layer]["epochs"].size:
                continue
            epochs = usage[layer]["epochs"]
            frac_strong_pos = usage[layer]["frac_strong_pos"]
            frac_strong_neg = usage[layer]["frac_strong_neg"]
            color = task_colors.get(run["label"], None)
            ax.plot(epochs, frac_strong_pos, label=f"{run['label']} strong +", color=color, linewidth=2)
            ax.plot(epochs, frac_strong_neg, label=f"{run['label']} strong -", color=color, linewidth=1.5, linestyle="--")

        ax.set_xlabel("Epoch")
        if idx == 0:
            ax.set_ylabel("Fraction of strongly signed weights")
        ax.set_title(layer_titles.get(layer, layer))
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3)

    fig.tight_layout(rect=[0, 0, 1, 0.83])
    out_path = os.path.join(NETWORK_ANALYSIS_DIR, "task_weight_usage_strong_sign.png")
    fig.savefig(out_path, dpi=600, transparent=True)
    plt.close(fig)


def plot_weight_usage_bar(runs):
    layers = ["i2h", "h2h", "h2o"]
    layer_titles = {"i2h": "Input weights", "h2h": "Hidden weights", "h2o": "Output weights"}
    task_colors = {
        "IMDB": thesis_red,
        "sMNIST": thesis_blue,
        "Mackey-Glass": ifisc_green,
    }

    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 6), sharey=True)
    if len(layers) == 1:
        axes = [axes]

    task_order = ["IMDB", "sMNIST", "Mackey-Glass"]

    for idx, layer in enumerate(layers):
        ax = axes[idx]
        task_labels = []
        before_vals = []
        after_vals = []
        colors_before = []
        colors_after = []

        for label in task_order:
            run_for_label = next((r for r in runs if r["label"] == label), None)
            if run_for_label is None:
                continue

            moments = compute_weight_moments_for_run(run_for_label["name"])
            if layer not in moments or not moments[layer]["epochs"].size:
                continue

            std_vals = moments[layer]["std"]
            var_vals = std_vals ** 2
            # assume epochs are sorted so index 0 is the earliest (epoch -1 if present)
            before_var = float(var_vals[0])
            after_var = float(var_vals[-1])

            task_labels.append(label)
            before_vals.append(before_var)
            after_vals.append(after_var)
            base_color = task_colors.get(label, "0.5")
            colors_before.append(base_color)
            colors_after.append(base_color)

        if not task_labels:
            continue

        all_vals = np.array(before_vals + after_vals, dtype=float)
        pos_vals = all_vals[all_vals > 0]
        floor = float(pos_vals.min() * 0.1) if pos_vals.size else 1e-12
        before_vals = [max(float(v), floor) for v in before_vals]
        after_vals = [max(float(v), floor) for v in after_vals]

        x = np.arange(len(task_labels))
        width = 0.35

        ax.bar(
            x - width / 2,
            before_vals,
            width,
            color=colors_before,
            alpha=0.4
        )
        ax.bar(
            x + width / 2,
            after_vals,
            width,
            color=colors_after,
            alpha=0.9
        )

        ax.set_xticks(x)
        ax.set_xticklabels(task_labels, rotation=0)
        ax.set_xlabel("Task")
        if idx == 0:
            ax.set_ylabel("Weight variance")
        ax.set_title(layer_titles.get(layer, layer))
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor)
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.grid(True, axis="y", alpha=0.3)

    handles = [
        Patch(facecolor="0.5", alpha=0.4, label="Before Training"),
        Patch(facecolor="0.5", alpha=0.9, label="After Training"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.85])
    out_path = os.path.join(NETWORK_ANALYSIS_DIR, "task_weight_variance_before_after.png")
    fig.savefig(out_path, dpi=600, transparent=True)
    plt.close(fig)


def main():
    runs = collect_runs()
    print("The used runs are:", [r["name"] for r in runs])
    if not runs:
        raise RuntimeError(f"No suitable runs found in {NETWORK_ANALYSIS_DIR}")
    plot_dimension_comparison(runs)
    plot_topological_restructuring(runs)
    plot_weight_usage(runs)
    plot_weight_usage_bar(runs)


if __name__ == "__main__":
    main()

