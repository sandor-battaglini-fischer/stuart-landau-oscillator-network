import numpy as np
import matplotlib.pyplot as plt

from .style import mycmap, thesis_blue, thesis_red, ifisc_green


def plot_parameter_evolution(parameters_history, output_dir, dynamics_type):
    if not parameters_history:
        return

    if "stats" not in parameters_history[0]:
        return

    epochs = [p["epoch"] for p in parameters_history]

    available_keys = set(parameters_history[0]["stats"].keys())
    if not any("weight" in k or "bias" in k for k in available_keys):
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    layers = ["i2h", "h2h", "h2o"]
    stats_to_plot = ["mean", "std", "abs_mean"]

    for layer_idx, layer in enumerate(layers):
        ax = axes[layer_idx * 2]
        for stat in stats_to_plot:
            key = f"{layer}_weight_{stat}"
            if key in parameters_history[0]["stats"]:
                values = [p["stats"].get(key, 0) for p in parameters_history]
                ax.plot(epochs, values, label=stat, linewidth=1.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel("weight value")
        ax.set_title(f"{layer.upper()} Weight Statistics")
        ax.legend()

        ax = axes[layer_idx * 2 + 1]
        for stat in stats_to_plot:
            key = f"{layer}_bias_{stat}"
            if key in parameters_history[0]["stats"]:
                values = [p["stats"].get(key, 0) for p in parameters_history]
                ax.plot(epochs, values, label=stat, linewidth=1.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel("bias value")
        ax.set_title(f"{layer.upper()} Bias Statistics")
        ax.legend()

    fig.tight_layout()
    try:
        output_path = f"{output_dir}/parameter_evolution_weights.png"
        fig.savefig(output_path)
        plt.close(fig)
    except Exception:
        plt.close(fig)
        raise

    if dynamics_type == "sl":
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        param_names = ["lambda_param", "omega_param", "gamma_real", "gamma_imag"]
        for idx, param_name in enumerate(param_names):
            ax = axes[idx]
            for stat in ["mean", "std", "min", "max"]:
                key = f"{param_name}_{stat}"
                if key in parameters_history[0]["stats"]:
                    values = [p["stats"].get(key, 0) for p in parameters_history]
                    ax.plot(epochs, values, label=stat, linewidth=1.5)
            ax.set_xlabel("epoch")
            ax.set_ylabel("value")
            ax.set_title(param_name.replace("_", " ").title())
            ax.legend()

        fig.tight_layout()
        try:
            output_path = f"{output_dir}/parameter_evolution_dynamics.png"
            fig.savefig(output_path)
            plt.close(fig)
        except Exception:
            plt.close(fig)
            raise

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for layer_idx, layer in enumerate(layers):
        ax_weight = axes[layer_idx * 2]
        ax_bias = axes[layer_idx * 2 + 1]

        weight_norms = [p["stats"].get(f"{layer}_weight_norm", 0) for p in parameters_history]
        bias_norms = [p["stats"].get(f"{layer}_bias_norm", 0) for p in parameters_history]

        ax_weight.plot(epochs, weight_norms, color=thesis_blue, linewidth=2)
        ax_weight.set_xlabel("epoch")
        ax_weight.set_ylabel("Frobenius norm")
        ax_weight.set_title(f"{layer.upper()} Weight Matrix Norm")

        ax_bias.plot(epochs, bias_norms, color=thesis_red, linewidth=2)
        ax_bias.set_xlabel("epoch")
        ax_bias.set_ylabel("L2 norm")
        ax_bias.set_title(f"{layer.upper()} Bias Vector Norm")

    fig.tight_layout()
    try:
        output_path = f"{output_dir}/parameter_evolution_norms.png"
        fig.savefig(output_path)
        plt.close(fig)
    except Exception:
        plt.close(fig)
        raise


def plot_weight_heatmaps(parameters_history, output_dir, epoch_indices=None):
    if not parameters_history:
        return

    if epoch_indices is None:
        total_epochs = len(parameters_history)
        epoch_indices = [0, total_epochs // 4, total_epochs // 2, 3 * total_epochs // 4, total_epochs - 1]
        epoch_indices = [min(i, total_epochs - 1) for i in epoch_indices if i < total_epochs]

    layers = ["i2h", "h2h", "h2o"]

    for layer in layers:
        fig, axes = plt.subplots(1, len(epoch_indices), figsize=(4 * len(epoch_indices), 4))
        if len(epoch_indices) == 1:
            axes = [axes]

        for idx, epoch_idx in enumerate(epoch_indices):
            if epoch_idx >= len(parameters_history):
                continue
            params = parameters_history[epoch_idx]["params"]
            weight_key = f"{layer}_weight"
            if weight_key in params:
                weight_matrix = np.array(params[weight_key])
                im = axes[idx].imshow(weight_matrix, aspect="auto", cmap=mycmap, interpolation="nearest")
                axes[idx].set_title(f"Epoch {parameters_history[epoch_idx]['epoch']}")
                axes[idx].set_xlabel("input dimension" if "i2h" in layer or "h2h" in layer else "hidden dimension")
                axes[idx].set_ylabel("output dimension")
                plt.colorbar(im, ax=axes[idx])

        fig.suptitle(f"{layer.upper()} Weight Matrices Over Training", fontsize=14)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/weight_heatmap_{layer}.png")
        plt.close(fig)


def plot_weight_distributions(parameters_history, output_dir, epoch_indices=None):
    if not parameters_history:
        return

    if epoch_indices is None:
        total_epochs = len(parameters_history)
        epoch_indices = [0, total_epochs // 2, total_epochs - 1]
        epoch_indices = [min(i, total_epochs - 1) for i in epoch_indices if i < total_epochs]

    layers = ["i2h", "h2h", "h2o"]

    for layer in layers:
        fig, axes = plt.subplots(1, len(epoch_indices), figsize=(5 * len(epoch_indices), 4))
        if len(epoch_indices) == 1:
            axes = [axes]

        for idx, epoch_idx in enumerate(epoch_indices):
            if epoch_idx >= len(parameters_history):
                continue
            params = parameters_history[epoch_idx]["params"]
            weight_key = f"{layer}_weight"
            if weight_key in params:
                weight_matrix = np.array(params[weight_key])
                weights_flat = weight_matrix.flatten()
                axes[idx].hist(weights_flat, bins=50, color=thesis_blue, alpha=0.7, edgecolor="black")
                axes[idx].axvline(0, color=thesis_red, linestyle="--", linewidth=1.5, label="zero")
                axes[idx].axvline(
                    np.mean(weights_flat),
                    color=ifisc_green,
                    linestyle="--",
                    linewidth=1.5,
                    label=f"mean: {np.mean(weights_flat):.4f}",
                )
                axes[idx].set_xlabel("weight value")
                axes[idx].set_ylabel("frequency")
                axes[idx].set_title(f"Epoch {parameters_history[epoch_idx]['epoch']}")
                axes[idx].legend()

        fig.suptitle(f"{layer.upper()} Weight Distributions Over Training", fontsize=14)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/weight_distribution_{layer}.png")
        plt.close(fig)


def plot_connection_strength_evolution(parameters_history, output_dir):
    if not parameters_history:
        return

    epochs = [p["epoch"] for p in parameters_history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    layers = ["i2h", "h2h", "h2o"]
    for layer_idx, layer in enumerate(layers[:3]):
        ax = axes[layer_idx]

        mean_abs_weights = [p["stats"].get(f"{layer}_weight_abs_mean", 0) for p in parameters_history]
        std_weights = [p["stats"].get(f"{layer}_weight_std", 0) for p in parameters_history]

        ax.plot(epochs, mean_abs_weights, label="mean |weight|", color=thesis_blue, linewidth=2)
        ax.plot(epochs, std_weights, label="std(weight)", color=thesis_red, linewidth=2, linestyle="--")
        ax.set_xlabel("epoch")
        ax.set_ylabel("value")
        ax.set_title(f"{layer.upper()} Connection Strength Evolution")
        ax.legend()

    ax = axes[3]
    h2h_norms = [p["stats"].get("h2h_weight_norm", 0) for p in parameters_history]
    h2h_abs_means = [p["stats"].get("h2h_weight_abs_mean", 0) for p in parameters_history]

    ax2 = ax.twinx()
    line1 = ax.plot(epochs, h2h_norms, label="Frobenius norm", color=thesis_blue, linewidth=2)
    line2 = ax2.plot(epochs, h2h_abs_means, label="mean |weight|", color=thesis_red, linewidth=2, linestyle="--")

    ax.set_xlabel("epoch")
    ax.set_ylabel("Frobenius norm", color=thesis_blue)
    ax2.set_ylabel("mean |weight|", color=thesis_red)
    ax.set_title("H2H Recurrent Layer Strength")
    ax.tick_params(axis="y", labelcolor=thesis_blue)
    ax2.tick_params(axis="y", labelcolor=thesis_red)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper left")

    fig.tight_layout()
    fig.savefig(f"{output_dir}/connection_strength_evolution.png")
    plt.close(fig)


def plot_weight_heatmap_single(parameters_history, output_dir, epoch, layer, vmin=None, vmax=None):
    if not parameters_history or epoch >= len(parameters_history):
        return False

    params = parameters_history[epoch]["params"]
    weight_key = f"{layer}_weight"
    if weight_key not in params:
        return False

    weight_matrix = np.array(params[weight_key])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(weight_matrix, aspect="auto", cmap=mycmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(f"{layer.upper()} Weight Matrix - Epoch {epoch}")
    if "i2h" in layer or "h2h" in layer:
        ax.set_xlabel("input dimension")
    else:
        ax.set_xlabel("hidden dimension")
    ax.set_ylabel("output dimension")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/weight_heatmap_{layer}_epoch{epoch:02d}.png")
    plt.close(fig)
    return True


def calculate_weight_limits(parameters_history):
    limits = {}
    layers = ["i2h", "h2h", "h2o"]

    for layer in layers:
        all_weights = []
        for params_dict in parameters_history:
            weight_key = f"{layer}_weight"
            if weight_key in params_dict["params"]:
                weights = np.array(params_dict["params"][weight_key])
                all_weights.append(weights.flatten())

        if all_weights:
            all_weights = np.concatenate(all_weights)
            limits[layer] = {
                "vmin": float(np.min(all_weights)),
                "vmax": float(np.max(all_weights)),
            }
        else:
            limits[layer] = {"vmin": None, "vmax": None}

    return limits
