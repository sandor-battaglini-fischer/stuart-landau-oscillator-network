import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from .style import ifisc_green, thesis_blue, thesis_red

EARLY_PIXEL_N = 196  # first 7 rows in row-major scan (28*7)
LATE_PIXEL_N = 196   # last 7 rows in row-major scan (28*7)


def prepare_smnist_sequence(images, shuffle_perm=None):
    seq = images.reshape(images.size(0), 1, 784).permute(2, 0, 1)
    if shuffle_perm is not None:
        seq = seq[shuffle_perm, :, :]
    return seq


def flatten_targets(images):
    return images.reshape(images.size(0), 784)


def compute_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def build_digit_prototypes(train_loader, num_classes=10):
    sums = np.zeros((num_classes, 784), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.float64)

    for images, labels in train_loader:
        flat = flatten_targets(images).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        for img, label in zip(flat, labels_np):
            sums[label] += img
            counts[label] += 1.0

    counts = np.maximum(counts, 1.0)
    return sums / counts[:, None]


def classify_reconstructed_digits(preds, prototypes):
    preds = np.asarray(preds, dtype=np.float64)
    prototypes = np.asarray(prototypes, dtype=np.float64)
    distances = np.sum((preds[:, None, :] - prototypes[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1)


def evaluate_pixel_reconstruction(
    data_loader,
    model,
    batch_size,
    shuffle_perm=None,
    pixel_threshold=0.1,
    digit_prototypes=None,
):
    model.eval()
    total_loss = 0.0
    total_count = 0
    sum_sq_error = np.zeros(784, dtype=np.float64)
    sum_abs_error = np.zeros(784, dtype=np.float64)
    correct_counts = np.zeros(784, dtype=np.float64)
    sample_count = 0

    all_preds = []
    all_targets = []
    all_labels = []

    loss_fn = torch.nn.MSELoss(reduction="sum")

    with torch.no_grad():
        for images, labels in data_loader:
            batch_size_current = images.size(0)
            inputs = prepare_smnist_sequence(images, shuffle_perm)
            targets = flatten_targets(images)

            output = model(inputs)
            preds = output["output"]

            batch_loss = loss_fn(preds, targets).item()
            total_loss += batch_loss
            total_count += batch_size_current * 784

            preds_np = preds.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            abs_err = np.abs(preds_np - targets_np)

            sum_sq_error += np.sum((preds_np - targets_np) ** 2, axis=0)
            sum_abs_error += np.sum(abs_err, axis=0)
            correct_counts += np.sum(abs_err < pixel_threshold, axis=0)
            sample_count += batch_size_current

            all_preds.append(preds_np)
            all_targets.append(targets_np)
            all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    per_pixel_mse = sum_sq_error / max(sample_count, 1)
    per_pixel_mae = sum_abs_error / max(sample_count, 1)
    per_pixel_acc = correct_counts / max(sample_count, 1)
    per_pixel_r2 = np.array(
        [compute_r2(all_targets[:, i], all_preds[:, i]) for i in range(784)],
        dtype=np.float64,
    )

    overall_mse = total_loss / max(total_count, 1)
    overall_r2 = compute_r2(all_targets.reshape(-1), all_preds.reshape(-1))
    mean_pixel_acc = float(np.mean(per_pixel_acc))

    per_digit = {}
    for digit in range(10):
        mask = all_labels == digit
        n_digit = int(np.sum(mask))
        if n_digit == 0:
            continue
        digit_preds = all_preds[mask]
        digit_targets = all_targets[mask]
        digit_abs_err = np.abs(digit_preds - digit_targets)
        per_digit[digit] = {
            "sample_count": n_digit,
            "per_pixel_mse": np.mean((digit_preds - digit_targets) ** 2, axis=0),
            "per_pixel_mae": np.mean(digit_abs_err, axis=0),
            "per_pixel_acc": np.mean(digit_abs_err < pixel_threshold, axis=0),
            "per_pixel_r2": np.array(
                [compute_r2(digit_targets[:, i], digit_preds[:, i]) for i in range(784)],
                dtype=np.float64,
            ),
            "overall_mse": float(np.mean((digit_preds - digit_targets) ** 2)),
            "overall_r2": float(compute_r2(digit_targets.reshape(-1), digit_preds.reshape(-1))),
            "mean_pixel_acc": float(np.mean(digit_abs_err < pixel_threshold)),
        }

    result = {
        "overall_mse": overall_mse,
        "overall_r2": overall_r2,
        "mean_pixel_acc": mean_pixel_acc,
        "per_pixel_mse": per_pixel_mse,
        "per_pixel_mae": per_pixel_mae,
        "per_pixel_acc": per_pixel_acc,
        "per_pixel_r2": per_pixel_r2,
        "per_digit": per_digit,
        "preds": all_preds,
        "targets": all_targets,
        "labels": all_labels,
        "sample_count": sample_count,
    }

    if digit_prototypes is not None:
        digit_preds = classify_reconstructed_digits(all_preds, digit_prototypes)
        digit_acc = 100.0 * np.mean(digit_preds == all_labels)
        result["digit_preds"] = digit_preds
        result["digit_acc"] = float(digit_acc)

    return result


def _save_heatmap(matrix_28x28, output_path, title, cbar_label, vmin=None, vmax=None, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix_28x28, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(output_path, transparent=True)
    plt.close(fig)


def plot_pixel_metric_heatmaps(metrics, output_dir, epoch=None, pixel_threshold=0.1):
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_epoch{epoch:02d}" if epoch is not None else ""

    acc_map = metrics["per_pixel_acc"].reshape(28, 28)
    mse_map = metrics["per_pixel_mse"].reshape(28, 28)
    r2_map = metrics["per_pixel_r2"].reshape(28, 28)

    _save_heatmap(
        acc_map,
        os.path.join(output_dir, f"pixel_accuracy_heatmap{suffix}.png"),
        f"Per-pixel accuracy ($|error| < {pixel_threshold:.2f}$)",
        "accuracy",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    _save_heatmap(
        mse_map,
        os.path.join(output_dir, f"pixel_mse_heatmap{suffix}.png"),
        "Per-pixel MSE",
        "MSE",
        cmap="magma_r",
    )
    _save_heatmap(
        r2_map,
        os.path.join(output_dir, f"pixel_r2_heatmap{suffix}.png"),
        "Per-pixel $R^2$",
        r"$R^2$",
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )


def plot_pixel_metrics_vs_scan_index(metrics, output_dir, epoch=None):
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_epoch{epoch:02d}" if epoch is not None else ""
    t = np.arange(784)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, metrics["per_pixel_acc"], color=thesis_blue, linewidth=1.5)
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Per-pixel accuracy vs scan index")

    axes[1].plot(t, metrics["per_pixel_mse"], color=thesis_red, linewidth=1.5)
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Per-pixel MSE vs scan index")

    axes[2].plot(t, metrics["per_pixel_r2"], color=ifisc_green, linewidth=1.5)
    axes[2].set_ylabel(r"$R^2$")
    axes[2].set_xlabel("pixel index (row-major scan)")
    axes[2].set_title("Per-pixel $R^2$ vs scan index")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"pixel_metrics_vs_index{suffix}.png"), transparent=True)
    plt.close(fig)


def plot_pixel_metrics_vs_scan_index_by_digit(metrics, output_dir, epoch=None):
    per_digit = metrics.get("per_digit")
    if not per_digit:
        return

    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_epoch{epoch:02d}" if epoch is not None else ""
    t = np.arange(784)
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    for digit in sorted(per_digit.keys()):
        digit_metrics = per_digit[digit]
        color = cmap(int(digit) % 10)
        label = f"{digit} (n={digit_metrics['sample_count']})"
        axes[0].plot(t, digit_metrics["per_pixel_acc"], color=color, linewidth=1.2, alpha=0.9, label=label)
        axes[1].plot(t, digit_metrics["per_pixel_mse"], color=color, linewidth=1.2, alpha=0.9, label=label)
        axes[2].plot(t, digit_metrics["per_pixel_r2"], color=color, linewidth=1.2, alpha=0.9, label=label)

    axes[0].plot(t, metrics["per_pixel_acc"], color="black", linewidth=2.0, linestyle="--", label="all")
    axes[1].plot(t, metrics["per_pixel_mse"], color="black", linewidth=2.0, linestyle="--", label="all")
    axes[2].plot(t, metrics["per_pixel_r2"], color="black", linewidth=2.0, linestyle="--", label="all")

    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Per-pixel accuracy vs scan index (by digit)")
    axes[0].legend(loc="upper right", ncol=2, fontsize=12)

    axes[1].set_ylabel("MSE")
    axes[1].set_title("Per-pixel MSE vs scan index (by digit)")

    axes[2].set_ylabel(r"$R^2$")
    axes[2].set_xlabel("pixel index (row-major scan)")
    axes[2].set_title("Per-pixel $R^2$ vs scan index (by digit)")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"pixel_metrics_vs_index_by_digit{suffix}.png"), transparent=True)
    plt.close(fig)


def plot_reconstruction_examples(
    preds,
    targets,
    labels,
    output_dir,
    epoch=None,
    num_examples=8,
    example_indices=None,
):
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_epoch{epoch:02d}" if epoch is not None else ""

    if example_indices is None:
        unique_labels = np.unique(labels)
        example_indices = []
        for digit in unique_labels:
            idx = np.where(labels == digit)[0]
            if len(idx) > 0:
                example_indices.append(int(idx[0]))
            if len(example_indices) >= num_examples:
                break
        while len(example_indices) < min(num_examples, len(labels)):
            for i in range(len(labels)):
                if i not in example_indices:
                    example_indices.append(i)
                    break
            else:
                break
    else:
        example_indices = list(example_indices[:num_examples])

    n = len(example_indices)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows * 3, ncols, figsize=(3.2 * ncols, 3.0 * nrows * 3))
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1:
        axes = axes.reshape(1, -1)
    if axes.shape[1] == 1:
        axes = axes.reshape(-1, 1)

    for col, idx in enumerate(example_indices):
        col_idx = col % ncols
        row_block = col // ncols

        true_img = targets[idx].reshape(28, 28)
        pred_img = preds[idx].reshape(28, 28)
        err_img = np.abs(pred_img - true_img)

        ax_true = axes[row_block * 3 + 0, col_idx]
        ax_pred = axes[row_block * 3 + 1, col_idx]
        ax_err = axes[row_block * 3 + 2, col_idx]

        ax_true.imshow(true_img, cmap="gray", vmin=0.0, vmax=1.0)
        ax_true.set_title(f"true ({labels[idx]})")
        ax_true.axis("off")

        ax_pred.imshow(pred_img, cmap="gray", vmin=0.0, vmax=1.0)
        ax_pred.set_title("predicted")
        ax_pred.axis("off")

        im = ax_err.imshow(err_img, cmap="magma", vmin=0.0, vmax=1.0)
        ax_err.set_title("|error|")
        ax_err.axis("off")
        fig.colorbar(im, ax=ax_err, fraction=0.046, pad=0.04)

    for ax in axes.flat:
        if not ax.images:
            ax.axis("off")

    fig.suptitle("Image reconstruction examples", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"reconstruction_examples{suffix}.png"), transparent=True)
    plt.close(fig)


def plot_reconstruction_scatter(metrics, output_dir, epoch=None, max_points=5000):
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_epoch{epoch:02d}" if epoch is not None else ""

    y_true = metrics["targets"].reshape(-1)
    y_pred = metrics["preds"].reshape(-1)
    if len(y_true) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y_true), size=max_points, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=4, alpha=0.15, color=thesis_blue)
    lim_min = min(y_true.min(), y_pred.min())
    lim_max = max(y_true.max(), y_pred.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color=thesis_red, linestyle="--", linewidth=1.5)
    ax.set_xlabel("true pixel value")
    ax.set_ylabel("predicted pixel value")
    ax.set_title(f"Pixel scatter ($R^2$={metrics['overall_r2']:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"reconstruction_scatter{suffix}.png"), transparent=True)
    plt.close(fig)


def plot_pixel_reconstruction_epoch(
    output_dir,
    ep_dir,
    epoch,
    train_losses,
    val_losses,
    test_losses,
    val_r2_scores,
    test_r2_scores,
    val_mean_pixel_accs,
    test_mean_pixel_accs,
    test_metrics,
    parameters_history,
    pixel_threshold=0.1,
    num_examples=8,
    is_last_epoch=False,
):
    from .classification import plot_confusion_matrix
    from .regression import plot_regression_metrics
    from .training import plot_epoch_weight_heatmaps

    plot_regression_metrics(
        train_losses,
        val_losses,
        test_losses,
        test_r2_scores=test_r2_scores,
        val_r2_scores=val_r2_scores,
        output_dir=output_dir,
    )

    plot_pixel_metric_heatmaps(test_metrics, ep_dir, epoch=epoch, pixel_threshold=pixel_threshold)
    plot_pixel_metrics_vs_scan_index(test_metrics, ep_dir, epoch=epoch)
    plot_pixel_metrics_vs_scan_index_by_digit(test_metrics, ep_dir, epoch=epoch)
    plot_reconstruction_examples(
        test_metrics["preds"],
        test_metrics["targets"],
        test_metrics["labels"],
        ep_dir,
        epoch=epoch,
        num_examples=num_examples,
    )
    plot_reconstruction_scatter(test_metrics, ep_dir, epoch=epoch)

    confusion_matrix = None
    if "digit_preds" in test_metrics:
        confusion_matrix = plot_confusion_matrix(
            test_metrics["labels"],
            test_metrics["digit_preds"],
            ep_dir,
            num_classes=10,
            epoch=epoch,
            title="Reconstructed digit confusion matrix",
        )

    if parameters_history:
        plot_epoch_weight_heatmaps(parameters_history, ep_dir, epoch)

    if is_last_epoch:
        import shutil

        promote_map = {
            f"pixel_accuracy_heatmap_epoch{epoch:02d}.png": "pixel_accuracy_heatmap.png",
            f"pixel_mse_heatmap_epoch{epoch:02d}.png": "pixel_mse_heatmap.png",
            f"pixel_r2_heatmap_epoch{epoch:02d}.png": "pixel_r2_heatmap.png",
            f"pixel_metrics_vs_index_epoch{epoch:02d}.png": "pixel_metrics_vs_index.png",
            f"pixel_metrics_vs_index_by_digit_epoch{epoch:02d}.png": "pixel_metrics_vs_index_by_digit.png",
            f"reconstruction_examples_epoch{epoch:02d}.png": "reconstruction_examples.png",
            f"reconstruction_scatter_epoch{epoch:02d}.png": "reconstruction_scatter.png",
            f"confusion_matrix_epoch{epoch:02d}.png": "confusion_matrix.png",
        }
        for src_name, dst_name in promote_map.items():
            src = os.path.join(ep_dir, src_name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, dst_name))

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs_so_far = np.arange(len(test_mean_pixel_accs))
    ax.plot(epochs_so_far, val_mean_pixel_accs, label="val mean pixel acc", color=thesis_blue, linestyle="--")
    ax.plot(epochs_so_far, test_mean_pixel_accs, label="test mean pixel acc", color=thesis_red)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean per-pixel accuracy")
    ax.set_title(f"Mean pixel accuracy ($|error| < {pixel_threshold:.2f}$)")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mean_pixel_accuracy_over_time.png"), transparent=True)
    plt.close(fig)

    return test_metrics, confusion_matrix


def truncate_decimals(x, n_decimals):
    if n_decimals is None:
        return x
    scale = 10.0 ** int(n_decimals)
    return torch.round(x * scale) / scale


def _oscillator_core(model):
    return model.slon if hasattr(model, "slon") else model


def extract_final_features(model, inputs):
    core = _oscillator_core(model)
    out = core(inputs, record=True)
    if "rec_z_real" in out and out["rec_z_real"] is not None:
        z_real = out["rec_z_real"][:, -1, :]
        z_imag = out["rec_z_imag"][:, -1, :]
        return torch.cat([z_real, z_imag], dim=1)
    if "rec_x_t" in out and out["rec_x_t"] is not None:
        return out["rec_x_t"][:, -1, :]
    raise ValueError("Model did not return recorded final-state features")


def collect_final_features_and_targets(data_loader, model, shuffle_perm=None):
    model.eval()
    all_features = []
    all_targets = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            inputs = prepare_smnist_sequence(images, shuffle_perm)
            targets = flatten_targets(images)
            features = extract_final_features(model, inputs)
            all_features.append(features.detach().cpu())
            all_targets.append(targets.detach().cpu())
            all_labels.append(labels.detach().cpu())

    return (
        torch.cat(all_features, dim=0),
        torch.cat(all_targets, dim=0),
        torch.cat(all_labels, dim=0).numpy(),
    )


def metrics_from_predictions(
    preds,
    targets,
    labels=None,
    pixel_threshold=0.1,
    early_n=EARLY_PIXEL_N,
    late_n=LATE_PIXEL_N,
):
    preds_np = preds.detach().cpu().numpy() if torch.is_tensor(preds) else np.asarray(preds)
    targets_np = targets.detach().cpu().numpy() if torch.is_tensor(targets) else np.asarray(targets)
    abs_err = np.abs(preds_np - targets_np)
    per_pixel_mse = np.mean((preds_np - targets_np) ** 2, axis=0)
    per_pixel_acc = np.mean(abs_err < pixel_threshold, axis=0)
    per_pixel_r2 = np.array(
        [compute_r2(targets_np[:, i], preds_np[:, i]) for i in range(preds_np.shape[1])],
        dtype=np.float64,
    )
    result = {
        "overall_mse": float(np.mean((preds_np - targets_np) ** 2)),
        "overall_r2": float(compute_r2(targets_np.reshape(-1), preds_np.reshape(-1))),
        "mean_pixel_acc": float(np.mean(per_pixel_acc)),
        "early_pixel_acc": float(np.mean(per_pixel_acc[:early_n])),
        "late_pixel_acc": float(np.mean(per_pixel_acc[-late_n:])),
        "early_pixel_mse": float(np.mean(per_pixel_mse[:early_n])),
        "late_pixel_mse": float(np.mean(per_pixel_mse[-late_n:])),
        "per_pixel_mse": per_pixel_mse,
        "per_pixel_acc": per_pixel_acc,
        "per_pixel_r2": per_pixel_r2,
        "preds": preds_np,
        "targets": targets_np,
    }
    if labels is not None:
        result["labels"] = np.asarray(labels)
    return result


def evaluate_precision_truncation(
    data_loader,
    model,
    decimal_levels,
    shuffle_perm=None,
    pixel_threshold=0.1,
    early_n=EARLY_PIXEL_N,
    late_n=LATE_PIXEL_N,
):
    features, targets, labels = collect_final_features_and_targets(
        data_loader, model, shuffle_perm=shuffle_perm
    )
    core = _oscillator_core(model)
    h2o = core.h2o

    sweep = []
    levels = list(decimal_levels)
    if None not in levels:
        levels = [None] + levels

    with torch.no_grad():
        for n_decimals in levels:
            truncated = truncate_decimals(features, n_decimals)
            preds = h2o(truncated)
            metrics = metrics_from_predictions(
                preds,
                targets,
                labels=labels,
                pixel_threshold=pixel_threshold,
                early_n=early_n,
                late_n=late_n,
            )
            metrics["n_decimals"] = n_decimals
            metrics["label"] = "full" if n_decimals is None else str(int(n_decimals))
            sweep.append(metrics)

    return {
        "levels": sweep,
        "early_n": early_n,
        "late_n": late_n,
        "pixel_threshold": pixel_threshold,
    }


def plot_precision_truncation(sweep_result, output_dir, epoch=None):
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_epoch{epoch:02d}" if epoch is not None else ""
    levels = sweep_result["levels"]

    x_labels = [m["label"] for m in levels]
    x = np.arange(len(levels))

    overall_r2 = [m["overall_r2"] for m in levels]
    overall_mse = [m["overall_mse"] for m in levels]
    mean_acc = [m["mean_pixel_acc"] for m in levels]
    early_acc = [m["early_pixel_acc"] for m in levels]
    late_acc = [m["late_pixel_acc"] for m in levels]
    early_mse = [m["early_pixel_mse"] for m in levels]
    late_mse = [m["late_pixel_mse"] for m in levels]

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(x, overall_r2, color=thesis_blue, marker="o", linewidth=2, label=r"overall $R^2$")
    axes[0].set_ylabel(r"$R^2$")
    axes[0].set_title("Reconstruction vs final-state decimal precision")
    axes[0].legend(loc="best")

    axes[1].plot(x, overall_mse, color=thesis_red, marker="o", linewidth=2, label="overall MSE")
    axes[1].plot(
        x,
        early_mse,
        color=thesis_blue,
        marker="s",
        linestyle="--",
        label=f"early MSE (first {sweep_result['early_n']} px)",
    )
    axes[1].plot(
        x,
        late_mse,
        color=ifisc_green,
        marker="^",
        linestyle="--",
        label=f"late MSE (last {sweep_result['late_n']} px)",
    )
    axes[1].set_ylabel("MSE")
    axes[1].legend(loc="best")

    axes[2].plot(x, mean_acc, color="black", marker="o", linewidth=2, label="mean pixel acc")
    axes[2].plot(
        x,
        early_acc,
        color=thesis_blue,
        marker="s",
        linestyle="--",
        label=f"early pixel acc (first {sweep_result['early_n']} px / top 7 rows)",
    )
    axes[2].plot(
        x,
        late_acc,
        color=ifisc_green,
        marker="^",
        linestyle="--",
        label=f"late pixel acc (last {sweep_result['late_n']} px / bottom 7 rows)",
    )
    axes[2].set_ylabel("accuracy")
    axes[2].set_xlabel("kept decimals in final state (full = no truncation)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(x_labels)
    axes[2].set_ylim(0.0, 1.05)
    axes[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"precision_truncation{suffix}.png"), transparent=True)
    plt.close(fig)

    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    t = np.arange(784)
    cmap = plt.get_cmap("viridis")
    n_curves = len(levels)
    for i, m in enumerate(levels):
        color = cmap(i / max(n_curves - 1, 1))
        label = f"decimals={m['label']}"
        axes2[0].plot(t, m["per_pixel_acc"], color=color, linewidth=1.2, alpha=0.9, label=label)
        axes2[1].plot(t, m["per_pixel_mse"], color=color, linewidth=1.2, alpha=0.9, label=label)
    axes2[0].set_ylabel("accuracy")
    axes2[0].set_title("Per-pixel accuracy vs scan index at truncated precision")
    axes2[0].legend(loc="upper right", ncol=2, fontsize=11)
    axes2[1].set_ylabel("MSE")
    axes2[1].set_xlabel("pixel index (row-major scan)")
    axes2[1].set_title("Per-pixel MSE vs scan index at truncated precision")
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, f"precision_truncation_vs_index{suffix}.png"), transparent=True)
    plt.close(fig2)

    return {
        "n_decimals": [m["n_decimals"] for m in levels],
        "labels": x_labels,
        "overall_r2": overall_r2,
        "overall_mse": overall_mse,
        "mean_pixel_acc": mean_acc,
        "early_pixel_acc": early_acc,
        "late_pixel_acc": late_acc,
        "early_pixel_mse": early_mse,
        "late_pixel_mse": late_mse,
    }


def run_and_save_precision_truncation(
    data_loader,
    model,
    output_dir,
    decimal_levels,
    shuffle_perm=None,
    pixel_threshold=0.1,
    epoch=None,
    fh_log=None,
    promote_to_dir=None,
):
    sweep_result = evaluate_precision_truncation(
        data_loader,
        model,
        decimal_levels=decimal_levels,
        shuffle_perm=shuffle_perm,
        pixel_threshold=pixel_threshold,
    )
    summary = plot_precision_truncation(sweep_result, output_dir, epoch=epoch)

    serializable = {
        "epoch": epoch,
        "pixel_threshold": sweep_result["pixel_threshold"],
        "early_n": sweep_result["early_n"],
        "late_n": sweep_result["late_n"],
        "early_definition": (
            f"mean over scan indices [0, {sweep_result['early_n']}) "
            f"= first {sweep_result['early_n']} pixels = top {sweep_result['early_n'] // 28} rows"
        ),
        "late_definition": (
            f"mean over scan indices [-{sweep_result['late_n']}:] "
            f"= last {sweep_result['late_n']} pixels = bottom {sweep_result['late_n'] // 28} rows"
        ),
        "levels": [
            {
                "n_decimals": m["n_decimals"],
                "label": m["label"],
                "overall_mse": m["overall_mse"],
                "overall_r2": m["overall_r2"],
                "mean_pixel_acc": m["mean_pixel_acc"],
                "early_pixel_acc": m["early_pixel_acc"],
                "late_pixel_acc": m["late_pixel_acc"],
                "early_pixel_mse": m["early_pixel_mse"],
                "late_pixel_mse": m["late_pixel_mse"],
            }
            for m in sweep_result["levels"]
        ],
    }
    json_name = (
        f"precision_truncation_epoch{epoch:02d}.json"
        if epoch is not None
        else "precision_truncation.json"
    )
    out_json = os.path.join(output_dir, json_name)
    with open(out_json, "w") as f:
        json.dump(serializable, f, indent=2)

    if promote_to_dir is not None:
        import shutil

        promote_map = {}
        if epoch is not None:
            promote_map = {
                f"precision_truncation_epoch{epoch:02d}.png": "precision_truncation.png",
                f"precision_truncation_vs_index_epoch{epoch:02d}.png": "precision_truncation_vs_index.png",
                f"precision_truncation_epoch{epoch:02d}.json": "precision_truncation.json",
            }
        for src_name, dst_name in promote_map.items():
            src = os.path.join(output_dir, src_name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(promote_to_dir, dst_name))

    if fh_log is not None:
        epoch_tag = f" epoch {epoch}" if epoch is not None else ""
        fh_log.write("\n" + "=" * 60 + "\n")
        fh_log.write(f"Final-state decimal precision truncation{epoch_tag}\n")
        fh_log.write(
            f"early = first {sweep_result['early_n']} scan indices "
            f"(top {sweep_result['early_n'] // 28} rows); "
            f"late = last {sweep_result['late_n']} "
            f"(bottom {sweep_result['late_n'] // 28} rows)\n"
        )
        fh_log.write("=" * 60 + "\n")
        for m in sweep_result["levels"]:
            fh_log.write(
                f"decimals={m['label']}: r2={m['overall_r2']:.4f}, mse={m['overall_mse']:.6f}, "
                f"mean_acc={m['mean_pixel_acc']:.4f}, "
                f"early_acc={m['early_pixel_acc']:.4f}, late_acc={m['late_pixel_acc']:.4f}\n"
            )
        fh_log.write("=" * 60 + "\n")
        fh_log.flush()

    return sweep_result, summary
