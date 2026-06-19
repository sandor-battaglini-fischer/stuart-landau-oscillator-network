import numpy as np
import matplotlib.pyplot as plt

from .style import ifisc_green, thesis_blue, thesis_red


def plot_mackey_glass_prediction(
    full_series,
    example_input_indices,
    example_target_idx,
    ex_pred,
    example_sampled_idx_full,
    start_idx,
    tau_steps,
    train_sampled_indices,
    test_start,
    args,
    output_dir,
    epoch=None,
    is_initial=False,
):
    fig, ax = plt.subplots(figsize=(10, 4))
    t_full = np.arange(len(full_series))
    ax.plot(t_full, full_series, color=thesis_blue, alpha=0.3, linewidth=0.5, label="series")
    ax.scatter(
        train_sampled_indices,
        full_series[train_sampled_indices],
        color=thesis_blue,
        s=10,
        alpha=0.4,
        label="samples",
        zorder=3,
    )
    ax.axvspan(start_idx, example_sampled_idx_full, color=ifisc_green, alpha=0.1, label="input window")
    ax.scatter(
        example_input_indices,
        full_series[example_input_indices],
        color=ifisc_green,
        s=35,
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        alpha=0.9,
        label="input samples",
        zorder=5,
    )
    if example_target_idx < len(full_series):
        ax.scatter(
            [example_target_idx],
            [full_series[example_target_idx]],
            color=thesis_red,
            s=60,
            marker="*",
            alpha=0.9,
            label="target (T+1)",
            zorder=5,
        )
        ax.scatter(
            [example_target_idx],
            [ex_pred],
            color=ifisc_green,
            s=60,
            marker="o",
            alpha=0.9,
            label="prediction" if not is_initial else "prediction (initial)",
            zorder=5,
        )
    series_shifted = np.roll(full_series, -args.horizon)
    ax.plot(
        t_full,
        series_shifted,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label="shifted series (T+1)",
    )
    ax.axvline(example_sampled_idx_full, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(example_target_idx, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ymax = np.max(full_series)
    ymin = np.min(full_series)
    yrange = ymax - ymin if ymax > ymin else 1.0
    ax.text(
        start_idx,
        ymax + 0.05 * yrange,
        fr"$L={len(example_input_indices)},\ \tau={tau_steps}$",
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )
    ax.set_xlabel("time (simulation steps)")
    ax.set_ylabel("value")
    title = "Input buffer and T+1 prediction on Mackey–Glass series"
    if is_initial:
        title += " (Initial, before training)"
    if args.remove_top_n_freqs > 0:
        title += f" (top {args.remove_top_n_freqs} frequencies removed)"
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    if epoch is not None:
        fig.savefig(f"{output_dir}/mg_pred_epoch{epoch:02d}.png")
    else:
        fig.savefig(f"{output_dir}/mg_pred_epoch_initial.png")
    plt.close(fig)


def plot_mackey_glass_zoom(
    full_series,
    example_input_indices,
    example_target_idx,
    ex_pred,
    example_sampled_idx_full,
    start_idx,
    args,
    train_sampled_indices,
    output_dir,
    epoch=None,
    is_initial=False,
):
    fig_z, ax_z = plt.subplots(figsize=(10, 4))
    t_full = np.arange(len(full_series))
    ax_z.plot(t_full, full_series, color=thesis_blue, alpha=0.15, linewidth=0.5)
    ax_z.scatter(
        train_sampled_indices,
        full_series[train_sampled_indices],
        color=thesis_blue,
        s=8,
        alpha=0.3,
        zorder=2,
    )
    ax_z.axvspan(start_idx, example_sampled_idx_full, color=ifisc_green, alpha=0.12)
    ax_z.scatter(
        example_input_indices,
        full_series[example_input_indices],
        color=ifisc_green,
        s=35,
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        alpha=0.9,
        zorder=5,
    )
    if example_target_idx < len(full_series):
        ax_z.scatter(
            [example_target_idx],
            [full_series[example_target_idx]],
            color=thesis_red,
            s=60,
            marker="*",
            alpha=0.9,
            zorder=6,
        )
        ax_z.scatter(
            [example_target_idx],
            [ex_pred],
            color=ifisc_green,
            s=60,
            marker="o",
            alpha=0.9,
            zorder=6,
        )
    series_shifted = np.roll(full_series, -args.horizon)
    ax_z.plot(t_full, series_shifted, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    zoom_before = 3 * args.input_length
    zoom_after = args.input_length
    x_min = max(0, example_sampled_idx_full - zoom_before)
    x_max = min(len(full_series) - 1, example_target_idx + zoom_after)
    ax_z.set_xlim(x_min, x_max)
    ax_z.set_xlabel("time (simulation steps)")
    ax_z.set_ylabel("value")
    title_z = "Zoomed view: skipped history and input buffer with T+1"
    if is_initial:
        title_z += " (Initial, before training)"
    if args.remove_top_n_freqs > 0:
        title_z += f" (top {args.remove_top_n_freqs} frequencies removed)"
    ax_z.set_title(title_z)
    fig_z.tight_layout()
    if epoch is not None:
        fig_z.savefig(f"{output_dir}/mg_pred_epoch{epoch:02d}_zoom.png")
    else:
        fig_z.savefig(f"{output_dir}/mg_pred_epoch_initial_zoom.png")
    plt.close(fig_z)


def plot_all_test_predictions(
    full_series,
    all_test_preds,
    all_test_targets,
    all_test_target_indices,
    train_sampled_indices,
    test_start,
    args,
    output_dir,
    epoch=None,
    is_initial=False,
):
    fig_all, axes_all = plt.subplots(1, 3, figsize=(18, 5))
    ax_ts, ax_scatter, ax_error = axes_all

    show_start = max(0, test_start - int(args.series_length * 0.1))
    show_end = len(full_series)
    t_shortened = np.arange(show_start, show_end)
    series_shortened = full_series[show_start:show_end]

    ax_ts.plot(t_shortened, series_shortened, color=thesis_blue, alpha=0.2, linewidth=0.5, label="series")
    train_indices_in_range = train_sampled_indices[
        (train_sampled_indices >= show_start) & (train_sampled_indices < show_end)
    ]
    if len(train_indices_in_range) > 0:
        ax_ts.scatter(
            train_indices_in_range,
            full_series[train_indices_in_range],
            color=thesis_blue,
            s=5,
            alpha=0.3,
            label="train samples",
            zorder=2,
        )
    valid_indices = [
        idx
        for idx in all_test_target_indices
        if idx is not None and idx < len(full_series) and idx >= show_start
    ]
    if valid_indices:
        valid_preds = [
            all_test_preds[i]
            for i, idx in enumerate(all_test_target_indices)
            if idx is not None and idx < len(full_series) and idx >= show_start
        ]
        valid_targets = [
            all_test_targets[i]
            for i, idx in enumerate(all_test_target_indices)
            if idx is not None and idx < len(full_series) and idx >= show_start
        ]
        ax_ts.scatter(
            valid_indices,
            valid_targets,
            color=thesis_red,
            s=15,
            alpha=0.6,
            marker="*",
            label="targets",
            zorder=4,
        )
        ax_ts.scatter(
            valid_indices,
            valid_preds,
            color=ifisc_green,
            s=15,
            alpha=0.6,
            marker="o",
            label="predictions" + (" (initial)" if is_initial else ""),
            zorder=5,
        )
    ax_ts.axvline(test_start, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="test start")
    ax_ts.set_xlabel("time (simulation steps)")
    ax_ts.set_ylabel("value")
    title_ts = "All Test Predictions Overlaid on Time Series (End Portion)"
    if is_initial:
        title_ts += " - Initial, before training"
    if args.remove_top_n_freqs > 0:
        title_ts += f" (top {args.remove_top_n_freqs} frequencies removed)"
    ax_ts.set_title(title_ts)
    ax_ts.legend(loc="upper right", fontsize=9)

    ax_scatter.scatter(
        all_test_targets,
        all_test_preds,
        alpha=0.5,
        s=20,
        color=thesis_blue,
        edgecolors="black",
        linewidths=0.3,
    )
    min_val = min(np.min(all_test_targets), np.min(all_test_preds))
    max_val = max(np.max(all_test_targets), np.max(all_test_preds))
    ax_scatter.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.7, linewidth=2, label="perfect prediction")
    ax_scatter.set_xlabel("target values")
    ax_scatter.set_ylabel("predicted values")
    title_scatter = "Predictions vs Targets (All Test Samples)"
    if is_initial:
        title_scatter += " - Initial, before training"
    elif epoch is not None:
        title_scatter += f" - Epoch {epoch}"
    ax_scatter.set_title(title_scatter)
    ax_scatter.legend()
    ax_scatter.set_aspect("equal", adjustable="box")

    errors = all_test_preds - all_test_targets
    ax_error.hist(errors, bins=50, color=thesis_blue, alpha=0.7, edgecolor="black")
    ax_error.axvline(0, color=thesis_red, linestyle="--", linewidth=2, label="zero error")
    ax_error.axvline(
        np.mean(errors),
        color=ifisc_green,
        linestyle="--",
        linewidth=2,
        label=f"mean error: {np.mean(errors):.4f}",
    )
    ax_error.set_xlabel("prediction error (pred - target)")
    ax_error.set_ylabel("frequency")
    title_error = "Error Distribution"
    if is_initial:
        title_error += " - Initial, before training"
    ax_error.set_title(title_error)
    ax_error.legend()

    fig_all.tight_layout()
    if epoch is not None:
        fig_all.savefig(f"{output_dir}/all_predictions_epoch{epoch:02d}.png")
    else:
        fig_all.savefig(f"{output_dir}/all_predictions_epoch_initial.png")
    plt.close(fig_all)


def plot_predictions_on_test_segment(
    full_series,
    all_test_preds,
    all_test_targets,
    all_test_target_indices,
    test_start,
    args,
    output_dir,
    epoch=None,
    is_initial=False,
):
    fig_pred, ax_pred = plt.subplots(figsize=(10, 4))
    t_test = np.arange(test_start, len(full_series))
    series_test = full_series[test_start:]
    ax_pred.plot(t_test, series_test, color=thesis_blue, alpha=0.2, linewidth=0.7, label="series (test)")

    valid_test_indices = [
        idx for idx in all_test_target_indices if idx is not None and idx >= test_start and idx < len(full_series)
    ]
    if valid_test_indices:
        valid_test_preds = [
            all_test_preds[i]
            for i, idx in enumerate(all_test_target_indices)
            if idx is not None and idx >= test_start and idx < len(full_series)
        ]
        valid_test_targets = [
            all_test_targets[i]
            for i, idx in enumerate(all_test_target_indices)
            if idx is not None and idx >= test_start and idx < len(full_series)
        ]
        ax_pred.scatter(
            valid_test_indices,
            valid_test_targets,
            color=thesis_red,
            s=20,
            alpha=0.9,
            marker="*",
            label="targets",
            zorder=4,
        )
        ax_pred.scatter(
            valid_test_indices,
            valid_test_preds,
            color=ifisc_green,
            s=20,
            alpha=0.9,
            marker="o",
            label="predictions" + (" (initial)" if is_initial else ""),
            zorder=5,
        )
    pred_ymin = np.min(full_series[test_start:])
    pred_ymax = np.max(full_series[test_start:])
    pred_ymargin = (pred_ymax - pred_ymin) * 0.1

    ax_pred.set_xlabel("time (simulation steps)")
    ax_pred.set_ylabel("value")
    title_pred = "Predictions vs Targets on Test Segment"
    if is_initial:
        title_pred += " - Initial, before training"
    if args.remove_top_n_freqs > 0:
        title_pred += f" (top {args.remove_top_n_freqs} frequencies removed)"
    ax_pred.set_title(title_pred)
    ax_pred.legend(loc="upper right", fontsize=9)
    ax_pred.set_xlim(test_start, len(full_series))
    ax_pred.set_ylim(pred_ymin - pred_ymargin, pred_ymax + pred_ymargin)
    fig_pred.tight_layout()
    if epoch is not None:
        fig_pred.savefig(f"{output_dir}/predictions_on_test_epoch{epoch:02d}.png")
    else:
        fig_pred.savefig(f"{output_dir}/predictions_on_test_epoch_initial.png")
    plt.close(fig_pred)


def plot_scatter_predictions(all_test_targets, all_test_preds, output_dir, epoch, scatter_xlim=None, scatter_ylim=None):
    fig_scatter, ax_scatter_single = plt.subplots(figsize=(8, 8))
    ax_scatter_single.scatter(
        all_test_targets,
        all_test_preds,
        alpha=0.5,
        s=20,
        color=thesis_blue,
        edgecolors="black",
        linewidths=0.3,
    )
    min_val = min(np.min(all_test_targets), np.min(all_test_preds))
    max_val = max(np.max(all_test_targets), np.max(all_test_preds))
    if scatter_xlim is not None and scatter_ylim is not None:
        diag_min = min(scatter_xlim[0], scatter_ylim[0])
        diag_max = max(scatter_xlim[1], scatter_ylim[1])
        ax_scatter_single.plot(
            [diag_min, diag_max],
            [diag_min, diag_max],
            "r--",
            alpha=0.7,
            linewidth=2,
            label="perfect prediction",
        )
        ax_scatter_single.set_xlim(scatter_xlim)
        ax_scatter_single.set_ylim(scatter_ylim)
    else:
        ax_scatter_single.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            alpha=0.7,
            linewidth=2,
            label="perfect prediction",
        )
    ax_scatter_single.set_xlabel("target values")
    ax_scatter_single.set_ylabel("predicted values")
    ax_scatter_single.set_title(f"Predictions vs Targets - Epoch {epoch}")
    ax_scatter_single.legend()
    ax_scatter_single.set_aspect("equal", adjustable="box")
    fig_scatter.tight_layout()
    fig_scatter.savefig(f"{output_dir}/scatter_epoch{epoch:02d}.png")
    plt.close(fig_scatter)


def plot_average_predictions(valid_pairs, test_start, args, output_dir, epoch):
    if not valid_pairs:
        return

    times_arr = np.array([p[0] for p in valid_pairs], dtype=int)
    preds_arr = np.array([p[1] for p in valid_pairs], dtype=float)
    targets_arr = np.array([p[2] for p in valid_pairs], dtype=float)

    unique_times, inverse = np.unique(times_arr, return_inverse=True)
    sum_preds = np.bincount(inverse, weights=preds_arr)
    sum_targets = np.bincount(inverse, weights=targets_arr)
    counts = np.bincount(inverse)
    avg_preds = sum_preds / np.maximum(counts, 1)
    avg_targets = sum_targets / np.maximum(counts, 1)

    pred_ymin = min(np.min(avg_targets), np.min(avg_preds))
    pred_ymax = max(np.max(avg_targets), np.max(avg_preds))
    pred_ymargin = (pred_ymax - pred_ymin) * 0.1

    fig_avg, ax_avg = plt.subplots(figsize=(10, 4))
    ax_avg.plot(unique_times, avg_targets, color=thesis_blue, linewidth=1.0, label="target (avg per step)")
    ax_avg.plot(unique_times, avg_preds, color=ifisc_green, linewidth=1.0, label="prediction (avg per step)")
    ax_avg.set_xlabel("time (simulation steps)")
    ax_avg.set_ylabel("value")
    title_avg = f"Average predictions on test segment - Epoch {epoch}"
    if args.remove_top_n_freqs > 0:
        title_avg += f" (top {args.remove_top_n_freqs} frequencies removed)"
    ax_avg.set_title(title_avg)
    ax_avg.set_xlim(test_start, test_start + len(unique_times))
    ax_avg.set_ylim(pred_ymin - pred_ymargin, pred_ymax + pred_ymargin)
    ax_avg.legend(loc="upper right", fontsize=9)
    fig_avg.tight_layout()
    fig_avg.savefig(f"{output_dir}/predictions_on_test_epoch{epoch:02d}_avg.png")
    plt.close(fig_avg)
