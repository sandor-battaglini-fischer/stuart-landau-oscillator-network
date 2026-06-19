from utils.run_dirs import create_gif_from_epoch_dirs

from .classification import plot_classification_metrics, plot_confusion_matrix
from .mackey_glass import (
    plot_all_test_predictions,
    plot_average_predictions,
    plot_mackey_glass_prediction,
    plot_mackey_glass_zoom,
    plot_predictions_on_test_segment,
    plot_scatter_predictions,
)
from .parameters import (
    calculate_weight_limits,
    plot_connection_strength_evolution,
    plot_parameter_evolution,
    plot_weight_distributions,
    plot_weight_heatmap_single,
    plot_weight_heatmaps,
)
from .regression import plot_regression_metrics

LAYERS = ["i2h", "h2h", "h2o"]


def plot_confusion_matrices(test_labels, test_preds, output_dir, num_classes, class_labels=None, epoch=None):
    cm = plot_confusion_matrix(
        test_labels,
        test_preds,
        output_dir,
        num_classes=num_classes,
        class_labels=class_labels,
        epoch=epoch,
        normalize=True,
    )
    plot_confusion_matrix(
        test_labels,
        test_preds,
        output_dir,
        num_classes=num_classes,
        class_labels=class_labels,
        epoch=epoch,
        normalize=False,
    )
    return cm


def plot_parameter_analysis(parameters_history, output_dir, dynamics_type="sl"):
    if len(parameters_history) <= 1:
        return

    if parameters_history and "stats" in parameters_history[0]:
        plot_parameter_evolution(parameters_history, output_dir, dynamics_type)
    plot_weight_heatmaps(parameters_history, output_dir)
    plot_weight_distributions(parameters_history, output_dir)
    plot_connection_strength_evolution(parameters_history, output_dir)


def plot_epoch_weight_heatmaps(parameters_history, ep_dir, epoch):
    weight_limits = calculate_weight_limits(parameters_history)
    for layer in LAYERS:
        vmin = weight_limits[layer]["vmin"]
        vmax = weight_limits[layer]["vmax"]
        if vmin is not None and vmax is not None:
            try:
                plot_weight_heatmap_single(parameters_history, ep_dir, epoch, layer, vmin=vmin, vmax=vmax)
            except Exception:
                pass


def plot_classification_epoch(
    output_dir,
    ep_dir,
    epoch,
    train_accs,
    val_accs,
    test_accs,
    train_losses,
    val_losses,
    test_losses,
    test_labels,
    test_preds,
    parameters_history,
    num_classes=10,
    class_labels=None,
    dynamics_type="sl",
    is_last_epoch=False,
):
    plot_classification_metrics(train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, output_dir)

    cm = None
    try:
        cm = plot_confusion_matrices(
            test_labels,
            test_preds,
            ep_dir,
            num_classes=num_classes,
            class_labels=class_labels,
            epoch=epoch,
        )
    except Exception:
        pass

    try:
        plot_parameter_analysis(parameters_history, output_dir, dynamics_type=dynamics_type)
    except Exception:
        pass

    plot_epoch_weight_heatmaps(parameters_history, ep_dir, epoch)

    if is_last_epoch:
        try:
            plot_confusion_matrices(
                test_labels,
                test_preds,
                output_dir,
                num_classes=num_classes,
                class_labels=class_labels,
                epoch=None,
            )
        except Exception:
            pass

    return cm


def create_classification_gifs(output_dir):
    for layer in LAYERS:
        create_gif_from_epoch_dirs(
            output_dir,
            f"weight_heatmap_{layer}_epoch*.png",
            f"weight_heatmap_{layer}.gif",
            target_size=(600, 500),
        )
    create_gif_from_epoch_dirs(
        output_dir,
        "confusion_matrix_epoch*.png",
        "confusion_matrix_normalized.gif",
        target_size=(1000, 800),
    )


def plot_mackey_glass_snapshots(
    output_dir,
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
    all_test_preds,
    all_test_targets,
    all_test_target_indices,
    epoch=None,
    is_initial=False,
    scatter_xlim=None,
    scatter_ylim=None,
):
    plot_mackey_glass_prediction(
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
        epoch=epoch,
        is_initial=is_initial,
    )
    plot_mackey_glass_zoom(
        full_series,
        example_input_indices,
        example_target_idx,
        ex_pred,
        example_sampled_idx_full,
        start_idx,
        args,
        train_sampled_indices,
        output_dir,
        epoch=epoch,
        is_initial=is_initial,
    )
    plot_all_test_predictions(
        full_series,
        all_test_preds,
        all_test_targets,
        all_test_target_indices,
        train_sampled_indices,
        test_start,
        args,
        output_dir,
        epoch=epoch,
        is_initial=is_initial,
    )
    plot_predictions_on_test_segment(
        full_series,
        all_test_preds,
        all_test_targets,
        all_test_target_indices,
        test_start,
        args,
        output_dir,
        epoch=epoch,
        is_initial=is_initial,
    )

    if not is_initial and epoch is not None:
        plot_scatter_predictions(
            all_test_targets,
            all_test_preds,
            output_dir,
            epoch,
            scatter_xlim=scatter_xlim,
            scatter_ylim=scatter_ylim,
        )
        valid_pairs = [
            (idx, all_test_preds[i], all_test_targets[i])
            for i, idx in enumerate(all_test_target_indices)
            if idx is not None and idx >= test_start and idx < len(full_series)
        ]
        plot_average_predictions(valid_pairs, test_start, args, output_dir, epoch)


def plot_mackey_glass_parameter_analysis(parameters_history, output_dir, epoch, fh_log=None, log_every=10):
    try:
        if parameters_history and "stats" in parameters_history[0]:
            plot_parameter_evolution(parameters_history, output_dir, "sl")
    except Exception as e:
        if (epoch + 1) % log_every == 0 or epoch == 0:
            if fh_log is not None:
                fh_log.write(f"Warning: Failed to generate parameter evolution plots at epoch {epoch}: {e}\n")
                fh_log.flush()

    if len(parameters_history) > 1:
        for plot_fn, name in [
            (plot_weight_heatmaps, "weight heatmaps"),
            (plot_weight_distributions, "weight distribution plots"),
            (plot_connection_strength_evolution, "connection strength plots"),
        ]:
            try:
                plot_fn(parameters_history, output_dir)
            except Exception as e:
                if (epoch + 1) % log_every == 0 or epoch == 0:
                    if fh_log is not None:
                        fh_log.write(f"Warning: Failed to generate {name} at epoch {epoch}: {e}\n")
                        fh_log.flush()


def create_mackey_glass_gifs(output_dir):
    create_gif_from_epoch_dirs(
        output_dir,
        "scatter_epoch*.png",
        "predictions_vs_targets.gif",
        target_size=(800, 800),
    )
    create_gif_from_epoch_dirs(
        output_dir,
        "predictions_on_test_epoch*.png",
        "predictions_on_test.gif",
        target_size=(1000, 400),
    )
    for layer in LAYERS:
        create_gif_from_epoch_dirs(
            output_dir,
            f"weight_heatmap_{layer}_epoch*.png",
            f"weight_heatmap_{layer}.gif",
            target_size=(600, 500),
        )
