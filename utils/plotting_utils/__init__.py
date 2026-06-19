from .style import apply_style, ifisc_green, mycmap, thesis_blue, thesis_red

apply_style()

from .classification import plot_classification_metrics, plot_confusion_matrix
from .gifs import create_gif_from_files
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
from .training import (
    create_classification_gifs,
    create_mackey_glass_gifs,
    plot_classification_epoch,
    plot_confusion_matrices,
    plot_epoch_weight_heatmaps,
    plot_mackey_glass_parameter_analysis,
    plot_mackey_glass_snapshots,
    plot_parameter_analysis,
)

__all__ = [
    "apply_style",
    "thesis_red",
    "thesis_blue",
    "ifisc_green",
    "mycmap",
    "plot_confusion_matrix",
    "plot_confusion_matrices",
    "plot_classification_metrics",
    "plot_classification_epoch",
    "plot_regression_metrics",
    "plot_mackey_glass_prediction",
    "plot_mackey_glass_zoom",
    "plot_all_test_predictions",
    "plot_predictions_on_test_segment",
    "plot_scatter_predictions",
    "plot_average_predictions",
    "plot_mackey_glass_snapshots",
    "plot_parameter_evolution",
    "plot_weight_heatmaps",
    "plot_weight_distributions",
    "plot_connection_strength_evolution",
    "plot_weight_heatmap_single",
    "plot_parameter_analysis",
    "plot_mackey_glass_parameter_analysis",
    "plot_epoch_weight_heatmaps",
    "calculate_weight_limits",
    "create_gif_from_files",
    "create_classification_gifs",
    "create_mackey_glass_gifs",
]
