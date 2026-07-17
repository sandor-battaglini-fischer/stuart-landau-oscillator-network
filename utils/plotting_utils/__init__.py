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
from .class_encoding import plot_smnist_digit_encoding_analysis
from .smnist_pixel_reconstruction import (
    build_digit_prototypes,
    classify_reconstructed_digits,
    evaluate_pixel_reconstruction,
    plot_pixel_metric_heatmaps,
    plot_pixel_metrics_vs_scan_index,
    plot_pixel_metrics_vs_scan_index_by_digit,
    plot_pixel_reconstruction_epoch,
    plot_reconstruction_examples,
    plot_reconstruction_scatter,
    prepare_smnist_sequence,
)
from .sentiment_encoding import (
    find_class_examples_batch,
    plot_imdb_sentiment_encoding_analysis,
)
from .mackey_glass_encoding import (
    collect_mg_forecast_batches,
    plot_mackey_glass_encoding_analysis,
    plot_mackey_glass_encoding_analysis_from_loader,
)
from .signal_stages import (
    find_class_examples,
    plot_imdb_sentiment_comparison,
    plot_signal_stages_for_example,
    prepare_imdb_sequence,
)
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
    "plot_signal_stages_for_example",
    "prepare_imdb_sequence",
    "find_class_examples",
    "plot_imdb_sentiment_comparison",
    "find_class_examples_batch",
    "plot_imdb_sentiment_encoding_analysis",
    "plot_smnist_digit_encoding_analysis",
    "prepare_smnist_sequence",
    "build_digit_prototypes",
    "classify_reconstructed_digits",
    "evaluate_pixel_reconstruction",
    "plot_pixel_metric_heatmaps",
    "plot_pixel_metrics_vs_scan_index",
    "plot_pixel_metrics_vs_scan_index_by_digit",
    "plot_reconstruction_examples",
    "plot_reconstruction_scatter",
    "plot_pixel_reconstruction_epoch",
    "collect_mg_forecast_batches",
    "plot_mackey_glass_encoding_analysis",
    "plot_mackey_glass_encoding_analysis_from_loader",
]
