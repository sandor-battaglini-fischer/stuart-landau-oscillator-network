#!/usr/bin/env python3
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.plotting_utils import (
    plot_connection_strength_evolution,
    plot_parameter_evolution,
    plot_weight_distributions,
    plot_weight_heatmaps,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate parameter plots from saved training data"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the run directory containing parameters.json",
    )
    parser.add_argument(
        "--dynamics",
        type=str,
        default="auto",
        choices=["dho", "sl", "auto"],
        help="Dynamics type: dho, sl, or auto (default: auto)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=None,
        help="Epoch numbers for heatmaps and distributions",
    )
    parser.add_argument(
        "--heatmap-epochs",
        type=int,
        nargs="+",
        default=None,
        help="Epoch numbers for weight heatmaps only",
    )
    parser.add_argument(
        "--distribution-epochs",
        type=int,
        nargs="+",
        default=None,
        help="Epoch numbers for weight distributions only",
    )
    args = parser.parse_args()

    params_file = os.path.join(args.output_dir, "parameters.json")
    if not os.path.exists(params_file):
        print(f"Error: {params_file} not found!")
        return

    with open(params_file) as f:
        parameters_history = json.load(f)

    if not parameters_history:
        print("Error: parameters.json is empty!")
        return

    print(f"Loaded {len(parameters_history)} parameter snapshots from {params_file}")

    if args.dynamics == "auto":
        dynamics_type = "sl" if "lambda_param" in parameters_history[0]["params"] else "dho"
        print(f"Auto-detected dynamics type: {dynamics_type}")
    else:
        dynamics_type = args.dynamics

    print("Generating parameter evolution plots...")
    plot_parameter_evolution(parameters_history, args.output_dir, dynamics_type)

    epoch_to_idx = {p["epoch"]: idx for idx, p in enumerate(parameters_history)}
    available_epochs = sorted(epoch_to_idx.keys())

    def get_epoch_indices(epoch_list):
        if epoch_list is None:
            return None
        indices = []
        for epoch_num in epoch_list:
            if epoch_num in epoch_to_idx:
                indices.append(epoch_to_idx[epoch_num])
            else:
                print(f"Warning: Epoch {epoch_num} not found. Available: {available_epochs}")
        return sorted(set(indices)) if indices else None

    heatmap_epochs = args.heatmap_epochs if args.heatmap_epochs is not None else args.epochs
    distribution_epochs = (
        args.distribution_epochs if args.distribution_epochs is not None else args.epochs
    )

    print("Generating weight heatmaps...")
    plot_weight_heatmaps(parameters_history, args.output_dir, get_epoch_indices(heatmap_epochs))

    print("Generating weight distribution plots...")
    plot_weight_distributions(
        parameters_history, args.output_dir, get_epoch_indices(distribution_epochs)
    )

    print("Generating connection strength evolution plots...")
    plot_connection_strength_evolution(parameters_history, args.output_dir)

    print(f"All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
