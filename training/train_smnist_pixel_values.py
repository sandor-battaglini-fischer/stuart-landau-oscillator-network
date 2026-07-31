import os
import argparse
import json
from datetime import datetime

import numpy as np
import torch
import torchvision
from tqdm import tqdm

import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TASK = "smnist_pixel_values"
NUM_PIXELS = 784

from utils.model_factory import build_oscillator
from utils.run_dirs import make_run_dir, sweep_summary_path, epoch_dir, save_training_checkpoint
from utils.slon_analysis import extract_model_parameters, compute_parameter_statistics
from utils.plotting_utils import plot_regression_metrics
from utils.plotting_utils.smnist_pixel_reconstruction import (
    prepare_smnist_sequence,
    flatten_targets,
    build_digit_prototypes,
    evaluate_pixel_reconstruction,
    plot_pixel_reconstruction_epoch,
    run_and_save_precision_truncation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SLON training for sMNIST full-image pixel reconstruction"
    )
    parser.add_argument("--num-hidden", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--omega", type=float, default=0.224)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--lambda-param", type=float, default=0.1)
    parser.add_argument("--gamma-real", type=float, default=-0.1)
    parser.add_argument("--gamma-imag", type=float, default=-0.1)
    parser.add_argument("--dynamics", type=str, default="sl", choices=["sl", "lo", "dho"])
    parser.add_argument("--no-tanh", action="store_true")
    parser.add_argument("--sweep-omega", action="store_true")
    parser.add_argument("--omega-min", type=float, default=None)
    parser.add_argument("--omega-max", type=float, default=None)
    parser.add_argument("--omega-steps", type=int, default=10)
    parser.add_argument("--sweep-lambda", action="store_true")
    parser.add_argument("--lambda-min", type=float, default=None)
    parser.add_argument("--lambda-max", type=float, default=None)
    parser.add_argument("--lambda-steps", type=int, default=10)
    parser.add_argument("--skip-epoch-plots", action="store_true")
    parser.add_argument("--sweep-mode", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--pixel-threshold",
        type=float,
        default=0.1,
        help="absolute error threshold for per-pixel accuracy",
    )
    parser.add_argument(
        "--reconstruction-examples",
        type=int,
        default=8,
        help="number of test examples in reconstruction figure",
    )
    parser.add_argument(
        "--precision-truncation",
        action="store_true",
        default=True,
        help="after training, sweep decimal truncation of the final state before readout",
    )
    parser.add_argument(
        "--no-precision-truncation",
        action="store_true",
        help="disable final-state precision truncation analysis",
    )
    parser.add_argument(
        "--precision-decimals",
        type=str,
        default="8,6,4,3,2,1,0",
        help="comma-separated decimal places to keep in the final state (e.g. 8,4,2,1,0)",
    )
    return parser.parse_args()


def build_dataloaders(batch_size_train, batch_size_test):
    size_validation = 1000
    train_set = torchvision.datasets.MNIST(
        root=DATA_DIR,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False,
    )
    test_set = torchvision.datasets.MNIST(
        root=DATA_DIR,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=False,
    )
    train_set, valid_set = torch.utils.data.random_split(
        train_set, [len(train_set) - size_validation, size_validation]
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size_train, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size_test, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size_test, shuffle=False
    )
    return train_loader, valid_loader, test_loader


def train_with_params(args, omega_value, lambda_value=None, sweep_idx=None, sweep_type=None):
    lambda_param = lambda_value if lambda_value is not None else args.lambda_param

    model = build_oscillator(
        args.dynamics,
        1,
        args.num_hidden,
        NUM_PIXELS,
        args.h,
        args.alpha,
        omega_value,
        args.gamma,
        lambda_param=lambda_param,
        gamma_real=args.gamma_real,
        gamma_imag=args.gamma_imag,
        use_tanh=not args.no_tanh,
    )

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is not None:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    elif sweep_idx is not None:
        if sweep_type == "omega":
            sweep_suffix = f"omega{omega_value:.6f}"
        elif sweep_type == "lambda":
            sweep_suffix = f"lambda{lambda_param:.6f}"
        else:
            sweep_suffix = None
        output_dir = make_run_dir(PROJECT_ROOT, TASK, timestamp, sweep_idx, sweep_suffix)
    else:
        output_dir = make_run_dir(PROJECT_ROOT, TASK, timestamp)

    fh_log = open(f"{output_dir}/log.txt", "a")
    fh_log.write("=" * 60 + "\n")
    fh_log.write("sMNIST pixel reconstruction training\n")
    fh_log.write("=" * 60 + "\n")
    for key, value in sorted(vars(args).items()):
        fh_log.write(f"{key}: {value}\n")
    fh_log.write(f"omega: {omega_value:.6f}\n")
    fh_log.write(f"lambda: {lambda_param:.6f}\n")
    fh_log.write(f"num_output: {NUM_PIXELS}\n")
    fh_log.write(
        "early/late pixel definition: early = first 196 scan indices "
        "(top 7 rows of 28x28); late = last 196 (bottom 7 rows)\n"
    )
    fh_log.write("=" * 60 + "\n")
    fh_log.flush()

    return model, loss_fn, optimizer, fh_log, output_dir


def run_epoch_loss(data_loader, model, loss_fn, shuffle_perm):
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for images, _ in data_loader:
            batch_size_current = images.size(0)
            inputs = prepare_smnist_sequence(images, shuffle_perm)
            targets = flatten_targets(images)
            preds = model(inputs)["output"]
            loss = loss_fn(preds, targets)
            total_loss += loss.item() * batch_size_current
            total_count += batch_size_current

    return total_loss / max(total_count, 1)


def run_training(args, omega_value, lambda_value=None, sweep_idx=None, sweep_type=None):
    batch_size_test = 1000
    train_loader, valid_loader, test_loader = build_dataloaders(
        args.batch_size, batch_size_test
    )

    shuffle_perm = torch.randperm(NUM_PIXELS) if args.shuffle else None
    digit_prototypes = build_digit_prototypes(train_loader)
    decimal_levels = [
        int(x.strip()) for x in args.precision_decimals.split(",") if x.strip() != ""
    ]
    do_precision = (
        not args.no_precision_truncation
        and args.precision_truncation
        and not args.sweep_mode
    )

    model, loss_fn, optimizer, fh_log, output_dir = train_with_params(
        args, omega_value, lambda_value, sweep_idx, sweep_type
    )

    best_val_r2 = float("-inf")
    best_test_r2 = float("-inf")
    best_val_mse = float("inf")
    best_test_mse = float("inf")
    parameters_history = []

    train_losses = []
    val_losses = []
    test_losses = []
    val_r2_scores = []
    test_r2_scores = []
    val_mean_pixel_accs = []
    test_mean_pixel_accs = []

    param_str = f"omega={omega_value:.6f}"
    if lambda_value is not None:
        param_str += f", lambda={lambda_value:.6f}"

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        tqdm.write(f"epoch {epoch} ({param_str})")
        model.train()

        for images, _ in tqdm(train_loader, total=len(train_loader), leave=False):
            inputs = prepare_smnist_sequence(images, shuffle_perm)
            targets = flatten_targets(images)

            optimizer.zero_grad()
            preds = model(inputs)["output"]
            loss = loss_fn(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_loss = run_epoch_loss(train_loader, model, loss_fn, shuffle_perm)
        val_loss = run_epoch_loss(valid_loader, model, loss_fn, shuffle_perm)
        test_loss = run_epoch_loss(test_loader, model, loss_fn, shuffle_perm)

        val_metrics = evaluate_pixel_reconstruction(
            valid_loader,
            model,
            batch_size_test,
            shuffle_perm=shuffle_perm,
            pixel_threshold=args.pixel_threshold,
            digit_prototypes=digit_prototypes,
        )
        test_metrics = evaluate_pixel_reconstruction(
            test_loader,
            model,
            batch_size_test,
            shuffle_perm=shuffle_perm,
            pixel_threshold=args.pixel_threshold,
            digit_prototypes=digit_prototypes,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        val_r2_scores.append(val_metrics["overall_r2"])
        test_r2_scores.append(test_metrics["overall_r2"])
        val_mean_pixel_accs.append(val_metrics["mean_pixel_acc"])
        test_mean_pixel_accs.append(test_metrics["mean_pixel_acc"])

        is_best = val_metrics["overall_r2"] > best_val_r2
        if is_best:
            best_val_r2 = val_metrics["overall_r2"]
            best_test_r2 = test_metrics["overall_r2"]
            best_val_mse = val_metrics["overall_mse"]
            best_test_mse = test_metrics["overall_mse"]

        if not args.sweep_mode:
            model_params = extract_model_parameters(model, args.dynamics)
            param_stats = compute_parameter_statistics(model_params)
            parameters_history.append(
                {"epoch": epoch, "params": model_params, "stats": param_stats}
            )
            with open(f"{output_dir}/parameters.json", "w") as f:
                json.dump(parameters_history, f, indent=2)

        confusion_matrix = None
        if not args.skip_epoch_plots:
            plot_regression_metrics(
                train_losses,
                val_losses,
                test_losses,
                test_r2_scores=test_r2_scores,
                val_r2_scores=val_r2_scores,
                output_dir=output_dir,
            )
            ep_dir = epoch_dir(output_dir, epoch)
            _, confusion_matrix = plot_pixel_reconstruction_epoch(
                output_dir=output_dir,
                ep_dir=ep_dir,
                epoch=epoch,
                train_losses=train_losses,
                val_losses=val_losses,
                test_losses=test_losses,
                val_r2_scores=val_r2_scores,
                test_r2_scores=test_r2_scores,
                val_mean_pixel_accs=val_mean_pixel_accs,
                test_mean_pixel_accs=test_mean_pixel_accs,
                test_metrics=test_metrics,
                parameters_history=parameters_history,
                pixel_threshold=args.pixel_threshold,
                num_examples=args.reconstruction_examples,
                is_last_epoch=(epoch == args.epochs - 1),
            )
            if do_precision:
                _, trunc_summary = run_and_save_precision_truncation(
                    test_loader,
                    model,
                    ep_dir,
                    decimal_levels=decimal_levels,
                    shuffle_perm=shuffle_perm,
                    pixel_threshold=args.pixel_threshold,
                    epoch=epoch,
                    fh_log=None,
                    promote_to_dir=output_dir if epoch == args.epochs - 1 else None,
                )
                coarsest = trunc_summary["labels"][-1]
                fh_log.write(
                    f"precision truncation epoch {epoch}: "
                    f"full_r2={trunc_summary['overall_r2'][0]:.4f}, "
                    f"decimals={coarsest}_r2={trunc_summary['overall_r2'][-1]:.4f}, "
                    f"full_early_acc={trunc_summary['early_pixel_acc'][0]:.4f}, "
                    f"full_late_acc={trunc_summary['late_pixel_acc'][0]:.4f}, "
                    f"decimals={coarsest}_early_acc={trunc_summary['early_pixel_acc'][-1]:.4f}, "
                    f"decimals={coarsest}_late_acc={trunc_summary['late_pixel_acc'][-1]:.4f}\n"
                )
                fh_log.flush()

        metrics_data = {
            "epoch": epoch,
            "train_mse": float(train_loss),
            "val_mse": float(val_loss),
            "test_mse": float(test_loss),
            "val_r2": float(val_metrics["overall_r2"]),
            "test_r2": float(test_metrics["overall_r2"]),
            "val_mean_pixel_acc": float(val_metrics["mean_pixel_acc"]),
            "test_mean_pixel_acc": float(test_metrics["mean_pixel_acc"]),
            "val_digit_acc": float(val_metrics.get("digit_acc", 0.0)),
            "test_digit_acc": float(test_metrics.get("digit_acc", 0.0)),
            "test_per_pixel_mse": test_metrics["per_pixel_mse"].tolist(),
            "test_per_pixel_acc": test_metrics["per_pixel_acc"].tolist(),
            "test_per_pixel_r2": test_metrics["per_pixel_r2"].tolist(),
        }
        metrics_file = f"{output_dir}/metrics.json"
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                all_metrics = json.load(f)
            all_metrics.append(metrics_data)
        else:
            all_metrics = [metrics_data]
        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=2)

        early_acc = float(np.mean(test_metrics["per_pixel_acc"][:196]))
        late_acc = float(np.mean(test_metrics["per_pixel_acc"][-196:]))

        msg = (
            f"epoch {epoch}: train_mse={train_loss:.6f}, val_mse={val_loss:.6f}, test_mse={test_loss:.6f}, "
            f"val_r2={val_metrics['overall_r2']:.4f}, test_r2={test_metrics['overall_r2']:.4f}, "
            f"val_mean_pixel_acc={val_metrics['mean_pixel_acc']:.4f}, "
            f"test_mean_pixel_acc={test_metrics['mean_pixel_acc']:.4f}, "
            f"val_digit_acc={val_metrics.get('digit_acc', 0.0):.2f}, "
            f"test_digit_acc={test_metrics.get('digit_acc', 0.0):.2f}, "
            f"early_pixel_acc={early_acc:.4f}, late_pixel_acc={late_acc:.4f}"
        )
        if is_best:
            msg += " [BEST]"
        fh_log.write(msg + "\n")
        if not args.skip_epoch_plots and confusion_matrix is not None:
            per_class_acc = confusion_matrix.diagonal() / (confusion_matrix.sum(axis=1) + 1e-10)
            per_class_acc_dict = {f"digit_{i}": f"{acc * 100:.2f}%" for i, acc in enumerate(per_class_acc)}
            fh_log.write(f"Reconstructed-digit per-class accuracies: {per_class_acc_dict}\n")
        fh_log.flush()
        tqdm.write(msg)

        if not args.sweep_mode:
            save_training_checkpoint(model, output_dir, is_best=is_best)

    if do_precision and args.skip_epoch_plots:
        tqdm.write("Running final-state precision truncation analysis (best checkpoint)...")
        best_path = os.path.join(output_dir, "best_model.pt")
        last_path = os.path.join(output_dir, "last_model.pt")
        ckpt_path = best_path if os.path.exists(best_path) else last_path
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        run_and_save_precision_truncation(
            test_loader,
            model,
            output_dir,
            decimal_levels=decimal_levels,
            shuffle_perm=shuffle_perm,
            pixel_threshold=args.pixel_threshold,
            fh_log=fh_log,
        )
        tqdm.write(f"Precision truncation results saved under {output_dir}")
    elif do_precision:
        tqdm.write(
            "Per-epoch precision truncation plots saved under epochs/; "
            "final epoch also promoted to run root."
        )
    summary = {
        "best_val_r2": best_val_r2,
        "best_test_r2": best_test_r2,
        "best_val_mse": best_val_mse,
        "best_test_mse": best_test_mse,
    }
    fh_log.write(
        f"best test_r2={best_test_r2:.4f} (val_r2={best_val_r2:.4f}), "
        f"best test_mse={best_test_mse:.6f} (val_mse={best_val_mse:.6f})\n"
    )
    fh_log.flush()
    fh_log.close()

    result = {"omega": omega_value, **summary}
    if lambda_value is not None:
        result["lambda"] = lambda_value
    return result


def main():
    args = parse_args()

    if args.sweep_mode:
        args.skip_epoch_plots = True

    if args.sweep_omega and args.sweep_lambda:
        raise ValueError("Cannot sweep both omega and lambda simultaneously.")

    if args.lambda_param > 0:
        print(
            f"WARNING: lambda_param={args.lambda_param} > 0 can destabilize dynamics over "
            f"{NUM_PIXELS} steps. Use a negative value, e.g. --lambda-param -0.04."
        )

    print(args)
    torch.manual_seed(args.seed)

    if args.sweep_omega:
        if args.omega_min is None or args.omega_max is None:
            raise ValueError("--omega-min and --omega-max are required for --sweep-omega")
        omega_values = np.linspace(args.omega_min, args.omega_max, args.omega_steps)
        sweep_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_results = []
        for sweep_idx, omega_val in enumerate(omega_values):
            result = run_training(args, omega_val, sweep_idx=sweep_idx, sweep_type="omega")
            sweep_results.append(result)
        best = max(sweep_results, key=lambda x: x["best_val_r2"])
        sweep_summary_file = sweep_summary_path(PROJECT_ROOT, TASK, "omega", sweep_timestamp)
        with open(sweep_summary_file, "w") as f:
            f.write("Omega sweep summary (sMNIST pixel reconstruction)\n")
            for result in sweep_results:
                f.write(
                    f"omega={result['omega']:.6f}, val_r2={result['best_val_r2']:.4f}, "
                    f"test_r2={result['best_test_r2']:.4f}\n"
                )
            f.write(
                f"best: omega={best['omega']:.6f}, val_r2={best['best_val_r2']:.4f}, "
                f"test_r2={best['best_test_r2']:.4f}\n"
            )
        print(f"Sweep summary saved to {sweep_summary_file}")
    elif args.sweep_lambda:
        if args.lambda_min is None or args.lambda_max is None:
            raise ValueError("--lambda-min and --lambda-max are required for --sweep-lambda")
        lambda_values = np.linspace(args.lambda_min, args.lambda_max, args.lambda_steps)
        sweep_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_results = []
        for sweep_idx, lambda_val in enumerate(lambda_values):
            result = run_training(
                args,
                args.omega,
                lambda_value=lambda_val,
                sweep_idx=sweep_idx,
                sweep_type="lambda",
            )
            sweep_results.append(result)
        best = max(sweep_results, key=lambda x: x["best_val_r2"])
        sweep_summary_file = sweep_summary_path(PROJECT_ROOT, TASK, "lambda", sweep_timestamp)
        with open(sweep_summary_file, "w") as f:
            f.write("Lambda sweep summary (sMNIST pixel reconstruction)\n")
            for result in sweep_results:
                f.write(
                    f"lambda={result['lambda']:.6f}, val_r2={result['best_val_r2']:.4f}, "
                    f"test_r2={result['best_test_r2']:.4f}\n"
                )
            f.write(
                f"best: lambda={best['lambda']:.6f}, val_r2={best['best_val_r2']:.4f}, "
                f"test_r2={best['best_test_r2']:.4f}\n"
            )
        print(f"Sweep summary saved to {sweep_summary_file}")
    else:
        result = run_training(args, args.omega)
        print(
            f"best test_r2={result['best_test_r2']:.4f} (val_r2={result['best_val_r2']:.4f}), "
            f"best test_mse={result['best_test_mse']:.6f}"
        )


if __name__ == "__main__":
    main()
