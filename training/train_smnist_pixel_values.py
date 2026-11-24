import argparse
import json
import os
import sys

import numpy as np
import torch
import torchvision
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TASK = "smnist_pixel_values"
NUM_PIXELS = 784

from utils.model_factory import build_oscillator
from utils.plotting_utils import plot_regression_metrics
from utils.plotting_utils.smnist_pixel_reconstruction import (
    build_digit_prototypes,
    evaluate_pixel_reconstruction,
    flatten_targets,
    plot_pixel_reconstruction_epoch,
    prepare_smnist_sequence,
    run_and_save_precision_truncation,
)
from utils.run_dirs import (
    append_metrics, epoch_dir, make_run_dir, save_initial_checkpoint, save_training_checkpoint, write_log_header,
)
from utils.slon_analysis import compute_parameter_statistics, extract_model_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="SLON training on sMNIST full-image pixel reconstruction")
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
    parser.add_argument("--skip-epoch-plots", action="store_true")
    parser.add_argument("--sweep-mode", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--pixel-threshold", type=float, default=0.1,
                        help="absolute error below which a pixel counts as correct")
    parser.add_argument("--reconstruction-examples", type=int, default=8)
    parser.add_argument("--no-precision-truncation", action="store_true",
                        help="skip the final-state decimal truncation analysis")
    parser.add_argument("--precision-decimals", type=str, default="8,6,4,3,2,1,0",
                        help="decimal places to keep in the final state before readout")
    return parser.parse_args()


def build_dataloaders(batch_size_train, batch_size_test):
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root=DATA_DIR, train=True, transform=transform, download=True)
    test_set = torchvision.datasets.MNIST(root=DATA_DIR, train=False, transform=transform, download=True)
    train_set, valid_set = torch.utils.data.random_split(train_set, [len(train_set) - 1000, 1000])
    return (
        torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True),
        torch.utils.data.DataLoader(valid_set, batch_size=batch_size_test, shuffle=False),
        torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False),
    )


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


def run_training(args):
    batch_size_test = 1000
    train_loader, valid_loader, test_loader = build_dataloaders(args.batch_size, batch_size_test)

    shuffle_perm = torch.randperm(NUM_PIXELS) if args.shuffle else None
    digit_prototypes = build_digit_prototypes(train_loader)
    decimal_levels = [int(x) for x in args.precision_decimals.split(",") if x.strip()]
    do_precision = not args.no_precision_truncation and not args.sweep_mode

    model = build_oscillator(
        args.dynamics, 1, args.num_hidden, NUM_PIXELS, args.h, args.alpha, args.omega, args.gamma,
        lambda_param=args.lambda_param, gamma_real=args.gamma_real, gamma_imag=args.gamma_imag,
        use_tanh=not args.no_tanh,
    )
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    output_dir = make_run_dir(PROJECT_ROOT, TASK, args.output_dir)
    if not args.sweep_mode:
        save_initial_checkpoint(model, output_dir)
    fh_log = open(os.path.join(output_dir, "log.txt"), "a")
    write_log_header(fh_log, {**dict(sorted(vars(args).items())), "num_output": NUM_PIXELS})

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

    for epoch in tqdm(range(args.epochs), total=args.epochs):
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

        append_metrics(os.path.join(output_dir, "metrics.json"), {
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
        })

        # early = first 196 scan indices (top 7 rows), late = last 196 (bottom 7 rows)
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
        best_path = os.path.join(output_dir, "best_model.pt")
        ckpt_path = best_path if os.path.exists(best_path) else os.path.join(output_dir, "last_model.pt")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        run_and_save_precision_truncation(
            test_loader, model, output_dir, decimal_levels=decimal_levels, shuffle_perm=shuffle_perm,
            pixel_threshold=args.pixel_threshold, fh_log=fh_log,
        )
    fh_log.write(
        f"best test_r2={best_test_r2:.4f} (val_r2={best_val_r2:.4f}), "
        f"best test_mse={best_test_mse:.6f} (val_mse={best_val_mse:.6f})\n"
    )
    fh_log.close()
    return best_test_r2, best_val_r2, best_test_mse


def main():
    args = parse_args()
    if args.sweep_mode:
        args.skip_epoch_plots = True
    if args.lambda_param > 0:
        print(f"WARNING: lambda_param={args.lambda_param} > 0 can make the dynamics diverge over {NUM_PIXELS} steps")
    print(args)
    torch.manual_seed(args.seed)
    test_r2, val_r2, test_mse = run_training(args)
    print(f"best test_r2={test_r2:.4f} (val_r2={val_r2:.4f}), best test_mse={test_mse:.6f}")


if __name__ == "__main__":
    main()
