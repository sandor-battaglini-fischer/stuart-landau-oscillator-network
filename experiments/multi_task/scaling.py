import argparse
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.plotting_utils.style import apply_style, thesis_red, thesis_blue, ifisc_green, mycmap
apply_style()


def run_mg_scaling(num_hidden_list, base_args, output_base_dir):
    """Run Mackey-Glass scaling experiments."""
    from experiments.mackey_glass.comparison import run_single_training, create_label
    
    print("\n" + "=" * 80)
    print("Running Mackey-Glass scaling experiments")
    print("=" * 80)
    
    args = argparse.Namespace(**base_args)
    args.num_hidden_list = ",".join(map(str, num_hidden_list))
    
    # Set default values for list attributes that generate_param_combinations expects
    if not hasattr(args, 'h_list'):
        args.h_list = None
    if not hasattr(args, 'alpha_list'):
        args.alpha_list = None
    if not hasattr(args, 'omega_list'):
        args.omega_list = None
    if not hasattr(args, 'gamma_list'):
        args.gamma_list = None
    if not hasattr(args, 'lambda_list'):
        args.lambda_list = None
    if not hasattr(args, 'mg_tau_list'):
        args.mg_tau_list = None
    if not hasattr(args, 'horizon_list'):
        args.horizon_list = None
    if not hasattr(args, 'remove_top_n_freqs_list'):
        args.remove_top_n_freqs_list = None
    if not hasattr(args, 'remove_top_n_freqs'):
        args.remove_top_n_freqs = 0
    
    from experiments.mackey_glass.comparison import generate_param_combinations
    param_configs = generate_param_combinations(args)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base_dir) / f"mg_scaling_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for run_idx, param_overrides in enumerate(param_configs):
        print(f"\nRun {run_idx + 1}/{len(param_configs)}: {create_label(param_overrides)}")
        print("-" * 80)
        
        try:
            result = run_single_training(args, param_overrides, run_idx, len(param_configs))
            all_results.append(result)
            print(f"Completed: test R² final = {result['test_r2_scores'][-1]:.6f}")
        except Exception as e:
            print(f"ERROR: Run {run_idx + 1} failed: {e}")
            continue
    
    full_data_file = output_dir / "full_results.json"
    full_data = {
        'timestamp': timestamp,
        'total_runs': len(all_results),
        'config': {
            'series_length': args.series_length,
            'input_length': args.input_length,
            'batch_size': args.batch_size,
            'val_fraction': args.val_fraction,
            'test_fraction': args.test_fraction,
            'mg_tau': args.mg_tau,
            'mg_delta_t': args.mg_delta_t,
            'mg_beta': args.mg_beta,
            'mg_gamma': args.mg_gamma,
            'mg_n': args.mg_n,
            'mg_x0': args.mg_x0,
            'seed': args.seed,
        },
        'results': []
    }
    
    for r in all_results:
        result_data = {
            'parameters': r['param_overrides'],
            'label': create_label(r['param_overrides']),
            'test_r2_scores': [float(x) for x in r['test_r2_scores']],
            'val_r2_scores': [float(x) for x in r['val_r2_scores']],
        }
        full_data['results'].append(result_data)
    
    with open(full_data_file, "w") as f:
        json.dump(full_data, f, indent=2)
    
    print(f"\nMackey-Glass results saved to {full_data_file}")
    return full_data_file


def run_imdb_scaling(num_hidden_list, base_args, output_base_dir):
    """Run IMDB scaling experiments."""
    print("\n" + "=" * 80)
    print("Running IMDB scaling experiments")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base_dir) / f"imdb_scaling_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_by_n = {}
    cwd = Path.cwd()
    
    for n in num_hidden_list:
        print(f"\nRunning IMDB with N={n}")
        print("-" * 80)
        
        cmd = [
            sys.executable, "training/train_imdb.py",
            "--num-hidden", str(n),
            "--epochs", str(base_args.get("epochs", 50)),
            "--batch-size", str(base_args.get("batch_size", 64)),
            "--seed", str(base_args.get("seed", 1)),
            "--lr", str(base_args.get("lr", 1e-3)),
            "--h", str(base_args.get("h", 1.0)),
            "--alpha", str(base_args.get("alpha", 0.04)),
            "--omega", str(base_args.get("omega", 0.035904)),
            "--gamma", str(base_args.get("gamma", 0.01)),
            "--lambda-param", str(base_args.get("lambda_param", -0.05)),
            "--gamma-real", str(base_args.get("gamma_real", -0.1)),
            "--gamma-imag", str(base_args.get("gamma_imag", 0.1)),
            "--dynamics", base_args.get("dynamics", "sl"),
        ]
        
        if "embed_dim" in base_args:
            cmd.extend(["--embed-dim", str(base_args["embed_dim"])])
        if "max_len" in base_args:
            cmd.extend(["--max-len", str(base_args["max_len"])])
        if "dropout" in base_args:
            cmd.extend(["--dropout", str(base_args["dropout"])])
        if "weight_decay" in base_args:
            cmd.extend(["--weight-decay", str(base_args["weight_decay"])])
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Completed N={n}")
            
            run_output_dir = None
            out_dirs = sorted(cwd.glob("results/imdb/*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if out_dirs:
                run_output_dir = out_dirs[0]
            
            if run_output_dir and run_output_dir.exists():
                log_path = run_output_dir / "log.txt"
                metrics_path = run_output_dir / "metrics.json"
                
                if metrics_path.exists():
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)
                    test_acc = np.asarray([float(m.get("test_acc", 0.0)) for m in sorted(metrics, key=lambda x: x.get("epoch", 0))], dtype=float)
                    results_by_n[n] = {"test": test_acc, "output_dir": str(run_output_dir)}
                elif log_path.exists():
                    test_acc = parse_log_file_for_test_acc(log_path)
                    if test_acc is not None:
                        results_by_n[n] = {"test": test_acc, "output_dir": str(run_output_dir)}
        except subprocess.CalledProcessError as e:
            print(f"ERROR: IMDB N={n} failed: {e}")
            continue
    
    return results_by_n


def run_smnist_scaling(num_hidden_list, base_args, output_base_dir):
    """Run sMNIST scaling experiments."""
    print("\n" + "=" * 80)
    print("Running sMNIST scaling experiments")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base_dir) / f"smnist_scaling_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_by_n = {}
    cwd = Path.cwd()
    
    for n in num_hidden_list:
        print(f"\nRunning sMNIST with N={n}")
        print("-" * 80)
        
        cmd = [
            sys.executable, "training/train_smnist.py",
            "--num-hidden", str(n),
            "--epochs", str(base_args.get("epochs", 50)),
            "--batch-size", str(base_args.get("batch_size", 64)),
            "--seed", str(base_args.get("seed", 1)),
            "--lr", str(base_args.get("lr", 1e-2)),
            "--h", str(base_args.get("h", 1.0)),
            "--alpha", str(base_args.get("alpha", 0.04)),
            "--omega", str(base_args.get("omega", 0.224)),
            "--gamma", str(base_args.get("gamma", 0.01)),
            "--lambda-param", str(base_args.get("lambda_param", 0.1)),
            "--gamma-real", str(base_args.get("gamma_real", -0.05)),
            "--gamma-imag", str(base_args.get("gamma_imag", 0.1)),
            "--dynamics", base_args.get("dynamics", "sl"),
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Completed N={n}")
            
            run_output_dir = None
            out_dirs = sorted(cwd.glob("results/smnist/*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if out_dirs:
                run_output_dir = out_dirs[0]
            
            if run_output_dir and run_output_dir.exists():
                log_path = run_output_dir / "log.txt"
                metrics_path = run_output_dir / "metrics.json"
                
                if metrics_path.exists():
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)
                    test_acc = np.asarray([float(m.get("test_acc", 0.0)) for m in sorted(metrics, key=lambda x: x.get("epoch", 0))], dtype=float)
                    results_by_n[n] = {"test": test_acc, "output_dir": str(run_output_dir)}
                elif log_path.exists():
                    test_acc = parse_log_file_for_test_acc(log_path)
                    if test_acc is not None:
                        results_by_n[n] = {"test": test_acc, "output_dir": str(run_output_dir)}
        except subprocess.CalledProcessError as e:
            print(f"ERROR: sMNIST N={n} failed: {e}")
            continue
    
    return results_by_n


def parse_log_file_for_test_acc(log_path: Path):
    """Parse log.txt to extract test accuracy per epoch."""
    if not log_path.exists():
        return None
    
    epochs = []
    test_accuracies = []
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('epoch'):
                parts = line.split(':', 1)
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
                rest_parts = rest.split(',')
                
                if len(rest_parts) < 2:
                    continue
                
                try:
                    test_str = None
                    for part in rest_parts:
                        if 'test:' in part:
                            test_str = part.strip()
                            break
                    
                    if test_str:
                        test_value_str = test_str.split(':', 1)[1].strip().split()[0]
                        test = float(test_value_str)
                        epochs.append(epoch)
                        test_accuracies.append(test)
                except (ValueError, IndexError):
                    continue
    
    if len(test_accuracies) == 0:
        return None
    
    if len(epochs) != len(test_accuracies):
        return None
    
    sorted_data = sorted(zip(epochs, test_accuracies))
    sorted_epochs, sorted_test = zip(*sorted_data)
    
    return np.asarray(sorted_test, dtype=float)


def load_mg_from_results(full_results_path: Path):
    """Load MG results from full_results.json."""
    with open(full_results_path, "r") as f:
        data = json.load(f)
    
    results = data.get("results", [])
    by_n = {}
    
    for r in results:
        params = r.get("parameters", {})
        n = params.get("num_hidden")
        if n is None:
            continue
        
        test_r2 = r.get("test_r2_scores", [])
        if not test_r2:
            continue
        
        best_score = max(test_r2)
        existing = by_n.get(n)
        if existing is None or best_score > existing["best_score"]:
            by_n[n] = {
                "test": np.asarray(test_r2, dtype=float) * 100.0,
                "best_score": best_score,
            }
    
    return by_n


def plot_scaling_curves(curves_by_n, task_name, ylabel, output_path: Path, use_r2=False):
    if not curves_by_n:
        print(f"  No data found for {task_name}, skipping plot.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ns = sorted(curves_by_n.keys())
    n_results = len(ns)
    
    all_values = []
    
    for idx, n in enumerate(ns):
        curve = curves_by_n[n]
        test = np.asarray(curve["test"], dtype=float)
        
        epochs = np.arange(len(test))
        frac = idx / max(1, n_results - 1) if n_results > 1 else 0.5
        color = mycmap(frac)
        
        label_test = rf"$N={n}$"
        ax.plot(epochs, test, label=label_test, color=color, linewidth=2.5)
        all_values.extend(test.tolist())
    
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    
    title_task = {
        "mg": "Mackey-Glass",
        "imdb": "IMDB",
        "smnist": "sMNIST",
    }.get(task_name, task_name)
    
    metric_name = "Test $R^2$" if use_r2 else "Test Accuracy"
    ax.set_title(f"{title_task}: {metric_name} vs epoch for different $N$")
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10, ncol=2)
    
    if all_values:
        if use_r2:
            vmin = min(all_values)
            vmax = max(all_values)
            vmin = min(vmin, 0.0)
            vmax = max(vmax, 100.0)
            if vmax > vmin:
                ax.set_ylim(vmin, vmax)
        else:
            ax.set_ylim(50.0, 100.0)
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Generated: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run scaling experiments for MG, IMDB, and sMNIST with different N values"
    )
    
    parser.add_argument(
        "--num-hidden-list",
        type=str,
        required=True,
        help="Comma-separated list of num_hidden values (e.g., '1,2,4,9,50,128')"
    )
    
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        choices=["mg", "imdb", "smnist"],
        default=["mg", "imdb", "smnist"],
        help="Tasks to run (default: all)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="multi_task_scaling_plots",
        help="Output directory for results and plots"
    )
    
    parser.add_argument("--seed", type=int, default=1)
    
    mg_group = parser.add_argument_group("Mackey-Glass parameters")
    mg_group.add_argument("--mg-epochs", type=int, default=50)
    mg_group.add_argument("--mg-batch-size", type=int, default=64)
    mg_group.add_argument("--mg-lr", type=float, default=1e-2)
    mg_group.add_argument("--mg-h", type=float, default=1.0)
    mg_group.add_argument("--mg-alpha", type=float, default=0.04)
    mg_group.add_argument("--mg-omega", type=float, default=0.15)
    mg_group.add_argument("--mg-gamma", type=float, default=0.01)
    mg_group.add_argument("--mg-lambda-param", type=float, default=-0.1)
    mg_group.add_argument("--mg-gamma-real", type=float, default=-0.1)
    mg_group.add_argument("--mg-gamma-imag", type=float, default=0.0)
    mg_group.add_argument("--mg-dynamics", type=str, default="sl", choices=["dho", "sl"])
    mg_group.add_argument("--mg-series-length", type=int, default=20000)
    mg_group.add_argument("--mg-input-length", type=int, default=100)
    mg_group.add_argument("--mg-horizon", type=int, default=1)
    mg_group.add_argument("--mg-tau", type=float, default=17.0)
    mg_group.add_argument("--mg-lr-decay-power", type=float, default=1.0,
                        help="Power factor for cosine decay (lower = slower decay, default: 1.0)")
    mg_group.add_argument("--mg-min-lr-ratio", type=float, default=0.0,
                        help="Minimum LR as fraction of initial LR (default: 0.0)")
    
    imdb_group = parser.add_argument_group("IMDB parameters")
    imdb_group.add_argument("--imdb-epochs", type=int, default=50)
    imdb_group.add_argument("--imdb-batch-size", type=int, default=64)
    imdb_group.add_argument("--imdb-lr", type=float, default=1e-3)
    imdb_group.add_argument("--imdb-h", type=float, default=1.0)
    imdb_group.add_argument("--imdb-alpha", type=float, default=0.04)
    imdb_group.add_argument("--imdb-omega", type=float, default=0.035904)
    imdb_group.add_argument("--imdb-gamma", type=float, default=0.01)
    imdb_group.add_argument("--imdb-lambda-param", type=float, default=-0.05)
    imdb_group.add_argument("--imdb-gamma-real", type=float, default=-0.1)
    imdb_group.add_argument("--imdb-gamma-imag", type=float, default=0.1)
    imdb_group.add_argument("--imdb-dynamics", type=str, default="sl", choices=["dho", "sl"])
    imdb_group.add_argument("--imdb-embed-dim", type=int, default=100)
    imdb_group.add_argument("--imdb-max-len", type=int, default=175)
    imdb_group.add_argument("--imdb-dropout", type=float, default=0.3)
    imdb_group.add_argument("--imdb-weight-decay", type=float, default=0.05)
    
    smnist_group = parser.add_argument_group("sMNIST parameters")
    smnist_group.add_argument("--smnist-epochs", type=int, default=50)
    smnist_group.add_argument("--smnist-batch-size", type=int, default=64)
    smnist_group.add_argument("--smnist-lr", type=float, default=1e-2)
    smnist_group.add_argument("--smnist-h", type=float, default=1.0)
    smnist_group.add_argument("--smnist-alpha", type=float, default=0.04)
    smnist_group.add_argument("--smnist-omega", type=float, default=0.224)
    smnist_group.add_argument("--smnist-gamma", type=float, default=0.01)
    smnist_group.add_argument("--smnist-lambda-param", type=float, default=0.1)
    smnist_group.add_argument("--smnist-gamma-real", type=float, default=-0.05)
    smnist_group.add_argument("--smnist-gamma-imag", type=float, default=0.1)
    smnist_group.add_argument("--smnist-dynamics", type=str, default="sl", choices=["dho", "sl"])
    
    args = parser.parse_args()
    
    num_hidden_list = [int(x.strip()) for x in args.num_hidden_list.split(",")]
    print(f"Running scaling experiments for N values: {num_hidden_list}")
    print(f"Tasks: {args.tasks}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mg_results_file = None
    imdb_results = {}
    smnist_results = {}
    
    if "mg" in args.tasks:
        mg_base_args = {
            "epochs": args.mg_epochs,
            "batch_size": args.mg_batch_size,
            "seed": args.seed,
            "lr": args.mg_lr,
            "h": args.mg_h,
            "alpha": args.mg_alpha,
            "omega": args.mg_omega,
            "gamma": args.mg_gamma,
            "lambda_param": args.mg_lambda_param,
            "gamma_real": args.mg_gamma_real,
            "gamma_imag": args.mg_gamma_imag,
            "dynamics": args.mg_dynamics,
            "series_length": args.mg_series_length,
            "input_length": args.mg_input_length,
            "horizon": args.mg_horizon,
            "mg_tau": args.mg_tau,
            "mg_delta_t": 1.0,
            "mg_beta": 0.2,
            "mg_gamma": 0.1,
            "mg_n": 10.0,
            "mg_x0": 1.2,
            "val_fraction": 0.1,
            "test_fraction": 0.1,
            "lr_decay_power": args.mg_lr_decay_power,
            "min_lr_ratio": args.mg_min_lr_ratio,
        }
        mg_results_file = run_mg_scaling(num_hidden_list, mg_base_args, output_dir)
    
    if "imdb" in args.tasks:
        imdb_base_args = {
            "epochs": args.imdb_epochs,
            "batch_size": args.imdb_batch_size,
            "seed": args.seed,
            "lr": args.imdb_lr,
            "h": args.imdb_h,
            "alpha": args.imdb_alpha,
            "omega": args.imdb_omega,
            "gamma": args.imdb_gamma,
            "lambda_param": args.imdb_lambda_param,
            "gamma_real": args.imdb_gamma_real,
            "gamma_imag": args.imdb_gamma_imag,
            "dynamics": args.imdb_dynamics,
            "embed_dim": args.imdb_embed_dim,
            "max_len": args.imdb_max_len,
            "dropout": args.imdb_dropout,
            "weight_decay": args.imdb_weight_decay,
        }
        imdb_results = run_imdb_scaling(num_hidden_list, imdb_base_args, output_dir)
    
    if "smnist" in args.tasks:
        smnist_base_args = {
            "epochs": args.smnist_epochs,
            "batch_size": args.smnist_batch_size,
            "seed": args.seed,
            "lr": args.smnist_lr,
            "h": args.smnist_h,
            "alpha": args.smnist_alpha,
            "omega": args.smnist_omega,
            "gamma": args.smnist_gamma,
            "lambda_param": args.smnist_lambda_param,
            "gamma_real": args.smnist_gamma_real,
            "gamma_imag": args.smnist_gamma_imag,
            "dynamics": args.smnist_dynamics,
        }
        smnist_results = run_smnist_scaling(num_hidden_list, smnist_base_args, output_dir)
    
    print("\n" + "=" * 80)
    print("Generating scaling plots...")
    print("=" * 80)
    
    if mg_results_file:
        mg_curves = load_mg_from_results(Path(mg_results_file))
        if mg_curves:
            plot_scaling_curves(
                mg_curves,
                task_name="mg",
                ylabel=r"Test $R^2$ (\%)",
                output_path=output_dir / "scaling_mg_num_hidden.png",
                use_r2=True,
            )
    
    if imdb_results:
        plot_scaling_curves(
            imdb_results,
            task_name="imdb",
            ylabel=r"Test Accuracy (\%)",
            output_path=output_dir / "scaling_imdb_num_hidden.png",
            use_r2=False,
        )
    
    if smnist_results:
        plot_scaling_curves(
            smnist_results,
            task_name="smnist",
            ylabel=r"Test Accuracy (\%)",
            output_path=output_dir / "scaling_smnist_num_hidden.png",
            use_r2=False,
        )
    
    print(f"\nAll results and plots saved to {output_dir}")


if __name__ == "__main__":
    main()
