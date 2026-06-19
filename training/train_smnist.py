# sMNIST train script for SLON

import os
import argparse
from tqdm import tqdm
import torch
import torchvision
from datetime import datetime
import numpy as np
import json

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TASK = 'smnist'

from utils.run_dirs import make_run_dir, sweep_summary_path, epoch_dir, save_training_checkpoint
from utils.slon_analysis import extract_model_parameters, compute_parameter_statistics
from utils.plotting_utils import plot_classification_epoch, create_classification_gifs
from utils.manifold_dimension_analysis import analyze_manifold_dimension, collect_and_save_final_states

# command line arguments
parser = argparse.ArgumentParser(description='SLON training script for sMNIST')
parser.add_argument('--num-hidden', type=int, default=50, help='number of units')
parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--shuffle', action = 'store_true', help='whether to shuffle stimulus time steps')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--h', type=float, default=1.0, help='microscopic time constant h (default: 1)')
parser.add_argument('--alpha', type=float, default=0.04, help='excitability coefficient alpha')
parser.add_argument('--omega', type=float, default=0.224, help='natural frequency omega') # 2 * pi / 28 for sMNIST
parser.add_argument('--gamma', type=float, default=0.01, help='damping coefficient gamma')
parser.add_argument('--lambda-param', type=float, default=0.1, help='Stuart-Landau: real part of linear coefficient lambda (default: -|gamma|)')
parser.add_argument('--gamma-real', type=float, default=-0.05, help='Stuart-Landau: real part of nonlinear coefficient (default: -0.1)')
parser.add_argument('--gamma-imag', type=float, default=0.1, help='Stuart-Landau: imaginary part of nonlinear coefficient (default: 0.0)')
parser.add_argument('--sweep-omega', action='store_true', help='enable parameter sweep for omega')
parser.add_argument('--omega-min', type=float, default=None, help='minimum omega value for sweep')
parser.add_argument('--omega-max', type=float, default=None, help='maximum omega value for sweep')
parser.add_argument('--omega-steps', type=int, default=10, help='number of steps for omega sweep (default: 10)')
parser.add_argument('--sweep-lambda', action='store_true', help='enable parameter sweep for lambda (Stuart-Landau only)')
parser.add_argument('--lambda-min', type=float, default=None, help='minimum lambda value for sweep')
parser.add_argument('--lambda-max', type=float, default=None, help='maximum lambda value for sweep')
parser.add_argument('--lambda-steps', type=int, default=10, help='number of steps for lambda sweep (default: 10)')
parser.add_argument('--analyze-manifold', action='store_true', default=True,
                    help='Enable manifold dimension analysis (runs at end of training and every 10 epochs)')

args = parser.parse_args()

if args.sweep_omega and args.sweep_lambda:
    raise ValueError("Cannot sweep both omega and lambda simultaneously. Choose one.")

print(args)

from models import SLON
print("Using Stuart-Landau dynamics")

# fix seed
torch.manual_seed(args.seed)

# sMNIST as 1-dim time series
dim_input = 1

# 10 MNIST classes
dim_output = 10

# batch size of the test set
batch_size_train = args.batch_size
batch_size_test = 1000

# to shuffle mnist digits
if args.shuffle:
    perm = torch.randperm(784)

# load dataset
size_validation = 1000 # size of validation dataset
train_set = torchvision.datasets.MNIST(root=DATA_DIR, train=True, transform=torchvision.transforms.ToTensor(), download=False)
test_set = torchvision.datasets.MNIST(root=DATA_DIR, train=False, transform=torchvision.transforms.ToTensor(), download=False)
train_set, valid_set = torch.utils.data.random_split(train_set, [len(train_set) - size_validation, size_validation])

# data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size_train, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=batch_size_test, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size_test, shuffle=False)

def train_with_params(omega_value, lambda_value=None, sweep_idx=None, sweep_type=None):
    lambda_param = lambda_value if lambda_value is not None else args.lambda_param
    
    model = SLON(dim_input, args.num_hidden, dim_output, args.h, args.alpha, omega_value, args.gamma,
                 lambda_param=lambda_param, gamma_real=args.gamma_real, gamma_imag=args.gamma_imag)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if sweep_idx is not None:
        if sweep_type == 'omega':
            sweep_suffix = f'omega{omega_value:.6f}'
        elif sweep_type == 'lambda':
            sweep_suffix = f'lambda{lambda_param:.6f}'
        else:
            sweep_suffix = None
        output_dir = make_run_dir(PROJECT_ROOT, TASK, timestamp, sweep_idx, sweep_suffix)
    else:
        output_dir = make_run_dir(PROJECT_ROOT, TASK, timestamp)

    fh_log = open(f'{output_dir}/log.txt', 'a')
    fh_log.write('='*60 + '\n')
    fh_log.write('Training Parameters\n')
    fh_log.write('='*60 + '\n')
    fh_log.write(f'dynamics: sl (Stuart-Landau)\n')
    fh_log.write(f'num_hidden: {args.num_hidden}\n')
    fh_log.write(f'epochs: {args.epochs}\n')
    fh_log.write(f'batch_size: {args.batch_size}\n')
    fh_log.write(f'shuffle: {args.shuffle}\n')
    fh_log.write(f'seed: {args.seed}\n')
    fh_log.write(f'lr: {args.lr}\n')
    fh_log.write(f'h: {args.h}\n')
    fh_log.write(f'alpha: {args.alpha}\n')
    fh_log.write(f'omega: {omega_value:.6f}\n')
    fh_log.write(f'gamma: {args.gamma}\n')
    fh_log.write(f'lambda: {lambda_param:.6f}\n')
    fh_log.write(f'gamma_real: {args.gamma_real}\n')
    fh_log.write(f'gamma_imag: {args.gamma_imag}\n')
    fh_log.write('='*60 + '\n')
    fh_log.flush()
    
    return model, loss, optimizer, fh_log, output_dir

# run inference on test set
def evaluate_model(data_loader, model, loss, output_dir, epoch = None, batch = None, return_predictions=False):
    model.eval()
    correct = 0
    test_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # loop over batches in data loader
        for i, (images, labels) in enumerate(data_loader):
            # reshape batch
            images = images.reshape(batch_size_test, 1, 784)
            images = images.permute(2, 0, 1)

            if args.shuffle:
                images = images[perm, :, :]

            # run model inference - record true returns dynamics
            output = model(images, record = True)
            prediction = output['output']

            # compute loss + number of correct predictions
            test_loss += loss(prediction, labels).item()
            pred_label = prediction.data.max(1, keepdim=True)[1]
            correct += pred_label.eq(labels.data.view_as(pred_label)).sum()
            
            if return_predictions:
                all_preds.append(pred_label.squeeze().cpu().numpy())
                all_labels.append(labels.cpu().numpy())


    # compute loss and accuracy
    test_loss /= len(data_loader)
    accuracy = 100. * correct / len(data_loader.dataset)
    
    if return_predictions:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        return accuracy.item(), all_preds, all_labels
    else:
        return accuracy.item()



def run_training(omega_value, lambda_value=None, sweep_idx=None, sweep_type=None):
    model, loss_fn, optimizer, fh_log, output_dir = train_with_params(omega_value, lambda_value, sweep_idx, sweep_type)
    
    best_eval = 0.
    final_test_acc = 0.
    parameters_history = []
    
    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []
    
    param_str = f'omega={omega_value:.6f}'
    if lambda_value is not None:
        param_str += f', lambda={lambda_value:.6f}'
    
    for epoch in tqdm(range(args.epochs), total = args.epochs):
        tqdm.write(f'epoch {epoch} ({param_str})')

        # loop over batches for one epoch
        for batch, (images, labels) in tqdm(enumerate(train_loader), total = len(train_loader)):
            # set model into train mode
            model.train()

            # reshape samples
            images = images.reshape(-1, 1, 784)

            # dimensions: time x batch x 1
            images = images.permute(2, 0, 1)

            if args.shuffle:
                # shuffle if requested
                images = images[perm, :, :]

            # zero gradients
            optimizer.zero_grad()

            # predict
            output = model(images)
            prediction = output['output']

            # compute loss
            train_loss = loss_fn(prediction, labels)

            # compute gradients
            train_loss.backward()

            # update parameters
            optimizer.step()

        # compute validation and test accuracy
        model.eval()
        train_correct = 0
        train_total = 0
        train_loss_sum = 0.0
        
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.reshape(-1, 1, 784)
                images = images.permute(2, 0, 1)
                if args.shuffle:
                    images = images[perm, :, :]
                output = model(images)
                prediction = output['output']
                train_loss_batch = loss_fn(prediction, labels)
                train_loss_sum += train_loss_batch.item() * len(labels)
                pred_label = prediction.data.max(1, keepdim=True)[1]
                train_correct += pred_label.eq(labels.data.view_as(pred_label)).sum()
                train_total += len(labels)
        
        train_acc = (100. * train_correct / train_total).item()
        train_loss_avg = train_loss_sum / train_total
        
        valid_acc = evaluate_model(valid_loader, model, loss_fn, output_dir)
        test_acc, test_preds, test_labels = evaluate_model(test_loader, model, loss_fn, output_dir, epoch, batch, return_predictions=True)
        
        model.eval()
        val_loss_sum = 0.0
        val_total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.reshape(batch_size_test, 1, 784)
                images = images.permute(2, 0, 1)
                if args.shuffle:
                    images = images[perm, :, :]
                output = model(images)
                prediction = output['output']
                val_loss_batch = loss_fn(prediction, labels)
                val_loss_sum += val_loss_batch.item() * len(labels)
                val_total += len(labels)
        val_loss_avg = val_loss_sum / val_total
        
        test_loss_sum = 0.0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.reshape(batch_size_test, 1, 784)
                images = images.permute(2, 0, 1)
                if args.shuffle:
                    images = images[perm, :, :]
                output = model(images)
                prediction = output['output']
                test_loss_batch = loss_fn(prediction, labels)
                test_loss_sum += test_loss_batch.item() * len(labels)
                test_total += len(labels)
        test_loss_avg = test_loss_sum / test_total
        
        is_best = valid_acc > best_eval
        if is_best:
            best_eval = valid_acc
            final_test_acc = test_acc

        train_accs.append(train_acc)
        val_accs.append(valid_acc)
        test_accs.append(test_acc)
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        test_losses.append(test_loss_avg)

        model_params = extract_model_parameters(model, 'sl')
        param_stats = compute_parameter_statistics(model_params)
        parameters_history.append({
            "epoch": epoch,
            "params": model_params,
            "stats": param_stats
        })

        params_file = f'{output_dir}/parameters.json'
        with open(params_file, 'w') as f:
            json.dump(parameters_history, f, indent=2)

        ep_dir = epoch_dir(output_dir, epoch)
        try:
            cm = plot_classification_epoch(
                output_dir=output_dir,
                ep_dir=ep_dir,
                epoch=epoch,
                train_accs=train_accs,
                val_accs=val_accs,
                test_accs=test_accs,
                train_losses=train_losses,
                val_losses=val_losses,
                test_losses=test_losses,
                test_labels=test_labels,
                test_preds=test_preds,
                parameters_history=parameters_history,
                num_classes=10,
                is_last_epoch=(epoch == args.epochs - 1),
            )
            if cm is not None:
                per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-10)
                per_class_acc_dict = {f"digit_{i}": f"{acc*100:.2f}%" for i, acc in enumerate(per_class_acc)}
                fh_log.write(f"Per-class accuracies: {per_class_acc_dict}\n")
                fh_log.flush()
        except Exception as e:
            tqdm.write(f"Warning: Failed to generate plots at epoch {epoch}: {e}")

        metrics_file = f"{output_dir}/metrics.json"
        metrics_data = {
            "epoch": epoch,
            "train_acc": float(train_acc),
            "val_acc": float(valid_acc),
            "test_acc": float(test_acc),
            "train_loss": float(train_loss_avg),
            "val_loss": float(val_loss_avg),
            "test_loss": float(test_loss_avg),
        }
        
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                all_metrics = json.load(f)
            all_metrics.append(metrics_data)
        else:
            all_metrics = [metrics_data]
        
        with open(metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=2)

        msg = f'epoch {epoch}: train: {train_acc:.2f}, val: {valid_acc:.2f}, test: {test_acc:.2f}, train_loss: {train_loss_avg:.4f}, val_loss: {val_loss_avg:.4f}, test_loss: {test_loss_avg:.4f}'
        if valid_acc == best_eval:
            msg += ' [BEST]'
        fh_log.write(msg + '\n')
        fh_log.flush()
        tqdm.write(msg)

        if args.analyze_manifold:
            try:
                tqdm.write(f"\nRunning manifold dimension analysis at epoch {epoch}...")
                manifold_results_epoch = analyze_manifold_dimension(
                    test_loader,
                    model,
                    'sl',
                    ep_dir,
                    epoch=epoch,
                    batch_size_test=batch_size_test,
                    max_samples=5000,
                    variance_threshold=0.95
                )
                if manifold_results_epoch is not None:
                    tqdm.write(f"  PCA effective dim: {manifold_results_epoch['effective_dim_pca']}, "
                             f"Correlation dim: {manifold_results_epoch['correlation_dim']:.4f}" 
                             if manifold_results_epoch['correlation_dim'] is not None 
                             else f"  PCA effective dim: {manifold_results_epoch['effective_dim_pca']}")
            except Exception as e:
                tqdm.write(f"Warning: Manifold dimension analysis failed at epoch {epoch}: {e}")

        save_training_checkpoint(model, output_dir, is_best=is_best)
        tqdm.write(f'wrote checkpoint last_model.pt{" + best_model.pt" if is_best else ""}')

    
    msg = f'best test: {final_test_acc:.2f} (val: {best_eval:.2f})'
    fh_log.write(msg + '\n')
    fh_log.flush()
    
    if len(parameters_history) > 0:
        create_classification_gifs(output_dir)
    
    # final manifold dimension analysis
    if args.analyze_manifold:
        print("\n" + "=" * 60)
        print("Computing final manifold dimension analysis...")
        print("=" * 60 + "\n")
        
        try:
            manifold_results = analyze_manifold_dimension(
                test_loader,
                model,
                'sl',
                output_dir,
                epoch=None,
                batch_size_test=batch_size_test,
                max_samples=10000,
                variance_threshold=0.95
            )
            
            if manifold_results is not None:
                manifold_file = f"{output_dir}/manifold_dimension_results.json"
                with open(manifold_file, "w") as f:
                    json.dump(manifold_results, f, indent=2)
                
                fh_log.write("\n" + "=" * 60 + "\n")
                fh_log.write("Manifold Dimension Analysis Results:\n")
                fh_log.write("=" * 60 + "\n")
                fh_log.write(f"PCA effective dimension (95% variance): {manifold_results['effective_dim_pca']}\n")
                fh_log.write(f"Correlation dimension: {manifold_results['correlation_dim']:.4f}\n" if manifold_results['correlation_dim'] is not None else "Correlation dimension: N/A\n")
                fh_log.write(f"State space dimension: {manifold_results['state_dim']}\n")
                fh_log.write(f"Number of samples analyzed: {manifold_results['n_samples']}\n")
                fh_log.write(f"Explained variance at 95% threshold: {manifold_results['explained_variance_95']:.4f}\n")
                fh_log.write("=" * 60 + "\n")
                fh_log.flush()
                
                print(f"Manifold dimension analysis complete!")
                print(f"  PCA effective dimension: {manifold_results['effective_dim_pca']}")
                if manifold_results['correlation_dim'] is not None:
                    print(f"  Correlation dimension: {manifold_results['correlation_dim']:.4f}")
                print(f"  State space dimension: {manifold_results['state_dim']}")
        except Exception as e:
            import traceback
            error_msg = f"Error during manifold dimension analysis: {e}\n{traceback.format_exc()}"
            fh_log.write(f"\n{error_msg}\n")
            fh_log.flush()
            print(f"Error during manifold dimension analysis: {e}")
        
        # Collect and save final states with animation
        try:
            print("\n" + "=" * 60)
            print("Collecting final states and creating animation...")
            print("=" * 60 + "\n")
            collect_and_save_final_states(
                test_loader,
                model,
                'sl',
                output_dir,
                batch_size_test=batch_size_test,
                max_samples=50000,
                is_imdb=False
            )
        except Exception as e:
            import traceback
            print(f"Warning: Failed to collect final states or create animation: {e}")
            fh_log.write(f"\nWarning: Failed to collect final states: {e}\n")
            fh_log.flush()
    
    fh_log.close()
    
    if lambda_value is not None:
        return {'omega': omega_value, 'lambda': lambda_value, 'best_val_acc': best_eval, 'best_test_acc': final_test_acc}
    else:
        return {'omega': omega_value, 'best_val_acc': best_eval, 'best_test_acc': final_test_acc}

if args.sweep_omega:
    if args.omega_min is None or args.omega_max is None:
        raise ValueError("--omega-min and --omega-max must be specified when --sweep-omega is enabled")
    
    omega_values = np.linspace(args.omega_min, args.omega_max, args.omega_steps)
    print(f"Starting omega sweep: {args.omega_steps} steps from {args.omega_min:.6f} to {args.omega_max:.6f}")
    print(f"Omega values: {omega_values}")
    
    sweep_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_results = []
    
    for sweep_idx, omega_val in enumerate(omega_values):
        print(f"\n{'='*60}")
        print(f"Sweep iteration {sweep_idx + 1}/{args.omega_steps}: omega = {omega_val:.6f}")
        print(f"{'='*60}\n")
        
        result = run_training(omega_val, sweep_idx=sweep_idx, sweep_type='omega')
        sweep_results.append(result)
        
        print(f"\nCompleted sweep {sweep_idx + 1}/{args.omega_steps}: omega={result['omega']:.6f}, val={result['best_val_acc']:.4f}, test={result['best_test_acc']:.4f}\n")
    
    sweep_summary_file = sweep_summary_path(PROJECT_ROOT, TASK, 'omega', sweep_timestamp)
    with open(sweep_summary_file, 'w') as f:
        f.write(f"Omega Sweep Summary\n")
        f.write(f"Timestamp: {sweep_timestamp}\n")
        f.write(f"Range: {args.omega_min:.6f} to {args.omega_max:.6f}\n")
        f.write(f"Steps: {args.omega_steps}\n")
        f.write(f"{'='*60}\n")
        f.write(f"{'Omega':<15} {'Best Val Acc':<15} {'Best Test Acc':<15}\n")
        f.write(f"{'-'*60}\n")
        
        best_overall = max(sweep_results, key=lambda x: x['best_val_acc'])
        
        for result in sweep_results:
            marker = " <-- BEST" if result == best_overall else ""
            f.write(f"{result['omega']:<15.6f} {result['best_val_acc']:<15.4f} {result['best_test_acc']:<15.4f}{marker}\n")
        
        f.write(f"{'='*60}\n")
        f.write(f"Best overall: omega={best_overall['omega']:.6f}, val={best_overall['best_val_acc']:.4f}, test={best_overall['best_test_acc']:.4f}\n")
    
    print(f"\n{'='*60}")
    print(f"Sweep complete! Summary saved to {sweep_summary_file}")
    print(f"Best result: omega={best_overall['omega']:.6f}, val={best_overall['best_val_acc']:.4f}, test={best_overall['best_test_acc']:.4f}")
    print(f"{'='*60}\n")
elif args.sweep_lambda:
    if args.lambda_min is None or args.lambda_max is None:
        raise ValueError("--lambda-min and --lambda-max must be specified when --sweep-lambda is enabled")
    
    lambda_values = np.linspace(args.lambda_min, args.lambda_max, args.lambda_steps)
    print(f"Starting lambda sweep: {args.lambda_steps} steps from {args.lambda_min:.6f} to {args.lambda_max:.6f}")
    print(f"Lambda values: {lambda_values}")
    
    sweep_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_results = []
    
    for sweep_idx, lambda_val in enumerate(lambda_values):
        print(f"\n{'='*60}")
        print(f"Sweep iteration {sweep_idx + 1}/{args.lambda_steps}: lambda = {lambda_val:.6f}")
        print(f"{'='*60}\n")
        
        result = run_training(args.omega, lambda_val, sweep_idx=sweep_idx, sweep_type='lambda')
        sweep_results.append(result)
        
        print(f"\nCompleted sweep {sweep_idx + 1}/{args.lambda_steps}: lambda={result['lambda']:.6f}, val={result['best_val_acc']:.4f}, test={result['best_test_acc']:.4f}\n")
    
    sweep_summary_file = sweep_summary_path(PROJECT_ROOT, TASK, 'lambda', sweep_timestamp)
    with open(sweep_summary_file, 'w') as f:
        f.write(f"Lambda Sweep Summary\n")
        f.write(f"Timestamp: {sweep_timestamp}\n")
        f.write(f"Range: {args.lambda_min:.6f} to {args.lambda_max:.6f}\n")
        f.write(f"Steps: {args.lambda_steps}\n")
        f.write(f"{'='*60}\n")
        f.write(f"{'Lambda':<15} {'Best Val Acc':<15} {'Best Test Acc':<15}\n")
        f.write(f"{'-'*60}\n")
        
        best_overall = max(sweep_results, key=lambda x: x['best_val_acc'])
        
        for result in sweep_results:
            marker = " <-- BEST" if result == best_overall else ""
            f.write(f"{result['lambda']:<15.6f} {result['best_val_acc']:<15.4f} {result['best_test_acc']:<15.4f}{marker}\n")
        
        f.write(f"{'='*60}\n")
        f.write(f"Best overall: lambda={best_overall['lambda']:.6f}, val={best_overall['best_val_acc']:.4f}, test={best_overall['best_test_acc']:.4f}\n")
    
    print(f"\n{'='*60}")
    print(f"Sweep complete! Summary saved to {sweep_summary_file}")
    print(f"Best result: lambda={best_overall['lambda']:.6f}, val={best_overall['best_val_acc']:.4f}, test={best_overall['best_test_acc']:.4f}")
    print(f"{'='*60}\n")
else:
    result = run_training(args.omega)
    print(f'best test: {result["best_test_acc"]:.2f} (val: {result["best_val_acc"]:.4f})')
