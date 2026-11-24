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
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TASK = 'smnist'

from utils.device import resolve_device, seed_all
from utils.manifold_dimension_analysis import analyze_manifold_dimension, run_final_manifold_analysis
from utils.model_factory import build_oscillator, manifold_dynamics_type
from utils.plotting_utils import (
    create_classification_gifs,
    find_class_examples_batch,
    plot_classification_epoch,
    plot_signal_stages_for_example,
    plot_smnist_digit_encoding_analysis,
)
from utils.run_dirs import (
    append_metrics, epoch_dir, make_run_dir, save_initial_checkpoint, save_training_checkpoint, write_log_header,
)
from utils.slon_analysis import compute_parameter_statistics, extract_model_parameters


def build_parser():
    p = argparse.ArgumentParser(description='SLON training on sequential MNIST')
    p.add_argument('--num-hidden', type=int, default=50)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--early-stop-patience', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--train-fraction', type=float, default=1.0,
                   help='class-balanced fraction of the training split to use')
    p.add_argument('--log-train-acc', action='store_true', help='evaluate train accuracy even in --sweep-mode')
    p.add_argument('--analyze-manifold-final', action='store_true',
                   help='lightweight manifold analysis at the end even in --sweep-mode')
    p.add_argument('--shuffle', action='store_true', help='fixed random permutation of the 784 pixels (psMNIST)')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--h', type=float, default=1.0)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--omega', type=float, default=0.224, help='default ~ 2*pi/28, one period per image row')
    p.add_argument('--gamma', type=float, default=0.01)
    p.add_argument('--lambda-param', type=float, default=0.1)
    p.add_argument('--gamma-real', type=float, default=-0.1)
    p.add_argument('--gamma-imag', type=float, default=-0.1)
    p.add_argument('--dynamics', type=str, default='sl', choices=['sl', 'lo', 'dho'])
    p.add_argument('--no-tanh', action='store_true')
    p.add_argument('--analyze-manifold', action='store_true', default=True)
    p.add_argument('--skip-epoch-plots', action='store_true')
    p.add_argument('--sweep-mode', action='store_true', help='only write log.txt and metrics.json')
    p.add_argument('--output-dir', type=str, default=None)
    p.add_argument('--digit-analysis-examples', type=int, default=20)
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    return p


def class_balanced_subsample(dataset, fraction, seed, n_classes=10):
    if fraction >= 1.0:
        return dataset
    base = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    labels = np.asarray(base.targets)
    if isinstance(dataset, torch.utils.data.Subset):
        labels = labels[np.asarray(dataset.indices)]
    rng = np.random.RandomState(seed)
    n_keep = max(n_classes, int(round(fraction * len(dataset))))
    per_class, remainder = divmod(n_keep, n_classes)
    chosen = []
    for c in range(n_classes):
        class_idx = np.flatnonzero(labels == c)
        rng.shuffle(class_idx)
        chosen.extend(class_idx[:min(per_class + (c < remainder), len(class_idx))].tolist())
    rng.shuffle(chosen)
    return torch.utils.data.Subset(dataset, chosen)


def main():
    args = build_parser().parse_args()
    if args.sweep_mode:
        args.skip_epoch_plots = True
    if not 0.0 < args.train_fraction <= 1.0:
        raise ValueError(f'--train-fraction must be in (0, 1], got {args.train_fraction}')
    if args.lambda_param > 0:
        print(f'WARNING: lambda_param={args.lambda_param} > 0 can make the dynamics diverge over 784 steps')

    print(args)
    device = resolve_device(args.device)
    seed_all(args.seed, device)
    print(f'Using device: {device}')

    perm = torch.randperm(784) if args.shuffle else None
    transform = torchvision.transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root=DATA_DIR, train=True, transform=transform, download=True)
    test_set = torchvision.datasets.MNIST(root=DATA_DIR, train=False, transform=transform, download=True)
    train_set, valid_set = torch.utils.data.random_split(train_set, [len(train_set) - 1000, 1000])
    train_set = class_balanced_subsample(train_set, args.train_fraction, args.seed)
    print(f'train examples: {len(train_set)}')

    batch_size_test = 1000
    pin = device.type == 'cuda'
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size_test, shuffle=False, pin_memory=pin)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False, pin_memory=pin)

    model = build_oscillator(
        args.dynamics, 1, args.num_hidden, 10, args.h, args.alpha, args.omega, args.gamma,
        lambda_param=args.lambda_param, gamma_real=args.gamma_real, gamma_imag=args.gamma_imag,
        use_tanh=not args.no_tanh,
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    output_dir = make_run_dir(PROJECT_ROOT, TASK, args.output_dir)
    if not args.sweep_mode:
        save_initial_checkpoint(model, output_dir)
    fh_log = open(os.path.join(output_dir, 'log.txt'), 'a')
    write_log_header(fh_log, {
        'dynamics': args.dynamics, 'use_tanh': not args.no_tanh, 'num_hidden': args.num_hidden,
        'epochs': args.epochs, 'early_stop_patience': args.early_stop_patience, 'batch_size': args.batch_size,
        'train_fraction': args.train_fraction, 'n_train': len(train_set), 'n_valid': len(valid_set),
        'n_test': len(test_set), 'shuffle': args.shuffle, 'seed': args.seed, 'lr': args.lr, 'h': args.h,
        'alpha': args.alpha, 'omega': f'{args.omega:.6f}', 'gamma': args.gamma,
        'lambda': f'{args.lambda_param:.6f}', 'gamma_real': args.gamma_real, 'gamma_imag': args.gamma_imag,
    })

    def log(msg):
        tqdm.write(msg)
        fh_log.write(msg + '\n')
        fh_log.flush()

    def to_sequence(images):
        seq = images.reshape(-1, 1, 784).permute(2, 0, 1).to(device, non_blocking=True)
        return seq[perm] if perm is not None else seq

    def evaluate(loader, return_predictions=False):
        model.eval()
        correct, loss_sum, n = 0, 0.0, 0
        preds, labels_all = [], []
        with torch.no_grad():
            for images, labels in loader:
                labels = labels.to(device, non_blocking=True)
                logits = model(to_sequence(images))['output']
                loss_sum += loss_fn(logits, labels).item() * len(labels)
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                n += len(labels)
                if return_predictions:
                    preds.append(pred.cpu().numpy())
                    labels_all.append(labels.cpu().numpy())
        accuracy, avg_loss = 100.0 * correct / n, loss_sum / n
        if return_predictions:
            return accuracy, avg_loss, np.concatenate(preds), np.concatenate(labels_all)
        return accuracy, avg_loss

    if not args.skip_epoch_plots:
        example_images, example_labels = next(iter(test_loader))
        example_label = int(example_labels[0].item())
        example_inputs = example_images[0:1].reshape(1, 1, 784).permute(2, 0, 1)
        digit_batches = find_class_examples_batch(test_loader, labels=tuple(range(10)),
                                                  num_per_class=args.digit_analysis_examples)

    best_val, best_test, best_epoch, patience_counter = 0.0, 0.0, -1, 0
    parameters_history = []
    history = {k: [] for k in ('train_acc', 'val_acc', 'test_acc', 'train_loss', 'val_loss', 'test_loss')}

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        model.train()
        loss_sum, n = 0.0, 0
        for images, labels in tqdm(train_loader, total=len(train_loader)):
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            train_loss = loss_fn(model(to_sequence(images))['output'], labels)
            train_loss.backward()
            optimizer.step()
            loss_sum += train_loss.item() * len(labels)
            n += len(labels)
        train_loss = loss_sum / max(n, 1)

        if args.sweep_mode:
            train_acc = evaluate(train_loader)[0] if args.log_train_acc else float('nan')
            val_acc, _ = evaluate(valid_loader)
            test_acc, _ = evaluate(test_loader)
            val_loss = test_loss = float('nan')
            test_preds = test_labels = None
        else:
            # outside sweep mode, report the eval-mode train loss rather than the running average
            train_acc, train_loss = evaluate(train_loader)
            val_acc, val_loss = evaluate(valid_loader)
            test_acc, test_loss, test_preds, test_labels = evaluate(test_loader, return_predictions=True)

        is_best = val_acc > best_val
        if is_best:
            best_val, best_test, best_epoch, patience_counter = val_acc, test_acc, epoch, 0
        else:
            patience_counter += 1

        for key, value in zip(history, (train_acc, val_acc, test_acc, train_loss, val_loss, test_loss)):
            history[key].append(value)

        if not args.sweep_mode:
            model_params = extract_model_parameters(model, args.dynamics)
            parameters_history.append({'epoch': epoch, 'params': model_params,
                                       'stats': compute_parameter_statistics(model_params)})
            with open(os.path.join(output_dir, 'parameters.json'), 'w') as f:
                json.dump(parameters_history, f, indent=2)

        if not args.skip_epoch_plots:
            ep_dir = epoch_dir(output_dir, epoch)
            plot_epoch(args, model, output_dir, ep_dir, epoch, history, test_labels, test_preds, parameters_history,
                       patience_counter, fh_log, example_inputs, example_label, digit_batches)

        append_metrics(os.path.join(output_dir, 'metrics.json'), {
            'epoch': epoch, 'train_acc': float(train_acc), 'val_acc': float(val_acc), 'test_acc': float(test_acc),
            'train_loss': float(train_loss), 'val_loss': float(val_loss), 'test_loss': float(test_loss),
        })
        log(f'epoch {epoch}: train: {train_acc:.2f}, val: {val_acc:.2f}, test: {test_acc:.2f}, '
            f'train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, test_loss: {test_loss:.4f}'
            + (' [BEST]' if is_best else ''))

        if args.analyze_manifold and not args.skip_epoch_plots:
            try:
                analyze_manifold_dimension(test_loader, model, manifold_dynamics_type(args.dynamics), ep_dir,
                                           epoch=epoch, batch_size_test=batch_size_test, max_samples=5000,
                                           variance_threshold=0.95)
            except Exception as e:
                tqdm.write(f'Warning: manifold analysis failed at epoch {epoch}: {e}')

        if not args.sweep_mode:
            save_training_checkpoint(model, output_dir, is_best=is_best)

        if patience_counter >= args.early_stop_patience:
            log(f'Early stopping at epoch {epoch}. Best validation: {best_val:.4f} at epoch {best_epoch}')
            break

    log(f'best test: {best_test:.2f} (at epoch {best_epoch}, val: {best_val:.2f})')

    if not args.skip_epoch_plots:
        if parameters_history:
            create_classification_gifs(output_dir)
        if args.analyze_manifold:
            run_final_manifold_analysis(test_loader, model, args.dynamics, output_dir, batch_size_test, fh_log)
    elif args.analyze_manifold_final:
        save_training_checkpoint(model, output_dir, is_best=False)
        run_final_manifold_analysis(test_loader, model, args.dynamics, output_dir, batch_size_test, fh_log,
                                    max_samples=5000, lightweight=True, save_states=False)
    fh_log.close()


def plot_epoch(args, model, output_dir, ep_dir, epoch, history, test_labels, test_preds, parameters_history,
               patience_counter, fh_log, example_inputs, example_label, digit_batches):
    n_units = min(5, args.num_hidden)
    try:
        cm = plot_classification_epoch(
            output_dir=output_dir, ep_dir=ep_dir, epoch=epoch,
            train_accs=history['train_acc'], val_accs=history['val_acc'], test_accs=history['test_acc'],
            train_losses=history['train_loss'], val_losses=history['val_loss'], test_losses=history['test_loss'],
            test_labels=test_labels, test_preds=test_preds, parameters_history=parameters_history, num_classes=10,
            is_last_epoch=(epoch == args.epochs - 1 or patience_counter >= args.early_stop_patience),
        )
        if cm is not None:
            per_class = cm.diagonal() / (cm.sum(axis=1) + 1e-10)
            per_class = {f'digit_{i}': f'{a * 100:.2f}%' for i, a in enumerate(per_class)}
            fh_log.write(f'Per-class accuracies: {per_class}\n')
    except Exception as e:
        tqdm.write(f'Warning: classification plots failed at epoch {epoch}: {e}')

    try:
        plot_signal_stages_for_example(
            model, example_inputs, ep_dir, epoch, input_mode='scalar', task_type='classification',
            class_labels=[str(i) for i in range(10)], true_label=example_label, num_units_plot=n_units,
            raw_input_label='pixel value', title_suffix=f'example digit: {example_label}',
        )
    except Exception as e:
        tqdm.write(f'Warning: signal stage plots failed at epoch {epoch}: {e}')

    try:
        summary = plot_smnist_digit_encoding_analysis(model, digit_batches, ep_dir, epoch,
                                                      num_units_plot=n_units, num_classes=10)
        if not summary.get('dynamics_finite', True):
            tqdm.write(f'Warning: non-finite dynamics at epoch {epoch}, check that lambda_param is negative')
        fh_log.write(
            f"Digit encoding epoch {epoch}: dominant={summary.get('dominant_encoding', 'n/a')}, "
            f"magnitude_score={summary.get('magnitude_separation_score', 0):.3f}, "
            f"phase_score={summary.get('phase_separation_score', 0):.3f}, "
            f"frequency_score={summary.get('frequency_separation_score', 0):.3f}\n"
        )
        nodes = summary.get('node_activity', {}).get('z_magnitude', {})
        if nodes:
            fh_log.write(f"  top active |z| nodes: {nodes.get('top_active_nodes', [])}, "
                         f"top discriminative |z| nodes: {nodes.get('top_discriminative_nodes', [])}\n")
    except Exception as e:
        tqdm.write(f'Warning: digit encoding analysis failed at epoch {epoch}: {e}')
    fh_log.flush()


if __name__ == '__main__':
    main()
