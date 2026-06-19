# IMDB grid search script for HORN Stuart-Landau dynamics

import os
import argparse
import re
import math
import json
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import urllib.request
import tarfile
import zipfile
import numpy as np
import itertools
from pathlib import Path
import matplotlib.patches as mpatches
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.plotting_utils.style import apply_style, thesis_red, thesis_blue, ifisc_green, mycmap
apply_style()

import matplotlib.colors as mcolors

parser = argparse.ArgumentParser(description='SLON grid search for IMDB')
parser.add_argument('--num-hidden', type=int, default=1, help='number of units')
parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--shuffle', action='store_true', help='whether to shuffle stimulus time steps')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--h', type=float, default=1.0, help='microscopic time constant h')
parser.add_argument('--alpha', type=float, default=0.04, help='excitability coefficient alpha')
parser.add_argument('--gamma', type=float, default=0.01, help='damping coefficient gamma')
parser.add_argument('--embed-dim', type=int, default=100, help='word embedding dimension')
parser.add_argument('--max-len', type=int, default=175, help='maximum sequence length')
parser.add_argument('--min-freq', type=int, default=2, help='minimum word frequency for vocabulary')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--early-stop-patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay for regularization')
parser.add_argument('--glove', type=str, default="glove.6B.100d.txt", help='GloVe embedding file path')
parser.add_argument('--glove-dir', type=str, default='data/glove', help='directory to store GloVe embeddings')
parser.add_argument('--lambda-min', type=float, default=-1.0, help='minimum lambda value')
parser.add_argument('--lambda-max', type=float, default=0.5, help='maximum lambda value')
parser.add_argument('--lambda-steps', type=int, default=16, help='number of lambda values')
parser.add_argument('--gamma-real-min', type=float, default=-0.1, help='minimum gamma_real value')
parser.add_argument('--gamma-real-max', type=float, default=0.2, help='maximum gamma_real value')
parser.add_argument('--gamma-real-steps', type=int, default=16, help='number of gamma_real values')
parser.add_argument('--gamma-imag-min', type=float, default=-0.1, help='minimum gamma_imag value')
parser.add_argument('--gamma-imag-max', type=float, default=0.2, help='maximum gamma_imag value')
parser.add_argument('--gamma-imag-steps', type=int, default=16, help='number of gamma_imag values')
parser.add_argument('--fix-omega', action='store_true', help='fix omega at 2*pi/175 and exclude from grid search')
parser.add_argument('--omega-min', type=float, default=None, help='minimum omega value (if None, uses uniform range)')
parser.add_argument('--omega-max', type=float, default=None, help='maximum omega value (if None, uses uniform range)')
parser.add_argument('--omega-steps', type=int, default=16, help='number of omega values')
parser.add_argument('--omega-multiples', action='store_true', help='use multiples of 2*pi/175 for omega instead of uniform range')
parser.add_argument('--results-dir', type=str, default='grid_search_results', help='directory to save results')
parser.add_argument('--save-interval', type=int, default=1, help='save results every N completed runs')
parser.add_argument('--resume', type=str, default=None, help='resume from saved results JSON file')
parser.add_argument('--plot-only', type=str, default=None, help='only generate plots from existing results JSON file')

args = parser.parse_args()

target_period = args.max_len
base_omega = (2 * math.pi) / (target_period * args.h)
fixed_omega = base_omega

if args.fix_omega:
    omega_values = [fixed_omega]
    print(f"Omega fixed at: {fixed_omega:.6f} (2*pi/{target_period})")
else:
    if args.omega_min is None or args.omega_max is None:
        if args.omega_multiples:
            multipliers = np.linspace(0.3, 2.5, args.omega_steps)
            omega_values = multipliers * base_omega
        else:
            omega_values = np.linspace(0.01, 0.08, args.omega_steps)
    else:
        omega_values = np.linspace(args.omega_min, args.omega_max, args.omega_steps)

lambda_values = np.linspace(args.lambda_min, args.lambda_max, args.lambda_steps)
gamma_real_values = np.linspace(args.gamma_real_min, args.gamma_real_max, args.gamma_real_steps)
gamma_imag_values = np.linspace(args.gamma_imag_min, args.gamma_imag_max, args.gamma_imag_steps)

print(f"Grid search configuration:")
print(f"  Lambda: {args.lambda_steps} values from {args.lambda_min:.3f} to {args.lambda_max:.3f}")
if args.fix_omega:
    print(f"  Omega: FIXED at {fixed_omega:.6f} (excluded from grid search)")
else:
    print(f"  Omega: {args.omega_steps} values from {omega_values[0]:.6f} to {omega_values[-1]:.6f}")
print(f"  Gamma_real: {args.gamma_real_steps} values from {args.gamma_real_min:.3f} to {args.gamma_real_max:.3f}")
print(f"  Gamma_imag: {args.gamma_imag_steps} values from {args.gamma_imag_min:.3f} to {args.gamma_imag_max:.3f}")
if args.fix_omega:
    print(f"  Total combinations: {len(lambda_values) * len(gamma_real_values) * len(gamma_imag_values)}")
else:
    print(f"  Total combinations: {len(lambda_values) * len(omega_values) * len(gamma_real_values) * len(gamma_imag_values)}")

if not args.plot_only:
    from models import SLON
    from training.train_imdb import (
        tokenize, Vocabulary, download_imdb, load_imdb_reviews,
        download_glove, load_glove_vectors, SLONWithEmbedding
    )

    torch.manual_seed(args.seed)

    dim_input = args.embed_dim
    dim_output = 2
    batch_size_train = args.batch_size
    batch_size_test = 1000

    print("Loading IMDB dataset...")
    imdb_path = download_imdb()
    train_texts, train_labels = load_imdb_reviews(imdb_path, 'train')
    test_texts, test_labels = load_imdb_reviews(imdb_path, 'test')

    print("Building vocabulary...")
    vocab = Vocabulary(min_freq=args.min_freq)
    vocab_size = vocab.build(train_texts)
    print(f"Vocabulary size: {vocab_size}")

    pad_idx = vocab.word2idx['<pad>']
    unk_idx = vocab.word2idx['<unk>']

    embedding_weights = None
    if args.glove:
        glove_path = args.glove
        if not os.path.isabs(glove_path):
            glove_path = os.path.join(args.glove_dir, glove_path)
        
        if not os.path.exists(glove_path):
            print(f"GloVe file not found at {glove_path}, downloading...")
            glove_path = download_glove(args.glove_dir, dim=args.embed_dim)
        
        embedding_weights = load_glove_vectors(glove_path, vocab, args.embed_dim)

    def process_data(text, label):
        tokens = tokenize(text)
        token_ids = vocab(tokens)
        
        if len(token_ids) > args.max_len:
            token_ids = token_ids[:args.max_len]
        else:
            token_ids = token_ids + [pad_idx] * (args.max_len - len(token_ids))
        
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    print("Processing datasets...")
    train_data = [process_data(text, label) for text, label in zip(train_texts, train_labels)]
    test_data = [process_data(text, label) for text, label in zip(test_texts, test_labels)]

    size_validation = 5000
    train_data, valid_data = train_data[size_validation:], train_data[:size_validation]

    def collate_batch(batch):
        texts, labels = zip(*batch)
        texts = torch.stack(texts)
        labels = torch.stack(labels)
        return texts, labels

    train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_data, batch_size=batch_size_test, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=False, collate_fn=collate_batch)

    def evaluate_model(data_loader, model, loss):
        model.eval()
        correct = 0
        test_loss = 0

        with torch.no_grad():
            for token_ids, labels in data_loader:
                if args.shuffle:
                    perm = torch.randperm(token_ids.size(1))
                    token_ids = token_ids[:, perm]

                output = model(token_ids)
                prediction = output['output']

                test_loss += loss(prediction, labels).item()
                pred_label = prediction.data.max(1, keepdim=True)[1]
                correct += pred_label.eq(labels.data.view_as(pred_label)).sum()

        test_loss /= len(data_loader)
        accuracy = 100. * correct / len(data_loader.dataset)
        return accuracy.item()

    def run_training(lambda_val, omega_val, gamma_real_val, gamma_imag_val):
        model = SLONWithEmbedding(vocab_size, dim_input, args.num_hidden, dim_output, 
                                  args.h, args.alpha, omega_val, args.gamma, pad_idx, 
                                  dropout=args.dropout, embedding_weights=embedding_weights,
                                  dynamics='sl',
                                  lambda_param=lambda_val, gamma_real=gamma_real_val, gamma_imag=gamma_imag_val)

        loss_fn = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW([
            {'params': model.embedding.parameters(), 'lr': args.lr * 0.5, 'weight_decay': args.weight_decay},
            {'params': model.horn.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}
        ], lr=args.lr, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
        )

        best_eval = 0.
        final_test_acc = 0.
        best_epoch = 0
        patience_counter = 0
        numerical_error = False
        
        for epoch in range(args.epochs):
            epoch_train_loss = 0.0
            model.train()

            try:
                for batch_idx, (token_ids, labels) in enumerate(train_loader):
                    if args.shuffle:
                        perm = torch.randperm(token_ids.size(1))
                        token_ids = token_ids[:, perm]

                    optimizer.zero_grad()

                    output = model(token_ids)
                    prediction = output['output']

                    train_loss = loss_fn(prediction, labels)
                    
                    if not torch.isfinite(train_loss):
                        numerical_error = True
                        break
                    
                    epoch_train_loss += train_loss.item()

                    train_loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                
                if numerical_error:
                    break

                valid_acc = evaluate_model(valid_loader, model, loss_fn)
                test_acc = evaluate_model(test_loader, model, loss_fn)
                
                if valid_acc > best_eval:
                    best_eval = valid_acc
                    final_test_acc = test_acc
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1

                scheduler.step(valid_acc)

                if patience_counter >= args.early_stop_patience:
                    break
                    
            except (RuntimeError, ValueError) as e:
                if 'nan' in str(e).lower() or 'inf' in str(e).lower():
                    numerical_error = True
                    break
                raise
        
        result = {
            'lambda': float(lambda_val),
            'omega': float(omega_val),
            'gamma_real': float(gamma_real_val),
            'gamma_imag': float(gamma_imag_val),
            'best_val_acc': float(best_eval),
            'best_test_acc': float(final_test_acc),
            'best_epoch': int(best_epoch),
            'numerical_error': numerical_error
        }
        
        return result

def generate_heatmaps(results, output_dir, omega_fixed=False, timestamp=None, config=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    valid_results = [r for r in results if not r.get('numerical_error', False)]
    
    if not valid_results and config is None:
        print("No valid results to plot (all had numerical errors)")
        return
    
    if config is not None:
        lambda_range = config.get('lambda_range', None)
        omega_range = config.get('omega_range', None)
        omega_value = config.get('omega_value', None)
        gamma_real_range = config.get('gamma_real_range', None)
        gamma_imag_range = config.get('gamma_imag_range', None)
        
        if lambda_range:
            lambda_vals = np.linspace(lambda_range[0], lambda_range[1], int(lambda_range[2])).tolist()
        else:
            lambda_vals = sorted(set(r['lambda'] for r in valid_results))
        
        if omega_fixed and omega_value is not None:
            omega_vals = [omega_value]
        elif omega_range:
            omega_vals = np.linspace(omega_range[0], omega_range[1], int(omega_range[2])).tolist()
        else:
            omega_vals = sorted(set(r['omega'] for r in valid_results))
        
        if gamma_real_range:
            gamma_real_vals = np.linspace(gamma_real_range[0], gamma_real_range[1], int(gamma_real_range[2])).tolist()
        else:
            gamma_real_vals = sorted(set(r['gamma_real'] for r in valid_results))
        
        if gamma_imag_range:
            gamma_imag_vals = np.linspace(gamma_imag_range[0], gamma_imag_range[1], int(gamma_imag_range[2])).tolist()
        else:
            gamma_imag_vals = sorted(set(r['gamma_imag'] for r in valid_results))
    else:
        lambda_vals = sorted(set(r['lambda'] for r in valid_results))
        omega_vals = sorted(set(r['omega'] for r in valid_results))
        gamma_real_vals = sorted(set(r['gamma_real'] for r in valid_results))
        gamma_imag_vals = sorted(set(r['gamma_imag'] for r in valid_results))
    
    if omega_fixed or len(omega_vals) == 1:
        param_pairs = [
            ('lambda', 'gamma_real', lambda_vals, gamma_real_vals, 'gamma_imag', None, gamma_imag_vals, None),
            ('lambda', 'gamma_imag', lambda_vals, gamma_imag_vals, 'gamma_real', None, gamma_real_vals, None),
            ('gamma_real', 'gamma_imag', gamma_real_vals, gamma_imag_vals, 'lambda', None, lambda_vals, None),
        ]
    else:
        param_pairs = [
            ('lambda', 'omega', lambda_vals, omega_vals, 'gamma_real', 'gamma_imag', gamma_real_vals, gamma_imag_vals),
            ('lambda', 'gamma_real', lambda_vals, gamma_real_vals, 'omega', 'gamma_imag', omega_vals, gamma_imag_vals),
            ('lambda', 'gamma_imag', lambda_vals, gamma_imag_vals, 'omega', 'gamma_real', omega_vals, gamma_real_vals),
            ('omega', 'gamma_real', omega_vals, gamma_real_vals, 'lambda', 'gamma_imag', lambda_vals, gamma_imag_vals),
            ('omega', 'gamma_imag', omega_vals, gamma_imag_vals, 'lambda', 'gamma_real', lambda_vals, gamma_real_vals),
            ('gamma_real', 'gamma_imag', gamma_real_vals, gamma_imag_vals, 'lambda', 'omega', lambda_vals, omega_vals),
        ]
    
    for param1_name, param2_name, param1_vals, param2_vals, avg_param1_name, avg_param2_name, avg_param1_vals, avg_param2_vals in param_pairs:
        def find_closest_idx(val, val_list, tol=1e-6):
            for idx, v in enumerate(val_list):
                if abs(v - val) < tol:
                    return idx
            return None

        def format_val(v, param_name):
            if param_name == 'omega':
                return f'{v:.4f}'
            else:
                return f'{v:.3f}'

        def build_and_save_heatmap(current_results, suffix='', title_suffix=''):
            heatmap_data = np.full((len(param2_vals), len(param1_vals)), np.nan)
            count_data = np.zeros((len(param2_vals), len(param1_vals)), dtype=int)
            max_data = np.full((len(param2_vals), len(param1_vals)), np.nan)

            for r in current_results:
                param1_val = r[param1_name]
                param2_val = r[param2_name]

                idx1 = find_closest_idx(param1_val, param1_vals)
                idx2 = find_closest_idx(param2_val, param2_vals)

                if idx1 is not None and idx2 is not None:
                    acc = 0.0 if r.get('numerical_error', False) else r['best_test_acc']

                    if np.isnan(heatmap_data[idx2, idx1]):
                        heatmap_data[idx2, idx1] = acc
                        max_data[idx2, idx1] = acc
                        count_data[idx2, idx1] = 1
                    else:
                        heatmap_data[idx2, idx1] += acc
                        max_data[idx2, idx1] = max(max_data[idx2, idx1], acc)
                        count_data[idx2, idx1] += 1

            heatmap_data_local = np.divide(heatmap_data, count_data, out=np.full_like(heatmap_data, np.nan), where=count_data!=0)

            missing_mask = np.isnan(heatmap_data_local)

            zero_mask = (heatmap_data_local == 0) & (~missing_mask)
            below_50_mask = (heatmap_data_local > 0) & (heatmap_data_local < 50) & (~missing_mask)

            valid_data = heatmap_data_local[~missing_mask]

            if len(valid_data) > 0:
                vmax = max(100.0, np.max(valid_data))
            else:
                vmax = 100.0

            heatmap_data_normalized = heatmap_data_local.copy()
            heatmap_data_normalized[below_50_mask] = 49.9
            heatmap_data_normalized[zero_mask] = -1
            heatmap_data_normalized[missing_mask] = -2

            vmin_norm = -2
            vmax_norm = vmax

            def normalize_value(val):
                return (val - vmin_norm) / (vmax_norm - vmin_norm)

            missing_pos = normalize_value(-2)
            zero_pos = normalize_value(-1)
            blue_pos = normalize_value(49.9)
            green_pos = normalize_value(50.0)

            cdict = {
                'red': [
                    (missing_pos, 1.0, 1.0),
                    (zero_pos, 0.0, 0.0),
                    (blue_pos, thesis_blue[0], thesis_blue[0]),
                    (green_pos, ifisc_green[0], ifisc_green[0]),
                    (1.0, thesis_red[0], thesis_red[0])
                ],
                'green': [
                    (missing_pos, 1.0, 1.0),
                    (zero_pos, 0.0, 0.0),
                    (blue_pos, thesis_blue[1], thesis_blue[1]),
                    (green_pos, ifisc_green[1], ifisc_green[1]),
                    (1.0, thesis_red[1], thesis_red[1])
                ],
                'blue': [
                    (missing_pos, 1.0, 1.0),
                    (zero_pos, 0.0, 0.0),
                    (blue_pos, thesis_blue[2], thesis_blue[2]),
                    (green_pos, ifisc_green[2], ifisc_green[2]),
                    (1.0, thesis_red[2], thesis_red[2])
                ]
            }

            custom_cmap = mcolors.LinearSegmentedColormap('custom_heatmap', cdict, N=256)

            norm = mcolors.Normalize(vmin=vmin_norm, vmax=vmax_norm)

            fig, ax = plt.subplots(figsize=(12, 10))

            im = ax.imshow(heatmap_data_normalized, aspect='auto', origin='lower', cmap=custom_cmap, norm=norm, interpolation='nearest')

            num_ticks_x = min(16, len(param1_vals))
            num_ticks_y = min(16, len(param2_vals))

            if len(param1_vals) > 16:
                x_indices = np.linspace(0, len(param1_vals) - 1, num_ticks_x, dtype=int)
                ax.set_xticks(x_indices)
                ax.set_xticklabels([format_val(param1_vals[i], param1_name) for i in x_indices], rotation=45, ha='right', fontsize=28)
            else:
                ax.set_xticks(np.arange(len(param1_vals)))
                ax.set_xticklabels([format_val(v, param1_name) for v in param1_vals], rotation=45, ha='right', fontsize=28)

            if len(param2_vals) > 16:
                y_indices = np.linspace(0, len(param2_vals) - 1, num_ticks_y, dtype=int)
                ax.set_yticks(y_indices)
                ax.set_yticklabels([format_val(param2_vals[i], param2_name) for i in y_indices], fontsize=28)
            else:
                ax.set_yticks(np.arange(len(param2_vals)))
                ax.set_yticklabels([format_val(v, param2_name) for v in param2_vals], fontsize=28)

            param1_label = param1_name.replace('_', ' ').title()
            param2_label = param2_name.replace('_', ' ').title()
            if param1_name == 'lambda':
                param1_label = 'Lambda ($\\lambda$)'
            elif param1_name == 'omega':
                param1_label = 'Omega ($\\omega$)'
            elif param1_name == 'gamma_real':
                param1_label = 'Gamma Real ($\\gamma_r$)'
            elif param1_name == 'gamma_imag':
                param1_label = 'Gamma Imaginary ($\\gamma_i$)'

            if param2_name == 'lambda':
                param2_label = 'Lambda ($\\lambda$)'
            elif param2_name == 'omega':
                param2_label = 'Omega ($\\omega$)'
            elif param2_name == 'gamma_real':
                param2_label = 'Gamma Real ($\\gamma_r$)'
            elif param2_name == 'gamma_imag':
                param2_label = 'Gamma Imaginary ($\\gamma_i$)'

            ax.set_xlabel(param1_label, fontsize=30)
            ax.set_ylabel(param2_label, fontsize=30)
            title = f'Test Accuracy Heatmap: {param1_label} vs {param2_label}'
            if title_suffix:
                title += f' ({title_suffix})'
            # ax.set_title(title)

            for j in range(len(param2_vals)):
                for i in range(len(param1_vals)):
                    if missing_mask[j, i]:
                        continue
                    avg_val = heatmap_data_local[j, i]
                    max_val = max_data[j, i]
                    if np.isnan(avg_val) or np.isnan(max_val):
                        continue
                    text = f'{avg_val:.1f}\n{max_val:.1f}'
                    ax.text(i, j, text, ha='center', va='center', color='white', fontsize=6)

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Test Accuracy (\\%)', fontsize=30)
            cbar.ax.tick_params(labelsize=28)

            if len(current_results) > 0:
                current_valid = [r for r in current_results if not r.get('numerical_error', False)]
                if current_valid:
                    best_result = max(current_valid, key=lambda x: x['best_test_acc'])
                    best_idx1 = find_closest_idx(best_result[param1_name], param1_vals)
                    best_idx2 = find_closest_idx(best_result[param2_name], param2_vals)
                    if best_idx1 is not None and best_idx2 is not None:
                        rect = mpatches.Rectangle(
                            (best_idx1 - 0.5, best_idx2 - 0.5),
                            1.0,
                            1.0,
                            linewidth=2,
                            edgecolor='white',
                            facecolor='none',
                            zorder=10,
                        )
                        ax.add_patch(rect)

            plt.tight_layout()
            base_name = f'heatmap_{param1_name}_vs_{param2_name}'
            if suffix:
                base_name += f'_{suffix}'
            output_path = output_dir / f'{base_name}_{timestamp}.png'
            plt.savefig(output_path)
            plt.close()
            print(f"Generated heatmap: {output_path}")

        build_and_save_heatmap(results)

        if avg_param1_name and avg_param1_vals:
            for v3 in avg_param1_vals:
                filtered = [r for r in results if abs(r.get(avg_param1_name, 0.0) - v3) < 1e-6]
                if not filtered:
                    continue
                label_val = format_val(v3, avg_param1_name)
                title_suffix = ''
                if avg_param1_name == 'lambda':
                    title_suffix = f'Lambda ($\\lambda$) = {label_val}'
                elif avg_param1_name == 'omega':
                    title_suffix = f'Omega ($\\omega$) = {label_val}'
                elif avg_param1_name == 'gamma_real':
                    title_suffix = f'Gamma Real ($\\gamma_r$) = {label_val}'
                elif avg_param1_name == 'gamma_imag':
                    title_suffix = f'Gamma Imaginary ($\\gamma_i$) = {label_val}'
                else:
                    title_suffix = f'{avg_param1_name} = {label_val}'
                build_and_save_heatmap(
                    filtered,
                    suffix=f'{avg_param1_name}_{label_val.replace(".", "p")}',
                    title_suffix=title_suffix,
                )

if args.plot_only:
    with open(args.plot_only, 'r') as f:
        data = json.load(f)
        results = data.get('results', [])
    
    base_results_dir = Path(args.plot_only).parent
    print(f"Generating heatmaps from {args.plot_only}...")
    print(f"Found {len(results)} results")
    config = data.get('config', {})
    omega_fixed = config.get('omega_fixed', False)
    
    json_filename = Path(args.plot_only).stem
    if 'grid_search_results_' in json_filename:
        run_id = json_filename.replace('grid_search_results_', '')
    else:
        run_id = json_filename
    timestamp = run_id
    plots_dir = base_results_dir / run_id
    
    generate_heatmaps(results, plots_dir, omega_fixed=omega_fixed, timestamp=timestamp, config=config)
    print(f"\nPlots saved in {plots_dir}")
    exit(0)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = Path(args.results_dir)
results_dir.mkdir(exist_ok=True)

results_file = results_dir / f'grid_search_results_{timestamp}.json'
completed_file = results_dir / f'grid_search_completed_{timestamp}.json'

if args.fix_omega:
    all_combinations = list(itertools.product(lambda_values, gamma_real_values, gamma_imag_values))
    total_combinations = len(all_combinations)
else:
    all_combinations = list(itertools.product(lambda_values, omega_values, gamma_real_values, gamma_imag_values))
    total_combinations = len(all_combinations)

results = []
completed_indices = set()

if args.resume:
    print(f"Resuming from {args.resume}...")
    with open(args.resume, 'r') as f:
        resume_data = json.load(f)
        results = resume_data.get('results', [])
        completed_indices = set(resume_data.get('completed_indices', []))
        print(f"  Loaded {len(results)} completed results")

print(f"\nStarting grid search: {total_combinations} total combinations")
print(f"  {len(completed_indices)} already completed")
print(f"  {total_combinations - len(completed_indices)} remaining\n")

pbar = tqdm(total=total_combinations, initial=len(completed_indices), desc="Grid search progress")

for idx, combo in enumerate(all_combinations):
    if idx in completed_indices:
        continue
    
    if args.fix_omega:
        lambda_val, gamma_real_val, gamma_imag_val = combo
        omega_val = fixed_omega
    else:
        lambda_val, omega_val, gamma_real_val, gamma_imag_val = combo
    
    pbar.set_description(f"λ={lambda_val:.3f}, ω={omega_val:.4f}, γr={gamma_real_val:.3f}, γi={gamma_imag_val:.3f}")
    
    result = run_training(lambda_val, omega_val, gamma_real_val, gamma_imag_val)
    result['index'] = idx
    results.append(result)
    completed_indices.add(idx)
    
    pbar.update(1)
    
    if (len(results) % args.save_interval == 0) or (idx == total_combinations - 1):
        with open(results_file, 'w') as f:
            json.dump({
                'config': {
                    'lambda_range': [float(args.lambda_min), float(args.lambda_max), args.lambda_steps],
                    'omega_fixed': args.fix_omega,
                    'omega_value': float(fixed_omega) if args.fix_omega else None,
                    'omega_range': [float(omega_values[0]), float(omega_values[-1]), args.omega_steps] if not args.fix_omega else None,
                    'gamma_real_range': [float(args.gamma_real_min), float(args.gamma_real_max), args.gamma_real_steps],
                    'gamma_imag_range': [float(args.gamma_imag_min), float(args.gamma_imag_max), args.gamma_imag_steps],
                    'num_hidden': args.num_hidden,
                    'epochs': args.epochs,
                    'early_stop_patience': args.early_stop_patience,
                },
                'results': results,
                'completed_indices': list(completed_indices)
            }, f, indent=2)
        pbar.write(f"Saved results to {results_file} ({len(results)}/{total_combinations} completed)")

pbar.close()

print(f"\nGrid search complete! Results saved to {results_file}")

print("\nGenerating heatmaps...")
config = {
    'lambda_range': [float(args.lambda_min), float(args.lambda_max), args.lambda_steps],
    'omega_fixed': args.fix_omega,
    'omega_value': float(fixed_omega) if args.fix_omega else None,
    'omega_range': [float(omega_values[0]), float(omega_values[-1]), args.omega_steps] if not args.fix_omega else None,
    'gamma_real_range': [float(args.gamma_real_min), float(args.gamma_real_max), args.gamma_real_steps],
    'gamma_imag_range': [float(args.gamma_imag_min), float(args.gamma_imag_max), args.gamma_imag_steps],
    'num_hidden': args.num_hidden,
    'epochs': args.epochs,
    'early_stop_patience': args.early_stop_patience,
}
plot_dir = results_dir / timestamp
generate_heatmaps(results, plot_dir, omega_fixed=args.fix_omega, timestamp=timestamp, config=config)

print(f"\nAll done! Results and plots saved in {plot_dir}")

