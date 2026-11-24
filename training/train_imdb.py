import argparse
import hashlib
import json
import math
import os
import pickle
import re
import sys
import tarfile
import urllib.request
import zipfile
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TASK = 'imdb'

from utils.device import resolve_device, seed_all
from utils.manifold_dimension_analysis import analyze_manifold_dimension, run_final_manifold_analysis
from utils.model_factory import build_oscillator, manifold_dynamics_type
from utils.plotting_utils import (
    create_classification_gifs,
    find_class_examples,
    find_class_examples_batch,
    plot_classification_epoch,
    plot_imdb_sentiment_comparison,
    plot_imdb_sentiment_encoding_analysis,
    plot_signal_stages_for_example,
    prepare_imdb_sequence,
)
from utils.run_dirs import (
    append_metrics, epoch_dir, make_run_dir, save_initial_checkpoint, save_training_checkpoint, write_log_header,
)
from utils.slon_analysis import compute_parameter_statistics, extract_model_parameters

CLASS_LABELS = ['Negative', 'Positive']


parser = argparse.ArgumentParser(description='SLON training on IMDb sentiment classification')
parser.add_argument('--num-hidden', type=int, default=50)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--shuffle', action='store_true', help='shuffle token order within each sequence')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--h', type=float, default=1.0, help='integration step')
parser.add_argument('--alpha', type=float, default=0.5, help='input excitability')
parser.add_argument('--omega', type=float, default=None, help='natural frequency (default: one period over --max-len)')
parser.add_argument('--gamma', type=float, default=0.01, help='damping (dho only, used when lambda is not given)')
parser.add_argument('--lambda-param', type=float, default=-0.05)
parser.add_argument('--gamma-real', type=float, default=-0.1)
parser.add_argument('--gamma-imag', type=float, default=0.1)
parser.add_argument('--dynamics', type=str, default='sl', choices=['sl', 'lo', 'dho'])
parser.add_argument('--no-tanh', action='store_true', help='linear input coupling instead of tanh')
parser.add_argument('--embed-dim', type=int, default=100)
parser.add_argument('--max-len', type=int, default=175, help='truncate/pad reviews to this many tokens')
parser.add_argument('--min-freq', type=int, default=2, help='minimum word count for the vocabulary')
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--early-stop-patience', type=int, default=100)
parser.add_argument('--weight-decay', type=float, default=0.05)
parser.add_argument('--glove', type=str, default='glove.6B.100d.txt', help='GloVe file name (empty for random embeddings)')
parser.add_argument('--freeze-embedding', action='store_true', help='keep the embedding at its initial (GloVe) values')
parser.add_argument('--glove-dir', type=str, default=os.path.join(DATA_DIR, 'glove'))
parser.add_argument('--cache-dir', type=str, default=os.path.join(DATA_DIR, 'imdb_cache'))
parser.add_argument('--force-reprocess', action='store_true')
parser.add_argument('--analyze-manifold', action='store_true', default=True)
parser.add_argument('--skip-epoch-plots', action='store_true', help='skip per-epoch plots and manifold analysis')
parser.add_argument('--sweep-mode', action='store_true', help='only write log.txt and metrics.json')
parser.add_argument('--output-dir', type=str, default=None)
parser.add_argument('--sentiment-analysis-examples', type=int, default=100)
parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])


def tokenize(text):
    text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    return text.split()


class Vocabulary:
    def __init__(self, min_freq=1):
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        self.min_freq = min_freq

    def build(self, token_lists):
        for tokens in token_lists:
            self.word_counts.update(tokens)
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        for word, count in self.word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = len(self.word2idx)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        return len(self.word2idx)

    def __call__(self, tokens):
        unk = self.word2idx['<unk>']
        return [self.word2idx.get(token, unk) for token in tokens]

    def __len__(self):
        return len(self.word2idx)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_counts': dict(self.word_counts),
                'min_freq': self.min_freq,
            }, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        vocab = cls(min_freq=data['min_freq'])
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        vocab.word_counts = Counter(data['word_counts'])
        return vocab


def download_imdb(data_dir=None):
    data_dir = data_dir or os.path.join(DATA_DIR, 'imdb')
    extracted = os.path.join(data_dir, 'aclImdb')
    if os.path.exists(extracted):
        return extracted
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, 'aclImdb_v1.tar.gz')
    if not os.path.exists(tar_path):
        print(f'Downloading IMDb to {tar_path}')
        urllib.request.urlretrieve('https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', tar_path)
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(data_dir)
    return extracted


def load_imdb_reviews(data_path, split='train'):
    reviews, labels = [], []
    for label, label_type in ((1, 'pos'), (0, 'neg')):
        dir_path = os.path.join(data_path, split, label_type)
        for filename in sorted(os.listdir(dir_path)):
            if filename.endswith('.txt'):
                with open(os.path.join(dir_path, filename), encoding='utf-8') as f:
                    reviews.append(f.read())
                labels.append(label)
    return reviews, labels


def _cache_paths(cache_dir, min_freq, max_len):
    key = hashlib.md5(f'{min_freq}_{max_len}'.encode()).hexdigest()[:8]
    names = ['vocab', 'train_tokens', 'test_tokens', 'train_labels', 'test_labels']
    return {name: os.path.join(cache_dir, f'{name}_{key}.pkl') for name in names}


def save_preprocessed_data(cache_dir, vocab, train_tokens, test_tokens, train_labels, test_labels, min_freq, max_len):
    os.makedirs(cache_dir, exist_ok=True)
    paths = _cache_paths(cache_dir, min_freq, max_len)
    vocab.save(paths['vocab'])
    for name, obj in (('train_tokens', train_tokens), ('test_tokens', test_tokens),
                      ('train_labels', train_labels), ('test_labels', test_labels)):
        with open(paths[name], 'wb') as f:
            pickle.dump(obj, f)


def load_preprocessed_data(cache_dir, min_freq, max_len):
    paths = _cache_paths(cache_dir, min_freq, max_len)
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    vocab = Vocabulary.load(paths['vocab'])
    loaded = []
    for name in ('train_tokens', 'test_tokens', 'train_labels', 'test_labels'):
        with open(paths[name], 'rb') as f:
            loaded.append(pickle.load(f))
    return (vocab, *loaded)


def encode(vocab, tokens, max_len):
    ids = vocab(tokens)[:max_len]
    return ids + [vocab.word2idx['<pad>']] * (max_len - len(ids))


def load_imdb_data(cache_dir, min_freq, max_len, force_reprocess=False):
    """Returns (vocab, train_data, test_data) with data as lists of (token_ids, label) tensors."""
    cached = None if force_reprocess else load_preprocessed_data(cache_dir, min_freq, max_len)
    if cached is not None:
        vocab, train_tokens, test_tokens, train_labels, test_labels = cached
    else:
        imdb_path = download_imdb()
        train_texts, train_labels = load_imdb_reviews(imdb_path, 'train')
        test_texts, test_labels = load_imdb_reviews(imdb_path, 'test')
        train_tokens = [tokenize(t) for t in tqdm(train_texts, desc='Tokenizing train')]
        test_tokens = [tokenize(t) for t in tqdm(test_texts, desc='Tokenizing test')]
        vocab = Vocabulary(min_freq=min_freq)
        vocab.build(train_tokens)
        save_preprocessed_data(cache_dir, vocab, train_tokens, test_tokens, train_labels, test_labels, min_freq, max_len)

    def to_dataset(token_lists, labels):
        return [(torch.tensor(encode(vocab, tokens, max_len), dtype=torch.long), torch.tensor(label, dtype=torch.long))
                for tokens, label in zip(token_lists, labels)]

    print(f'Vocabulary size: {len(vocab)}')
    return vocab, to_dataset(train_tokens, train_labels), to_dataset(test_tokens, test_labels)


def download_glove(glove_dir=None, dim=100):
    glove_dir = glove_dir or os.path.join(DATA_DIR, 'glove')
    os.makedirs(glove_dir, exist_ok=True)
    if dim not in (50, 100, 200, 300):
        raise ValueError(f'GloVe dimension {dim} not available')
    filename = f'glove.6B.{dim}d.txt'
    filepath = os.path.join(glove_dir, filename)
    if os.path.exists(filepath):
        return filepath
    zip_path = os.path.join(glove_dir, 'glove.6B.zip')
    if not os.path.exists(zip_path):
        print(f'Downloading GloVe to {zip_path}')
        urllib.request.urlretrieve('https://nlp.stanford.edu/data/glove.6B.zip', zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(filename, glove_dir)
    return filepath


def load_glove_vectors(glove_path, vocab, embed_dim):
    cache_dir = os.path.join(os.path.dirname(glove_path), '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    vocab_key = hashlib.md5(json.dumps(sorted(vocab.word2idx.items()), sort_keys=True).encode()).hexdigest()[:16]
    mtime_ns = os.stat(glove_path).st_mtime_ns
    cache_path = os.path.join(cache_dir, f'{os.path.basename(glove_path)}_{embed_dim}_{vocab_key}_{mtime_ns}.pt')
    if os.path.isfile(cache_path):
        return torch.load(cache_path, weights_only=True)

    glove = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in tqdm(f, desc='Loading GloVe'):
            parts = line.strip().split()
            vector = np.array(parts[1:], dtype=np.float32)
            if len(vector) == embed_dim:
                glove[parts[0]] = vector

    matrix = np.random.normal(0, 0.1, (len(vocab), embed_dim)).astype(np.float32)
    found = 0
    for word, idx in vocab.word2idx.items():
        if word in glove:
            matrix[idx] = glove[word]
            found += 1
        elif word == '<pad>':
            matrix[idx] = 0.0
        elif word == '<unk>':
            matrix[idx] = np.mean(list(glove.values()), axis=0) if glove else 0.0
    print(f'{found}/{len(vocab) - 2} vocabulary words initialised from GloVe')

    weights = torch.from_numpy(matrix)
    torch.save(weights, cache_path)
    return weights


class SLONWithEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_hidden, num_output, h, alpha, omega, gamma, pad_idx,
                 dropout=0.0, embedding_weights=None, lambda_param=None, gamma_real=None, gamma_imag=None,
                 dynamics='sl', use_tanh=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(embedding_weights)
        else:
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
            self.embedding.weight.data[pad_idx].fill_(0)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.slon = build_oscillator(
            dynamics, embed_dim, num_hidden, num_output, h, alpha, omega, gamma,
            lambda_param=lambda_param, gamma_real=gamma_real, gamma_imag=gamma_imag, use_tanh=use_tanh,
        )

    def forward(self, token_ids, random_init=None, record=False):
        embedded = self.embedding(token_ids) / math.sqrt(self.embedding.embedding_dim)
        embedded = self.dropout(embedded) * 3.0
        return self.slon(embedded.permute(1, 0, 2), random_init=random_init, record=record)


def collate_batch(batch):
    texts, labels = zip(*batch)
    return torch.stack(texts), torch.stack(labels)


def main():
    args = parser.parse_args()
    if args.sweep_mode:
        args.skip_epoch_plots = True
    if not os.path.isabs(args.glove_dir):
        args.glove_dir = os.path.join(PROJECT_ROOT, args.glove_dir)
    if not os.path.isabs(args.cache_dir):
        args.cache_dir = os.path.join(PROJECT_ROOT, args.cache_dir)
    if args.omega is None:
        args.omega = 2 * math.pi / (args.max_len * args.h)
        print(f'omega set to {args.omega:.6f} (one period over {args.max_len} steps)')

    print(args)
    device = resolve_device(args.device)
    seed_all(args.seed, device)
    print(f'Using device: {device}')

    vocab, train_data, test_data = load_imdb_data(args.cache_dir, args.min_freq, args.max_len, args.force_reprocess)
    pad_idx = vocab.word2idx['<pad>']

    embedding_weights = None
    if args.glove:
        glove_path = args.glove if os.path.isabs(args.glove) else os.path.join(args.glove_dir, args.glove)
        if not os.path.exists(glove_path):
            glove_path = download_glove(args.glove_dir, dim=args.embed_dim)
        embedding_weights = load_glove_vectors(glove_path, vocab, args.embed_dim)

    # reviews are stored class by class: shuffle before taking the validation split,
    # and shuffle the test set because the manifold analysis only reads its first samples
    split_gen = torch.Generator().manual_seed(args.seed)
    train_data = [train_data[i] for i in torch.randperm(len(train_data), generator=split_gen).tolist()]
    test_data = [test_data[i] for i in torch.randperm(len(test_data), generator=split_gen).tolist()]
    valid_data, train_data = train_data[:5000], train_data[5000:]
    for name, split in (('train', train_data), ('validation', valid_data), ('test', test_data)):
        n_pos = sum(int(label) for _, label in split)
        print(f'{name}: {len(split)} reviews ({n_pos} pos / {len(split) - n_pos} neg)')

    batch_size_test = 1000
    pin = device.type == 'cuda'
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, pin_memory=pin)
    valid_loader = DataLoader(valid_data, batch_size=batch_size_test, shuffle=False, collate_fn=collate_batch, pin_memory=pin)
    test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=False, collate_fn=collate_batch, pin_memory=pin)

    model = SLONWithEmbedding(
        len(vocab), args.embed_dim, args.num_hidden, 2, args.h, args.alpha, args.omega, args.gamma, pad_idx,
        dropout=args.dropout, embedding_weights=embedding_weights, lambda_param=args.lambda_param,
        gamma_real=args.gamma_real, gamma_imag=args.gamma_imag, dynamics=args.dynamics, use_tanh=not args.no_tanh,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    param_groups = [{'params': model.slon.parameters(), 'lr': args.lr}]
    if args.freeze_embedding:
        model.embedding.weight.requires_grad_(False)
    else:
        param_groups.insert(0, {'params': model.embedding.parameters(), 'lr': args.lr * 0.5})
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)

    output_dir = make_run_dir(PROJECT_ROOT, TASK, args.output_dir)
    if not args.sweep_mode:
        save_initial_checkpoint(model, output_dir)
    fh_log = open(os.path.join(output_dir, 'log.txt'), 'a')
    write_log_header(fh_log, {
        'dynamics': args.dynamics, 'use_tanh': not args.no_tanh, 'num_hidden': args.num_hidden,
        'epochs': args.epochs, 'batch_size': args.batch_size, 'shuffle': args.shuffle, 'seed': args.seed,
        'lr': args.lr, 'h': args.h, 'alpha': args.alpha, 'omega': f'{args.omega:.6f}', 'gamma': args.gamma,
        'lambda': f'{args.lambda_param:.6f}', 'gamma_real': f'{args.gamma_real:.6f}',
        'gamma_imag': f'{args.gamma_imag:.6f}', 'embed_dim': args.embed_dim, 'max_len': args.max_len,
        'min_freq': args.min_freq, 'dropout': args.dropout, 'weight_decay': args.weight_decay,
        'early_stop_patience': args.early_stop_patience, 'glove': args.glove,
        'freeze_embedding': args.freeze_embedding,
    })

    def log(msg):
        tqdm.write(msg)
        fh_log.write(msg + '\n')
        fh_log.flush()

    def maybe_shuffle(token_ids):
        if args.shuffle:
            token_ids = token_ids[:, torch.randperm(token_ids.size(1))]
        return token_ids

    def evaluate(loader, return_predictions=False):
        model.eval()
        correct, loss_sum, n = 0, 0.0, 0
        preds, labels_all = [], []
        with torch.no_grad():
            for token_ids, labels in loader:
                token_ids = maybe_shuffle(token_ids.to(device, non_blocking=True))
                labels = labels.to(device, non_blocking=True)
                logits = model(token_ids)['output']
                batch_loss = loss_fn(logits, labels)
                if not torch.isfinite(batch_loss):
                    tqdm.write('WARNING: NaN/Inf loss during evaluation, skipping batch')
                    continue
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                loss_sum += batch_loss.item() * len(labels)
                n += len(labels)
                if return_predictions:
                    preds.append(pred.cpu().numpy())
                    labels_all.append(labels.cpu().numpy())
        accuracy = 100.0 * correct / len(loader.dataset)
        avg_loss = loss_sum / max(n, 1)
        if return_predictions:
            return accuracy, avg_loss, np.concatenate(preds) if preds else np.array([]), \
                np.concatenate(labels_all) if labels_all else np.array([])
        return accuracy, avg_loss

    if not args.skip_epoch_plots:
        example_token_ids, example_label = next(iter(test_loader))
        example_token_ids = example_token_ids[0:1]
        example_label = int(example_label[0].item())
        examples = find_class_examples(test_loader, labels=(0, 1))
        neg_example, pos_example = examples[0], examples[1]
        sentiment_batch = find_class_examples_batch(test_loader, labels=(0, 1), num_per_class=args.sentiment_analysis_examples)

    best_val, best_test, best_epoch, patience_counter = 0.0, 0.0, 0, 0
    parameters_history = []
    history = {k: [] for k in ('train_acc', 'val_acc', 'test_acc', 'train_loss', 'val_loss', 'test_loss')}

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        model.train()
        epoch_loss, n_batches, n_skipped = 0.0, 0, 0
        for batch_idx, (token_ids, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            n_batches += 1
            model.train()
            token_ids = maybe_shuffle(token_ids.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            train_loss = loss_fn(model(token_ids)['output'], labels)
            if not torch.isfinite(train_loss):
                log(f'WARNING: Skipping epoch {epoch} batch {batch_idx} due to NaN/Inf loss')
                n_skipped += 1
                continue
            epoch_loss += train_loss.item()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if not args.sweep_mode and batch_idx % 100 == 0:
                test_acc, _ = evaluate(test_loader)
                tqdm.write(f'epoch {epoch} batch {batch_idx}: test acc {test_acc:.2f}, loss: {train_loss.item():.4f}')

        if n_batches and n_skipped == n_batches:
            log(f'epoch {epoch}: all batches NaN/Inf, stopping')
            break

        avg_train_loss = epoch_loss / len(train_loader)
        val_acc, val_loss = evaluate(valid_loader)
        if args.sweep_mode:
            test_acc, test_loss = evaluate(test_loader)
            test_preds = test_labels = None
            train_acc = float('nan')
        else:
            test_acc, test_loss, test_preds, test_labels = evaluate(test_loader, return_predictions=True)
            train_acc, _ = evaluate(train_loader)

        is_best = val_acc > best_val
        if is_best:
            best_val, best_test, best_epoch, patience_counter = val_acc, test_acc, epoch, 0
        else:
            patience_counter += 1

        for key, value in zip(history, (train_acc, val_acc, test_acc, avg_train_loss, val_loss, test_loss)):
            history[key].append(value)

        if not args.sweep_mode:
            model_params = extract_model_parameters(model, args.dynamics)
            parameters_history.append({'epoch': epoch, 'params': model_params,
                                       'stats': compute_parameter_statistics(model_params)})
            with open(os.path.join(output_dir, 'parameters.json'), 'w') as f:
                json.dump(parameters_history, f, indent=2)

        if not args.skip_epoch_plots:
            ep_dir = epoch_dir(output_dir, epoch)
            plot_epoch(args, model, output_dir, ep_dir, epoch, history, test_labels, test_preds,
                       parameters_history, patience_counter, fh_log, example_token_ids, example_label,
                       neg_example, pos_example, sentiment_batch)

        append_metrics(os.path.join(output_dir, 'metrics.json'), {
            'epoch': epoch, 'train_acc': float(train_acc), 'val_acc': float(val_acc), 'test_acc': float(test_acc),
            'train_loss': float(avg_train_loss), 'val_loss': float(val_loss), 'test_loss': float(test_loss),
        })
        log(f'epoch {epoch}: train_loss: {avg_train_loss:.4f}, val: {val_acc:.4f}, test: {test_acc:.4f}'
            + (' [BEST]' if is_best else ''))
        scheduler.step(val_acc)

        if args.analyze_manifold and not args.skip_epoch_plots:
            try:
                result = analyze_manifold_dimension(
                    test_loader, model, manifold_dynamics_type(args.dynamics), ep_dir, epoch=epoch,
                    batch_size_test=batch_size_test, max_samples=5000, variance_threshold=0.95, is_imdb=True,
                )
                if result is not None:
                    tqdm.write(f"  PCA effective dim: {result['effective_dim_pca']}, correlation dim: {result['correlation_dim']}")
            except Exception as e:
                tqdm.write(f'Warning: manifold analysis failed at epoch {epoch}: {e}')

        if not args.sweep_mode:
            save_training_checkpoint(model, output_dir, is_best=is_best)

        if patience_counter >= args.early_stop_patience:
            log(f'Early stopping at epoch {epoch}. Best validation: {best_val:.4f} at epoch {best_epoch}')
            break

    log(f'best test: {best_test:.2f} (at epoch {best_epoch}, val: {best_val:.4f})')

    if not args.skip_epoch_plots:
        if parameters_history:
            create_classification_gifs(output_dir)
        if args.analyze_manifold:
            run_final_manifold_analysis(test_loader, model, args.dynamics, output_dir, batch_size_test, fh_log,
                                        is_imdb=True)
    fh_log.close()


def plot_epoch(args, model, output_dir, ep_dir, epoch, history, test_labels, test_preds, parameters_history,
               patience_counter, fh_log, example_token_ids, example_label, neg_example, pos_example, sentiment_batch):
    try:
        cm = plot_classification_epoch(
            output_dir=output_dir, ep_dir=ep_dir, epoch=epoch,
            train_accs=history['train_acc'], val_accs=history['val_acc'], test_accs=history['test_acc'],
            train_losses=history['train_loss'], val_losses=history['val_loss'], test_losses=history['test_loss'],
            test_labels=test_labels, test_preds=test_preds, parameters_history=parameters_history,
            num_classes=2, class_labels=CLASS_LABELS,
            is_last_epoch=(epoch == args.epochs - 1 or patience_counter >= args.early_stop_patience),
        )
        if cm is not None:
            per_class = cm.diagonal() / (cm.sum(axis=1) + 1e-10)
            per_class = {f'class_{i}': f'{a * 100:.2f}%' for i, a in enumerate(per_class)}
            fh_log.write(f'Per-class accuracies: {per_class}\n')
    except Exception as e:
        tqdm.write(f'Warning: classification plots failed at epoch {epoch}: {e}')

    try:
        with torch.no_grad():
            example_inputs = prepare_imdb_sequence(model, example_token_ids)
        plot_signal_stages_for_example(
            model, example_inputs, ep_dir, epoch, input_mode='norm', task_type='classification',
            class_labels=CLASS_LABELS, true_label=example_label, num_units_plot=args.num_hidden,
            raw_input_label='embedding norm', title_suffix=f'example label: {CLASS_LABELS[example_label]}',
        )
    except Exception as e:
        tqdm.write(f'Warning: signal stage plots failed at epoch {epoch}: {e}')

    try:
        summary = plot_imdb_sentiment_comparison(
            model, neg_example, pos_example, ep_dir, epoch, num_units_plot=args.num_hidden, class_labels=CLASS_LABELS,
        )
        fh_log.write(
            f"Sentiment comparison epoch {epoch}: neg_pred={summary['negative_predicted_class']}, "
            f"pos_pred={summary['positive_predicted_class']}, final_z_diff={summary['final_mean_z_real_diff']:.4f}\n"
        )
    except Exception as e:
        tqdm.write(f'Warning: sentiment comparison failed at epoch {epoch}: {e}')

    try:
        enc = plot_imdb_sentiment_encoding_analysis(
            model, sentiment_batch[0], sentiment_batch[1], ep_dir, epoch, num_units_plot=args.num_hidden,
        )
        fh_log.write(
            f"Sentiment encoding epoch {epoch}: dominant={enc['dominant_encoding']}, "
            f"magnitude_score={enc['magnitude_separation_score']:.3f}, "
            f"phase_score={enc.get('phase_separation_score', 0):.3f}, "
            f"frequency_score={enc['frequency_separation_score']:.3f}, "
            f"top_feature={enc['top_separating_features'][0]['feature']}\n"
        )
    except Exception as e:
        tqdm.write(f'Warning: sentiment encoding analysis failed at epoch {epoch}: {e}')
    fh_log.flush()


if __name__ == '__main__':
    main()
