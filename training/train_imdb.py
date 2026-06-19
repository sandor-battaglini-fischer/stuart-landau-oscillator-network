# IMDB train script for SLON

import os
import argparse
import re
import math
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import urllib.request
import tarfile
import zipfile
import numpy as np
import pickle
import hashlib
import json
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TASK = 'imdb'

from models import SLON

from utils.run_dirs import make_run_dir, sweep_summary_path, epoch_dir, save_training_checkpoint
from utils.slon_analysis import extract_model_parameters, compute_parameter_statistics
from utils.plotting_utils import plot_classification_epoch, create_classification_gifs
from utils.manifold_dimension_analysis import analyze_manifold_dimension, collect_and_save_final_states


# command line arguments
parser = argparse.ArgumentParser(description='SLON training script for IMDB')
parser.add_argument('--num-hidden', type=int, default=50, help='number of units')
parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--shuffle', action = 'store_true', help='whether to shuffle stimulus time steps')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--h', type=float, default=1.0, help='microscopic time constant h (default: 1)')
parser.add_argument('--alpha', type=float, default=0.04, help='excitability coefficient alpha')
parser.add_argument('--omega', type=float, default=0.224, help='natural frequency omega')
parser.add_argument('--gamma', type=float, default=0.01, help='damping coefficient gamma')
parser.add_argument('--lambda-param', type=float, default=-0.05, help='Stuart-Landau: real part of linear coefficient lambda (default: -|gamma|)')
parser.add_argument('--gamma-real', type=float, default=-0.1, help='Stuart-Landau: real part of nonlinear coefficient (default: -0.1)')
parser.add_argument('--gamma-imag', type=float, default=0.1, help='Stuart-Landau: imaginary part of nonlinear coefficient (default: 0.1)')
parser.add_argument('--embed-dim', type=int, default=100, help='word embedding dimension')
parser.add_argument('--max-len', type=int, default=175, help='maximum sequence length (truncate/pad to this)')
parser.add_argument('--min-freq', type=int, default=2, help='minimum word frequency for vocabulary')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate (default: 0.3)')
parser.add_argument('--early-stop-patience', type=int, default=100, help='early stopping patience (default: 5)')
parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay for regularization (default: 0.05)')
parser.add_argument('--glove', type=str, default="glove.6B.100d.txt", help='GloVe embedding file path (e.g., "glove.6B.100d.txt"). If None, uses random embeddings')
parser.add_argument('--glove-dir', type=str, default=os.path.join(DATA_DIR, 'glove'), help='directory to store GloVe embeddings')
parser.add_argument('--sweep-omega', action='store_true', help='enable parameter sweep for omega')
parser.add_argument('--omega-min', type=float, default=None, help='minimum omega value for sweep')
parser.add_argument('--omega-max', type=float, default=None, help='maximum omega value for sweep')
parser.add_argument('--omega-steps', type=int, default=10, help='number of steps for omega sweep (default: 10)')
parser.add_argument('--sweep-lambda', action='store_true', help='enable parameter sweep for lambda (Stuart-Landau only)')
parser.add_argument('--lambda-min', type=float, default=None, help='minimum lambda value for sweep')
parser.add_argument('--lambda-max', type=float, default=None, help='maximum lambda value for sweep')
parser.add_argument('--lambda-steps', type=int, default=10, help='number of steps for lambda sweep (default: 10)')
parser.add_argument('--sweep-gamma-real', action='store_true', help='enable parameter sweep for gamma_real (Stuart-Landau only)')
parser.add_argument('--gamma-real-min', type=float, default=None, help='minimum gamma_real value for sweep')
parser.add_argument('--gamma-real-max', type=float, default=None, help='maximum gamma_real value for sweep')
parser.add_argument('--gamma-real-steps', type=int, default=10, help='number of steps for gamma_real sweep (default: 10)')
parser.add_argument('--sweep-gamma-imag', action='store_true', help='enable parameter sweep for gamma_imag (Stuart-Landau only)')
parser.add_argument('--gamma-imag-min', type=float, default=None, help='minimum gamma_imag value for sweep')
parser.add_argument('--gamma-imag-max', type=float, default=None, help='maximum gamma_imag value for sweep')
parser.add_argument('--gamma-imag-steps', type=int, default=10, help='number of steps for gamma_imag sweep (default: 10)')
parser.add_argument('--cache-dir', type=str, default=os.path.join(DATA_DIR, 'imdb_cache'), help='directory to cache preprocessed data')
parser.add_argument('--force-reprocess', action='store_true', help='force reprocessing even if cached data exists')
parser.add_argument('--analyze-manifold', action='store_true', default=True,
                    help='Enable manifold dimension analysis (runs at end of training and every 10 epochs)')

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    return tokens

# vocabulary class
class Vocabulary:
    def __init__(self, min_freq=1):
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        self.min_freq = min_freq
        
    def build(self, texts):
        for text in tqdm(texts, desc="Building vocabulary"):
            tokens = tokenize(text)
            self.word_counts.update(tokens)
        
        self.word2idx['<pad>'] = 0
        self.word2idx['<unk>'] = 1
        idx = 2
        for word, count in self.word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                idx += 1
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        return len(self.word2idx)
    
    def __call__(self, tokens):
        return [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_counts': dict(self.word_counts),
                'min_freq': self.min_freq
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

# download and load IMDB dataset
def download_imdb(data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(DATA_DIR, 'imdb')
    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    
    if os.path.exists(os.path.join(data_dir, 'aclImdb')):
        print(f"IMDB dataset already exists at {data_dir}")
        return os.path.join(data_dir, 'aclImdb')
    
    os.makedirs(data_dir, exist_ok=True)
    tar_path = os.path.join(data_dir, 'aclImdb_v1.tar.gz')
    
    if not os.path.exists(tar_path):
        print(f"Downloading IMDB dataset to {tar_path}...")
        urllib.request.urlretrieve(url, tar_path)
        print("Download complete.")
    
    print(f"Extracting IMDB dataset...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(data_dir)
    print("Extraction complete.")
    
    return os.path.join(data_dir, 'aclImdb')

def load_imdb_reviews(data_path, split='train'):
    reviews = []
    labels = []
    
    for label_type in ['pos', 'neg']:
        dir_path = os.path.join(data_path, split, label_type)
        label = 1 if label_type == 'pos' else 0
        
        for filename in os.listdir(dir_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(dir_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    reviews.append(text)
                    labels.append(label)
    
    return reviews, labels

def get_cache_hash(min_freq, max_len):
    return hashlib.md5(f"{min_freq}_{max_len}".encode()).hexdigest()[:8]

def save_preprocessed_data(cache_dir, vocab, train_tokens, test_tokens, train_labels, test_labels, min_freq, max_len):
    os.makedirs(cache_dir, exist_ok=True)
    cache_hash = get_cache_hash(min_freq, max_len)
    
    vocab_path = os.path.join(cache_dir, f'vocab_{cache_hash}.pkl')
    train_path = os.path.join(cache_dir, f'train_tokens_{cache_hash}.pkl')
    test_path = os.path.join(cache_dir, f'test_tokens_{cache_hash}.pkl')
    train_labels_path = os.path.join(cache_dir, f'train_labels_{cache_hash}.pkl')
    test_labels_path = os.path.join(cache_dir, f'test_labels_{cache_hash}.pkl')
    
    print(f"Saving preprocessed data to {cache_dir}...")
    vocab.save(vocab_path)
    
    with open(train_path, 'wb') as f:
        pickle.dump(train_tokens, f)
    with open(test_path, 'wb') as f:
        pickle.dump(test_tokens, f)
    with open(train_labels_path, 'wb') as f:
        pickle.dump(train_labels, f)
    with open(test_labels_path, 'wb') as f:
        pickle.dump(test_labels, f)
    
    print(f"Preprocessed data saved with hash {cache_hash}")

def load_preprocessed_data(cache_dir, min_freq, max_len):
    cache_hash = get_cache_hash(min_freq, max_len)
    
    vocab_path = os.path.join(cache_dir, f'vocab_{cache_hash}.pkl')
    train_path = os.path.join(cache_dir, f'train_tokens_{cache_hash}.pkl')
    test_path = os.path.join(cache_dir, f'test_tokens_{cache_hash}.pkl')
    train_labels_path = os.path.join(cache_dir, f'train_labels_{cache_hash}.pkl')
    test_labels_path = os.path.join(cache_dir, f'test_labels_{cache_hash}.pkl')
    
    if all(os.path.exists(p) for p in [vocab_path, train_path, test_path, train_labels_path, test_labels_path]):
        print(f"Loading preprocessed data from cache (hash: {cache_hash})...")
        vocab = Vocabulary.load(vocab_path)
        
        with open(train_path, 'rb') as f:
            train_tokens = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_tokens = pickle.load(f)
        with open(train_labels_path, 'rb') as f:
            train_labels = pickle.load(f)
        with open(test_labels_path, 'rb') as f:
            test_labels = pickle.load(f)
        
        print(f"Loaded vocabulary size: {len(vocab)}")
        print(f"Loaded {len(train_tokens)} training samples and {len(test_tokens)} test samples")
        return vocab, train_tokens, test_tokens, train_labels, test_labels
    
    return None

def download_glove(glove_dir=None, dim=100):
    if glove_dir is None:
        glove_dir = os.path.join(DATA_DIR, 'glove')
    os.makedirs(glove_dir, exist_ok=True)
    
    glove_files = {
        50: 'glove.6B.50d.txt',
        100: 'glove.6B.100d.txt',
        200: 'glove.6B.200d.txt',
        300: 'glove.6B.300d.txt'
    }
    
    if dim not in glove_files:
        raise ValueError(f"GloVe dimension {dim} not supported. Choose from {list(glove_files.keys())}")
    
    filename = glove_files[dim]
    filepath = os.path.join(glove_dir, filename)
    zip_path = os.path.join(glove_dir, 'glove.6B.zip')
    
    if os.path.exists(filepath):
        print(f"GloVe embeddings already exist at {filepath}")
        return filepath
    
    if not os.path.exists(zip_path):
        url = 'https://nlp.stanford.edu/data/glove.6B.zip'
        print(f"Downloading GloVe embeddings to {zip_path}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    
    print(f"Extracting GloVe embeddings...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract(filename, glove_dir)
    print("Extraction complete.")
    
    return filepath

def load_glove_vectors(glove_path, vocab, embed_dim):
    print(f"Loading GloVe vectors from {glove_path}...")
    glove_dict = {}
    
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading GloVe"):
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            if len(vector) == embed_dim:
                glove_dict[word] = vector
    
    print(f"Loaded {len(glove_dict)} GloVe vectors")
    
    vocab_size = len(vocab)
    embedding_matrix = np.random.normal(0, 0.1, (vocab_size, embed_dim)).astype(np.float32)
    
    found = 0
    for word, idx in vocab.word2idx.items():
        if word in glove_dict:
            embedding_matrix[idx] = glove_dict[word]
            found += 1
        elif word == '<pad>':
            embedding_matrix[idx] = np.zeros(embed_dim, dtype=np.float32)
        elif word == '<unk>':
            embedding_matrix[idx] = np.mean(list(glove_dict.values()), axis=0) if glove_dict else np.zeros(embed_dim, dtype=np.float32)
    
    print(f"Initialized {found}/{vocab_size - 2} vocabulary words with GloVe vectors")
    return torch.from_numpy(embedding_matrix)

# wrapper model that includes embedding layer
class SLONWithEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_hidden, num_output, h, alpha, omega, gamma, pad_idx, dropout=0.0, embedding_weights=None, lambda_param=None, gamma_real=None, gamma_imag=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(embedding_weights)
            self.embedding.weight.requires_grad = True
        else:
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
            self.embedding.weight.data[pad_idx].fill_(0)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.slon = SLON(embed_dim, num_hidden, num_output, h, alpha, omega, gamma, lambda_param=lambda_param, gamma_real=gamma_real, gamma_imag=gamma_imag)
    
    def forward(self, token_ids, random_init=None, record=False):
        embedded = self.embedding(token_ids)
        embedded = embedded / math.sqrt(self.embedding.embedding_dim)
        embedded = self.dropout(embedded)
        embedded = embedded * 3.0
        embedded = embedded.permute(1, 0, 2) 
        return self.slon(embedded, random_init=random_init, record=record)

if __name__ == '__main__':
    args = parser.parse_args()

    def resolve_data_path(path):
        if os.path.isabs(path):
            return path
        return os.path.join(PROJECT_ROOT, path)

    args.glove_dir = resolve_data_path(args.glove_dir)
    args.cache_dir = resolve_data_path(args.cache_dir)

    target_period = args.max_len 

    sweep_count = sum([args.sweep_omega, args.sweep_lambda, args.sweep_gamma_real, args.sweep_gamma_imag])
    if sweep_count > 1:
        raise ValueError("Cannot sweep multiple parameters simultaneously. Choose one sweep type.")

    if not args.sweep_omega:
        args.omega = (2 * math.pi) / (target_period * args.h)
        print(f"Auto-adjusted omega to {args.omega:.6f} for period length {target_period} (sequence length)")

    print(args)

    print("Using Stuart-Landau dynamics")

    # fix seed
    torch.manual_seed(args.seed)

    # embedding dimension as input to SLON
    dim_input = args.embed_dim

    # 2 classes for binary sentiment classification
    dim_output = 2

    # batch sizes
    batch_size_train = args.batch_size
    batch_size_test = 1000

    # try to load cached preprocessed data
    cached_data = None
    if not args.force_reprocess:
        cached_data = load_preprocessed_data(args.cache_dir, args.min_freq, args.max_len)
    
    if cached_data is not None:
        vocab, train_tokens, test_tokens, train_labels, test_labels = cached_data
        vocab_size = len(vocab)
        pad_idx = vocab.word2idx['<pad>']
        unk_idx = vocab.word2idx['<unk>']
        
        def process_tokens_to_ids(tokens_list, pad_idx):
            token_ids_list = []
            for tokens in tokens_list:
                token_ids = vocab(tokens)
                if len(token_ids) > args.max_len:
                    token_ids = token_ids[:args.max_len]
                else:
                    token_ids = token_ids + [pad_idx] * (args.max_len - len(token_ids))
                token_ids_list.append(token_ids)
            return token_ids_list
        
        print("Converting cached tokens to token IDs...")
        train_token_ids = process_tokens_to_ids(train_tokens, pad_idx)
        test_token_ids = process_tokens_to_ids(test_tokens, pad_idx)
        
        train_data = [(torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)) 
                      for token_ids, label in zip(train_token_ids, train_labels)]
        test_data = [(torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)) 
                     for token_ids, label in zip(test_token_ids, test_labels)]
    else:
        # download and load data
        print("Loading IMDB dataset...")
        imdb_path = download_imdb()
        train_texts, train_labels = load_imdb_reviews(imdb_path, 'train')
        test_texts, test_labels = load_imdb_reviews(imdb_path, 'test')

        # build vocabulary and tokenize
        print("Building vocabulary and tokenizing...")
        vocab = Vocabulary(min_freq=args.min_freq)
        
        print("Tokenizing training texts...")
        train_tokens = [tokenize(text) for text in tqdm(train_texts, desc="Tokenizing train")]
        print("Tokenizing test texts...")
        test_tokens = [tokenize(text) for text in tqdm(test_texts, desc="Tokenizing test")]
        
        for tokens in train_tokens:
            vocab.word_counts.update(tokens)
        
        vocab.word2idx['<pad>'] = 0
        vocab.word2idx['<unk>'] = 1
        idx = 2
        for word, count in vocab.word_counts.items():
            if count >= args.min_freq:
                vocab.word2idx[word] = idx
                idx += 1
        
        vocab.idx2word = {idx: word for word, idx in vocab.word2idx.items()}
        vocab_size = len(vocab)
        print(f"Vocabulary size: {vocab_size}")

        pad_idx = vocab.word2idx['<pad>']
        unk_idx = vocab.word2idx['<unk>']
        
        save_preprocessed_data(args.cache_dir, vocab, train_tokens, test_tokens, 
                              train_labels, test_labels, args.min_freq, args.max_len)

        # process data function
        def process_tokens_to_ids(tokens, pad_idx):
            token_ids = vocab(tokens)
            if len(token_ids) > args.max_len:
                token_ids = token_ids[:args.max_len]
            else:
                token_ids = token_ids + [pad_idx] * (args.max_len - len(token_ids))
            return token_ids

        # process all data
        print("Converting tokens to token IDs...")
        train_token_ids = [process_tokens_to_ids(tokens, pad_idx) for tokens in tqdm(train_tokens, desc="Processing train")]
        test_token_ids = [process_tokens_to_ids(tokens, pad_idx) for tokens in tqdm(test_tokens, desc="Processing test")]
        
        train_data = [(torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)) 
                      for token_ids, label in zip(train_token_ids, train_labels)]
        test_data = [(torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)) 
                     for token_ids, label in zip(test_token_ids, test_labels)]

    # load GloVe embeddings if specified
    embedding_weights = None
    if args.glove:
        glove_path = args.glove
        if not os.path.isabs(glove_path):
            glove_path = os.path.join(args.glove_dir, glove_path)
        
        if not os.path.exists(glove_path):
            print(f"GloVe file not found at {glove_path}, downloading...")
            glove_path = download_glove(args.glove_dir, dim=args.embed_dim)
        
        embedding_weights = load_glove_vectors(glove_path, vocab, args.embed_dim)

    # split train into train and validation
    size_validation = 5000
    train_data, valid_data = train_data[size_validation:], train_data[:size_validation]

    # helper function to collate batches
    def collate_batch(batch):
        texts, labels = zip(*batch)
        texts = torch.stack(texts)
        labels = torch.stack(labels)
        return texts, labels

    # data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_data, batch_size=batch_size_test, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=False, collate_fn=collate_batch)

    def train_with_params(omega_value, lambda_value=None, gamma_real_value=None, gamma_imag_value=None, sweep_idx=None, sweep_type=None):
        lambda_param = lambda_value if lambda_value is not None else args.lambda_param
        gamma_real_param = gamma_real_value if gamma_real_value is not None else args.gamma_real
        gamma_imag_param = gamma_imag_value if gamma_imag_value is not None else args.gamma_imag
        
        model = SLONWithEmbedding(vocab_size, dim_input, args.num_hidden, dim_output, 
                                  args.h, args.alpha, omega_value, args.gamma, pad_idx, 
                                  dropout=args.dropout, embedding_weights=embedding_weights,
                                  lambda_param=lambda_param, gamma_real=gamma_real_param, gamma_imag=gamma_imag_param)

        loss = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW([
            {'params': model.embedding.parameters(), 'lr': args.lr * 0.5, 'weight_decay': args.weight_decay},
            {'params': model.slon.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}
        ], lr=args.lr, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
        )

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if sweep_idx is not None:
            if sweep_type == 'omega':
                sweep_suffix = f'omega{omega_value:.6f}'
            elif sweep_type == 'lambda':
                sweep_suffix = f'lambda{lambda_param:.6f}'
            elif sweep_type == 'gamma_real':
                sweep_suffix = f'gammareal{gamma_real_param:.6f}'
            elif sweep_type == 'gamma_imag':
                sweep_suffix = f'gammaimag{gamma_imag_param:.6f}'
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
        fh_log.write(f'gamma_real: {gamma_real_param:.6f}\n')
        fh_log.write(f'gamma_imag: {gamma_imag_param:.6f}\n')
        fh_log.write(f'embed_dim: {args.embed_dim}\n')
        fh_log.write(f'max_len: {args.max_len}\n')
        fh_log.write(f'min_freq: {args.min_freq}\n')
        fh_log.write(f'dropout: {args.dropout}\n')
        fh_log.write(f'weight_decay: {args.weight_decay}\n')
        fh_log.write(f'early_stop_patience: {args.early_stop_patience}\n')
        fh_log.write(f'glove: {args.glove}\n')
        fh_log.write('='*60 + '\n')
        fh_log.flush()
        
        return model, loss, optimizer, scheduler, fh_log, output_dir

    # run inference on test set
    def evaluate_model(data_loader, model, loss, output_dir, epoch = None, batch = None, return_predictions=False):
        model.eval()
        correct = 0
        test_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            # loop over batches in data loader
            for i, (token_ids, labels) in enumerate(data_loader):
                if args.shuffle:
                    # shuffle sequence order
                    perm = torch.randperm(token_ids.size(1))
                    token_ids = token_ids[:, perm]

                # run model inference - record dynamics for first batch when epoch and batch are provided
                record_dynamics = (i == 0 and epoch is not None and batch is not None)
                output = model(token_ids, record=record_dynamics)
                prediction = output['output']

                # compute loss + number of correct predictions
                batch_loss = loss(prediction, labels)
                if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                    tqdm.write(f'WARNING: NaN/Inf loss detected during evaluation, skipping this batch.')
                    continue
                test_loss += batch_loss.item()
                pred_label = prediction.data.max(1, keepdim=True)[1]
                correct += pred_label.eq(labels.data.view_as(pred_label)).sum()
                
                if return_predictions:
                    all_preds.append(pred_label.squeeze().cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

                # if record_dynamics:
                #     plt.figure()

                #     if args.dynamics == 'sl':
                #         for unit_idx in range(args.num_hidden):
                #             z_magnitude = torch.sqrt(output['rec_z_real'][0, :, unit_idx]**2 + 
                #                                    output['rec_z_imag'][0, :, unit_idx]**2)
                #             plt.plot(z_magnitude)
                #         plt.ylabel('|z|')
                #     else:
                #         for unit_idx in range(args.num_hidden):
                #             plt.plot(output['rec_x_t'][0, :, unit_idx])
                #         plt.ylabel('amplitude')
                    
                #     plt.title(f'epoch {epoch}, batch {batch} ({args.dynamics})')
                #     plt.xlabel('time')
                #     timestamp_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
                #     # plt.savefig(f'{output_dir}/dynamics_epoch{epoch:02d}_batch{batch:03d}_{timestamp_suffix}.png')

                #     plt.close()

        # compute loss and accuracy
        test_loss /= len(data_loader)
        accuracy = 100. * correct / len(data_loader.dataset)
        
        if return_predictions:
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            return accuracy.item(), all_preds, all_labels
        else:
            return accuracy.item()



    def run_training(omega_value, lambda_value=None, gamma_real_value=None, gamma_imag_value=None, sweep_idx=None, sweep_type=None):
        model, loss_fn, optimizer, scheduler, fh_log, output_dir = train_with_params(omega_value, lambda_value, gamma_real_value, gamma_imag_value, sweep_idx, sweep_type)
        
        best_eval = 0.
        final_test_acc = 0.
        best_epoch = 0
        patience_counter = 0
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
        if gamma_real_value is not None:
            param_str += f', gamma_real={gamma_real_value:.6f}'
        if gamma_imag_value is not None:
            param_str += f', gamma_imag={gamma_imag_value:.6f}'
        
        for epoch in tqdm(range(args.epochs), total = args.epochs):
            tqdm.write(f'epoch {epoch} ({param_str})')
            
            epoch_train_loss = 0.0

            for batch_idx, (token_ids, labels) in tqdm(enumerate(train_loader), total = len(train_loader)):
                model.train()

                if args.shuffle:
                    perm = torch.randperm(token_ids.size(1))
                    token_ids = token_ids[:, perm]

                optimizer.zero_grad()

                output = model(token_ids)
                prediction = output['output']

                train_loss = loss_fn(prediction, labels)
                
                if torch.isnan(train_loss) or torch.isinf(train_loss):
                    tqdm.write(f'WARNING: NaN/Inf loss detected at epoch {epoch} batch {batch_idx}. Skipping this batch.')
                    msg = f'WARNING: Skipping epoch {epoch} batch {batch_idx} due to NaN/Inf loss'
                    fh_log.write(msg + '\n')
                    fh_log.flush()
                    optimizer.zero_grad()
                    continue

                epoch_train_loss += train_loss.item()

                train_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                if batch_idx % 100 == 0:
                    test_acc = evaluate_model(test_loader, model, loss_fn, output_dir, epoch, batch_idx)
                    tqdm.write(f'epoch {epoch} batch {batch_idx}: test acc {test_acc:.2f}, loss: {train_loss.item():.4f}')

            valid_acc = evaluate_model(valid_loader, model, loss_fn, output_dir)
            test_acc, test_preds, test_labels = evaluate_model(test_loader, model, loss_fn, output_dir, epoch, len(train_loader) - 1, return_predictions=True)
            
            is_best = valid_acc > best_eval
            if is_best:
                best_eval = valid_acc
                final_test_acc = test_acc
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            avg_train_loss = epoch_train_loss / len(train_loader)

            # compute training accuracy
            train_acc = evaluate_model(train_loader, model, loss_fn, output_dir)

            # compute per-epoch validation and test losses
            model.eval()
            val_loss_sum = 0.0
            val_total = 0
            test_loss_sum = 0.0
            test_total = 0
            with torch.no_grad():
                for token_ids, labels in valid_loader:
                    if args.shuffle:
                        perm = torch.randperm(token_ids.size(1))
                        token_ids = token_ids[:, perm]
                    output = model(token_ids)
                    prediction = output['output']
                    batch_loss = loss_fn(prediction, labels)
                    val_loss_sum += batch_loss.item() * len(labels)
                    val_total += len(labels)
                for token_ids, labels in test_loader:
                    if args.shuffle:
                        perm = torch.randperm(token_ids.size(1))
                        token_ids = token_ids[:, perm]
                    output = model(token_ids)
                    prediction = output['output']
                    batch_loss = loss_fn(prediction, labels)
                    test_loss_sum += batch_loss.item() * len(labels)
                    test_total += len(labels)
            val_loss_avg = val_loss_sum / max(val_total, 1)
            test_loss_avg = test_loss_sum / max(test_total, 1)

            train_accs.append(train_acc)
            val_accs.append(valid_acc)
            test_accs.append(test_acc)
            train_losses.append(avg_train_loss)
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
                    num_classes=2,
                    class_labels=['Negative', 'Positive'],
                    is_last_epoch=(epoch == args.epochs - 1 or patience_counter >= args.early_stop_patience),
                )
                if cm is not None:
                    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-10)
                    per_class_acc_dict = {f"class_{i}": f"{acc*100:.2f}%" for i, acc in enumerate(per_class_acc)}
                    fh_log.write(f"Per-class accuracies: {per_class_acc_dict}\n")
                    fh_log.flush()
            except Exception as e:
                tqdm.write(f"Warning: Failed to generate plots at epoch {epoch}: {e}")

            metrics_file = f"{output_dir}/metrics.json"
            metrics_data = {
                "epoch": epoch,
                "train_acc": float(train_accs[-1]),
                "val_acc": float(valid_acc),
                "test_acc": float(test_acc),
                "train_loss": float(avg_train_loss),
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

            msg = f'epoch {epoch}: train_loss: {avg_train_loss:.4f}, val: {valid_acc:.4f}, test: {test_acc:.4f}'
            if valid_acc == best_eval:
                msg += ' [BEST]'
            fh_log.write(msg + '\n')
            fh_log.flush()
            tqdm.write(msg)
            
            scheduler.step(valid_acc)

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
                        variance_threshold=0.95,
                        is_imdb=True
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

            if patience_counter >= args.early_stop_patience:
                tqdm.write(f'Early stopping at epoch {epoch}. Best validation: {best_eval:.4f} at epoch {best_epoch}')
                msg_stop = f'Early stopping at epoch {epoch}. Best validation: {best_eval:.4f} at epoch {best_epoch}'
                fh_log.write(msg_stop + '\n')
                fh_log.flush()
                break


        msg = f'best test: {final_test_acc:.2f} (at epoch {best_epoch}, val: {best_eval:.4f})'
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
                    variance_threshold=0.95,
                    is_imdb=True
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
                    is_imdb=True
                )
            except Exception as e:
                import traceback
                print(f"Warning: Failed to collect final states or create animation: {e}")
                fh_log.write(f"\nWarning: Failed to collect final states: {e}\n")
                fh_log.flush()
        
        fh_log.close()
        
        result = {'omega': omega_value, 'best_val_acc': best_eval, 'best_test_acc': final_test_acc, 'best_epoch': best_epoch}
        if lambda_value is not None:
            result['lambda'] = lambda_value
        if gamma_real_value is not None:
            result['gamma_real'] = gamma_real_value
        if gamma_imag_value is not None:
            result['gamma_imag'] = gamma_imag_value
        return result

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
            f.write(f"{'Omega':<15} {'Best Val Acc':<15} {'Best Test Acc':<15} {'Best Epoch':<12}\n")
            f.write(f"{'-'*60}\n")
            
            best_overall = max(sweep_results, key=lambda x: x['best_val_acc'])
            
            for result in sweep_results:
                marker = " <-- BEST" if result == best_overall else ""
                f.write(f"{result['omega']:<15.6f} {result['best_val_acc']:<15.4f} {result['best_test_acc']:<15.4f} {result['best_epoch']:<12}{marker}\n")
            
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
            f.write(f"{'Lambda':<15} {'Best Val Acc':<15} {'Best Test Acc':<15} {'Best Epoch':<12}\n")
            f.write(f"{'-'*60}\n")
            
            best_overall = max(sweep_results, key=lambda x: x['best_test_acc'])
            
            for result in sweep_results:
                marker = " <-- BEST" if result == best_overall else ""
                f.write(f"{result['lambda']:<15.6f} {result['best_val_acc']:<15.4f} {result['best_test_acc']:<15.4f} {result['best_epoch']:<12}{marker}\n")
            
            f.write(f"{'='*60}\n")
            f.write(f"Best overall: lambda={best_overall['lambda']:.6f}, val={best_overall['best_val_acc']:.4f}, test={best_overall['best_test_acc']:.4f}\n")
        
        print(f"\n{'='*60}")
        print(f"Sweep complete! Summary saved to {sweep_summary_file}")
        print(f"Best result: lambda={best_overall['lambda']:.6f}, val={best_overall['best_val_acc']:.4f}, test={best_overall['best_test_acc']:.4f}")
        print(f"{'='*60}\n")
    elif args.sweep_gamma_real:
        if args.gamma_real_min is None or args.gamma_real_max is None:
            raise ValueError("--gamma-real-min and --gamma-real-max must be specified when --sweep-gamma-real is enabled")
        
        gamma_real_values = np.linspace(args.gamma_real_min, args.gamma_real_max, args.gamma_real_steps)
        print(f"Starting gamma_real sweep: {args.gamma_real_steps} steps from {args.gamma_real_min:.6f} to {args.gamma_real_max:.6f}")
        print(f"Gamma_real values: {gamma_real_values}")
        
        sweep_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        sweep_results = []
        
        for sweep_idx, gamma_real_val in enumerate(gamma_real_values):
            print(f"\n{'='*60}")
            print(f"Sweep iteration {sweep_idx + 1}/{args.gamma_real_steps}: gamma_real = {gamma_real_val:.6f}")
            print(f"{'='*60}\n")
            
            result = run_training(args.omega, gamma_real_value=gamma_real_val, sweep_idx=sweep_idx, sweep_type='gamma_real')
            sweep_results.append(result)
            
            print(f"\nCompleted sweep {sweep_idx + 1}/{args.gamma_real_steps}: gamma_real={result['gamma_real']:.6f}, val={result['best_val_acc']:.4f}, test={result['best_test_acc']:.4f}\n")
        
        sweep_summary_file = sweep_summary_path(PROJECT_ROOT, TASK, 'gamma_real', sweep_timestamp)
        with open(sweep_summary_file, 'w') as f:
            f.write(f"Gamma_real Sweep Summary\n")
            f.write(f"Timestamp: {sweep_timestamp}\n")
            f.write(f"Range: {args.gamma_real_min:.6f} to {args.gamma_real_max:.6f}\n")
            f.write(f"Steps: {args.gamma_real_steps}\n")
            f.write(f"{'='*60}\n")
            f.write(f"{'Gamma_real':<15} {'Best Val Acc':<15} {'Best Test Acc':<15} {'Best Epoch':<12}\n")
            f.write(f"{'-'*60}\n")
            
            best_overall = max(sweep_results, key=lambda x: x['best_val_acc'])
            
            for result in sweep_results:
                marker = " <-- BEST" if result == best_overall else ""
                f.write(f"{result['gamma_real']:<15.6f} {result['best_val_acc']:<15.4f} {result['best_test_acc']:<15.4f} {result['best_epoch']:<12}{marker}\n")
            
            f.write(f"{'='*60}\n")
            f.write(f"Best overall: gamma_real={best_overall['gamma_real']:.6f}, val={best_overall['best_val_acc']:.4f}, test={best_overall['best_test_acc']:.4f}\n")
        
        print(f"\n{'='*60}")
        print(f"Sweep complete! Summary saved to {sweep_summary_file}")
        print(f"Best result: gamma_real={best_overall['gamma_real']:.6f}, val={best_overall['best_val_acc']:.4f}, test={best_overall['best_test_acc']:.4f}")
        print(f"{'='*60}\n")
    elif args.sweep_gamma_imag:
        if args.gamma_imag_min is None or args.gamma_imag_max is None:
            raise ValueError("--gamma-imag-min and --gamma-imag-max must be specified when --sweep-gamma-imag is enabled")
        
        gamma_imag_values = np.linspace(args.gamma_imag_min, args.gamma_imag_max, args.gamma_imag_steps)
        print(f"Starting gamma_imag sweep: {args.gamma_imag_steps} steps from {args.gamma_imag_min:.6f} to {args.gamma_imag_max:.6f}")
        print(f"Gamma_imag values: {gamma_imag_values}")
        
        sweep_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        sweep_results = []
        
        for sweep_idx, gamma_imag_val in enumerate(gamma_imag_values):
            print(f"\n{'='*60}")
            print(f"Sweep iteration {sweep_idx + 1}/{args.gamma_imag_steps}: gamma_imag = {gamma_imag_val:.6f}")
            print(f"{'='*60}\n")
            
            result = run_training(args.omega, gamma_imag_value=gamma_imag_val, sweep_idx=sweep_idx, sweep_type='gamma_imag')
            sweep_results.append(result)
            
            print(f"\nCompleted sweep {sweep_idx + 1}/{args.gamma_imag_steps}: gamma_imag={result['gamma_imag']:.6f}, val={result['best_val_acc']:.4f}, test={result['best_test_acc']:.4f}\n")
        
        sweep_summary_file = sweep_summary_path(PROJECT_ROOT, TASK, 'gamma_imag', sweep_timestamp)
        with open(sweep_summary_file, 'w') as f:
            f.write(f"Gamma_imag Sweep Summary\n")
            f.write(f"Timestamp: {sweep_timestamp}\n")
            f.write(f"Range: {args.gamma_imag_min:.6f} to {args.gamma_imag_max:.6f}\n")
            f.write(f"Steps: {args.gamma_imag_steps}\n")
            f.write(f"{'='*60}\n")
            f.write(f"{'Gamma_imag':<15} {'Best Val Acc':<15} {'Best Test Acc':<15} {'Best Epoch':<12}\n")
            f.write(f"{'-'*60}\n")
            
            best_overall = max(sweep_results, key=lambda x: x['best_val_acc'])
            
            for result in sweep_results:
                marker = " <-- BEST" if result == best_overall else ""
                f.write(f"{result['gamma_imag']:<15.6f} {result['best_val_acc']:<15.4f} {result['best_test_acc']:<15.4f} {result['best_epoch']:<12}{marker}\n")
            
            f.write(f"{'='*60}\n")
            f.write(f"Best overall: gamma_imag={best_overall['gamma_imag']:.6f}, val={best_overall['best_val_acc']:.4f}, test={best_overall['best_test_acc']:.4f}\n")
        
        print(f"\n{'='*60}")
        print(f"Sweep complete! Summary saved to {sweep_summary_file}")
        print(f"Best result: gamma_imag={best_overall['gamma_imag']:.6f}, val={best_overall['best_val_acc']:.4f}, test={best_overall['best_test_acc']:.4f}")
        print(f"{'='*60}\n")
    else:
        result = run_training(args.omega)
        print(f'best test: {result["best_test_acc"]:.2f} (at epoch {result["best_epoch"]}, val: {result["best_val_acc"]:.4f})')
