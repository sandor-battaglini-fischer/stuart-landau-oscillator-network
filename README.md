# Network of Stuart-Landau oscillators for information processing tasks

This network of oscillators is inspired by neuromorphic photonic architectures and can be used to efficiently solve a wide range of tasks such as sMNIST digit recognition, IMDb sentiment analysis and time series prediction such as Mackey-Glass. As a low-dimensional recurrent neural network (RNN), it remains mechanistically interpretable from a dynamical system point of view. This repository provides the functionality to analyse the learning behaviour of this oscillatory model and by extention allows insight into areas such as optical neural networks or neural dynamics as well.

This repository constitutes the code for the paper in XX by myself and X, which is my first publication as a PhD student at IFISC.

Citation:

## Repository Structure

```
stuart-landau-oscillator-network/
├── models/                    # Model implementations
│   ├── __init__.py
│   └── stuart_landau.py       # Stuart-Landau oscillator model (SLON)
├── training/                  # Training scripts for different tasks
│   ├── train_smnist.py        # Sequential MNIST classification
│   ├── train_imdb.py          # IMDB sentiment analysis
│   └── train_mackey_glass.py  # Mackey-Glass time series prediction
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── slon_analysis.py       # Model analysis utilities
│   └── manifold_dimension_analysis.py  # Manifold dimension analysis
├── analysis/                  # Post-training analysis scripts
├── results/                   # Training outputs and results
└── README.md
```

## Usage

The repository contains a single Stuart-Landau oscillator model that can be tested on three different tasks:

1. **Sequential MNIST (sMNIST)**: `python training/train_smnist.py`
2. **IMDB Sentiment Analysis**: `python training/train_imdb.py`
3. **Mackey-Glass Prediction**: `python training/train_mackey_glass.py`

Each training script supports various hyperparameters and parameter sweeps. Use `--help` for each script to see available options.

## Files

