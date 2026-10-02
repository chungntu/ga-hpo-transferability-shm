"""
A conventional 1D-CNN for vibration classification, as an alternative base
model to the WaveNet port.

Rationale. Under the leakage-free record-level split the WaveNet pipeline
memorises its training records (train acc 0.88-1.00) and does not transfer
across measurement setups (test 0.11-0.22, chance 0.20), at every window
length and learning rate tried. A PSD + logistic-regression baseline on the
same splits reaches 0.67-0.72. So the discriminative information survives the
split; the raw-waveform model is not using it.

This architecture is the one established for these benchmarks (Abdeljaber et
al. 2017, reference [5] of the paper, on QUGS): stacked conv + pooling blocks
that progressively downsample, then global pooling into a small classifier.
The essential difference from the WaveNet port is that the representation is
COMPRESSED before classification instead of being flattened at full length.

Hyperparameters are kept analogous to the paper's four genes so the
transferability study can run on either base model:
    lr, filters, n_blocks, kernel_size
"""
from __future__ import annotations

import math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    def __init__(self, n_classes, filters=16, n_blocks=4, kernel_size=9,
                 width=2048, pool=4, widen=True, dropout=0.3, in_channels=1):
        super().__init__()
        layers = []
        c_in = in_channels
        c = filters
        t = width
        for _ in range(n_blocks):
            layers += [
                nn.Conv1d(c_in, c, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(pool),
            ]
            c_in = c
            if widen:
                c = min(c * 2, 256)
            t //= pool
            if t < 2:
                break
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(c_in, n_classes)

    def forward(self, x):                    # (N, 1, T)
        h = self.features(x)
        h = h.mean(dim=2)                    # global average pooling
        return self.out(self.dropout(h))


def count_params(model):
    return sum(p.numel() for p in model.parameters())
