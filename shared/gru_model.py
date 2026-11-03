"""Causal GRU classifier. Input: [batch, past_and_current_frames, 52]."""
from __future__ import annotations

import numpy as np
from torch import nn


class PostureGRU(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1, dropout=0.2, input_dim=52, num_classes=6):
        super().__init__()
        if int(hidden_size) <= 0 or int(num_layers) not in (1, 2):
            raise ValueError('Invalid GRU architecture')
        self.recurrent = nn.GRU(input_dim, int(hidden_size), num_layers=int(num_layers),
                                batch_first=True, dropout=0.15 if int(num_layers) > 1 else 0.)
        self.head = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(int(hidden_size), num_classes))

    def forward(self, sequence):
        if sequence.ndim != 3 or sequence.shape[-1] != self.recurrent.input_size:
            raise ValueError('Expected [batch, frames, 52]')
        sequence_out, _ = self.recurrent(sequence)
        return self.head(sequence_out[:, -1])


def causal_windows(x, length=16, min_joints=4):
    """Left-pad with zeros; invalid frames break temporal history (no lookahead).

    Returns all frames' windows [T,L,52] and valid target mask [T].
    Matches PosePipeline's per-source causal reset policy.
    """
    array = np.asarray(x, dtype=np.float32)
    if array.ndim == 3 and array.shape[1:] == (13, 4):
        array = array.reshape(len(array), 52)
    if array.ndim != 2 or array.shape[1] != 52 or length < 1:
        raise ValueError('Expected [T,52] or [T,13,4] and positive window')
    valid = (array.reshape(-1, 13, 4)[:, :, 3].sum(axis=1) >= min_joints)
    windows = np.zeros((len(array), length, 52), dtype=np.float32)
    start = 0
    for t in range(len(array)):
        if not valid[t]:
            start = t + 1
            continue
        begin = max(start, t - length + 1)
        n = t - begin + 1
        windows[t, -n:] = array[begin:t+1]
    return windows, valid
