"""Uncalibrated hourly-risk baselines with explicit padding handling."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class LastHourLogistic(nn.Module):
    def __init__(self, channels, hidden=64):
        super().__init__()
        self.head = nn.Linear(channels, 1)

    def forward(self, x, lengths):
        rows = torch.arange(x.shape[0], device=x.device)
        return self.head(x[rows, lengths.to(x.device) - 1]).squeeze(-1)


class RiskLSTM(nn.Module):
    def __init__(self, channels, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(channels, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, lengths):
        # Inputs are right-padded, unlike the historical lazy Dataset.
        # pack_padded_sequence prevents padded rows from influencing h_n.
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.head(hidden[-1]).squeeze(-1)


def build_model(name, channels, hidden=64):
    classes = {'logistic': LastHourLogistic, 'lstm': RiskLSTM}
    if name not in classes:
        raise ValueError('Unknown model architecture')
    return classes[name](channels, hidden)
