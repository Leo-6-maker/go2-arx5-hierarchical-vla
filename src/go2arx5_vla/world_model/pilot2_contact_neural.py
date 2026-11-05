"""Small Torch heads, isolated from native NumPy replay construction."""
import torch
from torch import nn


class DirectContactWM(nn.Module):
    def __init__(self, history_dimension, plan_dimension=64, horizon_count=6, hidden=64):
        super().__init__()
        self.register_buffer('mean', torch.zeros(history_dimension+plan_dimension))
        self.register_buffer('scale', torch.ones_like(self.mean))
        self.horizon_count = horizon_count
        self.network = nn.Sequential(nn.Linear(len(self.mean), hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, horizon_count*10))

    def fit_normalizer(self, x):
        self.mean.copy_(x.mean(0)); self.scale.copy_(x.std(0, unbiased=False).clamp_min(1e-4))

    def forward(self, x):
        return self.network((x-self.mean)/self.scale).reshape(-1, self.horizon_count, 10)

