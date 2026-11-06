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


class PrefixContactWM(DirectContactWM):
    """Each direct head receives only the commands up to its forecast horizon.

    The parameters, initialization and TRAIN normalizer match the full-plan
    model. Masking happens after normalization: unknown later commands are a
    fixed TRAIN-mean prior, never values from the candidate's later suffix.
    This guarantees prefix invariance, not complete latent-state dynamics.
    """
    def __init__(self, history_dimension, plan_dimension=64,
                 horizons=(1, 4, 8, 16, 32, 64), hidden=64):
        if (history_dimension < 1 or not horizons
                or any(type(h) is not int or not 0 < h <= plan_dimension for h in horizons)
                or list(horizons) != sorted(set(horizons))):
            raise ValueError('DECLARED_FORECAST_PREFIXES_REQUIRED')
        super().__init__(history_dimension, plan_dimension, len(horizons), hidden)
        self.register_buffer('forecast_horizons', torch.tensor(horizons, dtype=torch.int64))
        self.register_buffer('prefix_mask', (torch.arange(plan_dimension)[None]
            < self.forecast_horizons[:, None]).to(torch.float32))

    def forward(self, x):
        z = (x-self.mean)/self.scale
        past = z[:, :-self.prefix_mask.shape[1]]
        future = z[:, -self.prefix_mask.shape[1]:]
        views = torch.cat((past[:, None].expand(-1, self.horizon_count, -1),
            future[:, None]*self.prefix_mask[None]), -1)
        heads = self.network(views).reshape(-1, self.horizon_count, self.horizon_count, 10)
        index = torch.arange(self.horizon_count, device=x.device)
        return heads[:, index, index]

