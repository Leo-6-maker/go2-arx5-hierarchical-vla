"""Steering-only components; no optimizer or online training loop is launched.

MLP layout adapted from Microsoft's FlowDAgger NoisePolicy (MIT), pinned
3426f246510fd67a9e39c3c08bd5b02d586ad688; see third_party/flowdagger/LICENSE.
Copyright (c) Microsoft Corporation.
"""
import torch
from torch import nn


def observation_features(context):
    """Only cached RGB/language/observable-state embeddings, never Teacher fields."""
    if not context.observation_id:
        raise ValueError('BOUND_ACTUAL_OBSERVATION_REQUIRED')
    values = torch.cat((context.backbone_features.mean(dim=1), context.state_features.mean(dim=1)), dim=-1).detach()
    if values.ndim != 2 or not torch.isfinite(values).all():
        raise ValueError('FINITE_OBSERVATION_FEATURES_REQUIRED')
    return values.float()


class SteeringMLP(nn.Module):
    def __init__(self, feature_dim, space, *, hidden_dims=(512, 512, 512)):
        super().__init__()
        if type(feature_dim) is not int or feature_dim <= 0 or not hidden_dims or any(type(n) is not int or n <= 0 for n in hidden_dims):
            raise ValueError('EXPLICIT_POSITIVE_STEERING_DIMENSIONS_REQUIRED')
        self.space, self.feature_dim = space, feature_dim
        layers, width = [], feature_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.GELU()])
            width = hidden
        layers.append(nn.Linear(width, space.horizon*space.padded_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        if features.ndim != 2 or features.shape[-1] != self.feature_dim or not torch.isfinite(features).all():
            raise ValueError('EXACT_FINITE_STEERING_FEATURE_SHAPE_REQUIRED')
        return self.net(features.detach()).reshape(-1, self.space.horizon, self.space.padded_dim)


def latent_loss(prediction, inverted_target):
    if prediction.shape != inverted_target.shape or prediction.ndim != 3:
        raise ValueError('EXACT_MODEL_LATENT_SHAPE_REQUIRED')
    if not torch.isfinite(prediction).all() or not torch.isfinite(inverted_target).all():
        raise ValueError('NONFINITE_LATENT_LOSS_REFUSED')
    return (prediction-inverted_target.detach()).square().mean()
