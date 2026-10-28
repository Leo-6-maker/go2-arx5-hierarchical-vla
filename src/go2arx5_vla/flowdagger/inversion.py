"""Bounded tensor-only adaptation of Microsoft's pinned FlowDAgger primitives.

Adapted from groot_inversion.py at 3426f246510fd67a9e39c3c08bd5b02d586ad688.
Copyright (c) Microsoft Corporation. MIT; see third_party/flowdagger/LICENSE.
No policy loading, environment, data collection, optimizer or run entry here.
"""
import torch


def _tensor(value, shape=None):
    if (not torch.is_tensor(value) or value.ndim != 3 or not value.is_floating_point()
            or any(n <= 0 for n in value.shape)
            or shape is not None and value.shape != shape or not torch.isfinite(value).all()):
        raise ValueError('FINITE_MATCHED_BATCH_HORIZON_CHANNEL_TENSOR_REQUIRED')
    return value


class CountedVelocity:
    """Charge before every evaluation, including failed and inversion calls."""
    def __init__(self, function, *, limit):
        if type(limit) is not int or limit <= 0:
            raise ValueError('EXPLICIT_POSITIVE_VELOCITY_EVALUATION_BUDGET_REQUIRED')
        self.function, self.limit, self.calls = function, limit, 0

    def __call__(self, value, time):
        _tensor(value)
        if self.calls >= self.limit:
            raise PermissionError('FLOW_VELOCITY_BUDGET_EXHAUSTED')
        self.calls += 1
        if not 0 <= time < 1:
            raise ValueError('ORIGINAL_FORWARD_TIME_GRID_REQUIRED')
        result = _tensor(self.function(value, time), value.shape)
        if result.dtype != value.dtype or result.device != value.device:
            raise ValueError('VELOCITY_MUST_PRESERVE_NATIVE_DEVICE_AND_DTYPE')
        return result


def _steps(steps):
    if type(steps) is not int or steps <= 0:
        raise ValueError('POSITIVE_EXPLICIT_SAMPLER_STEPS_REQUIRED')


@torch.no_grad()
def forward_euler(initial, velocity, *, steps):
    _steps(steps)
    value = _tensor(initial).clone()
    for k in range(steps):
        value = value + (1.0/steps)*velocity(value, k/steps)
    return _tensor(value)


@torch.no_grad()
def perstep_fp_reverse(target, velocity, *, steps, fp_per_step):
    _steps(steps)
    if type(fp_per_step) is not int or not 0 <= fp_per_step <= 100:
        raise ValueError('EXPLICIT_BOUNDED_FIXED_POINT_ITERATIONS_REQUIRED')
    value = _tensor(target).clone()
    for k in range(steps-1, -1, -1):
        next_value = value
        previous = next_value-(1.0/steps)*velocity(next_value, k/steps)
        for _ in range(fp_per_step):
            previous = next_value-(1.0/steps)*velocity(previous, k/steps)
        value = previous
    return _tensor(value)


def round_trip(target, velocity, *, steps, fp_per_step, active_dim):
    """Return evidence, not a scientific PASS; finite inverse may still be poor."""
    _tensor(target)
    if type(active_dim) is not int or not 0 < active_dim <= target.shape[-1]:
        raise ValueError('VERIFIED_ACTIVE_CHANNEL_COUNT_REQUIRED')
    before = velocity.calls
    noise = perstep_fp_reverse(target, velocity, steps=steps, fp_per_step=fp_per_step)
    reconstructed = forward_euler(noise, velocity, steps=steps)
    error = (reconstructed.double()-target.double()).square()
    return dict(noise=noise, reconstructed=reconstructed,
        active_normalized_MSE=float(error[..., :active_dim].mean()),
        full_padded_normalized_MSE=float(error.mean()),
        velocity_evaluations=velocity.calls-before,
        expected_velocity_evaluations=steps*(fp_per_step+2),
        convergence_qualified=False, requires_physical_decode_and_original_safety=True)
