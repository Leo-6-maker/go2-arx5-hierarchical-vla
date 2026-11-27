"""Go2 scalar condition for V7-A, independent of the state tokenizer and action codec."""

import jax.numpy as jnp
import numpy as np


MIN_WIDTH_M = .002
MAX_WIDTH_M = .042
PRECEDING_INTERVAL_TICKS = 25


def encode_previous_width(width_m, query_tick, consumed_start_tick, command_valid, state_kept, *, treatment):
    """Use the measured consumption clock and the same state-drop mask in both arms."""
    width, query, previous, valid, retained = map(
        np.asarray, (width_m, query_tick, consumed_start_tick, command_valid, state_kept))
    if not (width.ndim == 1 and width.shape == query.shape == previous.shape == valid.shape == retained.shape
            and np.issubdtype(query.dtype, np.integer) and np.issubdtype(previous.dtype, np.integer)
            and valid.dtype == retained.dtype == np.bool_ and (query >= 0).all()):
        raise ValueError("Malformed previous-command context batch")
    verified = width[valid]
    if not ((previous[valid] >= 0).all()
            and (previous[valid] + PRECEDING_INTERVAL_TICKS == query[valid]).all()
            and np.isfinite(verified).all()
            and (verified >= MIN_WIDTH_M - 1e-12).all()
            and (verified <= MAX_WIDTH_M + 1e-12).all()):
        raise ValueError("Expected the actual preceding complete 50 ms command in physical metres")
    visible = valid & retained if treatment else np.zeros_like(valid)
    condition = np.zeros(width.shape, dtype=np.float32)
    condition[visible] = 2 * (width[visible] - MIN_WIDTH_M) / (MAX_WIDTH_M - MIN_WIDTH_M) - 1
    return condition[:, None]


def condition_action_tokens(action_tokens, condition, projection):
    """Add the same bias-free projection to every Go2 action token."""
    if not (action_tokens.ndim == 3 and condition.shape == (action_tokens.shape[0], 1)
            and projection.shape == (1, action_tokens.shape[-1])):
        raise ValueError("Incompatible Go2 action-token condition")
    residual = (condition @ projection).astype(action_tokens.dtype)
    return action_tokens + jnp.expand_dims(residual, axis=1)
