"""Separate PD loss; no existing V6 engine or inference path imports this.

The FM network/draws follow batch_v2 and Spatial-Forcing405ca887 (Apache-2.0).
Only supervision_mask weights learning. Consumption and camera masks have
separate meanings and are deliberately absent from this API. Partial targets
are sanitized before noising; their action keys cannot enter attention at any
layer. Original token positions and all32 learned action dimensions remain.
Real SLA gradient/update qualification is required before training admission.
"""
import numpy as np

VERSION = 'SLA_SUPERVISION_MASKED_PLAN_FM32_V6_1'


def validate_host_batch(actions, supervision_mask):
    """Reject empty/invalid supervised rows before any device/model work.

    Invalid slots may contain NaN or arbitrary values: none is allowed into
    the flow input. This does not make an illegal H40 deployment-admissible.
    """
    actions, mask = np.asarray(actions), np.asarray(supervision_mask)
    if (actions.ndim != 3 or actions.shape[1:] != (40, 32)
            or actions.dtype != np.float32 or mask.dtype != np.bool_
            or mask.shape != actions.shape[:2] or not len(actions)):
        raise ValueError('PD_FP32_BATCH_H40_32_AND_BOOLEAN_SUPERVISION_MASK_REQUIRED')
    if not mask.any(axis=1).all():
        raise ValueError('ALL_FALSE_SUPERVISION_ROW_REJECTED_BEFORE_LOSS')
    if not np.isfinite(actions[mask]).all():
        raise ValueError('FINITE_SUPERVISED_TARGETS_REQUIRED')
    return mask


def _device_mask(actions, supervision_mask):
    import jax.numpy as jnp
    mask = jnp.asarray(supervision_mask)
    if (actions.ndim != 3 or actions.shape[1:] != (40, 32)
            or mask.shape != actions.shape[:2] or mask.dtype != jnp.bool_):
        raise ValueError('PD_H40_32_AND_BOOLEAN_SUPERVISION_MASK_REQUIRED')
    return mask


def error32(model, observation, actions, noise, flow_time, supervision_mask):
    """The pinned FM forward, with no unsupervised target in flow/attention."""
    import jax.numpy as jnp
    from openpi.models import model as upstream
    from openpi.models.pi0 import make_attn_mask
    supervision = _device_mask(actions, supervision_mask)
    observation = upstream.preprocess_observation(None, observation, train=False)
    safe_actions = jnp.where(supervision[..., None], actions, jnp.zeros_like(actions))
    expanded = flow_time[..., None, None]
    x_t = expanded * noise + (1 - expanded) * safe_actions
    target = noise - safe_actions
    prefix, prefix_mask, prefix_ar = model.embed_prefix(observation)
    suffix, suffix_mask, suffix_ar, condition = model.embed_suffix(observation, x_t, flow_time)
    present = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    attention = make_attn_mask(present, jnp.concatenate([prefix_ar, suffix_ar], axis=0))
    # Keep the original positions/query rows. Removing only invalid action
    # keys prevents content propagation without compressing the H40 clock or
    # creating all-masked query rows. The state token, when present, stays.
    allowed_suffix_keys = suffix_mask.at[:, -model.action_horizon:].set(
        suffix_mask[:, -model.action_horizon:] & supervision)
    allowed_keys = jnp.concatenate([prefix_mask, allowed_suffix_keys], axis=1)
    attention = attention & allowed_keys[:, None, :]
    positions = jnp.cumsum(present, axis=1) - 1
    (_, output), _ = model.PaliGemma.llm([prefix, suffix], mask=attention, positions=positions,
                                       adarms_cond=[None, condition])
    predicted = model.action_out_proj(output[:, -model.action_horizon:])
    return jnp.square(predicted - target)


def reduce32(error, supervision_mask):
    """Per-row sum(error)/(32*supervised_knots), then equal row mean.

    The all-true branch uses the exact existing reduction and diagnostics.
    Empty rows are rejected on the host; NaN is a second, device-side refusal
    so a caller bypassing that check cannot silently commit a zero-loss row.
    """
    import jax
    import jax.numpy as jnp
    mask = _device_mask(error, supervision_mask)
    def dense(_):
        return jnp.mean(jnp.mean(error, axis=-1)), (jnp.mean(error[..., :7]), jnp.mean(error[..., 7:]))
    def partial(_):
        counts = jnp.sum(mask, axis=1)
        def mean_channels(values):
            selected = jnp.where(mask, jnp.mean(values, axis=-1), jnp.zeros_like(mask, dtype=values.dtype))
            row = jnp.sum(selected, axis=1) / jnp.maximum(counts, 1)
            return jnp.mean(jnp.where(counts > 0, row, jnp.nan))
        return mean_channels(error), (mean_channels(error[..., :7]), mean_channels(error[..., 7:]))
    return jax.lax.cond(jnp.all(mask), dense, partial, operand=None)


def loss(model, observation, actions, noise, flow_time, supervision_mask):
    return reduce32(error32(model, observation, actions, noise, flow_time, supervision_mask), supervision_mask)
