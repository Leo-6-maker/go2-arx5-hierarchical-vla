"""Expose active/padding diagnostics for the pinned upstream FM equations.

Adapted from openpi/models/pi0.py, Spatial-Forcing commit 405ca887, Apache-2.0.
The network, time/noise draws, composition and 32-D reduction are unchanged.
The explicit train=False preprocessing implements this protocol's no-augment
recipe. Qualification compares this loss and its gradients to upstream.
"""
def error32(model, rng, observation, actions):
    import jax
    import jax.numpy as jnp
    from openpi.models import model as upstream
    from openpi.models.pi0 import make_attn_mask
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = upstream.preprocess_observation(preprocess_rng, observation, train=False)
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2])*.999+.001
    t = time[..., None, None]
    x_t, u_t = t*noise+(1-t)*actions, noise-actions
    prefix, prefix_mask, prefix_ar = model.embed_prefix(observation)
    suffix, suffix_mask, suffix_ar, condition = model.embed_suffix(observation, x_t, time)
    masks = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    attention = make_attn_mask(masks, jnp.concatenate([prefix_ar, suffix_ar], axis=0))
    positions = jnp.cumsum(masks, axis=1)-1
    (_, output), _ = model.PaliGemma.llm([prefix, suffix], mask=attention, positions=positions,
                                       adarms_cond=[None, condition])
    prediction = model.action_out_proj(output[:, -model.action_horizon:])
    return jnp.square(prediction-u_t)


def loss(model, rng, observation, actions):
    import jax.numpy as jnp
    error = error32(model, rng, observation, actions)
    return jnp.mean(jnp.mean(error, axis=-1)), (jnp.mean(error[..., :7]), jnp.mean(error[..., 7:]))
