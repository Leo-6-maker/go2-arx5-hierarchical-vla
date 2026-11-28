"""Go2-only V7-A paths using the V6.5 model, loss and optimizer conventions."""

from v7_previous_command import Go2ActionCondition


PROJECTION_KEY = "previous_executed_gripper_projection"


def extend_target_io(parameters):
    """Keep the projection in the existing Go2 optimizer and checkpoint partition."""
    from flax import nnx
    import jax.numpy as jnp

    target = parameters["target"]
    if PROJECTION_KEY in target:
        raise ValueError("The candidate must start from an unmodified R4 target IO tree")
    leaf = nnx.Param(jnp.zeros((1, 1024), dtype=jnp.float32)).to_state()
    return dict(parameters, target=nnx.State.merge(target, nnx.State({PROJECTION_KEY: leaf})))


def conditioned_policy(graph, shared, target, frozen, condition):
    from flax import nnx

    context, original_target = target.split(lambda path, _: path[0] == PROJECTION_KEY, ...)
    model = nnx.merge(graph, shared, original_target, frozen)
    return Go2ActionCondition(model, condition, context[PROJECTION_KEY].value)


def target_flow_loss(graph):
    import jax
    import jax.numpy as jnp
    from go2arx5_vla.sla_adapter.batch_v2 import error32

    def loss(active, frozen, observation, actions, noise, tau, support, condition):
        actions = jnp.where(support[..., None], actions, 0.)
        with jax.default_matmul_precision("highest"):
            policy = conditioned_policy(graph, active["shared"], active["private"], frozen, condition)
            error = error32(policy, observation, actions, noise, tau)
        valid = jnp.where(support[..., None], error, 0.)
        return jnp.mean(jnp.sum(valid, axis=(-2, -1)) / (32 * jnp.sum(support, axis=-1)))

    return jax.jit(jax.value_and_grad(loss))


def deployment_h40(graph, shared, target, frozen, observation, noise, condition, rng):
    policy = conditioned_policy(graph, shared, target, frozen, condition)
    return policy.sample_actions(rng, observation, num_steps=10, noise=noise)


def auxiliary_h40(graph, fields):
    """Keep the original differentiable cached 10-step solver and explicit noise."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def sample(active, frozen, observation, noise, condition):
        with jax.default_matmul_precision("highest"):
            policy = conditioned_policy(graph, active["shared"], active["private"], frozen, condition)
            prefix = fields["prefix_field"](policy, observation)

            def step(carry, _):
                value, tau = carry
                field = fields["cache_field"](policy, *prefix, value, tau)
                return (value + (-1.0 / 10) * field, tau + (-1.0 / 10)), None

            body = jax.checkpoint(step, prevent_cse=False)
            (value, _), _ = jax.lax.scan(body, (noise, jnp.asarray(1., jnp.float32)), xs=None, length=10)
            return value

    return sample


def target_auxiliary(graph, quantiles):
    import jax
    import jax.numpy as jnp
    from go2arx5_vla.sla_adapter import batch_v2, dense_g7_v6
    from go2arx5_vla.sla_adapter.grasp_supervision_d5 import width_losses
    from openpi.models import pi0
    from scripts.sla.mfree_c20_vectorfield_v3 import extracted_fields

    fields, _ = extracted_fields(batch_v2, pi0)
    sample = auxiliary_h40(graph, fields)
    lo, hi = (jnp.asarray(getattr(quantiles, key), jnp.float32) for key in ("q01", "q99"))

    def loss(active, frozen, observation, noise, previous_width, label_width, condition):
        prediction = sample(active, frozen, observation, noise, condition)
        g7, _ = dense_g7_v6.full_objective(prediction, lo, hi, previous_width, xp=jnp, sigmoid=jax.nn.sigmoid)
        latent = .5 * (prediction[..., 6] + 1) * (hi[6] - lo[6] + 1e-6) + lo[6]
        physical_width = .002 + .040 * jax.nn.sigmoid(latent)
        width, _ = width_losses(physical_width, label_width, previous_width, xp=jnp)
        return g7 + width

    return jax.jit(jax.value_and_grad(loss))
