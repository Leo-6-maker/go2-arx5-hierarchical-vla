"""Private-IO AdamW with the original shared Expert excluded before global clipping."""

import math


SCOPE = "V7_R4_FROZEN_SHARED_EXPERT_ADAMW_V1"
GROUPS = frozenset(("shared", "target", "robodojo", "libero"))
ACTIVE = GROUPS - {"shared"}


def make_frozen_expert_adamw(*, beta1, beta2, eps, weight_decay, clip_norm):
    """Retain the complete state schema; never invoke AdamW on shared parameters.

    FM/auxiliary losses and their composition are unchanged. The single clip
    uses only trainable IO gradients. Shared moments and count stay unchanged,
    including when reading an existing complete optimizer state.
    """
    import jax
    import optax

    if not (0 <= beta1 < 1 and 0 <= beta2 < 1 and eps > 0 and weight_decay >= 0 and clip_norm > 0
            and all(math.isfinite(x) for x in (beta1, beta2, eps, weight_decay, clip_norm))):
        raise ValueError("FROZEN_EXPERT_INVALID_ADAMW_HYPERPARAMETERS")
    transform = optax.chain(optax.scale_by_adam(b1=beta1, b2=beta2, eps=eps),
                            optax.add_decayed_weights(weight_decay))

    @jax.jit
    def apply_partition(parameters, state, gradient, lr, scale):
        clipped = jax.tree.map(lambda leaf: leaf*scale, gradient)
        direction, next_state = transform.update(clipped, state, parameters)
        updates = jax.tree.map(lambda leaf: -lr*leaf, direction)
        return optax.apply_updates(parameters, updates), next_state

    def initialize(parameters):
        if set(parameters) != GROUPS:
            raise ValueError("FROZEN_EXPERT_WRONG_PARAMETER_PARTITIONS")
        return {key: transform.init(value) for key, value in parameters.items()}

    def apply(parameters, states, gradient, lr):
        if set(parameters) != GROUPS or set(states) != GROUPS or set(gradient) != GROUPS:
            raise ValueError("FROZEN_EXPERT_WRONG_ACTIVE_DOMAINS_OR_OPTIMIZER_PARTITION")
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("FROZEN_EXPERT_INVALID_LEARNING_RATE")
        # A frozen-gradient NaN is still an upstream error, not a valid masked
        # value. Exclusion affects clipping/update scope, not finite checks.
        raw_norm = float(optax.global_norm(gradient))
        active = {key: value for key, value in gradient.items() if key in ACTIVE}
        norm = float(optax.global_norm(active))
        if not math.isfinite(raw_norm) or not math.isfinite(norm) or norm <= 0:
            raise ValueError("FROZEN_EXPERT_NONFINITE_OR_ZERO_ACTIVE_GRADIENT")
        scale = min(1., clip_norm/norm)
        next_parameters, next_states = dict(parameters), dict(states)
        for key, value in active.items():
            next_parameters[key], next_states[key] = apply_partition(parameters[key], states[key], value, lr, scale)
        if next_parameters["shared"] is not parameters["shared"] or next_states["shared"] is not states["shared"]:
            raise AssertionError("FROZEN_EXPERT_SHARED_PARAMETER_OR_ADAM_STATE_REPLACED")
        return next_parameters, next_states, norm

    return initialize, apply
