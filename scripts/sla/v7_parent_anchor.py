"""Add a shared-expert R4 parameter anchor before the existing global clip."""
import math


class SharedParentAnchor:
    def __init__(self, parent):
        import jax
        import jax.numpy as jnp

        self.parent = jax.tree.map(lambda value: jnp.array(value, copy=True), parent)

        @jax.jit
        def value_and_gradient(shared, origin):
            difference = jax.tree.map(jnp.subtract, shared, origin)
            loss = sum(jnp.vdot(value, value).real for value in jax.tree.leaves(difference))
            return loss, jax.tree.map(lambda value: 2 * value, difference)

        self.compute = value_and_gradient
        self.last = None

    def value_and_gradient(self, shared):
        return self.compute(shared, self.parent)

    def wrap(self, original, coefficient):
        import jax
        import optax

        coefficient = float(coefficient)
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError("Parent-anchor coefficient must be finite and nonnegative")

        def apply(parameters, optimizer, gradients, rate):
            loss, penalty = self.value_and_gradient(parameters["shared"])
            norm = float(optax.global_norm(penalty))
            self.last = dict(raw_loss=float(loss), coefficient=coefficient,
                             weighted_loss=coefficient * float(loss),
                             weighted_gradient_l2=coefficient * norm)
            if coefficient == 0:
                return original(parameters, optimizer, gradients, rate)
            shared = jax.tree.map(lambda gradient, regularizer: gradient + coefficient * regularizer,
                                  gradients["shared"], penalty)
            augmented = dict(gradients, shared=shared)
            return original(parameters, optimizer, augmented, rate)

        return apply
