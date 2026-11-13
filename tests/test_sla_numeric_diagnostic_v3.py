import time
from types import SimpleNamespace
import numpy as np
import pytest


def test_fixed_inputs_are_reused_and_optimizer_rng_cursor_remain_bit_exact(tmp_path, monkeypatch):
    import jax
    import jax.numpy as jnp
    from go2arx5_vla.sla_adapter import numeric_diagnostic_v3 as diagnostic
    from go2arx5_vla.sla_adapter.checkpoint_v1 import save
    e = SimpleNamespace(step=0, cursor=0, active={'w': jnp.asarray([.3, .7])},
                        frozen={}, opt_state={'moment': jnp.zeros(2)}, rng=jax.random.PRNGKey(3),
                        counts=dict(optimizer_updates=0, forward_backward_attempts=0, completed_forward_backward=0),
                        dataset=None, tokenizer=None, visit_map=None, bindings={'test': 'CPU_NUMERIC_NOT_SLA'},
                        spec=dict(deadline_epoch=time.time() + 120))
    e.checkpoint = lambda name: save(tmp_path / name, (e.active, e.opt_state, e.rng),
                                     step=e.step, cursor=e.cursor, bindings=e.bindings)
    e.verify_frozen = lambda: 'UNCHANGED_EMPTY_TOY_FROZEN'
    e.draw_bank = lambda *args: (jnp.zeros((32, 40, 32)), jnp.zeros(32))
    def objective(active, frozen, obs, actions, noise, flow_time):
        value = jnp.mean((obs @ active['w'] - actions)**2)
        return value, (value, value * 0)
    e.grad = jax.jit(jax.value_and_grad(objective, has_aux=True))
    calls = []
    def host(*args, **kwargs):
        calls.append(args[4])
        return dict(observation=jnp.arange(64, dtype=jnp.float32).reshape(32, 2) / 64,
                    actions=jnp.ones(32), indices=list(range(32)), state_token_dropped=[False] * 32)
    monkeypatch.setattr(diagnostic, 'host_batch', host)
    monkeypatch.setattr(diagnostic, 'device_batch', lambda batch: (batch['observation'], batch['actions']))
    result = diagnostic.diagnose(e, tmp_path)
    assert calls == [32]
    assert result['optimizer_updates'] == result['optimization_visits'] == result['physical_requests'] == 0
    assert result['diagnostic_forward_backward_visits'] == 192
    assert result['diagnostic_forward_backward_calls'] == 76
    assert result['unchanged_full_state']['bit_exact']
    assert all(row['passes'] for row in result['comparisons'])
    assert not result['candidate_training_admitted']
    with pytest.raises(RuntimeError, match='OPTIMIZER_FORBIDDEN'):
        e.apply()


def test_noninitial_engine_cannot_be_used_for_comparison(tmp_path):
    from go2arx5_vla.sla_adapter.numeric_diagnostic_v3 import diagnose
    with pytest.raises(ValueError, match='UNMODIFIED_PARENT'):
        diagnose(SimpleNamespace(step=1, cursor=32, counts={'optimizer_updates': 1}), tmp_path)


def test_compiler_inspection_preserves_array_only_typed_pytree_validation():
    import jax
    import jax.numpy as jnp
    from go2arx5_vla.sla_adapter.numeric_diagnostic_v3 import gradient_ir
    @jax.tree_util.register_pytree_node_class
    class Observation:
        def __init__(self, images):
            if not isinstance(images, (jax.Array, jax.core.Tracer)):
                raise TypeError('TYPED_OBSERVATION_REQUIRES_REAL_ARRAY_OR_TRACER')
            self.images = images
        def tree_flatten(self):
            return (self.images,), None
        @classmethod
        def tree_unflatten(cls, metadata, leaves):
            return cls(leaves[0])
    grad = jax.jit(jax.value_and_grad(lambda w, obs: jnp.mean((obs.images @ w)**2)))
    args = (jnp.ones(2), Observation(jnp.ones((1, 2))))
    with pytest.raises(TypeError, match='TYPED_OBSERVATION'):
        grad.lower(*args)
    text = gradient_ir(grad, args)
    assert 'stablehlo.dot_general' in text
    assert float(grad(*args)[0]) == 4
