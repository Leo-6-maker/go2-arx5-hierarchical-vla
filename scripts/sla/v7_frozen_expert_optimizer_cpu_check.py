"""CPU optimizer-mechanics checks on synthetic arrays, never on model weights."""

import hashlib
import json
from pathlib import Path
import sys
from tempfile import mkdtemp

import numpy as np

from v7_frozen_expert_optimizer import ACTIVE, GROUPS, SCOPE, make_frozen_expert_adamw


def run(work):
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp
    from go2arx5_vla.sla_adapter.scale20k_contract import make_adamw
    import go2arx5_vla.sla_adapter.scale20k_contract as original
    from v7_frozen_expert_checkpoint import validate_optimizer_clock

    assert jax.default_backend() == "cpu"
    output = Path(mkdtemp(prefix="v7-frozen-expert-optimizer-cpu.", dir=work))
    hyper = dict(beta1=.9, beta2=.999, eps=1e-8, weight_decay=.1, clip_norm=1.)
    initialize, update = make_frozen_expert_adamw(**hyper)
    original_init, original_update = make_adamw(**hyper)
    names = ("shared", "target", "robodojo", "libero")

    def fixture():
        return {key: dict(kernel=jnp.asarray([[1.+i, -2.-i], [3., .5]], dtype=jnp.float32),
                          bias=jnp.asarray([.2, -.4], dtype=jnp.float32)) for i, key in enumerate(names)}

    def gradient(step):
        return {key: dict(kernel=jnp.full((2,2), 1e6 if key == "shared" else .5*(i+1)+step*.1),
                          bias=jnp.asarray([.3,-.1])) for i, key in enumerate(names)}

    def equal(a, b):
        return jax.tree.structure(a) == jax.tree.structure(b) and all(
            np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True))

    parameters = fixture()
    state = initialize(parameters)
    origin = jax.tree.map(lambda x: np.asarray(x).copy(), parameters["shared"])
    shared_state = state["shared"]
    records = []
    for step in range(5):
        g = gradient(step)
        active = {key: value for key, value in g.items() if key in ACTIVE}
        expected_norm = float(optax.global_norm(active))
        assert expected_norm > hyper["clip_norm"]
        oracle_gradient = dict(g, shared=jax.tree.map(jnp.zeros_like, g["shared"]))
        expected_parameters, expected_state, expected_readback = original_update(parameters, state, oracle_gradient, .01)
        actual_parameters, actual_state, norm = update(parameters, state, g, .01)
        assert norm == expected_norm == expected_readback
        assert equal({key:actual_parameters[key] for key in ACTIVE}, {key:expected_parameters[key] for key in ACTIVE})
        assert equal({key:actual_state[key] for key in ACTIVE}, {key:expected_state[key] for key in ACTIVE})
        assert equal(actual_parameters["shared"], origin) and actual_state["shared"] is shared_state
        assert not equal(expected_parameters["shared"], origin), "The nonzero decay must expose a naive zero-gradient freeze"
        assert {key:int(s[0].count) for key,s in actual_state.items()} == {key:0 if key == "shared" else step+1 for key in GROUPS}
        records.append(dict(synthetic_step=step+1, active_gradient_l2=norm,
                            clip_factor=hyper["clip_norm"]/norm, private_matches_original=True,
                            frozen_shared_and_adam_unchanged=True))
        parameters, state = actual_parameters, actual_state
    snapshot = dict(parameters=parameters, optimizer=state, rng=jax.random.PRNGKey(529202611), step=5,
                    domain_cursors=dict(target=480, robodojo=80, libero=80))
    endpoint = output / "SYNTHETIC_STEP0005"
    with ocp.PyTreeCheckpointer() as writer:
        writer.save(endpoint.resolve(), snapshot)
    template = dict(parameters=fixture(), optimizer=initialize(fixture()), rng=jax.random.PRNGKey(0),step=0,
                    domain_cursors=dict(target=0,robodojo=0,libero=0))
    with ocp.PyTreeCheckpointer() as reader:
        recovered = reader.restore(endpoint.resolve(), item=template)
    assert equal(snapshot,recovered)
    assert validate_optimizer_clock(recovered["optimizer"],5) == {key:0 if key == "shared" else 5 for key in names}
    counter_errors = {}
    bad_states = {"dense_shared_clock":dict(state,shared=expected_state["shared"])}
    bad_shared = state["shared"][0]._replace(mu=jax.tree.map(jnp.ones_like,state["shared"][0].mu))
    bad_states["frozen_nonzero_moments"] = dict(state,shared=(bad_shared,*state["shared"][1:]))
    for name,bad in bad_states.items():
        try:
            validate_optimizer_clock(bad,5)
        except ValueError as exc:
            counter_errors[name] = str(exc)
        else:
            raise AssertionError("Required checkpoint-clock rejection did not occur: "+name)
    next_g = gradient(5)
    hot = update(parameters,state,next_g,.007)
    cold = update(recovered["parameters"],recovered["optimizer"],next_g,.007)
    assert equal(hot,cold)
    negatives = {}
    for name, bad in (("missing_domain", {k:v for k,v in next_g.items() if k != "libero"}),
                      ("nonfinite_frozen_gradient",dict(next_g,shared=dict(kernel=jnp.full((2,2),jnp.nan),bias=jnp.zeros(2)))),
                      ("zero_active_gradient",jax.tree.map(jnp.zeros_like,next_g))):
        try:
            update(parameters,state,bad,.007)
        except ValueError as exc:
            negatives[name] = str(exc)
        else:
            raise AssertionError("Required rejection did not occur: "+name)
    result = dict(status="SYNTHETIC_CPU_OPTIMIZER_MECHANICS_PASS_NOT_MODEL_QUALIFICATION", scope=SCOPE,
        original_optimizer_module_sha256=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
        optimizer_module_sha256=hashlib.sha256(Path(sys.modules[make_frozen_expert_adamw.__module__].__file__).read_bytes()).hexdigest(),
        synthetic_fixture_shapes={key:{n:list(v.shape) for n,v in group.items()} for key,group in parameters.items()},
        checks=records, nonzero_weight_decay_tested=True, shared_gradient_magnitude=1e6,
        shared_count_after_5=0, private_counts_after_5={key:int(state[key][0].count) for key in ACTIVE},
        synthetic_snapshot_restore_equal=True, synthetic_next_update_equal=True, rejected=negatives,
        frozen_checkpoint_clock_and_moments_rejections=counter_errors,
        synthetic_frozen_optimizer_calls=7, synthetic_original_oracle_calls=5,
        actual_model_optimizer_updates=0, formal_updates=0, engineering_model_updates=0,
        model_forwards=0, physics_integrations=0, full_model_and_H200_qualification=False)
    out = output / "ACTUAL.json"
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(out,hashlib.sha256(out.read_bytes()).hexdigest(),flush=True)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
