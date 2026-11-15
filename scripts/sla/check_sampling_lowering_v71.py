"""Lower the full SLA20 graph using abstract parameters; never execute it.

The unchanged production timer is stopped at its compile boundary after real
JAX lowering. Only package metadata, a tokenizer and synthetic RGB are read.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np


class StopBeforeCompile(Exception):
    pass


def run(helper, manifests, pins):
    if os.environ.get('JAX_PLATFORMS') != 'cpu' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise PermissionError('EXPLICIT_CPU_ONLY_ENVIRONMENT_REQUIRED')
    import flax.nnx as nnx
    from flax.traverse_util import unflatten_dict
    import jax
    import jax.numpy as jnp
    from go2arx5_vla.sla_adapter import model_v1
    from go2arx5_vla.sla_adapter.config_v1 import SEED, trainable
    from go2arx5_vla.sla_adapter.dataset_v1 import images224
    from go2arx5_vla.sla_adapter.provider_v1 import active_rows
    from openpi_client.image_tools import resize_with_pad

    assert jax.__version__ == '0.5.3' and all(d.platform == 'cpu' for d in jax.devices())
    assert len(manifests) == len(pins) == 2
    packages = []
    for path, pin in zip(manifests, pins, strict=True):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == pin
        package = json.loads(path.read_bytes())
        assert (package['step'], package['cursor'], package['sampler_steps']) == (500, 16000, 10)
        packages.append(package)
    assert packages[0]['parent_manifest'] == packages[1]['parent_manifest']
    started, cpu = time.time(), time.process_time()
    print(json.dumps(dict(phase='ABSTRACT_FULL_ARCHITECTURE_BEGIN')), flush=True)
    abstract = nnx.eval_shape(model_v1.configuration().create, jax.random.PRNGKey(SEED))
    graph, state = nnx.split(abstract)
    values = {path: jax.ShapeDtypeStruct(value.shape, model_v1.placement_dtype_name(path))
              for path, value in model_v1.flat(state.to_pure_dict()).items()}
    state.replace_by_pure_dict(unflatten_dict(values))
    graph, active, frozen = nnx.split(nnx.merge(graph, state),
        nnx.All(nnx.Param, lambda path, _: trainable(path)), ...)
    parameter_leaves = jax.tree_util.tree_leaves((active, frozen))
    assert all(isinstance(value, jax.ShapeDtypeStruct) for value in parameter_leaves)
    assert sum(math.prod(value.shape) for value in parameter_leaves) > 1_000_000_000
    for package in packages:
        rows, _ = active_rows(dict(arrays=package['adapter_arrays']), active)
        assert len(rows) == 28
    print(json.dumps(dict(phase='ABSTRACT_METADATA_MATCHED_BOTH_PACKAGES', active_leaves=28)), flush=True)

    tokenizer_path = manifests[0].parent/'paligemma_tokenizer.model'
    assert hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() == packages[0]['tokenizer_sha256']
    tokenizer = model_v1.Tokenizer(tokenizer_path)
    item = images224([np.zeros((256, 256, 3), np.uint8),
                      np.full((256, 256, 3), 127, np.uint8)], resize_with_pad)
    item.update(state=np.linspace(-.2, .2, 32, dtype=np.float32),
                actions=np.zeros((40, 32), np.float32), prompt='pick up the object')
    noise = jnp.zeros((1, 40, 32), jnp.float32)
    function = jax.jit(lambda a, f, obs, z: nnx.merge(graph, a, f).sample_actions(
        jax.random.PRNGKey(0), obs, noise=z, num_steps=20))
    captured = {}

    class CompileGuard:
        def __init__(self, lowered):
            self.lowered = lowered

        def compile(self):
            # Do not call the executable compiler or return an executable.
            text = self.lowered.as_text()
            output = self.lowered.out_info
            assert tuple(output.shape) == (1, 40, 32) and str(output.dtype) == 'float32'
            assert 'stablehlo.' in text
            captured.update(output_shape=list(output.shape), output_dtype=str(output.dtype),
                stablehlo_sha256=hashlib.sha256(text.encode()).hexdigest(),
                stablehlo_bytes=len(text.encode()), stablehlo_while_operations=text.count('stablehlo.while'))
            raise StopBeforeCompile

    class LowerOnlyJax:
        """Only the timer's jit entry is intercepted; real JAX stays unchanged."""
        def __init__(self):
            self.device_put = jax.device_put

        def __getattr__(self, name):
            return getattr(jax, name)

        def jit(self, wrapped):
            real_function = jax.jit(wrapped)

            class Boundary:
                def lower(self, *args):
                    return CompileGuard(real_function.lower(*args))

            return Boundary()

    spec = importlib.util.spec_from_file_location('sla_full_graph_timer', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    facade = LowerOnlyJax()
    timer = module.SamplingTimer(function, jax_module=facade, model_module=model_v1)
    original_observation, original_put = model_v1.observation, jax.device_put
    try:
        with timer.prediction():
            observation, _ = model_v1.observation(item, tokenizer)
            timer(active, frozen, observation, noise)
    except StopBeforeCompile:
        pass
    else:
        raise AssertionError('FULL_MODEL_CHECK_MUST_STOP_BEFORE_COMPILE')
    assert captured and timer.submitted == timer.completed == timer.compilations == 0
    assert timer.compiled is None
    assert model_v1.observation is original_observation and jax.device_put is original_put
    assert all(isinstance(v, jax.ShapeDtypeStruct) for v in jax.tree_util.tree_leaves((active, frozen)))
    return dict(schema='SLA20_FULL_ARCHITECTURE_ABSTRACT_LOWER_ONLY_V71', passed=True,
        helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(), package_manifest_sha256=pins,
        model_class=type(abstract).__module__+'.'+type(abstract).__name__, sampling_steps=20,
        active_leaves=28, frozen_leaves=len(jax.tree_util.tree_leaves(frozen)),
        abstract_parameter_numel=sum(math.prod(v.shape) for v in parameter_leaves),
        abstract_parameter_dtype_counts=dict(Counter(str(v.dtype) for v in parameter_leaves)),
        parameter_values_materialized=False, original_type_checks_disabled=False,
        synthetic_RGB=True, matched_both_active_package_metadata=True,
        **captured, sampling_executable_compilations=0, sampling_executable_submissions=0,
        sampling_executable_completions=0, SLA_parent_weight_loads=0, SLA_generations=0,
        optimizer_updates=0, training_visits=0, physical_requests=0, physics_integrations=0,
        GPU_model_qualification=False, numerical_output_equivalence_proven=False,
        CPU_s=time.process_time()-cpu, wall_s=time.time()-started,
        process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--helper', type=Path, required=True)
    parser.add_argument('--package-manifest', type=Path, action='append', required=True)
    parser.add_argument('--package-sha', action='append', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.helper, args.package_manifest, args.package_sha)), flush=True)
