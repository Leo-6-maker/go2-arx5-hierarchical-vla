"""Cold inference-package export or a separate process using ordinary RGB."""
import json
from pathlib import Path
import sys
import time
import numpy as np

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.model_v1 import configuration
from go2arx5_vla.sla_adapter.config_v1 import trainable
from go2arx5_vla.sla_adapter.provider_v1 import export, Provider
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin, data, probe_indices


def run(spec):
    import flax.nnx as nnx
    import jax
    R = Path(spec['registered_root'])
    if spec['operation'] == 'EXPORT':
        checkpoint = Path(spec['resume']['path'])
        if sha(checkpoint/'COMPLETE.json') != spec['resume']['sha256']:
            raise ValueError('COLD_EXPORT_CHECKPOINT_CHANGED')
        audit = read_pin(spec['target_audit'])
        template = nnx.eval_shape(configuration().create, jax.random.PRNGKey(0))
        _, active, _ = nnx.split(template, nnx.All(nnx.Param, lambda path, _: trainable(path)), ...)
        package = R/'inference-package'
        value = export(package, checkpoint, active_template=active,
                       statistics=Path(spec['target_audit']['path']).parent/'STATISTICS.private.json',
                       tokenizer=spec['tokenizer']['path'], parent_manifest=spec['parent_manifest'],
                       target_audit=spec['target_audit'])
        return dict(schema='SLA_COLD_INFERENCE_EXPORT_ACTUAL_V1', step=value['step'],
                    package=dict(path=str(package), sha256=sha(package/'MANIFEST.private.json')),
                    deployed_adapter_arrays=len(value['adapter_arrays']), tokenizer_saved=True, statistics_saved=True,
                    model_weights_loaded=0, optimizer_updates=0, physical_requests=0)
    if spec['operation'] != 'RECONSTRUCT_PACKAGE':
        raise ValueError('COLD_EXPORT_OR_FRESH_PACKAGE_RECONSTRUCTION_REQUIRED')
    pin = spec['package']
    provider = Provider(pin['path'], manifest_sha256=pin['sha256'], parent_params=spec['parent_params'])
    dataset = data(spec)
    checkpoint = Path(spec['resume']['path'])
    complete = json.loads((checkpoint/'COMPLETE.json').read_bytes())
    if sha(checkpoint/'COMPLETE.json') != spec['resume']['sha256']:
        raise ValueError('ORDINARY_RGB_REFERENCE_CHECKPOINT_CHANGED')
    reference = checkpoint.parent/f"step{complete['step']}-RGB"
    indices = probe_indices(dataset)
    original = json.loads((reference/'MANIFEST.private.json').read_bytes())
    if [(r['index'], r['noise']) for r in original] != [(i, n) for i in indices for n in range(2)]:
        raise ValueError('SAME_FIXED_TRAIN_RGB_AND_NOISE_REFERENCE_REQUIRED')
    R.joinpath('queries').mkdir()
    records = []
    for ordinal, row in enumerate(original):
        state, _, images, text = dataset.original.raw(row['index'])
        identity = dataset.original.identity(row['index'])
        frame = dataset.frames[identity['episode_id']]['T_policy_from_mount']
        request = dict(index=row['index'], noise=row['noise'], ordinal=ordinal)
        R.joinpath('queries', f'{ordinal:02d}.intent.json').write_text(json.dumps(request))
        output = provider.predict(state, images, text, frame, noise_id=row['noise'])
        path = R/'queries'/f'{ordinal:02d}.npz'
        np.savez(path, normalized=output['normalized'], physical_H40=output['physical_H40'])
        with np.load(reference/row['file'], allow_pickle=False) as expected:
            exact = all(np.array_equal(expected[k], output[k]) for k in ('normalized', 'physical_H40'))
        records.append(dict(**request, bit_exact=exact, file=path.name, sha256=sha(path)))
        R.joinpath('PROGRESS.private.json').write_text(json.dumps(dict(completed_generations=len(records), updates=0, physical_requests=0)))
        if not exact:
            raise ValueError('FRESH_ORDINARY_RGB_PACKAGE_NOT_BIT_EXACT:'+str(request))
    return dict(schema='SLA_FRESH_ORDINARY_RGB_PACKAGE_ACTUAL_V1', passes=True,
                step=complete['step'], package=pin, records=records, actual_parent_loads=1,
                ordinary_RGB_generations=len(records), optimizer_loaded=False, training_cache_used=False,
                teacher_loaded=False, deployed_LoRA_retained=True, optimizer_updates=0, physical_requests=0,
                frozen_fingerprint=provider.inventory['frozen_fingerprint'],
                GPU_memory=jax.devices()[0].memory_stats(), epoch=time.time())


if __name__ == '__main__':
    spec = json.loads(Path(sys.argv[1]).read_bytes())
    value = run(spec)
    (Path(spec['registered_root'])/'RESULT.private.json').write_text(json.dumps(value, indent=2))
    print(json.dumps(value), flush=True)
