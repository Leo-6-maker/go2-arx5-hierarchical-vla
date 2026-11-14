"""Export a complete dense pilot and replay ordinary RGB in a fresh process."""
import argparse
import hashlib
import json
from pathlib import Path


def run(spec, mode, *, engine_class=None, export_function=None, probe_selector=None):
    from go2arx5_vla.sla_adapter.site_v2 import validate
    validate(spec)
    from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
    from go2arx5_vla.sla_adapter.provider_v3 import Provider
    if engine_class is None:
        from go2arx5_vla.sla_adapter.runtime_v4 import DensePilotEngine
        engine_class = DensePilotEngine
    import jax, flax, optax
    root = Path(spec['registered_root'])
    if mode == 'export':
        if export_function is None:
            from go2arx5_vla.sla_adapter.provider_v4 import export
            export_function = export
        engine = engine_class(spec, root/'work')
        engine.bindings['training_site'] = dict(GPU_uuid=spec['GPU_uuid'], GPU_name=spec['GPU_name'],
                                              jax=jax.__version__, flax=flax.__version__, optax=optax.__version__)
        engine.resume(spec['resume'])
        package = root/'RGB_PACKAGE'
        export_function(package, spec['resume']['path'], active_template=engine.active, dataset=spec['dataset'],
               tokenizer=spec['tokenizer']['path'], parent_manifest=spec['parent_manifest'])
        result = dict(package=dict(path=str(package), sha256=sha(package/'MANIFEST.private.json')),
                      actual_parent_loads=1, optimizer_updates=0, ordinary_RGB_generations=0,
                      physical_requests=0, exit_code=0)
        engine.close()
        return result
    jax.config.update('jax_compilation_cache_dir', spec['compilation_cache_dir'])
    counts = dict(parent_load_attempts=1, actual_parent_loads=0,
                  ordinary_RGB_generation_attempts=0, ordinary_RGB_generations=0)
    from scripts.pi0292.durable import write
    write(root/'RGB_PROGRESS.private.json', counts)
    provider = Provider(spec['package']['path'], manifest_sha256=spec['package']['sha256'],
                        parent_params=spec['parent_params'], parent_manifest_path=spec['parent_manifest']['path'])
    counts['actual_parent_loads'] = 1
    write(root/'RGB_PROGRESS.private.json', counts)
    # The reader supplies only current state, RGB, prompt and mount to predict.
    dataset = engine_class.__new__(engine_class).dataset_for_spec(spec)
    from go2arx5_vla.sla_adapter.runtime_v1 import probe_indices
    rows = []
    for index in (probe_selector or probe_indices)(dataset):
        state, _, images, text = dataset.raw(index)
        frame = dataset.frames[dataset.identity(index)['episode_id']]['T_policy_from_mount']
        for noise in range(2):
            counts['ordinary_RGB_generation_attempts'] += 1
            write(root/'RGB_PROGRESS.private.json', counts)
            value = provider.predict(state, images, text, frame, noise_id=noise)
            counts['ordinary_RGB_generations'] += 1
            write(root/'RGB_PROGRESS.private.json', counts)
            rows.append(dict(index=index, noise=noise,
                normalized_sha256=hashlib.sha256(value['normalized'].tobytes()).hexdigest(),
                physical_sha256=hashlib.sha256(value['physical_H40'].tobytes()).hexdigest()))
    reference = json.loads(Path(spec['reference_probes']).read_bytes())
    if (len(rows) != len(reference) or len(rows) != 16 or not all(
            all(a[k] == b[k] for k in ('index', 'noise', 'normalized_sha256', 'physical_sha256'))
            for a, b in zip(rows, reference))):
        raise ValueError('FRESH_NAMED_PILOT_RGB_DIFFERS_FROM_COMPLETE_ENDPOINT')
    return dict(passes=True, ordinary_RGB_generations=16, all_bit_exact=True, records=rows,
                actual_parent_loads=1, teacher_loaded=False, optimizer_loaded=False,
                optimizer_updates=0, physical_requests=0, exit_code=0,
                GPU_memory=jax.devices()[0].memory_stats())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path)
    parser.add_argument('--mode', choices=['export', 'replay'], required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    result = run(spec, args.mode)
    with (Path(spec['registered_root'])/(args.mode.upper()+'_RESULT.private.json')).open('x') as stream:
        json.dump(result, stream, indent=2)
