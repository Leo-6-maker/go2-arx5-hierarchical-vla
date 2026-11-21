"""Private, nonintegrating phase-zero measurements. No endpoint or student queue writes."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np

CHECKPOINT = 'd7949c75ecc33253bea605a3aed738df4ecdc473d8737848a1bca787ab253c18'
PACKAGE = 'fdf58d6078f3b8259892e5f6d991c7737071b543254a64b865393063a85baa01'
PANEL = '3136404c06f890a9dd830b2cf5f40e487ef436ede16e805e14ffc0cb06b5eefd'
CORE_FAILURE = 'f934791b64896c1660d508242232c150003d0a6295d45e8b5eaba6a98fc87666'
ORDINALS = (0,7,14,21,28,35,36,43,50,57,64,71,72,79,86,93)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path, expected):
    buf = path.read_bytes()
    if hashlib.sha256(buf).hexdigest() != expected:
        raise ValueError('Pinned input changed: ' + path.name)
    return json.loads(buf)


def record(path, value):
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


def signature(value):
    a = np.asarray(value)
    return dict(dtype=str(a.dtype), shape=list(a.shape),
                sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())


def instrument(sampler):
    """Use the original bound graph and fields. Check the observed scan against production."""
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from go2arx5_vla.sla_adapter import full_h40_v6

    raw = inspect.unwrap(sampler)
    if inspect.getmodule(raw) != full_h40_v6:
        raise ValueError('Unexpected production sampler')
    binding = inspect.getclosurevars(raw).nonlocals
    if binding['precision'] != 'highest' or binding['rematerialize']:
        raise ValueError('Unexpected sampler precision or rematerialization')
    fields = binding['fields']

    @jax.jit
    def capture(active, frozen, observation, noise):
        with jax.default_matmul_precision('highest'):
            model = nnx.merge(binding['graph'], active, frozen)
            prefix = fields['prefix_field'](model, observation)
            def step(carry, _):
                value, tau = carry
                field = fields['cache_field'](model, *prefix, value, tau)
                value = value + (-1.0/10) * field
                return (value, tau + (-1.0/10)), value
            (final, _), flow = jax.lax.scan(step, (noise, jnp.asarray(1., jnp.float32)), None, length=10)
            return final, flow, prefix[1], prefix[2], prefix[3]
    return capture


def load(site, cfg):
    from scripts.sla.hopper_reference import core_provider, TargetProvider, package
    from scripts.sla.hopper_site import pinned
    if site == 'core':
        from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
        from openpi_client.image_tools import resize_with_pad
        src = pin(Path(cfg['receiver_config']), cfg['receiver_sha256'])
        slot = Path(cfg['package']).parent
        historical = pin(slot/'RESULT.json', CORE_FAILURE)
        if historical['exact'] != 0 or historical['checkpoint'] != CHECKPOINT:
            raise ValueError('Historical strict result changed')
        target = pinned(src['input'])['target_D1']
        data = AdaptTargetDataset(target['legacy'], target['captures'], resize_with_pad=resize_with_pad)
        data.bind_state(target['statistics'])
        panel = pin(Path(src['panel']['path']), PANEL)
        provider, manifest = core_provider(Path(cfg['package']), PACKAGE, src['model'], data, src['tokenizer']['path'])
        local = {n: json.loads((slot/f'{n:03d}.result.json').read_bytes()) for n in ORDINALS}
    else:
        from scripts.sla.hopper_formal_outputs import load_endpoint
        from scripts.sla.hopper_inputs import PARENT, PLAN
        from scripts.sla.hopper_panels import FixedPanels
        view = Path(cfg['view'])
        lineage = Path(cfg['lineage'])
        event = pin(lineage/'MAIN/sealed/phase00000.json', cfg['phase0_event_sha256'])
        manifest = package(Path(cfg['package']), PACKAGE)
        panels = FixedPanels(view/'assets.verified', Path(cfg['eval_assets']))
        panel, data = panels.formal, panels.datasets['target']
        inputs = dict(original_parent=dict(path=str(view/'parent.verified/COMPLETE.json'),sha256=PARENT),
                      train_assets=str(view/'assets.verified'),training_source=manifest['training_source'])
        receipt = dict(custody=event['checkpoint']['path'], checkpoint=CHECKPOINT, plan=PLAN,
                       parent=PARENT, role='MAIN', phase=0, arrays=140)
        graphs, state, frozen = load_endpoint(inputs, receipt)
        provider = TargetProvider(graphs['target'], frozen, state['parameters'], panels.tokenizer, data.statistics)
        local = {r['ordinal']: r for r in manifest['reference']}
    if (manifest['checkpoint_sha256'] != CHECKPOINT or manifest['panel'] != PANEL or
            tuple(panel['restore_ordinals']) != ORDINALS or panel['old_TEST_access']):
        raise ValueError('Checkpoint or fixed input panel changed')
    return provider, data, panel, local, manifest


def measure(out, provider, data, panel, local, manifest, ordinal, tracer):
    import jax
    from scripts.sla.hopper_reference import fingerprint
    from go2arx5_vla.sla_adapter.scale20k_contract import canonical_hash
    from go2arx5_vla.sla_adapter.scale20k_package import Provider
    from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze
    from go2arx5_vla.world_model.task_learning_executability_v3 import path_series
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
    from scripts.sla.replay_scale20k_core import guard_signature

    fixed = panel['formal']['target'][ordinal]
    index = fixed['index']
    ctx = panel['target_contexts'][str(index)]
    previous = data.reference(index)
    state, _, images, prompt = data.raw(index)
    frame = data.codecs[fixed['identity']['episode_id']].T_policy_from_mount
    raw = canonical_hash(dict(window=data.identity(index), state=fingerprint(state), prompt=prompt,
        body_wrist=[fingerprint(v) for v in images], frame=fingerprint(frame), previous=previous))
    expected = next(r for r in manifest['reference'] if r['ordinal']==ordinal)
    if (data.identity(index)!=fixed['identity'] or ctx['previous_command']!=previous or
            (raw,index,fixed['noise_id']) != (expected['input_sha256'],expected['index'],expected['noise_id'])):
        raise ValueError('Raw observation, window or previous actual command changed')
    obsig, private = {}, {}
    def capture(active, frozen, obs, noise):
        for field in ('image','image_mask','image_padding_mask','state','tokenized_prompt','tokenized_prompt_mask'):
            value = getattr(obs, field)
            items = value.items() if isinstance(value,dict) else [('',value)]
            for name, a in items:
                obsig[field+('/'+name if name else '')] = signature(a)
        obsig['noise'] = signature(noise)
        obsig['key'] = signature(jax.random.PRNGKey(fixed['noise_id']))
        private['noise'] = np.asarray(noise)
        if tracer is None:
            return provider.infer(active,frozen,obs,noise)
        final, steps, tokens, mask, cache = tracer(active,frozen,obs,noise)
        private['flow'] = np.asarray(steps)
        private['prefix_tokens'] = np.asarray(tokens,dtype=np.float32)
        obsig['prefix_tokens_native'] = signature(tokens)
        obsig['prefix_mask'] = signature(mask)
        for i,a in enumerate(jax.tree.leaves(cache)):
            # Full cache hashes and lossless FP32 representations for numerical comparison.
            obsig['prefix_cache_'+str(i)] = signature(a)
            private['prefix_cache_'+str(i)] = np.asarray(a,dtype=np.float32)
        return final
    proxy = SimpleNamespace(active=provider.active,frozen=provider.frozen,resize=provider.resize,
                            tokenizer=provider.tokenizer,statistics=provider.statistics,infer=capture)
    stem = f'{ordinal:03d}'
    record(out/(stem+'.intent.json'),dict(ordinal=ordinal,noise_id=fixed['noise_id'],input_sha256=raw))
    actual = Provider.predict(proxy,state,images,prompt,frame,noise_id=fixed['noise_id'],previous_reference=previous)
    physical = actual['physical_H40']
    normalized = actual['normalized'][0]
    envelope = analyze(physical,state,np.asarray(ctx['policy_from_world']),np.asarray(ctx['reference']))
    seam = actual['G7_gripper_speed']
    sig = guard_signature(envelope,seam)
    series = path_series(physical)
    private.update(normalized=normalized,physical=physical,rotations=rot6d_rows_to_matrix(physical[:,3:9]),
                   width_m=series['working_gripper_width_m'],previous_width_m=previous['width_m'])
    for channel,v in series.items():
        if isinstance(v,dict) and 'vectors' in v:
            private[channel] = np.asarray(v['vectors'])
    artifact = out/(stem+'.private.npz')
    with artifact.open('xb') as f:
        np.savez_compressed(f,**private)
        f.flush()
        os.fsync(f.fileno())
    digests = dict(normalized_sha256=signature(normalized)['sha256'],decoded_sha256=signature(physical)['sha256'],
                   original_guard_sha256=canonical_hash(sig))
    record(out/(stem+'.json'),dict(ordinal=ordinal,index=index,noise_id=fixed['noise_id'],input_sha256=raw,
        tensors=obsig,outputs=digests,own_v1={k:v==local[ordinal][k] for k,v in digests.items()},
        guard=sig,envelope=envelope,seam=seam,artifact_sha256=sha(artifact)))


def run(root, cfg, mode):
    import jax
    from importlib.metadata import version
    gpu = jax.local_devices()
    if not (jax.default_backend()=='gpu' and len(gpu)==1 and cfg['device'] in gpu[0].device_kind):
        raise PermissionError('Wrong local GPU')
    if not cfg['site'] in ('core','hopper') or os.getuid()!=cfg['uid'] or time.time()>=cfg['deadline']:
        raise PermissionError('Expired or unbound diagnostic')
    if cfg['site']=='core':
        from scripts.pi0292.durable import OwnerLock
        if not os.environ.get('INVOCATION_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!=cfg['gpu']:
            raise PermissionError('Unexpected Core owner')
        # The registered service holds this lock across both cold processes.
        try:
            lock=OwnerLock(cfg['gpu_lock'])
        except BlockingIOError:
            pass
        else:
            lock.close()
            raise PermissionError('Expected an already fenced parent')
        if os.getppid()!=json.loads((root/'OWNER.json').read_bytes())['pid']:
            raise PermissionError('Not the registered child')
    else:
        award=json.loads((root/'AWARD.json').read_bytes())
        if os.environ.get('PBS_JOBID')!=award['job'] or os.uname().nodename.split('.')[0]!=award['host']:
            raise PermissionError('Not the registered PBS award')
    if mode=='trace2':
        prior=json.loads((root/'ordinary2'/'RESULT.json').read_bytes())
        if prior['completed']!=2 or not prior['own_v1_exact']:
            raise ValueError('Cold ordinary path differed from its historical bytes')
    out=root/mode
    out.mkdir()
    ordinals=ORDINALS if mode=='ordinary16' else ORDINALS[:2]
    info=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],
        '--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
    record(out/'ENV.json',dict(device=info,jax=jax.__version__,jaxlib=version('jaxlib'),
        gpu_plugin=version('jax-cuda12-plugin'),gpu_runtime=gpu[0].client.platform_version,
        prng=str(jax.config.jax_default_prng_impl),x64=jax.config.jax_enable_x64,
        key_dtype=str(jax.random.PRNGKey(0).dtype),sampler_precision='highest',
        env={k:os.environ.get(k) for k in ('XLA_FLAGS','OMP_NUM_THREADS','JAX_DEFAULT_PRNG_IMPL',
            'JAX_DEFAULT_MATMUL_PRECISION','JAX_COMPILATION_CACHE_DIR')},pid=os.getpid()))
    provider,data,panel,local,manifest=load(cfg['site'],cfg)
    tracer=instrument(provider.infer) if mode=='trace2' else None
    for ordinal in ordinals:
        if time.time()>=cfg['deadline'] or (root/'STOP').exists():
            raise TimeoutError('Diagnostic stopped at a fixed window boundary')
        measure(out,provider,data,panel,local,manifest,ordinal,tracer)
    rows=[json.loads((out/f'{n:03d}.json').read_bytes()) for n in ordinals]
    record(out/'RESULT.json',dict(completed=len(rows),own_v1_exact=all(all(r['own_v1'].values()) for r in rows)))


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('root',type=Path)
    cli.add_argument('config_sha256')
    cli.add_argument('mode',choices=('ordinary2','trace2','ordinary16'))
    args=cli.parse_args()
    os.umask(0o077)
    config=pin(args.root/'CONFIG.json',args.config_sha256)
    if sha(Path(__file__))!=config['worker_sha256'] or args.mode not in config['modes']:
        raise ValueError('Unregistered measurement or diagnostic source')
    run(args.root,config,args.mode)
