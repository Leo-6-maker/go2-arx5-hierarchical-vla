"""Measure auxiliary shared-gradient alignment without modifying a trained endpoint."""
from argparse import ArgumentParser
from contextlib import closing
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import runpy
import time


CLOSURE_ORDINALS = (1, 10, 19)
AUX_SHA = {
    'target_auxiliary': 'd9d91280d2321396ba1aa68b86d3e7999a5b74e79282f6fa6640031da7e7e8b5',
    'grasp_supervision_d5': 'd57ca466026c4ae998507edc4a3a8b327ba37f7c3fcbc8a547afa0498bd04601',
    'dense_g7_v6': 'c9f35039a1fa43371bdaac49a6879f66d539030977d150aa055f5eb62c2a195a',
    'full_h40_v6': '8917492c2d65c771bdc0ea52b70f4dc2c60f26c63230bacf363bdc5583ad5eef',
}


def read(path, want):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != want:
        raise ValueError('Changed audit specification')
    return json.loads(raw)


def pairs(panel, source_root):
    import numpy as np
    from scripts.sla.hopper_site import pinned
    from scripts.sla.main3800_event_eval import observation

    samples = []
    for ordinal in CLOSURE_ORDINALS:
        first = panel['roster']['TRAIN'][ordinal]
        assert first['tag'] == 'P10_CLOSE_CROSSING'
        native_event = panel['episode_events'][first['result']['sha256']]
        episode = pinned(first['episode_pin'])
        path = Path(first['episode_pin']['path'])
        label = path.parent.parent/episode['labels']['file']
        with np.load(label,allow_pickle=False) as data:
            steps = data['native_step']
            anchor = int(np.searchsorted(steps,native_event['events']['target_grasp'],side='left'))
            assert data['consumed_mask'][first['anchor']][0] and data['consumed_mask'][anchor][0]
            after = dict(first,tag='FIRST_REQUEST_AFTER_GRASP',anchor=anchor,tick=int(steps[anchor]))
            assert first['anchor'] < anchor < len(steps)
        seen = json.loads((source_root/'A-phase04300'/f'{ordinal:03d}.intent.json').read_bytes())
        assert seen['ordinal'] == ordinal and seen['tag'] == first['tag'] and seen['split'] == 'TRAIN'
        samples.append((dict(ordinal=ordinal,episode=first['episode'],layout=first['layout'],
                             indices=[first['anchor'],after['anchor']],ticks=[first['tick'],after['tick']],
                             noise=seen['noise_id']),observation(first),observation(after)))
    assert len({p[0]['layout'] for p in samples}) == len(CLOSURE_ORDINALS)
    return samples


def alignment(left, right):
    import numpy as np

    dot = math.fsum(float(np.vdot(a,b)) for a,b in zip(left,right,strict=True))
    a = math.sqrt(math.fsum(float(np.vdot(x,x)) for x in left))
    b = math.sqrt(math.fsum(float(np.vdot(x,x)) for x in right))
    if not math.isfinite(dot+a+b) or a == 0 or b == 0:
        raise ValueError('Shared-gradient comparison has a zero norm or nonfinite value')
    return dict(cosine=dot/(a*b),dot=dot,close_norm=a,post_grasp_norm=b)


def audit(root, spec):
    from scripts.pi0292.durable import OwnerLock, write
    from scripts.sla.core_hopper_native_queue import capacity
    from scripts.sla.hopper_parent_custody import digest, write_once
    from scripts.sla.hopper_site import pinned
    from scripts.sla.main3800_cross_layout_proof import matched_rows
    from scripts.sla.main3800_event_eval import endpoint_packages
    from go2arx5_vla.sla_adapter import dense_g7_v6, full_h40_v6, grasp_supervision_d5, scale20k_training

    subjects = dict(target_auxiliary=hashlib.sha256(inspect.getsource(scale20k_training.target_auxiliary).encode()).hexdigest())
    for item in (dense_g7_v6,full_h40_v6,grasp_supervision_d5):
        subjects[item.__name__.rsplit('.',1)[-1]] = digest(Path(item.__file__).resolve())
    if subjects != AUX_SHA:
        raise ValueError('The original Hopper auxiliary objective differs in this environment')
    original = pinned(spec['offline_config'])
    offline = pinned(spec['offline_terminal'])
    native = pinned(spec['native_terminal'])
    if (offline['status']!='COMPLETE' or offline['generations']!=80 or native['status']!='COMPLETE'
            or native['native_requests']!=12 or time.time()>=spec['deadline']):
        raise ValueError('This audit is bounded to the closed matched A/B result')
    cfg = pinned(original['native_config'])
    expert = pinned(cfg['expert'])
    receiver = pinned(cfg['receiver'])
    if os.getuid()!=expert['uid'] or os.uname().nodename!=expert['host'] or not os.environ.get('INVOCATION_ID'):
        raise PermissionError('Expected the registered Corelab diagnostic service')
    cfg['root'] = str(root)
    while time.time()+300 < spec['deadline'] and not (root/'STOP').exists():
        if capacity(cfg,expert) is not None:
            try:
                lease = OwnerLock(receiver['gpu_lock'])
                break
            except BlockingIOError:
                pass
        write(root/'HEARTBEAT.json',dict(state='WAIT_ORIGINAL_RESOURCES',epoch=time.time(),pid=os.getpid()))
        time.sleep(20)
    else:
        write_once(root/'TERMINAL.json',dict(status='ORIGINAL_RESOURCES_UNAVAILABLE',
                   gradient_attempts=0,completed_gradients=0,model_generations=0,native_requests=0,updates=0))
        return
    with closing(lease):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from openpi_client.image_tools import resize_with_pad
        from go2arx5_vla.sla_adapter import scale20k_model, scale20k_package
        from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
        from go2arx5_vla.sla_adapter.codec_v1 import pad32
        from go2arx5_vla.sla_adapter.dataset_v1 import images224
        from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec
        from go2arx5_vla.sla_adapter.model_v1 import Tokenizer, observation
        from go2arx5_vla.sla_adapter.scale20k_contract import canonical_hash

        if jax.default_backend()!='gpu' or len(jax.local_devices())!=1 or 'RTX 4080 SUPER' not in jax.local_devices()[0].device_kind:
            raise PermissionError('Expected the original Core NVIDIA resource')
        panel = pinned(original['panel'])
        offline_root = Path(spec['offline_config']['path']).parent
        packages = endpoint_packages(cfg,original['auxiliary_export'])
        for arm,_,manifest in packages:
            matched_rows(offline_root,offline,panel,manifest['checkpoint_sha256'],arm)
        samples = pairs(panel,offline_root)
        write_once(root/'PAIRS.private.json',dict(pairs=[item[0] for item in samples],split='TRAIN',
                                                 differentiated_H40_cap=12,optimizer_updates=0))
        source = pinned(receiver['input'])['target_D1']
        if source['statistics']['sha256']!=receiver['model']['state_sha256']:
            raise ValueError('The original state statistics changed')
        data = AdaptTargetDataset(source['legacy'],source['captures'],resize_with_pad=resize_with_pad)
        data.bind_state(source['statistics'])
        tok = Path(receiver['tokenizer']['path'])
        assert digest(tok) == receiver['tokenizer']['sha256']
        tokenizer = Tokenizer(str(tok))
        graphs,weights,frozen,model = scale20k_model.load(receiver['model']['parent_manifest'],receiver['model']['parent_params'])
        template = {k:weights[k] for k in ('shared','target')}
        del weights
        gradient = scale20k_training.target_auxiliary(graphs['target'],data.statistics['actions'])
        results = {}
        for arm,path,manifest in packages:
            if (manifest['action_sha256']!=receiver['action']['sha256']
                    or manifest['state_sha256']!=source['statistics']['sha256']
                    or manifest['tokenizer_sha256']!=receiver['tokenizer']['sha256']
                    or manifest['partition_sha256']!=canonical_hash(model['inventory'])
                    or manifest['frozen_model']!=model['frozen_fingerprint']):
                raise ValueError('Changed endpoint, statistics or frozen model')
            weights = scale20k_package.read_weights(path,manifest['arrays'],template)
            active = dict(shared=weights['shared'],private=weights['target'])
            results[arm] = []
            for register,close,after in samples:
                derivatives = []
                values = []
                noise = jax.random.normal(jax.random.PRNGKey(register['noise']),(1,40,32))
                for label,inputs in (('close',close),('post_grasp',after)):
                    if time.time()>=spec['deadline'] or (root/'STOP').exists():
                        raise TimeoutError('The bounded gradient diagnostic has stopped')
                    state,images,prompt,mount,previous,_,_,target = inputs
                    codec = GripLogitCodec(mount)
                    state7,_ = codec.encode(state,target)
                    item = images224(images,resize_with_pad)
                    item.update(state=pad32(data.statistics['state'].normalize(state7)).astype(np.float32),
                                actions=np.zeros((40,32),np.float32),prompt=prompt)
                    obs,_ = observation(item,tokenizer)
                    width = dense_g7_v6.previous_command_context(previous)
                    suffix = f"{arm}-{register['ordinal']}-{label}"
                    write_once(root/f'{suffix}.intent.json',dict(arm=arm,checkpoint=manifest['checkpoint_sha256'],
                                                               sample=register['ordinal'],phase=label))
                    value,grads = gradient(active,frozen,obs,noise,jnp.asarray([width],jnp.float32),
                                           jnp.asarray((.002+.040*target[:,9])[None,:],jnp.float32))
                    jax.block_until_ready((value,grads))
                    derivatives.append([np.asarray(a,dtype=np.float64) for a in jax.tree.leaves(grads['shared'])])
                    values.append(float(value))
                    write_once(root/f'{suffix}.complete.json',dict(auxiliary_loss=values[-1],updates=0))
                    del grads
                results[arm].append(dict(ordinal=register['ordinal'],aux_losses=values,**alignment(*derivatives)))
            del weights,active
        complete = len(list(root.glob('*.complete.json')))
        if complete != 12:
            raise ValueError('Expected the twelve fixed forward/backward evaluations')
        write_once(root/'TERMINAL.json',dict(status='COMPLETE',completed_gradients=complete,model_generations=complete,
                   native_requests=0,updates=0,results=results,epoch=time.time()))


if __name__ == '__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('root',type=Path)
    cli.add_argument('sha')
    arg = cli.parse_args()
    os.umask(0o077)
    spec = read(arg.root/'CONFIG.json',arg.sha)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=spec['source_sha256']:
        raise ValueError('Changed gradient audit source')
    n = spec['numeric_entry']
    if hashlib.sha256(Path(n['path']).read_bytes()).hexdigest()!=n['sha256']:
        raise ValueError('Changed frozen Core launcher')
    binder = runpy.run_path(n['path'])
    binder['bind_runtime'](binder['read_verified'](spec['numeric_binding']))
    try:
        audit(arg.root,spec)
    except Exception as exc:
        from scripts.sla.hopper_parent_custody import write_once
        write_once(arg.root/'REVIEW_REQUIRED.json',dict(reason=f'{type(exc).__name__}: {exc}',
            gradient_attempts=len(list(arg.root.glob('*.intent.json'))),completed_gradients=len(list(arg.root.glob('*.complete.json'))),
            native_requests=0,updates=0,epoch=time.time()))
        raise
