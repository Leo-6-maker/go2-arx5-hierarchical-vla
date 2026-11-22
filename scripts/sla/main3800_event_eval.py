"""Evaluate the fixed TRAIN/VAL event roster on qualified Core Go2 endpoints."""
from argparse import ArgumentParser
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from scripts.pi0292.durable import OwnerLock, write
from scripts.sla.core_canonical_package import validate
from scripts.sla.core_main3800_native_proof import load_manifest
from scripts.sla.hopper_parent_custody import digest, write_once
from scripts.sla.hopper_site import pinned
from scripts.sla.main3800_auxiliary_readback import ARMS, verify_package
from scripts.sla.main3800_event_panel import SEED, TRAIN052


def observation(row):
    path = Path(row['episode_pin']['path'])
    ep = pinned(dict(path=str(path),sha256=row['episode_pin']['sha256']))
    index = row['anchor']
    if ep['id'] != row['episode'] or ep['native_result_sha256'] != row['result']['sha256']:
        raise ValueError('Panel and native episode differ')
    shard, = (s for s in ep['frame_shards'] if s['first_frame'] <= index < s['first_frame']+s['frames'])
    labels = path.parent.parent/ep['labels']['file']
    image = path.parent.parent/shard['file']
    if digest(labels) != row['label_sha256'] or digest(image) != shard['sha256']:
        raise ValueError('Original body/wrist or label evidence changed')
    with np.load(labels,allow_pickle=False) as stream:
        state, target = (stream[k][index] for k in ('state_eef10','nominal_policy_H40'))
        previous = stream['previous_actual_command'][index]
        step = stream['native_step'][index]
    with np.load(image,allow_pickle=False) as stream:
        local = index-shard['first_frame']
        rgb = stream['rgb'][local]
        if step != row['tick'] or stream['native_step'][local] != step or not np.array_equal(stream['state_eef10'][local],state):
            raise ValueError('RGB and native state do not share the admitted tick')
    native = Path(row['result']['path']).parent
    pinned(row['result'])
    live = native/'native-01/LIVE_DATA'
    capture = json.loads((live/'MANIFEST.private.json').read_bytes())
    # A migrated trace can still resolve the original data receipt by hash.
    query_file = live/Path(capture['labels']['path']).name
    query = pinned(dict(path=str(query_file),sha256=capture['labels']['sha256']))[index]
    spec = json.loads((native/'EXECUTION_SPEC.json').read_bytes())
    frame = np.asarray(spec['policy_from_world'])
    mount = np.asarray(ep['mount']['T_policy_from_mount'])
    previous = dict(zip(('width_m','source_native_s','query_native_s'),map(float,previous),strict=True))
    if (query['native_step'] != step or not np.array_equal(query['state_eef10'],state)
            or query['previous_confirmed_command'] != previous
            or not np.array_equal(query['episode_fixed_T_policy_from_mount'],mount)
            or not query['recent_reference']['exact_receiver_knot']):
        raise ValueError('Actual previous command, reference or native state differs')
    return state,rgb,ep['language'],mount,previous,frame,query['recent_reference']['accepted_pose7'],target


def measures(output, inputs, tag):
    from scripts.wm02.mfree_c20_query_audit_v3 import pose7
    from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze

    state,_,_,_,previous,frame,reference,target = inputs
    plan = output['physical_H40']
    certified = analyze(plan,state,frame,pose7(reference))
    grip = output['G7_gripper_speed']
    legal = certified['original_full_H40_pass'] and not grip['original_execution_speed_exceeded']
    up = np.linalg.inv(frame)[2,:3]
    actual_direction = target[9,:3]-state[:3]
    length = np.linalg.norm(actual_direction)
    displacement = plan[9,:3]-state[:3]

    def crossing(trajectory, closing):
        timeline = np.r_[previous['width_m'],.002+.040*trajectory[:,9]]
        opened = timeline >= .022
        hit = np.flatnonzero(opened[:-1] & ~opened[1:] if closing else ~opened[:-1] & opened[1:])
        return int(hit[0]) if len(hit) else None

    direction_close = tag == 'P10_CLOSE_CROSSING'
    reference_event = crossing(target,direction_close) if direction_close or tag == 'RELEASE' else None
    predicted_event = crossing(plan,direction_close) if direction_close or tag == 'RELEASE' else None
    due = reference_event is not None and reference_event < 10
    timely = due and predicted_event is not None and predicted_event < 10 and abs(predicted_event-reference_event) <= 2
    return dict(H40=certified['original_full_H40_pass'],G7_speed=not grip['original_execution_speed_exceeded'],
                legal=legal,first_H40_exceedance=certified['first_temporal_exceedance'],
                seam_mm_s=1000*grip['first_seam_speed_m_s'],maximum_finger_mm_s=1000*grip['maximum_speed_m_s'],
                expert_crossing=reference_event,model_crossing=predicted_event,due=due,
                timely_and_legal=bool(legal and timely) if due else None,
                expert_up_P10_mm=float(1000*actual_direction@up),model_up_P10_mm=float(1000*displacement@up),
                aligned_P10_mm=float(1000*displacement@actual_direction/length) if length>0 else None)


def endpoint_packages(cfg, comparison):
    if comparison is not None:
        export = pinned(comparison)
        if (export['status'] != 'MATCHED_PAIR_RECONCILED' or export['matched_FM_updates'] != 500
                or set(export['arms']) != set(ARMS)):
            raise ValueError('The two completed auxiliary arms must be verified together')
        parent = Path(comparison['path']).parent
        packages = []
        for arm in ARMS:
            receipt = export['arms'][arm]
            folder = parent/arm
            manifest = verify_package(folder,receipt['manifest_sha256'])
            if (manifest['arm'] != arm or manifest['checkpoint_sha256'] != receipt['training_state_sha256']
                    or manifest['training_result_sha256'] != receipt['result_sha256']):
                raise ValueError('Core inference package does not match the reconciled training endpoint')
            packages.append((f'{arm}-phase04300',folder,manifest))
        return packages
    original = Path(cfg['historic_run'])/'endpoints'
    arms = ('MAIN-phase00000','MAIN-phase02000')
    pkgs = [pinned(dict(path=str(original/(arm+'.json')),sha256=digest(original/(arm+'.json'))))['gate']['package']
            for arm in arms]
    packages = [(arm,Path(p['path']).resolve(strict=True),validate(Path(p['path']).resolve(strict=True),p['sha256']))
                for arm,p in zip(arms,pkgs,strict=True)]
    path = Path(cfg['package']['path']).resolve(strict=True)
    return [*packages,('MAIN-phase03800',path,load_manifest(cfg['package'],pkgs[0]))]


def evaluate(root, config, panel):
    from scripts.sla.core_hopper_native_queue import capacity

    cfg = pinned(config['native_config'])
    expert = pinned(cfg['expert'])
    receiver = pinned(cfg['receiver'])
    if (os.getuid()!=expert['uid'] or os.uname().nodename!=expert['host'] or not os.environ.get('INVOCATION_ID')
            or os.environ['CUDA_VISIBLE_DEVICES']!=receiver['gpu']) or panel['training_manifest']!=TRAIN052 or panel['seed']!=SEED:
        raise PermissionError('Expected the actual isolated Core evaluation resource and frozen event population')
    if {k:len(v) for k,v in panel['roster'].items()} != {'TRAIN':24,'VAL':16}:
        raise ValueError('Expected the sealed 24 TRAIN / 16 independent VAL windows')
    native = Path(cfg['root'])
    cfg['root'] = str(root)
    while time.time()+1800 < config['deadline'] and not (root/'STOP').exists():
        closed = (native/'TERMINAL.json').is_file() or (native/'REVIEW_REQUIRED.json').is_file()
        if closed and capacity(cfg,expert):
            try:
                lease = OwnerLock(receiver['gpu_lock'])
                break
            except BlockingIOError:
                pass
        write(root/'HEARTBEAT.json',dict(epoch=time.time(),state='WAIT_NATIVE_RELEASE' if not closed else 'WAIT_ORIGINAL_RESOURCES'))
        time.sleep(30)
    else:
        write_once(root/'TERMINAL.json',dict(status='NO_CAPACITY_OR_STOP',generations=0,native_requests=0,updates=0))
        return
    with closing(lease):
        import jax
        from flax import nnx
        from openpi_client.image_tools import resize_with_pad
        from go2arx5_vla.sla_adapter import scale20k_model, scale20k_package
        from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
        from go2arx5_vla.sla_adapter.model_v1 import Tokenizer
        from go2arx5_vla.sla_adapter.scale20k_contract import canonical_hash
        from scripts.sla.hopper_reference import TargetProvider

        if jax.default_backend()!='gpu' or len(jax.local_devices())!=1 or 'RTX 4080 SUPER' not in jax.local_devices()[0].device_kind:
            raise PermissionError('Original Core RTX numerical backend required')
        packages = endpoint_packages(cfg,config.get('auxiliary_export'))
        inputs = [(split,row,observation(row)) for split,rows in panel['roster'].items() for row in rows]
        source = pinned(receiver['input'])['target_D1']
        data = AdaptTargetDataset(source['legacy'],source['captures'],resize_with_pad=resize_with_pad)
        data.bind_state(source['statistics'])
        binding = receiver['model']
        graphs,weights,frozen,identity = scale20k_model.load(binding['parent_manifest'],binding['parent_params'])
        template = {key:weights[key] for key in ('shared','target')}
        tokenizer = Path(receiver['tokenizer']['path'])
        if any(m['frozen_model']!=identity['frozen_fingerprint']
               or m['partition_sha256']!=canonical_hash(identity['inventory'])
               or m['foundation_sha256']!=binding['parent_manifest']['sha256']
               or m['state_sha256']!=source['statistics']['sha256']
               or m['action_sha256']!=receiver['action']['sha256']
               or m['tokenizer_sha256']!=digest(tokenizer) for _,_,m in packages):
            raise ValueError('The original model, target statistics or tokenizer changed')
        model = None
        totals = {}
        for arm,folder,manifest in packages:
            scope = root/arm
            scope.mkdir()
            weights = scale20k_package.read_weights(folder,manifest['arrays'],template)
            if model is None:
                model = TargetProvider(graphs['target'],frozen,weights,Tokenizer(str(tokenizer)),data.statistics)
            else:
                model.active = nnx.State.merge(weights['shared'],weights['target'])
            rows = []
            for i,(split,row,values) in enumerate(inputs):
                if time.time()>=config['deadline'] or (root/'STOP').exists():
                    raise TimeoutError('Bounded offline evaluation stopped; existing attempts are retained')
                seed = int.from_bytes(hashlib.sha256(f'{SEED}/{split}/{i}'.encode()).digest()[:4],'little')
                state,rgb,language,mount,previous,*_ = values
                write_once(scope/f'{i:03d}.intent.json',dict(ordinal=i,split=split,tag=row['tag'],noise_id=seed,checkpoint=manifest['checkpoint_sha256']))
                start = time.monotonic()
                output = model.predict(state,rgb,language,mount,noise_id=seed,previous_reference=previous)
                write_once(scope/f'{i:03d}.generated.json',dict(ordinal=i,seconds=time.monotonic()-start))
                result = dict(ordinal=i,split=split,tag=row['tag'],**measures(output,values,row['tag']))
                write_once(scope/f'{i:03d}.result.json',result)
                rows.append(result)
            totals[arm] = rows
        predictions = len(packages)*len(inputs)
        write_once(root/'TERMINAL.json',dict(status='COMPLETE',attempts=predictions,generations=predictions,native_requests=0,
                   updates=0,visits=0,rows=totals,epoch=time.time()))


if __name__ == '__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('root',type=Path)
    cli.add_argument('config_sha')
    args = cli.parse_args()
    os.umask(0o077)
    try:
        cfg = pinned(dict(path=str(args.root/'CONFIG.json'),sha256=args.config_sha))
        evaluate(args.root,cfg,pinned(cfg['panel']))
    except Exception as exc:
        write_once(args.root/'REVIEW_REQUIRED.json',dict(reason=f'{type(exc).__name__}: {exc}',
            attempts=len(list(args.root.glob('*/*.intent.json'))),generations=len(list(args.root.glob('*/*.generated.json'))),
            native_requests=0,updates=0,visits=0,epoch=time.time()))
        raise
