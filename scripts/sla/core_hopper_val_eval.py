"""Evaluate the fixed expert VAL captures on Core using received endpoint weights."""
import argparse
from contextlib import closing
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from dataclasses import dataclass

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.sla.hopper_core_receive import KEY
from scripts.sla.hopper_parent_custody import PLAN, digest, write_once
from scripts.sla.hopper_site import DEADLINE, pinned


def pin(path):
    return dict(path=str(path), sha256=digest(path))


@dataclass
class Capture:
    """One selected capture for the original phase metric's dataset interface."""
    previous: dict
    expert: np.ndarray

    def reference(self, index):
        if index != 0:
            raise IndexError(index)
        return self.previous

    def raw(self, index):
        if index != 0:
            raise IndexError(index)
        return None, self.expert, None, None


def windows(row, producer):
    from go2arx5_vla.sla_adapter.codec_v1 import rigid
    from go2arx5_vla.sla_adapter.multitask_data_v6 import reference
    from scripts.sla.export_mfree_c20_v1 import _reference

    eid = row['condition']
    folder = producer/'admission/VAL'/eid
    card_dir = producer/'cases'/eid
    episode = pinned(dict(path=str(folder/'EPISODE.private.json'), sha256=row['episode_sha256']))
    result = pinned(dict(path=str(card_dir/'RESULT.private.json'), sha256=row['native_sha256']))
    card_pin = pin(card_dir/'EXECUTION_SPEC.json')
    card = pinned(card_pin)
    manifest = pinned(result['detail']['capture'])
    queries = pinned(manifest['labels'])
    labels = folder/'LABELS.npz'
    if (episode['split'] != 'VAL' or episode['id'] != eid or not episode['individual_pass']
            or not result['detail']['task_success'] or episode['native_result_sha256'] != row['native_sha256']
            or result['source_sha'] != card['source_sha'] or result['condition']['collection_episode'] != eid
            or manifest['queries'] != episode['windows'] or len(queries) != episode['windows']
            or digest(labels) != row['labels_sha256'] or row['labels_sha256'] != episode['labels']['sha256']):
        raise ValueError('VAL capture, labels or terminal evidence changed')
    selected = row['selected']
    if len(selected) > 4 or len({w['index'] for w in selected}) != len(selected):
        raise ValueError('Repeated VAL selection')
    with np.load(labels, allow_pickle=False) as lab:
        for w in selected:
            index, frame, tick = w['index'], w['frame'], w['native_tick']
            packet = queries[index]
            pri = reference(lab['previous_actual_command'][index])
            if (lab['frame_index'][index] != frame or packet['query'] != index or packet['native_step'] != tick
                    or lab['native_step'][index] != tick or lab['phase'][index] != w['phase']
                    or packet['previous_confirmed_command'] != pri or packet['recent_reference']['native_step'] != tick
                    or not lab['supervision_mask'][index].all()
                    or not np.array_equal(np.flatnonzero(lab['consumed_mask'][index]), np.arange(10))):
                raise ValueError('Held-out frame, H40 mask or actual command clock differs')
            shard, = (s for s in episode['frame_shards'] if s['first_frame'] <= frame < s['first_frame'] + s['frames'])
            path = folder.parent/shard['file']
            if path.parent != folder or digest(path) != shard['sha256']:
                raise ValueError('Original body/wrist shard changed')
            offset = frame-shard['first_frame']
            state = lab['state_eef10'][index]
            with np.load(path, allow_pickle=False) as rgb:
                images = rgb['rgb'][offset]
                if (images.shape != (2,256,256,3) or images.dtype != np.uint8 or rgb['native_step'][offset] != tick
                        or not np.array_equal(rgb['state_eef10'][offset], state)):
                    raise ValueError('Original two views and EEF do not match the selected query')
            noise = hashlib.sha256(f'{PLAN}/VAL/{eid}/{tick}/{w["criterion"]}'.encode()).digest()
            identity = dict(condition=eid, criterion=w['criterion'], native_tick=tick, index=index, phase=w['phase'],
                            spec=card_pin['sha256'], rgb=shard['sha256'], labels=row['labels_sha256'],
                            source_queries=manifest['labels']['sha256'], noise_id=int.from_bytes(noise[:4], 'big'))
            yield (identity, state, images, episode['language'], rigid(episode['mount']['T_policy_from_mount']),
                   pri, lab['nominal_policy_H40'][index], rigid(card['policy_from_world']), rigid(_reference(packet)))


def load_panel(cfg):
    selector = pinned(cfg['selector'])
    path = Path(cfg['selector']['path']).parent/'PANEL.private.json'
    panel_pin = pin(path)
    panel = pinned(panel_pin)
    if (selector['plan'] != PLAN or panel['plan'] != PLAN or panel['split'] != 'VAL' or panel['old_TEST_read']
            or panel['selector_sha256'] != selector['worker_sha256'] or panel['selector'] != selector['source']
            or panel['population_sha256'] != cfg['population'] or panel['registered_attempts'] != 24
            or len(panel['conditions']) != 24 or len(panel['episodes']) != panel['admitted']
            or not 0 <= panel['selected_windows'] <= 96
            or sum(len(r['selected']) for r in panel['episodes']) != panel['selected_windows']):
        raise ValueError('Unbound VAL selection')
    expected = [r['condition'] for r in panel['conditions'] if r['status'] == 'ADMITTED']
    if expected != [r['condition'] for r in panel['episodes']] or len(set(expected)) != len(expected):
        raise ValueError('Held-out members differ from admitted conditions')
    return panel_pin, panel, Path(selector['producer'])


def evaluate(root, cfg, key):
    import jax
    from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
    from go2arx5_vla.sla_adapter.codec_v1 import roundtrip_error
    from go2arx5_vla.sla_adapter.scale20k_evaluation import bytes_sha
    from go2arx5_vla.world_model.mfree_query_diagnostics_v2 import analyze
    from openpi_client.image_tools import resize_with_pad
    from scripts.sla.hopper_reference import core_provider
    from scripts.sla.evaluate_adapt5k import diagnostic_metrics

    slot = root/'endpoints'/key
    start = read(slot/'START.json')
    site = pinned(cfg['receiver'])
    package = Path(cfg['receiver']['path']).parent/'endpoints'/key
    received = pinned(start['receipt'])
    panel_pin, panel, producer = load_panel(cfg)
    def guard():
        if (time.time() >= start['end']-10 or (root/'STOP').exists() or start['owner'] != os.getppid()
                or os.environ['CUDA_VISIBLE_DEVICES'] != site['gpu'] or not os.environ.get('INVOCATION_ID')):
            raise PermissionError('Offline VAL GPU lease expired')
    guard()
    if (panel_pin != start['panel'] or jax.default_backend() != 'gpu' or len(jax.local_devices()) != 1
            or 'RTX' not in jax.local_devices()[0].device_kind):
        raise PermissionError('Expected immutable VAL panel and Core GPU')
    jax.config.update('jax_compilation_cache_dir', str(root/'jax-cache'))
    target = pinned(site['input'])['target_D1']
    data = AdaptTargetDataset(target['legacy'],target['captures'],resize_with_pad=resize_with_pad)
    data.bind_state(target['statistics'])
    model, manifest = core_provider((package/'payload').resolve(), received['manifest'], site['model'], data,
                                    site['tokenizer']['path'])
    if (received['checkpoint'] != manifest['checkpoint_sha256'] or target['statistics']['sha256'] != manifest['state_sha256']
            or digest(Path(site['action']['path'])) != manifest['action_sha256']
            or f"{manifest['role']}-phase{manifest['phase']:05d}" != key):
        raise ValueError('Received checkpoint or frozen statistics changed')
    results = []
    for row in panel['episodes']:
        for identity,state,images,prompt,mount,prev,truth,frame,reference in windows(row, producer):
            guard()
            n = len(results)
            stem = f'{n:03d}'
            write_once(slot/(stem+'.intent.json'), identity)
            pred = model.predict(state,images,prompt,mount,noise_id=identity['noise_id'],previous_reference=prev)
            physical = pred['physical_H40']
            write_once(slot/(stem+'.generated.json'), dict(sha256=bytes_sha(physical)))
            envelope = analyze(physical,state,frame,reference)
            seam = pred['G7_gripper_speed']
            metrics = dict(envelope=envelope, seam=seam, pose=roundtrip_error(truth,physical),
                           H40_and_seam_pass=bool(envelope['original_full_H40_pass'] and not seam['original_execution_speed_exceeded']))
            metrics['timing'] = diagnostic_metrics(Capture(prev,truth),dict(index=0,phase_bucket='P10_DUE_CLOSE'
                if identity['criterion']=='p10_due_close' else identity['criterion']),physical,metrics)
            speed = np.diff(np.r_[prev['width_m'],.002+.040*physical[:10,9]])/.050
            metrics.update(max_P10_speed_m_s=float(np.max(np.abs(speed))),
                           max_P10_acceleration_m_s2=float(np.max(np.abs(np.diff(speed)/.050))))
            out = slot/(stem+'.npz')
            with out.open('xb') as f:
                np.savez(f, physical=physical, normalized=pred['normalized'][0])
                f.flush()
                os.fsync(f.fileno())
            result = dict(identity=identity, metrics=metrics, output=pin(out))
            write_once(slot/(stem+'.result.json'), result)
            results.append(result)
    if len(results) != panel['selected_windows']:
        raise ValueError('Missing selected VAL windows')
    write_once(slot/'RESULT.json',dict(status='VAL_OFFLINE_SEALED',panel=panel_pin,checkpoint=received['checkpoint'],
        attempted=len(results),generated=len(results),h40_seam=sum(r['metrics']['H40_and_seam_pass'] for r in results),
        expert_due=sum(r['metrics']['timing']['expert_P10_due_close'] for r in results),
        timely_and_gates=sum(r['metrics']['timing']['timely_P10_close_and_original_gates'] for r in results),
        updates=0,native_requests=0,native_ticks=0))


def dispatch(root,cfg,sha,endpoint):
    from scripts.sla.scale20k_core_prefix import resources

    site = pinned(cfg['receiver'])
    panel_pin,panel,_ = load_panel(cfg)
    receipt = pin(endpoint/'RECEIVED.json')
    identity = pinned(receipt)
    replay = endpoint/'RESULT.json'
    if replay.is_file():
        restored = read(replay)
        if restored['checkpoint'] != identity['checkpoint']:
            raise ValueError('RGB16 and received checkpoint differ')
        if 'manifest' in restored and restored['manifest'] != identity['manifest']:
            raise ValueError('RGB16 package identity differs')
    elif time.time() < identity['received_epoch']+300:
        return 'WAIT_RGB16_PRIORITY'
    slot = root/'endpoints'/endpoint.name
    if (slot/'START.json').exists():
        raise RuntimeError('Unreconciled VAL launch')
    if panel['selected_windows'] == 0:
        slot.mkdir(exist_ok=True)
        write_once(slot/'RESULT.json', dict(status='NO_ELIGIBLE_VAL_WINDOWS',panel=panel_pin,
                   checkpoint=identity['checkpoint'],attempted=0,generated=0,updates=0,native_requests=0,native_ticks=0))
        return 'NO_ELIGIBLE_VAL_WINDOWS'
    try:
        lock = OwnerLock(site['gpu_lock'])
    except BlockingIOError:
        return 'WAIT_GPU_OWNER'
    with closing(lock):
        available = resources(site['gpu'])
        if (available['free_bytes'] <= available['total_bytes']*site['memory_fraction']+2*1024**3
                or available['host_bytes'] <= 30*1024**3):
            return 'WAIT_CAPACITY'
        end = min(time.time()+3600, DEADLINE, site['deadline'])
        if time.time()+300 >= end or (root/'STOP').exists():
            return 'STOP'
        slot.mkdir(exist_ok=True)
        write_once(slot/'START.json',dict(end=end,owner=os.getpid(),receipt=receipt,panel=panel_pin,
                   windows=panel['selected_windows'],capacity=available,core_rgb16=pin(replay) if replay.is_file() else None,
                   numeric_profile='CORE_OFFLINE_ONLY',native_authorized=False))
        env = dict(os.environ,JAX_PLATFORMS='cuda',CUDA_VISIBLE_DEVICES=site['gpu'],
                   XLA_PYTHON_CLIENT_MEM_FRACTION=str(site['memory_fraction']))
        cmd = [sys.executable,'-B',__file__,'evaluate',str(root),sha,'--endpoint',endpoint.name]
        try:
            with (slot/'worker.log').open('xb') as log:
                subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,end-time.time()))
        except subprocess.TimeoutExpired:
            pass
        if not (slot/'RESULT.json').exists():
            write_once(slot/'RESULT.json',dict(status='VAL_OFFLINE_INTERRUPTED',attempted=len(list(slot.glob('*.intent.json'))),
                       generated=len(list(slot.glob('*.generated.json'))),measured=len(list(slot.glob('*.result.json'))),
                       log='worker.log',updates=0,native_requests=0))
        return read(slot/'RESULT.json')['status']


def watch(root,cfg,sha):
    receiver = Path(cfg['receiver']['path']).parent
    selector = Path(cfg['selector']['path']).parent
    with closing(OwnerLock(root/'owner.lock')):
        while time.time()<DEADLINE-300 and not (root/'STOP').exists():
            status = 'WAIT_FIXED_VAL'
            if (selector/'PANEL.private.json').is_file():
                status = 'WAIT_SEALED_CHECKPOINT'
                ready = sorted((receiver/'endpoints').glob('*/RECEIVED.json'))
                if len(ready)>7 or any(not KEY.fullmatch(p.parent.name) for p in ready):
                    raise ValueError('Invalid endpoint population')
                for receipt in ready:
                    endpoint = receipt.parent
                    if (root/'endpoints'/endpoint.name/'RESULT.json').exists():
                        continue
                    status = dispatch(root,cfg,sha,endpoint)
                    break
            write(root/'HEARTBEAT.json',dict(status=status,
                       completed=len(list((root/'endpoints').glob('*/RESULT.json'))),pid=os.getpid(),epoch=time.time()))
            time.sleep(90)
    write_once(root/'CLOSED.json',dict(status='STOP_OR_DEADLINE',epoch=time.time()))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['watch','evaluate'])
    p.add_argument('root',type=Path)
    p.add_argument('sha')
    p.add_argument('--endpoint')
    a=p.parse_args()
    config=pinned(dict(path=str(a.root/'CONFIG.json'),sha256=a.sha))
    if (config['plan']!=PLAN or config['deadline']!=DEADLINE or os.uname().nodename!='corelab-4080s'
            or os.getuid()!=2009 or not os.environ.get('INVOCATION_ID') or digest(Path(__file__))!=config['worker_sha']):
        raise PermissionError('Expected the registered local VAL service')
    if a.action=='watch':
        watch(a.root,config,a.sha)
    elif KEY.fullmatch(a.endpoint or ''):
        evaluate(a.root,config,a.endpoint)
    else:
        raise ValueError('Unrecognized VAL endpoint')

