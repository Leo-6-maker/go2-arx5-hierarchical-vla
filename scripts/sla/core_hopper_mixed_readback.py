"""Qualify the extended TRAIN reader on the last original prefix and the first real mixed seal."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from hopper_reader_compat import compare

LEGACY = 'ab60d9da64e63812528ae1df6c1fdf82c470559676753da6d2ed1008b125db2d'


def read(path):
    return json.loads(path.read_bytes())


def inputs(cfg, receipt, prefix):
    import numpy as np
    from openpi_client.image_tools import resize_with_pad
    from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
    from scripts.sla import hopper_snapshot_relay as relay
    from scripts.sla.hopper_curriculum import WithCoreTrain
    from scripts.sla.hopper_site import pinned
    from core_hopper_course_readback import window

    tree = Path(cfg['capsule'])/cfg['mode']
    if Path(relay.__file__).resolve() != tree/'scripts/sla/hopper_snapshot_relay.py':
        raise ValueError('Unexpected snapshot verifier import')
    config = pinned(cfg['d2'])['data']['target']
    data = AdaptTargetDataset(config['legacy'],config['captures'],resize_with_pad=resize_with_pad)
    data.bind_state(config['statistics'])
    seal = pinned(dict(path=receipt['path'],sha256=receipt['sha256']))
    if seal['episode_sha256'] != receipt['membership'] or seal['count'] != receipt['count']:
        raise ValueError('Catalog membership differs from its seal')
    folder = Path(receipt['path']).parent
    view = WithCoreTrain(data,folder,receipt['sha256'],Path(cfg['population']['path']))
    manifest = read(folder/'MANIFEST.private.json')
    proof = read(folder/'ACTUAL.private.json')
    if (manifest['windows']!=len(view)-len(data) or manifest['windows']!=receipt['windows']
            or [e['id'] for e in manifest['episodes']]!=[c['id'] for c in proof['cases']]):
        raise ValueError('Unexpected number or order of Core rows')
    per_case = {}
    cursor = len(data)
    for ep, source in zip(manifest['episodes'], proof['cases'], strict=True):
        count = ep['windows']
        with np.load(folder/ep['labels']['file'],allow_pickle=False) as arrays:
            supervised, consumed = arrays['supervision_mask'], arrays['consumed_mask']
            if (supervised.shape != (count,40) or consumed.shape!=supervised.shape or
                    supervised.dtype!=np.bool_ or consumed.dtype!=np.bool_ or not supervised.all()
                    or not consumed[:,:10].all() or consumed[:,10:].any()):
                raise ValueError('A G7 mask changed in '+ep['id'])
        stream = hashlib.sha256()
        for i in range(cursor,cursor+count):
            row,age = window(view,i)
            if abs(age-.05)>1e-8 or view.identity(i)['episode_id']!=ep['id']:
                raise ValueError('Unexpected native command clock or window identity')
            stream.update(row+b'\n')
        per_case[ep['id']] = dict(windows=count,full_inputs=stream.hexdigest(),
            producer_spec=source.get('producer_spec',proof.get('producer_spec')),
            native_result=source['native_result_sha256'])
        cursor += count
    if prefix and (receipt['membership'][:prefix['episodes']] != prefix['member_sha256']
            or any(per_case.get(k)!=v for k,v in prefix['cases'].items())):
        raise ValueError('A previously sealed full reader input differs')
    return dict(episodes=receipt['count'],windows=manifest['windows'],d2_windows=len(data),
                supervised_atoms=40*(len(view)-len(data)),consumed_atoms=10*(len(view)-len(data)),
                cases=per_case,member_sha256=receipt['membership'],seal=receipt['sha256'],
                manifest=seal['manifest_sha256'],mixed='schema' in proof)


def child(root,cfg,mode,record,prefix=None):
    tree = Path(cfg['capsule'])/mode
    env = dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,[tree,tree/'src',cfg['dependencies']])))
    cmd = [sys.executable,'-B',str(Path(__file__).resolve()),str(root),'read',mode,str(record)]
    if prefix:
        cmd.append(str(prefix))
    return json.loads(subprocess.check_output(cmd,env=env,text=True))


def watch(root,cfg):
    from scripts.pi0292.durable import OwnerLock, write
    from scripts.sla.hopper_parent_custody import write_once
    from scripts.sla.hopper_site import DEADLINE
    from contextlib import closing

    catalog = Path(cfg['catalog'])
    capsule = Path(cfg['capsule'])
    original = catalog/'TRAIN-016.json'
    if read(original)['sha256']!=LEGACY or 'INVOCATION_ID' not in os.environ or os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Expected the original sealed TRAIN-016 and an isolated CPU service')
    with closing(OwnerLock(root/'owner.lock')):
        verified = compare(capsule/'original',capsule/'candidate')
        write_once(root/'MODULES.json',verified)
        legacy = root/'LEGACY.json'
        if not legacy.exists():
            a = child(root,cfg,'original',original)
            b = child(root,cfg,'candidate',original)
            if a!=b or a['mixed'] or a['windows']!=1317:
                raise ValueError('The two readers differ on the published original examples')
            write_once(legacy,a)
        while time.time()<DEADLINE and not (root/'STOP').exists():
            for f in sorted(catalog.glob('TRAIN-???.json')):
                rec = read(f)
                if not rec['aggregate']:
                    continue
                if rec['count']<=16:
                    raise ValueError('Unexpected aggregate before the fixed prefix')
                value = child(root,cfg,'candidate',f,legacy)
                repair = value['cases'].get(cfg['repair_condition'])
                if (not value['mixed'] or repair is None or repair['producer_spec']!=cfg['repair_spec']
                        or repair['native_result']!=cfg['repair_result'] or repair['windows']!=86):
                    raise ValueError('First real aggregate lacks the verified red HOLD admission')
                write_once(root/'RESULT.json',dict(status='MIXED_TRAIN_COLD_READER_VERIFIED',plan=cfg['plan'],
                    original_and_candidate=verified, unchanged_prefix=16, unchanged_prefix_windows=1317,
                    actual=value, epoch=time.time(), worker=cfg['worker'], updates=0, optimization_visits=0,
                    native_requests=0, model_generations=0, course_switches=0))
                return
            write(root/'HEARTBEAT.json',dict(pid=os.getpid(),epoch=time.time(),stage='WAIT_FIRST_MIXED_CATALOG'))
            time.sleep(min(45,max(0,DEADLINE-time.time())))
        write_once(root/'CLOSED.json',dict(status='STOP_OR_DEADLINE',epoch=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('action',choices=['read','watch'])
    p.add_argument('args',nargs='*')
    a=p.parse_args()
    cfg=read(a.root/'CONFIG.json')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=cfg['worker']:
        raise ValueError('Unregistered reader audit source')
    if a.action=='watch':
        watch(a.root,cfg)
    else:
        mode,record,*prefix=a.args
        if mode not in ('original','candidate') or len(prefix)>1:
            raise ValueError('Unregistered reader or prefix')
        print(json.dumps(inputs(dict(cfg,mode=mode),read(Path(record)),read(Path(prefix[0])) if prefix else None)))
