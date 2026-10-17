"""Bounded read-only directory observer; its possible cache effect is disclosed.

No packet is written/replaced/read as an action. Enumerating NFS directories
can affect lookup visibility, so an observed run is not an unassisted timing
control. Numeric model/plant/watchdog/transaction identities remain unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import time

from scripts.pi0292.durable import read, write
from scripts.pi0294.native_dataset import sha
from .scope import validate, charge_cpu


def observe(directories, *, duration_s, interval_s=.2, clock=time.monotonic, sleep=time.sleep):
    if not 0 < duration_s <= 1800 or interval_s != .2:
        raise PermissionError('BOUNDED_READ_ONLY_DIRECTORY_OBSERVER')
    paths=[Path(p) for p in directories]
    if any(p.is_symlink() or not p.is_dir() for p in paths):
        raise PermissionError('EXISTING_REAL_PACKET_DIRECTORIES_ONLY')
    began=clock();samples=[];last=None;scans=0
    while clock()-began < duration_s:
        counts=[]
        for path in paths:
            with os.scandir(path) as stream:
                count=sum(1 for _ in stream)
            if count>1280:raise PermissionError('DIRECTORY_ENTRY_BOUND')
            counts.append(count)
        scans+=1
        if counts!=last:
            samples.append(dict(elapsed_s=clock()-began,observed_epoch=time.time(),counts=counts))
            last=counts
        sleep(interval_s)
    return dict(status='READ_ONLY_DIRECTORY_VISIBILITY_OBSERVATION_COMPLETE',scans=scans,
                count_change_samples=samples,wall_s=clock()-began,
                packets_modified=0,new_model_queries=0,new_physics_integrations=0,
                potential_NFS_lookup_cache_observer_effect=True,
                unassisted_transport_qualification=False)


def run(root):
    root=Path(root);m=validate(root,'hopper-neural');binding=root/'plans/DIRECTORY_VISIBILITY_AUDIT_01.json'
    if m['execution'].get('directory_visibility_binding_sha256')!=sha(binding):
        raise PermissionError('SEALED_DIRECTORY_OBSERVER_BINDING')
    q=read(binding);child=root/'children/e1-panel-recovery-03'
    if (q['target_run']!=child.name or q['parent_run']!=root.name or q['duration_s']!=1800
            or q['maximum_CPU_s']!=600 or q['packet_mutations']!=0
            or sha(child/'RUN_MANIFEST.json')!=q['target_manifest_sha256']):
        raise PermissionError('FIXED_EXISTING_CHILD_ONLY_NO_INPUT_MUTATIONS')
    folders=[]
    # Only already existing sessions are observed. No future model/session is
    # created or requested by the observer.
    for label in q['existing_sessions']:
        if label!='E1REC-M2-BC-512-TRAIN4-ALLOCATION-01':
            raise PermissionError('ONE_ALREADY_LIVE_SESSION_ONLY')
        folders.extend(child/'evidence'/label/'resident'/name for name in ['inbox','outbox'])
    if len(folders)!=2:raise PermissionError('ONE_EXISTING_INBOX_OUTBOX_PAIR')
    output=root/'evidence/DIRECTORY_VISIBILITY_AUDIT_01';output.mkdir(mode=0o700)
    start=time.process_time()
    write(output/'START.json',dict(epoch=time.time(),source_sha=m['code']['vla_commit'],
          target_source_sha=read(child/'RUN_MANIFEST.json')['code']['vla_commit'],
          observer_effect_disclosed=True,model_or_physics_source_hot_changed=False))
    try:
        value=observe(folders,duration_s=q['duration_s'])
        write(output/'RESULT.json',value);return value
    finally:
        charge_cpu(root,time.process_time()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_CPU,(600,600))
    def expire(signum,frame):raise TimeoutError('DIRECTORY_OBSERVER_1805S_BOUND')
    signal.signal(signal.SIGALRM,expire);signal.alarm(1805)
    try:print(json.dumps(run(args.root)),flush=True)
    finally:signal.alarm(0)
