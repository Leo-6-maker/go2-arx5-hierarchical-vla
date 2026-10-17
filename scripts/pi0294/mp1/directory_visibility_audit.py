"""Bounded read-only directory observer; its possible cache effect is disclosed.

No packet is written/replaced/read as an action. Enumerating NFS directories
can affect lookup visibility, so an observed run is not an unassisted timing
control. Numeric model/plant/watchdog/transaction identities remain unchanged.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import resource
import signal
import time

from scripts.pi0292.durable import read, write
from scripts.pi0294.native_dataset import sha
from .scope import validate, charge_cpu


def force_server_attributes(path):
    """statx(2) metadata read, not packet mutation or cache-wide drop/remount."""
    libc=ctypes.CDLL(None,use_errno=True)
    function=libc.statx
    function.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_uint,ctypes.c_void_p]
    function.restype=ctypes.c_int
    buffer=ctypes.create_string_buffer(256)
    # AT_SYMLINK_NOFOLLOW | AT_NO_AUTOMOUNT | AT_STATX_FORCE_SYNC;
    # STATX_BASIC_STATS. See the Linux statx(2) synchronization contract.
    if function(-100,os.fsencode(path),0x100|0x800|0x2000,0x7ff,buffer):
        error=ctypes.get_errno();raise OSError(error,os.strerror(error))


def observe(directories, *, duration_s, interval_s=.2, clock=time.monotonic, sleep=time.sleep,
            force_sync=False,allow_pending=False):
    if not 0 < duration_s <= 3600 or interval_s != .2:
        raise PermissionError('BOUNDED_READ_ONLY_DIRECTORY_OBSERVER')
    paths=[Path(p) for p in directories]
    if any(p.is_symlink() or (not p.is_dir() and not allow_pending) for p in paths):
        raise PermissionError('EXISTING_REAL_PACKET_DIRECTORIES_ONLY')
    began=clock();samples=[];last=None;scans=0
    while clock()-began < duration_s:
        counts=[]
        for path in paths:
            if path.is_symlink():raise PermissionError('PACKET_DIRECTORY_SYMLINK_FORBIDDEN')
            if allow_pending and not path.is_dir():counts.append(None);continue
            if force_sync:force_server_attributes(path)
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
                server_attribute_synchronization_requested=force_sync,
                unassisted_transport_qualification=False)


def run(root,attempt=1):
    if attempt not in [1,2,3]:raise PermissionError('TWO_OBSERVER_ALGORITHMS_AND_ONE_FIXED_CORR_CONTINUATION')
    name='DIRECTORY_VISIBILITY_AUDIT_'+str(attempt).zfill(2)
    root=Path(root);m=validate(root,'hopper-neural');binding=root/'plans'/(name+'.json')
    if m['execution'].get('directory_visibility_binding_sha256')!=sha(binding):
        raise PermissionError('SEALED_DIRECTORY_OBSERVER_BINDING')
    q=read(binding);child=root/'children/e1-panel-recovery-03'
    if (q['target_run']!=child.name or q['parent_run']!=root.name or q['duration_s']!=(3600 if attempt==3 else 1800)
            or q['maximum_CPU_s']!=600 or q['packet_mutations']!=0 or q.get('force_server_attributes',False)!=(attempt>=2)
            or sha(child/'RUN_MANIFEST.json')!=q['target_manifest_sha256']):
        raise PermissionError('FIXED_EXISTING_CHILD_ONLY_NO_INPUT_MUTATIONS')
    folders=[]
    # V2 may wait for the second predeclared session to appear naturally; it
    # never creates a future directory or asks a model/session to start.
    for label in q['existing_sessions']:
        permitted=['E1REC-M2-BC-512-TRAIN4-ALLOCATION-01']
        if attempt==2:permitted.append('E1REC-M2-CORR-512-TRAIN4-ALLOCATION-01')
        if attempt==3:permitted=['E1REC-M2-CORR-512-TRAIN4-ALLOCATION-01']
        if label not in permitted:raise PermissionError('ONLY_FIXED_TWO_MODEL_SESSION_DIRECTORIES')
        folders.extend(child/'evidence'/label/'resident'/name for name in ['inbox','outbox'])
    if len(folders)!=(4 if attempt==2 else 2):raise PermissionError('FIXED_SESSION_INBOX_OUTBOX_PAIRS')
    output=root/'evidence'/name;output.mkdir(mode=0o700)
    start=time.process_time()
    write(output/'START.json',dict(epoch=time.time(),source_sha=m['code']['vla_commit'],
          target_source_sha=read(child/'RUN_MANIFEST.json')['code']['vla_commit'],
          observer_effect_disclosed=True,model_or_physics_source_hot_changed=False))
    try:
        value=observe(folders,duration_s=q['duration_s'],force_sync=attempt>=2,allow_pending=attempt>=2)
        write(output/'RESULT.json',value);return value
    finally:
        charge_cpu(root,time.process_time()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--attempt',type=int,default=1);args=p.parse_args()
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_CPU,(600,600))
    def expire(signum,frame):raise TimeoutError('DIRECTORY_OBSERVER_FIXED_WALL_BOUND')
    signal.signal(signal.SIGALRM,expire);signal.alarm(3605 if args.attempt==3 else 1805)
    try:print(json.dumps(run(args.root,args.attempt)),flush=True)
    finally:signal.alarm(0)
