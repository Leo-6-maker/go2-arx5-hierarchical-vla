"""Own a bounded nonintegrating measurement on the same site/device profile as phase zero."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.pi0292.durable import OwnerLock
from scripts.sla.hopper_parent_custody import digest
from scripts.sla.hopper_site import DEADLINE, pinned, site_epoch


def once(file, data):
    with file.open('x') as f:
        json.dump(data,f,sort_keys=True)
        f.flush()
        os.fsync(f.fileno())


def execute(root, cfg, config_hash, locks):
    if cfg['uid'] != os.getuid() or cfg['deadline'] > DEADLINE or time.time() >= cfg['deadline']:
        raise PermissionError('Expected a current site-bound scope')
    worker = root/'hopper_numeric_probe.py'
    if digest(worker)!=cfg['worker_sha256'] or digest(Path(__file__))!=cfg['driver_sha256']:
        raise ValueError('Cold execution files changed')
    if cfg['modes'] not in (['ordinary2','trace2'],['ordinary14'],['ordinary16']):
        raise ValueError('Unexpected diagnostic population')
    end = cfg['deadline']
    env = os.environ.copy()
    if cfg['site']=='core':
        from scripts.sla.scale20k_core_prefix import resources
        src=pinned(dict(path=cfg['receiver_config'],sha256=cfg['receiver_sha256']))
        if (os.uname().nodename!=src['host'] or not env.get('INVOCATION_ID') or src['gpu']!=cfg['gpu']
                or src['gpu_lock']!=cfg['gpu_lock'] or env.get('CUDA_VISIBLE_DEVICES')!=src['gpu']):
            raise PermissionError('Wrong Core service or original deployment GPU')
        locks.callback(OwnerLock(src['gpu_lock']).close)
        available=resources(src['gpu'])
        if (available['free_bytes']<=available['total_bytes']*src['memory_fraction']+2*1024**3
                or available['host_bytes']<=30*1024**3):
            raise PermissionError('Insufficient Core headroom')
        once(root/'AWARD.json',dict(unit=env['INVOCATION_ID'],host=os.uname().nodename,
                                   gpu=src['gpu'],capacity=available))
    elif cfg['site']=='hopper':
        import hopper_project_admission as project
        job=env['PBS_JOBID']
        ticket=json.loads((root/'PBS_JOB.json').read_bytes())
        mountfile=root/'MOUNT.private.json'
        if (ticket['job']!=job or ticket['config']!=config_hash or
                ticket['mount_sha256']!=digest(mountfile)):
            raise PermissionError('Unexpected PBS job, site mount or specification')
        mount=json.loads(mountfile.read_bytes())
        if (mount['pbs_project_basis']!=cfg['project_basis'] or mount['lineage']!=cfg['lineage']
                or mount['qstat']!=cfg['qstat'] or mount['uid']!=cfg['uid']):
            raise PermissionError('Current site mount differs from the preregistered scope')
        # Login-side project checks have already run before PBS release. Reuse the
        # production compute-node verifier, including the live site billing cache.
        project.running(root)
        award=json.loads((root/'PBS_PROJECT_AWARD.json').read_bytes())
        grant=pinned(mount['site_grant'])
        if award['wall_seconds']!=cfg['wall_seconds']:
            raise PermissionError('Unexpected duration on the assigned GPU')
        end=min(end,site_epoch(award['site_start'])+cfg['wall_seconds']-30,grant['expiry_epoch'])
        once(root/'AWARD.json',dict(job=job,project=project.PROJECT,host=award['host'],
                                   actual_site_start=award['site_start'],rate=award['queue_rate'],
                                   wall_seconds=cfg['wall_seconds'],remaining_seconds=end-time.time()))
    else:
        raise ValueError('Unknown site')
    once(root/'OWNER.json',dict(pid=os.getpid(),started=time.time(),end=end))
    for mode in cfg['modes']:
        if time.time()>=end-5 or (root/'STOP').exists():
            raise TimeoutError('STOP or expired allocation')
        if mode=='trace2':
            normal=json.loads((root/'ordinary2/RESULT.json').read_bytes())
            if not normal['own_v1_exact']:
                return 'SAME_SITE_V1_OUTPUT_DIFFERED'
        cmd=[sys.executable,'-B',str(worker),str(root),config_hash,mode]
        with (root/(mode+'.log')).open('xb') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=end-time.time())
    return 'DIAGNOSTIC_SEALED'


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('config_sha')
    args=p.parse_args()
    os.umask(0o077)
    config=pinned(dict(path=str(args.root/'CONFIG.json'),sha256=args.config_sha))
    with ExitStack() as stack:
        stack.callback(OwnerLock(args.root/'owner.lock').close)
        if (args.root/'RESULT.json').exists() or (args.root/'OWNER.json').exists():
            raise RuntimeError('Previously started diagnostic')
        try:
            status=execute(args.root,config,args.config_sha,stack)
            reason=None
        except Exception as exc:
            status='DIAGNOSTIC_INTERRUPTED'
            reason=f'{type(exc).__name__}: {exc}'
        once(args.root/'RESULT.json',dict(status=status,reason=reason,time=time.time(),
            intents=len(list(args.root.glob('*/*.intent.json'))),completed=len(list(args.root.glob('*/*.private.npz'))),
            updates=0,native_requests=0,native_ticks=0))
    sys.exit(1 if reason else 0)
