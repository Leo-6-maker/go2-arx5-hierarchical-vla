"""Bounded native release, with no account login inside a compute node."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from scripts.pi0292.durable import read,write
from scripts.pi0294.native_dataset import sha
from .science import validate_runtime


def run(root,request_path):
    root=Path(root);validate_runtime(root,'hopper-neural');directory=Path(request_path).parent;job=os.environ.get('PBS_JOBID')
    if not job:raise PermissionError('MP1_NEURAL_LAUNCH_ONLY_IN_OWN_PBS')
    started=time.time();write(directory/'PBS_READY.json',dict(run_uuid=root.name,job_id=job,pid=os.getpid(),observed_epoch=started,model_loaded=False))
    while not (directory/'START_NEURAL.json').exists():
        if time.time()-started>=180:raise TimeoutError('MP1_OWNER_RELEASE_NOT_READY_NO_IDLE_MODEL')
        time.sleep(.2)
    v=read(directory/'START_NEURAL.json');r=read(request_path)
    if v['run_uuid']!=root.name or v['job_id']!=job or v['request_sha256']!=sha(request_path) or r['mode']!='MP1_FIXED_SOURCE_PAIR':
        raise PermissionError('MP1_EXACT_BOUND_NATIVE_RELEASE')
    return subprocess.run([r['python'],'-m','scripts.pi0294.mp1.neural','--root',str(root),'--request',str(request_path)],check=False).returncode


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--request',required=True);a=p.parse_args();raise SystemExit(run(a.root,a.request))
