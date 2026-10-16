"""Fresh login attestation release around the own PBS neural child."""
import argparse
import os
from pathlib import Path
import resource
import subprocess
import time

from scripts.pi0292.durable import read,write
from .scope import validate
from scripts.pi0294.native_dataset import sha


def run(root,request_path):
    root=Path(root);validate(root,'hopper-neural');directory=Path(request_path).parent
    job=os.environ.get('PBS_JOBID')
    if not job:raise PermissionError('E1_LAUNCH_ONLY_IN_PBS')
    started=time.time();write(directory/'PBS_READY.json',dict(run_uuid=root.name,job_id=job,pid=os.getpid(),observed_epoch=started,model_loaded=False))
    while not (directory/'START_NEURAL.json').exists():
        if time.time()-started>=180:raise TimeoutError('E1_OWNER_RELEASE_NOT_READY_NO_IDLE_MODEL')
        time.sleep(.2)
    release=read(directory/'START_NEURAL.json')
    if release['run_uuid']!=root.name or release['job_id']!=job or release['request_sha256']!=sha(request_path):raise PermissionError('E1_EXACT_FRESH_NATIVE_RELEASE')
    r=read(request_path)
    modules={'E1_M1_TRAIN_PROBE_EVAL':'scripts.pi0294.e1.neural','E1_T1_COLLECTION_MATCHED_M2':'scripts.pi0294.e1.neural2'}
    if r['mode'] not in modules:raise PermissionError('E1_EXACT_SCIENTIFIC_WORKLOAD')
    return subprocess.run([r['python'],'-m',modules[r['mode']],'--root',str(root),'--request',str(request_path)],check=False).returncode


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--request',required=True);a=p.parse_args();raise SystemExit(run(a.root,a.request))
