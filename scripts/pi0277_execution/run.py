"""Finite, locked stage owner; actual receipts, same-state continuation."""
import argparse,json,os,subprocess,sys
from pathlib import Path
RELEASE='63151a4548243ab0fb4ba172eea6f7d92e02c576'

def hopper():
    import fcntl
    root=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0277');code=root/'code'
    (root/'runtime').mkdir(parents=True,exist_ok=True)
    with (root/'runtime/owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert os.environ.get('PBS_JOBID')
        def call(name,*args):subprocess.run([sys.executable,'-u',str(code/name),*args],check=True)
        if not (root/'evidence/LOCAL_BINDING.json').exists():call('prepare.py')
        stages=[('cpu','CPU_SELECTION.json'),('h200','H200_SELECTION.json')]
        for stage,receipt in stages:
            if not (root/'evidence'/receipt).exists():call('qualify.py','--root',str(root),'--stage',stage)
        call('qualify.py','--root',str(root),'--stage','canary')
        call('qualify.py','--root',str(root),'--stage','train')
        from cpu_inputs import write
        write(root/'evidence/HOPPER_TRAINING_COMPLETE.json',dict(status='BOTH_FINAL4096_EXECUTED_FOUR_TRAINERS_PENDING_ARCHIVE',
            selected_checkpoint='NONE',deployment_selected_checkpoint='NONE',quality='PENDING_FIXED_EVALUATION'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('hopper',));p.add_argument('--release',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    assert a.release==RELEASE
    hopper()
