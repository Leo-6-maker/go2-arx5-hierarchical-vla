"""Actual canary then sole CLEAR4096, using existing resume and retirement helpers."""
import argparse,fcntl,importlib.util,os
from pathlib import Path
import cpu_inputs as data
import train
import persistence
SOURCE=Path(__file__).with_name('qualified_run.py')
spec=importlib.util.spec_from_file_location('pi0279_qualified_run',SOURCE)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


def main(root):
    assert os.environ.get('PBS_JOBID')
    with (root.parent/'runtime/owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for r in data.read(root/'EXECUTION_FILESET.json')['files']:assert data.sha(r['path'])==r['sha256'],r['path']
        receipt=root/'CLEAR/canary_compare.json'
        if not receipt.exists():
            continuous=q.segment(root,'continuous',8,arm='CLEAR');split=root/'CLEAR/split/training'
            if not (split/'checkpoint-4/trainer_state.json').exists():q.segment(root,'split',4,arm='CLEAR')
            q.segment(root,'split',8,4,arm='CLEAR');value=q.compare(root,continuous,split)
            value.update(status='PASS_PI0279_REAL_EXACT_RESUME',governing_release=data.read(root/'RUN_CONFIG.json')['release'],data_plan_sha256=data.sha(root/'design/TRAIN_PLAN.json'))
            data.write(receipt,value)
        for phase in ('continuous','split'):q.retire_canary(root,phase,arm='CLEAR')
        q.segment(root,'main',4096,arm='CLEAR')
        data.write(root/'evidence/TRAINING_COMPLETE.json',dict(status='ACTUAL_CLEAR4096_SEGMENT_ENDED',quality='PENDING_FIXED_LOCAL_SCREENS',HF='DEFERRED_USER_PRIORITY',selected_checkpoint='NONE',deployment='NONE'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
