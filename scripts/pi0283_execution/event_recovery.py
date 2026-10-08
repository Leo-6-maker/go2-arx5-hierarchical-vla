"""One bounded technical EVENT recovery; MORE/canary and ledgers are retained."""
import fcntl,json,os,time
from pathlib import Path
from training_launch import call,write
from authority import load
ROOT=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0283')
BASE='/scratch/e1192548/Projects/go2-arx5/archive/stage_runs/stage22_handoff_deb920a_20260902/payload/GR00T-N1.7-3B'

def run():
    assert os.environ.get('PBS_JOBID') and os.environ.get('CUDA_VISIBLE_DEVICES')
    r=ROOT;caps=load(r);route=json.loads((r/'design/EVENT_DECISION.json').read_text())
    assert caps['technical_recoveries_used']==2 and route['launch_allowed']
    assert json.loads((r/'runtime/MORE_COMPLETE.json').read_text())['status']=='MORE2048_FULL_TRAINER_COMPLETE'
    ledger=[json.loads(x) for x in (r/'learning/completed_updates.jsonl').read_text().splitlines()]
    assert len(ledger)==2064 and sum(x['work_kind']=='formal' for x in ledger)==2048
    assert not any(x.get('arm')=='EVENT' for x in ledger)
    with (r/'runtime/TRAINING_ALLOCATION.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=r/'runtime/TRAINING_ALLOCATION.json';assert not p.exists()
        started=time.time();write(p,dict(PBS_JOBID=os.environ['PBS_JOBID'],pid=os.getpid(),started_epoch=started,scheduler_CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],recovery='EVENT_ONLY_AFTER_ZERO_UPDATE_LEGACY_GUARD_FAILURE',previous_GPU_seconds=caps['previous_training_GPU_seconds']))
        call(r,'EVENT-technical-recovery-1024','train.py',['--root',str(r),'--base',BASE,'--data-root',str(r/'data'),'--microbatch','4','--arm','EVENT','--continuation-card',str(r/'design/EVENT_CARD.json'),'--phase','main','--stop','1024'])
        cp=r/'learning/EVENT/main/checkpoint-1024';assert json.loads((cp/'PI0282_RECOVERY.json').read_text())['cursor']==8192
        write(r/'runtime/TRAINING_COMPLETE.json',dict(status='MORE_AND_ROUTED_EVENT_COMPLETE',EVENT_executed=True,started_epoch=started,ended_epoch=time.time(),PBS_JOBID=os.environ['PBS_JOBID'],selected='NONE',deployment='NONE'))

if __name__=='__main__':run()
