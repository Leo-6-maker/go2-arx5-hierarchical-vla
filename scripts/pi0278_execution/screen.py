"""One fixed point's finite TRAIN/DEV/conditional confirmation; no new monitor."""
import argparse,json,time
from call import call,remote,ROOT
from reconcile_ready import reconcile

def phase(script,args,native=False,common=False):
    while True:
        v=call(script,args,native,common)
        if v['status']=='COMPLETE':return
        assert v['status']=='RUNNING'
        time.sleep(30)

def screen(point):
    assert point in ['COVER931024','COVER932048','COVER934096']
    while True:
        ready=json.loads(remote('core',f"import json\nfrom pathlib import Path\np=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0278/R2/models/{point}/READY.json');print(p.read_text() if p.is_file() else 'null')"))
        if ready is not None:break
        time.sleep(30)
    if ready['release']!='83af9099f3bab9a4ac08eaa85380c2cfd6351f34':reconcile(point)
    phase('panel.py',['admit','--point',point])
    for lane,threshold in [('TRAIN_PHYSICS',3),('TEST',6),('CONFIRM',12)]:
        phase('panel.py',[lane,'--point',point],native=True)
        phase('fast_decision.py',[point,lane]);phase('replay_panel.py',[point,lane])
        proof=json.loads(remote('core',f"from pathlib import Path\nprint(Path({(ROOT+'/evidence/CPU_REPLAY_'+point+'_'+lane+'.json')!r}).read_text())"))
        assert proof['status']=='PASS_SAVED_S4_DECODE_FRESH_INPUT_NATIVE_PREDICATES'
        n=proof['task']['persistent_stacks'];print('ACTUAL_NATIVE_VERDICT',point,lane,n,proof['task']['denominator'],flush=True)
        if n<threshold:return
        if lane=='TEST':
            phase('candidate.py',['freeze',point],common=True)
            frozen=remote('core',f"from pathlib import Path\nprint(Path({(ROOT+'/evidence/FROZEN_CANDIDATE.json')!r}).read_text())")
            print(remote('hop',f"import json\nfrom pathlib import Path\nr=Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0278/run');v=json.loads({frozen!r});p=r/'runtime/STOP_TRAINING.json'\nif p.exists():assert json.loads(p.read_text())==v\nelse:p.write_text(json.dumps(v,sort_keys=True)+chr(10))\nprint('FROZEN_CANDIDATE_STOP_AT_NEXT_SAFE_UPDATE')"),flush=True)
            phase('frozen.py',[point],common=True);phase('candidate.py',['confirmation',point],common=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('point');screen(p.parse_args().point)
