"""Exactly the frozen TRAIN stage starts; diagnostic placement, never task credit."""
import isaacgym
import argparse
import os
import subprocess
from virtual_episode import BASE,R1,ROOT,read,message,run_episode

def main(point,family_id):
    # Native renderer Python stays separate from the modern processor/model runtime.
    audit=read(R1/'evidence/MAIN_TRAINING_AUDIT.json')
    assert audit['status']=='PASS_BOTH_4096_MATCHED_TRAINING_INTEGRITY'
    assert audit['optimizer_updates']=={'UNIFORM':4096,'PREFIX':4096} and len(audit['complete_checkpoints'])==4
    card=read(R1/'design/COMPONENT_CARD.json')
    rows=[r for r in card['rows'] if r['family']==family_id];assert len(rows)==16
    scene=read(ROOT/'design/SCENE_CARD.json');families={f['family_id']:f for f in scene['families']}
    assert families[family_id]['split']=='FEEDBACK_TRAIN'
    env=dict(os.environ);env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None);env['PYTHONPATH']=str(R1/'code')
    with (R1/'evidence'/('component-worker-'+point+'-'+family_id+'.log')).open('ab') as log:
        worker=subprocess.Popen([str(BASE/'PI0250/runtime/venv/bin/python'),str(R1/'code/infer_worker.py'),'--point',point],
             stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=env)
        try:
            assert message(worker)==dict(ready=True,point=point)
            for row in rows:
                run_episode(point,families[family_id],row['task'],row['command'],row['backend'],worker,start=row)
            worker.stdin.close();assert worker.wait()==0
        finally:
            if worker.poll() is None:worker.terminate();worker.wait()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',required=True,choices=('ROT2048','UNIFORM4096','PREFIX4096'))
    p.add_argument('--family',required=True);a=p.parse_args();main(a.point,a.family)
