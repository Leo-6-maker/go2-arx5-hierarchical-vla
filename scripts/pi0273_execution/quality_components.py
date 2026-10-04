"""Exactly the frozen TRAIN stage starts; diagnostic placement, never task credit."""
import isaacgym
import argparse
import os
import subprocess
from quality_virtual import BASE,R1,ROOT,read,message,run_episode

def main(point,family_id):
    # Native renderer Python stays separate from the modern processor/model runtime.
    from quality_gate import admitted
    admitted();assert point in ('BROAD5120','TARGET5120')
    card=read(R1/'design/RUNTIME_COMPONENT_CARD.json')
    rows=[r for r in card['rows'] if r['family']==family_id];assert len(rows)==16
    scene=read(ROOT/'design/SCENE_CARD.json');families={f['family_id']:f for f in scene['families']}
    assert families[family_id]['split']=='FEEDBACK_TRAIN'
    env=dict(os.environ);env.pop('LD_PRELOAD',None);env.pop('LD_LIBRARY_PATH',None);env['PYTHONPATH']=str(R1/'code')
    with (R1/'evidence'/('component-worker-'+point+'-'+family_id+'.log')).open('ab') as log:
        worker=subprocess.Popen([str(BASE/'PI0250/runtime/venv/bin/python'),str(R1/'code/quality_worker.py'),'--point',point],
             stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,env=env)
        try:
            assert message(worker)==dict(ready=True,point=point)
            for row in rows:
                run_episode(point,families[family_id],row['task'],row['command'],row['backend'],worker,start=row)
            worker.stdin.close();assert worker.wait()==0
        finally:
            if worker.poll() is None:worker.terminate();worker.wait()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--point',required=True,choices=('BROAD5120','TARGET5120'))
    p.add_argument('--family',required=True);a=p.parse_args();main(a.point,a.family)
