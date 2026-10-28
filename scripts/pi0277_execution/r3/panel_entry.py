"""Original physical panel, with only the explicitly authorized decision guard changed."""
import argparse,fcntl,inspect,sys
sys.path.insert(0,'/home/liuyu-nfs/Projects/go2-arx5/PI0277/R2/code')
import panel
from fast_decision import decision,R3

def main(point,lane):
    assert lane in ('TEST','CONFIRM')
    for r in panel.read(R3/'EXECUTION_FILESET.json')['files']:assert panel.sha(r['path'])==r['sha256']
    source=inspect.getsource(panel.panel)
    old="assert read(ROOT/'evidence'/f'CPU_REPLAY_{point}_TRAIN_PHYSICS.json')['task']['persistent_stacks']>=3"
    new="decision_path,decision_value=decision(point,'TRAIN_PHYSICS');assert decision_value['task']['persistent_stacks']>=3"
    assert source.count(old)==1;source=source.replace(old,new)
    old="TRAIN_CPU_receipt_sha256=sha(ROOT/'evidence'/f'CPU_REPLAY_{point}_TRAIN_PHYSICS.json')"
    new="TRAIN_decision_receipt_sha256=sha(decision_path),TRAIN_decision_receipt_path=str(decision_path),TRAIN_decision_receipt_schema=decision_value['status']"
    assert source.count(old)==1;source=source.replace(old,new)
    scope=dict(vars(panel),decision=decision)
    exec(compile(source,str(R3/'code/panel_entry.py'),'exec'),scope)
    with (panel.COMMON/'runtime/quality_owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);scope['panel'](point,lane)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('TEST','CONFIRM'));p.add_argument('--point',required=True);a=p.parse_args();main(a.point,a.stage)
