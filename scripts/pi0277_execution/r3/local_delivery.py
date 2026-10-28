"""Reuse saved-RGB delivery and policy handoff with truthful R3 decision provenance."""
import argparse,inspect,sys
sys.path[:0]=['/home/liuyu-nfs/Projects/go2-arx5/PI0277/R2/delivery_code','/home/liuyu-nfs/Projects/go2-arx5/PI0277/R2/code']
import deliver_panel,handoff
from fast_decision import decision,R3
from evaluation_card import ROOT,read,sha

FULL='PASS_SAVED_S4_DECODE_FRESH_INPUT_NATIVE_PREDICATES'

def edit(source,changes):
    for old,new in changes.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    compile(source,__file__,'exec');return source

def deliver(point,lane):
    source=edit(inspect.getsource(deliver_panel.main),{
        "receipt=read(ROOT/'evidence'/f'CPU_REPLAY_{point}_{lane}.json')":"decision_path,receipt=decision(point,lane)",
        "assert receipt['status']=='PASS_SAVED_S4_DECODE_FRESH_INPUT_NATIVE_PREDICATES'":"assert receipt['status'] in (FULL,'PASS_NATIVE_TASK_RECONSTRUCTION_FOR_DECISION')",
        "video_receipt_sha256=sha(video_receipt),new_model_queries=0":"video_receipt_sha256=sha(video_receipt),decision_receipt_path=str(decision_path),decision_receipt_sha256=sha(decision_path),full_S4_replay=receipt.get('full_S4_replay','PASS'),new_model_queries=0",
    })
    scope=dict(vars(deliver_panel),decision=decision,FULL=FULL)
    exec(compile(source,__file__,'exec'),scope);scope['main'](point,lane)

def package(point):
    source=edit(inspect.getsource(handoff.handoff),{
        "cpu=read(ROOT/'evidence'/f'CPU_REPLAY_{point}_TRAIN_PHYSICS.json')":"decision_path,cpu=decision(point,'TRAIN_PHYSICS')",
        "assert cpu['status']=='PASS_SAVED_S4_DECODE_FRESH_INPUT_NATIVE_PREDICATES'":"assert cpu['status'] in (FULL,'PASS_NATIVE_TASK_RECONSTRUCTION_FOR_DECISION')",
        "for p in sorted((ROOT/'evidence').glob('CPU_REPLAY_*_*.json')):":"for p in decision_paths():",
        "v=read(p);assert v['status']=='PASS_SAVED_S4_DECODE_FRESH_INPUT_NATIVE_PREDICATES'":"identity=read(p);p,v=decision(identity['point'],identity['lane'])",
        "path=str(p),sha256=sha(p),task=v['task']":"path=str(p),sha256=sha(p),task=v['task'],receipt_status=v['status'],full_S4_replay=v.get('full_S4_replay','PASS')",
        "status='SOURCE_PICKPLACE_PILOT_CONFIRMED'":"qualified=qualified and cpu['status']==FULL and dev['receipt_status']==FULL\n    status='SOURCE_PICKPLACE_PILOT_CONFIRMED'",
        "target=ROOT/'handoff'/point":"target=R3/'handoff'/point",
        "selection_trail=evidence,saved_RGB_videos=videos":"selection_trail=evidence,decision_receipt_path=str(decision_path),decision_receipt_sha256=sha(decision_path),full_S4_replay=cpu.get('full_S4_replay','PASS'),governing_release='a5a7e569684d52170f02b0819ae006a7587800dd',saved_RGB_videos=videos",
    })
    scope=dict(vars(handoff),decision=decision,decision_paths=decision_paths,R3=R3,FULL=FULL)
    exec(compile(source,__file__,'exec'),scope);scope['handoff'](point)

def decision_paths():
    paths=sorted((ROOT/'evidence').glob('CPU_REPLAY_*_*.json'))
    pairs={(read(p)['point'],read(p)['lane']) for p in paths}
    return paths+[p for p in sorted((R3/'evidence').glob('FAST_*_*.json')) if (read(p)['point'],read(p)['lane']) not in pairs]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('point');p.add_argument('lane',choices=('TRAIN_PHYSICS','TEST','CONFIRM'));p.add_argument('--package',action='store_true');a=p.parse_args()
    for row in read(R3/'DELIVERY_FILESET.json')['files']:assert sha(row['path'])==row['sha256']
    deliver(a.point,a.lane)
    if a.package:package(a.point)
