"""Saved-data exposure and exact observation/future hashes; zero model calls."""
import hashlib,json,os,resource,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

def digest(arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a);h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def run(root):
    root=Path(root);old=root.parent/'PI0283';card=root.parent/'PI0276/design/DATASET_CARD.json'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU only'
    began=time.time();source=json.loads(card.read_text());rows=[r for r in source['rows'] if r['eligible'] and r['split']=='TRAIN']
    grouped=defaultdict(list);packets={};maps={}
    for r in rows:
        ep=r['seed']
        if ep not in maps:maps[ep]={k:np.load(old/f'data/episodes/TRAIN-{ep}/{k}.npy',mmap_mode='r') for k in ('state_v1','action','base_camera_raw','hand_camera_raw')}
        m=maps[ep];a=r['anchor'];obs=digest([m[k][a] for k in ('state_v1','base_camera_raw','hand_camera_raw')]);future=digest([m['action'][a:a+40]])
        record=dict(uid=r['uid'],seed=ep,phase=r['group'],anchor=a,observation_packet_sha256=obs,future_command_packet_sha256=future)
        grouped[obs].append(record);packets[r['uid']]=record
    plan=[json.loads(x) for x in (root/'data/PAIRED_PLAN16384.jsonl').read_text().splitlines()]
    assert all(v['uid'] in packets for v in plan)
    duplicates=[dict(observation_packet_sha256=h,members=g,distinct_futures=len({x['future_command_packet_sha256'] for x in g})) for h,g in grouped.items() if len(g)>1]
    exposure=[]
    for phase in range(4):
        part=[r for r in plan if r['phase']==phase];opening=[]
        for r in part:
            a=r['anchor'];commands=maps[r['seed']]['action'][a:a+40,9];opening.append(commands)
        exposure.append(dict(phase=phase,draws=len(part),unique_UIDs=len({r['uid'] for r in part}),event_draws=sum(r['event_draw'] for r in part),stable_open_draws=sum(bool(np.all(c>.8)) for c in opening),opening_first8_draws=sum(bool(np.any(np.diff(c[:8])>.1)) for c in opening)))
    output=root/'data/ACTUAL_TRAIN_PACKET_INDEX.jsonl';output.write_text(''.join(json.dumps(packets[k],sort_keys=True)+'\n' for k in sorted(packets)))
    report=dict(status='PASS_SAVED_TRAIN_DATA_AND_COMMON_PLAN_AUDIT_NOT_MODEL_FIT',dataset_card_sha256=hashlib.sha256(card.read_bytes()).hexdigest(),plan_sha256=hashlib.sha256((root/'data/PAIRED_PLAN16384.jsonl').read_bytes()).hexdigest(),eligible_TRAIN_windows=len(rows),exact_packet_index_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),duplicate_observation_groups=len(duplicates),different_future_groups=sum(v['distinct_futures']>1 for v in duplicates),duplicates=duplicates,exposure=exposure,interpretation='Different futures are retained; duplicates alone do not prove a data error or partial observability.',elapsed_s=time.time()-began,CPU_user_s=resource.getrusage(resource.RUSAGE_SELF).ru_utime,CPU_system_s=resource.getrusage(resource.RUSAGE_SELF).ru_stime,new_neural_calls=0,new_optimizer_updates=0,new_physics=0)
    (root/'data/DUPLICATE_AND_EXPOSURE_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='duplicates'}))

if __name__=='__main__':
    import sys
    run(sys.argv[1])
