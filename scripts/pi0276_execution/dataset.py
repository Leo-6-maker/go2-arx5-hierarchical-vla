"""Frozen window indices into actual EEF replay arrays; no duplicated RGB banks."""
import hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from run import ROOT,sha,write

RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0276')

def freeze(path,value):
    if path.exists():assert json.loads(path.read_text())==value,str(path)
    else:write(path,value)

def main():
    source=json.loads((ROOT/'evidence/SOURCE.json').read_text())
    assert source['status']=='PASS_FIXED_SOURCE_AND_EEF_TASK_GATES'
    cpu=json.loads((ROOT/'evidence/PHYSICS_CPU_REPLAY.json').read_text())
    assert cpu['status']=='PASS_SAVED_PHYSICAL_PREDICATES'
    rows=[];episodes=[];coverage_failures=[]
    for r in source['rows']:
        if r['split'] not in ('TRAIN','VAL'):continue
        folder=Path(r['eef_path']);rec=json.loads((folder/'RESULT.json').read_text())
        for n,h in rec['files'].items():assert sha(folder/n)==h
        with np.load(folder/'commands.npz') as z:phase=z['phase'].copy();action=z['action'].copy()
        with np.load(folder/'observations.npz') as z:
            assert z['state_v1'].shape==(len(action),10)
            for k in ('base_camera_96','hand_camera_96'):
                assert z[k].shape==(len(action),96,96,3) and z[k].dtype==np.uint8
        gates={q['anchor']:q['result']['decision'] for q in json.loads((folder/'FULL_H40_GATE.json').read_text())}
        eligible=[]
        for anchor in range(len(action)):
            reasons=[]
            if not r['eef_success']:reasons.append('SOURCE_OR_REPLAY_NOT_POSITIVE')
            if anchor+40>len(action):reasons.append('NO_GENUINE_FULL_FUTURE')
            elif gates.get(anchor)=='REJECT':reasons.append('GATE_REJECT')
            elif anchor not in gates:reasons.append('GATE_EVIDENCE_MISSING')
            row=dict(uid=f"{r['split']}-{r['seed']}-a{anchor:03d}",split=r['split'],seed=r['seed'],anchor=anchor,
                group=int(phase[anchor]),episode=str(folder),eligible=not reasons,reasons=reasons,
                command_sha256=rec['files']['commands.npz'],observation_sha256=rec['files']['observations.npz'])
            rows.append(row)
            if row['eligible']:eligible.append(row)
        counts=Counter(q['group'] for q in eligible)
        good=r['eef_success'] and all(counts[g]>=2 for g in range(4))
        if r['eef_success'] and not good:coverage_failures.append(dict(split=r['split'],seed=r['seed'],groups=dict(counts)))
        episodes.append(dict(split=r['split'],seed=r['seed'],path=str(folder),positive=r['eef_success'],admitted=good,groups=dict(counts)))
    assert sum(e['admitted'] for e in episodes if e['split']=='TRAIN')>=72
    assert sum(e['admitted'] for e in episodes if e['split']=='VAL')>=12
    allowed={(e['split'],e['seed']) for e in episodes if e['admitted']}
    bank=[r for r in rows if r['eligible'] and (r['split'],r['seed']) in allowed]
    stream=[]
    for m in range(16384):
        group=m%4;es=sorted({r['seed'] for r in bank if r['split']=='TRAIN' and r['group']==group})
        seed=es[(m//4)%len(es)];own=sorted((r for r in bank if r['split']=='TRAIN' and r['group']==group and r['seed']==seed),key=lambda r:r['anchor'])
        row=own[((m//4)//len(es))%len(own)]
        stream.append(dict(micro_index=m,unit=m//2+1,micro=m%2,uid=row['uid'],group=group))
    panel=[]
    for split,n in [('TRAIN',32),('VAL',16)]:
        for group in range(4):
            own=[r for r in bank if r['split']==split and r['group']==group]
            own.sort(key=lambda r:hashlib.sha256(f"{r['seed']}/{r['anchor']}".encode()).hexdigest())
            assert len(own)>=n;panel+=own[:n]
    assert len(panel)==192 and Counter(v['group'] for v in stream)=={g:4096 for g in range(4)}
    freeze(ROOT/'design/DATASET_CARD.json',dict(schema='PI0276_ACTUAL_EEF_REPLAY_WINDOWS_V1',episodes=episodes,rows=rows,
        source_receipt_sha256=sha(ROOT/'evidence/SOURCE.json'),coverage_failures=coverage_failures))
    freeze(ROOT/'design/TRAIN_PLAN.json',dict(schema='PI0276_GROUP_BALANCED16384_V1',seed=27603,updates=8192,stream=stream))
    freeze(ROOT/'design/OFFLINE_CARD.json',dict(rows=panel,TEST_used=False))
    write(ROOT/'evidence/DATA_PRECODEC.json',dict(status='PASS_FROZEN_ACTUAL_REPLAY_WINDOWS_AND_SCHEDULE',
        bank_windows=len(bank),episodes=episodes,coverage_failures=coverage_failures,codec_admission='PENDING',launch_allowed=False,
        new_model_queries=0,new_physics_steps=0,dataset_card_sha256=sha(ROOT/'design/DATASET_CARD.json'),train_plan_sha256=sha(ROOT/'design/TRAIN_PLAN.json')))
    print('PASS_FROZEN_ACTUAL_REPLAY_WINDOWS_AND_SCHEDULE',len(bank),flush=True)

if __name__=='__main__':main()
