"""Freeze unchanged two-task training visits and offline evaluation source rows."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from pi0260_r2 import ROOT, read, sha

OUT=ROOT/'R3'


@lru_cache(maxsize=160)
def source(name):
    p=ROOT/'dataset'/name
    rec=read(p.with_suffix('.json'))
    with np.load(p.with_suffix('.npz')) as z:
        arrays={k:z[k].copy() for k in ('state_v1','raw_action')}
    return rec,arrays


def row_for_quartile(rec,q):
    active=[r for r in rec['rows'] if r['active'] and r['usable']]
    end=active[-1]['time']
    candidates=[r for r in active if q*end/4<=r['time']<(q+1)*end/4 or q==3 and r['time']==end]
    assert candidates
    return min(candidates,key=lambda r:(abs(r['time']-(q+.5)*end/4),r['index']))


def row_id(seq,row):
    return dict(sequence=seq,index=row['index'],time_s=row['time'],frame=row['frame'],frame_sha256=row['frame_sha256'])


def main():
    codec=read(ROOT/'dataset/codec_receipt.json')
    assert codec['status']=='COMPACT_30_10'
    selected=codec['selected_post_codec']
    rng=np.random.default_rng(26003)
    visits=[]
    for update in range(6000):
        task=('M1','M3')[update%2]
        visit=update//2
        command=visit%2
        family=selected['TRAIN'][int(rng.integers(len(selected['TRAIN'])))]
        seq=f'{family}-{task}-c{command}'
        rec,_=source(seq)
        quartile=int(rng.integers(4))
        active=[r for r in rec['rows'] if r['active']]
        end=active[-1]['time']
        candidates=[r for r in active if r['usable'] and (quartile*end/4<=r['time']<(quartile+1)*end/4 or quartile==3 and r['time']==end)]
        assert candidates,(seq,quartile)
        row=candidates[int(rng.integers(len(candidates)))]
        visits.append(dict(update=update+1,task=task,command=command,family=family,quartile=quartile,**row_id(seq,row)))
    assert len(visits)==6000 and {t:sum(r['task']==t for r in visits) for t in ('M1','M3')}=={'M1':3000,'M3':3000}
    assert all(sum(r['task']==t and r['command']==c for r in visits)==1500 for t in ('M1','M3') for c in (0,1))
    cases=[];semantic=[]
    for task in ('M1','M3'):
        for split,n in (('TRAIN',32),('KIN-VAL',16)):
            seqs=[f'{family}-{task}-c{command}' for family in selected[split] for command in (0,1)]
            assert len(seqs)>=n
            for i,seq in enumerate(seqs[:n]):
                rec,_=source(seq)
                cases.append(dict(task=task,kind=split,quartile=i%4,language=rec['sequence']['instruction'],**row_id(seq,row_for_quartile(rec,i%4))))
        for family in selected['KIN-VAL'][:6]:
            seqs=[f'{family}-{task}-c{c}' for c in (0,1)]
            pair=[source(s) for s in seqs]
            selected_pair=None
            for j,(a,b) in enumerate(zip(pair[0][0]['rows'],pair[1][0]['rows'])):
                if not(a['active'] and b['active'] and a['usable'] and b['usable']):continue
                if a['frame_sha256']!=b['frame_sha256'] or not np.array_equal(pair[0][1]['state_v1'][j],pair[1][1]['state_v1'][j]):continue
                pa,pb=pair[0][1]['raw_action'][j,:,:3],pair[1][1]['raw_action'][j,:,:3]
                difference=np.linalg.norm(pa-pb,axis=1)
                if difference.max()>1e-5:
                    selected_pair=(a,b,float(np.sqrt(np.mean(difference**2))),float(difference.max()))
                    break
            assert selected_pair is not None,('SEMANTIC_REFERENCE_NOT_SEPARABLE',family,task)
            a,b,rmse,peak=selected_pair
            semantic.append(dict(task=task,family=family,source_times_s=[a['time'],b['time']],reference_xyz_rmse_m=rmse,
                                 reference_max_pointwise_m=peak,common_frame_sha256=a['frame_sha256']))
            for c,(seq,row) in enumerate(zip(seqs,(a,b))):
                rec,_=pair[c]
                cases.append(dict(task=task,kind='SEMANTIC',command=c,language=rec['sequence']['instruction'],**row_id(seq,row)))
                para=next(p for p in reversed(rec['sequence']['paraphrases']) if p!=rec['sequence']['instruction'])
                cases.append(dict(task=task,kind='PARAPHRASE',command=c,language=para,**row_id(seq,row)))
            cases.append(dict(task=task,kind='EMPTY_LANGUAGE',language='',**row_id(seqs[0],a)))
    assert len(cases)==156
    counts={task:{kind:sum(c['task']==task and c['kind']==kind for c in cases) for kind in ('TRAIN','KIN-VAL','SEMANTIC','PARAPHRASE','EMPTY_LANGUAGE')} for task in ('M1','M3')}
    assert all(list(x.values())==[32,16,12,12,6] for x in counts.values())
    OUT.mkdir(exist_ok=True)
    for name,value in [('train_visits.json',dict(seed=26003,visits=visits,rule='alternate tasks; alternating command per task; uniform family, then uniform quartile and usable window')),
                       ('motion2_eval_cases.json',dict(cases=cases,semantic_pairs=semantic,counts=counts,source_rule='first families/sequences, phase-balanced; first same-frame separable paired state'))]:
        p=OUT/name
        assert not p.exists()
        p.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
        print(json.dumps(dict(path=str(p),sha256=sha(p),rows=len(visits) if name.startswith('train') else len(cases))))


if __name__=='__main__':main()
