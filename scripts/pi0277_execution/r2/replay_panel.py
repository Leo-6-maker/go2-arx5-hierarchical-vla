"""Saved S4, processor, actual input and native-task replay, zero new neural calls."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from evaluation_card import ROOT,OLD,RAW,OLD_RAW,read,sha,write,freeze,packet
from panel import function,scores

def replay(point,lane):
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    def forbidden(*a,**kw):raise RuntimeError('SAVED_ARRAY_REPLAY_NO_MODEL_OR_OPTIMIZER')
    torch.nn.Module.__call__=forbidden;torch.optim.Optimizer.__init__=forbidden;assert not torch.cuda.is_available()
    data=OLD_RAW.parent/'PI0260/R2';base=ROOT.parent.parent
    sys.path[:0]=[str(data/'R3'),str(data/'runtime'),str(data/'qa'),str(base/'PI0250/runtime/reference-317a/src'),'/home/liuyu-nfs/Projects/isaac-gr00t',str(OLD/'project-src')]
    import pi0260_r2_codec as codec
    proc=codec.processor(read(ROOT/'contracts/TASK/source/dataset/statistics.json'))
    primitive=function(base/'PI0268/R1/code/report.py','replay',dict(np=np,sha=sha))
    metrics=function(OLD/'code/model_replay.py','metrics',dict(np=np,Rotation=Rotation));proof=[];rows=[]
    if lane=='TRAIN_PHYSICS':
        fit={r['uid']:r for r in read(ROOT/'design/FIT_CARD.json')['rows']}
        for r in read(ROOT/'evidence'/f'FIT16_{point}.json')['results']:
            path=Path(r['output']);assert sha(path)==r['sha256'];proof.append(primitive(path,proc,torch,codec))
            item=fit[r['uid']];expected=packet(item)
            with np.load(path.with_name('input.npz')) as z:assert all(np.array_equal(z[k],v) for k,v in expected.items())
            with np.load(Path(item['episode'])/'commands.npz') as z:target=z['action'][item['anchor']:item['anchor']+40]
            with np.load(path) as z:pred=z['decoded'][0]
            rows.append(dict(point=point,uid=item['uid'],seed=item['seed'],group=item['group'],**metrics(pred,target)))
    card=read(ROOT/'design/EVALUATION_CARD.json');seeds=card[{'TRAIN_PHYSICS':'TRAIN4','TEST':'DEV8','CONFIRM':'CONFIRM16'}[lane]]
    for seed in seeds:
        paths=[p for p in (RAW/'episodes'/f'{lane}-{seed}-{point}').glob('attempt-*/RESULT.json') if read(p)['status'].startswith('VALID')];assert len(paths)==1
        folder=paths[0].parent
        with np.load(folder/'observations.npz') as obs:
            for q in read(folder/'QUERIES.json'):
                path=Path(q['output']);receipt=read(path.with_suffix('.receipt.json'));assert receipt['output_sha256']==sha(path)
                proof.append(primitive(path,proc,torch,codec));index=8*q['index'];assert q['time_s']==index*.05
                with np.load(path.with_name('input.npz')) as inp:
                    assert np.array_equal(inp['state_v1'],obs['state_v1'][index])
                    assert np.array_equal(inp['body_rgb'],obs['base_camera_96'][index]) and np.array_equal(inp['wrist_rgb'],obs['hand_camera_96'][index])
    task=scores(point,lane,seeds)
    if rows:
        p=RAW/'delivery'/f'FIT16_{point}.csv';p.parent.mkdir(exist_ok=True)
        with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    freeze(ROOT/'evidence'/f'CPU_REPLAY_{point}_{lane}.json',dict(status='PASS_SAVED_S4_DECODE_FRESH_INPUT_NATIVE_PREDICATES',point=point,lane=lane,
        predictions=proof,FIT16=rows,task=task,new_model_queries=0,new_physics_steps=0,archive='DEFERRED_USER_PRIORITY'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('point');p.add_argument('lane',choices=('TRAIN_PHYSICS','TEST','CONFIRM'));a=p.parse_args();replay(a.point,a.lane)
