"""Frozen evaluation packets; no simulator or model imports."""
import json,hashlib
from pathlib import Path
import numpy as np
from run import ROOT,write
RAW=Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0276')

def freeze_noise():
    card=json.loads((ROOT/'design/CONDITIONS.json').read_text());data=json.loads((ROOT/'design/DATASET_CARD.json').read_text())
    train=sorted(e['seed'] for e in data['episodes'] if e['split']=='TRAIN' and e['admitted'])[:16];assert len(train)==16
    panel=json.loads((ROOT/'design/OFFLINE_CARD.json').read_text())['rows'];generator=np.random.Generator(np.random.PCG64(27611));rows=[]
    for lane,conditions,queries in [('TEST',card['conditions']['TEST'],80),('TRAIN_PHYSICS',train,80),('OFFLINE',[r['uid'] for r in panel],1)]:
        for condition in conditions:
            values={f'q{q:03d}':generator.standard_normal((1,40,132),dtype=np.float32) for q in range(queries)}
            path=RAW/'noise'/f'{lane}-{condition}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                with np.load(path) as z:
                    assert set(z.files)==set(values)
                    for n,v in values.items():assert np.array_equal(z[n],v)
            else:np.savez(path,**values)
            rows.extend(dict(lane=lane,condition=condition,index=q,file=str(path),name=n,sha256=hashlib.sha256(v.tobytes()).hexdigest()) for q,(n,v) in enumerate(values.items()))
    value=dict(seed=27611,rows=rows,TRAIN_diagnostic_seeds=train,independent_of_outcomes=True)
    path=ROOT/'design/EVALUATION_NOISE.json'
    if path.exists():assert json.loads(path.read_text())==value
    else:write(path,value)
    return value
