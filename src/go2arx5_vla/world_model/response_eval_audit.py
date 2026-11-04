"""Frozen evaluation semantics audit on current FIT/DEV; no optimizer or physics."""
from pathlib import Path
import hashlib,time,itertools
import numpy as np
import torch
from .response_diagnostics import groups,forecast,EVENTS
from .contact_training import select
from .decision_training import read
from .decision_diagnostics import stable_choice,frozen_model
from .neural import WorldModel
from .neural_training import parameter_hash
from .replay import SCHEMA,SLICES
from .storage import digest,write_json

def run(spec,output):
    start=time.monotonic();cpu=time.process_time();torch.set_num_threads(len(spec['cpus']));output=Path(output)
    if digest(spec['replay'])!=spec['replay_sha256']:raise ValueError('Replay identity differs')
    data,meta=read(spec['replay'],'cpu');keep=meta['partition']!='ONCE_HOLDOUT';data=select(data,torch.tensor(np.flatnonzero(keep)));meta={k:v[keep]for k,v in meta.items()}
    result={}
    for name,binding in spec['models'].items():
        if digest(binding['path'])!=binding['sha256']:raise ValueError('Checkpoint differs')
        if name=='M0':model=frozen_model(binding)
        else:
            c=torch.load(binding['path'],map_location='cpu',weights_only=False)
            if c['step']!=200:raise ValueError('Fixed endpoint required')
            model=WorldModel(**c['model_config']);model.load_state_dict(c['model']);model.eval();del c
            for p in model.parameters():p.requires_grad_(False)
        pin=parameter_hash(model);rows=[]
        with torch.no_grad():
            for part in('FIT','DIAGNOSTIC_DEV'):
                for scene,ids,b,names in groups(data,meta,part):
                    p,e,_=forecast(model,b);singles=[forecast(model,select(b,slice(i,i+1)))for i in range(5)];p1=torch.cat([v[0]for v in singles]);e1=torch.cat([v[1]for v in singles]);order=torch.tensor([4,2,0,3,1]);pp,ep,_=forecast(model,select(b,order));pp=pp[torch.argsort(order)];ep=ep[torch.argsort(order)]
                    raw=meta['physical_raw'][ids]
                    if raw.dtype!=np.float64:raise ValueError('Original float64 physical command identity required')
                    checks=[]
                    for h in(1,8,16):
                        scores=[]
                        for pred,event in[(p,e),(p1,e1),(pp,ep)]:scores.append((event[:,h-1,EVENTS['native_grasp']]-10*(pred[:,h-1,SLICES['object_position']][:,:2]-b['context'][:,:2]).norm(dim=-1)).numpy())
                        choices=[names[stable_choice(score,raw)]for score in scores]
                        hashes=[hashlib.sha256(np.asarray(raw[names.index(s)],dtype='<f8').tobytes()).hexdigest()for s in choices]
                        checks.append(dict(horizon=h,choice_batch5_batch1_permuted=choices,physical_command_sha256=hashes[0],same_command=len(set(hashes))==1,score_max_abs=max(float(np.max(abs(scores[0]-x)))for x in scores[1:])))
                    rows.append(dict(partition=part,scene=scene,state_max_abs=max(float((p1-p).abs().max()),float((pp-p).abs().max())),checks=checks))
        if parameter_hash(model)!=pin:raise RuntimeError('Audit changed model')
        result[name]=dict(checkpoint_sha256=binding['sha256'],rows=rows,all_choices_same=all(c['same_command']for r in rows for c in r['checks']),max_state_abs=max(r['state_max_abs']for r in rows),max_score_abs=max(c['score_max_abs']for r in rows for c in r['checks']))
        del model
    value=dict(source_sha=spec['source_sha'],data_sha256=spec['replay_sha256'],named_events=EVENTS,support_prediction_head=False,models=result,no_holdout_evaluation=True,WM_updates=0,Actor_updates=0,physics=0,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    write_json(output/'RESULT.json',value);return value
