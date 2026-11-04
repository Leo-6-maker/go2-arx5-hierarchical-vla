"""Native-contact support derivative and transparent correction of V1 release scoring."""
from pathlib import Path
import json,time
import numpy as np
from .storage import digest,write_json
from .paired_replay import read as read_rows
from .native_adapter import SourceAdapter
from .contact_identification import supported

PURPOSE='WM02_DECISION_SUPPORT_EVALUATION_V2'


def derive(original,native_root,spec,output):
    native_root=Path(native_root);output=Path(output);output.mkdir(exist_ok=False);start=time.monotonic();cpu=time.process_time()
    if digest(original)!=spec['replay_sha256']:raise ValueError('Original training replay differs')
    with np.load(original,allow_pickle=False)as f:arrays={k:f[k].copy()for k in f.files}
    supports=[];adapter=None;current_scene=None;constructors=0;pins=[]
    try:
        for i,kind,name in zip(arrays['scene'],arrays['anchor'],arrays['candidate']):
            scene=native_root/f'scene-{i:02d}'
            if current_scene!=int(i):
                if adapter:adapter.close()
                card=json.loads((scene/'construction/CARD.json').read_text());adapter=SourceAdapter(card,spec['asset_root'],executor_id=spec['executor_id'],seed=card['seed']);constructors+=1;current_scene=int(i)
                if adapter.engine.steps!=0:raise RuntimeError('Source constructor unexpectedly integrated')
            parent=scene/'branches'/str(kind);h=read_rows(parent/'COMMON_HISTORY.pkl.gz')['rows'];p=parent/(str(name)+'.pkl.gz');rows=read_rows(p)['rows'];seq=h+rows
            if len(seq)!=24 or any(r['context']['model_sha256']!=adapter.model_sha256 for r in seq):raise ValueError('Original native model/history differs')
            supports.append([supported(adapter,r['state_after'])for r in seq]);pins.append(digest(p))
    finally:
        if adapter:adapter.close()
    arrays['support']=np.array(supports,dtype=bool);arrays['parent_replay_sha256']=np.array(spec['replay_sha256']);arrays['purpose']=np.array(PURPOSE)
    with(output/'SUPPORTED.npz').open('xb')as f:np.savez_compressed(f,**arrays)
    result=dict(purpose=PURPOSE,parent_replay_sha256=spec['replay_sha256'],derived_sha256=digest(output/'SUPPORTED.npz'),
        support_label='Original native table/cube contact pair and normal_force_N>=0.05, no geometric proxy',source_model_constructors=constructors,
        original_stream_hashes=pins,new_physics=0,new_updates=0,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    write_json(output/'AUDIT.json',result);return result


def evaluate(spec,output):
    import torch
    from .decision_training import read,evaluate as score
    from .decision_diagnostics import frozen_model
    from .contact_training import select
    from .neural_training import parameter_hash
    output=Path(output);torch.set_num_threads(len(spec['cpus']));start=time.monotonic();cpu=time.process_time()
    if digest(spec['support_replay'])!=spec['support_replay_sha256']:raise ValueError('Support derivative differs')
    with np.load(spec['support_replay'],allow_pickle=False)as f:
        if str(f['parent_replay_sha256'])!=spec['replay_sha256']:raise ValueError('Original training data link differs')
    data,meta=read(spec['support_replay'],'cpu',purpose=PURPOSE);models=[frozen_model(spec['models'][n])for n in ('action_conditioned','no_future_action')];pins=[parameter_hash(m)for m in models];result={}
    for part in ('DIAGNOSTIC_DEV','ONCE_HOLDOUT'):
        ids=np.flatnonzero(meta['partition']==part);folder=output/part;folder.mkdir()
        result[part]=score(models,select(data,torch.tensor(ids)),{k:v[ids]for k,v in meta.items()},folder)
    if pins!=[parameter_hash(m)for m in models]:raise RuntimeError('Scoring changed frozen models')
    result=dict(status='V2_NATIVE_SUPPORT_RESCORE_FIXED200_MODELS',source_sha=spec['source_sha'],original_data_sha256=spec['replay_sha256'],support_data_sha256=spec['support_replay_sha256'],
        evaluation=result,preserves='V1 release score used illegal_contact as support and is INVALID. V2 reuses exact saved models/data and native contacts, no retraining or selection; original once-heldout is now corrected-rescore data, not a new blind test.',
        WM_updates=0,Actor_updates=0,new_physics=0,parameters_unchanged=True,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start)
    write_json(output/'RESULT.json',result);return result
