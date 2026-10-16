"""E1 model loading and fixed TRAIN probes using original official primitives."""
import base64
import gc
import hashlib
import io
from pathlib import Path
import uuid

import numpy as np
from PIL import Image

from scripts.pi0292.durable import read,write
from scripts.pi0292.eval_contract import FRAME,TCP,gate_action,error_metrics,expert_decode
from scripts.pi0294.native_dataset import official_import,sha
from scripts.pi0294.resident import ResidentModel
from scripts.pi0294.neural import COSMOS_REVISION
from scripts.pi0294.local_backbone import local_model_name
from .artifacts import atomic_npz
from .scope import STATS,DATA


def load(root,checkpoint,identity,*,trainable=True):
    root=Path(root);checkpoint=Path(checkpoint);bindings=read(root/'NEURAL_BINDINGS.private.json')
    if sha(Path(bindings['dataset'])/'DATASET_MANIFEST.json')!=DATA or sha(Path(bindings['dataset'])/'statistics.json')!=STATS:
        raise ValueError('E1_ORIGINAL_DATA_STATS_CHANGED')
    for name,digest in identity['files'].items():
        p=checkpoint/name
        if not p.resolve().is_relative_to(checkpoint.resolve()) or sha(p)!=digest:raise ValueError('E1_CHECKPOINT_CHANGED:'+name)
    official_import(bindings['source'])
    import torch
    from transformers import AutoModel
    import gr00t.model.gr00t_n1d7.gr00t_n1d7
    from scripts.pi0292.train_adapter import build_processor
    model,info=AutoModel.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=True,output_loading_info=True,
        model_name=local_model_name(root,bindings),transformers_loading_kwargs=dict(local_files_only=True,
            trust_remote_code=True,cache_dir=bindings['cosmos_hub'],revision=COSMOS_REVISION),
        tune_llm=False,tune_visual=False,tune_projector=True,tune_diffusion_model=True,tune_vlln=True,
        state_dropout_prob=0.,backbone_trainable_params_fp32=True,load_bf16=False)
    if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys']):raise ValueError('E1_EXACT_FULL_M2048_LOAD_REQUIRED')
    if model.action_head.num_inference_timesteps!=4:raise ValueError('E1_FOUR_STEP_SAMPLER_REQUIRED')
    model.to('cuda').eval()
    if not trainable:model.requires_grad_(False)
    instance=ResidentModel.__new__(ResidentModel);instance.root=root;instance.bindings=bindings;instance.model=model
    instance.processor=build_processor(bindings['source'],bindings['cosmos'],read(Path(bindings['dataset'])/'statistics.json'))
    instance.identity=identity
    return instance


def payload(state,images,text):
    rows=[]
    for name,array in zip(['external_rgb','wrist_rgb'],images):
        stream=io.BytesIO();Image.fromarray(array).save(stream,format='PNG');raw=stream.getvalue()
        rows.append(dict(camera=name,png_base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest()))
    return dict(snapshot_uuid=str(uuid.uuid4()),state_eef10=np.asarray(state).tolist(),images=rows,language=text,frame=FRAME,tcp=TCP)


def probes(root,label,instance,dataset,ledger,guard):
    root=Path(root);folder=root/'evidence'/label/'probes';folder.mkdir(parents=True,exist_ok=False)
    manifest=read(root/'plans/PROBE_MANIFEST.json');indices={(r['id'],a):i for i,(r,a) in enumerate(dataset.rows)}
    records=[]
    for row in manifest['windows']:
        guard();state,expert,images,text=dataset.raw(indices[row['episode_id'],row['anchor']])
        request=payload(state,images,text);ledger.charge('generation_queries',1)
        value=instance.generate(request,row['noise_seed']);physical=value['physical_H40']
        decoded,clip=expert_decode(instance.processor,state,expert)
        atomic_npz(folder/(row['id']+'.npz'),physical_H40=physical,normalized=value['normalized'],expert_absolute10=expert,
            expert_native_roundtrip=decoded,state=state)
        records.append(dict(id=row['id'],task=row['task'],episode_id=row['episode_id'],anchor=row['anchor'],split='TRAIN',
            errors=error_metrics(physical,expert,state),gate=gate_action(physical,state,row['anchor']*.05,row['id']),
            native_expert_roundtrip_maximum_absolute_difference=float(np.max(np.abs(decoded-expert))),expert_clip=clip,
            noise=value['noise'],seed=row['noise_seed'],arrays_sha256=sha(folder/(row['id']+'.npz'))))
        write(folder/'ON_EXPERT_RESULTS.json',dict(status='ACTUAL_NATIVE_TRAIN_PROBES_NOT_TASK_SUCCESS',records=records,complete=len(records)==manifest['count']))
    students=read(root/'plans/STUDENT_PROBES.private.json');student_records=[]
    for row in students['states']:
        guard()
        if label=='M0':
            student_records.append(dict(id=row['id'],new_query=False,origin='ORIGINAL_E0_GENERATION_REUSED',
                original_action_sha256=row['original_action_sha256'],physical_H40=row['M0_physical_H40'],noise_seed=row['noise_seed']))
        else:
            ledger.charge('generation_queries',1);value=instance.generate(row['payload'],row['noise_seed'])
            atomic_npz(folder/(row['id']+'.npz'),physical_H40=value['physical_H40'],normalized=value['normalized'])
            student_records.append(dict(id=row['id'],new_query=True,noise=value['noise'],noise_seed=row['noise_seed'],
                physical_H40=value['physical_H40'].tolist(),arrays_sha256=sha(folder/(row['id']+'.npz')),
                gate=gate_action(value['physical_H40'],row['payload']['state_eef10'],row['tick']*.002,row['id'])))
    write(folder/'ON_STUDENT_RESULTS.private.json',dict(status='FIXED_12_TRAIN_OBSERVATIONS_ONLY_NOT_LIVE_OR_TEACHER_LABELS',records=student_records))
    return dict(on_expert=len(records),on_student=len(student_records),new_queries=len(records)+(0 if label=='M0' else len(student_records)))


def unload(instance):
    import torch
    del instance.model;gc.collect();torch.cuda.empty_cache()
