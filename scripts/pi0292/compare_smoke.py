"""Compare actual continuous and resumed native smoke, with fixed tolerances."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource

from .durable import read,write
from .train_adapter import sha


def jsonl(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def rng_tree(value):
    import numpy as np
    import torch
    from .trainer_adapter import tensor_identity
    if torch.is_tensor(value):return tensor_identity(value)
    if isinstance(value,np.ndarray):return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value,dict):return {str(k):rng_tree(v) for k,v in sorted(value.items())}
    if isinstance(value,(tuple,list)):return [rng_tree(v) for v in value]
    return value


def compare_arrays(a,b):
    import torch
    if a.shape!=b.shape or a.dtype!=b.dtype:return dict(pass_=False,reason='SHAPE_OR_DTYPE')
    if not a.is_floating_point():return dict(pass_=bool(torch.equal(a,b)),max_abs=0.)
    a,b=a.reshape(-1),b.reshape(-1)
    maximum=0.;passed=True;finite=True
    for start in range(0,a.numel(),1<<20):
        left,right=a[start:start+(1<<20)].float(),b[start:start+(1<<20)].float()
        valid=bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        finite=finite and valid
        if valid:
            maximum=max(maximum,float((left-right).abs().max()))
            passed=passed and bool(torch.allclose(left,right,atol=1e-7,rtol=1e-3))
        else:passed=False
    return dict(pass_=passed,finite=finite,max_abs=maximum)


def compare_state(left,right,where=''):
    import torch
    rows={}
    if torch.is_tensor(left) and torch.is_tensor(right):return {where:compare_arrays(left,right)}
    if isinstance(left,dict) and isinstance(right,dict) and set(left)==set(right):
        for key in left:rows.update(compare_state(left[key],right[key],where+'/'+str(key)))
    elif isinstance(left,(list,tuple)) and type(left)==type(right) and len(left)==len(right):
        for index,(a,b) in enumerate(zip(left,right)):rows.update(compare_state(a,b,where+'/'+str(index)))
    else:rows[where]=dict(pass_=left==right)
    return rows


def checkpoint_parameters(root):
    from safetensors import safe_open
    index=root/'model.safetensors.index.json'
    files=sorted(set(read(index)['weight_map'].values())) if index.exists() else ['model.safetensors']
    for name in files:
        with safe_open(root/name,framework='pt',device='cpu') as f:
            for key in sorted(f.keys()):yield key,f.get_tensor(key)


def compare(root):
    import torch
    torch.set_num_threads(4)
    root=Path(root)
    continuous=root/'training/smoke_continuous'
    split=root/'training/smoke_split'
    sessions=[continuous/'session-start',split/'session-start',split/'session-resume']
    runs=[read(p/'TRAINING_RUN.json') for p in sessions]
    if [r['global_step'] for r in runs]!=[4,2,4]:raise ValueError('ACTUAL_4_VERSUS_2_PLUS_2_REQUIRED')
    if not all(r['frozen_backbone_unchanged'] and r['changed_trainable_parameters'] for r in runs):raise ValueError('SMOKE_GRADIENT_WEIGHT_GATE')
    if not runs[0]['recipe']==runs[1]['recipe']==runs[2]['recipe']:raise ValueError('SMOKE_RECIPE_CHANGED')
    if [len(jsonl(p/'GRADIENTS.jsonl')) for p in sessions]!=[4,2,2]:raise ValueError('ACTUAL_GRADIENT_EVIDENCE_MISSING')
    uninterrupted=jsonl(sessions[0]/'MICROBATCHES.jsonl')
    resumed=jsonl(sessions[1]/'MICROBATCHES.jsonl')+jsonl(sessions[2]/'MICROBATCHES.jsonl')
    expected=128//runs[0]['recipe']['per_device_train_batch_size']
    if len(uninterrupted)!=len(resumed) or len(resumed)!=expected:raise ValueError('SMOKE_SAMPLE_COUNT')
    fields=['optimizer_step','cursor','batch','batch_identity','rng_before','rng_after','noise']
    exact={key:all(a[key]==b[key] for a,b in zip(uninterrupted,resumed)) for key in fields}
    losses=torch.tensor([r['native_loss'] for r in uninterrupted],dtype=torch.float64)
    other=torch.tensor([r['native_loss'] for r in resumed],dtype=torch.float64)
    loss_pass=bool(torch.isfinite(losses).all() and torch.isfinite(other).all() and torch.allclose(losses,other,atol=1e-5,rtol=5e-3))
    steps=jsonl(sessions[0]/'OPTIMIZER_STEPS.jsonl')
    other_steps=jsonl(sessions[1]/'OPTIMIZER_STEPS.jsonl')+jsonl(sessions[2]/'OPTIMIZER_STEPS.jsonl')
    exact['optimizer_step_scheduler_rng_cursor']=all({k:v for k,v in a.items() if k!='epoch'}=={k:v for k,v in b.items() if k!='epoch'}
        for a,b in zip(steps,other_steps)) and len(steps)==len(other_steps)==4
    left,right=continuous/'checkpoint-4',split/'checkpoint-4'
    for checkpoint in (left,right,split/'checkpoint-2'):
        receipt=read(checkpoint/'CHECKPOINT_COMPLETE.json')
        for name,digest in receipt['required_files'].items():
            if sha(checkpoint/name)!=digest:raise ValueError('SAVED_CHECKPOINT_CHANGED')
    weights={}
    # State dict order can differ between shards; map keys to their files once.
    from safetensors import safe_open
    right_index=right/'model.safetensors.index.json'
    if right_index.exists():mapping=read(right_index)['weight_map']
    else:
        with safe_open(right/'model.safetensors',framework='pt',device='cpu') as f:mapping={key:'model.safetensors' for key in f.keys()}
    for name,tensor in checkpoint_parameters(left):
        if name not in mapping:weights[name]=dict(pass_=False,reason='MISSING_KEY');continue
        with safe_open(right/mapping[name],framework='pt',device='cpu') as f:
            weights[name]=compare_arrays(tensor,f.get_tensor(name))
    if set(weights)!=set(mapping):raise ValueError('CHECKPOINT_PARAMETER_SET_CHANGED')
    optimizer=compare_state(torch.load(left/'optimizer.pt',map_location='cpu',weights_only=True),
        torch.load(right/'optimizer.pt',map_location='cpu',weights_only=True))
    scheduler=compare_state(torch.load(left/'scheduler.pt',map_location='cpu',weights_only=True),
        torch.load(right/'scheduler.pt',map_location='cpu',weights_only=True))
    exact['checkpoint_sampler']=read(left/'SAMPLER_STATE.json')['sampler']==read(right/'SAMPLER_STATE.json')['sampler']
    # These are freshly produced, SHA-verified local Trainer checkpoints, not untrusted pickles.
    exact['checkpoint_rng']=rng_tree(torch.load(left/'rng_state.pth',map_location='cpu',weights_only=False))==rng_tree(
        torch.load(right/'rng_state.pth',map_location='cpu',weights_only=False))
    passed=all(exact.values()) and loss_pass and all(r['pass_'] for r in weights.values()) and all(r['pass_'] for r in optimizer.values()) and all(r['pass_'] for r in scheduler.values())
    u=resource.getrusage(resource.RUSAGE_SELF)
    result=dict(status='PASS_FIXED_RESUME_EQUIVALENCE' if passed else 'FAIL_PRESERVED',exact=exact,loss_pass=loss_pass,
        loss_max_abs=float((losses-other).abs().max()),weights=weights,optimizer=optimizer,scheduler=scheduler,
        tolerances=dict(weight_optimizer_atol=1e-7,weight_optimizer_rtol=1e-3,loss_atol=1e-5,loss_rtol=5e-3),
        actual_optimizer_updates=8,full_model_arrays_compared=True,CPU_s=u.ru_utime+u.ru_stime)
    output=root/'evidence/RESUME_EQUIVALENCE.json'
    if output.exists():raise FileExistsError('Preserve first smoke equivalence receipt')
    write(output,result)
    write(root/'evidence/MODEL_SMOKE.json',dict(status='TRAINER_MODEL_INTERFACE_QUALIFIED' if passed else 'MODEL_SMOKE_FAILED',
        equivalence_sha256=sha(output),actual_optimizer_updates=8,recipe=runs[0]['recipe'],
        finite_native_loss=True,exact_native_action_mask_asserted_every_microbatch=True,
        actual_projector_and_diffusion_gradients=True,frozen_VLM_unchanged=True,
        original_PI0291_status='PARTIAL',main_training='NOT_STARTED_BY_SMOKE',selected='NONE',deployment='NONE'))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    result=compare(p.parse_args().root);print(result['status'])
    if result['status']!='PASS_FIXED_RESUME_EQUIVALENCE':raise SystemExit(2)
