"""Seal read-only original SMALL8/probe inputs and the resolved historical recipe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from scripts.pi0292.durable import read,write
from scripts.pi0294.native_dataset import sha
from .dataset import select_small8
from .scope import validate,Ledger,STATS,DATA

RECIPE_SHA='67319790d4391dcb670577a3ace7dac592a1e2c0ce0ece6d54b682646a3ab53b'


def prepare(root):
    root=Path(root);validate(root,'hopper-neural');ledger=Ledger(root,'hopper')
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    start=time.process_time();inputs=read(root/'E1_INPUTS.private.json')
    b=read(root/'NEURAL_BINDINGS.private.json');data=Path(b['dataset'])
    assert sha(data/'DATASET_MANIFEST.json')==DATA and sha(data/'statistics.json')==STATS
    assert sha(inputs['original_recipe'])==RECIPE_SHA
    original=read(inputs['original_recipe'])
    assert original['effective_batch']==32 and original['per_device_train_batch_size']*original['gradient_accumulation_steps']==32
    assert original['numerical_configuration']['mode']=='DETERMINISTIC_CUDA_MATH_SDP_V1'
    rows=select_small8(read(data/'DATASET_MANIFEST.json'));bindings=read(b['raw_bindings'])
    episodes=[];probes=[]
    for row in rows:
        numeric=data/row['file'];assert sha(numeric)==row['sha256']
        raw=Path(bindings.get(row['id'],row['raw_root']))
        with np.load(numeric,allow_pickle=False) as z:
            assert z['label_representation'].item()=='ABSOLUTE_COMMITTED_EEF10'
            valid=np.flatnonzero(z['valid_window'])
            assert len(valid)==row['windows'] and np.array_equal(valid,np.arange(row['windows']))
            assert z['committed_absolute10'].shape[1:]==(10,) and np.isfinite(z['committed_absolute10']).all()
            chosen=[a for a in [0,80,160,240,320,400,480,560] if a<row['windows']]
            for phase in ['contact','transport','withdraw','hold']:
                if row['stage_anchors'].get(phase):
                    a=min(row['stage_anchors'][phase])
                    if a not in chosen:chosen.append(a)
            for anchor in sorted(chosen):
                assert anchor in valid and anchor+40<=len(z['committed_absolute10'])
                images=[]
                for camera in range(2):
                    p=raw/str(z['rgb_paths'][anchor,camera]);assert p.resolve().is_relative_to(raw.resolve())
                    assert sha(p)==str(z['rgb_sha256'][anchor,camera]);images.append(sha(p))
                command=z['committed_absolute10'][anchor:anchor+40]
                probes.append(dict(id=row['id']+'-A'+str(anchor).zfill(3),episode_id=row['id'],task=row['task'],split='TRAIN',anchor=anchor,
                    numeric_sha256=row['sha256'],state_sha256=hashlib.sha256(z['actual_state10'][anchor].tobytes()).hexdigest(),
                    absolute_H40_sha256=hashlib.sha256(command.tobytes()).hexdigest(),RGB_sha256=images,
                    stage=next(p for p,anchors in row['stage_anchors'].items() if anchor in anchors),
                    noise_seed=int.from_bytes(hashlib.sha256(('E1_PROBE/'+row['id']+'/'+str(anchor)).encode()).digest()[:8],'big')%(2**63)))
        result=read(raw/'RESULT.json');score=read(raw/'SCORE.json')
        assert result['status']=='VALID_COMPLETE' and score['persistent_success'] is True
        episodes.append({k:row[k] for k in ['id','task','family','split','file','sha256','bytes','windows','source_sha','samples_sha256','physics_sha256']})
    assert len(probes)<=96 and len(episodes)==8
    resolved=dict(original_recipe_sha256=RECIPE_SHA,original_recipe=original,
        new_optimizer=True,exact_resume=False,parent='M2048_INFERENCE_WEIGHTS_ONLY',
        learning_rate=1e-5,end_learning_rate=1e-6,scheduler='COSINE_MIN_LR_FLOOR',warmup_updates=32,
        max_steps=1024,seed=20261016,effective_batch=32,
        per_device_train_batch_size=original['per_device_train_batch_size'],gradient_accumulation_steps=original['gradient_accumulation_steps'],
        workers=original['workers'],optimizer=original['optimizer'],weight_decay=original['weight_decay'],
        adam_beta1=original['adam_beta1'],adam_beta2=original['adam_beta2'],adam_epsilon=original['adam_epsilon'],
        max_grad_norm=original['max_grad_norm'],bf16=original['bf16'],tf32=original['tf32'],
        gradient_checkpointing=original['gradient_checkpointing'],numerical_configuration=original['numerical_configuration'],
        trainable_modules='EXACT_ORIGINAL_PROJECTOR_ACTION_DIT_MAP',sampling='TASK_1_TO_1_EPISODE_UNIFORM_WINDOW_UNIFORM',
        VLM_frozen=True,statistics_fit=False)
    out=root/'plans'
    for name,value in [('SMALL8_BINDING.json',dict(schema='E1_SMALL8_BINDING_V1',original_manifest_sha256=DATA,statistics_sha256=STATS,
            selection='MANDATORY_F0_F1_000_THEN_NEXT_LEXICOGRAPHIC_COMPLETE_ADMITTED_PAIR_PER_FAMILY',episodes=episodes,original_data_rewritten=False)),
        ('PROBE_MANIFEST.json',dict(schema='E1_SOURCE_PROBE_MANIFEST_V1',windows=probes,count=len(probes),selection='BASE_8_ANCHORS_PLUS_FIRST_VALID_CONTACT_TRANSPORT_WITHDRAW_HOLD_ANCHORS_DEDUP',TRAIN_only=True)),
        ('M1_RECIPE_RESOLVED.json',resolved)]:
        if (out/name).exists():raise FileExistsError('PRESERVE_EXISTING_PRE_GPU_INPUT_SEAL')
        write(out/name,value)
    cpu=time.process_time()-start;ledger.cpu(cpu)
    result=dict(status='PASS_ACTUAL_SMALL8_SUCCESS_BYTES_AND_PROBES_RECIPE_SEALED',episodes=8,probes=len(probes),
                CPU_s=cpu,model_loaded=False,optimizer_updates=0,files={n:sha(out/n) for n in ['SMALL8_BINDING.json','PROBE_MANIFEST.json','M1_RECIPE_RESOLVED.json']})
    write(root/'evidence/INPUT_PREPARATION.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();print(json.dumps(prepare(a.root)),flush=True)
