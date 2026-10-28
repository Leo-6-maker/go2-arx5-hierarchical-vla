"""Full-state qualification restore through the original Trainer methods.

This library grants no model/allocation permission and performs no optimizer
step or inference query. The native owner must still verify actual query/output
equivalence before declaring the complete real-model qualification successful.
"""
from pathlib import Path
import hashlib
import json
import math

from . import minimal_study as study

FILENAME = 'FULL_TRAINING_STATE_IDENTITY.json'
SCHEMA = 'ARX5_MINIMAL_ORIGINAL_TRAINER_FULL_STATE_IDENTITY_V1'


def tree_identity(value):
    """Device/dtype/value identity, including nested AdamW and scheduler state."""
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        data=value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return dict(kind='tensor',shape=list(value.shape),dtype=str(value.dtype),
                    device=str(value.device),sha256=hashlib.sha256(data).hexdigest())
    if isinstance(value,np.ndarray):
        study.need(not value.dtype.hasobject,'NO_OBJECT_ARRAY_IN_TRAINING_STATE')
        return dict(kind='numpy',shape=list(value.shape),dtype=str(value.dtype),
                    sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
    if isinstance(value,np.generic):return tree_identity(value.item())
    if isinstance(value,dict):
        study.need(all(type(k)in(str,int)for k in value),'EXPLICIT_TRAINING_STATE_KEYS')
        return dict(kind='dict',items=[dict(key_type=type(k).__name__,key=k,value=tree_identity(value[k]))
                    for k in sorted(value,key=lambda k:(type(k).__name__,repr(k)))])
    if isinstance(value,(tuple,list)):
        return dict(kind=type(value).__name__,items=[tree_identity(v)for v in value])
    study.need(value is None or type(value)in(str,int,bool,float),'SUPPORTED_FULL_STATE_VALUE')
    study.need(type(value)is not float or math.isfinite(value),'FINITE_FULL_TRAINING_STATE')
    return dict(kind=type(value).__name__,value=value)


def fingerprint(value):return study.digest(tree_identity(value))


def saved_rng(checkpoint):
    """Load only the caller's hash-validated, self-produced Trainer RNG member."""
    import torch
    path=Path(checkpoint)/'rng_state.pth'
    study.need(path.is_file()and not path.is_symlink(),'ORIGINAL_REGULAR_TRAINER_RNG_MEMBER')
    value=torch.load(path,map_location='cpu',weights_only=False)
    study.need(type(value)is dict and set(value)in(
        {'python','numpy','cpu'},{'python','numpy','cpu','cuda'}),'ORIGINAL_SINGLE_GPU_RNG_SCHEMA')
    return value


def current_rng(saved):
    import random
    import numpy as np
    import torch
    value=dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.random.get_rng_state())
    if 'cuda'in saved:
        study.need(torch.cuda.is_available()and torch.cuda.device_count()==1,'SAME_SINGLE_REAL_CUDA_RNG_DEVICE')
        value['cuda']=(torch.cuda.get_rng_state_all()if isinstance(saved['cuda'],list)
                       else torch.cuda.get_rng_state())
    return value


def optimizer_identity(trainer):
    names={id(p):name for name,p in trainer.model.named_parameters()}
    groups=[[names[id(p)]for p in group['params']]for group in trainer.optimizer.param_groups]
    flat=[name for group in groups for name in group]
    study.need(len(flat)==len(set(flat)) and set(flat)==set(trainer.parameter_contract.trainable_names),
               'EXACT_UNIQUE_OPTIMIZER_PARAMETER_GROUP_NAMES')
    return dict(optimizer_sha256=fingerprint(trainer.optimizer.state_dict()),
                scheduler_sha256=fingerprint(trainer.lr_scheduler.state_dict()),
                optimizer_parameter_groups=groups)


def save_identity(trainer,checkpoint):
    """Called after the original Trainer writes its complete optimizer and RNG."""
    study.need(trainer.optimizer is not None and trainer.lr_scheduler is not None
        and trainer.consumed==trainer.state.global_step*32,'ACTUAL_COMMITTED_FULL_TRAINER_STATE')
    return dict(schema=SCHEMA,arm=trainer.recipe['arm'],step=trainer.state.global_step,
        cursor=trainer.consumed,scheduler_horizon=1024,**optimizer_identity(trainer),
        rng_sha256=fingerprint(saved_rng(checkpoint)),
        sampler=trainer.sampler.state_dict(trainer.consumed),
        optimizer_parameter_names=list(trainer.parameter_contract.trainable_names),
        native_model_qualification_complete=False)


def restore_qualification(trainer,checkpoint,*,guard):
    """Restore own step2 for audit only; never turn qualification into formal512."""
    from scripts.pi0292.durable import read
    from .minimal_training import checkpoint_members
    from .minimal_adapter import frozen_fingerprint,parameter_state_fingerprint
    study.need(callable(guard),'CURRENT_ORIGINAL_NATIVE_OWNER_GUARD_REQUIRED')
    guard()
    root=Path(checkpoint);expected=Path(trainer.args.output_dir)/'checkpoint-2'
    study.need(root.is_dir()and not root.is_symlink()and root.resolve()==expected.resolve()
        and trainer.qualification_only is True and trainer.session_stop==2
        and trainer.args.max_steps==1024 and trainer.args.gradient_accumulation_steps==16
        and trainer.args.per_device_train_batch_size==2 and getattr(trainer.args,'world_size',1)==1
        and not getattr(trainer,'_train_invoked',False)
        and not getattr(trainer,'_qualification_restore_audit',False),
        'ONLY_FRESH_OWN_QUALIFICATION_RESTORE_NOT_FORMAL_OR_REPEAT')
    complete=read(root/'CHECKPOINT_COMPLETE.json');files=checkpoint_members(root,complete['files'])
    required={'config.json','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json',
        'SAMPLER_STATE.json','statistics.json','ADAPTER_PARAMETER_CONTRACT.json',
        'FROZEN_PARAMETER_IDENTITY.json','MODEL_PARAMETER_STATE_IDENTITY.json',FILENAME}
    study.need(complete.get('qualification_only')is True and complete.get('global_step')==2
        and complete.get('full_resume')is True and complete.get('scheduler_horizon')==1024
        and complete.get('training_identity')==trainer.recipe['arm']
        and required<=set(files) and any(n.endswith('.safetensors')for n in files)
        and any(n.startswith('processor/')for n in files)
        and files==complete['files'], 'COMPLETE_OWN_STEP2_QUALIFICATION_STATE')
    saved=read(root/FILENAME);sampler=read(root/'SAMPLER_STATE.json')
    expected_sampler=trainer.train_dataset.make_sampler(trainer.train_dataset,start=64,
        stop=32768,seed=trainer.args.data_seed)
    study.need(saved['schema']==SCHEMA and saved['arm']==trainer.recipe['arm']
        and saved['step']==2 and saved['cursor']==64 and saved['scheduler_horizon']==1024
        and saved['optimizer_parameter_names']==list(trainer.parameter_contract.trainable_names)
        and sampler['recipe']==trainer.recipe
        and sampler['sampler']==saved['sampler']==expected_sampler.state_dict(64)
        and sampler['manifest_sha256']==trainer.train_dataset.manifest_sha
        and sampler['statistics_sha256']==study.R0 and files['statistics.json']==study.R0
        and read(root/'ADAPTER_PARAMETER_CONTRACT.json')==trainer.parameter_contract.describe()
        and read(root/'FROZEN_PARAMETER_IDENTITY.json')==trainer.frozen_baseline,
        'EXACT_RECIPE_R0_CATEGORY_OPTIMIZER_ORDER_AND_CURSOR64')
    effects=trainer.known_effects()
    study.need(effects['optimizer_updates']==2 and effects['pending_updates']==effects['unknown_effects']==[],
        'RESTORE_ONLY_AFTER_TWO_KNOWN_COMMITTED_QUALIFICATION_UPDATES')
    # Mark before loading: a failed/unknown restore is not automatically retried.
    trainer._qualification_restore_audit=True
    guard();trainer.create_optimizer_and_scheduler(num_training_steps=1024)
    trainer._load_from_checkpoint(str(root))
    trainer._load_optimizer_and_scheduler(str(root))
    trainer.state=type(trainer.state).load_from_json(str(root/'trainer_state.json'))
    trainer.start_cursor=trainer.consumed=64;trainer.sampler=expected_sampler
    trainer.parameter_contract.validate();trainer.parameter_contract.enforce_modes()
    study.need(trainer.state.global_step==2
        and optimizer_identity(trainer)=={k:saved[k]for k in
            ('optimizer_sha256','scheduler_sha256','optimizer_parameter_groups')}
        and frozen_fingerprint(trainer.model)==trainer.frozen_baseline
        and parameter_state_fingerprint(trainer.model)==read(root/'MODEL_PARAMETER_STATE_IDENTITY.json'),
        'ACTUAL_RELOADED_OPTIMIZER_SCHEDULER_MODEL_AND_FROZEN_VALUES_MATCH')
    rng=saved_rng(root)
    # Last state-changing operation: original framework restores Python, NumPy,
    # CPU and actual CUDA RNG. Hashing does not advance their random streams.
    trainer._load_rng_state(str(root))
    study.need(fingerprint(current_rng(rng))==saved['rng_sha256'], 'ACTUAL_ORIGINAL_FRAMEWORK_RNG_RESTORED')
    guard();study.need(checkpoint_members(root,complete['files'])==files,'CHECKPOINT_UNCHANGED_DURING_FULL_RESTORE')
    return dict(status='PASS_ORIGINAL_TRAINER_OWN_QUALIFICATION_FULL_STATE_RESTORE_ONLY',
        arm=trainer.recipe['arm'],step=2,cursor=64,scheduler_horizon=1024,
        optimizer_scheduler_model_RNG_sampler_checked=True,new_optimizer_updates=0,new_queries=0,
        qualification_never_becomes_formal=True,original_fixed_draw_equivalence_still_required=True,
        native_model_qualification_complete=False)
