"""Raw-parent-bound M32 restoration, without constructing a second model.

The seven full raw files are audited once at an immutable transfer boundary.
Only the 42 declared trainable tensors are read again at save/restore points;
the caller verifies the complete frozen tensor fingerprint independently.
"""
import json
import re
from pathlib import Path

from .raw_parent_delivery import REPO, REVISION, PARENT_ID

SCHEMA = 'SHARED4_RAW_PARENT_BOUND_M32_TRAINING_DELTA_V1'


def parameter_map(index):
    from scripts.pi0294.generalization.minimal_adapter import LEAVES
    mapping={path+'.target_'+suffix:(path+'.'+suffix,10)
             for path in LEAVES for suffix in ('W','b')}
    blocks={name:int(m[1])for name in index if
            (m:=re.match(r'^action_head\.model\.transformer_blocks\.(\d+)\.',name))}
    if set(blocks.values())!=set(range(32)):
        raise PermissionError('RAW_PINNED_32_DIT_BLOCKS')
    mapping.update({name:(name,None)for name,block in blocks.items()if block in(30,31)})
    if len(mapping)!=42 or not {key for key,slice_index in mapping.values()}<=set(index):
        raise PermissionError('EXACT_RAW_FOURTEEN_TARGET_AND_28_LAST_BLOCK_TENSORS')
    return mapping


def raw_trainables(root,audit,*,guard):
    """Read selected slices in their original dtype; never instantiate GR00T."""
    from safetensors import safe_open
    from scripts.pi0294.generalization.minimal_model import unchanged_raw_parent
    root=Path(root);unchanged_raw_parent(root,audit);guard()
    index=json.loads((root/'model.safetensors.index.json').read_bytes())['weight_map']
    mapping=parameter_map(index);out={}
    for shard in sorted({index[key]for key,part in mapping.values()}):
        if shard not in ('model-00001-of-00002.safetensors','model-00002-of-00002.safetensors'):
            raise PermissionError('ONLY_PINNED_RAW_SHARDS')
        with safe_open(root/shard,framework='pt',device='cpu')as source:
            for name,(key,part)in mapping.items():
                if index[key]!=shard:continue
                guard()
                value=source.get_tensor(key)if part is None else source.get_slice(key)[part]
                out[name]=value.clone()
    unchanged_raw_parent(root,audit)
    if sum(t.numel()for t in out.values())!=77847684:
        raise PermissionError('ACTUAL_RAW_M32_77847684_TRAINABLE_VALUES')
    return out


def raw_reference():
    from scripts.pi0294.generalization.minimal_model import FILES
    return dict(repo=REPO,revision=REVISION,raw_parent_files_identity=PARENT_ID,files=dict(FILES),
                standalone_model=False,adapted_parent_or_optimizer_inherited=False)


def raw_parameter_identity(model,contract,root,audit,*,guard,restore=False):
    import torch
    from scripts.pi0294.native_train import tensor_identity
    contract.validate();parameters=dict(model.named_parameters())
    values=raw_trainables(root,audit,guard=guard)
    if set(contract.trainable_names)!=set(values):
        raise PermissionError('EXACT_RAW_M32_PARAMETER_NAMES')
    proof={}
    for name,value in values.items():
        parameter=parameters[name]
        if value.shape!=parameter.shape:raise ValueError('RAW_PARAMETER_SHAPE:'+name)
        value=value.to(dtype=parameter.dtype)
        identity=tensor_identity(value)
        if restore:
            with torch.no_grad():parameter.copy_(value)
        if tensor_identity(parameter)!=identity:
            raise ValueError('INITIAL_PARAMETER_IS_NOT_ORIGINAL_RAW:'+name)
        proof[name]=identity
    return dict(schema='SHARED4_ACTUAL_ORIGINAL_RAW_M32_PARAMETERS_V1',
                **raw_reference(),trainable_tensors=proof,trainable_tensor_count=len(proof),
                trainable_numel=sum(t.numel()for t in values.values()),optimizer_updates=0,
                operation='RESTORE_RAW_TRAINABLES'if restore else'COMPARE_FRESH_LOADED_MODEL')


def validate_header(value):
    from .shared_training import is_recipe
    recipe=value.get('recipe',{});step=value.get('global_step')
    required={'target_parameters.pt','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json',
              'statistics.json','TRAIN_DEV_SEAL.json','TRAIN_POPULATION.json','TASK_DEV_POPULATION.json',
              'SAMPLER_STATE.json','FROZEN_PARENT_TENSORS.json','ADAPTER_PARAMETER_CONTRACT.json',
              'RAW_PARENT_REFERENCE.json','M32_MEMORY_EXECUTION.json'}
    if (value.get('schema')!=SCHEMA or not is_recipe(recipe)
            or step not in recipe['checkpoint_steps']or value.get('cursor')!=step*32
            or value.get('standalone_model')is not False or value.get('full_resume_with_required_parent')is not True
            or value.get('required_raw_parent_files_identity')!=PARENT_ID
            or not required<=set(value.get('files',{}))
            or value['files'].get('statistics.json')!=recipe['statistics_sha256']
            or value['files'].get('TRAIN_POPULATION.json')!=recipe['TRAIN_population_sha256']
            or value['files'].get('TASK_DEV_POPULATION.json')!=recipe['task_DEV_population_sha256']):
        raise PermissionError('COMPLETE_RAW_SHARED_DELTA_WITH_FIXED_DATA_AND_ALL_RESUME_STATE')
    return value


def load_delta(model,contract,checkpoint,*,manifest_sha256,raw_parent,audit,guard,inference_package_sha256=None):
    """Reconstruct a saved inference candidate after strict raw model loading.

    Producer CLOSED/full-state qualification and device inference admission
    remain the existing owner's responsibility. A native10 is never a policy.
    """
    import hashlib
    import torch
    from scripts.pi0294.multitask.book_fit import validate_delta
    from scripts.pi0294.multitask.book_m32_learning import LimitedContract
    from scripts.pi0294.generalization.minimal_adapter import frozen_fingerprint
    from scripts.pi0294.native_train import tensor_identity
    from scripts.pi0294.native_dataset import sha
    root=Path(checkpoint);body=(root/'CHECKPOINT_COMPLETE.json').read_bytes()
    if hashlib.sha256(body).hexdigest()!=manifest_sha256:raise PermissionError('EXACT_RAW_SHARED_DELTA_MANIFEST')
    complete=validate_header(json.loads(body))
    if complete['recipe']['capacity_qualification_only']:
        raise PermissionError('RAW_NATIVE10_IS_NOT_A_POLICY_CANDIDATE')
    members=complete['files']
    if inference_package_sha256 is not None:
        from .shared_delivery import validate,ARCHIVE_ONLY
        validate(root,manifest_sha256=inference_package_sha256,checkpoint_manifest_sha256=manifest_sha256)
        members={n:p for n,p in members.items()if n not in ARCHIVE_ONLY}
    for name,pin in members.items():
        guard();path=root/name
        if (path.is_symlink()or not path.resolve().is_relative_to(root.resolve())
                or sha(path)!=pin):
            raise PermissionError('RAW_SHARED_SAVED_MEMBER_CHANGED:'+name)
    if json.loads((root/'RAW_PARENT_REFERENCE.json').read_bytes())!=json.loads(json.dumps(raw_reference())):
        raise PermissionError('RAW_SHARED_EXACT_ORIGINAL_PARENT_REFERENCE')
    if len(contract.trainable_names)==14:contract=LimitedContract(contract,2)
    raw_parameter_identity(model,contract,raw_parent,audit,guard=guard)
    if contract.describe()!=json.loads((root/'ADAPTER_PARAMETER_CONTRACT.json').read_bytes()):
        raise PermissionError('RAW_SHARED_SAVED_M32_ARCHITECTURE')
    frozen=frozen_fingerprint(model)
    if frozen!=json.loads((root/'FROZEN_PARENT_TENSORS.json').read_bytes()):
        raise PermissionError('RAW_SHARED_ALL_FROZEN_PARAMETERS_AND_BUFFERS')
    values=torch.load(root/'target_parameters.pt',map_location='cpu',weights_only=True)
    parameters=dict(model.named_parameters());validate_delta(values,parameters,contract.trainable_names,expected_count=42)
    if not all(torch.isfinite(t).all()for t in values.values()):raise ValueError('RAW_SHARED_NONFINITE_DELTA')
    with torch.no_grad():
        for name in contract.trainable_names:parameters[name].copy_(values[name])
    if frozen_fingerprint(model)!=frozen:raise ValueError('RAW_SHARED_RECONSTRUCTION_CHANGED_FROZEN_PARENT')
    return dict(delta_checkpoint_manifest_sha256=manifest_sha256,required_raw_parent_files_identity=PARENT_ID,
        exact_trainable_tensors={n:tensor_identity(parameters[n])for n in contract.trainable_names},
        frozen_parent_unchanged=True,standalone_model=False,inference_reconstruction_only=True)
