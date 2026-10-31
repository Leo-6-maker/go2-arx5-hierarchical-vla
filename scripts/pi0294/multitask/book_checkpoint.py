"""Strict wrapped-model restore, including the official tied language output."""
import json
from pathlib import Path

LM_HEAD='backbone.model.lm_head.weight'


def validate_missing(model_keys,saved_keys,proven_aliases):
    missing=set(model_keys)-set(saved_keys);unexpected=set(saved_keys)-set(model_keys)
    if unexpected or missing not in(set(),{LM_HEAD}):
        raise ValueError('BOOK_RESTORE_NO_UNEXPLAINED_MISSING_OR_UNEXPECTED_KEYS')
    if missing and not any(name in saved_keys and name.endswith('embed_tokens.weight')for name in proven_aliases):
        raise ValueError('BOOK_RESTORE_LM_HEAD_MUST_SHARE_THE_ACTUAL_SAVED_EMBEDDING_PARAMETER')
    return missing


def load_wrapped(model,checkpoint):
    from transformers.modeling_utils import load_sharded_checkpoint
    root=Path(checkpoint);index=json.loads((root/'model.safetensors.index.json').read_bytes())
    saved=set(index['weight_map']);keys=set(model.state_dict());aliases=[]
    if LM_HEAD in keys-saved:
        head=model.get_parameter(LM_HEAD)
        aliases=[name for name,p in model.named_parameters(remove_duplicate=False)if name!=LM_HEAD and p is head]
    missing=validate_missing(keys,saved,aliases)
    result=load_sharded_checkpoint(model,str(root),strict=not missing,prefer_safe=True)
    if set(result.missing_keys)!=missing or result.unexpected_keys:
        raise ValueError('BOOK_RESTORE_ACTUAL_RESULT_MUST_MATCH_EXACT_PROVEN_ALIAS')
    if missing and not any(model.get_parameter(LM_HEAD)is model.get_parameter(name)for name in aliases if name in saved):
        raise ValueError('BOOK_RESTORE_ACTUAL_PARAMETER_ALIAS_CHANGED')
    return dict(all_non_alias_model_members_strict=True,missing_serialized_keys=sorted(missing),
        actual_shared_saved_parameter_aliases=aliases,arbitrary_missing_keys_allowed=False)


def validate_delta_header(value,parent_sha256):
    required={'target_parameters.pt','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json',
        'statistics.json','SAMPLER_STATE.json','FROZEN_PARENT_TENSORS.json','ADAPTER_PARAMETER_CONTRACT.json'}
    if(value.get('schema')!='BOOK_TRAIN16_PARENT_BOUND_TARGET_DELTA_V1'or value.get('global_step')!=256
            or value.get('cursor')!=8192 or value.get('standalone_model')is not False
            or value.get('full_resume_with_required_parent')is not True
            or value.get('required_parent_checkpoint_manifest_sha256')!=parent_sha256
            or not required<=set(value.get('files',{}))):
        raise PermissionError('EXACT_RETAINED_PARENT_AND_COMPLETE_FIT256_DELTA_REQUIRED')
    return value


def load_parent_delta(model,contract,checkpoint,*,manifest_sha256,parent_sha256):
    """Apply only after strict parent reconstruction and producer qualification.

    The caller separately verifies the producer's CLOSED lease and native
    model/optimizer/scheduler/RNG/cursor qualification, not an old PENDING flag.
    """
    import torch
    from scripts.pi0292.durable import read
    from scripts.pi0294.native_dataset import sha
    from scripts.pi0294.native_train import tensor_identity
    from scripts.pi0294.multitask.book_fit import validate_delta
    from scripts.pi0294.generalization.minimal_adapter import frozen_fingerprint
    root=Path(checkpoint)
    if sha(root/'CHECKPOINT_COMPLETE.json')!=manifest_sha256:raise PermissionError('EXACT_FIT256_DELTA_MANIFEST')
    complete=validate_delta_header(read(root/'CHECKPOINT_COMPLETE.json'),parent_sha256)
    for name,pin in complete['files'].items():
        file=root/name
        if not file.resolve().is_relative_to(root.resolve())or sha(file)!=pin:raise PermissionError('FIT256_DELTA_MEMBER_CHANGED:'+name)
    contract.validate()
    if contract.describe()!=read(root/'ADAPTER_PARAMETER_CONTRACT.json'):raise PermissionError('FIT256_DELTA_SAME_WRAPPED_ARCHITECTURE')
    frozen=frozen_fingerprint(model)
    if frozen!=read(root/'FROZEN_PARENT_TENSORS.json'):raise PermissionError('FIT256_DELTA_EXACT_FROZEN_PARENT_TENSORS')
    delta=torch.load(root/'target_parameters.pt',map_location='cpu',weights_only=True);params=dict(model.named_parameters())
    validate_delta(delta,params,contract.trainable_names)
    if not all(torch.isfinite(x).all()for x in delta.values()):raise ValueError('FIT256_FINITE_TARGET_TENSORS')
    with torch.no_grad():
        for name in contract.trainable_names:params[name].copy_(delta[name])
    if frozen_fingerprint(model)!=frozen:raise ValueError('FIT256_INFERENCE_RECONSTRUCTION_CHANGED_FROZEN_PARENT')
    return dict(delta_checkpoint_manifest_sha256=manifest_sha256,parent_checkpoint_manifest_sha256=parent_sha256,
        exact_target_tensors={n:tensor_identity(params[n])for n in contract.trainable_names},
        frozen_parent_unchanged=True,standalone_model=False,inference_reconstruction_only=True)
