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
