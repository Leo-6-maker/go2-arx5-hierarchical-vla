import pytest
from scripts.pi0294.multitask.book_checkpoint import validate_missing,LM_HEAD,validate_delta_header


def test_only_proven_saved_language_embedding_alias_can_be_omitted():
    embed='backbone.model.model.embed_tokens.weight';keys={embed,LM_HEAD,'action_head.weight'}
    assert validate_missing(keys,keys-{LM_HEAD},[embed])=={LM_HEAD}
    with pytest.raises(ValueError):validate_missing(keys,keys-{LM_HEAD},[])
    with pytest.raises(ValueError):validate_missing(keys,keys-{'action_head.weight'},[embed])
    with pytest.raises(ValueError):validate_missing(keys,keys|{'random.weight'},[embed])
    assert validate_missing(keys,keys,[])==set()


def test_parent_bound_delta_requires_full_state_and_exact_retained_parent():
    from copy import deepcopy
    fields=('target_parameters.pt','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json',
        'statistics.json','SAMPLER_STATE.json','FROZEN_PARENT_TENSORS.json','ADAPTER_PARAMETER_CONTRACT.json')
    c=dict(schema='BOOK_TRAIN16_PARENT_BOUND_TARGET_DELTA_V1',global_step=256,cursor=8192,standalone_model=False,
        full_resume_with_required_parent=True,required_parent_checkpoint_manifest_sha256='retained-parent',files=dict.fromkeys(fields,'hash'))
    validate_delta_header(c,'retained-parent')
    with pytest.raises(PermissionError):validate_delta_header(c,'another-parent')
    for key,value in [('standalone_model',True),('global_step',128),('cursor',4096),('full_resume_with_required_parent',False)]:
        bad=deepcopy(c);bad[key]=value
        with pytest.raises(PermissionError):validate_delta_header(bad,'retained-parent')
    bad=deepcopy(c);del bad['files']['optimizer.pt']
    with pytest.raises(PermissionError):validate_delta_header(bad,'retained-parent')
