import pytest
from scripts.pi0294.multitask.book_checkpoint import validate_missing,LM_HEAD,validate_delta_header,qualification_file_name


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


def test_full_book_delta_requires_exact_recipe_and_committed_endpoint():
    from copy import deepcopy
    from scripts.pi0294.multitask.book_fit import full_plan
    p=full_plan()
    fields=('target_parameters.pt','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json',
        'statistics.json','SAMPLER_STATE.json','FROZEN_PARENT_TENSORS.json','ADAPTER_PARAMETER_CONTRACT.json')
    value=dict(schema='BOOK_PARENT_BOUND_TARGET_DELTA_V2',global_step=1024,cursor=32768,recipe=p,
        standalone_model=False,full_resume_with_required_parent=True,
        required_parent_checkpoint_manifest_sha256=p['checkpoint_manifest_sha256'],files=dict.fromkeys(fields,'hash'))
    assert validate_delta_header(value,p['checkpoint_manifest_sha256'])is value
    for key,changed in [('global_step',256),('cursor',8192),('schema','UNKNOWN'),('standalone_model',True)]:
        bad=deepcopy(value);bad[key]=changed
        with pytest.raises(PermissionError):validate_delta_header(bad,p['checkpoint_manifest_sha256'])
    for key,changed in [('DEV_or_TEST_used',True),('shared_DiT_frozen',False),('TRAIN_windows',16)]:
        bad=deepcopy(value);bad['recipe'][key]=changed
        with pytest.raises(PermissionError):validate_delta_header(bad,p['checkpoint_manifest_sha256'])


def test_recovery_delta_requires_new_population_statistics_and_committed512():
    from copy import deepcopy
    from scripts.pi0294.multitask.book_fit import recovery_plan
    p=recovery_plan()
    fields=('target_parameters.pt','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json',
        'statistics.json','SAMPLER_STATE.json','FROZEN_PARENT_TENSORS.json','ADAPTER_PARAMETER_CONTRACT.json','TRAIN_DEV_SEAL.json')
    value=dict(schema='BOOK_PARENT_BOUND_TARGET_DELTA_V2',global_step=512,cursor=16384,recipe=p,
        standalone_model=False,full_resume_with_required_parent=True,
        required_parent_checkpoint_manifest_sha256=p['checkpoint_manifest_sha256'],files=dict.fromkeys(fields,'hash'))
    validate_delta_header(value,p['checkpoint_manifest_sha256'])
    for key,changed in [('statistics_sha256',p['parent_restore_statistics_sha256']),('TRAIN_windows',3005),('shared_DiT_frozen',False)]:
        bad=deepcopy(value);bad['recipe'][key]=changed
        with pytest.raises(PermissionError):validate_delta_header(bad,p['checkpoint_manifest_sha256'])
    bad=deepcopy(value);bad['cursor']=32768
    with pytest.raises(PermissionError):validate_delta_header(bad,p['checkpoint_manifest_sha256'])
    bad=deepcopy(value);del bad['files']['TRAIN_DEV_SEAL.json']
    with pytest.raises(PermissionError):validate_delta_header(bad,p['checkpoint_manifest_sha256'])


def test_intermediate_curve_requires_its_own_exact_full_state_qualification_file():
    from scripts.pi0294.multitask.book_m32_learning import curve_plan,plan
    p=curve_plan('1'*64,'2'*64,'3'*64)
    for step in(1024,2048):
        c=dict(recipe=p,global_step=step)
        name=f'ACTUAL_DELTA_CHECKPOINT_RESTORE_{step}.json'
        assert qualification_file_name(dict(delta_native_qualification_file=name),c)==name
        with pytest.raises(PermissionError):qualification_file_name({},c)
        for invalid in('../'+name,'/tmp/'+name,'ACTUAL_DELTA_CHECKPOINT_RESTORE_4096.json',name.replace(str(step),'512')):
            with pytest.raises(PermissionError):qualification_file_name(dict(delta_native_qualification_file=invalid),c)
    assert qualification_file_name({},dict(recipe=p,global_step=4096))=='ACTUAL_DELTA_CHECKPOINT_RESTORE.json'
    short=plan('1'*64,'2'*64,'3'*64,blocks=2,steps=512)
    assert qualification_file_name({},dict(recipe=short,global_step=512))=='ACTUAL_DELTA_CHECKPOINT_RESTORE.json'
    with pytest.raises(PermissionError):qualification_file_name(dict(delta_native_qualification_file='ACTUAL_DELTA_CHECKPOINT_RESTORE_1024.json'),dict(recipe=short,global_step=512))
