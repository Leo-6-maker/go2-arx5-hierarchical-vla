"""CPU input/release/loading contracts; never loads a pretrained model."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.pi0294.generalization import minimal_model as m
from scripts.pi0294.generalization import minimal_study as study


def release_fixture():
    request = dict(scope=m.SCOPE, raw_parent_files_identity=m.RAW_PARENT_FILES_ID,
                   statistics_sha256=study.R0, panel_sha256=m.PANEL_SHA,
                   request_id='CPU_REQUEST_FIXTURE_NOT_REGISTERED', submission_nonce='CPU_NONCE_FIXTURE')
    body = (json.dumps(request, indent=2) + '\n').encode()
    allocation = dict(status=m.LOAD_STATUS, job_id='CPU_PBS_FIXTURE', request_id=request['request_id'],
        submission_nonce=request['submission_nonce'], request_sha256=hashlib.sha256(body).hexdigest(),
        observed_epoch=100., allocation_end_epoch=200., native_proof=dict(complete_queue_verified=True,
        project_eligible=True, actual_project_quota_covers_remaining_output=True,
        whole_remaining_package_shortfalls={}, global_unknown_effects=[]))
    return request, body, allocation


def check(a, r, b, **kw):
    return m.check_release(a, r, request_bytes=b, job_id=kw.get('job_id', 'CPU_PBS_FIXTURE'),
                           hostname=kw.get('hostname', 'GN-A40-CPU-FIXTURE'), now=kw.get('now', 110.))


def test_request_binds_exact_file_bytes_not_a_different_json_serialization():
    r, b, a = release_fixture(); check(a, r, b)
    with pytest.raises(ValueError, match='FRESH_SAME'):
        check(a, r, study.raw(r))


@pytest.mark.parametrize('change', ['old_scope', 'old_status', 'wrong_job', 'login_node', 'stale', 'unknown', 'shortfall'])
def test_old_scope_stale_release_or_incomplete_resources_never_admit_model(change):
    r, b, a = release_fixture(); kw = {}
    if change == 'old_scope':
        r['scope'] = 'V5_OLD_C50'; b = json.dumps(r).encode(); a['request_sha256'] = hashlib.sha256(b).hexdigest()
    elif change == 'old_status': a['status'] = 'PASS_V5_CD_OWN_NATIVE_A40_ALLOCATION'
    elif change == 'wrong_job': kw['job_id'] = 'OTHER_CPU_FIXTURE'
    elif change == 'login_node': kw['hostname'] = 'stdct-login-01'
    elif change == 'stale': kw['now'] = 221.
    elif change == 'unknown': a['native_proof']['global_unknown_effects'] = ['UNKNOWN']
    elif change == 'shortfall': a['native_proof']['whole_remaining_package_shortfalls'] = {'GPU': 1}
    with pytest.raises(ValueError): check(a, r, b, **kw)


def test_full_input_audit_guard_and_stat_change_refusal(tmp_path, monkeypatch):
    # Tiny stand-ins exercise streaming mechanics only, not actual raw hashes.
    expected = {}
    for i, name in enumerate(m.FILES):
        data = ('CPU_FIXTURE_' + str(i)).encode(); (tmp_path/name).write_bytes(data)
        expected[name] = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(m, 'FILES', expected)
    calls = []
    audited = m.audit_raw_parent(tmp_path, guard=lambda: calls.append('guard'))
    assert len(calls) >= 7 and audited['models_loaded'] == 0
    m.unchanged_raw_parent(tmp_path, audited)
    (tmp_path/'config.json').write_bytes(b'CHANGED')
    with pytest.raises(ValueError, match='RAW_PARENT_CHANGED'):
        m.unchanged_raw_parent(tmp_path, audited)


def test_header_only_and_observation_timeout_cannot_substitute_for_full_stream(tmp_path):
    for status in ['ACTUAL_METADATA_AND_TARGET_ROW_BYTES', 'STATUS_OBSERVATION_TIMEOUT']:
        with pytest.raises(ValueError, match='ACTUAL_FULL_RAW_AUDIT'):
            m.unchanged_raw_parent(tmp_path, dict(status=status))


def test_partial_or_new_parameter_load_is_rejected_and_no_padding_truncation():
    model = SimpleNamespace(config=SimpleNamespace(action_horizon=40, max_action_dim=132,
        max_state_dim=132, max_num_embodiments=32), action_head=SimpleNamespace(
        num_inference_timesteps=4, state_dropout_prob=0.))
    m.validate_loading(model, {})
    for field in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']:
        with pytest.raises(ValueError, match='NO_SILENT'):
            m.validate_loading(model, {field: ['CPU_FIXTURE_MISSING_OR_CHANGED']})
    model.config.max_action_dim = 10
    with pytest.raises(ValueError, match='EXACT_H40'):
        m.validate_loading(model, {})


def test_loading_flags_are_identical_for_both_arms_and_keep_original_local_backbone():
    a = m.loader_kwargs('CPU_LOCAL_BACKBONE_FIXTURE', 'CPU_HUB_FIXTURE', 'CPU_REVISION_FIXTURE')
    assert a['state_dropout_prob'] == 0. and a['load_bf16'] is False
    assert a['tune_projector'] and a['tune_diffusion_model'] and a['tune_vlln']
    assert a['tune_llm'] is False and a['tune_visual'] is False
    assert a['local_files_only'] and a['transformers_loading_kwargs']['local_files_only']
    assert m.FILES['statistics.json'] != study.R0


def test_owner_rejection_happens_before_any_framework_or_checkpoint_loading():
    calls = []
    def guard():
        calls.append('owner'); raise PermissionError('CPU_OWNER_REJECTED_NO_NATIVE_SCOPE')
    with pytest.raises(PermissionError, match='OWNER_REJECTED'):
        m.load('NO_ROOT_OPENED', 'NO_REQUEST_OPENED', {}, {}, checkpoint='NO_CHECKPOINT_OPENED',
               audit={}, mode='E0_RAW', guard=guard)
    assert calls == ['owner']


def test_unload_releases_contract_reference_before_original_model_cleanup():
    import gc,weakref
    from types import SimpleNamespace
    from test_minimal_arx5_adapter_contract import fixture_pair
    _,model,_,contract,_=fixture_pair()
    weak=weakref.ref(model);instance=SimpleNamespace(model=model,parameter_contract=contract)
    del model,contract
    m.unload(instance);gc.collect()
    assert instance.parameter_contract is None and not hasattr(instance,'model')
    assert weak()is None


@pytest.fixture
def saved_tiny_model(tmp_path,monkeypatch):
    """Real tiny tensor values and file hashes; no GR00T/CUDA qualification."""
    from test_minimal_arx5_adapter_contract import fixture_pair
    from scripts.pi0294.generalization.minimal_adapter import parameter_state_fingerprint
    from scripts.pi0292.durable import write
    _,model,_,contract,_=fixture_pair()
    stats=b'{"CPU_FIXTURE_NOT_R0":true}\n';(tmp_path/'statistics.json').write_bytes(stats)
    monkeypatch.setattr(study,'R0',hashlib.sha256(stats).hexdigest())
    files=['optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','config.json',
        'SAMPLER_STATE.json','FROZEN_PARAMETER_IDENTITY.json','FULL_TRAINING_STATE_IDENTITY.json',
        'model.safetensors','processor/processor_config.json']
    for name in files:
        p=tmp_path/name;p.parent.mkdir(exist_ok=True);p.write_bytes(b'CPU_MEMBER_FIXTURE')
    write(tmp_path/'ADAPTER_PARAMETER_CONTRACT.json',contract.describe())
    write(tmp_path/'MODEL_PARAMETER_STATE_IDENTITY.json',parameter_state_fingerprint(model))
    members={str(p.relative_to(tmp_path)):hashlib.sha256(p.read_bytes()).hexdigest()for p in tmp_path.rglob('*')if p.is_file()}
    complete=dict(global_step=2,training_identity=contract.arm,qualification_only=True,full_resume=True,
        session_terminal=True,scheduler_horizon=1024,parameter_state_format='WRAPPED_CATEGORY_TABLES_FULL_RESUME_ONLY',files=members)
    write(tmp_path/'CHECKPOINT_COMPLETE.json',complete)
    instance=SimpleNamespace(model=model,parameter_contract=contract,identity=dict(mode=contract.arm,
        global_step=0,pretrained_revision=m.RAW_REVISION,statistics_sha256=study.R0,files=dict(m.FILES)))
    return instance,tmp_path,complete


def test_trained_or_restored_queries_bind_actual_saved_state_not_raw_parent_hashes(saved_tiny_model):
    instance,root,_=saved_tiny_model;raw=dict(instance.identity);calls=[]
    out=m.bind_saved_checkpoint(instance,root,guard=lambda:calls.append('CURRENT_OWNER'))
    assert calls==['CURRENT_OWNER','CURRENT_OWNER']
    assert instance.identity['global_step']==2 and instance.identity['files']!=raw['files']
    assert instance.identity['checkpoint_manifest_sha256']==hashlib.sha256((root/'CHECKPOINT_COMPLETE.json').read_bytes()).hexdigest()
    assert instance.identity['raw_parent_files_identity']==m.RAW_PARENT_FILES_ID
    assert instance.identity['raw_parent_files']==m.FILES
    assert out['actual_parameter_identity']


@pytest.mark.parametrize('change',['changed_memory','changed_member','other_arm','not_full','wrong_step','missing_optimizer','wrong_contract'])
def test_query_provenance_cannot_label_other_model_or_partial_state_as_trained_checkpoint(saved_tiny_model,change):
    from scripts.pi0292.durable import write
    instance,root,complete=saved_tiny_model;before=deepcopy(instance.identity)
    if change=='changed_memory':
        import torch
        with torch.no_grad():next(instance.model.parameters()).add_(1)
    elif change=='changed_member':(root/'model.safetensors').write_bytes(b'CHANGED')
    elif change=='other_arm':complete['training_identity']='E2_SHARED_ACTION_HEAD'
    elif change=='not_full':complete['full_resume']=False
    elif change=='wrong_step':complete['global_step']=1024
    elif change=='missing_optimizer':complete['files'].pop('optimizer.pt')
    else:
        path=root/'ADAPTER_PARAMETER_CONTRACT.json';write(path,dict(wrong=True));complete['files']['ADAPTER_PARAMETER_CONTRACT.json']=hashlib.sha256(path.read_bytes()).hexdigest()
    write(root/'CHECKPOINT_COMPLETE.json',complete)
    with pytest.raises(ValueError):m.bind_saved_checkpoint(instance,root,guard=lambda:None)
    assert instance.identity==before
