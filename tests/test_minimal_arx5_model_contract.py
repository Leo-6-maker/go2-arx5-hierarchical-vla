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
