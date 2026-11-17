import json

import pytest

from scripts.wm02 import mobile_evaluation_v52 as evaluation


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')
    return evaluation.pin(path)


def terminal(tmp_path):
    case_root = tmp_path/'cases'/'CONFIRM-F1-T0'
    case = dict(registered_root=str(case_root), sim_only_role='STUDENT', arm='B2',
        panel_entry_id='CONFIRM-F1-T0', mobile_condition=dict(family='F1', layout='T0'),
        source_sha='actual-runtime', model_path='scene', frozen_files={'scene': 'scene-sha'})
    case_pin = save(case_root/'EXECUTION_SPEC.json', case)
    save(case_root/'control/LEASE.json', dict(status='CLOSED', spec_sha256=case_pin['sha256']))
    actual = dict(physical_requests=1, actual_native_integrations=23703,
        technical_failure=False, unsafe=False, completed=True, source_sha='actual-runtime',
        detail=dict(terminal_reason='TASK_SUCCESS'),
        metrics=dict(task_success=True, task_and_required_motion_success=True))
    save(case_root/'RESULT.private.json', actual)
    row = evaluation.compact_case(case, actual)
    spec = dict(registered_root=str(tmp_path), source_sha='actual-runtime',
        cases=[case_pin['path']])
    spec_pin = save(tmp_path/'EXECUTION_SPEC.json', spec)
    save(tmp_path/'control/LEASE.json', dict(status='CLOSED', spec_sha256=spec_pin['sha256']))
    result_pin = save(tmp_path/'RESULT.private.json', dict(source_sha='actual-runtime',
        attempted_case_launches=1, new_verified_native_requests=1,
        unresolved_case_launches=0, unstarted_cases=0, records=[row]))
    return spec_pin, result_pin, case_root


def test_predecessor_is_verified_from_real_closed_case_not_a_passed_flag(tmp_path):
    spec_pin, result_pin, case_root = terminal(tmp_path)
    _, rows = evaluation.verified_terminal(spec_pin, result_pin, 1)
    assert rows[0]['task_and_mobile_success']
    raw = json.loads((case_root/'RESULT.private.json').read_text())
    raw['metrics']['task_and_required_motion_success'] = False
    save(case_root/'RESULT.private.json', raw)
    with pytest.raises(ValueError, match='CASE_IDENTITY_OR_CONTROL'):
        evaluation.verified_terminal(spec_pin, result_pin, 1)


def test_changed_case_spec_and_open_parent_are_not_accepted_as_old_evidence(tmp_path):
    spec_pin, result_pin, case_root = terminal(tmp_path)
    original = (case_root/'EXECUTION_SPEC.json').read_text()
    case = json.loads(original)
    case['mobile_condition']['layout'] = 'T1'
    save(case_root/'EXECUTION_SPEC.json', case)
    with pytest.raises(ValueError, match='CASE_IDENTITY_OR_CONTROL'):
        evaluation.verified_terminal(spec_pin, result_pin, 1)
    (case_root/'EXECUTION_SPEC.json').write_text(original)
    save(tmp_path/'control/LEASE.json', dict(status='ACTIVE', spec_sha256=spec_pin['sha256']))
    with pytest.raises(ValueError, match='CLOSED_ACTUAL_PREDECESSOR'):
        evaluation.verified_terminal(spec_pin, result_pin, 1)


def test_successor_keeps_native_profile_candidate_restore_and_case_isolation(tmp_path, monkeypatch):
    root = tmp_path/'new-confirmation'
    packages = {a: dict(path=a, sha256=a+'-model') for a in ('B0', 'B1', 'B2')}
    templates, panel = {}, []
    for family in ('F1', 'F2'):
        template = dict(python='/frozen-native/bin/python', arm='B2',
            mobile_condition=dict(family=family, mode='M_FREE', method='SHIFT7_SINGLE'),
            max_ticks=35703, physics_dt_s=.002, model_service_timeout_s=30,
            packages=packages, SHIFT7_package=packages['B2'], model_path='old-scene',
            frozen_files={'old-scene': 'old', 'native': 'frozen-binary'},
            ordinary_RGB_restore=dict(path='actual-B2-restore', sha256='B2-restore-sha'),
            native_environment={'CUDA_VISIBLE_DEVICES': '', 'ROS_LOCALHOST_ONLY': '1'},
            mobile_world_lock='same-world', sim_only_role='STUDENT')
        templates[family] = (template, save(tmp_path/(family+'-template.json'), template))
        for i in range(4):
            panel.append(dict(id='CONFIRM-%s-T%d' % (family, i), family=family,
                layout='T%d' % i, reset=500+i, noise_seed=700+i, arm='B2', language='unchanged',
                scene=dict(path='scene-'+family, sha256='sha-'+family),
                template_spec=templates[family][1]))
    env = dict(TMPDIR='/old-private/tmp', XDG_CACHE_HOME='/old-private/cache',
        ROS_LOG_DIR='/old-private/logs/ros', CUDA_VISIBLE_DEVICES='authorized-GPU')
    server_pin = save(tmp_path/'old-server.json', dict(packages=packages,
        site_binding={}, native_environment=env, gpu_environment=env,
        request_bindings_by_arm={a: dict(source_sha='old-source', arm=a) for a in packages}))
    original = dict(resident_server_spec=server_pin, mobile_world_lock='same-world',
        prior_case_results=[{}, {}])
    proof = dict(admission=dict(requests=8, stage='CONFIRM', arm='B2'))
    monkeypatch.setattr(evaluation, 'evidence', lambda spec: (proof, panel, templates, original, {}))
    monkeypatch.setattr(evaluation, 'site_binding', lambda *args, **kwargs: {})
    registered = []
    def register(parent, entries):
        for folder, spec in entries:
            save(folder/'EXECUTION_SPEC.json', spec)
            registered.append(spec)
    monkeypatch.setattr(evaluation, 'register', register)
    monkeypatch.setattr(evaluation, 'carried_rows', lambda prior: [])
    monkeypatch.setattr(evaluation, 'interface_gate', lambda rows: True)
    def validate(case):
        assert case['python'] == '/frozen-native/bin/python'
        assert case['ordinary_RGB_restore']['path'] == 'actual-B2-restore'
        assert case['arm'] == 'B2' and case['SHIFT7_package'] == packages['B2']
        assert case['max_ticks'] == 35703 and case['physics_dt_s'] == .002
        assert case['WBC_changes'] is case['safety_changes'] is False
    monkeypatch.setattr(evaluation, 'validate', validate)
    spec = {k: 'same' for k in ('source_root', 'registry', 'owner', 'host', 'world_control',
        'goal_plan_sha', 'allocation_deadline_epoch', 'campaign_deadline_epoch', 'deadline_epoch')}
    spec.update(source_sha='new-immutable-source', source_files={}, trusted_source_parent={},
        source_seal={}, plan_commit=evaluation.PLAN, registered_root=str(root),
        campaign_root=str(root), python='/sla-pt/venv/bin/python', uid=2009,
        native_environment={}, frozen_arm='B2', evaluation_stage='CONFIRM', packages=packages)
    result = evaluation.prepare(spec)
    assert result['new_physical_requests_budget'] == 8
    assert result['new_model_generations_budget'] == 960
    assert len(result['cases']) == 8
    worker = registered[0]
    assert worker['new_complete_generations_budget'] == 960 and not worker['physics']
    assert worker['gpu_environment']['CUDA_VISIBLE_DEVICES'] == 'authorized-GPU'
    assert worker['gpu_environment']['TMPDIR'] == str(root/'worker/tmp')
    assert env['TMPDIR'] == '/old-private/tmp'
    assert all(s['python'] == '/frozen-native/bin/python' for s in registered[2:])
    assert all(s['frozen_files']['native'] == 'frozen-binary' for s in registered[2:])
