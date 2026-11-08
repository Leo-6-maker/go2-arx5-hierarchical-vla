"""Synthetic file/proof fixtures only; no real proposal execution or model fit."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.pi0292.relay_protocol import sealed
from go2arx5_vla.world_model import fusion_live_proposal_qualification_contract as c
from go2arx5_vla.world_model import fusion_trajectory_qualification_contract as old
from go2arx5_vla.world_model.fusion_trajectory_adapter import TrajectoryAdaptation
from go2arx5_vla.world_model.storage import digest

BASE = [0., 0., .3, 0., 0., 0., 1.]


def put(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))
    return digest(path)


def close(root, spec, *, code=0):
    path = root / 'EXECUTION_SPEC.json'
    pin = put(path, spec)
    put(root / 'control/LEASE.json', dict(status='CLOSED', run_uuid=root.name,
        worker_exit_code=code, spec_sha256=pin))
    put(root / 'WORKER_EXIT.private.json', dict(exit_code=code, failure=None))
    return pin


def old_episode(root):
    return dict(schema=old.SLOW_SCHEMA, phase=old.SLOW_PHASE, purpose='TRAJECTORY_GEOMETRY_QUALIFICATION',
        experiment_role='NON_VLA_CALIBRATION', split='TRAIN', reference_rule=old.SLOW_RECIPE,
        reference_anchor_mode='INIT_TCP_2503_REFERENCE_V1', prehold_atoms=40, screen_atom=48,
        screen=old.SCREEN, tail_hold_atoms=8, desired_outcome_provenance='CALIBRATION_INTENT',
        physics_dt_s=.002, base_program='MEASURED_INIT_BASE_POSE_HOLD', time_stretch_factor=2,
        query_pins={k: v for k, v in old.PINS.items() if k != 'first_predecessor'},
        run=str(root), scene_id=c.SCENE_ID, parent=c.PARENT, geometry_scale=1., planned_end_atom=1034)


def new_episode(root, raw, scale=.25):
    adaptation = TrajectoryAdaptation(raw, np.eye(4), np.eye(4), np.eye(4), np.eye(4), BASE,
                                     scale, time_stretch_factor=2)
    return dict(c._fixed_episode(), run=str(root), geometry_scale=scale, planned_end_atom=adaptation.end_atom)


def prior_files(root, episode, spec, *, source='SYNTHETIC_LATER_SOURCE'):
    end = old.endpoint(episode) if episode['phase'] == old.SLOW_PHASE else c.endpoint(episode)
    windows = []
    for i in range((end-c.INIT_STEP)//25):
        start = c.INIT_STEP + 25*i
        sent = dict(sequence=i, opening=.95, expiry=start*.002+.4,
                    source_mpc_s=start*.002, native_start_step=start)
        memory = dict(status='ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY', values=dict(
            opening=sent['opening'], expiry=sent['expiry'], sequence=i, steps=start+25,
            initialized=True, has_command=True))
        windows.append(dict(native_start_step=start, native_end_step=start+25, actual_ticks=25,
            solver_policies_with_exact_installed_target=[{'synthetic': True}],
            gripper=dict(command_consumed=True, submitted=sent, consumed_wrapper=memory)))
    report = dict(episode=episode, native_endpoint_step=end, all_40_knots_consumed=True,
        raw_knot_consumed=[True]*40, complete_task_atoms=len(windows), solver_target_confirmed_atoms=len(windows),
        gripper_confirmed_atoms=len(windows), windows=windows, actual_trace_binding=dict(actual_trace_capacity=524288))
    previous = dict(status='NATIVE_STAGE_PASSED', phase=episode['phase'], source_sha=source,
        actual_target_integrations=end, reached_exact_pause=True, shutdown_clean=True, failure=None,
        physics_stops=[], original_joint_or_warning_samples=0, native_outer_guard_rejection_count=0,
        native_outer_guard_rejections={}, trace_audit_error=None,
        owned_process_exits=[dict(name=n, exit_code=0) for n in ('sim','controller','solver','planner')],
        trace_audit=dict(status='RECORDED_TRACE_CHECKS_PASS', errors=[], missing=[], native_step_records=end,
                         writer=dict(events=end, written=end, dropped=0, sealed_at_explicit_pause=True, stop_latched=False)),
        campaign_uuid=Path(spec['campaign_root']).name, candidate_index=spec['candidate_index'],
        native_binary_sha256=digest(spec['native_binary']), observer_sha256=digest(spec['observer_library']),
        scene_sha256=digest(spec['model_path']), reference_qualification=report)
    ep = root / 'EPISODE_MANIFEST.private.json'
    prior_spec = dict(registered_root=str(root), source_sha=source, phase=episode['phase'], max_ticks=end,
                     episode_manifest=str(ep), episode_manifest_sha256=put(ep, episode))
    spec_pin = close(root, prior_spec)
    path = root / 'native-01/RESULT.json'; pin = put(path, previous)
    return previous, path, pin, spec_pin


@pytest.fixture
def recipe(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[2]
    source = tmp_path / 'SYNTHETIC_CAPSULE'
    files = {}
    for name in c.SOURCE_MEMBERS:
        path = source / name; path.parent.mkdir(parents=True, exist_ok=True)
        original = project / name
        # The parent's new native/CLI seam may not be written yet. Nothing in
        # this fixture imports or executes these synthetic capsule members.
        path.write_bytes(original.read_bytes() if original.is_file() else b'SYNTHETIC_READY_SOURCE_MEMBER')
        files[name] = digest(path)
    model, binary, observer, trace = [tmp_path / name for name in ('SYNTHETIC_SCENE', 'SYNTHETIC_BINARY', 'SYNTHETIC_OBSERVER', 'SYNTHETIC_TRACE')]
    for path in (model, binary, observer, trace): path.write_bytes(path.name.encode())
    root = tmp_path / 'SYNTHETIC_NEW_GRADE'
    spec = dict(phase=c.PHASE, profile='wm02-auto1-target', domain='WM02_AUTO1_TARGET_COMMISSION_V1',
        site_profile='4080-simulation', scope='WM02_PILOT1_SIM_ONLY', physics=True,
        pilot1_binding='/fixture/existing-original-P1', source_root=str(source), source_files=files,
        registered_root=str(root), campaign_root=str(tmp_path / 'ORIGINAL_CAMPAIGN'), candidate_index=10,
        WBC_changes=False, physics_dt_s=.002, wall_limit_s=1200,
        reference_binding_version='MEASURED_EEF_AND_BASE_POSE_HOLD_V2', model_path=str(model),
        native_binary=str(binary), observer_library=str(observer), trace_library=str(trace),
        trajectory_trace_capacity=524288, live_proposal_query_pins=copy.deepcopy(c.PINS),
        live_proposal_native_spec='/fixture/query02-native/EXECUTION_SPEC.json',
        live_proposal_gpu_spec='/fixture/query02-gpu/EXECUTION_SPEC.json')
    catalog = tmp_path / 'STATIC_SCENES.json'
    spec.update(scene_allowlist=str(catalog), scene_allowlist_sha256=put(catalog, dict(scenes=[dict(
        scene_id=c.SCENE_ID, parent=c.PARENT, split='TRAIN', model_path=str(model), model_sha256=digest(model),
        static_check='PASSED_NO_INTEGRATION')], native_binary_sha256=digest(binary))))
    # Exercise the original CF06 proof validator with tiny explicit synthetic
    # receipts; monkeypatch only its independently frozen artifact SHA.
    failed_root = tmp_path / 'SYNTHETIC_RETAINED_CF06'
    failed = dict(status='NATIVE_STAGE_FAILED_PRESERVED', phase=old.SLOW_PHASE, actual_target_integrations=29753,
        reached_exact_pause=True, shutdown_clean=True, failure=None, physics_stops=[], original_joint_or_warning_samples=0,
        native_outer_guard_rejection_count=0, native_binary_sha256=digest(binary), observer_sha256=digest(observer),
        scene_sha256=digest(model), owned_process_exits=[dict(name=n, exit_code=0) for n in ('sim','controller','solver','planner')],
        trace_audit=dict(status='RECORDED_TRACE_CHECKS_FAIL', errors=['WRITER_EVENT_COUNT_DIFFERS','WRITER_REPORTED_DROPPED_EVENTS'],
            missing=[], native_step_records=29753, writer=dict(events=266763, written=262144, dropped=4619,
                sealed_at_explicit_pause=True, stop_latched=False)),
        reference_qualification=dict(all_40_knots_consumed=True, complete_task_atoms=1090,
                                    solver_target_confirmed_atoms=1090, gripper_confirmed_atoms=1090))
    close(failed_root, dict(registered_root=str(failed_root), max_ticks=29753), code=1)
    failed_path = failed_root / 'native-01/RESULT.json'
    failed_pin = put(failed_path, failed)
    monkeypatch.setattr(old, 'TRACE_CAPACITY_FAILED_RESULT', failed_pin)
    spec.update(trace_capacity_failed_result=str(failed_path), trace_capacity_failed_result_sha256=failed_pin)
    build_root = tmp_path / 'SYNTHETIC_BUILD03'
    close(build_root, dict(registered_root=str(build_root)))
    build_path = build_root / 'build-01/RESULT.private.json'
    build_pin = put(build_path, dict(status='TRACE_ONLY_524288_CAPACITY_COMPILED_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED',
        trace_capacity=524288, new_target_integrations=0, trace_library_sha256=digest(trace),
        native_binary_sha256=digest(binary), observer_sha256=digest(observer), source_files={
            n: files[n] for n in ('src/go2arx5_vla/target_mujoco/init_trace.cpp','src/go2arx5_vla/target_mujoco/init_trace.hpp')}))
    monkeypatch.setattr(c, 'TRACE_BUILD', build_pin)
    spec.update(trajectory_trace_capacity_build=str(build_path), trajectory_trace_capacity_build_sha256=build_pin)
    raw = np.zeros((40, 10), dtype=np.float32); raw[:, 3:9] = [1,0,0,0,1,0]
    raw[:, 0] = np.linspace(.001, .009, 40); raw[:, 2] = .002*np.sin(np.linspace(0, 3, 40)); raw[:, 9] = np.linspace(.99, 1, 40)
    body = dict(world=dict(scene_sha256=digest(model)), task_sha256=c.PINS['task_sha256'],
        task={'SYNTHETIC_TEST_TASK': True}, original_shadow_ABI=sealed(dict(source_policy_view=dict(T_source_from_target=np.eye(4).tolist()))))
    archived = dict(status='PAIR_COMPLETED_AND_PAUSE_REVALIDATED', paired_return=dict(established=True),
        external_pins=copy.deepcopy(c.PINS), historical_read=True, current_native_pause_checked=False,
        native_execution_authorized=False, observation=sealed(body), physical_H40=raw,
        payload=dict(state_eef10=[0,0,0,1,0,0,0,1,0,1]))
    calls = []
    def owned_reader(native, gpu, *, pins):
        calls.append((native, gpu, pins))
        assert pins == c.PINS
        return archived
    monkeypatch.setattr(c, 'read_archive', owned_reader)
    ep = new_episode(root, raw)
    ep_path = root / 'EPISODE_MANIFEST.private.json'
    spec.update(episode_manifest=str(ep_path), episode_manifest_sha256=put(ep_path, ep), max_ticks=c.endpoint(ep))
    prev_root = tmp_path / 'SYNTHETIC_FULL1_67'
    previous, prev_path, pin, spec_pin = prior_files(prev_root, old_episode(prev_root), spec, source=c.FIRST_PREDECESSOR_SOURCE)
    monkeypatch.setattr(c, 'FIRST_PREDECESSOR', pin)
    monkeypatch.setattr(c, 'FIRST_PREDECESSOR_SPEC', spec_pin)
    spec.update(previous_stage=str(prev_path), previous_stage_sha256=pin)
    return spec, previous, ep, archived, calls


def rewrite_episode(spec, episode):
    spec['episode_manifest_sha256'] = put(spec['episode_manifest'], episode)
    spec['max_ticks'] = c.endpoint(episode)


def test_query02_historical_archive_and_explicit_adapter_are_the_only_new_inputs(recipe):
    spec, previous, episode, archive, calls = recipe
    actual, inputs = c.episode_config(spec)
    assert actual == episode and inputs['archive'] is archive
    assert inputs['task_sha256'] == c.PINS['task_sha256']
    assert c.episode_limit(spec, previous) == c.endpoint(episode)
    assert all(x[0] == Path(spec['live_proposal_native_spec']) and x[1] == Path(spec['live_proposal_gpu_spec']) for x in calls)
    program = c.make_program(inputs, episode, np.eye(4), np.eye(4), BASE)
    assert program.time_stretch_factor == 2 and len(program.knot_indices) == 40
    assert program.end_atom == episode['planned_end_atom']
    np.testing.assert_array_equal(program.decoded_H40, archive['physical_H40'])
    assert program.decoded_H40.dtype == np.float32 and not program.decoded_H40.flags.writeable
    assert inputs['provenance']['task_intent_sufficient_for_task_success'] is False
    assert inputs['provenance']['execution_condition'].startswith('FROZEN_QUERY02_REFERENCE_REPLAY')


@pytest.mark.parametrize('key,value', [
    ('phase', old.SLOW_PHASE), ('profile', 'wm02-fusion-vla-shadow'), ('physics', False),
    ('scope', 'WM02_FUSION_REFERENCE_SIM_ONLY'), ('pilot1_binding', None), ('site_profile', 'GPU_PHYSICS'),
    ('fusion_binding', '/old-FUSION'), ('next1_binding', '/old-NEXT1'), ('parallel_batch_spec', '/batch'),
    ('physics_dt_s', .004), ('WBC_changes', True), ('wall_limit_s', 1201), ('wall_limit_s', True),
])
def test_backend_scope_and_frozen_native_clock_are_not_grants_or_bypasses(recipe, key, value):
    spec, *_ = recipe
    with pytest.raises(ValueError): c.episode_config(dict(spec, **{key: value}))


@pytest.mark.parametrize('key', sorted(c.PINS))
def test_every_external_query02_pin_is_required_and_checked_before_reader(recipe, key):
    spec, _, episode, _, calls = recipe
    value = copy.deepcopy(spec); value['live_proposal_query_pins'][key] = 'ALTERED'
    with pytest.raises(ValueError, match='14_REGISTERED_QUERY02_PINS'): c.episode_config(value)
    assert calls == []
    altered = copy.deepcopy(episode); altered['query_pins'][key] = 'ALTERED'
    with pytest.raises(ValueError, match='EXACT_V3'): c.episode_contract(altered)


@pytest.mark.parametrize('change', [dict(geometry_scale=.03), dict(geometry_scale=.75),
    dict(time_stretch_factor=1), dict(time_stretch_factor=True), dict(split='DEV'), dict(scene_id='OTHER'),
    dict(online_closed_loop_claim=True), dict(full_task_success_claim=True), dict(WM_updates=1), dict(clip=True)])
def test_episode_recipe_cannot_change_geometry_semantics_or_claims(recipe, change):
    _, _, episode, *_ = recipe
    with pytest.raises(ValueError): c.episode_contract(dict(episode, **change))


@pytest.mark.parametrize('fault', ['incomplete_archive', 'native_failed', 'live_grant', 'scene', 'raw_OOD', 'clock', 'source_member', 'trace_capacity', 'build_source', 'CF06', 'build_open'])
def test_missing_archive_native_or_trace_proof_fails_closed(recipe, fault):
    spec, _, episode, archive, _ = recipe
    if fault == 'incomplete_archive': archive['status'] = 'SAVED_SNAPSHOT_QUERY_ARCHIVED_NATIVE_PAIR_NOT_COMPLETED'
    elif fault == 'native_failed': archive['paired_return']['established'] = False
    elif fault == 'live_grant': archive['native_execution_authorized'] = True
    elif fault == 'scene': Path(spec['model_path']).write_bytes(b'CHANGED_SCENE')
    elif fault == 'raw_OOD': archive['physical_H40'][0, 9] = 1.01
    elif fault == 'clock': rewrite_episode(spec, dict(episode, planned_end_atom=episode['planned_end_atom']+1))
    elif fault == 'source_member': (Path(spec['source_root']) / c.SOURCE_MEMBERS[-1]).write_bytes(b'CHANGED')
    elif fault == 'trace_capacity': spec['trajectory_trace_capacity'] = 262144
    elif fault == 'build_source':
        path = Path(spec['trajectory_trace_capacity_build']); value = json.loads(path.read_text()); value['new_target_integrations'] = 1
        spec['trajectory_trace_capacity_build_sha256'] = put(path, value)
    elif fault == 'CF06': Path(spec['trace_capacity_failed_result']).write_text('{}')
    else:
        path = Path(spec['trajectory_trace_capacity_build']).parent.parent / 'control/LEASE.json'
        value = json.loads(path.read_text()); value['status'] = 'ACTIVE'; put(path, value)
    with pytest.raises(ValueError): c.episode_config(spec)


def test_new_half_requires_new_quarter_not_old_geometry_or_skipped_grade(recipe, tmp_path):
    spec, old_previous, episode, archive, _ = recipe
    current = new_episode(Path(spec['registered_root']), archive['physical_H40'], .5)
    rewrite_episode(spec, current)
    with pytest.raises(ValueError): c.episode_limit(spec, old_previous)
    previous_root = tmp_path / 'SYNTHETIC_NEW_QUARTER'
    prior_episode = new_episode(previous_root, archive['physical_H40'], .25)
    previous, path, pin, _ = prior_files(previous_root, prior_episode, spec)
    spec.update(previous_stage=str(path), previous_stage_sha256=pin)
    assert c.episode_limit(spec, previous) == c.endpoint(current)
    rewrite_episode(spec, new_episode(Path(spec['registered_root']), archive['physical_H40'], 1.))
    with pytest.raises(ValueError): c.episode_limit(spec, previous)


@pytest.mark.parametrize('fault', ['failed', 'trace_drop', 'missing_trace_counts', 'wrong_family', 'truncated', 'missing_knots',
    'pose_unconsumed', 'grip_unconsumed', 'grip_expired', 'grip_uninitialized', 'wrong_query', 'open_lease', 'wrong_worker'])
def test_rehashed_predecessor_cannot_replace_complete_new_query_grade(recipe, tmp_path, fault):
    spec, _, _, archive, _ = recipe
    rewrite_episode(spec, new_episode(Path(spec['registered_root']), archive['physical_H40'], .5))
    prior_root = tmp_path / 'SYNTHETIC_NEW_QUARTER'
    prior_episode = new_episode(prior_root, archive['physical_H40'], .25)
    previous, path, pin, _ = prior_files(prior_root, prior_episode, spec)
    spec.update(previous_stage=str(path), previous_stage_sha256=pin)
    report = previous['reference_qualification']
    if fault == 'failed': previous['status'] = 'NATIVE_STAGE_FAILED_PRESERVED'
    elif fault == 'trace_drop': previous['trace_audit']['writer']['dropped'] = 1
    elif fault == 'missing_trace_counts':
        del previous['trace_audit']['writer']['events']; del previous['trace_audit']['writer']['written']
    elif fault == 'wrong_family': previous['phase'] = old.SLOW_PHASE
    elif fault == 'truncated':
        previous['actual_target_integrations'] -= 25; report['native_endpoint_step'] -= 25
        report['windows'].pop(); report['complete_task_atoms'] -= 1; report['solver_target_confirmed_atoms'] -= 1; report['gripper_confirmed_atoms'] -= 1
    elif fault == 'missing_knots': report['raw_knot_consumed'][-1] = False
    elif fault == 'pose_unconsumed': report['windows'][-1]['solver_policies_with_exact_installed_target'] = []
    elif fault == 'grip_unconsumed': report['windows'][-1]['gripper']['command_consumed'] = False
    elif fault == 'grip_expired':
        grip = report['windows'][-1]['gripper']; grip['submitted']['expiry'] -= 1.; grip['consumed_wrapper']['values']['expiry'] -= 1.
    elif fault == 'grip_uninitialized': report['windows'][-1]['gripper']['consumed_wrapper']['values']['initialized'] = False
    elif fault == 'wrong_query': report['episode']['query_pins']['action_sha256'] = '0'*64
    elif fault == 'open_lease':
        lease = json.loads((prior_root/'control/LEASE.json').read_text()); lease['status'] = 'ACTIVE'; put(prior_root/'control/LEASE.json', lease)
    else: put(prior_root/'WORKER_EXIT.private.json', dict(exit_code=1, failure='SYNTHETIC_FAILURE'))
    spec['previous_stage_sha256'] = put(path, previous)
    with pytest.raises(ValueError): c.episode_limit(spec, previous)


def test_finite_clock_and_full_curve_are_preserved_without_clipping(recipe):
    spec, _, episode, archive, _ = recipe
    for value in (True, 39, 2000):
        with pytest.raises(ValueError): c.episode_contract(dict(episode, planned_end_atom=value))
    for scale in c.SCALES:
        current = new_episode(Path(spec['registered_root']), archive['physical_H40'], scale)
        c.episode_contract(current)
        assert c.endpoint(current) == 3703 + 25*(current['planned_end_atom'] + 8) <= 33000


def test_import_is_numpy_only_and_does_not_call_live_or_model_backends():
    code = ('import sys; from go2arx5_vla.world_model import fusion_live_proposal_qualification_contract;'
            'assert not {"torch","mujoco","rclpy","transformers","gr00t"} & sys.modules.keys()')
    subprocess.run([sys.executable, '-B', '-c', code], check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
