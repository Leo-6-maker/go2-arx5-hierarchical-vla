"""Synthetic bias-fade lineage and consumption checks, never native evidence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from go2arx5_vla.world_model import auto1_native, target_sequential
from go2arx5_vla.world_model import fusion_task_phase_bias_fade_v8 as phase
from go2arx5_vla.world_model import fusion_task_phase_bias_fade_contract_v8 as contract
from go2arx5_vla.world_model import fusion_task_phase_bias_fade_native_v8 as native
from go2arx5_vla.world_model import fusion_task_phase_initialization_native as original_native
from go2arx5_vla.world_model import fusion_task_phase_pose_continuation_v7 as v7
from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_native_v4 as n4
from go2arx5_vla.world_model import fusion_task_phase_xyz_feedback_evidence as evidence
from go2arx5_vla.world_model import fusion_trajectory_qualification_native as common_native
from go2arx5_vla.world_model import fusion_live_proposal_qualification_contract as common
from go2arx5_vla.world_model.storage import digest
from scripts.wm02 import fusion_task_phase_bias_fade_check_v8 as cli
from test_fusion_task_phase_initialization import setup as pure_setup, samples as hold_samples
from test_fusion_task_phase_pose_continuation_v7 import (
    Adapter as V7Adapter, completed as completed_v7, recipe as v7_recipe)
from test_fusion_reference_qualification_contract import reference
from test_fusion_live_proposal_qualification_contract import put
from test_fusion_trajectory_qualification import setup_flow


@pytest.fixture
def setup(monkeypatch):
    return pure_setup.__wrapped__(monkeypatch)


def programs(setup):
    episode = phase.make_episode(run='/fixture/full1-bias-fade-v8', geometry_scale=1.)
    args = (setup['inputs'], setup['initial'], setup['decision'], setup['base'])
    current = phase.make_program(args[0], episode, *args[1:])
    old_episode = v7.make_episode(run=episode['run'], geometry_scale=1.)
    old = v7.make_program(args[0], old_episode, *args[1:])
    return current, old, episode


def test_only_translation_fades_and_all_desired_goal_so3_grip_and_early_bytes_match_v7(setup):
    current, old, episode = programs(setup)
    assert phase.endpoint(episode) == 46303
    assert episode['ramp_atoms'] == 1024 and episode['planned_end_atom'] == 1696
    assert phase.FADE_START_ATOM == 552 and phase.FADE_END_ATOM == 672
    assert phase.FADE_START_STEP == 3703+25*552 == 17503
    assert phase.FADE_END_STEP == 3703+25*672 == 20503
    assert phase.FADE_DURATION_S == (672-552)*.05 == 6.
    np.testing.assert_array_equal(current.desired_path, old.desired_path)
    np.testing.assert_array_equal(current.full_goal_TCP_target, old.full_goal_TCP_target)
    np.testing.assert_array_equal(current.graded_goal_TCP_target, old.graded_goal_TCP_target)
    np.testing.assert_array_equal(current.frozen_left_rotation, old.frozen_left_rotation)
    np.testing.assert_array_equal(current.reference_path[:, 3:], old.reference_path[:, 3:])
    np.testing.assert_array_equal(current.reference_path[:553], old.reference_path[:553])
    assert current.desired_path.tobytes() == old.desired_path.tobytes()
    assert current.reference_path[:, 3:].tobytes() == old.reference_path[:, 3:].tobytes()
    assert current.reference_path[:553].tobytes() == old.reference_path[:553].tobytes()
    np.testing.assert_array_equal(current.reference_path[672:, :3], current.desired_path[672:, :3])
    assert not np.array_equal(current.reference_path[612, :3], old.reference_path[612, :3])
    for atom, u in ((582, .25), (612, .5), (642, .75)):
        weight = 10*u**3-15*u**4+6*u**5
        expected = old.reference_path[atom, :3]+weight*old.frozen_translation_bias_m
        np.testing.assert_array_equal(current.reference_path[atom, :3], expected)
    for atom in (0, 511, 512, 513):
        np.testing.assert_array_equal(current.preview(atom), old.preview(atom))
    proof = current.evidence()
    assert proof['original_full_bias_reference_path_sha256'] == phase.fixed._sha(old.reference_path)
    assert proof['unchanged_desired_path_sha256'] == phase.fixed._sha(old.desired_path)
    assert proof['frozen_predecision_reference_binding']['frozen_SO3_bias_unchanged'] is True
    assert proof['saved_state_geometry_diagnostic_is_dynamic_success'] is False
    assert proof['analytical_rates']['reference']['peak_translation_speed_m_s'] <= .05
    assert proof['analytical_rates']['reference']['peak_translation_acceleration_m_s2'] <= .25
    with pytest.raises(ValueError):
        v7.episode_contract(episode)


@pytest.mark.parametrize('grade', [.03, .1, .25, .5, True])
def test_no_repeated_lower_grade_or_boolean_full_grade(grade):
    with pytest.raises(ValueError):
        phase.make_episode(run='/fixture/refused-fade', geometry_scale=grade)


@pytest.mark.parametrize('fault', ['fade-start', 'fade-end', 'fade-law', 'dose', 'slots', 'clock', 'phase'])
def test_static_manifest_cannot_silently_change_fade_law_dose_or_clock(fault):
    episode = phase.make_episode(run='/fixture/fixed-fade', geometry_scale=1.)
    if fault == 'fade-start':
        episode['translation_bias_fade']['start_elapsed_atom'] += 1
    elif fault == 'fade-end':
        episode['translation_bias_fade']['end_elapsed_atom'] -= 1
    elif fault == 'fade-law':
        episode['translation_bias_fade']['time_law'] = 'LINEAR'
    elif fault == 'dose':
        episode['feedback_step_norm_cap_m'] = .009
    elif fault == 'slots':
        episode['finite_feedback_decisions'] = 25
    elif fault == 'clock':
        episode['planned_end_atom'] += 1
    else:
        episode['phase'] = v7.PHASE
    with pytest.raises(ValueError):
        phase.episode_contract(episode)


def test_twentyfour_saved_feedback_doses_retain_fade_desired_labels_and_original_caps(setup):
    program, old, episode = programs(setup)
    original_path = program.reference_path.copy()
    desired = program.desired_path.copy()
    assert list(phase.feedback_steps(episode)) == [29503+700*i for i in range(24)]
    for index, step in enumerate(phase.feedback_steps(episode)):
        actual = program.graded_goal_TCP_target.copy()
        actual[:3, 3] -= [.05, 0., 0.]
        actual[:3, :3] = v7.previous._exp_so3([0., .06, 0.])@actual[:3, :3]
        update = program.prepare_feedback(step, actual)
        assert update.evidence['schema'] == 'WM02_FINITE_CAUSAL_POSE_FEEDBACK_DECISION_V8'
        assert np.isclose(update.evidence['applied_delta_norm_m'], .008)
        assert np.isclose(update.evidence['applied_rotation_norm_rad'], .01)
        with pytest.raises(ValueError, match='NO_SCHEDULED_REFERENCE'):
            phase.validate_install_preview(program.preview((step-3703)//25),
                initial_tcp=setup['initial'], program=program, step=step,
                base=setup['base'], frozen_base=setup['base'], end_step=46303)
        program.commit_feedback(update, saved_decision_sha256=hashlib.sha256(str(index).encode()).hexdigest())
        phase.validate_install_preview(program.preview((step-3703)//25),
            initial_tcp=setup['initial'], program=program, step=step,
            base=setup['base'], frozen_base=setup['base'], end_step=46303)
    assert program.feedback_count == 24
    assert np.isclose(program.evidence()['cumulative_feedback_reference_path_length_m'], .192)
    assert np.isclose(program.evidence()['cumulative_feedback_reference_rotation_path_rad'], .24)
    np.testing.assert_array_equal(program.desired_path, desired)
    np.testing.assert_array_equal(program.reference_path[:1033], original_path[:1033])
    np.testing.assert_array_equal(program.reference_path[:553], old.reference_path[:553])
    with pytest.raises(ValueError):
        program.prepare_feedback(46303, actual)
    rows = hold_samples(setup, 1.)
    for row, step in zip(rows, range(46103, 46304, 25)):
        row.update(native_step=step, time_s=step*.002)
    readiness = phase.query_readiness(rows, episode, setup['inputs'])
    assert readiness['query_ready'] is True
    assert readiness['actual_sample_steps'] == list(range(46103, 46304, 25))
    assert readiness['native_execution_authorized'] is readiness['GPU_query_authorized'] is False


def build_archive(tmp_path, monkeypatch):
    spec, _, _, setup = v7_recipe.__wrapped__(tmp_path, monkeypatch)
    project, capsule = Path(__file__).resolve().parents[2], Path(spec['source_root'])
    for name in contract.SOURCE_MEMBERS:
        path = capsule/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((project/name).read_bytes())
        spec['source_files'][name] = digest(path)
    prior, previous, previous_path, previous_pin = completed_v7(
        tmp_path/'SYNTHETIC_QUALIFIED_V7_HALF', spec, setup, .5)
    monkeypatch.setattr(contract, 'FIRST_PREDECESSOR_SOURCE', prior['source_sha'])
    monkeypatch.setattr(contract, 'FIRST_PREDECESSOR_SPEC', digest(previous_path.parent.parent/'EXECUTION_SPEC.json'))
    monkeypatch.setattr(contract, 'FIRST_PREDECESSOR', previous_pin)

    failed_root = tmp_path/'SYNTHETIC_RETAINED_FAILED_V7_FULL1'
    failed_root.mkdir()
    failed_spec = copy.deepcopy(prior)
    failed_episode = v7.make_episode(run=str(failed_root), geometry_scale=1.)
    failed_manifest = failed_root/'EPISODE_MANIFEST.private.json'
    failed_spec.update(registered_root=str(failed_root), max_ticks=46303,
        episode_manifest=str(failed_manifest), episode_manifest_sha256=put(failed_manifest, failed_episode),
        previous_stage=str(previous_path), previous_stage_sha256=previous_pin)
    failed_spec_path = failed_root/'EXECUTION_SPEC.json'
    failed_spec_pin = put(failed_spec_path, failed_spec)
    failed_result = copy.deepcopy(previous)
    failed_result.update(status='NATIVE_STAGE_FAILED_PRESERVED', actual_target_integrations=20503,
        complete_atomic_transitions=820, reached_exact_pause=False, reference_qualification=None,
        failure='ValueError: '+contract.FAILED_REASON+'\n', shutdown_clean=True,
        physics_stops=[], original_joint_or_warning_samples=0, native_outer_guard_rejection_count=0,
        native_outer_guard_rejections={}, trace_audit_error=None)
    failed_result['trace_audit'].update(native_step_records=20503)
    failed_result['trace_audit']['writer'].update(events=20503, written=20503,
        dropped=0, stop_latched=False, sealed_at_explicit_pause=True)
    failed_result_path = failed_root/'native-01/RESULT.json'
    failed_result_pin = put(failed_result_path, failed_result)
    lease = dict(status='CLOSED', run_uuid=failed_root.name, source_sha=prior['source_sha'],
                 spec_sha256=failed_spec_pin, worker_exit_code=1)
    worker = dict(exit_code=1, failure=None)
    lease_path, worker_path = failed_root/'control/LEASE.json', failed_root/'WORKER_EXIT.private.json'
    put(lease_path, lease)
    put(worker_path, worker)
    commands_path = failed_root/'native-01/COMMANDS.private.jsonl'
    rejected = dict(phase='reference-certificate-rejected', native_step=20503,
        reason=contract.FAILED_REASON, submitted_or_executed=False)
    commands_path.write_text(json.dumps(rejected)+'\n')
    for name, value in (('FAILED_SOURCE', prior['source_sha']), ('FAILED_SPEC', failed_spec_pin),
                        ('FAILED_RESULT', failed_result_pin), ('FAILED_LEASE', digest(lease_path)),
                        ('FAILED_WORKER', digest(worker_path)), ('FAILED_COMMANDS_SHA', digest(commands_path))):
        monkeypatch.setattr(contract, name, value)
    monkeypatch.setattr(contract, '_unit_status', lambda root: dict(
        MainPID='0', ExecMainStatus='1' if Path(root) == failed_root else '0'))
    root = tmp_path/'SYNTHETIC_NEW_V8_BIAS_FADE'
    root.mkdir()
    spec.update(registered_root=str(root), phase=phase.PHASE, kind=contract.KIND,
        previous_stage=str(previous_path), previous_stage_sha256=previous_pin,
        repaired_full1_failed_spec=str(failed_spec_path), repaired_full1_failed_spec_sha256=failed_spec_pin,
        repaired_full1_failed_result=str(failed_result_path), repaired_full1_failed_result_sha256=failed_result_pin,
        episode_manifest=str(root/'EPISODE_MANIFEST.private.json'), max_ticks=46303)
    episode = phase.make_episode(run=str(root), geometry_scale=1.)
    spec['episode_manifest_sha256'] = put(spec['episode_manifest'], episode)
    return dict(spec=spec, previous=previous, prior=prior, episode=episode, setup=setup,
        failed_root=failed_root, failed_spec=failed_spec, failed_result=failed_result,
        lease=lease, worker=worker, rejected=rejected, commands_path=commands_path)


@pytest.fixture(scope='module')
def archive(tmp_path_factory):
    # Build the real synthetic predecessor chain once. Per-case edits below
    # restore the small pinned documents, without rerunning every old wrapper.
    with pytest.MonkeyPatch.context() as patch:
        value = build_archive(tmp_path_factory.mktemp('bias-fade-archive'), patch)
        root = value['failed_root']
        paths = [Path(value['spec']['previous_stage']), root/'EXECUTION_SPEC.json',
            root/'native-01/RESULT.json', root/'control/LEASE.json',
            root/'WORKER_EXIT.private.json', value['commands_path']]
        yield value, {path: path.read_bytes() for path in paths}


@pytest.fixture
def recipe(archive):
    value, documents = archive
    for path, raw in documents.items():
        path.write_bytes(raw)
    try:
        yield copy.deepcopy(value)
    finally:
        for path, raw in documents.items():
            path.write_bytes(raw)


def repin_failed(recipe, monkeypatch, *, spec=False, result=False, lease=False, worker=False, commands=False):
    root, current = recipe['failed_root'], recipe['spec']
    if spec:
        pin = put(root/'EXECUTION_SPEC.json', recipe['failed_spec'])
        current['repaired_full1_failed_spec_sha256'] = pin
        recipe['lease']['spec_sha256'] = pin
        monkeypatch.setattr(contract, 'FAILED_SPEC', pin)
        lease = True
    if result:
        pin = put(root/'native-01/RESULT.json', recipe['failed_result'])
        current['repaired_full1_failed_result_sha256'] = pin
        monkeypatch.setattr(contract, 'FAILED_RESULT', pin)
    if lease:
        monkeypatch.setattr(contract, 'FAILED_LEASE', put(root/'control/LEASE.json', recipe['lease']))
    if worker:
        monkeypatch.setattr(contract, 'FAILED_WORKER', put(root/'WORKER_EXIT.private.json', recipe['worker']))
    if commands:
        recipe['commands_path'].write_text(json.dumps(recipe['rejected'])+'\n')
        monkeypatch.setattr(contract, 'FAILED_COMMANDS_SHA', digest(recipe['commands_path']))


def test_exact_half_pass_and_separate_closed1_failure_check_without_reclassifying_either(recipe, tmp_path):
    spec, previous = recipe['spec'], recipe['previous']
    assert common._closed(Path(spec['previous_stage']).parent.parent)[0] == recipe['prior']
    with pytest.raises(ValueError, match='ACTUAL_CLOSED0'):
        common._closed(recipe['failed_root'])
    checked = evidence.audit_completed_feedback(recipe['prior'], previous, recipe=v7)
    assert checked['actual_feedback_decisions'] == 16
    assert checked['all_nine_non_full_grade_predicates_passed'] is True
    assert contract.episode_limit(spec, previous) == 46303
    assert recipe['failed_result']['status'] == 'NATIVE_STAGE_FAILED_PRESERVED'
    assert recipe['failed_result']['reference_qualification'] is None
    path = tmp_path/'CHECK_V8.json'
    put(path, spec)
    value = cli.check(path)
    assert value['qualified_half_all_nine_predicates_passed'] is True
    assert value['failed_full1_closed1_preserved'] is True
    assert value['native_execution_authorized'] is value['native_qualification_claim'] is False
    assert value['new_integrations'] == value['VLA_queries'] == value['WM_updates'] == 0


@pytest.mark.parametrize('field', ['repaired_full1_failed_spec', 'repaired_full1_failed_spec_sha256',
    'repaired_full1_failed_result', 'repaired_full1_failed_result_sha256'])
def test_missing_explicit_failed_supplement_cannot_be_substituted_by_half_pass(recipe, field):
    recipe['spec'].pop(field)
    with pytest.raises((ValueError, KeyError)):
        contract._failed_full1_supplement(recipe['spec'], recipe['prior'])


@pytest.mark.parametrize('fault', ['closed0', 'open-lease', 'wrong-source', 'different-half',
    'relabeled-pass', 'partial-clock', 'bad-trace', 'executed-rejection', 'wrong-reason', 'live-failed-unit'])
def test_rehashed_failed_supplement_still_requires_exact_closed1_unpublished_failure(recipe, monkeypatch, fault):
    if fault == 'closed0':
        recipe['lease']['worker_exit_code'] = recipe['worker']['exit_code'] = 0
        repin_failed(recipe, monkeypatch, lease=True, worker=True)
    elif fault == 'open-lease':
        recipe['lease']['status'] = 'ACTIVE'
        repin_failed(recipe, monkeypatch, lease=True)
    elif fault == 'wrong-source':
        recipe['failed_spec']['source_sha'] = '0'*40
        repin_failed(recipe, monkeypatch, spec=True)
    elif fault == 'different-half':
        recipe['failed_spec']['previous_stage_sha256'] = '0'*64
        repin_failed(recipe, monkeypatch, spec=True)
    elif fault == 'relabeled-pass':
        recipe['failed_result']['status'] = 'NATIVE_STAGE_PASSED'
        repin_failed(recipe, monkeypatch, result=True)
    elif fault == 'partial-clock':
        recipe['failed_result']['actual_target_integrations'] = 20528
        repin_failed(recipe, monkeypatch, result=True)
    elif fault == 'bad-trace':
        recipe['failed_result']['trace_audit_error'] = 'SYNTHETIC_TRACE_ERROR'
        repin_failed(recipe, monkeypatch, result=True)
    elif fault == 'executed-rejection':
        recipe['rejected']['submitted_or_executed'] = True
        repin_failed(recipe, monkeypatch, commands=True)
    elif fault == 'wrong-reason':
        recipe['rejected']['reason'] = 'OTHER_FAILURE'
        repin_failed(recipe, monkeypatch, commands=True)
    else:
        monkeypatch.setattr(contract, '_unit_status', lambda root: dict(
            MainPID='123' if Path(root) == recipe['failed_root'] else '0',
            ExecMainStatus='1' if Path(root) == recipe['failed_root'] else '0'))
    with pytest.raises((ValueError, KeyError)):
        contract._failed_full1_supplement(recipe['spec'], recipe['prior'])


@pytest.mark.parametrize('fault', ['false-measured-gate', 'stale-result-pin', 'failed-as-predecessor',
    'live-half-unit', 'missing-original-reader-source'])
def test_half_qualification_cannot_be_replaced_by_a_failure_or_stale_aggregate(recipe, monkeypatch, fault):
    spec, previous = recipe['spec'], recipe['previous']
    if fault == 'false-measured-gate':
        previous['reference_qualification']['task_phase_initialization']['readiness']['predicates']['position'] = False
        pin = put(spec['previous_stage'], previous)
        spec['previous_stage_sha256'] = pin
        monkeypatch.setattr(contract, 'FIRST_PREDECESSOR', pin)
    elif fault == 'stale-result-pin':
        spec['previous_stage_sha256'] = '0'*64
    elif fault == 'failed-as-predecessor':
        spec['previous_stage'] = spec['repaired_full1_failed_result']
        spec['previous_stage_sha256'] = spec['repaired_full1_failed_result_sha256']
        previous = recipe['failed_result']
    elif fault == 'missing-original-reader-source':
        spec['source_files'].pop('scripts/pi0294/task_state.py')
    else:
        monkeypatch.setattr(contract, '_unit_status', lambda root: dict(MainPID='0', ExecMainStatus='1'))
    with pytest.raises((ValueError, KeyError, PermissionError)):
        contract.episode_limit(spec, previous)


class FadeAdapter(V7Adapter):
    """Perfect desired tracking is synthetic and independent of the reference."""
    def install_reference(self, action):
        native.validate_adapter_install(self, action)
        value = reference(round(self.step*.002*1e9)/1e9)
        wire = json.dumps(dict(synthetic_not_native_command=True, native_step=self.step, H40=action.tolist()))
        pin = hashlib.sha256(wire.encode()).hexdigest()
        value.update(wire=wire, wire_sha256=pin)
        for row in value['receiver_events']:
            row['ref_id'] = pin
            if row['kind'] == 'accept':
                row['raw'] = wire
        self.active_reference = dict(value, action_EEF10=action.tolist())
        self.installs.append((self.step, action.copy()))
        return self.active_reference


def test_twentyfour_native_wrapper_decisions_and_every_revision_rejoin_original_saved_evidence(recipe, tmp_path, monkeypatch):
    spec, setup = copy.deepcopy(recipe['spec']), recipe['setup']
    root = tmp_path/'SYNTHETIC_V8_FULL24'
    output = root/'native-01'
    output.mkdir(parents=True)
    episode = phase.make_episode(run=str(root), geometry_scale=1.)
    spec.update(registered_root=str(root), episode_manifest=str(root/'EPISODE_MANIFEST.private.json'))
    spec['episode_manifest_sha256'] = put(spec['episode_manifest'], episode)
    adapter = FadeAdapter(spec, output, setup)
    setup_flow(monkeypatch, adapter)
    consumed = common_native.consumed_window
    monkeypatch.setattr(common_native, 'consumed_window', lambda a, *args: dict(
        consumed(a, *args), reference_sha256=a.active_reference['wire_sha256']))
    def fk(a, packet, state, **kwargs):
        return dict(status='SYNTHETIC_NOT_FK_OR_NATIVE_EVIDENCE', native_step=packet['step'],
            native_time_s=packet['time_s'], model_sha256=phase.original.SCENE,
            reader_sha256=a.spec['source_files']['scripts/pi0294/task_state.py'],
            packet_qpos_qvel_ctrl_exact=True, original_reader_instance_retained=True)
    monkeypatch.setattr(original_native, '_verify_actual_fk', fk)
    monkeypatch.setattr(n4, '_verify_actual_fk', fk)
    report = native.run(adapter)
    outer = copy.deepcopy(recipe['previous'])
    outer.update(phase=phase.PHASE, source_sha=spec['source_sha'], actual_target_integrations=46303,
        complete_atomic_transitions=1852, reference_qualification=report)
    outer['trace_audit']['native_step_records'] = 46303
    outer['trace_audit']['writer'].update(events=46303, written=46303)
    proof = evidence.audit_completed_feedback(spec, outer, recipe=phase)
    assert proof['actual_feedback_decisions'] == 24 and proof['complete_task_atoms'] == 1752
    assert proof['all_nine_non_full_grade_predicates_passed'] is True
    assert report['schema'] == 'WM02_NATIVE_TASK_PHASE_BIAS_FADE_V8'
    assert report['episode'] == episode and report['actual_spec_phase'] == phase.PHASE
    assert report['finite_xyz_feedback']['committed_feedback_decisions'] == 24
    assert [r['native_step'] for r in report['finite_xyz_feedback']['decisions']] == list(phase.feedback_steps(episode))
    assert adapter.sealed == 46303
    assert not proof['native_execution_authorized'] and not proof['GPU_query_authorized']
    assert proof['new_integrations'] == proof['model_loads'] == proof['queries'] == 0


def test_new_route_quota_and_cli_still_delegate_only_to_original_worker(recipe, tmp_path, monkeypatch):
    spec, previous = recipe['spec'], recipe['previous']
    assert auto1_native.phase_limit(phase.PHASE, previous, spec) == 46303
    assert target_sequential.native_task_bucket(phase.PHASE, True) == '96'
    adapter = target_sequential.TargetSequentialAdapter({'phase': phase.PHASE}, tmp_path)
    token = object()
    monkeypatch.setattr(native, 'run', lambda actual: token)
    assert adapter.dynamic_diagnostic() is token
    path = tmp_path/'CLI_SPEC.json'
    put(path, spec)
    calls = []
    monkeypatch.setitem(sys.modules, 'scripts.wm02.pilot1_native_worker', SimpleNamespace(
        main=lambda argv: calls.append(argv) or 75))
    assert cli.main(['--spec', str(path), '--execute']) == 75
    assert calls == [['--spec', str(path)]]


def test_clean_import_and_cli_help_do_not_load_a_runtime_backend():
    root = Path(__file__).resolve().parents[2]
    code = '''
import importlib
import sys
for name in (
    'go2arx5_vla.world_model.fusion_task_phase_bias_fade_v8',
    'go2arx5_vla.world_model.fusion_task_phase_bias_fade_contract_v8',
    'go2arx5_vla.world_model.fusion_task_phase_bias_fade_native_v8',
    'scripts.wm02.fusion_task_phase_bias_fade_check_v8',
):
    importlib.import_module(name)
forbidden = {'torch', 'mujoco', 'rclpy', 'transformers'}
def check_no_backend():
    loaded = forbidden.intersection(name.split('.')[0] for name in sys.modules)
    if loaded:
        raise AssertionError('runtime backend imported: '+repr(loaded))
check_no_backend()
from scripts.wm02 import fusion_task_phase_bias_fade_check_v8 as cli
try:
    cli.main(['--help'])
except SystemExit as exc:
    if exc.code != 0:
        raise
else:
    raise AssertionError('CLI help did not terminate before checking a recipe')
check_no_backend()
'''
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(root/'src'), str(root), os.environ.get('PYTHONPATH', '')]))
    command = [sys.executable]+(['-O'] if sys.flags.optimize else [])+['-c', code]
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '--execute' in result.stdout and '--check-only' in result.stdout
