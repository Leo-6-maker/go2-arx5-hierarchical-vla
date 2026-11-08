"""Pure math and synthetic native seams; never a physical calibration result."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.pi0292.eval_contract import matrix10
from go2arx5_vla.world_model import auto1_native, target_sequential
from go2arx5_vla.world_model import fusion_task_phase_initialization as original
from go2arx5_vla.world_model import fusion_task_phase_initialization_native as original_native
from go2arx5_vla.world_model import fusion_task_phase_initialization_contract as operational
from go2arx5_vla.world_model import fusion_task_phase_bias_calibration_v2 as bias
from go2arx5_vla.world_model import fusion_task_phase_bias_calibration_contract_v2 as contract
from go2arx5_vla.world_model import fusion_task_phase_bias_calibration_native_v2 as native
from go2arx5_vla.world_model.storage import digest
from scripts.wm02 import fusion_task_phase_bias_calibration_check_v2 as cli
from test_fusion_task_phase_initialization import setup as pure_setup
from test_fusion_task_phase_initialization_native import recipe as original_recipe, TaskAdapter
from test_fusion_live_proposal_qualification_contract import prior_files, put
from test_fusion_trajectory_qualification import setup_flow
from test_fusion_reference_qualification_contract import reference


@pytest.fixture
def setup(monkeypatch):
    return pure_setup.__wrapped__(monkeypatch)


def program(setup):
    episode = bias.make_episode(run='/fixture/fresh-frozen-bias', geometry_scale=.03)
    return bias.make_program(setup['inputs'], episode, setup['initial'], setup['decision'], setup['base']), episode


def test_desired_goal_clock_are_identical_given_same_inputs_only_reference_is_calibrated(setup):
    p, episode = program(setup)
    baseline = original.make_program(setup['inputs'], bias.parent_recipe_view(episode),
        setup['initial'], setup['decision'], setup['base'])
    assert p.end_atom == baseline.end_atom == 128 and bias.endpoint(episode) == 7103
    assert episode['endpoint_transport'] == 'FROZEN_PREDECISION_XYZ_OFFSET_AND_LEFT_SO3_REFERENCE_ENDPOINT'
    assert bias.parent_recipe_view(episode)['endpoint_transport'] == 'NONE_BOTH_PATHS_SHARE_ABSOLUTE_GRADED_WORLD_GOAL'
    with pytest.raises(ValueError):
        bias.episode_contract(dict(episode, endpoint_transport='NONE_BOTH_PATHS_SHARE_ABSOLUTE_GRADED_WORLD_GOAL'))
    assert np.array_equal(p.desired_path, baseline.desired_path)
    assert np.array_equal(p.full_goal_TCP_target, baseline.full_goal_TCP_target)
    assert np.array_equal(p.graded_goal_TCP_target, baseline.graded_goal_TCP_target)
    desired = matrix10(p.desired_path)
    ref = matrix10(p.reference_path)
    b = setup['decision'][:3, 3] - setup['initial'][:3, 3]
    rotation = setup['initial'][:3, :3] @ setup['decision'][:3, :3].T
    np.testing.assert_allclose(ref[:, :3, 3], desired[:, :3, 3]-b, atol=1e-15, rtol=0)
    np.testing.assert_allclose(ref[:, :3, :3], np.einsum('ij,njk->nik', rotation, desired[:, :3, :3]), atol=1e-14, rtol=0)
    np.testing.assert_allclose(ref[0], setup['initial'], atol=1e-15, rtol=0)
    assert not np.array_equal(p.reference_path[-1], baseline.reference_path[-1])
    evidence = p.evidence()
    assert evidence['schema'] == bias.SCHEMA and evidence['recipe'] == bias.RECIPE
    assert evidence['desired_path_construction'] == original.RECIPE
    binding = evidence['frozen_predecision_reference_binding']
    assert binding['largest_measurement_native_step'] == 3703
    assert binding['translation_bias_DEC_minus_INIT_m'] == b.tolist()
    assert binding['reference_path_sha256'] == evidence['reference_path_sha256']
    assert binding['desired_path_sha256'] == evidence['desired_path_sha256']
    assert not binding['historical_terminal_error_used'] and not binding['future_tracking_error_used']
    assert not binding['original_TrajectoryAdaptation_byte_equivalence_claim']
    assert evidence['analytical_rates']['reference'] == evidence['analytical_rates']['desired']


def test_world_component_translation_is_not_silently_replaced_by_a_coupled_se3_transport(setup):
    initial = setup['initial'].copy()
    angle = .2
    initial[:3, :3] = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    episode = bias.make_episode(run='/fixture/noncommuting-rotation', geometry_scale=.03)
    p = bias.make_program(setup['inputs'], episode, initial, setup['decision'], setup['base'])
    desired, ref = matrix10(p.desired_path), matrix10(p.reference_path)
    expected = desired[:, :3, 3] - (setup['decision'][:3, 3]-initial[:3, 3])
    np.testing.assert_allclose(ref[:, :3, 3], expected, atol=1e-15, rtol=0)
    coupled = initial @ np.linalg.inv(setup['decision']) @ desired[-1]
    assert np.linalg.norm(ref[-1, :3, 3]-coupled[:3, 3]) > 1e-5
    np.testing.assert_allclose(ref[0], initial, atol=1e-15, rtol=0)


def test_frozen_arrays_suffix_and_old_version_identity_are_strict(setup):
    p, episode = program(setup)
    for item in (p.reference_path, p.desired_path, p.frozen_translation_bias_m, p.frozen_left_rotation):
        with pytest.raises(ValueError): item.setflags(write=True)
    p.evidence()['frozen_predecision_reference_binding']['largest_measurement_native_step'] = 7103
    assert p.evidence()['frozen_predecision_reference_binding']['largest_measurement_native_step'] == 3703
    for elapsed in (0, 7, 39, 127, 128, 135):
        expected = p.reference_path[np.minimum(elapsed+np.arange(40), 128)]
        assert np.array_equal(p.preview(elapsed), expected)
        result = bias.validate_install_preview(p.preview(elapsed), initial_tcp=setup['initial'], program=p,
            step=3703+25*elapsed, base=setup['base'], frozen_base=setup['base'], end_step=7103)
        assert result['points_checked'] == 40 and not result['native_execution_authorized']
    altered = p.preview(0); altered[39, 0] += .001
    with pytest.raises(ValueError, match='ENTIRE_TRANSFORMED_H40'):
        bias.validate_install_preview(altered, initial_tcp=setup['initial'], program=p,
            step=3703, base=setup['base'], frozen_base=setup['base'], end_step=7103)
    old = original.make_program(setup['inputs'], bias.parent_recipe_view(episode),
        setup['initial'], setup['decision'], setup['base'])
    with pytest.raises(ValueError, match='EXACT_NEW_TYPED'):
        bias.validate_install_preview(old.preview(), initial_tcp=setup['initial'], program=old,
            step=3703, base=setup['base'], frozen_base=setup['base'], end_step=7103)
    for alpha in (.1, .25, .5, 1., True):
        with pytest.raises(ValueError): bias.make_episode(run='/fixture/rejected-expansion', geometry_scale=alpha)
    with pytest.raises(ValueError): bias.episode_contract(bias.parent_recipe_view(episode))
    distant = setup['initial'].copy(); distant[0, 3] -= 10
    with pytest.raises(ValueError, match='NO_RETIME_NO_CLIP'):
        bias.make_program(setup['inputs'], episode, distant, setup['decision'], setup['base'])


@pytest.fixture
def recipe(tmp_path, monkeypatch):
    spec, _, _, setup = original_recipe.__wrapped__(tmp_path, monkeypatch)
    capsule, project = Path(spec['source_root']), Path(__file__).resolve().parents[2]
    for name in contract.SOURCE_MEMBERS:
        path = capsule/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((project/name).read_bytes()); spec['source_files'][name] = digest(path)
    prior_root = tmp_path/'SYNTHETIC_ABSOLUTE_INIT_POINT03'
    prior_episode = original.make_episode(run=str(prior_root), geometry_scale=.03)
    previous, result_path, _, spec_pin = prior_files(prior_root, prior_episode, spec,
                                                   source=contract.FIRST_PREDECESSOR_SOURCE)
    artifact = {}
    for key in ('goal_binding', 'readiness', 'hold_samples'):
        path = prior_root/(key+'.SYNTHETIC.json')
        artifact.update({key+'_path': str(path), key+'_sha256': put(path, {'synthetic': True})})
    previous['reference_qualification']['task_phase_initialization'] = dict(artifact,
        actual_VLA_commands=0, VLA_queries=0, object_and_goal_are_policy_inputs=False)
    prior_pin = put(result_path, previous)
    monkeypatch.setattr(contract, 'FIRST_PREDECESSOR', prior_pin)
    monkeypatch.setattr(contract, 'FIRST_PREDECESSOR_SPEC', spec_pin)
    spec.update(phase=contract.PHASE, kind=contract.KIND, previous_stage=str(result_path), previous_stage_sha256=prior_pin)
    episode = bias.make_episode(run=spec['registered_root'], geometry_scale=.03)
    spec['episode_manifest_sha256'] = put(spec['episode_manifest'], episode)
    spec['max_ticks'] = 7103
    return spec, previous, episode, setup


def test_new_contract_and_cli_require_complete_exact_absolute_point03_predecessor(recipe, tmp_path):
    spec, previous, episode, _ = recipe
    assert contract.episode_limit(spec, previous) == 7103
    actual, _ = contract.episode_config(spec)
    assert actual == episode and actual['phase'] == bias.PHASE
    path = tmp_path/'SPEC.json'; put(path, spec)
    value = cli.check(path)
    assert value['initialization_condition'] == bias.CONDITION
    assert not value['historical_terminal_error_used'] and not value['native_execution_authorized']
    assert value['new_integrations'] == value['VLA_queries'] == value['WM_updates'] == 0


def test_generic_operational_seam_retains_known_caller_and_all_original_source_checks(recipe):
    spec, _, episode, _ = recipe
    with pytest.raises(ValueError, match='EXPLICIT_KNOWN_RECIPE_CALLER'):
        operational.recipe_config(spec, episode, actual_phase=contract.PHASE,
            actual_kind=contract.KIND, source_members=())
    with pytest.raises(ValueError, match='EXPLICIT_KNOWN_RECIPE_CALLER'):
        operational.recipe_config(spec, episode, actual_phase='UNDECLARED_NEW_PHASE',
            actual_kind='UNDECLARED_NEW_KIND', source_members=contract.SOURCE_MEMBERS)


def test_new_cli_execute_delegates_original_worker_without_new_dispatch_or_model(recipe, tmp_path, monkeypatch):
    spec, *_ = recipe
    path = tmp_path/'SPEC_CLI.json'; put(path, spec)
    calls = []
    monkeypatch.setitem(sys.modules, 'scripts.wm02.pilot1_native_worker',
        SimpleNamespace(main=lambda argv: calls.append(argv) or 75))
    assert cli.main(['--spec', str(path), '--execute']) == 75
    assert calls == [['--spec', str(path)]]


@pytest.mark.parametrize('fault', ['wrong-kind', 'old-episode', 'wrong-result', 'open-lease',
    'trace-drop', 'truncated', 'unconsumed-gripper', 'lost-goal'])
def test_new_condition_cannot_relabel_old_inputs_or_missing_native_evidence(recipe, fault, monkeypatch):
    spec, previous, episode, _ = recipe
    if fault == 'wrong-kind': spec['kind'] = 'FUSION_TASK_PHASE_INITIALIZATION'
    elif fault == 'old-episode': spec['episode_manifest_sha256'] = put(spec['episode_manifest'], bias.parent_recipe_view(episode))
    elif fault == 'wrong-result': spec['previous_stage_sha256'] = 'e'*64
    elif fault == 'open-lease': put(Path(spec['previous_stage']).parent.parent/'control/LEASE.json', {'status': 'ACTIVE'})
    elif fault == 'trace-drop': previous['trace_audit']['writer']['dropped'] = 1
    elif fault == 'truncated': previous['reference_qualification']['windows'].pop()
    elif fault == 'unconsumed-gripper': previous['reference_qualification']['windows'][-1]['gripper']['command_consumed'] = False
    else: Path(previous['reference_qualification']['task_phase_initialization']['goal_binding_path']).write_text('{}')
    if fault in ('trace-drop', 'truncated', 'unconsumed-gripper'):
        spec['previous_stage_sha256'] = put(spec['previous_stage'], previous)
        monkeypatch.setattr(contract, 'FIRST_PREDECESSOR', spec['previous_stage_sha256'])
    with pytest.raises((ValueError, PermissionError)):
        contract.episode_limit(spec, previous)


class BiasAdapter(TaskAdapter):
    def _sample(self):
        result = super()._sample()
        if self.step > 2503:
            result['tcp'] = self.setup['decision'].copy()
        return result

    def install_reference(self, action):
        native.validate_adapter_install(self, action)
        if self.step >= 3703:
            saved = json.loads((self.output/'TASK_PHASE_GOAL_BINDING.private.json').read_text())
            assert saved['actual_spec_phase'] == bias.PHASE
            frozen = saved['frozen_predecision_reference_calibration']
            assert frozen['largest_measurement_native_step'] == 3703
            assert not frozen['future_tracking_error_used']
        self.active_reference = dict(reference(round(self.step*.002*1e9)/1e9), action_EEF10=action.tolist())
        self.installs.append((self.step, action.copy()))
        return self.active_reference


def test_original_loop_saves_frozen_bias_before_install_and_real_tail_without_model_or_future_error(recipe, tmp_path, monkeypatch):
    spec, _, episode, setup = recipe
    output = tmp_path/'SYNTHETIC_BIAS_OUTPUT'; output.mkdir()
    adapter = BiasAdapter(spec, output, setup); setup_flow(monkeypatch, adapter)
    _, source = contract.episode_config(spec)
    monkeypatch.setattr(contract, 'episode_config', lambda value: (episode, source))
    monkeypatch.setattr(original_native, '_verify_actual_fk', lambda *a: {'status': 'SYNTHETIC_NO_NATIVE_CLAIM'})
    report = native.run(adapter)
    assert adapter.sealed == 7103 and report['complete_task_atoms'] == report['gripper_confirmed_atoms'] == 184
    assert report['all_40_knots_consumed'] and report['actual_trace_binding']['actual_trace_capacity'] == 524288
    assert report['initialization_condition'] == bias.CONDITION
    assert report['initialization_calibration_not_VLA_task_transfer_result']
    binding = report['task_phase_initialization']
    frozen = binding['frozen_predecision_reference_calibration']
    assert frozen['translation_bias_DEC_minus_INIT_m'] == (setup['decision'][:3, 3]-setup['initial'][:3, 3]).tolist()
    assert digest(binding['goal_binding_path']) == binding['goal_binding_sha256']
    assert digest(binding['hold_samples_path']) == binding['hold_samples_sha256']
    assert binding['readiness_sample_native_steps'] == list(range(6903, 7104, 25))
    decision = json.loads((output/'TRAJECTORY_QUALIFICATION_DECISION.private.json').read_text())
    assert decision['episode'] == episode and len(decision['desired_path']) == len(decision['nominal_reference_path']) == 129
    assert decision['adaptation']['frozen_predecision_reference_binding'] == frozen
    assert report['actual_VLA_commands'] == report['VLA_queries'] == report['WM_updates'] == report['Actor_updates'] == 0
    assert not report['historical_terminal_tracking_error_used'] and not report['full_task_success']


def test_new_route_is_typed_bounded_and_import_has_no_backends(tmp_path, monkeypatch):
    monkeypatch.setattr(contract, 'episode_limit', lambda s, p: 7103)
    assert auto1_native.phase_limit(bias.PHASE, {}, {'phase': bias.PHASE}) == 7103
    adapter = target_sequential.TargetSequentialAdapter({'phase': bias.PHASE}, tmp_path)
    token = object(); monkeypatch.setattr(native, 'run', lambda a: token)
    assert adapter.dynamic_diagnostic() is token
    assert target_sequential.native_task_bucket(bias.PHASE, True) == '64'
    code = ('import sys;from go2arx5_vla.world_model import fusion_task_phase_bias_calibration_v2;'
        'from go2arx5_vla.world_model import fusion_task_phase_bias_calibration_native_v2;'
        'assert not {"torch","mujoco","rclpy","transformers","gr00t"}&sys.modules.keys()')
    subprocess.run([sys.executable, '-B', '-c', code], check=True)
