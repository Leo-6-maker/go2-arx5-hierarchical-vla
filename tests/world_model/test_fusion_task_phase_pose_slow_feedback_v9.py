"""Pure V9 geometry and causal-clock checks; no native dynamics qualification."""
import copy
from dataclasses import replace
import hashlib

import numpy as np
import pytest

from scripts.pi0292.eval_contract import matrix10
from go2arx5_vla.world_model import fusion_task_phase_pose_slow_feedback_v9 as phase
from go2arx5_vla.world_model import fusion_task_phase_pose_continuation_v7 as v7
from go2arx5_vla.world_model import fusion_task_phase_pose_feedback_v6 as v6
from test_fusion_task_phase_initialization import setup as pure_setup, samples


@pytest.fixture
def setup(monkeypatch):
    return pure_setup.__wrapped__(monkeypatch)


def make(setup):
    episode = phase.make_episode(run='/fixture/pure-v9', geometry_scale=.5)
    return phase.make_program(setup['inputs'], episode, setup['initial'], setup['decision'], setup['base']), episode


def measured(program, *, xyz=(.03, -.04, .02), rotation=(.03, -.02, .04)):
    actual = program.graded_goal_TCP_target.copy()
    actual[:3, 3] -= xyz
    actual[:3, :3] = v6._exp_so3(-np.asarray(rotation))@actual[:3, :3]
    return actual


def install_check(setup, program, step, *, action=None, end=34303):
    if action is None:
        action = program.preview((step-phase.DECISION_STEP)//25)
    return phase.validate_install_preview(action, initial_tcp=setup['initial'], program=program,
        step=step, base=setup['base'], frozen_base=setup['base'], end_step=end)


def test_exact_twelve_slot_clock_same_endpoint_and_original_frozen_geometry(setup):
    program, episode = make(setup)
    old_episode = v7.make_episode(run=episode['run'], geometry_scale=.5)
    old = v7.make_program(setup['inputs'], old_episode, setup['initial'], setup['decision'], setup['base'])
    assert phase.endpoint(episode) == v7.endpoint(old_episode) == 34303
    assert episode['ramp_atoms'] == 768 and episode['planned_end_atom'] == program.end_atom == 1216
    assert phase.feedback_steps(episode) == tuple(23103+900*i for i in range(12))
    assert episode['feedback_move_atoms'] == 28 and episode['feedback_settle_atoms'] == 8
    assert phase.feedback_steps(episode)[-1]+28*25 == 33703
    assert (phase.endpoint(episode)-33703)//25 == phase.FINAL_HOLD_ATOMS == 24
    assert program.reference_path.shape == program.desired_path.shape == (1217, 10)
    np.testing.assert_array_equal(program.desired_path, old.desired_path)
    np.testing.assert_array_equal(program.reference_path, old.reference_path)
    np.testing.assert_array_equal(program.graded_goal_TCP_target, old.graded_goal_TCP_target)
    np.testing.assert_array_equal(program.full_goal_TCP_target, old.full_goal_TCP_target)
    np.testing.assert_array_equal(program.base_pose_hold, old.base_pose_hold)
    view = phase.parent_recipe_view(episode)
    assert view == phase.original.make_episode(run=episode['run'], geometry_scale=.5)
    assert view['planned_end_atom'] == 768 and episode['schema'] != view['schema']
    proof = program.evidence()
    assert proof['schema'] == phase.SCHEMA and proof['baseline_condition'] == v7.CONDITION
    assert proof['clock']['end_atom'] == 1216 and proof['clock']['feedback_interval_atoms'] == 36
    assert proof['clock']['native_dt_s'] == .002 and proof['clock']['final_hold_atoms'] == 24


@pytest.mark.parametrize('grade', [.03, .1, .25, 1., True, np.float64(.5), '0.5'])
def test_only_half_grade_is_registered(grade):
    with pytest.raises(ValueError, match='HALF_GRADE_ONLY'):
        phase.make_episode(run='/fixture/refuse', geometry_scale=grade)


@pytest.mark.parametrize('fault', ['extra-key', 'wrong-phase', 'wrong-slots', 'float-slot',
                                  'extra-slot', 'move20', 'wrong-end', 'wrong-ramp', 'extra-dose'])
def test_manifest_rejects_unregistered_keys_or_dose_and_clock_changes(setup, fault):
    _, episode = make(setup)
    broken = copy.deepcopy(episode)
    if fault == 'extra-key':
        broken['permission'] = True
    elif fault == 'wrong-phase':
        broken['phase'] = v7.PHASE
    elif fault == 'wrong-slots':
        broken['feedback_decision_native_steps'][1] -= 200
    elif fault == 'float-slot':
        broken['feedback_decision_native_steps'][0] = float(broken['feedback_decision_native_steps'][0])
    elif fault == 'extra-slot':
        broken['feedback_decision_native_steps'].append(33903)
    elif fault == 'move20':
        broken['feedback_move_atoms'] = 20
    elif fault == 'wrong-end':
        broken['planned_end_atom'] = 1208
    elif fault == 'wrong-ramp':
        broken['ramp_atoms'] = 1024
    else:
        broken['finite_feedback_decisions'] = 13
    with pytest.raises(ValueError):
        phase.episode_contract(broken)
    for old in (v6, v7):
        with pytest.raises(ValueError):
            old.episode_contract(episode)


def test_bounded_correction_parity_with_new_quintic_and_past_preview_hashes(setup):
    program, episode = make(setup)
    step = phase.feedback_steps(episode)[0]
    atom = (step-phase.DECISION_STEP)//25
    actual = measured(program)
    before, desired = program.reference_path.copy(), program.desired_path.copy()
    # Call the original pure calculation on this same state and caps: only its
    # old candidate clock differs, and neither candidate is installed here.
    bounded = v6.FeedbackProgram.prepare_feedback(program, step, actual)
    update = program.prepare_feedback(step, actual)
    proof = update.evidence
    for key in ('error_world_xyz_m', 'applied_delta_world_m', 'raw_gain_delta_world_m',
                'angular_error_target_world_rotvec_rad', 'applied_rotation_target_world_rad',
                'cumulative_reference_path_length_m', 'cumulative_reference_rotation_path_rad'):
        assert proof[key] == bounded.evidence[key]
    delta = np.asarray(proof['applied_delta_world_m'])
    rotation = np.asarray(proof['applied_rotation_target_world_rad'])
    start = matrix10(before[atom])
    for offset in (0, 7, 14, 21, 28):
        u = offset/28
        weight = 10*u**3-15*u**4+6*u**5
        np.testing.assert_allclose(update.reference_path[atom+offset, :3],
                                   before[atom, :3]+weight*delta, rtol=0, atol=1e-15)
        np.testing.assert_allclose(matrix10(update.reference_path[atom+offset])[:3, :3],
                                   v6._exp_so3(weight*rotation)@start[:3, :3], rtol=0, atol=2e-14)
    np.testing.assert_array_equal(update.reference_path[:atom+1], before[:atom+1])
    np.testing.assert_array_equal(update.reference_path[:, 9], before[:, 9])
    endpoint = update.reference_path[atom+28]
    np.testing.assert_array_equal(update.reference_path[atom+28:],
                                  np.repeat(endpoint[None], len(before)-atom-28, axis=0))
    np.testing.assert_array_equal(program.reference_path, before)
    np.testing.assert_array_equal(program.desired_path, desired)
    np.testing.assert_allclose(update.reference_path[-1], bounded.reference_path[-1], rtol=0, atol=0)
    assert not np.array_equal(update.reference_path[atom+20], endpoint)
    preview = update.reference_path[np.minimum(atom+np.arange(40), program.end_atom)]
    assert proof['new_H40_sha256'] == phase.fixed._sha(preview)
    assert proof['reference_path_before_sha256'] == phase.fixed._sha(before)
    assert proof['reference_path_after_sha256'] == phase.fixed._sha(update.reference_path)
    assert proof['unchanged_desired_path_sha256'] == phase.fixed._sha(desired)
    assert proof['move_atoms'] == 28 and proof['settle_atoms'] == 8
    assert proof['move_duration_s'] == pytest.approx(1.4)
    assert proof['native_execution_authorized'] is False
    for speed, acceleration in (('peak_translation_speed_m_s', 'peak_translation_acceleration_m_s2'),
                                ('peak_angular_speed_rad_s', 'peak_angular_acceleration_rad_s2')):
        assert proof[speed] == pytest.approx(bounded.evidence[speed]/1.4)
        assert proof[acceleration] == pytest.approx(bounded.evidence[acceleration]/1.4**2)
    dt = .05
    xyz = update.reference_path[atom:atom+29, :3]
    assert np.linalg.norm(np.diff(xyz, axis=0)/dt, axis=1).max() <= proof['peak_translation_speed_m_s']+1e-12
    assert np.linalg.norm(np.diff(xyz, n=2, axis=0)/dt**2, axis=1).max() <= proof['peak_translation_acceleration_m_s2']+1e-12
    with pytest.raises(ValueError):
        update.reference_path[atom+1, 0] += .001


def test_all_twelve_commits_preserve_desired_and_caps_with_no_extra_decision(setup):
    program, episode = make(setup)
    desired = program.desired_path.copy()
    final_update = None
    for i, step in enumerate(phase.feedback_steps(episode)):
        before = program.reference_path.copy()
        actual = measured(program, xyz=(((-1)**i)*.05, 0., 0.), rotation=(0., .05, 0.))
        update = program.prepare_feedback(step, actual)
        with pytest.raises(ValueError, match='NO_SCHEDULED_REFERENCE'):
            install_check(setup, program, step)
        pin = hashlib.sha256(str(i).encode()).hexdigest()
        program.commit_feedback(update, saved_decision_sha256=pin)
        assert program.feedback_count == i+1
        atom = (step-phase.DECISION_STEP)//25
        np.testing.assert_array_equal(program.reference_path[:atom+1], before[:atom+1])
        np.testing.assert_array_equal(program.desired_path, desired)
        np.testing.assert_array_equal(program.preview(atom), update.reference_path[atom:atom+40])
        accepted = install_check(setup, program, step)
        assert accepted['original_certify_chunk_required'] and not accepted['native_execution_authorized']
        assert program.evidence()['feedback_decisions'][-1]['saved_decision_sha256'] == pin
        final_update = update
    assert program.evidence()['cumulative_feedback_reference_path_length_m'] == pytest.approx(.096)
    assert program.evidence()['cumulative_feedback_reference_rotation_path_rad'] == pytest.approx(.12)
    np.testing.assert_array_equal(program.preview(1210)[6:], np.repeat(program.reference_path[-1][None], 34, axis=0))
    with pytest.raises(ValueError):
        program.prepare_feedback(33903, actual)
    with pytest.raises(ValueError):
        program.commit_feedback(final_update, saved_decision_sha256='a'*64)


@pytest.mark.parametrize('fault', ['no-sha', 'wrong-step', 'wrong-revision', 'tampered-path', 'tampered-proof'])
def test_commit_recomputes_same_v9_update_and_rejects_unsaved_or_tampered_candidate(setup, fault):
    program, episode = make(setup)
    step = phase.feedback_steps(episode)[0]
    update = program.prepare_feedback(step, measured(program))
    before = program.reference_path.copy()
    pin = 'a'*64
    if fault == 'no-sha':
        pin = ''
    elif fault == 'wrong-step':
        update = replace(update, native_step=step+25)
    elif fault == 'wrong-revision':
        update = replace(update, revision_before=1)
    elif fault == 'tampered-path':
        path = update.reference_path.copy()
        path[-1, 0] += .001
        update = replace(update, reference_path=path)
    else:
        proof = copy.deepcopy(update.evidence)
        proof['move_atoms'] = 20
        update = replace(update, evidence=proof)
    with pytest.raises(ValueError):
        program.commit_feedback(update, saved_decision_sha256=pin)
    assert program.feedback_count == 0
    np.testing.assert_array_equal(program.reference_path, before)


def test_independent_deadbands_pending_motion_and_exact_decision_clock(setup):
    program, episode = make(setup)
    step = phase.feedback_steps(episode)[0]
    hold = program.prepare_feedback(step, measured(program, xyz=(.003, 0., 0.), rotation=(0., 0., 0.)))
    assert hold.evidence['deadband_hold'] and hold.evidence['angular_deadband_hold']
    np.testing.assert_array_equal(hold.reference_path, program.reference_path)
    rotation_only = program.prepare_feedback(step, measured(program, xyz=(0., 0., 0.)))
    assert rotation_only.evidence['applied_delta_norm_m'] == 0
    assert rotation_only.evidence['applied_rotation_norm_rad'] == pytest.approx(.01)
    for bad_step in (step-25, step+25, float(step), phase.feedback_steps(episode)[1]):
        with pytest.raises(ValueError):
            program.prepare_feedback(bad_step, measured(program))
    pending = program.reference_path.copy()
    pending[(step-phase.DECISION_STEP)//25+3, 0] += .001
    object.__setattr__(program, 'reference_path', phase.fixed._immutable(pending))
    with pytest.raises(ValueError, match='PENDING_MOTION_OVERWRITE'):
        program.prepare_feedback(step, measured(program))


@pytest.mark.parametrize('field,used', [('_cumulative_feedback_m', .095),
                                       ('_cumulative_feedback_rad', .115)])
def test_sum_of_norm_caps_reject_a_correction_without_mutating_the_path(setup, field, used):
    program, episode = make(setup)
    before = program.reference_path.copy()
    object.__setattr__(program, field, used)
    with pytest.raises(ValueError, match='FINITE_XYZ_AND_ANGULAR_SUM_OF_NORMS'):
        program.prepare_feedback(phase.feedback_steps(episode)[0], measured(program))
    assert program.feedback_count == 0
    np.testing.assert_array_equal(program.reference_path, before)


def test_shape_typed_program_full_preview_and_endpoint_are_still_required(setup):
    program, episode = make(setup)
    step = phase.DECISION_STEP
    install_check(setup, program, step)
    broken = program.preview(0)
    broken[39, 0] += .001
    with pytest.raises(ValueError, match='ENTIRE_TRANSFORMED_H40'):
        install_check(setup, program, step, action=broken)
    with pytest.raises(ValueError):
        install_check(setup, program, step, end=34328)
    old_episode = v7.make_episode(run=episode['run'], geometry_scale=.5)
    old = v7.make_program(setup['inputs'], old_episode, setup['initial'], setup['decision'], setup['base'])
    with pytest.raises(ValueError, match='EXACT_TYPED_V9'):
        install_check(setup, old, step)
    object.__setattr__(program, 'reference_path', phase.fixed._immutable(program.reference_path[:-1]))
    with pytest.raises(ValueError, match='EXACT_ORIGINAL_END_SHAPE'):
        program.prepare_feedback(phase.feedback_steps(episode)[0], measured(program))


def test_original_measurement_readiness_stays_half_grade_not_query_permission(setup):
    _, episode = make(setup)
    actual = samples(setup, .5)
    for row, step in zip(actual, range(34103, 34304, 25)):
        row.update(native_step=step, time_s=step*.002)
    readiness = phase.query_readiness(actual, episode, setup['inputs'])
    old_episode = v7.make_episode(run=episode['run'], geometry_scale=.5)
    old = v7.query_readiness(actual, old_episode, setup['inputs'])
    assert readiness['predicates'] == old['predicates']
    assert readiness['metrics'] == old['metrics']
    assert readiness['predicates']['full_grade'] is False and readiness['query_ready'] is False
    assert not readiness['native_execution_authorized'] and not readiness['GPU_query_authorized']
    assert readiness['actual_sample_steps'] == list(range(34103, 34304, 25))
    actual[-1]['native_step'] += 25
    actual[-1]['time_s'] += .05
    with pytest.raises(ValueError, match='ABSOLUTE_CLOCK'):
        phase.query_readiness(actual, episode, setup['inputs'])
