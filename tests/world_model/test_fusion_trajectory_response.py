"""Small synthetic fixtures ONLY: no real fit, native execution, or network."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from go2arx5_vla.world_model import fusion_trajectory_response_contract as c
from go2arx5_vla.world_model import fusion_trajectory_response_replay as r
from go2arx5_vla.world_model import fusion_trajectory_response as m
from go2arx5_vla.world_model.fusion_trajectory_adapter import TrajectoryAdaptation, WORKING_OPEN_FRACTION


BASE = np.array([0., 0., .4, 0., 0., 0., 1.])


def panel():
    return c.make_panel(physical_parents={**{f'T{i}': 'TRAIN' for i in range(4)},
                                        **{f'D{i}': 'DEV' for i in range(2)}},
                        source_sha='a' * 40,
                        query_pins={key: 'b' * 64 for key in (
                            'checkpoint', 'spec', 'result', 'action', 'observation', 'frame_map')})


def eef(xyz):
    xyz = np.asarray(xyz)
    return np.column_stack((xyz, np.tile([1., 0., 0., 0., 1., 0.], (len(xyz), 1)),
                            np.full(len(xyz), WORKING_OPEN_FRACTION)))


def tcp(xyz):
    result = np.eye(4); result[:3, 3] = xyz
    return result


def episode(p, parent='T0', context='C0', theta=0, restart=0, end=80):
    return dict(schema=c.EPISODE_SCHEMA, phase=c.PHASE,
        purpose='TRAJECTORY_RESPONSE_IDENTIFICATION', reference_rule=c.RECIPE,
        panel_sha256=c.canonical_sha256(p), geometry_scale=1., query_pins=p['query_pins'].copy(),
        run=f'synthetic-only/{parent}/{context}/{theta}/{restart}', scene_id='synthetic-' + parent,
        parent=parent, split=p['physical_parents'][parent], context_id=context,
        theta_level=theta, restart_index=restart, planned_end_atom=end)


def record(p, parent='T0', context='C0', theta=0, restart=0):
    """Fabricated mathematical records explicitly confined to this test file."""
    ep = episode(p, parent, context, theta, restart)
    atom = np.arange(ep['planned_end_atom'] + 1)
    nominal = eef(np.column_stack((.00002 * atom, .00005 * np.sin(atom / 5), -.00001 * atom)))
    basis = c.pulse_basis(atom)
    corrected = nominal.copy(); corrected[:, 0] += theta * c.EPSILON_M * basis
    raw = nominal[:40].astype(np.float32)
    parent_number = int(parent[1:]) + (0 if parent[0] == 'T' else 4)
    lag = (-1 if context == 'C0' else 1) * .00015 + parent_number * .000003
    velocity = (1 if context == 'C0' else -1) * .000002
    anchor = np.array([lag, .00001 * parent_number, .2])
    context_program = c.ContextProgram(tcp([0., 0., .2]), BASE, context)
    windows = []
    states = []
    for index in range(48 + ep['planned_end_atom'] + 8):
        phase = index - 48
        if phase < 0:
            xyz = anchor + [velocity * (phase + 1), 0., 0.]
            consumed = context_program.preview(index)[0]
            installed = context_program.preview(index)
        else:
            t = min(phase + 1, len(atom) - 1)
            noise = (1 if restart else -1) * .0000008 * min(t, 16) / 16
            # Nonzero state×theta effect; also an additive observed-lag effect.
            gain = .0003 + 1.5 * lag
            xyz = anchor + nominal[t, :3] - nominal[0, :3]
            xyz += [theta * gain * basis[t] + lag * min(t, 16) / 40 + noise, 0., 0.]
            consumed = corrected[min(phase, len(atom) - 1)]
            origin = 7 * (phase // 7)
            installed = corrected[np.minimum(origin + np.arange(40), len(atom) - 1)].copy()
        state = np.zeros(106); state[86:89] = xyz
        state[89:95] = [1., 0., 0., 0., 1., 0.]
        state[104:106] = .042
        states.append(state)
        start = 2503 + 25 * index
        row = dict(native_start_step=start, native_end_step=start + 25, actual_ticks=25,
            solver_policies_with_exact_installed_target=[dict(synthetic_only=True)],
            gripper=dict(command_consumed=True), full_final_installed_H40=installed,
            actual_installed_consumed_EEF10_BASE7=np.r_[consumed, BASE],
            probe_elapsed_atom=phase if phase >= 0 else None,
            actual_TCP=tcp(xyz), actual_state_vector=state,
            clock=dict(native_start_s=start * .002, source_mpc_s=start * .002,
                       remaining_at_install_s=.4))
        windows.append(row)
    history = np.asarray(states[39:48])
    windows[48]['actual_history_before_atom'] = dict(actual_state_history=history,
        actual_history_native_steps=3703 + 25 * np.arange(-8, 1),
        past_consumed_EEF10_BASE7=np.asarray([w['actual_installed_consumed_EEF10_BASE7'] for w in windows[40:48]]),
        past_reference_native_end_steps=3703 + 25 * np.arange(-7, 1),
        full_MPC_hidden_state_restored=False)
    pins = {key: hashlib.sha256((ep['run'] + key).encode()).hexdigest() for key in (
        'native_result_sha256', 'report_sha256', 'episode_sha256', 'decision_sha256', 'raw_action_sha256')}
    pins['raw_action_sha256'] = p['query_pins']['action']
    assignment = dict(native_step=3703, theta_level=theta, future_response_observed=False)
    decision = dict(schema='WM02_TRAJECTORY_RESPONSE_DECISION_V1', assignment_before_install=assignment,
        full_nominal_reference_path=nominal, full_corrected_reference_path=corrected,
        desired_path=nominal.copy(), raw_H40=raw.tolist(), raw_H40_dtype=raw.dtype.str,
        raw_H40_bytes_sha256=hashlib.sha256(raw.tobytes()).hexdigest())
    report = dict(schema='WM02_NATIVE_TRAJECTORY_RESPONSE_COLLECTION_V1', episode=ep, panel_sha256=c.canonical_sha256(p), all_40_knots_consumed=True,
        assignment_before_install=assignment, decision_record=dict(sha256=pins['decision_sha256']),
        complete_task_atoms=len(windows), solver_target_confirmed_atoms=len(windows),
        gripper_confirmed_atoms=len(windows), windows=windows)
    native = dict(phase=c.PHASE, status='NATIVE_STAGE_PASSED', shutdown_clean=True, physics_stops=[],
        trace_audit=dict(status='RECORDED_TRACE_CHECKS_PASS', errors=[], missing=[]), actual_target_integrations=windows[-1]['native_end_step'])
    return dict(episode=ep, native=native, report=report, decision=decision, raw_H40=raw, pins=pins)


@pytest.fixture(scope='module')
def cohort():
    p = panel()
    records = [record(p, parent, context, theta, restart)
               for parent in p['physical_parents'] for context in c.CONTEXT_IDS
               for theta in c.THETA_LEVELS for restart in c.REPEAT_INDICES]
    data, meta = r.build_replay(records, p)
    return p, data, meta


def test_panel_episode_are_exact_new72_contract_and_do_not_claim_native_permission():
    p = panel(); ep = episode(p)
    c.validate_episode(ep, p)
    assert p['time_stretch_factor'] == 2
    assert p['nominal_recipe'].endswith('_2X_V2')
    for change in [dict(geometry_scale=.03), dict(theta_levels=[0, 1]), dict(epsilon_m=.001),
                   dict(history_atoms=7), dict(native_dt_s=.001), dict(extra_permission=True)]:
        with pytest.raises(ValueError): c.validate_panel(dict(p, **change))
    for change in [dict(split='DEV'), dict(theta_level=True), dict(restart_index=2),
                   dict(panel_sha256='f' * 64), dict(arbitrary_correction=True)]:
        with pytest.raises(ValueError): c.validate_episode(dict(ep, **change), p)


def test_context_roundtrip_is_independent_of_future_theta_and_continuous_at_decision():
    initial = tcp([.2, -.1, .4])
    c0, c1 = [c.ContextProgram(initial, BASE, name) for name in c.CONTEXT_IDS]
    np.testing.assert_array_equal(c0.preview(48), c1.preview(48))
    assert c1.preview(8)[0, 0] - initial[0, 3] == pytest.approx(.0005)
    assert c1.preview(40)[0, 0] - initial[0, 3] == pytest.approx(.0005)
    assert c1.preview(44)[0, 0] - initial[0, 3] == pytest.approx(.00025)
    np.testing.assert_array_equal(c1.end_TCP, initial)
    assert not c1.evidence()['theta_published']
    assert not c1.evidence()['native_execution_authorized']


def test_reference_only_pulse_preserves_full_geometry_and_common_2x_clock():
    raw = eef(np.column_stack((np.linspace(0, .004, 40), np.zeros((40, 2)))))
    base = TrajectoryAdaptation(raw, np.eye(4), np.eye(4), np.eye(4), np.eye(4), BASE, 1., time_stretch_factor=2)
    programs = [c.ResponseProgram(base, theta, panel()) for theta in c.THETA_LEVELS]
    for theta, program in zip(c.THETA_LEVELS, programs):
        assert program.end_atom == base.end_atom
        assert program.knot_indices == base.knot_indices
        np.testing.assert_array_equal(program.desired_path, base.desired_path)
        np.testing.assert_array_equal(program.reference_path[:, 3:], base.reference_path[:, 3:])
        np.testing.assert_allclose(program.reference_path[:, :3] - base.reference_path[:, :3],
            theta * c.EPSILON_M * c.pulse_basis(np.arange(base.end_atom + 1))[:, None] * c.DIRECTION)
        assert program.preview(8)[0, 0] - program.nominal_preview(8)[0, 0] == pytest.approx(theta * .0005)
        np.testing.assert_array_equal(program.preview(24), program.nominal_preview(24))
        json.dumps(program.evidence(), allow_nan=False)
        assert not program.evidence()['theta_specific_retiming']
    with pytest.raises(ValueError):
        c.ResponseProgram(TrajectoryAdaptation(raw, np.eye(4), np.eye(4), np.eye(4), np.eye(4), BASE, 1.), 0, panel())


@pytest.mark.parametrize('remaining,continuous,whole', [(.4, 8., 7), (.375, 7.5, 7), (.35, 7., 6), (.05, 1., 0)])
def test_ttl_seconds_to_atoms_matches_original_strict_expiry(remaining, continuous, whole):
    count, live = r.ttl_atoms(remaining)
    assert count == pytest.approx(continuous)
    assert live == whole
    with pytest.raises(ValueError): r.ttl_atoms(0)


def test_replay_preserves_complete_installed_H40_and_one_probe_per_physical_run(cohort):
    p, data, metadata = cohort
    assert data['history_state'].shape == (72, 9, 106)
    assert data['all_installed_H40'].shape == (72, 136, 40, 10)
    assert data['raw_H40'].dtype == np.float32
    assert metadata['one_predeclared_probe_per_run']
    assert not metadata['overlapping_windows_are_independent_samples']
    assert metadata['state_rotation'] == 'COLUMNS' and metadata['EEF10_rotation'] == 'ROWS'
    split = r.validate_cohort(data, p)
    assert (len(split['TRAIN']), len(split['DEV'])) == (48, 24)
    for parent, fit, held in r.whole_parent_folds(data, p):
        assert len(fit) == 36 and len(held) == 12
        assert parent not in data['parent'][fit]
        assert set(data['split'][fit]) == {'TRAIN'}


@pytest.mark.parametrize('failure', ['future_assignment', 'unsafe', 'missing_grip', 'future_history',
                                  'past_ref_leak', 'corrected_grip', 'clock_units', 'unconsumed_suffix', 'later_suffix', 'raw_dtype', 'old_phase', 'wrong_action_pin'])
def test_replay_rejects_invalid_causal_or_consumption_records(failure):
    p = panel(); rec = record(p)
    if failure == 'future_assignment': rec['decision']['assignment_before_install'] = dict(rec['decision']['assignment_before_install'], future_response_observed=True)
    elif failure == 'unsafe': rec['native']['physics_stops'] = ['synthetic-stop']
    elif failure == 'missing_grip': rec['report']['windows'][60]['gripper']['command_consumed'] = False
    elif failure == 'future_history': rec['report']['windows'][48]['actual_history_before_atom']['actual_history_native_steps'][-1] += 25
    elif failure == 'past_ref_leak': rec['report']['windows'][48]['actual_history_before_atom']['past_consumed_EEF10_BASE7'][0, 0] += 1.
    elif failure == 'corrected_grip': rec['decision']['full_corrected_reference_path'][15, 9] = .4
    elif failure == 'clock_units': rec['report']['windows'][48]['clock']['remaining_at_install_s'] = 8.
    elif failure == 'unconsumed_suffix': rec['report']['windows'][48]['full_final_installed_H40'][39, 0] += .001
    elif failure == 'later_suffix': rec['report']['windows'][62]['full_final_installed_H40'][39, 0] += .001
    elif failure == 'raw_dtype': rec['raw_H40'] = rec['raw_H40'].astype(np.float64)
    elif failure == 'old_phase': rec['native']['phase'] = 'collect-target-fusion-trajectory-qualification-v1'
    elif failure == 'wrong_action_pin': rec['pins']['raw_action_sha256'] = 'f' * 64
    with pytest.raises(ValueError): r.response_window(rec, p)


def test_twelve_run_single_parent_pilot_cannot_fit_or_claim_a72_cohort(cohort):
    p, data, _ = cohort
    pilot = {key: value[:12] for key, value in data.items()}
    with pytest.raises(ValueError, match='COMPLETE_72_RUN'): r.validate_cohort(pilot, p)
    with pytest.raises(ValueError): m.TrajectoryResponseStudy(p).fit(pilot, min_lopo_gain_mm=.001, threshold_source={})


def test_candidate_level_must_match_entire_corrected_H40_even_at_unexecuted_H39(cohort):
    p, data, _ = cohort
    poisoned = dict(data, corrected_H40=data['corrected_H40'].copy())
    poisoned['corrected_H40'][0, 39, 2] += .001
    with pytest.raises(ValueError, match='H40_ASSOCIATION'): r.validate_cohort(poisoned, p)
    poisoned = dict(data, correction_basis_H40=data['correction_basis_H40'].copy())
    poisoned['correction_basis_H40'][0, 12] += .1
    with pytest.raises(ValueError): r.validate_cohort(poisoned, p)


def test_causal_features_exclude_future_observations_ids_context_and_global_translation(cohort):
    _, data, _ = cohort
    changed = {key: value.copy() for key, value in data.items()}
    changed['future_TCP'][:] = np.nan
    changed['desired_path'][:] = 1000.
    for key in ('context_id', 'parent', 'run_id'):
        changed[key] = np.full(72, 'not-a-feature')
    shift = np.array([10., -20., 30.])
    changed['history_state'][:, :, 86:89] += shift
    changed['past_reference'][:, :, :3] += shift
    changed['nominal_H40'][:, :, :3] += shift
    for model in m.MODELS:
        np.testing.assert_allclose(m.causal_features(data, model=model), m.causal_features(changed, model=model), atol=1e-8)


def test_future_correction_mask_removes_theta_and_corrected_path_but_keeps_nominal(cohort):
    _, data, _ = cohort
    masked = m.mask_future_correction(data)
    np.testing.assert_array_equal(masked['corrected_H40'], data['nominal_H40'])
    np.testing.assert_array_equal(masked['corrected_path'], data['nominal_path'])
    assert np.all(masked['theta_level'] == 0)
    np.testing.assert_array_equal(masked['history_state'], data['history_state'])
    np.testing.assert_array_equal(masked['nominal_H40'], data['nominal_H40'])
    poisoned = dict(data, theta_level=np.full(72, 999), corrected_H40=np.full_like(data['corrected_H40'], np.nan))
    np.testing.assert_array_equal(m.causal_features(data, model='CORRECTION_BLIND'), m.causal_features(poisoned, model='CORRECTION_BLIND'))


@pytest.fixture(scope='module')
def fitted(cohort):
    p, data, _ = cohort
    threshold = m.train_noise_threshold(data, p, fraction=.25, floor_mm=.000001)
    model = m.TrajectoryResponseStudy(p).fit(data, min_lopo_gain_mm=threshold['min_lopo_gain_mm'], threshold_source=threshold)
    return model, threshold


def test_train_threshold_is_new_train_scatter_and_ignores_all_DEV_labels(cohort, fitted):
    p, data, _ = cohort
    _, threshold = fitted
    changed = dict(data, future_TCP=data['future_TCP'].copy())
    changed['future_TCP'][data['split'] == 'DEV'] = np.nan
    assert m.train_noise_threshold(changed, p, fraction=.25, floor_mm=.000001) == threshold
    assert threshold['independent_restart_pairs'] == 24
    assert threshold['noise_RMSE_mm'] > 0
    assert threshold['DEV_labels_read'] is False and threshold['old_XY_threshold_reused'] is False
    wrong = dict(threshold, training_manifest_sha256='f' * 64)
    with pytest.raises(ValueError, match='THRESHOLD_SOURCE'):
        m.TrajectoryResponseStudy(p).fit(data, min_lopo_gain_mm=threshold['min_lopo_gain_mm'], threshold_source=wrong)
    with pytest.raises(ValueError, match='THRESHOLD_SOURCE'):
        m.TrajectoryResponseStudy(p).fit(data, min_lopo_gain_mm=.08124448952310876, threshold_source=threshold)


def test_whole_parent_CV_base_and_normalizers_exclude_held_parent(cohort, fitted):
    p, data, _ = cohort; model, _ = fitted
    for fold, (parent, train, held) in zip(model.report['folds'], r.whole_parent_folds(data, p)):
        assert parent == fold['held_parent'] and parent not in fold['train_parents']
        target = data['future_TCP'][train] - data['history_state'][train, -1:, 86:89]
        curves = np.stack([target[data['theta_level'][train] == level].mean(0) for level in c.THETA_LEVELS])
        np.testing.assert_allclose(fold['static_curves_m'], curves, atol=0, rtol=0)
        for name in m.MODELS:
            expected = m.causal_features(data, train, model=name).mean(0)
            np.testing.assert_array_equal(fold['normalizers'][name]['mean'], expected)
    assert model.report['closed_form_ridge_solves'] == 20
    assert model.report['same_history_additive_comparator'] == 'ADDITIVE_LAG_VELOCITY'


def test_exact_save_reload_before_DEV_and_blind_prediction_invariance(cohort, fitted, tmp_path):
    p, data, _ = cohort; model, _ = fitted
    dev = np.flatnonzero(data['split'] == 'DEV')
    train = np.flatnonzero(data['split'] == 'TRAIN')
    with pytest.raises(ValueError, match='SAVE_RELOAD'): model.predict(data, dev)
    no_roles = {key: value for key, value in data.items() if key != 'split'}
    with pytest.raises(ValueError, match='EXPLICIT_OFFLINE'): model.predict(no_roles, dev)
    path = tmp_path / 'synthetic-only-model.npz'; model.save(path)
    with pytest.raises(FileExistsError): model.save(path)
    restored = m.TrajectoryResponseStudy.load(path)
    with pytest.raises(ValueError): restored.predict(data, dev)
    with pytest.raises(ValueError): model.verify_reload(model, data)
    with pytest.raises(ValueError): model.verify_reload(m.TrajectoryResponseStudy.from_arrays(model.arrays()), data)
    assert model.verify_reload(restored, data)['TRAIN_candidate_max_error_m'] == 0.
    assert restored.predict(data, dev, model='CONDITIONAL').shape == (24, 16, 3)
    candidates = restored.candidate_forecasts(data, train, model='CORRECTION_BLIND')
    np.testing.assert_array_equal(candidates[:, 0], candidates[:, 1])
    np.testing.assert_array_equal(candidates[:, 1], candidates[:, 2])
    poisoned = dict(data, theta_level=np.full(72, 999), corrected_H40=np.full_like(data['corrected_H40'], np.nan))
    np.testing.assert_array_equal(restored.predict(data, dev, model='CORRECTION_BLIND'),
                                  restored.predict(poisoned, dev, model='CORRECTION_BLIND', candidate_theta=999))
    damaged = restored.arrays(); damaged['CONDITIONAL__scale'][0] = 0.
    with pytest.raises(ValueError): m.TrajectoryResponseStudy.from_arrays(damaged)


def test_refit_invalidates_previous_file_reload_DEV_gate(cohort, fitted, tmp_path):
    p, data, _ = cohort; model, threshold = fitted
    path = tmp_path / 'synthetic-only-refit.npz'; model.save(path)
    restored = m.TrajectoryResponseStudy.load(path)
    model.verify_reload(restored, data)
    dev = np.flatnonzero(data['split'] == 'DEV')
    restored.predict(data, dev)
    restored.fit(data, min_lopo_gain_mm=threshold['min_lopo_gain_mm'], threshold_source=threshold)
    with pytest.raises(ValueError, match='SAVE_RELOAD'): restored.predict(data, dev)


def test_synthetic_state_interaction_is_compared_against_same_history_additive_model(cohort, fitted):
    _, data, _ = cohort; model, _ = fitted
    report = model.report['TRAIN_LOPO_RMSE_mm']
    assert report['CONDITIONAL'] < report['ADDITIVE_LAG_VELOCITY']
    assert report['CONDITIONAL'] < report['STATIC']
    assert not model.report['native_gain_established']
    train = np.flatnonzero(data['split'] == 'TRAIN')
    baselines = m.ordinary_baselines(data, train)
    assert set(baselines) == {'CURRENT_LAG_TRANSPORT', 'CONSTANT_VELOCITY', 'HOLD_CURRENT'}
    for prediction in baselines.values():
        metrics = m.forecast_metrics(prediction, data['future_TCP'][train])
        assert set(metrics['horizons']) == {'1', '4', '8', '16'}


def test_import_has_no_neural_physics_or_network_runtime():
    src = str(Path(m.__file__).parents[2])
    code = (f'import sys;sys.path.insert(0,{src!r})\n'
            'from go2arx5_vla.world_model import fusion_trajectory_response\n'
            "if any(x in sys.modules for x in ['torch','gr00t','mujoco','transformers','huggingface_hub']):\n"
            "    raise SystemExit('unexpected runtime import')\n")
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
