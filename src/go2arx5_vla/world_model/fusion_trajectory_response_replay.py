"""Causal one-probe/run replay for the NEW 72-episode trajectory experiment.

This module consumes already authenticated private receipts. File/hash/lease
authentication stays with the caller. It cannot turn a fixture or pilot into
physical evidence, and never fits or invokes a model.
"""
from collections import Counter

import numpy as np

from .fusion_trajectory_adapter import _identity
from .fusion_trajectory_response_contract import (
    ATOM_DT_S, CONTEXT_IDS, DECISION_STEP, DIRECTION, EPSILON_M, PHASE,
    REPEAT_INDICES, THETA_LEVELS, canonical_sha256, need, pulse_basis,
    validate_episode, validate_panel, _sha,
)

SCHEMA = 'WM02_TRAJECTORY_RESPONSE_REPLAY_V1'
STATE_LAYOUT = dict(qpos=[0, 34], qvel=[34, 66], ctrl=[66, 86],
                    tcp_xyz=[86, 89], tcp_rot6d_COLUMNS=[89, 95],
                    object_xyz=[95, 98], object_rot6d_COLUMNS=[98, 104], finger_q=[104, 106])


def ttl_atoms(remaining_s):
    """Continuous atoms and whole live atoms under the ORIGINAL strict expiry."""
    need(np.isfinite(remaining_s) and 0 < remaining_s <= .4 + 1e-9, 'MEASURED_TTL_REMAINING_SECONDS')
    continuous = float(remaining_s / ATOM_DT_S)
    complete = int(np.floor(continuous))
    if complete * ATOM_DT_S >= remaining_s - 1e-10:
        complete -= 1
    return continuous, max(0, complete)


def _array(value, shape, label):
    value = np.asarray(value, dtype=np.float64)
    need(value.shape == shape and np.isfinite(value).all(), 'FINITE_' + label)
    return value


def _suffix(path, atom):
    return path[np.minimum(atom + np.arange(40), len(path) - 1)]


def response_window(record, panel):
    """Extract only the predeclared probe; every future observation is a label.

    ``record`` has episode/native/report/decision dictionaries plus hashes
    authenticated by the caller. ``report`` preserves the complete collector
    windows and full paths, not just this selected learning window.
    """
    episode = validate_episode(record['episode'], panel)
    pins = record['pins']
    need(isinstance(pins, dict) and set(pins) == {'native_result_sha256', 'report_sha256',
        'episode_sha256', 'decision_sha256', 'raw_action_sha256'}
        and all(_sha(pin) for pin in pins.values()), 'EXPLICIT_AUTHENTICATED_SOURCE_ARTIFACT_PINS')
    need(pins['raw_action_sha256'] == panel['query_pins']['action'], 'SAME_PINNED_ORIGINAL_QUERY_ACTION')
    native, report, decision = record['native'], record['report'], record['decision']
    assignment = decision['assignment_before_install']
    need(report.get('assignment_before_install') == assignment
         and report.get('decision_record', {}).get('sha256') == pins['decision_sha256'],
         'SAME_SEALED_DECISION_AND_ASSIGNMENT')
    need(native.get('phase') == PHASE
         and native.get('status') == 'NATIVE_STAGE_PASSED' and native.get('shutdown_clean') is True
         and native.get('physics_stops') == []
         and native.get('trace_audit', {}).get('status') == 'RECORDED_TRACE_CHECKS_PASS'
         and native.get('trace_audit', {}).get('errors') == []
         and native.get('trace_audit', {}).get('missing') == [],
         'OUTER_COMPLETE_NATIVE_PASS_TRACE_AND_SHUTDOWN_REQUIRED')
    need(report.get('schema') == 'WM02_NATIVE_TRAJECTORY_RESPONSE_COLLECTION_V1'
         and decision.get('schema') == 'WM02_TRAJECTORY_RESPONSE_DECISION_V1'
         and report.get('episode') == episode and report.get('panel_sha256') == canonical_sha256(panel)
         and report.get('all_40_knots_consumed') is True,
         'SAME_FROZEN_PANEL_EPISODE_AND_COMPLETE_40_KNOTS')
    need(assignment.get('native_step') == DECISION_STEP
         and assignment.get('theta_level') == episode['theta_level']
         and assignment.get('future_response_observed') is False,
         'THETA_ASSIGNED_AT_REAL_DECISION_BEFORE_FUTURE_RESPONSE')
    n = episode['planned_end_atom'] + 1
    nominal = _array(decision['full_nominal_reference_path'], (n, 10), 'FULL_NOMINAL_PATH')
    corrected = _array(decision['full_corrected_reference_path'], (n, 10), 'FULL_CORRECTED_PATH')
    desired = _array(decision['desired_path'], (n, 10), 'FULL_DESIRED_PATH')
    raw = np.asarray(record['raw_H40'])
    need(raw.shape == (40, 10) and raw.dtype.kind == 'f' and np.isfinite(raw).all(),
         'ORIGINAL_FLOAT_RAW_H40')
    need(raw.dtype.str == decision['raw_H40_dtype']
         and _identity(raw)['sha256'] == decision['raw_H40_bytes_sha256']
         and np.array_equal(raw, np.asarray(decision['raw_H40'])),
         'ORIGINAL_RAW_H40_BYTES_DTYPE_AND_DECISION_READBACK')
    basis = pulse_basis(np.arange(n))
    expected = nominal.copy()
    expected[:, :3] += episode['theta_level'] * EPSILON_M * basis[:, None] * DIRECTION
    need(np.allclose(corrected, expected, rtol=0, atol=1e-12),
         'ONLY_DECLARED_PULSE_NO_THETA_SPECIFIC_RETIME_ROTATION_OR_GRIP_CHANGE')
    windows = report['windows']
    need(len(windows) == report.get('complete_task_atoms')
         == report.get('solver_target_confirmed_atoms') == report.get('gripper_confirmed_atoms'),
         'ALL_RECORDED_ATOMS_HAVE_POSE_AND_GRIPPER_CONSUMPTION')
    selected = [w for w in windows if w['native_start_step'] >= DECISION_STEP]
    need(len(selected) == episode['planned_end_atom'] + 8,
         'COMPLETE_COMMON_ENDPOINT_AND_TAIL')
    all_installed = []
    for index, row in enumerate(windows):
        need(row['native_end_step'] - row['native_start_step'] == row.get('actual_ticks') == 25
             and row['native_start_step'] == 2503 + 25 * index
             and bool(row.get('solver_policies_with_exact_installed_target'))
             and row.get('gripper', {}).get('command_consumed') is True,
             'EXACT_ATOMIC_SEQUENCE_AND_ACTUAL_CONSUMPTION')
        all_installed.append(_array(row['full_final_installed_H40'], (40, 10), 'ACTUAL_INSTALLED_H40'))
    need(native.get('actual_target_integrations') == windows[-1]['native_end_step'],
         'OUTER_ENDPOINT_EQUALS_FULL_RECORDED_PROGRAM')
    for atom, row in enumerate(selected):
        need(row['native_start_step'] == DECISION_STEP + 25 * atom
             and row.get('probe_elapsed_atom') == atom, 'CONTIGUOUS_POST_DECISION_ATOMS')
        consumed = _array(row['actual_installed_consumed_EEF10_BASE7'], (17,), 'CONSUMED_EEF10_BASE7')
        need(np.allclose(consumed[:10], corrected[min(atom, n - 1)], atol=1e-12, rtol=0),
             'ACTUAL_CONSUMED_PROGRAM_EQUALS_DECLARED_CORRECTION')
        installed_origin = 7 * (atom // 7)
        need(np.allclose(row['full_final_installed_H40'], _suffix(corrected, installed_origin),
                         atol=1e-12, rtol=0), 'EVERY_RENEWAL_ENTIRE_INSTALLED_H40_MATCHES_SEALED_PROGRAM')
    first = selected[0]
    history = first['actual_history_before_atom']
    state = _array(history['actual_state_history'], (9, 106), 'OBSERVED_HISTORY_9X106')
    past = _array(history['past_consumed_EEF10_BASE7'], (8, 17), 'EIGHT_PAST_CONSUMED_REFERENCES')
    steps = np.asarray(history['actual_history_native_steps'])
    need(np.array_equal(steps, DECISION_STEP + 25 * np.arange(-8, 1))
         and np.array_equal(history['past_reference_native_end_steps'],
                            DECISION_STEP + 25 * np.arange(-7, 1))
         and history.get('full_MPC_hidden_state_restored') is False,
         'ACTUAL_PREFIX_CLOCK_NOT_EXACT_HIDDEN_STATE_FORK')
    prefix_rows = [row for row in windows if DECISION_STEP - 200 <= row['native_end_step'] <= DECISION_STEP]
    past_rows = [row for row in windows if DECISION_STEP - 200 < row['native_end_step'] <= DECISION_STEP]
    need(len(prefix_rows) == 9 and len(past_rows) == 8
         and np.array_equal(state, np.asarray([row['actual_state_vector'] for row in prefix_rows]))
         and np.array_equal(past, np.asarray([row['actual_installed_consumed_EEF10_BASE7'] for row in past_rows])),
         'PREFIX_AND_PAST_REFERENCES_JOIN_ACTUAL_PREDECISION_RECORDS')
    truth = np.stack([_array(row['actual_TCP'], (4, 4), 'ACTUAL_FUTURE_TCP')[:3, 3]
                      for row in selected[:16]])
    clock = first['clock']
    age = float(clock['native_start_s'] - clock['source_mpc_s'])
    remaining = float(clock['remaining_at_install_s'])
    need(-1e-8 <= age <= .025 + 1e-9
         and np.isclose(remaining, .4 - age, atol=1e-9, rtol=0),
         'OBSERVED_PROBE_INSTALL_CLOCK_SECONDS')
    continuous, live = ttl_atoms(remaining)
    # The new collector reinstalls at the decision, so the first issued row
    # and all 40 preview points have the canonical probe origin.
    need(np.allclose(first['full_final_installed_H40'], _suffix(corrected, 0), rtol=0, atol=1e-12),
         'DECISION_INSTALLED_ENTIRE_H40_MATCHES_SEALED_CORRECTED_PROGRAM')
    return dict(history_state=state, past_reference=past, future_TCP=truth,
        raw_H40=raw.copy(), nominal_path=nominal, corrected_path=corrected, desired_path=desired,
        nominal_H40=_suffix(nominal, 0), corrected_H40=_suffix(corrected, 0),
        correction_basis_H40=basis[np.minimum(np.arange(40), n - 1)],
        all_installed_H40=np.stack(all_installed),
        all_installed_native_start_steps=np.array([w['native_start_step'] for w in windows]),
        source_age_s=age, ttl_remaining_s=remaining,
        ttl_remaining_continuous_atoms=continuous, complete_live_atoms=live,
        run_id=episode['run'], parent=episode['parent'], split=episode['split'],
        context_id=episode['context_id'], theta_level=episode['theta_level'],
        restart_index=episode['restart_index'], probe_native_step=DECISION_STEP,
        raw_H40_sha256=_identity(raw)['sha256'], source_record_sha256=canonical_sha256(pins))


def build_replay(records, panel, *, require_complete=True):
    validate_panel(panel)
    need(bool(records), 'NONEMPTY_NEW_EPISODE_RECORDS')
    rows = [response_window(record, panel) for record in records]
    need(len({row['run_id'] for row in rows}) == len(rows), 'UNIQUE_PHYSICAL_RUN_IDENTITIES')
    try:
        data = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    except ValueError as error:
        raise ValueError('TRAJECTORY_RESPONSE_COMMON_PANEL_CLOCK_AND_ARRAY_LAYOUT') from error
    metadata = dict(schema=SCHEMA, panel_sha256=canonical_sha256(panel),
                    state_layout=STATE_LAYOUT, EEF10_rotation='ROWS', state_rotation='COLUMNS',
                    observed_runs=len(rows), one_predeclared_probe_per_run=True,
                    overlapping_windows_are_independent_samples=False,
                    forecast_conditioning='PREDECLARED_PROGRAM_CONDITIONAL',
                    full_actual_installed_H40_retained=True,
                    raw_array_hashes=data['raw_H40_sha256'].tolist(),
                    source_records_sha256=[canonical_sha256(record.get('pins', {})) for record in records],
                    TEST_read=False, exact_hidden_state_forks=False,
                    native_admission_or_hash_authentication_granted=False)
    if require_complete:
        validate_cohort(data, panel)
    return data, metadata


def validate_cohort(data, panel):
    """A pilot cannot fit; check membership/shapes without reading DEV labels."""
    validate_panel(panel)
    n = len(data['parent'])
    need(n == 72 and len(set(map(str, data['run_id']))) == 72, 'COMPLETE_72_RUN_PANEL_REQUIRED_FOR_FIT')
    shapes = dict(history_state=(n, 9, 106), past_reference=(n, 8, 17), future_TCP=(n, 16, 3),
                  nominal_H40=(n, 40, 10), corrected_H40=(n, 40, 10), correction_basis_H40=(n, 40))
    need(all(np.asarray(data[key]).shape == shape for key, shape in shapes.items()), 'EXACT_NEW_REPLAY_LAYOUT')
    parents = np.asarray(data['parent']); contexts = np.asarray(data['context_id'])
    theta = np.asarray(data['theta_level']); restarts = np.asarray(data['restart_index'])
    need(theta.dtype.kind in 'iu' and restarts.dtype.kind in 'iu', 'INTEGER_LEVEL_RESTART_IDENTITIES')
    nominal = np.asarray(data['nominal_H40'], float)
    corrected = np.asarray(data['corrected_H40'], float)
    basis = np.asarray(data['correction_basis_H40'], float)
    expected_corrected = nominal.copy()
    expected_corrected[:, :, :3] += theta[:, None, None] * EPSILON_M * basis[:, :, None] * DIRECTION
    need(np.isfinite(nominal).all() and np.isfinite(corrected).all()
         and np.array_equal(basis, np.broadcast_to(pulse_basis(np.arange(40)), basis.shape))
         and np.allclose(corrected, expected_corrected, rtol=0, atol=1e-12),
         'CANDIDATE_LEVEL_BASIS_AND_FULL_H40_ASSOCIATION')
    expected = Counter((c, t, r) for c in CONTEXT_IDS for t in THETA_LEVELS for r in REPEAT_INDICES)
    need(set(map(str, parents)) == set(panel['physical_parents']), 'EXACT_FROZEN_PHYSICAL_PARENTS')
    for parent, split in panel['physical_parents'].items():
        ids = np.flatnonzero(parents == parent)
        need(np.all(np.asarray(data['split'])[ids] == split)
             and Counter(zip(contexts[ids], theta[ids], restarts[ids])) == expected,
             'WHOLE_PARENT_BALANCED_CONTEXT_THETA_RESTART_CELLS')
    return {split: np.flatnonzero(np.asarray(data['split']) == split) for split in ('TRAIN', 'DEV')}


def whole_parent_folds(data, panel):
    train = validate_cohort(data, panel)['TRAIN']
    parents = np.asarray(data['parent'])
    return [(str(parent), train[parents[train] != parent], train[parents[train] == parent])
            for parent in sorted(set(parents[train]))]
