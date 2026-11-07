"""Consume a finite post-query suffix on an existing, unsealed native adapter.

This is a loop seam, not a phase route, admission check, query entry or native
qualification. The future wrapper owns preregistration, query/pause identity,
shutdown and the one final trace seal. An explicitly registered shadow forecast
may observe the causal inputs before publication; it cannot change references.
"""
from __future__ import annotations

import copy
import numpy as np

from .fusion_trajectory_adapter import TrajectoryAdaptation
from . import fusion_trajectory_qualification_native as native
from .fusion_reference_qualification_native import require_original_install_ack
from .next2_episode import state_vector
from .storage import jsonable, write_json
from .target_reference import require_install_clock, require_live_window

PHASE = 'collect-target-fusion-task-phase-live-skill-v8'
SCHEMA = 'WM02_FUSION_LIVE_SKILL_CONSUMED_SEGMENT_V8'
ATOM_TICKS, NATIVE_DT_S, TAIL_ATOMS, RENEW_TICKS = 25, .002, 8, 175


def _need(condition, reason):
    if not condition:
        raise ValueError('LIVE_SKILL_SEGMENT_' + reason)


def _failed_evidence(value):
    """Retain invalid numeric feedback without emitting nonstandard JSON NaN."""
    value = jsonable(value)
    if isinstance(value, dict):
        return {k: _failed_evidence(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_failed_evidence(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {'invalid_nonfinite_observed': repr(value)}
    return value


def _sample(sample, step):
    vector = np.asarray(state_vector(sample), dtype=float)
    _need(vector.shape == (106,) and np.isfinite(vector).all()
          and sample['native_step'] == step
          and abs(sample['time_s'] - step * NATIVE_DT_S) <= 1e-8
          and sample['safety_valid'] is True and sample['native_abort'] is False,
          'FINITE_ACTUAL_SAFE_106_STATE_AT_EXACT_CLOCK')
    return vector


def _history(samples, windows):
    """Only samples and confirmed references preceding the next installation."""
    last = samples[-9:]
    steps = [s['native_step'] for s in last]
    _need(len(last) == 9 and steps == list(range(steps[-1]-200, steps[-1]+1, 25)),
          'NINE_CONTIGUOUS_ACTUAL_SAMPLES')
    state = np.asarray([_sample(s, step) for s, step in zip(last, steps)])
    previous = windows[-8:]
    _need(len(previous) == 8, 'EIGHT_ACTUALLY_CONSUMED_PAST_REFERENCES')
    for index, row in enumerate(previous):
        reference = np.asarray(row['actual_installed_consumed_EEF10_BASE7'], float)
        grip = row['gripper']
        submitted, wrapper = grip['submitted'], grip['consumed_wrapper']
        values = wrapper['values']
        _need(row['native_start_step'] == steps[index]
              and row['native_end_step'] == steps[index+1] and row['actual_ticks'] == 25
              and bool(row['solver_policies_with_exact_installed_target'])
              and row['gripper']['command_consumed'] is True
              and reference.shape == (17,) and np.isfinite(reference).all()
              and np.array_equal(np.asarray(row['actual_TCP']), last[index+1]['tcp']),
              'PAST_REFERENCE_SOLVER_GRIPPER_AND_MEASURED_ENDPOINT_JOIN')
        _need(wrapper['status'] == 'ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY'
              and values['initialized'] is True and values['has_command'] is True
              and values['steps'] == steps[index+1]
              and values['sequence'] == submitted['sequence']
              and submitted['native_start_step'] == steps[index]
              and values['opening'] == submitted['opening'] == reference[9]
              and values['expiry'] == submitted['expiry'] == submitted['source_mpc_s']+.4
              and last[index+1]['time_s'] <= values['expiry']+1e-9
              and np.array_equal(np.asarray(grip['actual_finger_q_m']), last[index+1]['finger_q']),
              'PAST_ACTUAL_WRAPPER_SEQUENCE_CLOCK_OPENING_AND_FINGER_JOIN')
    return dict(actual_state_history=state.tolist(), actual_state_history_shape=[9, 106],
        actual_history_native_steps=steps,
        past_consumed_EEF10_BASE7=[copy.deepcopy(w['actual_installed_consumed_EEF10_BASE7'])
                                  for w in previous],
        past_reference_native_end_steps=[w['native_end_step'] for w in previous],
        history_provenance='ACTUAL_SAMPLES_AND_EXACT_INSTALLED_SOLVER_JOIN_PER_ATOM',
        full_MPC_hidden_state_restored=False)


def _preflight(adapter, program, start_step, end_step, samples, windows):
    spec = adapter.spec
    _need(spec.get('phase') in (PHASE,'collect-target-fusion-task-phase-live-skill-v9'),
          'EXACT_FUTURE_V8_OR_V9_TYPED_PHASE_REQUIRED')
    _need(type(program) is TrajectoryAdaptation, 'EXACT_TRAJECTORY_ADAPTATION_NO_WM_OR_DUCK_TYPE')
    _need(type(start_step) is int and type(end_step) is int
          and 2503 <= start_step < end_step
          and (start_step-2503) % 25 == (end_step-2503) % 25 == 0
          and spec.get('live_skill_start_step') == start_step
          and spec.get('max_ticks') == end_step,
          'DISTINCT_PREREGISTERED_B_E_ON_ORIGINAL_25_TICK_GRID')
    _need(spec.get('physics_dt_s') == NATIVE_DT_S and spec.get('WBC_changes') is False,
          'UNCHANGED_ORIGINAL_NATIVE_DT_AND_WBC')
    _need(getattr(adapter, 'trace_sealed', None) is False, 'UNSEALED_EXISTING_WORLD_REQUIRED')
    _need(not (adapter.output/'trace_seal_request').exists()
          and not (adapter.output/'init.trace.summary').exists(),
          'NATIVE_TRACE_SEAL_OR_TERMINAL_SUMMARY_CANNOT_BE_UNDONE')
    _need(adapter.latest['mode'].data == 'WBC', 'CALLER_MUST_RESTORE_ORIGINAL_WBC_AFTER_QUERY_CANCEL')
    atoms = (end_step-start_step)//25
    _need(atoms >= program.end_atom+TAIL_ATOMS,
          'FULL_40_KNOTS_AND_EIGHT_HOLD_ATOMS_MUST_FIT_FIXED_SUFFIX_NO_CLIP')
    _need(np.array_equal(program.base_pose_hold, np.asarray(adapter.base_pose_hold)),
          'SAME_FIXED_MEASURED_BASE_HOLD')
    _need(len(samples) >= 9 and len(windows) >= 8
          and len(windows) == (start_step-2503)//25,
          'EXISTING_CONTIGUOUS_GLOBAL_WINDOWS_REQUIRED')
    for index, row in enumerate(windows):
        _need(row['native_start_step'] == 2503+25*index
              and row['native_end_step'] == 2503+25*(index+1)
              and row['actual_ticks'] == 25
              and row['gripper']['submitted']['sequence'] == index
              and row['gripper']['command_consumed'] is True,
              'PAST_WINDOWS_AND_GLOBAL_GRIPPER_SEQUENCE_MUST_CONTINUE_TO_B')
    _need(samples[-1]['native_step'] == start_step, 'ACTUAL_HISTORY_MUST_END_AT_B')
    history = _history(samples, windows)
    packet = adapter.observe()
    _need(packet['step'] == start_step and abs(packet['time_s']-start_step*NATIVE_DT_S) <= 1e-8,
          'SAME_CURRENT_PAUSED_WORLD_AT_B')
    actual = adapter.measured_task_state()
    _need(np.array_equal(_sample(actual, start_step), _sample(samples[-1], start_step)),
          'HISTORY_ENDPOINT_MUST_EQUAL_CURRENT_ACTUAL_STATE')
    _need(np.array_equal(program.decision_TCP_target, actual['tcp']),
          'PROGRAM_DESIRED_ANCHOR_MUST_BE_ACTUAL_QUERY_BOUNDARY_TCP')
    _need(isinstance(adapter.reference_history, list)
          and len(adapter.reference_history) == len(windows),
          'ORIGINAL_REFERENCE_HISTORY_MUST_NOT_BE_RESET')
    from scripts.pi0292.eval_contract import matrix10
    for saved, row in zip(adapter.reference_history[-8:], windows[-8:]):
        _need(saved['end_native_step'] == row['native_end_step']
              and np.allclose(saved['pose'], matrix10(np.asarray(
                  row['actual_installed_consumed_EEF10_BASE7'])[:10]), rtol=0, atol=1e-12),
              'ORIGINAL_REFERENCE_HISTORY_MUST_JOIN_CONSUMED_PAST')
    return atoms, history


def run_consumed_segment(adapter, program, *, start_step, end_step,
                         past_samples, past_windows, sample_observer=None):
    """Return ``(suffix_report, all_samples, all_windows)`` without finalizing.

    B is ``spec.live_skill_start_step`` and E is ``spec.max_ticks``, fixed
    before querying. Excess budget is only the original preview's final HOLD.
    Input lists are retained; returned lists append actual suffix observations.
    Failures are saved once and re-raised, with no retry or lifecycle action.
    """
    from scripts.pi0292.eval_contract import matrix10

    atoms, history = _preflight(adapter, program, start_step, end_step,
                                past_samples, past_windows)
    _need(sample_observer is None or callable(sample_observer), 'CALLABLE_MEASUREMENT_OBSERVER')
    if sample_observer is not None:
        sample_observer(past_samples[-1])
    decision_path = adapter.output/'LIVE_SKILL_SEGMENT_DECISION.private.json'
    result_path = adapter.output/'LIVE_SKILL_SEGMENT.private.json'
    _need(not decision_path.exists() and not result_path.exists(),
          'EXISTING_ATTEMPT_MUST_NOT_BE_REPLAYED')
    samples, windows = list(past_samples), list(past_windows)
    suffix, acks, installs = [], [], []
    report = dict(schema=SCHEMA, phase=adapter.spec['phase'], status='STARTED_NOT_NATIVE_QUALIFIED',
        start_step=start_step, native_endpoint_step=end_step, finite_suffix_atoms=atoms,
        movement_interval_atoms=program.end_atom, minimum_tail_hold_atoms=TAIL_ATOMS,
        extra_final_hold_atoms=atoms-program.end_atom-TAIL_ATOMS,
        adaptation=program.evidence(), original_install_ACKs=acks,
        install_attempts=installs, windows=suffix,
        original_raw_H40=program.decoded_H40.tolist(),
        original_raw_H40_label='DECODED_VLA_INTENT_NOT_ACHIEVED_OR_EXPERT_ACTION_LABEL',
        raw_knot_labels='ALL_40_ADAPTED_REFERENCE_KNOTS_REQUIRE_ACTUAL_CONSUMPTION',
        full_MPC_hidden_state_restored=False,
        controller_selected_reference='DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED',
        source_relative_ttl_s=.4, replan_cadence_s=.35, command_clips=0,
        Processor_decodes=0, VLA_inferences=0, WM_updates=0, Actor_updates=0,
        GR00T_updates=0, optimizer_updates=0, learned_models_loaded=0,
        WBC_changes=False, physics_dt_s=NATIVE_DT_S,
        final_trace_seal_owned_by_caller=True, target_task_success=False,
        native_stage_RESULT_json_required=True, automatic_retries=0)
    write_json(decision_path, dict(schema=SCHEMA+'_DECISION', start_step=start_step,
        end_step=end_step, adaptation=program.evidence(), raw_H40=program.decoded_H40,
        desired_path=program.desired_path, nominal_reference_path=program.reference_path,
        actual_history=history, first_gripper_sequence=len(windows),
        future_response_observed=False, Processor_decodes=0))
    attempt = None
    forecaster = None

    def install(elapsed):
        if forecaster is not None:
            forecaster.save(_history(samples,windows),windows[-8:],program.reference_path,
                            install_atom=elapsed)
        action = program.preview(elapsed)
        installation = dict(elapsed_atom=elapsed, native_step=start_step+25*elapsed,
            full_proposed_H40=action.tolist(), stage='BEFORE_ORIGINAL_INSTALL')
        installs.append(installation)
        reference = adapter.install_reference(action)
        installation.update(stage='RETURNED_ORIGINAL_INSTALL',
            wire_sha256=reference['wire_sha256'], source_mpc_s=reference['source_mpc_s'],
            receiver_events=copy.deepcopy(reference['receiver_events']))
        ack = require_original_install_ack(reference)
        packet = adapter.observe()
        require_install_clock(reference['source_mpc_s'], packet['time_s'])
        _need(packet['step'] == start_step+25*elapsed
              and adapter.active_reference['wire_sha256'] == reference['wire_sha256']
              and np.array_equal(np.asarray(reference['action_EEF10']), action)
              and np.array_equal(np.asarray(adapter.active_reference['action_EEF10']), action),
              'FRESH_FULL_H40_INSTALL_ACK_WITHOUT_INTEGRATION_REQUIRED')
        acks.append(dict(ack, native_step=packet['step'], elapsed_atom=elapsed))
        installation['stage'] = 'ORIGINAL_ACK_AND_FULL_SUFFIX_CONFIRMED'
        return packet['step']

    try:
        if adapter.spec.get('program_response_forecast') is not None:
            from .fusion_program_forecast_v1 import ShadowForecaster
            forecaster=ShadowForecaster(adapter.spec,adapter.output)
            report.update(learned_models_loaded=len(forecaster.models),
                shadow_forecast_records=forecaster.records,model_reference_changes=0)
        # A previous cancel/query leaves no usable live reference: always install.
        installed_step = install(0)
        for elapsed in range(atoms):
            before = adapter.observe(); step = start_step+25*elapsed
            _need(before['step'] == step and abs(before['time_s']-step*NATIVE_DT_S) <= 1e-8
                  and not adapter.trace_sealed,
                  'EXACT_UNSEALED_CONTIGUOUS_ATOM_CLOCK')
            history = _history(samples, windows)
            attempt = dict(trajectory_elapsed_atom=elapsed, native_start_step=step,
                actual_history_before_atom=history,
                nominal_submitted_EEF10=program.preview(elapsed)[0].tolist(),
                desired_EEF10=program.desired_at(elapsed).tolist(),
                desired_interval_end_EEF10=program.desired_at(elapsed+1).tolist(),
                stage='BEFORE_INSTALL_OR_EFFECT')
            if step-installed_step >= RENEW_TICKS:
                installed_step = install(elapsed)
            clock = require_live_window(adapter.active_reference['source_mpc_s'], before['time_s'], 25)
            index = (step-installed_step)//25
            installed_row = np.asarray(adapter.active_reference['action_EEF10'])[index]
            _need(np.array_equal(installed_row, program.preview(elapsed)[0]),
                  'INSTALLED_H40_INDEX_MUST_EQUAL_CURRENT_FULL_PROGRAM_SUFFIX')
            attempt.update(actual_installed_consumed_EEF10_BASE7=
                np.r_[installed_row, adapter.base_pose_hold].tolist(),
                full_final_installed_H40=np.asarray(adapter.active_reference['action_EEF10']).tolist(),
                original_reference_sha256=adapter.active_reference['wire_sha256'], clock=clock)
            submitted = native.submit_gripper(adapter, len(windows), installed_row[9])
            attempt.update(gripper_submitted=submitted, stage='SUBMITTED_BEFORE_EFFECT')
            _need(submitted['native_start_step'] == step and submitted['sequence'] == len(windows),
                  'ONE_CONTIGUOUS_GRIPPER_WRITE_AT_THIS_ACTUAL_ATOM')
            adapter.command_event(dict(phase='live-skill-before-atom', **attempt))
            event_start = len(adapter.ros_evidence)
            after = adapter.advance_to(step+25)
            adapter.spin(.15)
            actual = adapter.measured_task_state()
            samples.append(actual)
            attempt.update(actual_TCP=np.asarray(actual['tcp']).tolist(),
                actual_state_vector=np.asarray(state_vector(actual)).tolist(),
                actual_native_step=actual['native_step'],
                actual_packet=dict(step=after['step'], time_s=after['time_s']),
                stage='ACTUAL_EFFECT_OBSERVED_NOT_YET_CONFIRMED')
            _sample(actual, step+25)
            if sample_observer is not None:
                sample_observer(actual)
            _need(after['step'] == step+25 and abs(after['time_s']-actual['time_s']) <= 1e-8,
                  'EXACT_ORIGINAL_NATIVE_PAUSE_AFTER_ATOM')
            _need(adapter.latest['mode'].data == 'WBC'
                  and not any(r.get('reason') == 'PREFIX_TTL_EXPIRED'
                              for r in adapter.reference_events()), 'ORIGINAL_WBC_AND_TTL')
            grip = native.confirm_gripper(adapter, submitted, actual)
            attempt['gripper'] = grip
            consumed = native.consumed_window(adapter, event_start, before, after, clock)
            attempt.update(consumed)
            _need(consumed['native_start_step'] == step
                  and consumed['native_end_step'] == step+25 and consumed['actual_ticks'] == 25
                  and consumed['reference_sha256'] == adapter.active_reference['wire_sha256']
                  and bool(consumed['solver_policies_with_exact_installed_target']),
                  'EXACT_INSTALLED_SENDER_TO_SOLVER_CONSUMPTION_REQUIRED')
            attempt['stage'] = 'ACTUAL_CONSUMPTION_CONFIRMED'
            windows.append(attempt); suffix.append(attempt)
            adapter.reference_history.append(dict(end_native_step=step+25,
                pose=matrix10(adapter.active_reference['action_EEF10'])[index]))
            adapter.command_event(dict(phase='live-skill-controller-consumed', **attempt))
            attempt = None
        consumed_atoms = {row['trajectory_elapsed_atom'] for row in suffix}
        knots = [index in consumed_atoms for index in program.knot_indices]
        _need(len(suffix) == atoms and len(knots) == 40 and all(knots),
              'EVERY_ADAPTED_RAW_KNOT_AND_FIXED_SUFFIX_MUST_ACTUALLY_CONSUME')
        hold=(program.reference_intervention is not None
              and program.reference_intervention.candidate=='HOLD')
        report.update(status='SUFFIX_CONSUMED_REQUIRES_OUTER_TRACE_AND_SHUTDOWN_AUDIT',
            complete_suffix_atoms=len(suffix), gripper_confirmed_atoms=len(suffix),
            solver_target_confirmed_atoms=len(suffix), all_40_knots_consumed=not hold,
            raw_knot_consumed=[False]*40 if hold else knots,final_gripper_sequence=len(windows)-1,
            actual_endpoint_TCP=np.asarray(samples[-1]['tcp']).tolist(),
            trace_sealed_by_segment=False)
        if program.reference_intervention is not None:
            report.update(all_declared_reference_knot_times_consumed=True,post_query_HOLD_control=hold,
                reference_intervention=program.reference_intervention.document())
        if forecaster is not None:report['shadow_forecast_wall_s']=forecaster.wall_s
        write_json(result_path, report)
        return report, samples, windows
    except Exception as error:
        report.update(status='FAILED_RETAINED_NO_RETRY', failure_type=type(error).__name__,
            failure=str(error), failure_attempt=jsonable(attempt),
            complete_suffix_atoms=len(suffix),
            last_observed_native_step=samples[-1]['native_step'], trace_sealed_by_segment=False)
        write_json(result_path, _failed_evidence(report))
        raise
