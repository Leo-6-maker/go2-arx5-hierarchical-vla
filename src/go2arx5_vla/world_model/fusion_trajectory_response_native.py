"""Collect one predeclared response probe through the original native owner.

No scheduler, predictor, query or controller replacement. Context and future
pulse are separate programs: theta cannot change the predecision reference.
"""
import hashlib
import numpy as np

from .fusion_reference_qualification_native import require_original_install_ack, consumed_window
from .fusion_trajectory_qualification_native import actual_trace_binding, submit_gripper, confirm_gripper
from .fusion_trajectory_response_native_contract import (
    INIT_STEP, DECISION_STEP, need, episode_config, endpoint, make_program,
    validate_install_preview)
from .fusion_trajectory_response_contract import ContextProgram
from .target_reference import require_live_window


def validate_adapter_install(adapter, action):
    need(hasattr(adapter, 'trajectory_response_context'), 'ACTUAL_CONTEXT_PREPARATION_REQUIRED')
    return validate_install_preview(action, context=adapter.trajectory_response_context,
        program=adapter.trajectory_response_program, step=adapter.observe()['step'],
        base=adapter.base_pose_hold, frozen_base=adapter.trajectory_response_initial_base,
        end_step=adapter.spec['max_ticks'])


def run(adapter):
    from scripts.pi0292.durable import write
    from scripts.pi0292.eval_contract import matrix10
    from .next2_episode import state_vector
    from .target_stability import motion, screen

    episode, panel, inputs = episode_config(adapter.spec)
    end_step = endpoint(episode)
    adapter.prepare_dynamic()
    trace_binding = actual_trace_binding(adapter)
    initial = adapter.measured_task_state()
    need(initial['native_step'] == INIT_STEP, 'ACTUAL_INIT_2503')
    adapter.trajectory_response_initial_base = list(adapter.base_pose_hold)
    context = ContextProgram(initial['tcp'], adapter.base_pose_hold, episode['context_id'])
    adapter.trajectory_response_context = context
    adapter.trajectory_response_program = None
    samples, windows, grippers, acks = [initial], [], [], []

    def history_input():
        state = np.asarray([state_vector(s) for s in samples[-9:]])
        need(np.isfinite(state).all() and all(w['solver_policies_with_exact_installed_target']
             and w['gripper']['command_consumed'] for w in windows[-8:]),
             'FINITE_ACTUAL_HISTORY_AND_CONFIRMED_PAST_COMMANDS')
        return dict(actual_state_history=state.tolist(), actual_state_history_shape=list(state.shape),
            actual_history_native_steps=[s['native_step'] for s in samples[-9:]],
            past_consumed_EEF10_BASE7=[w['actual_installed_consumed_EEF10_BASE7'] for w in windows[-8:]],
            past_reference_native_end_steps=[w['native_end_step'] for w in windows[-8:]],
            input_available_equally_to_all_later_baselines=True,
            history_provenance='ACTUAL_FULL_STATE_AND_ACTUALLY_INSTALLED_SOLVER_JOINED_COMMANDS',
            full_MPC_hidden_state_restored=False)

    def nominal_now():
        step = adapter.observe()['step']
        return (context.preview((step-INIT_STEP)//25) if step < DECISION_STEP else
                adapter.trajectory_response_program.nominal_preview((step-DECISION_STEP)//25))

    def action_now():
        step = adapter.observe()['step']
        return (context.preview((step-INIT_STEP)//25) if step < DECISION_STEP else
                adapter.trajectory_response_program.preview((step-DECISION_STEP)//25))

    def install():
        reference = adapter.install_reference(action_now())
        acks.append(require_original_install_ack(reference))
        return adapter.observe()['step']

    installed_step = install()
    while adapter.observe()['step'] < end_step:
        before = adapter.observe(); step = before['step']
        if step == INIT_STEP+40*25:
            installed_step = install()
        if step == DECISION_STEP:
            last = samples[-9:]
            relative = motion(np.asarray([s['qpos'][:3] for s in last]),
                np.asarray([s['qpos'][3:7] for s in last]),
                np.asarray([s['tcp'][:3, 3] for s in last]))['relative_tcp']
            prefix = screen(np.asarray([s['time_s'] for s in last]), relative)
            prefix.update(schema='WM02_TRAJECTORY_RESPONSE_PREFIX_V1', native_step=step,
                          context_id=episode['context_id'], theta_not_used_in_context=True,
                          thresholds_not_native_safety=True, physics_paused_at_assessment=True)
            write(adapter.output/'TRAJECTORY_RESPONSE_PREFIX.private.json', prefix)
            need(prefix['qualified'], 'UNCHANGED_STEADY_PREFIX_SCREEN_FAILED')
            decision = adapter.measured_task_state()
            program = make_program(inputs, episode, panel, initial['tcp'], decision['tcp'],
                                   adapter.trajectory_response_initial_base)
            adapter.trajectory_response_program = program
            history = history_input()
            need(history['actual_state_history_shape'] == [9, 106]
                 and len(history['past_consumed_EEF10_BASE7']) == 8, 'ACTUAL_9X106_AND_8X17_HISTORY')
            context_endpoint = matrix10(context.preview(48)[0])
            evidence = dict(schema='WM02_TRAJECTORY_RESPONSE_DECISION_V1', episode=episode,
                actual_history=history, context=context.evidence(), program=program.evidence(),
                original_INIT_TCP=initial['tcp'].tolist(), planned_context_endpoint_TCP=context_endpoint.tolist(),
                actual_decision_TCP=decision['tcp'].tolist(),
                actual_context_endpoint_lag_xyz_m=(decision['tcp'][:3, 3]-context_endpoint[:3, 3]).tolist(),
                raw_H40=inputs['raw'].tolist(), full_nominal_reference_path=program.nominal_reference_path.tolist(),
                raw_H40_dtype=inputs['raw'].dtype.str,
                raw_H40_bytes_sha256=hashlib.sha256(inputs['raw'].tobytes()).hexdigest(),
                full_corrected_reference_path=program.reference_path.tolist(),
                desired_path=program.desired_path.tolist(), archived_input_provenance=inputs['provenance'],
                actual_native_step=step, future_response_observed=False, theta_first_installed_at_native_step=step,
                assignment_before_install=dict(native_step=step, theta_level=episode['theta_level'], future_response_observed=False),
                future_theta_available_in_predecision_state_features=False,
                scene_or_context_separation_qualified='NOT_ESTABLISHED_BY_LABELS',
                exact_state_fork=False, online_VLA=False, VLA_queries=0)
            write(adapter.output/'TRAJECTORY_RESPONSE_DECISION.private.json', evidence)
            need(adapter.observe()['step'] == step, 'NO_PHYSICS_DURING_DECISION')
            adapter.command_event(dict(phase='trajectory-response-before-first-theta-install', **evidence))
            installed_step = install()
        elif step-installed_step >= 175:
            installed_step = install()
        corrected, nominal = action_now(), nominal_now()
        expected = corrected[0]
        history = history_input() if step >= DECISION_STEP else None
        clock = require_live_window(adapter.active_reference['source_mpc_s'], before['time_s'], 25)
        index = (step-installed_step)//25
        installed_row = np.asarray(adapter.active_reference['action_EEF10'])[index]
        need(np.allclose(installed_row, expected, rtol=0, atol=1e-12), 'ACTUAL_INSTALLED_CURRENT_CORRECTED_ROW')
        submitted = submit_gripper(adapter, len(windows), expected[9])
        event_start = len(adapter.ros_evidence)
        after = adapter.advance_to(step+25); adapter.spin(.15)
        actual = adapter.measured_task_state(); samples.append(actual)
        need(actual['native_step'] == after['step'] == step+25 and actual['safety_valid'] is True
             and actual['native_abort'] is False, 'ORIGINAL_ACTUAL_NATIVE_FEEDBACK_AND_SAFETY')
        need(adapter.latest['mode'].data == 'WBC'
             and not any(r.get('reason') == 'PREFIX_TTL_EXPIRED' for r in adapter.reference_events()),
             'ORIGINAL_WBC_AND_TTL')
        grip = confirm_gripper(adapter, submitted, actual); grippers.append(grip)
        row = consumed_window(adapter, event_start, before, after, clock)
        elapsed = (step-DECISION_STEP)//25
        row.update(probe_elapsed_atom=elapsed if step >= DECISION_STEP else None,
            actual_history_before_atom=history, nominal_EEF10=nominal[0].tolist(),
            corrected_submitted_EEF10=expected.tolist(),
            full_nominal_H40=nominal.tolist(), full_corrected_H40=corrected.tolist(),
            actual_installed_full_H40=np.asarray(adapter.active_reference['action_EEF10']).tolist(),
            full_final_installed_H40=np.asarray(adapter.active_reference['action_EEF10']).tolist(),
            actual_installed_consumed_EEF10_BASE7=np.r_[installed_row, adapter.base_pose_hold].tolist(),
            actual_installed_source_age_s=before['time_s']-adapter.active_reference['source_mpc_s'],
            future_prefix_remaining_s=clock['remaining_at_install_s'], gripper=grip,
            desired_interval_end_EEF10=(adapter.trajectory_response_program.desired_at(elapsed+1).tolist()
                                       if step >= DECISION_STEP else None),
            actual_TCP=actual['tcp'].tolist(), actual_state_vector=state_vector(actual).tolist(),
            actual_state_native_step=actual['native_step'], raw_query_identity=episode['query_pins'])
        windows.append(row)
        adapter.command_event(dict(phase='controller-consumed' if row['solver_policies_with_exact_installed_target']
                                   else 'consumption_unconfirmed', **row))
        adapter.reference_history.append(dict(end_native_step=after['step'],
            pose=matrix10(adapter.active_reference['action_EEF10'])[index]))
    adapter.capture_loaded_identity()
    cancel = adapter.cancel_reference(); adapter.seal_trace(end_step)
    program = adapter.trajectory_response_program
    need(program is not None, 'ACTUAL_PROBE_PROGRAM_REQUIRED')
    confirmed = sum(bool(w['solver_policies_with_exact_installed_target']) for w in windows)
    consumed = {w['probe_elapsed_atom'] for w in windows if w['probe_elapsed_atom'] is not None}
    retained = [i in consumed for i in program.knot_indices]
    report = dict(schema='WM02_NATIVE_TRAJECTORY_RESPONSE_COLLECTION_V1',
        status='PROGRAM_COMPLETED_REQUIRES_OUTER_NATIVE_TRACE_AND_SHUTDOWN_AUDIT',
        experiment_role='NON_VLA_CALIBRATION', desired_outcome_provenance='CALIBRATION_INTENT',
        episode=episode, program=program.evidence(), context=context.evidence(),
        panel_sha256=episode['panel_sha256'],
        assignment_before_install=dict(native_step=DECISION_STEP, theta_level=episode['theta_level'], future_response_observed=False),
        decision_record=dict(path=str(adapter.output/'TRAJECTORY_RESPONSE_DECISION.private.json'),
                             sha256=hashlib.sha256((adapter.output/'TRAJECTORY_RESPONSE_DECISION.private.json').read_bytes()).hexdigest()),
        actual_trace_binding=trace_binding, original_install_ACKs=acks,
        actual_collection_start_step=INIT_STEP, native_endpoint_step=end_step,
        complete_task_atoms=len(windows), solver_target_confirmed_atoms=confirmed,
        gripper_confirmed_atoms=len(grippers), windows=windows,
        all_40_knots_consumed=all(retained), raw_knot_consumed=retained,
        retained_40_corrected_knot_indices_consumed=all(retained), original_raw_curve_directly_submitted=False,
        actual_endpoint_TCP=samples[-1]['tcp'].tolist(), operator_cancel=cancel,
        source_relative_ttl_s=.4, replan_cadence_s=.35, physics_dt_s=.002,
        controller_selected_reference='DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED',
        state_context_separation='REQUIRES_ACTUAL_PILOT_ANALYSIS',
        theta_response_identifiable='REQUIRES_ACTUAL_PILOT_ANALYSIS',
        WM_benefit='NOT_TESTED_BY_PROBE_COLLECTION', exact_state_fork=False,
        full_task_success=False, online_VLA=False, VLA_queries=0, VLA_commands=0,
        WM_updates=0, Actor_updates=0, GR00T_updates=0, WBC_changes=False,
        safety_STOP_qualification='NOT_TESTED', native_stage_RESULT_json_required=True)
    write(adapter.output/'TRAJECTORY_RESPONSE_COLLECTION.private.json', report)
    need(len(windows) == (end_step-INIT_STEP)//25 and confirmed == len(windows)
         and len(grippers) == len(windows) and all(retained), 'COMPLETE_ORIGINAL_POSE_GRIPPER_AND_RAW40_CONSUMPTION')
    adapter.reference_qualification = report
    return report
