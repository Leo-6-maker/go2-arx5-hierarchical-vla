"""Thin full-trajectory qualification on the original native lifecycle.

Only the new phase defers install-time gripper writes. One contiguous gripper
packet per actual atom is consumed by the unchanged original wrapper.
"""
import struct
import numpy as np

from .fusion_trajectory_qualification_contract import (
    INIT_STEP, DECISION_STEP, TAIL_ATOMS, need, hold_preview, episode_config,
    endpoint, make_program, validate_install_preview)
from .fusion_reference_qualification_native import require_original_install_ack, consumed_window
from .target_reference import require_live_window, require_install_clock


def actual_trace_binding(adapter):
    from .storage import digest
    from .init_trace_reader import RECORD
    loaded = adapter.capture_loaded_identity()
    path = adapter.spec['trace_library']
    need(loaded['sim']['loaded_files_sha256'].get(path) == digest(path),
         'ACTUAL_OWNED_SIM_MUST_LOAD_PINNED_NEW_TRACE_LIBRARY')
    with (adapter.output/'init.trace').open('rb') as stream:
        header = stream.read(16)
    capacity = (adapter.spec.get('trajectory_trace_capacity',262144) if adapter.spec.get('phase') in (
        'collect-target-fusion-trajectory-slow-qualification-v2','collect-target-fusion-trajectory-response-v1','collect-target-fusion-paused-query-v2',
        'collect-target-fusion-live-proposal-qualification-v3','collect-target-fusion-source-posture-qualification-v1','collect-target-fusion-source-posture-query-v3',
        'collect-target-fusion-task-phase-initialization-v1','collect-target-fusion-task-phase-bias-calibration-v2',
        'collect-target-fusion-task-phase-bias-continuation-v3','collect-target-fusion-task-phase-xyz-feedback-v4','collect-target-fusion-task-phase-xyz-feedback-continuation-v5','collect-target-fusion-task-phase-pose-feedback-v6') else 131072)
    need(len(header) == 16 and header[:8] == b'WM02IT01'
         and struct.unpack('<II', header[8:]) == (RECORD.size, capacity),
         'ACTUAL_NEW_TRACE_HEADER_CAPACITY_'+str(capacity))
    return dict(trace_library_sha256=digest(path), actual_trace_capacity=capacity,
                actual_trace_record_bytes=RECORD.size, owned_sim_loaded_identity=loaded['sim'])


def validate_adapter_install(adapter, action):
    need(hasattr(adapter, 'trajectory_initial_tcp'), 'MEASURED_INIT_REQUIRED')
    return validate_install_preview(action, initial_tcp=adapter.trajectory_initial_tcp,
        program=adapter.trajectory_program, step=adapter.observe()['step'],
        base=adapter.base_pose_hold, frozen_base=adapter.trajectory_initial_base,
        end_step=adapter.spec['max_ticks'])


def submit_gripper(adapter, sequence, opening):
    from scripts.pi0294.task_gripper_contract import require_working_target
    from scripts.pi0294.task_state import write_gripper
    require_working_target(opening)
    adapter.wait_source()
    packet = adapter.observe()
    source = float(adapter.latest['ee'].time)
    require_install_clock(source, packet['time_s'])
    write_gripper(adapter.output/'gripper.bin', sequence, source, float(opening))
    return dict(sequence=sequence, opening=float(opening), source_mpc_s=source,
                expiry=source+.4, native_start_step=packet['step'])


def confirm_gripper(adapter, submitted, actual):
    from .next2_contact import gripper_memory
    memory = gripper_memory(adapter)
    values = memory['values']
    need(memory.get('status') == 'ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY'
         and values.get('initialized') is True and values.get('has_command') is True
         and values.get('steps') == actual['native_step']
         and values.get('sequence') == submitted['sequence']
         and abs(values['opening']-submitted['opening']) <= 1e-12
         and abs(values['expiry']-submitted['expiry']) <= 1e-12
         and actual['time_s'] <= values['expiry']+1e-9,
         'ACTUAL_WRAPPER_CONTIGUOUS_GRIPPER_CONSUMPTION_REQUIRED')
    return dict(submitted=submitted, consumed_wrapper=memory,
                actual_finger_q_m=np.asarray(actual['finger_q'], float).tolist(),
                actual_gripper_fraction=float(actual['gripper_fraction']),
                command_consumed=True, achieved_opening_equals_command='NOT_ASSUMED')


def run(adapter):
    episode, inputs = episode_config(adapter.spec)
    return run_program(adapter, episode, inputs)


def run_program(adapter, episode, inputs, *, program_factory=None,
                report_context=None, task_score_contract=None, report_finalizer=None,
                sample_observer=None, before_preview=None):
    """Reuse actual native consumption for an independently checked recipe.

    The original entry supplies its unchanged contract. A versioned caller may
    supply a different verified proposal, but cannot replace the native safety,
    full-preview certificate, TTL, gripper ACK or exact-step consumption below.
    """
    from scripts.pi0292.durable import write
    from scripts.pi0292.eval_contract import matrix10
    from .target_stability import motion, screen
    from .next2_episode import state_vector
    program_factory = make_program if program_factory is None else program_factory
    report_context = {} if report_context is None else report_context
    end_step = endpoint(episode)
    adapter.prepare_dynamic()
    trace_binding = actual_trace_binding(adapter)
    initial = adapter.measured_task_state()
    need(initial['native_step'] == INIT_STEP, 'ACTUAL_INIT_2503_REQUIRED')
    if sample_observer is not None:
        sample_observer(initial)
    adapter.trajectory_initial_tcp = initial['tcp'].copy()
    adapter.trajectory_initial_base = list(adapter.base_pose_hold)
    adapter.trajectory_program = None
    samples, windows, acks, grippers = [initial], [], [], []

    def history_input():
        state = np.asarray([state_vector(s) for s in samples[-9:]])
        need(np.isfinite(state).all() and all(w['solver_policies_with_exact_installed_target']
             and w['gripper']['command_consumed'] for w in windows[-8:]),
             'HISTORY_REQUIRES_FINITE_ACTUAL_STATES_AND_CONFIRMED_PAST_CONSUMPTION')
        past = [w['actual_installed_consumed_EEF10_BASE7'] for w in windows[-8:]]
        return dict(actual_state_history=state.tolist(), actual_state_history_shape=list(state.shape),
                    actual_history_native_steps=[s['native_step'] for s in samples[-9:]],
                    past_consumed_EEF10_BASE7=past,
                    past_reference_native_end_steps=[w['native_end_step'] for w in windows[-8:]],
                    history_provenance='ACTUAL_SAMPLES_AND_EXACT_INSTALLED_SOLVER_JOIN_PER_ATOM',
                    full_MPC_hidden_state_restored=False)

    def action_now():
        step = adapter.observe()['step']
        return (hold_preview(initial['tcp']) if step < DECISION_STEP else
                adapter.trajectory_program.preview((step-DECISION_STEP)//25))

    def install():
        reference = adapter.install_reference(action_now())
        acks.append(require_original_install_ack(reference))
        return adapter.observe()['step']

    installed_step = install()
    while adapter.observe()['step'] < end_step:
        before = adapter.observe(); step = before['step']
        force_install = False
        if before_preview is not None:
            force_install = before_preview(adapter, before, samples[-1], windows[-1] if windows else None)
            need(type(force_install) is bool, 'EXPLICIT_BEFORE_PREVIEW_FORCE_INSTALL_BOOLEAN')
        if step == INIT_STEP+40*25:
            installed_step = install()
        if step == DECISION_STEP:
            last = samples[-9:]
            relative = motion(np.array([s['qpos'][:3] for s in last]),
                np.array([s['qpos'][3:7] for s in last]),
                np.array([s['tcp'][:3, 3] for s in last]))['relative_tcp']
            prefix = screen(np.array([s['time_s'] for s in last]), relative)
            prefix.update(schema='WM02_TRAJECTORY_QUALIFICATION_PREFIX_V1', native_step=step,
                          physics_paused_at_assessment=True, thresholds_not_native_safety=True)
            write(adapter.output/'TRAJECTORY_QUALIFICATION_PREFIX.private.json', prefix)
            need(prefix['qualified'], 'ORIGINAL_STEADY_PREFIX_FAILED_NO_NEW_ACTION')
            decision = adapter.measured_task_state()
            program = program_factory(inputs, episode, initial['tcp'], decision['tcp'],
                                   adapter.trajectory_initial_base)
            adapter.trajectory_program = program
            history = history_input()
            need(history['actual_state_history_shape'] == [9, 106]
                 and len(history['past_consumed_EEF10_BASE7']) == 8,
                 'ACTUAL_DECISION_HISTORY_9X106_AND_EIGHT_CONSUMED_REFERENCES')
            evidence = dict(schema='WM02_TRAJECTORY_QUALIFICATION_DECISION_V1',
                episode=episode, adaptation=program.evidence(),
                raw_H40=inputs['raw'].tolist(), desired_path=program.desired_path.tolist(),
                nominal_reference_path=program.reference_path.tolist(),
                archived_input_provenance=inputs['provenance'],
                actual_history=history,
                actual_native_step=step, future_response_observed=False,
                online_query=False, actual_VLA_commands=0, original_Processor_decodes=0)
            evidence.update(report_context)
            write(adapter.output/'TRAJECTORY_QUALIFICATION_DECISION.private.json', evidence)
            need(adapter.observe()['step'] == step, 'NO_DECISION_INTEGRATION')
            adapter.command_event(dict(phase='trajectory-decision-before-install', **evidence))
            installed_step = install()
        elif force_install or step-installed_step >= 175:
            installed_step = install()
        expected = action_now()[0]
        history = history_input() if step >= DECISION_STEP else None
        clock = require_live_window(adapter.active_reference['source_mpc_s'], before['time_s'], 25)
        index = (step-installed_step)//25
        installed_row = np.asarray(adapter.active_reference['action_EEF10'])[index]
        need(np.allclose(installed_row, expected, rtol=0, atol=1e-12),
             'ACTUAL_INSTALLED_PREVIEW_INDEX_MUST_MATCH_CURRENT_PERSISTENT_PATH')
        # Independent sequence: pose renewal must not produce a second write
        # before this integration. The original wrapper accepts exactly +1.
        submitted = submit_gripper(adapter, len(windows), expected[9])
        event_start = len(adapter.ros_evidence)
        after = adapter.advance_to(step+25); adapter.spin(.15)
        actual = adapter.measured_task_state(); samples.append(actual)
        need(actual['native_step'] == after['step'] == step+25
             and actual['safety_valid'] is True and actual['native_abort'] is False,
             'ORIGINAL_NATIVE_FEEDBACK_AND_SAFETY')
        if sample_observer is not None:
            sample_observer(actual)
        need(adapter.latest['mode'].data == 'WBC'
             and not any(r.get('reason') == 'PREFIX_TTL_EXPIRED' for r in adapter.reference_events()),
             'ORIGINAL_WBC_AND_TTL')
        grip = confirm_gripper(adapter, submitted, actual); grippers.append(grip)
        row = consumed_window(adapter, event_start, before, after, clock)
        elapsed = (step-DECISION_STEP)//25
        row.update(trajectory_elapsed_atom=elapsed if step >= DECISION_STEP else None,
            nominal_submitted_EEF10=expected.tolist(), gripper=grip,
            actual_installed_consumed_EEF10_BASE7=np.r_[installed_row, adapter.base_pose_hold].tolist(),
            actual_installed_source_age_s=before['time_s']-adapter.active_reference['source_mpc_s'],
            future_prefix_remaining_s=clock['remaining_at_install_s'],
            full_final_installed_H40=np.asarray(adapter.active_reference['action_EEF10']).tolist(),
            actual_history_before_atom=history,
            desired_EEF10=(adapter.trajectory_program.desired_at(elapsed).tolist()
                          if step >= DECISION_STEP else None),
            desired_interval_end_EEF10=(adapter.trajectory_program.desired_at(elapsed+1).tolist()
                                       if step >= DECISION_STEP else None),
            raw_query_identity=episode['query_pins'],
            actual_TCP=actual['tcp'].tolist())
        windows.append(row)
        adapter.command_event(dict(phase='controller-consumed' if row['solver_policies_with_exact_installed_target']
                                   else 'consumption_unconfirmed', **row))
        adapter.reference_history.append(dict(end_native_step=after['step'],
            pose=matrix10(adapter.active_reference['action_EEF10'])[index]))
    adapter.capture_loaded_identity()
    cancel = adapter.cancel_reference()
    adapter.seal_trace(end_step)
    program = adapter.trajectory_program
    need(program is not None, 'ACTUAL_PROGRAM_REQUIRED')
    confirmed = sum(bool(w['solver_policies_with_exact_installed_target']) for w in windows)
    elapsed_consumed = {w['trajectory_elapsed_atom'] for w in windows if w['trajectory_elapsed_atom'] is not None}
    knots_consumed = [i in elapsed_consumed for i in program.knot_indices]
    final = samples[-1]['tcp']
    desired_final = matrix10(program.desired_at(program.end_atom))
    reference_final = matrix10(program.preview(program.end_atom)[0])
    report = dict(schema='WM02_NATIVE_TRAJECTORY_QUALIFICATION_V1',
        status='PROGRAM_COMPLETED_REQUIRES_OUTER_NATIVE_TRACE_AND_SHUTDOWN_AUDIT',
        experiment_role='NON_VLA_CALIBRATION', desired_outcome_provenance='CALIBRATION_INTENT',
        episode=episode, adaptation=program.evidence(), original_install_ACKs=acks,
        actual_trace_binding=trace_binding,
        actual_collection_start_step=INIT_STEP, native_endpoint_step=end_step,
        task_native_ticks=end_step-INIT_STEP, complete_task_atoms=len(windows),
        post_decision_atoms=program.end_atom+TAIL_ATOMS, tail_hold_atoms=TAIL_ATOMS,
        solver_target_confirmed_atoms=confirmed, gripper_confirmed_atoms=len(grippers),
        all_40_knots_consumed=all(knots_consumed), raw_knot_consumed=knots_consumed,
        windows=windows, actual_endpoint_TCP=final.tolist(),
        actual_minus_desired_xyz_m=(final[:3, 3]-desired_final[:3, 3]).tolist(),
        actual_minus_reference_xyz_m=(final[:3, 3]-reference_final[:3, 3]).tolist(),
        operator_cancel=cancel, source_relative_ttl_s=.4, replan_cadence_s=.35,
        controller_selected_reference='DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED',
        full_task_success=False, online_VLA=False, VLA_inferences=0, VLA_commands=0,
        Processor_decodes=0, WM_updates=0, Actor_updates=0, WBC_changes=False, physics_dt_s=.002,
        safety_STOP_qualification='NOT_TESTED', target_task_qualification=False,
        target_collision_qualification='ORIGINAL_PER_PREVIEW_CERTIFIER_ONLY_NOT_A_GENERAL_DOMAIN',
        domain_generalization_claim=False, native_stage_RESULT_json_required=True)
    report.update(report_context)
    if task_score_contract is not None:
        from .fusion_short_skill import score
        report['short_skill_score'] = score(
            [sample for sample in samples if sample['native_step'] >= DECISION_STEP],
            task_score_contract)
    if report_finalizer is not None:
        report_finalizer(report)
    write(adapter.output/'TRAJECTORY_QUALIFICATION.private.json', report)
    need(len(windows) == (end_step-INIT_STEP)//25 and confirmed == len(windows)
         and len(grippers) == len(windows) and all(knots_consumed),
         'COMPLETE_POSE_AND_GRIPPER_CONSUMPTION_ALL_40_KNOTS_REQUIRED')
    adapter.reference_qualification = report
    return report
