"""Thin new-phase glue for the existing AUTO1/PILOT1 native lifecycle.

No simulator/controller/scheduler replacement. The adapter alone owns native
steps, certificates, gripper writes, ACKs, expiry, STOP and trace shutdown.
"""
import hashlib
import numpy as np

from .fusion_reference_qualification_contract import (
    DECISION_STEP, END_STEP, INIT_STEP, ReferenceProgram, episode_config,
    hold_preview, need, validate_install_preview)
from .target_reference import installed_target_identities, require_live_window, trajectory_identity


def validate_adapter_install(adapter, action):
    """Called only for this new phase, before the unchanged certifier/publish."""
    need(hasattr(adapter, 'qualification_initial_tcp'), 'NEW_PHASE_MUST_PREPARE_MEASURED_INITIAL_STATE')
    return validate_install_preview(action, initial_tcp=adapter.qualification_initial_tcp,
        program=adapter.qualification_program, native_step=adapter.observe()['step'],
        base_pose_hold=adapter.base_pose_hold,
        frozen_base_pose_hold=adapter.qualification_initial_base_pose_hold)


def require_original_install_ack(reference):
    """Bind original full-body receiver ACK; no fabricated EEF-only ACK ABI."""
    wire, ref, source = reference['wire'], reference['wire_sha256'], reference['source_mpc_s']
    need(hashlib.sha256(wire.encode()).hexdigest() == ref, 'UNCHANGED_ORIGINAL_WIRE')
    rows = reference['receiver_events']
    need(not any(r['kind'] == 'reject' and r.get('wire_sha256') == ref for r in rows), 'ORIGINAL_RECEIVER_REJECTED')
    accepted = [r for r in rows if r['kind'] == 'accept' and r.get('ref_id') == ref]
    installed = [r for r in rows if r['kind'] == 'installed' and r.get('ref_id') == ref]
    need(accepted and installed, 'ORIGINAL_ACCEPT_AND_INSTALLED_REQUIRED')
    for row in accepted:
        need(row.get('raw') == wire and row.get('source_mpc_s') == source
             and row.get('deadline') == source+.4
             and 0 <= row['receiver_mpc_s']-source <= .025, 'EXACT_ORIGINAL_ACCEPT_WIRE_AND_TTL')
    for row in installed:
        need(row.get('bound_start') == source
             and 0 <= row['observation_time_s']-source <= .025, 'EXACT_ORIGINAL_INSTALLED_SOURCE')
    return dict(wire_sha256=ref, accepted=len(accepted), installed=len(installed),
                physics_advance_authorized=False)


def consumed_window(adapter, event_start, before, after, clock):
    """Use the same exact installed-target/solver-policy join as the old path."""
    active = adapter.active_reference
    installed = [r for r in active['receiver_events'] if r['kind'] == 'installed'][-1]
    sent = installed_target_identities(adapter.reference_events(), active['wire_sha256'])
    matched, arm = [], []
    for row in adapter.ros_evidence[event_start:]:
        msg = row['message']
        if row['topic_key'] == 'policy':
            target = msg['plan_target_trajectories']
            key = trajectory_identity(target['time_trajectory'],
                [s['value'] for s in target['state_trajectory']],
                [s['value'] for s in target['input_trajectory']])
            if key in sent and sent[key]['monotonic_s'] <= row['received_monotonic_s']:
                matched.append(dict(trajectory_sha256=key,
                    receiver_send_monotonic_s=sent[key]['monotonic_s'],
                    received_monotonic_s=row['received_monotonic_s']))
        if row['topic_key'] == 'arm_request':
            arm.append(row)
    return dict(native_start_step=before['step'], native_end_step=after['step'],
        actual_ticks=after['step']-before['step'], clock=clock,
        reference_sha256=active['wire_sha256'],
        target_trajectory_sha256=trajectory_identity(installed['times'], installed['poses'], installed['inputs']),
        solver_policies_with_exact_installed_target=matched, arm_request_count=len(arm),
        controller_selected_reference='DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED')


def run(adapter):
    from scripts.pi0292.eval_contract import matrix10
    from scripts.pi0292.durable import write
    from .target_stability import motion, screen

    episode = episode_config(adapter.spec)
    adapter.prepare_dynamic()
    initial = adapter.measured_task_state()
    need(initial['native_step'] == INIT_STEP, 'ACTUAL_INIT_BOUNDARY_2503')
    adapter.qualification_initial_tcp = initial['tcp'].copy()
    adapter.qualification_initial_base_pose_hold = list(adapter.base_pose_hold)
    adapter.qualification_program = None
    samples, windows, acknowledgements = [initial], [], []

    def install():
        step = adapter.observe()['step']
        if step < DECISION_STEP:
            action = hold_preview(initial['tcp'])
        else:
            action = adapter.qualification_program.preview((step-DECISION_STEP)//25)
        reference = adapter.install_reference(action)  # Original certificate, BASE, gripper and receiver.
        acknowledgements.append(require_original_install_ack(reference))
        return step

    installed_step = install()
    while adapter.observe()['step'] < END_STEP:
        before = adapter.observe(); step = before['step']
        if step == INIT_STEP+40*25:
            installed_step = install()  # Same fixed initial hold at the original history boundary.
        if step == DECISION_STEP:
            last = samples[-9:]
            relative = motion(np.array([s['qpos'][:3] for s in last]),
                np.array([s['qpos'][3:7] for s in last]),
                np.array([s['tcp'][:3, 3] for s in last]))['relative_tcp']
            prefix = screen(np.array([s['time_s'] for s in last]), relative)
            prefix.update(schema='WM02_REFERENCE_QUALIFICATION_PREFIX_V1', actual_native_step=step,
                          physics_paused_at_assessment=True, thresholds_not_native_safety=True)
            write(adapter.output/'REFERENCE_QUALIFICATION_PREFIX.private.json', prefix)
            need(prefix['qualified'], 'ORIGINAL_STEADY_PREFIX_SCREEN_FAILED_NO_NEW_ACTION')
            decision = adapter.measured_task_state()
            program = ReferenceProgram(episode['candidate_id'], episode['reference_anchor_mode'],
                initial['tcp'], decision['tcp'], adapter.qualification_initial_base_pose_hold)
            adapter.qualification_program = program
            evidence = dict(program.evidence(), schema='WM02_REFERENCE_QUALIFICATION_DECISION_V1',
                native_step=step, native_time_s=before['time_s'], future_response_observed=False,
                initial_qpos=initial['qpos'].tolist(), actual_decision_qpos=decision['qpos'].tolist(),
                proposed_H40=program.preview().tolist(), new_domain_native_qualified=False)
            write(adapter.output/'REFERENCE_QUALIFICATION_DECISION.private.json', evidence)
            need(adapter.observe()['step'] == step, 'NO_PHYSICS_ADVANCE_DURING_DECISION')
            adapter.command_event(dict(phase='new-reference-qualification-decision-before-install', **evidence))
            installed_step = install()
        elif step-installed_step >= 175:
            installed_step = install()
        clock = require_live_window(adapter.active_reference['source_mpc_s'], before['time_s'], 25)
        event_start = len(adapter.ros_evidence)
        after = adapter.advance_to(step+25); adapter.spin(.15)
        actual = adapter.measured_task_state(); samples.append(actual)
        need(actual['native_step'] == after['step'] and actual['safety_valid'] is True
             and actual['native_abort'] is False, 'ORIGINAL_NATIVE_FEEDBACK_AND_SAFETY')
        need(adapter.latest['mode'].data == 'WBC'
             and not any(r.get('reason') == 'PREFIX_TTL_EXPIRED' for r in adapter.reference_events()),
             'ORIGINAL_WBC_MODE_AND_SOURCE_TTL')
        row = consumed_window(adapter, event_start, before, after, clock)
        windows.append(row)
        adapter.command_event(dict(phase='controller-consumed' if row['solver_policies_with_exact_installed_target']
                                   else 'consumption_unconfirmed', **row))
        adapter.command_event(dict(phase='achieved', native_step=after['step'], sim_time_s=after['time_s'],
                                  tcp=actual['tcp'].tolist()))
        index = (step-installed_step)//25
        adapter.reference_history.append(dict(end_native_step=after['step'],
            pose=matrix10(adapter.active_reference['action_EEF10'])[index]))
    adapter.capture_loaded_identity()
    cancel = adapter.cancel_reference()
    adapter.seal_trace(END_STEP)
    need(adapter.qualification_program is not None, 'ACTUAL_NEW_DECISION_WAS_INSTALLED')
    program = adapter.qualification_program
    endpoint = samples[-1]['tcp']
    confirmed = sum(bool(w['solver_policies_with_exact_installed_target']) for w in windows)
    report = dict(schema='WM02_NATIVE_REFERENCE_PROGRAM_QUALIFICATION_V1',
        status='PROGRAM_COMPLETED_REQUIRES_OUTER_NATIVE_TRACE_AND_SHUTDOWN_AUDIT',
        episode=episode, program=program.evidence(), original_install_ACKs=acknowledgements,
        actual_collection_start_step=INIT_STEP, native_endpoint_step=END_STEP,
        task_native_ticks=END_STEP-INIT_STEP, complete_task_atoms=len(windows),
        post_decision_atoms=16, solver_target_confirmed_atoms=confirmed, windows=windows,
        actual_endpoint_TCP=endpoint.tolist(),
        actual_minus_desired_xyz_m=(endpoint[:3, 3]-program.desired_goal[:3, 3]).tolist(),
        actual_minus_reference_goal_xyz_m=(endpoint[:3, 3]-program.reference_goal[:3, 3]).tolist(),
        operator_cancel=cancel, source_relative_ttl_s=.4, replan_cadence_s=.35,
        controller_selected_reference='DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED',
        full_task_success=False, VLA_inferences=0, WM_updates=0, Actor_updates=0,
        WBC_changes=False, physics_dt_s=.002, safety_STOP_qualification='NOT_TESTED',
        domain_generalization_claim=False, native_stage_RESULT_json_required=True)
    report['experiment_role'] = 'NON_VLA_CALIBRATION'
    write(adapter.output/'REFERENCE_PROGRAM_QUALIFICATION.private.json', report)
    need(len(windows) == 64 and confirmed == 64, 'ORIGINAL_SOLVER_TARGET_CONSUMPTION_REQUIRED_FOR_ALL_ATOMS')
    adapter.reference_qualification = report
    return report
