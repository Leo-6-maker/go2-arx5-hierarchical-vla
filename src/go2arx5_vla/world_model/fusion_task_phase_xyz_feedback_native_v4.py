"""Causal finite XYZ reference commissioning on the original native loop."""
import copy
import json
import os

import numpy as np

from scripts.pi0294.task_state import parse_state
from . import fusion_task_phase_xyz_feedback_v4 as recipe
from .fusion_task_phase_xyz_feedback_contract_v4 import PHASE, need
from .fusion_task_phase_initialization_native import _plain, _verify_actual_fk
from .fusion_reference_qualification_native import require_original_install_ack
from .storage import digest


def validate_adapter_install(adapter, action):
    need(adapter.spec.get('phase') == PHASE and hasattr(adapter, 'trajectory_initial_tcp'),
         'ACTUAL_V4_PHASE_AND_MEASURED_INIT')
    return recipe.validate_install_preview(action, initial_tcp=adapter.trajectory_initial_tcp,
        program=adapter.trajectory_program, step=adapter.observe()['step'],
        base=adapter.base_pose_hold, frozen_base=adapter.trajectory_initial_base,
        end_step=adapter.spec['max_ticks'])


def run(adapter):
    from .fusion_task_phase_xyz_feedback_contract_v4 import episode_config
    episode, source = episode_config(adapter.spec)
    return run_feedback_program(adapter, episode, source)


def run_feedback_program(adapter, episode, source, *, recipe_module=recipe, actual_phase=PHASE,
        report_schema='WM02_NATIVE_TASK_PHASE_XYZ_FEEDBACK_V4',
        goal_schema='WM02_NATIVE_TASK_PHASE_XYZ_FEEDBACK_GOAL_BINDING_V4',
        decision_schema='WM02_NATIVE_FINITE_XYZ_FEEDBACK_DECISION_V4',
        feedback_report_schema='WM02_NATIVE_FINITE_XYZ_FEEDBACK_REPORT_V4', continuation=None):
    """Original lifecycle with a checked caller's typed finite fixed-law recipe."""
    from scripts.pi0292.durable import write
    from .fusion_task_phase_initialization_native import run_initialization_program
    if actual_phase == PHASE:
        from . import fusion_task_phase_xyz_feedback_v4 as declared_recipe
    elif actual_phase == 'collect-target-fusion-task-phase-xyz-feedback-continuation-v5':
        from . import fusion_task_phase_xyz_feedback_continuation_v5 as declared_recipe
    elif actual_phase == 'collect-target-fusion-task-phase-pose-feedback-v6':
        from . import fusion_task_phase_pose_feedback_v6 as declared_recipe
    elif actual_phase == 'collect-target-fusion-task-phase-pose-continuation-v7':
        from . import fusion_task_phase_pose_continuation_v7 as declared_recipe
    elif actual_phase == 'collect-target-fusion-task-phase-bias-fade-v8':
        from . import fusion_task_phase_bias_fade_v8 as declared_recipe
    elif actual_phase == 'collect-target-fusion-task-phase-live-skill-v8':
        from . import fusion_task_phase_pose_continuation_v7 as declared_recipe
    elif actual_phase in ('collect-target-fusion-task-phase-live-skill-v9','collect-target-stable-start-short-v1','collect-target-stable-task-data-v1','collect-target-stable-task-eval-v1','collect-target-stable-task-closed-loop-v1'):
        from . import fusion_task_phase_pose_slow_feedback_v9 as declared_recipe
    else:
        raise ValueError('EXACT_KNOWN_FINITE_FEEDBACK_NATIVE_CALLER_REQUIRED')
    need(recipe_module is declared_recipe, 'EXACT_TYPED_VERSIONED_FEEDBACK_RECIPE_MODULE')
    recipe = recipe_module
    phase = actual_phase
    steps = tuple(episode['feedback_decision_native_steps'])
    receipts = []

    def freeze(inputs, actual_episode, initial_tcp, decision_tcp, base):
        program = recipe.make_program(inputs, actual_episode, initial_tcp, decision_tcp, base)
        return program, program.evidence()['frozen_predecision_reference_binding']

    def before_preview(current_adapter, before, last_sample, closed_window):
        step = before['step']
        if step not in steps:
            return False
        need(current_adapter is adapter and len(receipts) == steps.index(step),
             'ONE_SAVED_DECISION_PER_PREDECLARED_STEP_NO_RETRY')
        program = adapter.trajectory_program
        need(type(program) is recipe.FeedbackProgram and closed_window is not None
             and closed_window.get('native_start_step') == step-25
             and closed_window.get('native_end_step') == step
             and closed_window.get('actual_ticks') == 25
             and bool(closed_window.get('solver_policies_with_exact_installed_target'))
             and (closed_window.get('gripper') or {}).get('command_consumed') is True
             and closed_window.get('reference_sha256') == adapter.active_reference['wire_sha256']
             and last_sample.get('native_step') == step,
             'PREVIOUS_REAL_COMPLETE_POSE_GRIPPER_AND_SOLVER_WINDOW_REQUIRED')
        ack = require_original_install_ack(adapter.active_reference)
        state_path = adapter.output/'state.bin'
        raw = state_path.read_bytes()
        packet = parse_state(raw)
        original_reader = getattr(adapter, 'actual_state', None)
        measured = adapter.measured_task_state()
        need(getattr(adapter, 'actual_state', None) is original_reader
             and packet['step'] == before['step'] == step
             and measured['native_step'] == last_sample['native_step'] == step
             and all(np.array_equal(measured[key], last_sample[key])
                     for key in ('qpos', 'qvel', 'ctrl', 'tcp', 'object_pose')),
             'FRESH_ORIGINAL_READER_SAME_ACTUAL_CLOSED_WINDOW_AND_PACKET')
        witness = _verify_actual_fk(adapter, packet, measured, expected_step=step)
        need(measured.get('safety_valid') is True and measured.get('native_abort') is False
             and measured.get('robot_object_contact') is False
             and measured.get('gripper_state_valid') is True and measured.get('object_present') is True,
             'ORIGINAL_SAFE_NONCONTACT_FEEDBACK_MEASUREMENT')
        update = program.prepare_feedback(step, measured['tcp'])
        prefix = 'POSE_FEEDBACK' if actual_phase in ('collect-target-fusion-task-phase-pose-feedback-v6','collect-target-fusion-task-phase-pose-continuation-v7','collect-target-fusion-task-phase-bias-fade-v8','collect-target-fusion-task-phase-live-skill-v8','collect-target-fusion-task-phase-live-skill-v9','collect-target-stable-start-short-v1','collect-target-stable-task-data-v1','collect-target-stable-task-eval-v1','collect-target-stable-task-closed-loop-v1') else 'XYZ_FEEDBACK'
        paths = {name: adapter.output/f'{prefix}_{len(receipts)+1:02d}_{name}' for name in (
            'STATE.bin', 'MEASURED.private.json', 'REFERENCE_PATH.private.json',
            'REMAINING_PATH.private.json', 'H40.private.json', 'DECISION.private.json')}
        need(not any(path.exists() or path.is_symlink() for path in paths.values()),
             'FRESH_FEEDBACK_ARTIFACTS_NO_OVERWRITE_OR_UNCERTAIN_REPLAY')
        atom = (step-recipe.DECISION_STEP)//25
        remaining = update.reference_path[atom:]
        preview = update.reference_path[np.minimum(atom+np.arange(40), program.end_atom)]
        documents = {'MEASURED.private.json': _plain(measured),
            'REFERENCE_PATH.private.json': update.reference_path.tolist(),
            'REMAINING_PATH.private.json': remaining.tolist(), 'H40.private.json': preview.tolist()}
        with paths['STATE.bin'].open('xb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        for name, value in documents.items():
            write(paths[name], value)
        decision = dict(schema=decision_schema,
            actual_spec_phase=phase, actual_spec_source_sha=adapter.spec['source_sha'],
            episode_sha256=adapter.spec['episode_manifest_sha256'],
            original_goal_binding_sha256=digest(adapter.output/'TASK_PHASE_GOAL_BINDING.private.json'),
            decision=copy.deepcopy(update.evidence), actual_fk_witness=witness,
            previous_complete_window=copy.deepcopy(closed_window), previous_original_install_ACK=ack,
            artifact_files={name: dict(path=str(path), sha256=digest(path))
                            for name, path in paths.items() if name != 'DECISION.private.json'},
            saved_before_program_commit_and_reference_install=True,
            planned_install_outcome='NOT_YET_KNOWN_RETAIN_ORIGINAL_ACK_AND_CONSUMPTION_RECORDS',
            old_unconsumed_previews_are_expert_labels=False, native_execution_authorized=False,
            original_WBC_safety_and_physics_unchanged=True, WM_updates=0, VLA_queries=0)
        write(paths['DECISION.private.json'], decision)
        need(json.loads(paths['DECISION.private.json'].read_bytes()) == decision
             and all(json.loads(paths[name].read_bytes()) == value for name, value in documents.items())
             and paths['STATE.bin'].read_bytes() == raw == state_path.read_bytes()
             and adapter.observe()['step'] == step,
             'DURABLE_READBACK_AND_UNCHANGED_PACKET_BEFORE_COMMIT_OR_INSTALL')
        receipt = dict(native_step=step, decision_path=str(paths['DECISION.private.json']),
                       decision_sha256=digest(paths['DECISION.private.json']))
        program.commit_feedback(update, saved_decision_sha256=receipt['decision_sha256'])
        receipts.append(receipt)
        event = ('finite-xyz-so3-feedback-saved-before-reference-install'
            if actual_phase in ('collect-target-fusion-task-phase-pose-feedback-v6','collect-target-fusion-task-phase-pose-continuation-v7','collect-target-fusion-task-phase-bias-fade-v8','collect-target-fusion-task-phase-live-skill-v8','collect-target-fusion-task-phase-live-skill-v9','collect-target-stable-start-short-v1','collect-target-stable-task-data-v1','collect-target-stable-task-eval-v1','collect-target-stable-task-closed-loop-v1')
            else 'finite-xyz-feedback-saved-before-reference-install')
        adapter.command_event(dict(phase=event, **receipt))
        return True

    def extend_report(report):
        program = adapter.trajectory_program
        need(len(receipts) == program.feedback_count == len(steps)
             and [row['native_step'] for row in receipts] == list(steps),
             'ALL_PREDECLARED_SAVED_FINITE_DECISIONS_REQUIRED')
        report['finite_xyz_feedback'] = dict(schema=feedback_report_schema,
            decisions=copy.deepcopy(receipts), actual_feedback_reference_updates=len(steps),
            committed_feedback_decisions=len(steps),
            nonzero_XYZ_corrections=sum(row['applied_delta_norm_m'] > 0
                                       for row in program.evidence()['feedback_decisions']),
            deadband_HOLD_count=sum(row['deadband_hold'] for row in program.evidence()['feedback_decisions']),
            sum_applied_XYZ_delta_norms_m=sum(row['applied_delta_norm_m']
                                            for row in program.evidence()['feedback_decisions']),
            final_reference_path_sha256=recipe.fixed._sha(program.reference_path),
            unchanged_desired_path_sha256=recipe.fixed._sha(program.desired_path),
            causal_completed_window_feedback=True, ordinary_feedback_not_model_benefit=True,
            original_windows_retain_ACK_solver_pose_and_gripper_consumption=True,
            WM_updates=0, VLA_queries=0, Actor_updates=0)
        if actual_phase in ('collect-target-fusion-task-phase-pose-feedback-v6','collect-target-fusion-task-phase-pose-continuation-v7','collect-target-fusion-task-phase-bias-fade-v8','collect-target-fusion-task-phase-live-skill-v8','collect-target-fusion-task-phase-live-skill-v9','collect-target-stable-start-short-v1','collect-target-stable-task-data-v1','collect-target-stable-task-eval-v1','collect-target-stable-task-closed-loop-v1'):
            rows = program.evidence()['feedback_decisions']
            report['finite_xyz_feedback'].update(
                nonzero_SO3_corrections=sum(row['applied_rotation_norm_rad'] > 0 for row in rows),
                angular_deadband_HOLD_count=sum(row['angular_deadband_hold'] for row in rows),
                sum_applied_SO3_delta_norms_rad=sum(row['applied_rotation_norm_rad'] for row in rows),
                rotation_error_frame='ACTUAL_TARGET_WORLD_LEFT_LOG_R_GOAL_R_ACTUAL_TRANSPOSE',
                reference_rotation_from_previous_committed_endpoint=True,
                feedback_components=['TARGET_WORLD_XYZ','TARGET_WORLD_LEFT_SO3'])

    context = dict(initialization_condition=recipe.CONDITION,
        baseline_condition=recipe.BASELINE_CONDITION,
        historical_terminal_tracking_error_used=False, learned_reference_adapter_used=False,
        ordinary_XYZ_feedback_commissioning_not_WM_or_VLA_gain=True,
        independent_restart_not_complete_MPC_state_fork=True)
    if actual_phase in ('collect-target-fusion-task-phase-pose-feedback-v6','collect-target-fusion-task-phase-pose-continuation-v7','collect-target-fusion-task-phase-bias-fade-v8','collect-target-fusion-task-phase-live-skill-v8','collect-target-fusion-task-phase-live-skill-v9','collect-target-stable-start-short-v1','collect-target-stable-task-data-v1','collect-target-stable-task-eval-v1','collect-target-stable-task-closed-loop-v1'):
        context['ordinary_XYZ_SO3_feedback_commissioning_not_WM_or_VLA_gain'] = True

    return run_initialization_program(adapter, episode, source, recipe_module=recipe,
        actual_phase=phase, report_schema=report_schema,
        goal_schema=goal_schema,
        reference_calibration_factory=freeze, before_preview=before_preview,
        report_extension=extend_report,
        report_context=context, continuation=continuation)
