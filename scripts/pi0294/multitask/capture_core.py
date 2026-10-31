"""Finite capture loop for an existing, separately admitted source owner.

This library does not check authority or instantiate an engine. It must only
be called after the original owner has admitted the exact new source scope,
entry/capsule, node-local paths, lease and complete resource reservation.
No CLI, scheduler, extra lease or S1-target permission reuse is introduced.
"""
import hashlib
import json
import math

import numpy as np

from go2arx5_vla.task_data.codec import pose10
from .object_relations import pose


def capture(engine, teacher, *, instruction, guard, emit, duration_s=32., execution_prefix=8):
    """Emit actual observations, submitted plans and completed physical intervals.

    `guard` enforces the already-reserved owner lifetime/budgets; `emit` writes
    to its existing durable recorder and must raise on persistence failure.
    Full H40s remain unexecuted reference evidence. Only first-eight points
    that each actually finish 25 native ticks become interval records. An
    independent complete-episode task/safety/data audit still admits training.
    """
    if (not callable(guard) or not callable(emit) or not isinstance(instruction, str) or not instruction
            or type(duration_s) not in (int, float) or not math.isfinite(duration_s)
            or not .4 <= duration_s <= 32 or abs(duration_s/.4-round(duration_s/.4)) > 1e-8
            or teacher.index != -1 or type(execution_prefix) is not int or execution_prefix not in (4, 8)):
        raise ValueError('FINITE_SOURCE_CAPTURE_WITH_FRESH_TEACHER_AND_OWNER_CALLBACKS')
    intervals = round(duration_s/.05)
    begin_steps = engine.steps
    begin_time = None
    active = None
    plan_id = None
    completed, plans, rgb_pairs = 0, 0, 0
    for index in range(intervals+1):
        guard()
        state = engine.snapshot()
        if begin_time is None: begin_time = float(state['time_s'])
        if (engine.steps != begin_steps+index*25
                or abs(state['time_s']-begin_time-index*.05) > 1e-8):
            raise ValueError('ACTUAL_NATIVE_CLOCK_OR_INTEGRATION_COUNT_MISMATCH')
        teacher.observe(state)
        images, calibration = engine.render(state)
        if (set(images) != {'body_rgb', 'wrist_rgb'} or any(
                image.shape != (256, 256, 3) or image.dtype != np.uint8 for image in images.values())
                or any(abs(calibration[name]['timestamp_s']-state['time_s']) > 1e-8 for name in images)):
            raise ValueError('EXACT_TWO_ACTUAL_RGB_AND_CALIBRATION_CLOCKS')
        for name in images:
            pose(calibration[name]['T_world_optical'])
            k = np.asarray(calibration[name]['K'], dtype=float)
            if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
                raise ValueError('ACTUAL_FINITE_CAMERA_CALIBRATION_REQUIRED')
        emit('observation', dict(index=index, native_step=engine.steps, time_s=state['time_s'],
            policy_obs=dict(instruction=instruction, actual_tcp=state['tcp'].copy(),
                actual_gripper=float(state['gripper_fraction']), body_rgb=images['body_rgb'], wrist_rgb=images['wrist_rgb']),
            actual_state=state, calibration=calibration, privileged_teacher_stage=teacher.stage))
        rgb_pairs += 1
        if index == intervals: break
        if index % execution_prefix == 0:
            active = teacher.plan()
            rows = np.stack([pose10(p, g) for p, g in zip(active['poses'], active['opening'])])
            if rows.shape != (40, 10) or not np.isfinite(rows).all():
                raise ValueError('FINITE_H40_EEF10_REFERENCE_REQUIRED')
            plan_id = hashlib.sha256(json.dumps(dict(index=index, source_time_s=state['time_s'],
                rows10=rows.tolist()), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            emit('planned_H40', dict(ref_id=plan_id, observation_index=index,
                source_time_s=state['time_s'], rows10=rows, stage=teacher.stage,
                role='UNEXECUTED_REFERENCE_EVIDENCE_NOT_TRAINING_LABEL', committed_prefix_count=0))
            plans += 1
        k = index % execution_prefix
        target, opening = active['poses'][k].copy(), float(active['opening'][k])
        command = pose10(target, opening)
        start_step = engine.steps
        emit('submitted_interval', dict(ref_id=plan_id, observation_index=index, prefix_index=k,
            source_time_s=state['time_s'], action_EEF10=command.copy(), start_native_step=start_step,
            expected_end_native_step=start_step+25, completed=False, training_admitted=False))
        for _ in range(25):
            guard()
            before = engine.steps
            try:
                engine.step(target, opening)
            except Exception as exc:
                # The native driver can stop after integrating this tick. Keep
                # its actual post-error state and count; do not complete labels.
                emit('native_tick_failure', dict(ref_id=plan_id, native_step=engine.steps,
                    actual_state=engine.snapshot(), error=type(exc).__name__+': '+str(exc)))
                raise
            if engine.steps != before+1:
                raise ValueError('SOURCE_DRIVER_MUST_ADVANCE_EXACTLY_ONE_NATIVE_TICK')
            actual = engine.snapshot()
            emit('native_tick', dict(ref_id=plan_id, native_step=engine.steps, actual_state=actual))
            if actual.get('safety_valid') is not True:
                raise PermissionError('ORIGINAL_NATIVE_SOURCE_HIGH_RATE_SAFETY_STOP')
        if engine.steps != start_step+25:
            raise ValueError('INCOMPLETE_INTERVAL_CANNOT_BECOME_COMMITTED_LABEL')
        emit('completed_interval', dict(ref_id=plan_id, observation_index=index, prefix_index=k,
            source_time_s=state['time_s'], action_EEF10=command.copy(), start_native_step=start_step,
            end_native_step=engine.steps, completed=True, training_admitted=False,
            selected_MPC_policy_consumption='NOT_APPLICABLE_SOURCE_ARM_DRIVER'))
        completed += 1
    return dict(status='RAW_SOURCE_COMPLETE_REQUIRES_INDEPENDENT_TASK_SAFETY_DATA_AUDIT',
        physics_integrations=engine.steps-begin_steps, completed_intervals=completed,
        observations=intervals+1, RGB_pairs=rgb_pairs, H40_plans=plans,
        final_stage=teacher.stage, teacher_candidate_complete=teacher.candidate_complete,
        task_success=None, training_admitted=False, model_calls=0)
