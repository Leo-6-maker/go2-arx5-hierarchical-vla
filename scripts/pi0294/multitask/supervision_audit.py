"""Saved observations and submitted commands; diagnostic sidecar, no labels."""
import hashlib
import json
import math
from pathlib import Path


def audit(path, *, expected_sha256):
    path = Path(path)
    initial = path.stat()
    digest = hashlib.sha256()
    bounds = None
    stage = 'INITIAL'
    stages, events, previous = {}, [], {}
    command = pending = task_start = None
    submitted = completed = ticks = 0
    last_time = -math.inf
    with path.open('rb') as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            kind, value = row['kind'], row['value']
            if kind == 'qualification':
                if bounds is not None:
                    raise ValueError('DUPLICATE_NATIVE_QUALIFICATION')
                bounds = value['original_arm_bounds']
            elif kind == 'observation':
                stage = value.get('privileged_teacher_stage', 'STUDENT_PHASE_UNANNOTATED')
                if task_start is None:
                    task_start = value['time_s']
            elif kind == 'submitted_interval':
                if pending is not None:
                    raise ValueError('UNFINISHED_PREVIOUS_INTERVAL')
                command, pending = value['action_EEF10'][9], value
                submitted += 1
            elif kind == 'completed_interval':
                if pending is None or any(value[k] != pending[k] for k in (
                        'ref_id', 'observation_index', 'action_EEF10', 'start_native_step')):
                    raise ValueError('ACTUAL_SUBMISSION_COMPLETION_PAIR')
                pending = None
                completed += 1
            if kind not in ('warmup_tick', 'native_tick', 'native_tick_failure'):
                continue
            state = value['actual_state']
            q, timestamp = state['arm_q'], state['time_s']
            ticks += 1
            if (bounds is None or len(bounds) != 6 or len(q) != 6
                    or not all(map(math.isfinite, q)) or timestamp <= last_time
                    or any(len(b) != 2 or not all(map(math.isfinite, b))
                           or b[0] >= b[1] for b in bounds)):
                raise ValueError('ORIGINAL_BOUNDS_AND_MONOTONIC_ACTUAL_NATIVE_STATE')
            last_time = timestamp
            current = 'WARMUP' if kind == 'warmup_tick' else stage
            record = stages.setdefault(current, dict(
                ticks=0, start_s=timestamp, end_s=timestamp,
                min_joint_margin_rad=[math.inf] * 6,
                min_gap_m=math.inf, max_gap_m=-math.inf,
                min_finger_q=[math.inf] * 2, max_finger_q=[-math.inf] * 2,
                grasp_ticks=0, opposing_ticks=0, supported_ticks=0))
            record['ticks'] += 1
            record['end_s'] = timestamp
            for j, (a, (low, high)) in enumerate(zip(q, bounds)):
                record['min_joint_margin_rad'][j] = min(
                    record['min_joint_margin_rad'][j], a - low, high - a)
            gap = state['gripper_gap_m']
            fingers = state['finger_q']
            if not math.isfinite(gap) or len(fingers) != 2 or not all(map(math.isfinite, fingers)):
                raise ValueError('DIRECT_INNER_GAP_AND_TWO_FINGERS')
            record['min_gap_m'] = min(record['min_gap_m'], gap)
            record['max_gap_m'] = max(record['max_gap_m'], gap)
            for j, a in enumerate(fingers):
                record['min_finger_q'][j] = min(record['min_finger_q'][j], a)
                record['max_finger_q'][j] = max(record['max_finger_q'][j], a)
            for key, count in (
                    ('native_grasp', 'grasp_ticks'), ('opposing_contact', 'opposing_ticks'),
                    ('object_supported_by_shelf', 'supported_ticks')):
                present = bool(state.get(key, False))
                record[count] += int(present)
                if key in previous and present != previous[key]:
                    events.append(dict(
                        event=key + ('_ON' if present else '_OFF'), time_s=timestamp,
                        stage=current, submitted_gripper=command,
                        normalized_actual_gripper=state['gripper_fraction'],
                        finger_q=fingers, inner_gap_m=gap,
                        native_grasp=state['native_grasp'],
                        opposing_contact=state['opposing_contact'],
                        supported=state.get('object_supported_by_shelf')))
                previous[key] = present
    final = path.stat()
    keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if (any(getattr(initial, k) != getattr(final, k) for k in keys)
            or digest.hexdigest() != expected_sha256):
        raise PermissionError('SAVED_RAW_BYTES_OR_IDENTITY_CHANGED')
    return dict(
        schema='SAVED_BOOK_JOINT_GRIPPER_DIAGNOSTIC_V1',
        raw_sha256=digest.hexdigest(), bytes=initial.st_size, native_ticks=ticks,
        submitted_intervals=submitted, completed_intervals=completed,
        unfinished_interval=pending is not None, task_start_s=task_start,
        stages=stages, events=events, joint_evidence='DIRECT_RECORDED_ACTUAL_Q',
        command_evidence='DIRECT_SUBMITTED_EEF10_NOT_JOINT_TARGET',
        joint_tracking_error='REQUIRES_SAVED_OR_VERIFIED_RECONSTRUCTED_JOINT_COMMAND',
        labels='DIAGNOSTIC_SIDECAR_NOT_STUDENT_INPUT_OR_EXPERT_EXPORT',
        physics_integrations=0, render_calls=0, model_loads=0, optimizer_updates=0)
