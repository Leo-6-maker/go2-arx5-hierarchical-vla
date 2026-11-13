"""Audit closed native expert executions for the SLA gripper endpoint gate."""
from pathlib import Path

import numpy as np

from . import task_learning_grasp_v1 as task
from .sla_expert_epsilon_v3 import SCHEMA, configuration
from .stable_task_dataset_v1 import load
from ..sla_adapter.gripper_latent_v2 import EPSILON, VERSION, decode_latent, label_latent


def audit_episode(root, *, expected_spec_sha256, expected_result_sha256):
    from scripts.pi0292.relay_protocol import unseal
    from .task_learning_grasp_native_v1 import complete_consumption
    from .task_learning_sequence_v1 import sequence
    from .storage import digest

    root = Path(root)
    native_root = root / 'native-01'
    spec, spec_pin = load(root / 'EXECUTION_SPEC.json', expected_spec_sha256)
    lease, lease_pin = load(root / 'control/LEASE.json')
    worker, worker_pin = load(root / 'WORKER_EXIT.private.json')
    result, result_pin = load(native_root / 'RESULT.json', expected_result_sha256)
    children = result['owned_process_exits']
    task.need(configuration(spec) and spec['registered_root'] == str(root)
              and lease['status'] == 'CLOSED' and lease['spec_sha256'] == spec_pin['sha256']
              and lease['source_sha'] == spec['source_sha'] == result['source_sha']
              and lease['worker_exit_code'] == worker['exit_code'] == 0 and worker['failure'] is None
              and result['status'] == 'NATIVE_STAGE_PASSED' and result['shutdown_clean'] is True
              and result['actual_target_integrations'] == task.E and not result['physics_stops']
              and result['original_joint_or_warning_samples'] == result['native_outer_guard_rejection_count'] == 0
              and len(children) == 4 and {v['name'] for v in children} == {'sim', 'solver', 'controller', 'planner'}
              and all(v['exit_code'] == 0 for v in children),
              'ACTUAL_CLOSED_EPSILON_EXPERT_WITH_ORIGINAL_NATIVE_SAFETY')
    q = result['reference_qualification']
    epsilon = q['sla_expert_epsilon']
    task.need(complete_consumption(spec, q)
              and q['task_condition_success'] is True and q['new_task_score']['task_success'] is True
              and q['new_task_score']['contract'] == task.score_contract()
              and q['GR00T_queries'] == q['VLA_queries'] == q['GR00T_updates'] == 0
              and epsilon['configuration'] == spec['sla_expert_epsilon_v3']
              and epsilon['fully_consumed_H40'] == len(epsilon['decisions']) == task.QUERIES,
              'ALL_ORIGINAL_NATIVE_SOLVER_AND_GRIPPER_WINDOWS_CONFIRMED')
    captures, capture_pin = load(native_root / 'TASK_LEARNING_CAPTURES.private.json')
    confirmed, command_pin = load(native_root / 'TASK_LEARNING_CONFIRMED_COMMANDS.private.json')
    task.need(captures['complete'] is True and captures['task_success'] is True
              and confirmed['task_success'] is True, 'CLOSED_EXPERT_CAPTURE_AND_COMMAND_ARCHIVES')
    decisions = []
    decision_pins = []
    frame_pins = []
    max_bias = 0.0
    for index, (descriptor, summary, capture) in enumerate(
            zip(q['teacher_decisions'], epsilon['decisions'], captures['records'], strict=True)):
        path = native_root / f'TASK_LEARNING_DECISION_{index:03d}.private.json'
        task.need(descriptor['path'] == str(path), 'ORIGINAL_NUMBERED_EPSILON_DECISION')
        decision, pin = load(path, descriptor['sha256'])
        ticket = decision['observation']
        query = ticket['query_uuid']
        frame_path = native_root / 'PAUSED_OBSERVATIONS' / query / 'PAUSED_OBSERVATION.private.json'
        task.need(Path(query).name == query and not query.startswith('.')
                  and str(frame_path) == capture['path'] == ticket['path']
                  and capture['committed'] is True and ticket['archive_state'] == 'PENDING_NOT_READABLE'
                  and ticket['sha256'] is None, 'EXACT_PENDING_TO_COMMITTED_FRAME')
        for key in ('query_uuid', 'native_step', 'native_time_s', 'state_sha256', 'task_command',
                    'memory_snapshot_sha256', 'GR00T_queries', 'actual_cube_truth_is_student_input'):
            task.need(ticket[key] == capture[key], 'IMMUTABLE_DECISION_FRAME_IDENTITY_MATCHES_ARCHIVE')
        frame, frame_pin = load(frame_path, capture['sha256'])
        body = unseal(frame)
        state = frame_path.parent / 'STATE.private.bin'
        task.need(body['query_uuid'] == query and body['native_step'] == ticket['native_step']
                  and body['native_time_s'] == ticket['native_time_s']
                  and body['world']['execution_spec_sha256'] == spec_pin['sha256']
                  and body['step_quota_sha256'] == ticket['quota_sha256']
                  and body['task_sha256'] == ticket['task_sha256']
                  and body['transport']['immutable_snapshot_sha256'] == ticket['memory_snapshot_sha256']
                  and not state.is_symlink() and body['files']['STATE.private.bin'] == ticket['state_sha256'] == digest(state),
                  'PINNED_ORIGINAL_PAUSED_STATE_BEFORE_EXPERT_DECISION')
        frame_pins.append(frame_pin)
        original = np.asarray(decision['original_teacher_policy_H40'], dtype=float)
        adjusted = np.asarray(decision['submitted_policy_H40'], dtype=float)
        record = decision['gripper_epsilon']
        task.need(original.shape == adjusted.shape == (40, 10)
                  and np.isfinite(original).all() and np.isfinite(adjusted).all()
                  and np.array_equal(original[:, :9], adjusted[:, :9])
                  and record['active_pose_coordinates_exact'] is True
                  and record['schema'] == SCHEMA and record['codec'] == VERSION and record['epsilon'] == EPSILON,
                  'EPSILON_TRANSFORM_CHANGES_ONLY_NOMINAL_GRIPPER')
        latent = label_latent(original[:, 9])
        bias = float(np.max(np.abs(adjusted[:, 9] - original[:, 9])))
        task.need(np.array_equal(record['original_nominal'], original[:, 9])
                  and np.array_equal(record['latent'], latent)
                  and np.array_equal(record['decoded_nominal'], adjusted[:, 9])
                  and np.array_equal(adjusted[:, 9], decode_latent(latent))
                  and bias == record['maximum_nominal_bias'] == summary['maximum_nominal_bias']
                  and bias <= EPSILON + np.finfo(np.float64).eps
                  and summary['native_step'] == decision['native_step'] == descriptor['native_step'],
                  'EVERY_NOMINAL_GRIPPER_KNOT_HAS_THE_EXACT_LOGIT_ROUNDTRIP')
        max_bias = max(max_bias, bias)
        decisions.append(dict(decision, _sha256=pin['sha256']))
        decision_pins.append(pin)

    checked = sequence(decisions, confirmed, [decision['observation'] for decision in decisions])
    installed = [window['actual_installed_consumed_EEF10_BASE7'][:10]
                 for cycle in q['generated_cycles'][1:-1] for window in cycle['windows']]
    task.need(np.array_equal(checked['installed_audit_only'], installed),
              'COMMAND_ARCHIVE_MATCHES_ACTUAL_INSTALLED_NATIVE_WINDOWS')
    return dict(kind='NATIVE_EXPERT', lease=lease_pin, worker=worker_pin, result=result_pin, spec=spec_pin,
                decisions=decision_pins, captures=capture_pin, frames=frame_pins, confirmed_commands=command_pin,
                native_integrations=task.E, fully_consumed_H40=epsilon['fully_consumed_H40'],
                confirmed_commands_count=checked['confirmed_intervals'], maximum_nominal_gripper_bias=max_bias)
