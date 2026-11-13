"""Verify and encode consumed dense TRAIN windows from a closed native episode."""
from pathlib import Path

import numpy as np

from scripts.pi0291_arx5_data.native_dataset import matrix_from10, pose10
from scripts.pi0292.relay_protocol import unseal
from ..world_model.corelab_expert_semantics_v3 import ROLE
from ..world_model.fusion_observation import shadow_inference_payload
from ..world_model.native_body_memory_v3 import DENSE_VERSION, VERSION as MEMORY_VERSION, nominal_windows
from ..world_model.stable_task_dataset_v1 import load
from ..world_model.storage import digest
from ..world_model.task_learning_grasp_v1 import B, need
from .calibration_v1 import captured_mount_frame
from .codec_v1 import TOLERANCES, pad32, rigid, roundtrip_error
from .gripper_v2 import EPSILON, VERSION, GripLogitCodec


def _anchor(native_root, record, spec_pin, policy_from_world):
    query = record['query_uuid']
    path = native_root / 'PAUSED_OBSERVATIONS' / query / 'PAUSED_OBSERVATION.private.json'
    need(Path(query).name == query and not query.startswith('.') and record['path'] == str(path)
         and record['committed'] is True, 'OWN_COMMITTED_DENSE_FRAME')
    frame, pin = load(path, record['sha256'])
    body = unseal(frame)
    shadow = unseal(body['original_shadow_ABI'])
    target = shadow['target_observation']
    state = np.asarray(shadow_inference_payload(body['original_shadow_ABI'])['state_eef10'], dtype=float)
    need(body['query_uuid'] == query and body['native_step'] == record['native_step'] == shadow['native_step']
         and body['native_time_s'] == record['native_time_s']
         and body['world']['execution_spec_sha256'] == target['target_runtime_sha256'] == spec_pin['sha256']
         and body['transport']['immutable_snapshot_sha256'] == record['memory_snapshot_sha256']
         and body['files']['STATE.private.bin'] == target['native_state_sha256'] == record['state_sha256']
         and digest(path.parent / 'STATE.private.bin') == record['state_sha256']
         and np.array_equal(shadow['source_policy_view']['T_source_from_target'], policy_from_world),
         'ORIGINAL_CURRENT_NATIVE_STATE_AND_POLICY_FRAME')
    return state, dict(pin, native_step=record['native_step'])


def encode_episode(root, *, expected_result_sha256, expected_archive_sha256):
    """Encode a sealed engineering episode for audit. Returns arrays and provenance."""
    root = Path(root)
    native_root = root / 'native-01'
    spec, spec_pin = load(root / 'EXECUTION_SPEC.json')
    lease, _ = load(root / 'control/LEASE.json')
    worker, _ = load(root / 'WORKER_EXIT.private.json')
    result, result_pin = load(native_root / 'RESULT.json', expected_result_sha256)
    archive, archive_pin = load(native_root / 'BODY_MEMORY_FRAME_ARCHIVE.private.json', expected_archive_sha256)
    q = result['reference_qualification']
    need(spec['registered_root'] == str(root) and spec['task_learning_role'] == ROLE
         and spec['expert_dense_observation_v3']['version'] == DENSE_VERSION
         and lease['status'] == 'CLOSED' and lease['spec_sha256'] == spec_pin['sha256']
         and lease['worker_exit_code'] == worker['exit_code'] == 0 and worker['failure'] is None
         and spec['source_sha'] == result['source_sha'] == archive['source_sha']
         and result['status'] == 'NATIVE_STAGE_PASSED' and result['shutdown_clean'] is True
         and q['task_condition_success'] is True and q['memory_frame_archive'] == archive_pin
         and archive['schema'] == MEMORY_VERSION and archive['recording_complete'] is True
         and archive['frames_committed'] is True and archive['BC_eligible'] is False,
         'CLOSED_DENSE_TRAIN_ENGINEERING_EPISODE')
    from ..world_model.task_learning_grasp_native_v1 import complete_consumption
    need(complete_consumption(spec, q), 'COMPLETE_ORIGINAL_NATIVE_CONSUMPTION')
    confirmed, command_pin = load(native_root / 'TASK_LEARNING_CONFIRMED_COMMANDS.private.json')
    need(confirmed['complete'] is True and confirmed['task_success'] is True
         and confirmed['unexecuted_suffix_included'] is False and confirmed['future_achieved_labels'] is False,
         'ACTUALLY_CONSUMED_COMMANDS_ONLY')
    windows_by_step = {row['native_start_step']: row for row in q['windows']}
    for command in confirmed['records']:
        native = windows_by_step[command['native_start_step']]
        need(command['native_end_step'] == native['native_end_step']
             and native['gripper']['command_consumed'] is True
             and bool(native['solver_policies_with_exact_installed_target'])
             and np.array_equal(command['installed_consumed_EEF10'],
                                native['actual_installed_consumed_EEF10_BASE7'][:10]),
             'LABEL_COMMAND_JOINS_ACTUALLY_INSTALLED_NATIVE_WINDOW')
    records = archive['records']
    windows = nominal_windows(records, confirmed['records'])
    need(windows == archive['nominal_windows'], 'EVERY_ARCHIVED_WINDOW_JOINS_ORIGINAL_CONSUMED_COMMANDS')
    prefix_atoms = spec['task_learning_grasp']['prefix_atoms']
    decisions = {row['sha256']: (row['native_step'], prefix_atoms) for row in q['teacher_decisions']}
    tail = q.get('recorded_expert_success_tail')
    if tail:
        tail_record, tail_pin = load(native_root / 'TASK_LEARNING_TERMINATION_HOLD.private.json', tail['command_sha256'])
        need(tail_record['passes'] is tail['passes'] is True and tail['consumed_commands'] == 40
             and tail['end_step'] - tail['start_step'] == 1000
             and tail_record['score'] == tail['score'] and tail['score']['longest_contiguous_grasp_lift_hold_s'] >= 2.,
             'ACTUALLY_RECORDED_CONTINUOUS_TWO_SECOND_SUCCESS_TAIL')
        decisions[tail_pin['sha256']] = (tail['start_step'], tail['consumed_commands'])
    policy_from_world = rigid(spec['task_phase_source_from_target'])
    first = records[0]
    first_state, first_pin = _anchor(native_root, first, spec_pin, policy_from_world)
    need(first['native_step'] == B, 'ORIGINAL_POST_HOLD_MOUNT_FRAME_ANCHOR')
    _, readiness = load(native_root / 'TASK_LEARNING_START_READINESS.private.json')
    mount = captured_mount_frame(spec_pin, readiness, first_pin, policy_from_world, first_state)
    codec = GripLogitCodec(mount['T_policy_from_mount'])

    states, actions, indices = [], [], []
    errors = dict(xyz_m=0.0, so3_rad=0.0, gripper=0.0)
    cross_replan = 0
    for window, record in zip(windows, (row for row in records if row['dense_anchor']), strict=True):
        if not window['complete']:
            continue
        need(window['query_uuid'] == record['query_uuid'] and window['native_step'] == record['native_step'],
             'SAME_ANCHOR_AS_CURRENT_OBSERVATION')
        state, _ = _anchor(native_root, record, spec_pin, policy_from_world)
        for tick, decision in zip(window['native_start_steps'], window['Teacher_decision_sha256'], strict=True):
            need(decision in decisions, 'PINNED_ISSUING_QUERY_FOR_EACH_NOMINAL_KNOT')
            issue_step, consumed_atoms = decisions[decision]
            need(0 <= tick - issue_step < consumed_atoms * 25,
                 'NOMINAL_WORLD_KNOT_COMES_FROM_ITS_ISSUING_QUERY')
        world = np.asarray(window['nominal_policy_EEF10_target_world'], dtype=float)
        # Issuing expert knots already share target-world coordinates. Move
        # the whole verified window to the policy frame and encode at this anchor.
        nominal = pose10(policy_from_world @ matrix_from10(world), world[:, 9])
        state7, latent7 = codec.encode(state, nominal)
        error = roundtrip_error(nominal, codec.decode(state, pad32(latent7)))
        need(error['xyz_m'] <= TOLERANCES['xyz_m'] and error['so3_rad'] <= TOLERANCES['so3_rad']
             and error['gripper'] <= EPSILON + np.finfo(np.float64).eps and state7[6] == state[9],
             'DENSE_EEF_LOGIT_ROUNDTRIP_AT_REAL_ANCHOR')
        for key, value in error.items():
            errors[key] = max(errors[key], value)
        cross_replan += len(set(window['Teacher_decision_sha256'])) > 1
        states.append(state7)
        actions.append(latent7)
        indices.append(record['native_step'])
    need(len(indices) == archive['complete_consumption_joined_nominal_windows'] > 0,
         'COMPLETE_DENSE_WINDOW_DENOMINATOR')
    arrays = dict(state7=np.stack(states), nominal_latent7=np.stack(actions), native_step=np.asarray(indices))
    return arrays, dict(schema='SLA_DENSE_NATIVE_EEF_CODEC_AUDIT_V3', codec=VERSION, passes=True,
                        spec=spec_pin, result=result_pin, archive=archive_pin, commands=command_pin,
                        mount=mount, encoded_windows=len(indices), cross_replan_windows=cross_replan,
                        excluded_incomplete_windows=len(windows) - len(indices), max_roundtrip_error=errors,
                        actual_two_second_success_tail=bool(tail),
                        additional_physical_requests=0, additional_native_integrations=0,
                        optimizer_updates=0, optimization_visits=0, model_calls=0, BC_examples_admitted=0)
