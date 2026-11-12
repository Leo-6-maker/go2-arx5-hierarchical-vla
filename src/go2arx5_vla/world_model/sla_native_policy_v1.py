"""SLA 32-wide response through the existing endpoint basis and native H40 gate."""
from pathlib import Path
import numpy as np
from . import task_learning_grasp_v1 as task
from .storage import digest, write_json


def capture_task_frame(adapter, initial):
    """Own the measured post-hold mount frame once, without moving the clock."""
    from .fusion_observation import rigid
    sample = adapter.actual_state
    need = task.need
    need(initial['native_step'] == task.B and adapter.observe()['step'] == task.B
         and sample.data.time == initial['time_s']
         and np.array_equal(sample.data.qpos, initial['qpos']), 'SLA_ACTUAL_POST_HOLD_FRAME')
    mount = sample.model.body('x5_base_link').id
    need(sample.model.body_parentid[mount] == sample.model.body('base_link').id
         and sample.model.body_jntnum[mount] == 0, 'SLA_FIXED_CALIBRATED_ARX_MOUNT')
    transform = np.eye(4)
    transform[:3, :3] = sample.data.xmat[mount].reshape(3, 3)
    transform[:3, 3] = sample.data.xpos[mount]
    policy_from_world = rigid(adapter.spec['task_phase_source_from_target'])
    path = adapter.output/'SLA_TASK_FRAME.private.json'
    need(not path.exists(), 'SLA_TASK_FRAME_CAPTURED_EXACTLY_ONCE')
    write_json(path, dict(schema='SLA_EPISODE_FIXED_NATIVE_TASK_FRAME_V1',
        T_policy_from_mount=rigid(policy_from_world@transform).tolist(),
        native_step=task.B, native_time_s=initial['time_s'],
        native_spec_sha256=digest(Path(adapter.spec['registered_root'])/'EXECUTION_SPEC.json'),
        scene_sha256=digest(adapter.spec['model_path']), mount_body='x5_base_link',
        fixed_for_episode=True, object_truth_used=False, additional_TCP_offset_m=0., native_integrations=0))
    return dict(path=str(path), sha256=digest(path))


def compile_proposal(action, pin, *, consumed_nominal_grip, source_from_target,
                     query_state10, actual_TCP, consumed_reference_TCP, base_hold):
    from .task_learning_student_v1 import StudentProposalRejected
    from .task_learning_endpoint_basis_v3 import decode, VERSION
    from .fusion_observation import rigid
    from scripts.pi0292.eval_contract import matrix10
    from scripts.pi0291_arx5_data.native_dataset import pose10
    task.need(digest(action) == pin, 'SLA_ORIGINAL_RESPONSE_BYTES')
    with np.load(action, allow_pickle=False) as packet:
        raw = packet['physical_H40'].copy()
        normalized = packet['normalized']
        task.need(raw.shape == (40, 10) and normalized.shape == (1, 40, 32)
                  and np.isfinite(raw).all() and np.isfinite(normalized).all(),
                  'SLA_COMPLETE_FINITE_H40_AND_NATIVE32_CONTAINER')
    frame = rigid(source_from_target)
    task.need(np.allclose(matrix10(query_state10), frame@np.asarray(actual_TCP), rtol=0, atol=1e-9),
              'SLA_SAME_POLICY_QUERY_AND_NATIVE_TCP')
    try:
        decoded, audit = decode(raw, np.asarray(query_state10), consumed_nominal_grip=consumed_nominal_grip)
        target = pose10(np.linalg.inv(frame)@matrix10(decoded), decoded[:, 9])
        program = task.FiftyMsProgram(target, actual_TCP, consumed_reference_TCP, base_hold)
    except ValueError as exc:
        raise StudentProposalRejected('TASK_LEARNING_STUDENT_FULL_PROPOSAL_REJECTED: SLA_ORIGINAL_FULL_H40_GATE:'+str(exc)) from exc
    return program, dict(raw_policy_H40=raw.tolist(), explicit_endpoint_policy_H40=decoded.tolist(),
        mapped_target_H40=target.tolist(), SLA_codec_decode_calls=1, action_sha256=pin,
        policy_frame_mapping_count=1, command_clips=0, teacher_fallback=False, expert_training_eligible=False,
        composed_policy_method='SLA_TARGET500_PLUS_ORIGINAL_ENDPOINT_BASIS',
        explicit_decoder_applied=True, explicit_decoder_execution_version=VERSION,
        representation_audit=audit, SLA_raw_H40_consumed=False, normalized_container_dimensions=32,
        composed_H40_requires_original_native_reference_and_consumption_audits=True)
