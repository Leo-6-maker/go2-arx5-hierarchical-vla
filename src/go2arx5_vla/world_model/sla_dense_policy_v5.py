"""Compile decoded dense SLA H40 without replacing its interior trajectory.

This is the raw-action seam for the separately registered P10 native pilot.
It grants no scene, owner, model or physics admission on its own. The existing
FiftyMsProgram still checks all forty knots before a ten-atom prefix can run.
"""
import numpy as np

from . import task_learning_grasp_v1 as task
from .storage import digest

VERSION = 'SLA_DENSE_RAW_H40_NATIVE_50MS_P10_V5'
PREFIX_ATOMS = 10


def compile_proposal(action, pin, *, source_from_target, query_state10,
                     actual_TCP, consumed_reference_TCP, base_hold):
    from scripts.pi0292.eval_contract import matrix10
    from scripts.pi0291_arx5_data.native_dataset import pose10
    from .fusion_observation import rigid
    from .task_learning_student_v1 import StudentProposalRejected

    task.need(digest(action) == pin, 'SLA_DENSE_ORIGINAL_RESPONSE_BYTES')
    with np.load(action, allow_pickle=False) as packet:
        raw = packet['physical_H40'].copy()
        normalized = packet['normalized']
        task.need(raw.shape == (40, 10) and normalized.shape == (1, 40, 32)
                  and np.isfinite(raw).all() and np.isfinite(normalized).all(),
                  'SLA_DENSE_FINITE_DECODED_H40_AND_NATIVE32_CONTAINER')
    frame = rigid(source_from_target)
    task.need(np.allclose(matrix10(query_state10), frame @ np.asarray(actual_TCP),
                          rtol=0, atol=1e-9), 'SLA_DENSE_SAME_CAPTURED_QUERY_AND_ACTUAL_TCP')
    try:
        # The provider has already decoded logit grip and task-frame actions.
        # Change coordinates once. Do not apply sigmoid, an endpoint basis,
        # expert segment generation, clipping, or timing changes here.
        target = pose10(np.linalg.inv(frame) @ matrix10(raw), raw[:, 9])
        program = task.FiftyMsProgram(target, actual_TCP, consumed_reference_TCP, base_hold)
    except ValueError as error:
        raise StudentProposalRejected(
            'TASK_LEARNING_STUDENT_FULL_PROPOSAL_REJECTED: SLA_DENSE_ORIGINAL_FULL_H40_GATE:'
            + str(error)) from error
    return program, dict(
        raw_policy_H40=raw.tolist(), mapped_target_H40=target.tolist(),
        action_sha256=pin, SLA_codec_decode_calls=1, policy_frame_mapping_count=1,
        normalized_container_dimensions=32, consumer_version=VERSION,
        requested_prefix_atoms=PREFIX_ATOMS, original_H40_interior_preserved=True,
        endpoint_basis_applied=False, expert_segment_generator_applied=False,
        additional_gripper_decode_calls=0, command_clips=0, teacher_fallback=False,
        expert_training_eligible=False, SLA_raw_H40_consumed=False,
        native_execution_authorized=False,
        raw_install_and_consumption_must_be_reported_separately=True)
