"""Separately scored JOIN10 consumer; raw H40 and every native gate remain.

The fixed offline method changes only the first nine installed reference knots.
It uses two completed references, never scene truth, Teacher or learned state.
"""
from pathlib import Path
import numpy as np

from . import task_learning_grasp_v1 as task
from .storage import digest
from ..trajectory.so3 import rot6d_rows_to_matrix, rotation_matrix_to_rot6d_rows
from scripts.sla.reference_join10_v71 import connect, METHOD

VERSION = 'SLA_PD4_REFERENCE_JOIN10_H40_P10_V72'
HELPER = 'scripts/sla/reference_join10_v71.py'
HELPER_SHA256 = '38358895a71ab17c9660a8884515cb3cdd9cb9be5bbcc9117296def47a5a493d'


def configuration():
    return dict(version=VERSION, method=METHOD, helper_sha256=HELPER_SHA256,
                changed_knot_indices=list(range(9)), preserved_suffix_start=9,
                label_interval_s=.05, prefix_atoms=10, teacher_fallback=False)


class ReferenceJoinProgram(task.FiftyMsProgram):
    """One fixed candidate, including unchanged raw and baseline reference."""

    def __init__(self, baseline, history, native_step):
        task.need(len(history) == 2 and
                  [r['end_native_step'] for r in history] == [native_step-25, native_step],
                  'JOIN10_TWO_COMPLETED_NATIVE_50MS_REFERENCES')
        previous = np.stack([r['pose'] for r in history])
        task.need(np.array_equal(previous[-1], baseline.initial_reference_TCP_target),
                  'JOIN10_CURRENT_REFERENCE_IS_LAST_COMPLETED_REFERENCE')
        candidate = connect(baseline.reference_path, previous)
        self.decoded_H40 = baseline.decoded_H40
        self.decision_TCP_target = baseline.decision_TCP_target
        self.initial_reference_TCP_target = baseline.initial_reference_TCP_target
        self.base_pose_hold = baseline.base_pose_hold
        self.original_reference_path = baseline.reference_path
        self.previous_reference_poses = previous.copy()
        self.history_native_steps = (native_step-25, native_step)
        desired = candidate.copy()
        actual, reference = self.decision_TCP_target, self.initial_reference_TCP_target
        desired[:, :3] += actual[:3, 3]-reference[:3, 3]
        desired[:, 3:9] = rotation_matrix_to_rot6d_rows(
            actual[:3, :3] @ reference[:3, :3].T @ rot6d_rows_to_matrix(candidate[:, 3:9]))
        self.desired_path, self.reference_path = desired, candidate
        # Same original full-H40 EEF limits, in both actual and reference frames.
        for value in (desired, candidate, self.previous_reference_poses):
            value.setflags(write=False)
        try:
            self.validate_envelope()
        except ValueError as error:
            error.reference_connection_evidence=self.evidence()
            raise

    def evidence(self):
        return dict(super().evidence(), schema=VERSION+'_TRANSPORT',
                    consumer_version=VERSION, method=METHOD,
                    reference_transport='ORIGINAL_TRANSPORT_THEN_FIXED_JOIN10',
                    installed_reference_changed=True,
                    original_reference_H40=self.original_reference_path.tolist(),
                    candidate_reference_H40=self.reference_path.tolist(),
                    previous_reference_poses=self.previous_reference_poses.tolist(),
                    history_native_steps=list(self.history_native_steps),
                    suffix_k9_through39_byte_exact=bool(np.array_equal(
                        self.reference_path[9:], self.original_reference_path[9:])),
                    grip_byte_exact=bool(np.array_equal(
                        self.reference_path[:, 9], self.original_reference_path[:, 9])),
                    maximum_translation_change_m=float(np.linalg.norm(
                        self.reference_path[:, :3]-self.original_reference_path[:, :3], axis=1).max()),
                    expert_training_eligible=False)


def compile_proposal(action, pin, *, reference_history, query_native_step, **kwargs):
    from .sla_dense_policy_v5 import compile_proposal as raw_compile
    from .task_learning_student_v1 import StudentProposalRejected
    # Reject the original invalid raw proposal before adaptation. No rescue of
    # an original EEF failure and no silent fallback after candidate rejection.
    baseline, proposal = raw_compile(action, pin, **kwargs)
    # A missing/stale producer history is an engineering failure, not a valid
    # generated-action rejection. Verify it outside the policy rejection path.
    task.need(type(query_native_step)is int and len(reference_history)==2 and
              [r['end_native_step']for r in reference_history]==[query_native_step-25,query_native_step]
              and np.array_equal(reference_history[-1]['pose'],baseline.initial_reference_TCP_target),
              'JOIN10_EXACT_COMPLETED_REFERENCE_HISTORY')
    try:
        program = ReferenceJoinProgram(baseline, reference_history, query_native_step)
    except ValueError as error:
        rejected=StudentProposalRejected(
            'TASK_LEARNING_STUDENT_FULL_PROPOSAL_REJECTED: JOIN10_FULL_H40_GATE:'+str(error))
        rejected.reference_connection_evidence=getattr(error,'reference_connection_evidence',None)
        raise rejected from error
    proposal.update(consumer_version=VERSION, reference_connection=configuration(),
                    original_raw_H40_unchanged=True,
                    original_H40_interior_preserved=False,
                    installed_reference_changed=True,
                    reference_connection_evidence=program.evidence())
    return program, proposal


def source_members():
    return ('src/go2arx5_vla/world_model/sla_reference_join_v72.py',
            'src/go2arx5_vla/world_model/pd4_join_v72.py', HELPER)


def source_custody(spec):
    if spec.get('reference_join10_v72') != configuration():
        raise PermissionError('EXACT_FIXED_JOIN10_METHOD_REQUIRED')
    source = Path(spec['source_root'])
    for name in source_members():
        if digest(source/name) != spec['source_files'].get(name):
            raise PermissionError('SEALED_JOIN10_SOURCE:'+name)
    if digest(source/HELPER) != HELPER_SHA256:
        raise PermissionError('ORIGINAL_PREDECLARED_JOIN10_HELPER_BYTES')
