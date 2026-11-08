"""One declared under-close intervention, followed by real Teacher feedback.

The intervention intentionally stops short of closing. It is not an expert
label. Only a later real observation can establish whether a correction was
needed and whether native grasp subsequently recovered. No slip capability is
claimed and no object pose, contact, dynamics or student input is changed.
"""
import numpy as np
from . import task_learning_grasp_v1 as task

NORMAL='NORMAL_TEACHER_V7'
UNDER_CLOSE='UNDER_CLOSE_INTERVENTION_THEN_TEACHER_CORRECTION_V1'
EXPERT='EXECUTED_TEACHER_INTENT'
INTERVENTION='INTERVENTION_NOT_EXPERT'


def contract():
    return dict(schema='TASK_LEARNING_UNDER_CLOSE_RECOVERY_CONTRACT_V1',
        intervention='FIRST_REAL_CLOSE_DECISION_ONLY',partial_close_nominal_grip=.55,
        partial_close_is_expert_label=False,subsequent_correction='ORIGINAL_TEACHER_ON_NEW_REAL_OBSERVATION',
        actual_failed_grasp_required_for_recovery_count=True,
        successful_native_grasp_after_correction_required=True,
        complete_original_task_success_required=True,changes_to_student_inputs=False,
        native_runtime_qualified=False,slip_or_release_recovery_proven=False)


class UnderCloseRecoveryTeacher(task.Teacher):
    def __init__(self,initial):
        super().__init__(initial)
        self.intervention_issued=False
        self.awaiting_intervention_observation=False
        self.awaiting_correction_observation=False
        self.actual_under_close_failed=False
        self.native_grasp_after_correction=False

    def decide(self,sample,reference,base):
        # These events come only from the newly measured input at this call.
        observed_intervention=self.awaiting_intervention_observation
        if observed_intervention:
            self.actual_under_close_failed=sample.get('native_grasp')is False
            self.awaiting_intervention_observation=False
        if self.awaiting_correction_observation and sample.get('native_grasp')is True:
            self.native_grasp_after_correction=True
            self.awaiting_correction_observation=False
        previous_grip=self.grip
        program,proposal=super().decide(sample,reference,base)
        role=EXPERT
        if self.stage=='CLOSE'and not self.intervention_issued:
            raw=program.decoded_H40.copy();u=np.linspace(0.,1.,40);w=u**3*(10+u*(-15+6*u))
            self.grip=.55;raw[:,9]=previous_grip+w*(self.grip-previous_grip)
            program=task.FiftyMsProgram(raw,sample['tcp'],reference,base)
            proposal.update(submitted_policy_H40=raw.tolist())
            self.intervention_issued=True;self.awaiting_intervention_observation=True;role=INTERVENTION
        elif observed_intervention and self.actual_under_close_failed and self.stage=='CLOSE':
            self.awaiting_correction_observation=True
        proposal.update(label_role=role,recovery_protocol=UNDER_CLOSE,
            recovery_evidence=dict(actual_under_close_failed=self.actual_under_close_failed,
                native_grasp_after_correction=self.native_grasp_after_correction,
                newly_observed_intervention_outcome=observed_intervention),
            intervention_is_expert_label=False)
        return program,proposal
