"""Unified24-request evidence around two unchanged, qualified controllers.

No delegated trajectory, recovery decision, native contact gate or score is
changed. This is a new qualification identity, never a continuation of V8.
"""
from . import task_learning_grasp_v1 as task
from . import task_learning_teacher_v8 as v8
from . import task_learning_teacher_e1_v9 as small
from .storage import write_json

VERSION='TEACHER_V9_REPAIRED_E1_RECOVERY24_UNCHANGED_ORACLE_V1'
ROLE='TEACHER_V9_FULL_RECOVERY24_QUALIFICATION_NOT_BC_V1'
TYPES=v8.TYPES
INTERVENTIONS={'UNDER_CLOSE':(0.,.55),**small.INTERVENTIONS}


class RecoveryUnavailable(ValueError):
    def __init__(self,reason,evidence):
        self.reason,self.evidence=reason,evidence
        super().__init__('TEACHER_V9_FULL24_RECOVERY_UNAVAILABLE:'+reason)


def contract():
    return dict(v8.contract(),stage=ROLE,task_revision=VERSION,recovery_Oracle=v8.VERSION,
        requested_conditions=24,interventions={k:dict(tool_Y_offset_m=v[0],ending_nominal_grip=v[1])
            for k,v in INTERVENTIONS.items()},
        controller_delegation='UNCHANGED_V8_UNDER_CLOSE_AND_QUALIFIED_V9_SMALL4_SIDE_SINGLE',
        E1_E2_E3_separate=True,actual_force_contact_threshold_N=.05,
        actual_side_entry_threshold_m=.002,automatic_BC_or_DAgger=False)


def entry_condition(kind,witness):
    task.need(kind in TYPES,'FULL24_DECLARED_RECOVERY_CONDITION')
    if kind=='UNDER_CLOSE':
        return bool(witness['supported']and witness['native_grasp']is False
            and witness['actual_mechanical_grip_fraction']>=.5)
    return small.entry_condition(kind,witness)


class NativeQualificationTeacher:
    def __init__(self,adapter,initial,history):
        self.adapter,self.history=adapter,history;self.kind=adapter.spec['teacher_full24_recovery_condition']
        sign=adapter.spec['teacher_full24_offset_sign']
        task.need(self.kind in TYPES and sign in(-1,1)
            and adapter.spec['teacher_v8_recovery_condition']==adapter.spec['teacher_v9_recovery_condition']==self.kind
            and adapter.spec['teacher_v8_offset_sign']==adapter.spec['teacher_v9_offset_sign']==sign,
            'FULL24_SAME_FROZEN_DELEGATE_CONDITION_AND_SIGN')
        factory=v8.NativeQualificationTeacher if self.kind=='UNDER_CLOSE'else small.NativeQualificationTeacher
        self.delegate=factory(adapter,initial,history)
        self.stage,self.cube=self.delegate.stage,self.delegate.cube
        self.entry=None;self.recovered=None

    @property
    def intervention_issued(self):return self.delegate.intervention_issued

    @property
    def native_grasp_after_correction(self):return self.recovered is not None

    def evidence(self):
        return dict(Teacher_version=VERSION,recovery_Oracle=v8.VERSION,recovery_condition=self.kind,
            intervention_issued=self.intervention_issued,E1=self.entry,E2=self.recovered,
            actual_recovery_entry_observed=bool(self.entry and self.entry['E1_condition_established']),
            native_grasp_after_correction=self.native_grasp_after_correction,
            delegate_evidence=self.delegate.evidence(),new_Oracle_not_student_DAgger=True)

    def decide(self,sample,reference,base):
        awaiting=self.delegate.awaiting
        if awaiting or(self.entry and self.entry['E1_condition_established']and not self.recovered
            and sample['native_step']>self.entry['native_step']and sample['native_grasp']is True):
            cycle,previous=self.history();consumed=v8.command_context(sample,cycle,previous)
            witness=dict(small.current_witness(self.adapter,sample,self.delegate.oracle,consumed),
                Teacher_version=VERSION,recovery_condition=self.kind)
            if awaiting:
                self.entry=dict(witness,E1_condition_established=entry_condition(self.kind,witness))
                write_json(self.adapter.output/'TEACHER_FULL24_E1_ENTRY.private.json',self.entry)
            else:
                task.need(witness['native_grasp']and witness['opposing_contact']and witness['relative_motion_valid'],
                    'FULL24_ORIGINAL_ACTUAL_RECOVERED_GRASP')
                self.recovered=dict(witness,E1_entry_native_step=self.entry['native_step'],E2_actual_recovered=True)
                write_json(self.adapter.output/'TEACHER_FULL24_E2_RECOVERED.private.json',self.recovered)
        try:program,proposal=self.delegate.decide(sample,reference,base)
        except(v8.RecoveryUnavailable,small.RecoveryUnavailable)as error:
            event=dict(self.evidence(),reason=error.reason,actual=error.evidence)
            write_json(self.adapter.output/'TEACHER_FULL24_RECOVERY_EXIT.private.json',event)
            raise RecoveryUnavailable(error.reason,event)from error
        self.stage=self.delegate.stage
        proposal.update(Teacher_version=VERSION,qualified_delegate_Teacher_version=(v8.VERSION
            if self.kind=='UNDER_CLOSE'else small.VERSION),teacher_full24_recovery=self.evidence(),
            delegated_command_changed=False,expert_training_eligible=False)
        return program,proposal
