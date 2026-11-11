"""Four independent intervention-establishment trials; never training data.

Only the declared intervention changes. The actual V8 recovery Oracle, full
H40 consumer, contact predicates, original score and safety remain in use.
"""
import numpy as np
from . import task_learning_grasp_v1 as task
from . import task_learning_teacher_v8 as v8
from .fusion_trajectory_adapter import _geometry
from .storage import write_json

VERSION='TEACHER_V9_E1_ESTABLISHMENT_THEN_UNCHANGED_V8_ORACLE_V1'
ROLE='TEACHER_V9_E1_SMALL4_QUALIFICATION_NOT_BC_V1'
TYPES=('SUPPORTED_SIDE_OFFSET','SUPPORTED_SINGLE_PAD')
INTERVENTIONS={'SUPPORTED_SIDE_OFFSET':(.0045,.8),'SUPPORTED_SINGLE_PAD':(.0045,.5)}


class RecoveryUnavailable(ValueError):
    def __init__(self,reason,evidence):
        self.reason,self.evidence=reason,evidence
        super().__init__('TEACHER_V9_RECOVERY_UNAVAILABLE:'+reason)


def contract():
    return dict(v8.contract(),stage=ROLE,task_revision=VERSION,
        recovery_Oracle=v8.VERSION,requested_conditions=4,
        intervention_tool_Y_offset_m=.0045,
        intervention_nominal_grip={'SUPPORTED_SIDE_OFFSET':.8,'SUPPORTED_SINGLE_PAD':.5},
        E1='SUPPORTED_NON_GRASP_AND_ACTUAL_SIDE_OFFSET_GE_2MM_OR_EXACTLY_ONE_PAD_GE_0.05N',
        E2='LATER_ACTUAL_NATIVE_BILATERAL_OPPOSING_FORCE_AND_RELATIVE_MOTION_GRASP',
        E3='ORIGINAL_COMPLETE_GRASP_LIFT20MM_HOLD1S_STANDING_AND_CONSUMPTION',
        automatic_new24_or_BC_or_DAgger=False)


def entry_condition(kind,witness):
    """Use actual original force/contact observations, never geometry overlap."""
    task.need(kind in TYPES,'V9_DECLARED_E1_CONDITION')
    forces=witness['pad_normal_force_N']
    left,right=(float(forces[side])>=.05 for side in ('left','right'))
    task.need(left==witness['left_object_contact']and right==witness['right_object_contact'],
        'V9_CONTACT_FLAGS_MATCH_ORIGINAL_PACKET_FORCE_THRESHOLD')
    return bool(witness['supported']and witness['native_grasp']is False and (
        abs(witness['actual_goal_tool_Y_offset_m'])>=.002 if kind==TYPES[0]else left!=right))


def current_witness(adapter,sample,oracle,consumed):
    """Record native forces and cloned FK gaps at the same paused input clock."""
    import mujoco
    packet=adapter.observe();reader=adapter.actual_state
    task.need(packet['step']==sample['native_step']and abs(packet['time_s']-sample['time_s'])<=1e-9,
        'V9_CURRENT_NATIVE_CONTACT_PACKET_CLOCK')
    task.need(np.array_equal(reader.data.qpos,sample['qpos'])and reader.data.time==sample['time_s'],
        'V9_CLONED_GEOMETRY_IS_CURRENT_NATIVE_QPOS')
    table=reader.model.geom('task_table').id;cube=reader.cube_geom
    forces={'left':0.,'right':0.};support=0.;contacts=[];gaps={}
    for c in packet['contacts']:
        pair={c['geom0'],c['geom1']}
        if cube not in pair:continue
        force=float(c['wrench'][0])
        for side,pad in zip(('left','right'),reader.pad):
            if pair=={pad,cube}:forces[side]=max(forces[side],force)
        if pair=={table,cube}:support=max(support,force)
        contacts.append(dict(geoms=[reader.model.geom(g).name for g in (c['geom0'],c['geom1'])],
            normal_force_N=force,distance_m=float(c['distance_m']),normal=np.asarray(c['normal']).tolist()))
    for side,pad in zip(('left','right'),reader.pad):
        line=np.zeros(6)
        gaps[side]=float(mujoco.mj_geomDistance(reader.model,reader.data,pad,cube,.25,line))
    axis=oracle.target_rotation(sample)[:,1]
    return dict(native_step=sample['native_step'],time_s=sample['time_s'],contact_time_s=sample['contact_time_s'],
        supported=bool(support>=.05),table_normal_force_N=support,native_grasp=sample['native_grasp'],
        left_object_contact=sample['left_object_contact'],right_object_contact=sample['right_object_contact'],
        opposing_contact=sample['opposing_contact'],relative_motion_valid=sample['relative_motion_valid'],
        actual_goal_tool_Y_offset_m=float((sample['tcp'][:3,3]-oracle.grasp_goal(sample))@axis),
        pad_normal_force_N=forces,signed_pad_cube_gap_m=gaps,actual_cube_contacts=contacts,
        actual_TCP_position_m=sample['tcp'][:3,3].tolist(),actual_cube_position_m=sample['object_pose'][:3,3].tolist(),
        object_linear_velocity_m_s=np.asarray(sample['object_linear_velocity']).tolist(),
        actual_mechanical_grip_fraction=float(sample['gripper_fraction']),
        actual_finger_q_m=np.asarray(sample['finger_q']).tolist(),
        previous_consumed_nominal_grip=v8.consumed_grip(sample,consumed),
        previous_complete_H40_solver_gripper_consumption=True,
        geometry_overlap_is_contact_qualification=False)


class NativeQualificationTeacher:
    def __init__(self,adapter,initial,history):
        self.adapter,self.history=adapter,history;self.oracle=v8.Teacher(initial)
        self.mode=adapter.spec['teacher_v9_recovery_condition'];self.sign=adapter.spec['teacher_v9_offset_sign']
        task.need(self.mode in TYPES and self.sign in(-1,1),'V9_DECLARED_INTERVENTION')
        self.stage=self.oracle.stage;self.cube=self.oracle.cube
        self.intervention_issued=False;self.awaiting=False;self.entry_observed=False
        self.entry_step=None;self.entry=None;self.before=None;self.native_grasp_after_correction=False

    def evidence(self):
        return dict(Teacher_version=VERSION,recovery_Oracle=v8.VERSION,recovery_condition=self.mode,
            intervention_issued=self.intervention_issued,actual_recovery_entry_observed=self.entry_observed,
            recovery_entry_native_step=self.entry_step,native_grasp_after_correction=self.native_grasp_after_correction,
            E1=self.entry,E2_requires_E1=True,E3_is_independent_original_score=True,
            new_Oracle_not_student_DAgger=True)

    def unavailable(self,reason,witness):
        event=dict(self.evidence(),reason=reason,actual=witness)
        write_json(self.adapter.output/'TEACHER_V9_RECOVERY_EXIT.private.json',event)
        raise RecoveryUnavailable(reason,event)

    def decide(self,sample,reference,base):
        cycle,previous=self.history();consumed=v8.command_context(sample,cycle,previous)
        witness=dict(current_witness(self.adapter,sample,self.oracle,consumed),
            Teacher_version=VERSION,recovery_condition=self.mode)
        if self.awaiting:
            task.need(sample['native_step']==self.before['native_step']+1000,
                'V9_INTERVENTION_FULL40_AT_ORIGINAL_NEXT_DECISION')
            self.entry=dict(witness,E1_condition_established=entry_condition(self.mode,witness),
                intervention_native_start_step=self.before['native_step'],
                actual_TCP_motion_since_intervention_m=(np.asarray(witness['actual_TCP_position_m'])-
                    self.before['actual_TCP_position_m']).tolist(),
                actual_cube_motion_since_intervention_m=(np.asarray(witness['actual_cube_position_m'])-
                    self.before['actual_cube_position_m']).tolist())
            write_json(self.adapter.output/'TEACHER_V9_E1_ENTRY.private.json',self.entry)
            if not self.entry['E1_condition_established']:
                self.unavailable('DECLARED_RECOVERY_CONDITION_NOT_ESTABLISHED',self.entry)
            self.entry_observed=True;self.entry_step=sample['native_step'];self.awaiting=False
            self.oracle.recovery_entry=True
        if self.entry_observed and sample['native_step']>self.entry_step and sample['native_grasp']is True:
            if not self.native_grasp_after_correction:
                write_json(self.adapter.output/'TEACHER_V9_E2_RECOVERED.private.json',
                    dict(witness,E1_entry_native_step=self.entry_step,E2_actual_recovered=True))
            self.native_grasp_after_correction=True
        try:
            program,proposal=self.oracle.decide(sample,reference,base,consumed_reference=consumed,
                supported=witness['supported'])
        except v8.RecoveryUnavailable as error:self.unavailable(error.reason,error.evidence)
        if self.oracle.stage=='CLOSE'and not self.intervention_issued:
            task.need(witness['supported']and sample['native_grasp']is False,
                'V9_INTERVENTION_ONLY_ACTUAL_SUPPORTED_NON_GRASP')
            raw=program.decoded_H40.copy();intent=raw.tolist();offset,ending=INTERVENTIONS[self.mode]
            u=np.linspace(0.,1.,40);w=u**3*(10+u*(-15+6*u))
            raw[:,9]=consumed['policy_EEF10'][9]+w*(ending-consumed['policy_EEF10'][9])
            raw[:,:3]+=w[:,None]*(self.sign*offset*self.oracle.target_rotation(sample)[:,1])
            program=task.FiftyMsProgram(raw,sample['tcp'],reference,base)
            metrics=_geometry(program.desired_path,sample['tcp'])
            task.need(metrics['maximum_finger_speed_m_s']<=.03*(1+1e-10),
                'V9_INTERVENTION_RETAINS_ORIGINAL_TEACHER_SPEED_MARGIN')
            proposal.update(submitted_policy_H40=raw.tolist(),label_role='INTERVENTION_NOT_EXPERT',
                declared_intervention_nominal_grip=ending,declared_intervention_tool_Y_offset_m=self.sign*offset,
                unexecuted_Teacher_intent_H40=intent,unexecuted_intent_label_role='ORACLE_QUERY_ONLY',
                physical_envelope_metrics=metrics,
                physical_margin_to_original_finger_speed_m_s=.04-metrics['maximum_finger_speed_m_s'])
            self.intervention_issued=True;self.awaiting=True;self.before=witness
            write_json(self.adapter.output/'TEACHER_V9_INTERVENTION_BEFORE.private.json',witness)
        self.stage=self.oracle.stage
        proposal.update(Teacher_version=VERSION,unchanged_recovery_Oracle_version=v8.VERSION,
            teacher_v9_recovery=self.evidence(),expert_training_eligible=False)
        return program,proposal
