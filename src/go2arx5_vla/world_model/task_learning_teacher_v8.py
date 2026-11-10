"""Independent current-state Teacher/RecoveryOracle candidate.

Original V7 is unchanged. Proposals use the original full40 transport and native
certificate. Every replan starts from measured TCP and the last *consumed*
nominal command; measured opening is an outcome, never command history.
"""
import numpy as np

from . import task_learning_grasp_v1 as task
from .fusion_trajectory_adapter import WORKING_CLOSED_M,WORKING_OPEN_M,MECHANICAL_TRAVEL_M,_geometry
from ..trajectory.so3 import matrix_to_quat,quat_slerp,quat_to_matrix,rotation_matrix_to_rot6d_rows

VERSION='TEACHER_V8_CURRENT_STATE_MARGIN_RECOVERY_V1'
ROLE='TEACHER_V8_RECOVERY24_QUALIFICATION_NOT_BC_V1'
TYPES=('UNDER_CLOSE','SUPPORTED_SIDE_OFFSET','SUPPORTED_SINGLE_PAD')
MAX_GRIP_DELTA=.7
RECENTER_OPEN=.8
GRASP_Z=.0135


class RecoveryUnavailable(ValueError):
    def __init__(self,reason,evidence):
        self.reason,self.evidence=reason,evidence
        super().__init__('TEACHER_V8_RECOVERY_UNAVAILABLE:'+reason)


def contract():
    base=task.protocol(task.QUALIFICATION_ROLE)
    return dict(base,stage=ROLE,task_revision=VERSION,grasp_above_cube_center_m=GRASP_Z,
        close_increment_per_decision=MAX_GRIP_DELTA,maximum_Teacher_finger_speed_m_s=.03,
        observation='CURRENT_NATIVE_OBJECT_CONTACT_TCP_AND_CONSUMED_COMMAND_PRIVILEGED_TEACHER_ONLY',
        previous_command='LAST_SOLVER_AND_GRIPPER_CONFIRMED_NOMINAL_POLICY_COMMAND',
        recovery_opening='ONLY_SUPPORTED_AND_NOT_NATIVE_GRASPED',
        original_native_reference_and_score_unchanged=True,expert_training_eligible=False)


def consumed_grip(sample,consumed):
    policy=np.asarray(consumed['policy_EEF10'],float);installed=np.asarray(consumed['installed_consumed_EEF10'],float)
    task.need(consumed['native_end_step']==sample['native_step']and consumed['solver_gripper_confirmed']is True
        and policy.shape==installed.shape==(10,)and np.isfinite(policy).all()and np.isfinite(installed).all()
        and 0<=policy[9]<=1,'V8_SAME_CLOCK_ACTUALLY_CONSUMED_COMMAND')
    expected=(WORKING_CLOSED_M+(WORKING_OPEN_M-WORKING_CLOSED_M)*policy[9])/MECHANICAL_TRAVEL_M
    task.need(abs(installed[9]-expected)<=1e-12,'V8_WORKING_MAPPING_ONCE_AND_NOT_ACTUAL_OPENING')
    return float(policy[9])


def command_context(sample,cycle,program):
    """Use the completed original consumer's real solver and gripper receipts."""
    rows=cycle['windows'];task.need(len(rows)==40 and cycle['all_40_knots_consumed']is True,
        'V8_COMPLETE_PREVIOUS_H40_CONSUMPTION')
    last=rows[-1];task.need(last['stage']=='ACTUAL_CONSUMPTION_CONFIRMED'
        and last['native_end_step']==sample['native_step']and last['actual_ticks']==25
        and last['solver_policies_with_exact_installed_target']and last['gripper']['command_consumed']is True,
        'V8_ORIGINAL_NATIVE_RECEIPTS_BEFORE_ORACLE')
    value=dict(native_end_step=last['native_end_step'],policy_EEF10=program.decoded_H40[-1].tolist(),
        installed_consumed_EEF10=last['actual_installed_consumed_EEF10_BASE7'][:10],solver_gripper_confirmed=True)
    consumed_grip(sample,value);return value


def native_support(adapter,sample):
    """Current original2ms contact packet, not a stage or predicted support bit."""
    packet=adapter.observe();reader=adapter.actual_state
    task.need(packet['step']==sample['native_step']and abs(packet['time_s']-sample['time_s'])<=1e-9,
        'V8_SAME_CURRENT_NATIVE_SUPPORT_PACKET')
    table=reader.model.geom('task_table').id;cube=reader.cube_geom
    return any({c['geom0'],c['geom1']}=={table,cube}and c['wrench'][0]>=.05 for c in packet['contacts'])


class Teacher:
    def __init__(self,initial,*,recovery_entry=False):
        self.cube=np.asarray(initial['object_pose'],float).copy()
        self.rotation=np.asarray(initial['tcp'],float)[:3,:3].copy()
        self.stage='PREGRASP';self.substage='APPROACH';self.grip=None;self.lift_goal=None
        self.recovery_entry=recovery_entry;self.previous_decision_step=None

    def target_rotation(self,sample):
        return np.asarray(sample['object_pose'])[:3,:3]@self.cube[:3,:3].T@self.rotation

    def grasp_goal(self,sample):
        return np.asarray(sample['object_pose'])[:3,3]+[0,-.0017,GRASP_Z]+self.target_rotation(sample)@[-.012,0,0]

    def decide(self,sample,reference,base,*,consumed_reference,supported):
        task.need(type(supported)is bool and all(sample.get(k)is True for k in
            ('safety_valid','contact_valid','gripper_state_valid','object_present')),
            'V8_ORIGINAL_SAFE_CURRENT_OBSERVATION')
        step=sample['native_step'];task.need(type(step)is int and task.B<=step<task.B+1000*task.QUERIES
            and(step-task.B)%1000==0 and(self.previous_decision_step is None or step==self.previous_decision_step+1000),
            'V8_ORIGINAL_FIFTEEN_TWO_SECOND_DECISIONS_NO_REBASED_CLOCK')
        previous=consumed_grip(sample,consumed_reference);actual=np.asarray(sample['tcp'],float)
        cube=np.asarray(sample['object_pose'],float);grasp=sample.get('native_grasp')is True
        snapshot=dict(native_step=step,cube_supported=supported,native_grasp=grasp,
            left_object_contact=sample.get('left_object_contact'),right_object_contact=sample.get('right_object_contact'),
            actual_gripper_fraction=float(sample['gripper_fraction']),previous_consumed_nominal_grip=previous)
        if not supported and not grasp:raise RecoveryUnavailable('UNSUPPORTED_NOT_GRASPED_NO_AIR_OPENING',snapshot)
        if cube[2,3]<self.cube[2,3]-.005:raise RecoveryUnavailable('OBJECT_BELOW_DECLARED_SUPPORT_DOMAIN',snapshot)
        if not grasp and np.linalg.norm(sample['object_linear_velocity'])>.02:
            raise RecoveryUnavailable('MOVING_UNATTACHED_OBJECT_OUTSIDE_RECOVERY_DOMAIN',snapshot)
        if not grasp and np.linalg.norm(cube[:2,3]-self.cube[:2,3])>.012:
            raise RecoveryUnavailable('OBJECT_OUTSIDE_DECLARED_LOCAL_RECOVERY_DOMAIN',snapshot)
        relative_q=matrix_to_quat(cube[:3,:3]@self.cube[:3,:3].T)
        orientation_change=2*np.arctan2(np.linalg.norm(relative_q[:3]),abs(relative_q[3]))
        if not grasp and orientation_change>.03:
            raise RecoveryUnavailable('OBJECT_ORIENTATION_OUTSIDE_LOCAL_RECOVERY_DOMAIN',snapshot)
        goal=self.grasp_goal(sample);desired_grip=previous;rotation_goal=self.target_rotation(sample)
        if grasp:
            remaining=max(0.,self.cube[2,3]+.026-cube[2,3])
            self.lift_goal=actual[:3,3]+[0,0,remaining]
            self.stage='HOLD'if cube[2,3]-self.cube[2,3]>=.020 else'LIFT'
            self.substage='RETAIN_ACTUAL_GRASP';goal=self.lift_goal;desired_grip=0.
        elif self.stage in ('LIFT','HOLD'):
            if not supported:raise RecoveryUnavailable('GRASP_LOST_ABOVE_SUPPORT',snapshot)
            self.stage='CLOSE';self.substage='SUPPORTED_REACQUIRE';self.lift_goal=None
        if not grasp:
            lateral=float(np.linalg.norm((actual[:3,3]-goal)[:2]));vertical=abs(actual[2,3]-goal[2])
            one_pad=bool(sample.get('left_object_contact'))!=bool(sample.get('right_object_contact'))
            if self.recovery_entry or self.stage=='CLOSE':
                self.stage='CLOSE'
                if (one_pad or lateral>.002)and previous<RECENTER_OPEN-1e-12:
                    self.substage='SUPPORTED_RELEASE';goal=actual[:3,3].copy();desired_grip=RECENTER_OPEN
                    rotation_goal=actual[:3,:3].copy()
                elif lateral>.002 or vertical>.004:
                    self.substage='SUPPORTED_ALIGN';desired_grip=max(previous,RECENTER_OPEN)
                    if np.linalg.norm(goal-actual[:3,3])>.015*(task.QUERIES-(step-task.B)//1000-2):
                        raise RecoveryUnavailable('INSUFFICIENT_REMAINING_ORIGINAL_DECISIONS',snapshot)
                else:self.substage='CLOSE_CURRENT_OBJECT';desired_grip=0.
            else:
                if self.stage=='PREGRASP':
                    above=cube[:3,3]+[0,-.0017,.080]
                    if np.linalg.norm(actual[:3,3]-above)<=.004:self.stage='DESCEND'
                    else:goal=above
                if self.stage=='DESCEND':
                    goal=self.grasp_goal(sample)
                    if lateral<=.002 and vertical<=.004:self.stage='CLOSE';self.substage='CLOSE_CURRENT_OBJECT';desired_grip=0.
                if self.stage!='CLOSE':desired_grip=1.
        # This bound designs the new Teacher command; it never clips a student
        # action. The original final H40 still passes the unchanged envelope.
        next_grip=previous+np.sign(desired_grip-previous)*min(abs(desired_grip-previous),MAX_GRIP_DELTA)
        if next_grip>previous+1e-12 and(not supported or grasp):raise RecoveryUnavailable('OPENING_REQUIRES_ACTUAL_SUPPORTED_NON_GRASP',snapshot)
        delta=goal-actual[:3,3];distance=float(np.linalg.norm(delta));delta*=min(1.,.015/distance)if distance else 1.
        u=np.linspace(0.,1.,40);w=u**3*(10+u*(-15+6*u));raw=np.empty((40,10))
        raw[:,:3]=actual[:3,3]+w[:,None]*delta
        q=quat_slerp(matrix_to_quat(actual[:3,:3]),matrix_to_quat(rotation_goal),w[:,None])
        raw[:,3:9]=rotation_matrix_to_rot6d_rows(quat_to_matrix(q));raw[:,9]=previous+w*(next_grip-previous)
        program=task.FiftyMsProgram(raw,actual,reference,base)
        margins=_geometry(program.desired_path,actual)
        task.need(margins['maximum_finger_speed_m_s']<=.03*(1+1e-10),'V8_DESIGNED_FINGER_SPEED_MARGIN')
        self.previous_decision_step=step;self.grip=next_grip
        return program,dict(stage=self.stage,Teacher_version=VERSION,recovery_substage=self.substage,
            submitted_policy_H40=raw.tolist(),privileged_goal=np.asarray(goal).tolist(),actual_context=snapshot,
            original_task_initial_cube_z_m=float(self.cube[2,3]),current_cube_z_m=float(cube[2,3]),
            original_task_clock_and_score_preserved=True,physical_envelope_metrics=margins,
            physical_margin_to_original_finger_speed_m_s=.04-margins['maximum_finger_speed_m_s'],
            label_role='EXECUTED_TEACHER_INTENT',student_metadata=False,no_future_achieved_state_labels=True)


class NativeQualificationTeacher:
    """One declared intervention, then recovery from its newly observed outcome."""
    def __init__(self,adapter,initial,history):
        self.adapter,self.history=adapter,history;self.oracle=Teacher(initial)
        self.mode=adapter.spec['teacher_v8_recovery_condition'];task.need(self.mode in TYPES,'V8_DECLARED_RECOVERY_TYPE')
        self.sign=adapter.spec['teacher_v8_offset_sign'];task.need(self.sign in (-1,1),'V8_PREREGISTERED_OFFSET_SIGN')
        self.stage=self.oracle.stage;self.cube=self.oracle.cube
        self.intervention_issued=False;self.awaiting=False;self.entry_observed=False
        self.entry_step=None;self.native_grasp_after_correction=False

    def evidence(self):
        return dict(Teacher_version=VERSION,recovery_condition=self.mode,intervention_issued=self.intervention_issued,
            actual_recovery_entry_observed=self.entry_observed,recovery_entry_native_step=self.entry_step,
            native_grasp_after_correction=self.native_grasp_after_correction,new_Oracle_not_student_DAgger=True)

    def decide(self,sample,reference,base):
        from .storage import write_json
        cycle,previous=self.history();consumed=command_context(sample,cycle,previous)
        supported=native_support(self.adapter,sample)
        if self.awaiting:
            lateral=abs(float((np.asarray(sample['tcp'])[:3,3]-self.oracle.grasp_goal(sample))@self.oracle.target_rotation(sample)[:,1]))
            one_pad=bool(sample.get('left_object_contact'))!=bool(sample.get('right_object_contact'))
            condition=sample.get('native_grasp')is False and supported and (
                sample['gripper_fraction']>=.5 if self.mode=='UNDER_CLOSE'else
                lateral>=.002 if self.mode=='SUPPORTED_SIDE_OFFSET'else one_pad)
            if not condition:
                event=dict(self.evidence(),native_step=sample['native_step'],reason='DECLARED_RECOVERY_CONDITION_NOT_ESTABLISHED',
                    supported=supported,native_grasp=sample.get('native_grasp'),one_pad=one_pad,lateral_offset_m=lateral)
                write_json(self.adapter.output/'TEACHER_V8_RECOVERY_EXIT.private.json',event)
                raise RecoveryUnavailable(event['reason'],event)
            self.entry_observed=True;self.entry_step=sample['native_step'];self.awaiting=False;self.oracle.recovery_entry=True
        if self.entry_observed and sample['native_step']>self.entry_step and sample.get('native_grasp')is True:
            self.native_grasp_after_correction=True
        try:program,proposal=self.oracle.decide(sample,reference,base,consumed_reference=consumed,supported=supported)
        except RecoveryUnavailable as error:
            write_json(self.adapter.output/'TEACHER_V8_RECOVERY_EXIT.private.json',dict(self.evidence(),reason=error.reason,actual=error.evidence));raise
        if self.oracle.stage=='CLOSE'and not self.intervention_issued:
            raw=program.decoded_H40.copy();u=np.linspace(0,1,40);w=u**3*(10+u*(-15+6*u))
            unexecuted_intent=raw.tolist()
            ending=.8 if self.mode=='SUPPORTED_SIDE_OFFSET'else .55
            raw[:,9]=consumed['policy_EEF10'][9]+w*(ending-consumed['policy_EEF10'][9])
            offset=0. if self.mode=='UNDER_CLOSE'else .003 if self.mode=='SUPPORTED_SIDE_OFFSET'else .0045
            raw[:,:3]+=w[:,None]*(self.sign*offset*self.oracle.target_rotation(sample)[:,1])
            program=task.FiftyMsProgram(raw,sample['tcp'],reference,base)
            task.need(_geometry(program.desired_path,sample['tcp'])['maximum_finger_speed_m_s']<=.03*(1+1e-10),
                'V8_INTERVENTION_RETAINS_FINGER_SPEED_MARGIN')
            proposal.update(submitted_policy_H40=raw.tolist(),label_role='INTERVENTION_NOT_EXPERT',
                declared_intervention_nominal_grip=ending,declared_intervention_tool_Y_offset_m=self.sign*offset,
                unexecuted_Teacher_intent_H40=unexecuted_intent,unexecuted_intent_label_role='ORACLE_QUERY_ONLY',
                physical_envelope_metrics=_geometry(program.desired_path,sample['tcp']))
            proposal['physical_margin_to_original_finger_speed_m_s']=.04-proposal['physical_envelope_metrics']['maximum_finger_speed_m_s']
            self.intervention_issued=True;self.awaiting=True
        self.stage=self.oracle.stage;proposal.update(teacher_v8_recovery=self.evidence())
        return program,proposal
