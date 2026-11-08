"""New grasp/lift/hold task: real 50ms commands, no achieved-state labels.

The construction pilot is distinct from the later frozen Teacher panel. This
module supplies a privileged Teacher and explicit reference transport, never
a student input, native admission grant, WBC change or task-success shortcut.
"""
from pathlib import Path
import numpy as np

from .storage import digest
from .fusion_trajectory_adapter import WORKING_CLOSED_M, WORKING_OPEN_M, MECHANICAL_TRAVEL_M
from ..trajectory.so3 import rotation_matrix_to_rot6d_rows, rot6d_rows_to_matrix, matrix_to_quat, quat_to_matrix, quat_slerp

PHASE='collect-target-task-learning-grasp-v1'
KIND='TASK_LEARNING_GRASP_LIFT_HOLD_V1'
B=34303
PREFIX_ATOMS,QUERIES,FINAL_HOLD=40,28,20
E=B+25*(PREFIX_ATOMS*QUERIES+FINAL_HOLD)
COMMAND='GRASP_RED_LIFT20MM_HOLD1S'
LANGUAGE='Grasp the red cube, lift it at least 20 millimeters, and hold it steady for one second.'


def need(ok,why):
    if not ok:raise ValueError('TASK_LEARNING_GRASP_'+why)


def protocol():
    return dict(schema=KIND,stage='TASK_CONSTRUCTION_PILOT_NOT_TEACHER_QUALIFICATION',
        prefix_atoms=PREFIX_ATOMS,policy_queries=QUERIES,final_hold_atoms=FINAL_HOLD,
        task_command=COMMAND,language=LANGUAGE,action_period_s=.05,physics_dt_s=.002,
        expert_commands='ACTUAL_50MS_TEACHER_INTENT_WITH_EXACT_CONFIRMED_REFERENCE_TRANSPORT',
        H40_labels='ROLLING40_CONFIRMED_INTERVALS_NEVER_UNEXECUTED_SUFFIX_OR_FUTURE_ACHIEVED_STATE',
        observation='CURRENT_DUAL_RGB_AND_ACTUAL_EEF9_GRIP1',privileged_student_inputs=False,
        full_H40_preview=True,time_retiming=False,reference_transport='CURRENT_REFERENCE_PLUS_DESIRED_WORLD_DISPLACEMENT',
        teacher_max_chunk_translation_m=.030,pregrasp_above_cube_m=.080,grasp_above_cube_center_m=.012,
        commanded_lift_m=.026,required_object_lift_m=.020,required_hold_s=1.,
        position_stage_tolerance_m=.004,close_increment_per_decision=.15,
        WBC_changes=False,expert_training_eligible=False,automatic_formal_training=False)


def declared(spec):
    need(spec.get('phase')==PHASE and spec.get('kind')==KIND and spec.get('task_learning_grasp')==protocol()
        and spec.get('task_command')==COMMAND and spec.get('live_skill_start_step')==B
        and spec.get('max_ticks')==E and spec.get('physics_dt_s')==.002 and spec.get('WBC_changes')is False,
        'EXACT_NEW_TASK_AND_NATIVE_CLOCK')


def score_contract():
    return dict(schema='TASK_LEARNING_GRASP_LIFT_HOLD_SCORE_V1',minimum_lift_m=.020,
        continuous_hold_s=1.,clock='NATIVE_2MS_WITH_CONTIGUOUS_50MS_SAMPLES',
        grasp='ORIGINAL_ACTUALSTATE_BILATERAL_FORCE_OPPOSING_NORMALS_RELATIVE_MOTION',
        maximum_object_linear_speed_m_s=.02,standing='ORIGINAL_UNCHANGED_REQUIRE_STANDING',
        hard_safety='ORIGINAL_UNCHANGED_NATIVE_OBSERVER',success_requires_all=True)


def score(samples,initial_cube_z):
    need(bool(samples),'REAL_SAMPLES_REQUIRED')
    for a,b in zip(samples,samples[1:]):need(b['native_step']-a['native_step']==25,'CONTIGUOUS_SCORE_CLOCK')
    first=None;longest=0.;maximum=0.;bad=0
    for s in samples:
        height=float(np.asarray(s['object_pose'])[2,3]-initial_cube_z);maximum=max(maximum,height)
        safe=all(s.get(k)is True for k in ('safety_valid','contact_valid','gripper_state_valid','object_present'))
        safe=safe and not any(s.get(k)for k in ('native_abort','illegal_contact','joint_limit_penetration','reset','takeover'))
        if not safe:bad+=1
        holds=safe and s.get('native_grasp')is True and height>=.020 and np.linalg.norm(s['object_linear_velocity'])<=.02
        if holds:
            if first is None:first=s['native_step']
            longest=max(longest,(s['native_step']-first)*.002)
        else:first=None
    return dict(contract=score_contract(),samples=len(samples),maximum_object_lift_m=maximum,
        longest_contiguous_grasp_lift_hold_s=longest,unsafe_samples=bad,task_success=bad==0 and longest>=1.)


class FiftyMsProgram:
    """Preserve a full Teacher H40 at its actual 50ms clock without retiming."""
    geometry_scale=1.
    time_stretch_factor=1
    knot_indices=tuple(range(40))
    end_atom=39

    def __init__(self,raw,actual,reference,base):
        raw,actual,reference,base=(np.asarray(v,float).copy()for v in (raw,actual,reference,base))
        need(raw.shape==(40,10) and actual.shape==reference.shape==(4,4) and base.shape==(7,)
            and all(np.isfinite(v).all()for v in (raw,actual,reference,base))
            and np.all((raw[:,9]>=0)&(raw[:,9]<=1)),'FINITE_FULL40_WITHOUT_CLIPPING')
        rotations=rot6d_rows_to_matrix(raw[:,3:9])
        self.decoded_H40=raw
        self.decision_TCP_target=actual
        self.initial_reference_TCP_target=reference
        self.base_pose_hold=base
        desired=raw.copy()
        desired[:,9]=(WORKING_CLOSED_M+(WORKING_OPEN_M-WORKING_CLOSED_M)*raw[:,9])/MECHANICAL_TRAVEL_M
        installed=desired.copy()
        installed[:,:3]=reference[:3,3]+raw[:,:3]-actual[:3,3]
        installed[:,3:9]=rotation_matrix_to_rot6d_rows(reference[:3,:3]@actual[:3,:3].T@rotations)
        self.desired_path,self.reference_path=desired,installed
        self.validate_envelope()
        for value in (raw,actual,reference,base,desired,installed):value.setflags(write=False)

    def preview(self,elapsed):
        need(type(elapsed)is int and elapsed>=0,'EXACT_50MS_CURSOR')
        return self.reference_path[np.minimum(np.arange(elapsed,elapsed+40),39)].copy()

    def validate_envelope(self):
        from .fusion_trajectory_adapter import _geometry,MAX_TRANSLATION_SPEED_M_S,MAX_ANGULAR_SPEED_RAD_S,MAX_FINGER_SPEED_M_S
        from .fusion_trajectory_adapter import MAX_TRANSLATION_ACCELERATION_M_S2,MAX_ANGULAR_ACCELERATION_RAD_S2
        for path,anchor in ((self.desired_path,self.decision_TCP_target),(self.reference_path,self.initial_reference_TCP_target)):
            m=_geometry(path,anchor)
            need(m['maximum_translation_speed_m_s']<=MAX_TRANSLATION_SPEED_M_S*(1+1e-10)
                and m['maximum_angular_speed_rad_s']<=MAX_ANGULAR_SPEED_RAD_S*(1+1e-10)
                and m['maximum_finger_speed_m_s']<=MAX_FINGER_SPEED_M_S*(1+1e-10)
                and m['maximum_discrete_translation_acceleration_m_s2']<=MAX_TRANSLATION_ACCELERATION_M_S2*(1+1e-10)
                and m['maximum_discrete_angular_acceleration_rad_s2']<=MAX_ANGULAR_ACCELERATION_RAD_S2*(1+1e-10),
                'ORIGINAL_FULL_PREVIEW_RATE_AND_ACCELERATION_LIMITS')

    def evidence(self):
        return dict(schema='TASK_LEARNING_EXACT_50MS_REFERENCE_TRANSPORT_V1',geometry_scale=1.,
            time_stretch_factor=1,policy_action_period_s=.05,physics_dt_s=.002,
            clock=dict(knot_indices=list(range(40)),knot_native_tick_offsets=[25*i for i in range(40)]),
            original_raw_H40_unchanged=True,unexecuted_suffix_is_expert_data=False,
            reference_transport='INIT_REFERENCE_PLUS_DESIRED_WORLD_DISPLACEMENT',
            reference_rotation='INIT_REFERENCE_ROTATION_TIMES_ACTUAL_INVERSE_TIMES_DESIRED',
            original_native_certificate_required=True,native_execution_authorized=False)


class Teacher:
    def __init__(self,initial):
        self.cube=np.asarray(initial['object_pose']).copy();self.rotation=np.asarray(initial['tcp'])[:3,:3].copy()
        self.stage='PREGRASP';self.grip=1.;self.lift_origin=None

    def decide(self,sample,reference,base):
        actual=np.asarray(sample['tcp']);goal=self.cube[:3,3].copy()+[0,-.0017,.080]
        if self.stage=='PREGRASP'and np.linalg.norm(actual[:3,3]-goal)<=.004:self.stage='DESCEND'
        if self.stage=='DESCEND':
            goal=self.cube[:3,3]+[0,-.0017,.012]
            if np.linalg.norm(actual[:3,3]-goal)<=.004:self.stage='CLOSE'
        previous_grip=self.grip
        if self.stage=='CLOSE':
            goal=self.cube[:3,3]+[0,-.0017,.012]
            if sample.get('native_grasp')is True:
                self.lift_origin=actual[:3,3].copy();self.stage='LIFT'
            else:self.grip=max(0.,self.grip-.15)
        if self.stage in('LIFT','HOLD'):
            goal=self.lift_origin+[0,0,.026]
            if float(np.asarray(sample['object_pose'])[2,3]-self.cube[2,3])>=.020:self.stage='HOLD'
        delta=goal-actual[:3,3];distance=float(np.linalg.norm(delta))
        step=delta*min(1.,.030/distance)if distance else delta
        u=np.linspace(0.,1.,40);w=u**3*(10+u*(-15+6*u))
        raw=np.empty((40,10));raw[:,:3]=actual[:3,3]+w[:,None]*step
        quaternions=quat_slerp(matrix_to_quat(actual[:3,:3]),matrix_to_quat(self.rotation),w[:,None])
        raw[:,3:9]=rotation_matrix_to_rot6d_rows(quat_to_matrix(quaternions))
        raw[:,9]=previous_grip+w*(self.grip-previous_grip)
        return FiftyMsProgram(raw,actual,reference,base),dict(stage=self.stage,privileged_goal=goal.tolist(),
            submitted_policy_H40=raw.tolist(),goal_step_limited_by_Teacher_design=True,
            no_future_achieved_state_labels=True,student_metadata=False)
