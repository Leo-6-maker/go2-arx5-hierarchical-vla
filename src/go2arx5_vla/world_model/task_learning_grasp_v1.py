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
CALIBRATION_B=4703
PREACTION_HOLD_ATOMS=40
B=CALIBRATION_B+25*PREACTION_HOLD_ATOMS
PREFIX_ATOMS,QUERIES,FINAL_HOLD=40,15,20
E=B+25*(PREFIX_ATOMS*QUERIES+FINAL_HOLD)
COMMAND='GRASP_RED_LIFT20MM_HOLD1S'
LANGUAGE='Grasp the red cube, lift it at least 20 millimeters, and hold it steady for one second.'
HOLD_CONTROL='HOLD_ONLY_DIAGNOSTIC_V1'
CONSTRUCTION_ROLE='TASK_CONSTRUCTION_PILOT_NOT_TEACHER_QUALIFICATION'
QUALIFICATION_ROLE='TEACHER_QUALIFICATION_NOT_BC_V1'
EXPANSION_QUALIFICATION_ROLE='NEXT_TRAIN_LAYOUT_TEACHER_FEASIBILITY_NOT_BC_V1'
RECOVERY_QUALIFICATION_ROLE='UNDER_CLOSE_RECOVERY_QUALIFICATION_NOT_BC_V1'
COLLECTION_ROLE='CONTINUOUS_TEACHER_COLLECTION_PENDING_EXPORT_V1'
DEV_REFERENCE_ROLE='FIXED_DEV_TEACHER_REFERENCES_NOT_BC_V1'
STUDENT_DEV_ROLE='NATIVE_GR00T_STUDENT_DEV_NOT_BC_V1'
CORE_LOCAL_ROLE='CORE_LOCAL_POLICY_DIAGNOSTIC_NOT_DEV_TEST_V1'
CONIC_ROLE='CORE_LOCAL_C500_EXPLICIT_CONIC_DIAGNOSTIC_NOT_DEV_TEST_V3'
ENDPOINT_ROLE='CORE_LOCAL_C500_ENDPOINT_BASIS_DIAGNOSTIC_NOT_DEV_TEST_V3'
MATCHED_ROLE='CORE_MATCHED_WEIGHTS_PROCESSOR_SMOOTH_TRAIN_NOT_DEV_TEST_V3_1'


def student_role(role):
    return role in(STUDENT_DEV_ROLE,CORE_LOCAL_ROLE,CONIC_ROLE,ENDPOINT_ROLE,MATCHED_ROLE)


def need(ok,why):
    if not ok:raise ValueError('TASK_LEARNING_GRASP_'+why)


def protocol(role=CONSTRUCTION_ROLE):
    if role=='TEACHER_V9_FULL_RECOVERY24_QUALIFICATION_NOT_BC_V1':
        from .task_learning_teacher_v9_full24 import contract
        return contract()
    if role=='TEACHER_V9_E1_SMALL4_QUALIFICATION_NOT_BC_V1':
        from .task_learning_teacher_e1_v9 import contract
        return contract()
    if role==MATCHED_ROLE:
        from .task_learning_matched_native_v3_1 import protocol as matched_protocol
        return matched_protocol()
    if role==ENDPOINT_ROLE:
        from .task_learning_endpoint_native_v3 import protocol as endpoint_protocol
        return endpoint_protocol()
    if role==CONIC_ROLE:
        from .task_learning_conic_native_v3 import protocol as conic_protocol
        return conic_protocol()
    if role=='TEACHER_V8_RECOVERY24_QUALIFICATION_NOT_BC_V1':
        from .task_learning_teacher_v8 import contract
        return contract()
    need(role in(CONSTRUCTION_ROLE,QUALIFICATION_ROLE,EXPANSION_QUALIFICATION_ROLE,RECOVERY_QUALIFICATION_ROLE,COLLECTION_ROLE,DEV_REFERENCE_ROLE,STUDENT_DEV_ROLE,CORE_LOCAL_ROLE),
        'EXPLICIT_CONSTRUCTION_QUALIFICATION_OR_NEW_CONTINUOUS_COLLECTION_ROLE')
    return dict(schema=KIND,task_revision='V7_TOOL_X_GRASP_OFFSET_WITH_ORIGINAL_REFERENCE_FLANGE_CLEARANCE',reset_revision='V4_ORIGINAL_STARTUP_AND_STEADY_HOLD_THEN_FOUR_SECOND_REFERENCE_HOLD',stage=role,
        calibration_boundary=CALIBRATION_B,start_step=B,preaction_hold_atoms=PREACTION_HOLD_ATOMS,
        preparation=dict(original_startup_to_step=2503,original_steady_prefix_to_step=3703,
            unchanged_initial_TCP_reference_hold_to_step=4703,source_pose_alignment=False,
            total_extra_hold_ticks_after_steady_prefix=2000,native_clock_rebased=False),
        prefix_atoms=PREFIX_ATOMS,policy_queries=QUERIES,final_hold_atoms=FINAL_HOLD,
        task_command=COMMAND,language=LANGUAGE,action_period_s=.05,physics_dt_s=.002,
        expert_commands=('NONE_STUDENT_NOT_EXPERT'if student_role(role) else
            'ACTUAL_50MS_TEACHER_INTENT_WITH_EXACT_CONFIRMED_REFERENCE_TRANSPORT'),
        H40_labels='ROLLING40_CONFIRMED_INTERVALS_NEVER_UNEXECUTED_SUFFIX_OR_FUTURE_ACHIEVED_STATE',
        observation='CURRENT_DUAL_RGB_AND_ACTUAL_EEF9_GRIP1',privileged_student_inputs=False,
        full_H40_preview=True,time_retiming=False,reference_transport='CURRENT_REFERENCE_PLUS_DESIRED_WORLD_DISPLACEMENT',
        teacher_max_chunk_translation_m=.015,pregrasp_above_cube_m=.080,grasp_above_cube_center_m=.012,
        grasp_TCP_offset_tool_m=[-.012,0.,0.],grasp_TCP_world_y_offset_m=-.0017,
        scene_family=('TRAIN_SCENES_INDEPENDENT_CORE_LOCAL_DIAGNOSTIC_NOT_DEV_TEST'if role==CORE_LOCAL_ROLE else
            'SEPARATE_UNDER_HAND_NEW_DEV_LAYOUTS_NOT_TRAIN'if role in(DEV_REFERENCE_ROLE,STUDENT_DEV_ROLE)else
            'SEPARATE_UNDER_HAND_TRAIN_PILOT01_NOT_OLD_WORKSPACE_TASKS'),
        commanded_lift_m=.026,required_object_lift_m=.020,required_hold_s=1.,
        position_stage_tolerance_m=.004,close_increment_per_decision=1.,
        standing=standing_contract(),WBC_changes=False,expert_training_eligible=False,automatic_formal_training=False)


def declared(spec):
    need(spec.get('phase')==PHASE and spec.get('kind')==KIND and spec.get('task_learning_grasp')==protocol(spec.get('task_learning_role',CONSTRUCTION_ROLE))
        and spec.get('task_command')==COMMAND and spec.get('live_skill_start_step')==CALIBRATION_B
        and spec.get('max_ticks')==E and spec.get('physics_dt_s')==.002 and spec.get('WBC_changes')is False,
        'EXACT_NEW_TASK_AND_NATIVE_CLOCK')


def start_readiness_contract():
    from .target_gr00t_collection_v1 import start_readiness_contract as original
    return dict(original(),decision_step=B,extra_settling_integrations=25*PREACTION_HOLD_ATOMS,
        reset_revision='V4_TASK_LOCAL_FOUR_SECOND_REFERENCE_HOLD',native_clock_rebased=False)


def require_start_readiness(adapter,sample):
    from .storage import write_json
    c=start_readiness_contract()
    need(adapter.spec.get('task_learning_start_readiness')==c,'DECLARED_START_READINESS_BEFORE_ANY_NATIVE_STARTUP')
    v=np.asarray(sample['qvel'],float)
    need(v.shape==(32,)and np.isfinite(v).all()and sample['native_step']==B
        and abs(sample['time_s']-B*.002)<=1e-8,'REAL_POST_HOLD_NATIVE_CLOCK_AND_VELOCITIES')
    linear,angular=float(np.linalg.norm(v[:3])),float(np.linalg.norm(v[3:6]))
    result=dict(contract=c,native_step=B,BASE_qvel6=v[:6].tolist(),BASE_linear_speed_m_s=linear,
        BASE_angular_speed_rad_s=angular,passes=linear<=c['maximum_BASE_linear_speed_m_s']
            and angular<=c['maximum_BASE_angular_speed_rad_s'],new_task_teacher_proposals_before_check=0,
        original_thresholds_changed=False,native_clock_rebased=False)
    path=adapter.output/'TASK_LEARNING_START_READINESS.private.json'
    write_json(path,result)
    need(result['passes'],'POST_HOLD_START_READINESS_REJECTED:'+repr(result))
    return dict(path=str(path),sha256=digest(path))


def score_contract():
    return dict(schema='TASK_LEARNING_GRASP_LIFT_HOLD_SCORE_V1',minimum_lift_m=.020,
        continuous_hold_s=1.,clock='NATIVE_2MS_WITH_CONTIGUOUS_50MS_SAMPLES',
        grasp='ORIGINAL_ACTUALSTATE_BILATERAL_FORCE_OPPOSING_NORMALS_RELATIVE_MOTION',
        maximum_object_linear_speed_m_s=.02,standing=standing_contract(),
        hard_safety='ORIGINAL_UNCHANGED_NATIVE_OBSERVER',success_requires_all=True)


def standing_contract():
    return dict(schema='TASK_LEARNING_GRASP_STANDING_SCORE_V3',maximum_base_displacement_m=.020,
        maximum_base_rotation_rad=float(np.deg2rad(5.)),maximum_foot_planar_displacement_m=.005,
        minimum_supported_feet=3,minimum_actual_floor_normal_force_N=.05,
        old_short_task_pose_diagnostic=dict(maximum_base_displacement_m=.002,maximum_base_rotation_rad=float(np.deg2rad(.5))),
        original_native_safety_changed=False,task_score_bounds_are_native_safety=False,
        scoring_reference='ACTUAL_POST_HOLD_BASE_AND_FOUR_FOOT_GEOMETRY')


def standing_metrics(base,feet,supported_feet,initial):
    c=standing_contract();base=np.asarray(base,float);feet=np.asarray(feet,float)
    before=np.asarray(initial['qpos'][:7],float);original_feet=np.asarray(initial['feet'],float)
    need(base.shape==before.shape==(7,)and feet.shape==original_feet.shape==(4,3)
        and all(np.isfinite(v).all()for v in (base,before,feet,original_feet))
        and abs(np.linalg.norm(base[3:])-1)<=1e-6 and abs(np.linalg.norm(before[3:])-1)<=1e-6,
        'REAL_UNIT_BASE_POSES_AND_FOUR_ACTUAL_FEET')
    displacement=float(np.linalg.norm(base[:3]-before[:3]))
    angle=float(2*np.arccos(np.clip(abs(base[3:]@before[3:]),0.,1.)))
    slip=float(np.max(np.linalg.norm(feet[:,:2]-original_feet[:,:2],axis=1)))
    support=len(set(supported_feet))
    need(set(supported_feet)<={'FL','FR','RL','RR'},'ACTUAL_NAMED_FLOOR_SUPPORT_ONLY')
    return dict(contract=c,base_displacement_m=displacement,base_rotation_rad=angle,
        maximum_foot_planar_displacement_m=slip,supported_feet=sorted(set(supported_feet)),
        passes=displacement<=c['maximum_base_displacement_m']and angle<=c['maximum_base_rotation_rad']
            and slip<=c['maximum_foot_planar_displacement_m']and support>=c['minimum_supported_feet'],
        old_short_task_pose_bound_passes=displacement<=.002 and angle<=float(np.deg2rad(.5)))


def score(samples,initial_cube_z,*,standing):
    need(bool(samples),'REAL_SAMPLES_REQUIRED')
    need(len(standing)==len(samples)and all(v['native_step']==s['native_step']and v['contract']==standing_contract()
        for v,s in zip(standing,samples)),'SAME_CLOCK_ACTUAL_STANDING_WITNESSES')
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
    stable=all(v['passes']for v in standing)
    return dict(contract=score_contract(),samples=len(samples),maximum_object_lift_m=maximum,
        longest_contiguous_grasp_lift_hold_s=longest,unsafe_samples=bad,standing_passes=stable,
        maximum_base_displacement_m=max(v['base_displacement_m']for v in standing),
        maximum_base_rotation_rad=max(v['base_rotation_rad']for v in standing),
        maximum_foot_planar_displacement_m=max(v['maximum_foot_planar_displacement_m']for v in standing),
        minimum_supported_feet=min(len(v['supported_feet'])for v in standing),
        old_short_task_pose_bound_all_samples_pass=all(v['old_short_task_pose_bound_passes']for v in standing),
        task_success=bad==0 and stable and longest>=1.)


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

    def grasp_goal(self):
        # Preserve the real pad contact region while keeping the complete
        # transported reference clear of the original link6 flange collider.
        return self.cube[:3,3]+[0,-.0017,.012]+self.rotation@np.array([-.012,0.,0.])

    def decide(self,sample,reference,base):
        actual=np.asarray(sample['tcp']);goal=self.cube[:3,3].copy()+[0,-.0017,.080]
        if self.stage=='PREGRASP'and np.linalg.norm(actual[:3,3]-goal)<=.004:self.stage='DESCEND'
        if self.stage=='DESCEND':
            goal=self.grasp_goal()
            if np.linalg.norm(actual[:3,3]-goal)<=.004:self.stage='CLOSE'
        previous_grip=self.grip
        if self.stage=='CLOSE':
            goal=self.grasp_goal()
            if sample.get('native_grasp')is True:
                self.lift_origin=actual[:3,3].copy();self.stage='LIFT'
            else:self.grip=0.
        if self.stage in('LIFT','HOLD'):
            goal=self.lift_origin+[0,0,.026]
            if float(np.asarray(sample['object_pose'])[2,3]-self.cube[2,3])>=.020:self.stage='HOLD'
        delta=goal-actual[:3,3];distance=float(np.linalg.norm(delta))
        step=delta*min(1.,.015/distance)if distance else delta
        u=np.linspace(0.,1.,40);w=u**3*(10+u*(-15+6*u))
        raw=np.empty((40,10));raw[:,:3]=actual[:3,3]+w[:,None]*step
        quaternions=quat_slerp(matrix_to_quat(actual[:3,:3]),matrix_to_quat(self.rotation),w[:,None])
        raw[:,3:9]=rotation_matrix_to_rot6d_rows(quat_to_matrix(quaternions))
        raw[:,9]=previous_grip+w*(self.grip-previous_grip)
        return FiftyMsProgram(raw,actual,reference,base),dict(stage=self.stage,privileged_goal=goal.tolist(),
            submitted_policy_H40=raw.tolist(),goal_step_limited_by_Teacher_design=True,
            no_future_achieved_state_labels=True,student_metadata=False)


class HoldControl:
    """Matched independent restart control, never a grasp Teacher or BC label."""
    stage='HOLD_CONTROL'

    def __init__(self,initial):
        self.cube=np.asarray(initial['object_pose']).copy()

    def decide(self,sample,reference,base):
        actual=np.asarray(sample['tcp']);raw=np.empty((40,10))
        raw[:,:3]=actual[:3,3];raw[:,3:9]=rotation_matrix_to_rot6d_rows(actual[:3,:3]);raw[:,9]=1.
        program=FiftyMsProgram(raw,actual,reference,base)
        need(np.max(np.abs(program.reference_path[:,:3]-np.asarray(reference)[:3,3]))<=1e-12,
            'HOLD_CONTROL_ZERO_REFERENCE_TRANSLATION')
        return program,dict(stage=self.stage,privileged_goal=actual[:3,3].tolist(),
            submitted_policy_H40=raw.tolist(),diagnostic_control=HOLD_CONTROL,
            expert_training_eligible=False,no_future_achieved_state_labels=True,student_metadata=False)
