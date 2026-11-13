"""New full target PICK_PLACE construction over the original grasp/safety chain.

The old grasp task, its scores, Teacher implementation and saved episodes are
unchanged. This version requires actual transfer, support, release, withdrawal
and stable object termination. A generated reference is never a success label.
"""
from pathlib import Path
import numpy as np

from . import task_learning_grasp_v1 as grasp
from .storage import digest

VERSION='SLA_TARGET_PICK_PLACE_CONSTRUCTION_V5_1'
PANEL='SLA_TARGET_PICK_PLACE_FIXED_PILOT1_V5_1'
COMMAND='GRASP_RED_TRANSFER150MM_PLACE_RELEASE_WITHDRAW_V5'
LANGUAGE='Pick up the red cube, place it on the left side of the table, release it, and withdraw the gripper.'
QUERIES=120
ENDPOINT=grasp.B+25*(QUERIES*10+40)
TRANSFER=np.array([0.,.15,0.])
ROLE='CORELAB_EXPERT_TRAIN4_SEMANTICS_NOT_BC_V3'


def configuration(spec):
    value=spec.get('sla_pick_place_v5')
    if value is None:return False
    grasp.need(value==dict(version=VERSION) and spec.get('task_learning_role')==ROLE
        and spec.get('task_learning_teacher_mode')=='NORMAL_TEACHER_V7'
        and spec.get('task_learning_control')is None and not spec.get('target_gr00t_policy'),
        'EXPLICIT_NEW_FULL_PLACE_CPU_EXPERT_ONLY')
    return True


def score_contract():
    return dict(schema='SLA_TARGET_PICK_PLACE_SCORE_V5',minimum_grasp_s=.3,minimum_lift_m=.020,
        minimum_carried_xy_m=.10,source_to_goal_xy_m=.15,goal_xy_half_width_m=.010,
        maximum_height_error_m=.005,minimum_actual_table_support_force_N=.05,
        minimum_measured_open_fraction=.85,maximum_linear_speed_m_s=.02,
        maximum_angular_speed_rad_s=.1,minimum_actual_TCP_object_clearance_m=.08,
        continuous_terminal_s=2.,original_native_safety_unchanged=True,standing=grasp.standing_contract(),
        original_GO2_PLACE_or_grasp_score_changed=False,clock='ORIGINAL_CONTIGUOUS_2MS_50MS_SAMPLES')


def protocol():
    p=grasp.protocol(grasp.CONSTRUCTION_ROLE)
    p.update(schema=VERSION,stage=ROLE,task_revision=VERSION,task_command=COMMAND,language=LANGUAGE,
        prefix_atoms=10,policy_queries=QUERIES,final_hold_atoms=40,expert_consumer='EEF_STREAM_V2',
        reference_TTL_s=.4,reference_renewal_max_s=.35,goal_offset_world_m=TRANSFER.tolist(),
        task_score=score_contract(),construction_only_not_D100=True,postgrasp_step_limit_m=.004)
    return p


def declared(spec):
    grasp.need(configuration(spec),'EXPLICIT_PLACE_CONFIGURATION_REQUIRED')
    grasp.need(spec.get('phase')==grasp.PHASE and spec.get('kind')==grasp.KIND
        and spec.get('task_learning_grasp')==protocol()and spec.get('task_command')==COMMAND
        and spec.get('live_skill_start_step')==grasp.CALIBRATION_B and spec.get('max_ticks')==ENDPOINT
        and spec.get('physics_dt_s')==.002 and spec.get('WBC_changes')is False,
        'EXACT_FULL_PLACE_PROTOCOL_WITH_UNCHANGED_NATIVE_PREPARATION')


def validate_panel(panel):
    from .task_learning_scene_v1 import scene_id
    from .task_learning_teacher_panel_v1 import teacher_fingerprint,FROZEN_TEACHER
    from .task_learning_collection_v1 import qualified_base
    rows=panel.get('assignments')or[]
    grasp.need(panel.get('schema')==PANEL and panel.get('protocol')==protocol()
        and panel.get('source_sha') and panel.get('tasks')==['PICK_PLACE']and panel.get('D100_new')is False
        and panel.get('base_grasp_Teacher_sha256')==teacher_fingerprint()==FROZEN_TEACHER
        and len(rows)==1 and rows[0]['layout']=='TRAIN01'and rows[0]['restart']==10
        and rows[0]['parent']==scene_id('TRAIN01')and rows[0]['split']=='TRAIN'
        and rows[0]['command']==COMMAND and rows[0]['purpose']=='NEW_FULL_PICK_PLACE_CONSTRUCTION_NOT_D100'
        and rows[0]['prefix_atoms']==10 and Path(rows[0]['native_run']).name==rows[0]['native_run'],
        'FIXED_NEW_PLACE_PILOT_NOT_OLD_HOLD_OR_FORMAL_DATA')
    qualified_base(panel['base_Teacher_qualification'])
    return rows


def native_scene_admission(spec):
    from .stable_task_scene_v1 import pinned
    from .task_learning_scene_v1 import validate_static_provenance
    from .native_body_camera_capture_v1 import session_configuration
    declared(spec)
    grasp.need(session_configuration(spec)and spec.get('expert_eef_stream_v2')==dict(version='EEF_STREAM_V2',prefix_atoms=10),
        'PLACE_REUSES_PERSISTENT_RGB_AND_ORIGINAL_FULL_PREVIEW_P10_CONSUMER')
    proof=pinned(spec['task_learning_scene_reference'])
    validate_static_provenance(spec,proof,expected_role=ROLE)
    panel=pinned(dict(path=spec['stable_task_panel'],sha256=spec['stable_task_panel_sha256']))
    row,=validate_panel(panel)
    grasp.need(panel['source_sha']==spec['source_sha']and panel['place_Teacher_sha256']==spec['source_files'][
        'src/go2arx5_vla/world_model/sla_pick_place_v5.py']and row['model_sha256']==digest(spec['model_path'])
        and row['native_run']==Path(spec['registered_root']).name and row['parent']==spec['data_parent']
        and spec['expert_cadence_restart']==row['restart'],'EXACT_NEW_PLACE_MODEL_TEACHER_AND_REQUEST')
    return proof


def with_actual_support(adapter,sample):
    """Attach same-clock original packet forces for scoring, never policy input."""
    state=adapter.actual_state;packet=adapter.observe()
    grasp.need(packet['step']==sample['native_step']and state.data.time==sample['time_s'],
        'PLACE_ACTUAL_SAME_PACKET_SUPPORT_FORCE')
    cube=state.model.geom('task_cube_geom').id;table=state.model.geom('task_table').id
    force=sum(max(0.,float(c['wrench'][0]))for c in packet['contacts']if{c['geom0'],c['geom1']}=={cube,table})
    return dict(sample,place_actual_table_support_force_N=force)


def _duration(samples,predicate):
    first=None;longest=0.
    for s in samples:
        if predicate(s):
            if first is None:first=s['native_step']
            longest=max(longest,(s['native_step']-first)*.002)
        else:first=None
    return longest


def terminal_score(samples,initial_z,*,standing,goal):
    c=score_contract();goal=np.asarray(goal,float)
    grasp.need(bool(samples)and len(samples)==len(standing)and goal.shape==(3,), 'PLACE_REAL_SAMPLES_AND_FIXED_GOAL')
    for i,(s,w)in enumerate(zip(samples,standing)):
        grasp.need(w['native_step']==s['native_step']and w['contract']==grasp.standing_contract(), 'PLACE_SAME_STANDING_CLOCK')
        if i:grasp.need(s['native_step']-samples[i-1]['native_step']==25,'PLACE_CONTIGUOUS_OBSERVATION_CLOCK')
    def safe(s):
        return all(s.get(k)is True for k in('safety_valid','contact_valid','gripper_state_valid','object_present'))and not any(
            s.get(k)for k in('native_abort','illegal_contact','joint_limit_penetration','reset','takeover'))
    def placed(s):
        obj=np.asarray(s['object_pose'])[:3,3];tcp=np.asarray(s['tcp'])[:3,3]
        return bool(safe(s)and np.max(np.abs(obj[:2]-goal[:2]))<=c['goal_xy_half_width_m']
            and abs(obj[2]-initial_z)<=c['maximum_height_error_m']
            and s.get('place_actual_table_support_force_N',0.)>=c['minimum_actual_table_support_force_N']
            and s['gripper_fraction']>=c['minimum_measured_open_fraction']
            and not any(s.get(k)for k in('left_object_contact','right_object_contact','robot_object_contact','native_grasp'))
            and np.linalg.norm(tcp-obj)>=c['minimum_actual_TCP_object_clearance_m']
            and np.linalg.norm(s['object_linear_velocity'])<=c['maximum_linear_speed_m_s']
            and np.linalg.norm(s['object_angular_velocity'])<=c['maximum_angular_speed_rad_s'])
    longest=_duration(samples,placed);unsafe=sum(not safe(s)for s in samples);stable=all(w['passes']for w in standing)
    return dict(contract=c,samples=len(samples),unsafe_samples=unsafe,standing_passes=stable,
        longest_contiguous_terminal_success_s=longest,goal_center_world=goal.tolist(),
        task_success=unsafe==0 and stable and longest>=c['continuous_terminal_s'],
        maximum_base_displacement_m=max(w['base_displacement_m']for w in standing),
        maximum_base_rotation_rad=max(w['base_rotation_rad']for w in standing),
        maximum_foot_planar_displacement_m=max(w['maximum_foot_planar_displacement_m']for w in standing),
        minimum_supported_feet=min(len(w['supported_feet'])for w in standing),
        old_short_task_pose_bound_all_samples_pass=all(w['old_short_task_pose_bound_passes']for w in standing))


def score(samples,initial_z,*,standing,goal):
    result=terminal_score(samples,initial_z,standing=standing,goal=goal);initial=np.asarray(samples[0]['object_pose'])[:3,3]
    grasp_s=_duration(samples,lambda s:s.get('native_grasp')is True)
    lift=max(float(np.asarray(s['object_pose'])[2,3]-initial_z)for s in samples if s.get('native_grasp'))if any(s.get('native_grasp')for s in samples)else 0.
    airborne=[s for s in samples if s.get('native_grasp')is True and np.asarray(s['object_pose'])[2,3]-initial_z>=.020]
    carry=max((float(np.linalg.norm(np.asarray(s['object_pose'])[:2,3]-initial[:2]))for s in airborne),default=0.)
    result.update(longest_actual_grasp_s=grasp_s,maximum_grasped_lift_m=lift,maximum_grasped_carry_xy_m=carry,
        task_success=bool(result['task_success']and grasp_s>=.3 and lift>=.020 and carry>=.10))
    return result


class Teacher:
    """Use the unchanged grasp Teacher, then real-state transfer/place feedback."""
    def __init__(self,initial):
        self.grasper=grasp.Teacher(initial);self.cube=self.grasper.cube;self.rotation=self.grasper.rotation
        self.goal_center=self.cube[:3,3]+TRANSFER;self.stage='PREGRASP';self.grip=1.;self.lift_since=None
        self.lift_TCP_z=None;self.release_TCP=None;self.withdraw_goal=None

    def decide(self,sample,reference,base):
        actual=np.asarray(sample['tcp']);obj=np.asarray(sample['object_pose'])[:3,3]
        if self.stage in('PREGRASP','DESCEND','CLOSE','LIFT','HOLD'):
            program,proposal=self.grasper.decide(sample,reference,base);self.stage=self.grasper.stage;self.grip=self.grasper.grip
            if sample.get('native_grasp')is True and obj[2]-self.cube[2,3]>=.020:
                if self.lift_since is None:self.lift_since=sample['native_step']
                if(sample['native_step']-self.lift_since)*.002>=.3:
                    self.stage='TRANSFER';self.lift_TCP_z=float(actual[2,3])
            else:self.lift_since=None
            return program,dict(proposal,stage=self.stage,base_grasp_stage=proposal['stage'],full_place_not_established=True)
        goal=actual[:3,3].copy();previous_grip=self.grip
        if self.stage=='TRANSFER':
            grasp.need(sample.get('native_grasp')is True,'PLACE_TRANSFER_REQUIRES_ACTUAL_BILATERAL_GRASP')
            goal[:2]+=self.goal_center[:2]-obj[:2];goal[2]=self.lift_TCP_z
            if np.max(np.abs(obj[:2]-self.goal_center[:2]))<=.004:self.stage='LOWER'
        if self.stage=='LOWER':
            goal=actual[:3,3].copy()
            goal[:2]+=self.goal_center[:2]-obj[:2];goal[2]+=self.goal_center[2]-obj[2]
            if(sample.get('place_actual_table_support_force_N',0.)>=.05 and abs(obj[2]-self.goal_center[2])<=.003):
                self.stage='RELEASE';self.release_TCP=actual[:3,3].copy()
        if self.stage=='RELEASE':
            goal=self.release_TCP.copy();self.grip=1.
            if sample['gripper_fraction']>=.85 and not any(sample.get(k)for k in('native_grasp','left_object_contact','right_object_contact')):
                self.stage='WITHDRAW';self.withdraw_goal=actual[:3,3]+[0.,0.,.10]
        if self.stage in('WITHDRAW','PLACED'):
            self.grip=1.;goal=self.withdraw_goal.copy()
            if np.linalg.norm(actual[:3,3]-obj)>=.08:self.stage='PLACED'
        # New transfer/lower/withdraw design only: the actual held TRAIN pose
        # reached1.02032rad/s² at5mm/prefix.4mm gives.81651 under the unchanged
        # original1rad/s² certificate. The qualified grasp Teacher is intact.
        delta=goal-actual[:3,3];distance=np.linalg.norm(delta);step=delta*min(1.,.004/distance)if distance else delta
        u=np.linspace(0.,1.,40);w=u**3*(10+u*(-15+6*u));raw=np.empty((40,10))
        raw[:,:3]=actual[:3,3]+w[:,None]*step
        raw[:,3:9]=grasp.rotation_matrix_to_rot6d_rows(grasp.quat_to_matrix(grasp.quat_slerp(
            grasp.matrix_to_quat(actual[:3,:3]),grasp.matrix_to_quat(self.rotation),w[:,None])))
        raw[:,9]=previous_grip+w*(self.grip-previous_grip)
        return grasp.FiftyMsProgram(raw,actual,reference,base),dict(stage=self.stage,privileged_goal=goal.tolist(),
            submitted_policy_H40=raw.tolist(),no_future_achieved_state_labels=True,student_metadata=False,
            full_place_not_established=True)
