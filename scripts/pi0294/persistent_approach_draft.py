"""CPU-only candidate task approach primitive, deliberately not a launcher.

Persistent world goals and bounded reference progress replace repeated measured
pose recentering. Values are a proposal, not adopted scientific settings. This
component cannot assert grasp/task success or start collection/GR00T.
"""
from dataclasses import dataclass,asdict
import numpy as np
from .base_intent import SPEED,ACCEL,rotation
from go2arx5_vla.trajectory.so3 import geodesic_angle,matrix_to_quat,quat_to_matrix,quat_slerp

VERSION='FULLBODY_PERSISTENT_APPROACH_DRAFT_V1_NOT_AUTHORIZED'
LAUNCH_ALLOWED=False


def pose(value):
    p=np.asarray(value,dtype=float)
    if (p.shape!=(4,4) or not np.isfinite(p).all() or not np.allclose(p[3],[0,0,0,1],atol=1e-10)
            or not np.allclose(p[:3,:3].T@p[:3,:3],np.eye(3),atol=1e-8) or np.linalg.det(p[:3,:3])<.999999):
        raise ValueError('ACTUAL_FINITE_SE3_REQUIRED')
    return p.copy()


@dataclass(frozen=True)
class DraftConfig:
    query_s:float=.4
    observation_s:float=.05
    physical_dt_s:float=.002
    linear_speed_m_s:float=.06
    angular_speed_rad_s:float=.3
    no_progress_window_s:float=2.
    # One native command interval's distance, not fitted to old tracking bias.
    minimum_progress_m:float=.003
    no_progress_windows:int=2
    duration_s:float=32.


class PersistentApproachDraft:
    def __init__(self,*,task,actual_tcp,stage_target,actual_BASE,station_world,pad_radius_m,table_top_m=.44,config=DraftConfig()):
        if task not in ('T1','T2'):raise ValueError('TWO_DECLARED_TASKS_ONLY')
        if config!=DraftConfig():raise ValueError('NO_PARAMETER_SEARCH_IN_DRAFT_FIXTURE')
        self.task=task;self.initial=pose(actual_tcp);self.target=pose(stage_target);self.cursor=self.initial.copy()
        self.initial_base=pose(actual_BASE);self.station=pose(station_world);self.config=config
        if not np.isfinite(pad_radius_m) or not 0<pad_radius_m<.2:raise ValueError('MEASURED_PAD_SWEEP_BOUND_REQUIRED')
        self.pad_radius_m=float(pad_radius_m);self.table_top_m=float(table_top_m)
        # Conservative TOOL-PAD envelope only. Full robot swept/IK validity is
        # separately required; no collision-free whole-robot claim is made.
        safe_z=max(float(self.initial[2,3]),float(self.target[2,3]),table_top_m+pad_radius_m+.04+.004)
        raised=self.initial.copy();raised[2,3]=safe_z
        oriented=raised.copy();oriented[:3,:3]=self.target[:3,:3]
        translated=oriented.copy();translated[:2,3]=self.target[:2,3]
        self.route=[raised,oriented,translated,self.target.copy()];self.segment=0
        self.last_time=None;self.last_observation=None;self.history=[];self.stop_reason=None
        self.previous_BASE_velocity=np.zeros(3)

    def observe(self,*,time_s,actual_tcp,actual_BASE,native_safe=True):
        if (not np.isfinite(time_s) or time_s<0 or self.last_observation is not None and time_s<=self.last_observation):
            raise ValueError('STRICT_ACTUAL_OBSERVATION_CLOCK')
        tcp,base=pose(actual_tcp),pose(actual_BASE)
        self.last_observation=float(time_s)
        distance=float(np.linalg.norm(tcp[:3,3]-self.target[:3,3]))
        station_error=float(np.linalg.norm(base[:2,3]-self.station[:2,3]))
        self.history.append(dict(time_s=float(time_s),goal_distance_m=distance,station_distance_m=station_error))
        if not native_safe:self.stop_reason='ORIGINAL_NATIVE_SAFETY_REJECT'
        if time_s>=self.config.duration_s:self.stop_reason=self.stop_reason or 'BOUNDED_TASK_TIME_EXHAUSTED'
        window=self.config.no_progress_window_s*self.config.no_progress_windows
        old=[r for r in self.history if r['time_s']<=time_s-window+1e-8]
        if old:
            change=old[-1]['goal_distance_m']-distance
            base_change=old[-1]['station_distance_m']-station_error
            if change<self.config.minimum_progress_m and base_change<self.config.minimum_progress_m:
                self.stop_reason=self.stop_reason or 'NO_TASK_APPROACH_PROGRESS_NO_FORCED_STAGE_TRANSITION'
        return dict(task=self.task,distance_m=distance,station_error_m=station_error,stop_reason=self.stop_reason,
            manipulation_stage='NOT_ENTERED_WITHOUT_ACTUAL_GEOMETRY',task_success=False)

    def _advance(self,cursor,segment,actual,dt):
        if segment==len(self.route):return cursor,segment
        goal=self.route[segment];delta=goal[:3,3]-cursor[:3,3];distance=np.linalg.norm(delta)
        step=min(distance,self.config.linear_speed_m_s*dt)
        candidate=cursor.copy();candidate[:3,3]+=delta*(step/max(distance,1e-15))
        # Only the persistent reference is bounded. Actual state is never clipped.
        lead=self.config.linear_speed_m_s*self.config.query_s
        if np.linalg.norm(candidate[:3,3]-actual[:3,3])>lead+1e-10:return cursor,segment
        angle=float(geodesic_angle(cursor[:3,:3],goal[:3,:3]))
        q=quat_slerp(matrix_to_quat(cursor[:3,:3]),matrix_to_quat(goal[:3,:3]),min(1.,self.config.angular_speed_rad_s*dt/max(angle,1e-15)))
        candidate[:3,:3]=quat_to_matrix(q)
        angular_lead=self.config.angular_speed_rad_s*self.config.query_s
        if geodesic_angle(candidate[:3,:3],actual[:3,:3])>angular_lead+1e-10:return cursor,segment
        if residual_zero(candidate,goal):segment+=1
        return candidate,segment

    def plan(self,*,time_s,actual_tcp,actual_BASE):
        actual,base=pose(actual_tcp),pose(actual_BASE)
        if self.stop_reason:raise PermissionError(self.stop_reason)
        if self.last_observation is None or time_s!=self.last_observation:raise ValueError('PLAN_REQUIRES_SAME_ACTUAL_OBSERVATION')
        if self.last_time is not None and abs(time_s-self.last_time-self.config.query_s)>1e-8:
            raise ValueError('EXPLICIT_PREFIX8_QUERY_CADENCE_REQUIRED')
        # Advance only by elapsed physical time; repeated calls cannot consume
        # future progress. Full lookahead remains capped by the current lead.
        if self.last_time is not None:
            self.cursor,self.segment=self._advance(self.cursor,self.segment,actual,self.config.query_s)
        self.last_time=float(time_s)
        cursor,segment=self.cursor.copy(),self.segment;trajectory=[]
        for i in range(40):
            if i:cursor,segment=self._advance(cursor,segment,actual,self.config.observation_s)
            trajectory.append(cursor.copy())
        yaw=np.arctan2(base[1,0],base[0,0]);error=rotation(-yaw)@(self.station[:3,3]-base[:3,3])
        desired=np.r_[np.clip(error[:2],-.06,.06),0.]
        velocities=[];previous=self.previous_BASE_velocity.copy()
        for i in range(40):
            previous=previous+np.clip(desired-previous,-ACCEL*.05,ACCEL*.05)
            if np.any(np.abs(previous)>SPEED):raise ValueError('ORIGINAL_BASE_ENVELOPE')
            velocities.append(previous.copy())
        self.previous_BASE_velocity=velocities[7].copy()
        return dict(version=VERSION,launch_allowed=False,task=self.task,source_time_s=float(time_s),
            prefix_points=8,execute_physics_steps=200,record_every_physics_steps=25,
            H40_world_poses=np.asarray(trajectory),BASE_soft_velocity=np.asarray(velocities),
            persistent_station_world=self.station.copy(),persistent_stage_target_world=self.target.copy(),
            planned_segment=self.segment,no_hidden_joint_control=True,task_success=False,
            labels='EXACT_SUBMITTED_PREFIX_AND_H40_ONLY_AFTER_FUTURE_AUTHORIZED_INSTALL_ACK')

    def snapshot(self):
        return dict(version=VERSION,launch_allowed=False,task=self.task,config=asdict(self.config),
            initial=self.initial.tolist(),target=self.target.tolist(),cursor=self.cursor.tolist(),
            initial_base=self.initial_base.tolist(),station=self.station.tolist(),route=[p.tolist() for p in self.route],
            segment=self.segment,last_time=self.last_time,last_observation=self.last_observation,
            history=[dict(r) for r in self.history],stop_reason=self.stop_reason,
            previous_BASE_velocity=self.previous_BASE_velocity.tolist(),pad_radius_m=self.pad_radius_m,table_top_m=self.table_top_m)

    @classmethod
    def restore(cls,snapshot):
        if snapshot['version']!=VERSION or snapshot['launch_allowed'] is not False:raise ValueError('EXACT_DRAFT_ONLY_RESTORE')
        p=cls(task=snapshot['task'],actual_tcp=snapshot['initial'],stage_target=snapshot['target'],actual_BASE=snapshot['initial_base'],
            station_world=snapshot['station'],pad_radius_m=snapshot['pad_radius_m'],table_top_m=snapshot['table_top_m'],config=DraftConfig(**snapshot['config']))
        p.cursor=pose(snapshot['cursor']);p.route=[pose(v) for v in snapshot['route']]
        if not 0<=snapshot['segment']<=len(p.route) or len(p.route)!=4:raise ValueError('DRAFT_ROUTE_CURSOR')
        p.segment=snapshot['segment'];p.last_time=snapshot['last_time'];p.last_observation=snapshot['last_observation']
        p.history=[dict(r) for r in snapshot['history']];p.stop_reason=snapshot['stop_reason']
        p.previous_BASE_velocity=np.asarray(snapshot['previous_BASE_velocity'],dtype=float)
        if p.previous_BASE_velocity.shape!=(3,) or not np.isfinite(p.previous_BASE_velocity).all():raise ValueError('FINITE_RESTORED_BASE')
        return p


def residual_zero(a,b):
    return np.linalg.norm(a[:3,3]-b[:3,3])<=1e-10 and geodesic_angle(a[:3,:3],b[:3,:3])<=1e-7


def precontact_geometry(*,pad_corners_world,object_pose,object_half_size=.02,clearance_m=.004):
    """Candidate geometric diagnostic, not final task scoring or launch approval.

    Requires actual object-frame enclosing gap in XY, pads above the cube and
    within the declared8cm precontact slab. Far TCP or a timer cannot satisfy it.
    Full robot/arm collision and reachability remain unqualified.
    """
    obj=pose(object_pose);pads=np.asarray(pad_corners_world,dtype=float)
    if pads.shape!=(2,8,3) or not np.isfinite(pads).all():raise ValueError('ACTUAL_TWO_PAD_CORNERS_REQUIRED')
    local=(pads-obj[:3,3])@obj[:3,:3]
    low=local.min(axis=1);high=local.max(axis=1)
    order=np.argsort(local.mean(axis=1)[:,1]);negative,positive=order
    gap=low[positive,1]-high[negative,1]
    encloses_y=high[negative,1]<=-object_half_size-clearance_m and low[positive,1]>=object_half_size+clearance_m
    covers_x=bool(np.all(low[:,0]<=object_half_size) and np.all(high[:,0]>=-object_half_size))
    above=bool(np.all(low[:,2]>=object_half_size+clearance_m))
    within_precontact_slab=bool(np.all(low[:,2]<=object_half_size+.08))
    ready=bool(encloses_y and covers_x and above and within_precontact_slab)
    return dict(precontact_geometry_ready=ready,actual_gap_m=float(gap),encloses_object_y=bool(encloses_y),
        overlaps_object_x=covers_x,pads_above_actual_object=above,within_precontact_slab=within_precontact_slab,
        launch_allowed=False,grasp_success=False,whole_robot_clearance='UNQUALIFIED')
