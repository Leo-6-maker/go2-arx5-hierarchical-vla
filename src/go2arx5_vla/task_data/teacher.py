from __future__ import annotations

import numpy as np
from .config import PARAMETERS as P
from ..trajectory.so3 import geodesic_angle,matrix_to_quat,quat_slerp,quat_to_matrix

STAGES=('approach','align','close','verify_grasp','lift','carry','lower','release','withdraw','hold')


class Teacher:
    def __init__(self,card,initial,mode='normal'):
        self.card,self.initial,self.mode=card,initial.copy(),mode
        self.stage='approach';self.events=[];self.grasp_since=None;self.stage_since=0.
        self.lifted=False;self.carried=False;self.release_tcp=None;self.final_tcp=None

    def target(self,s):
        target=np.eye(4);yaw=self.card['cube_yaw'];c,z=np.cos(yaw),np.sin(yaw)
        target[:3,:3]=[[c,-z,0],[z,c,0],[0,0,1]]
        source=self.initial[:3,3].copy()
        if self.mode=='empty_grasp':source[:2]=self.card['source_xy']
        target[:3,3]=source;opening=1.
        if self.stage=='approach':target[2,3]+=.08
        elif self.stage in ['align','close','verify_grasp']:target[2,3]+=.004
        elif self.stage in ['lift','carry','lower']:
            target[:3,3]=source
            if self.stage!='lift':target[:2,3]=self.card['goal_xy']
            target[2,3]+=.004 if self.stage=='lower' else P['lift_clearance']+.004
        elif self.stage=='release':target=self.release_tcp.copy()
        elif self.stage in ['withdraw','hold']:target=self.final_tcp.copy()
        if self.stage in ['close','verify_grasp','lift','carry','lower']:opening=0.
        return target,opening

    def observe(self,s):
        t=s['time_s'];grasp=s['native_grasp']
        if grasp and self.grasp_since is None:self.grasp_since=t
        if not grasp:self.grasp_since=None
        self.lifted |= bool(grasp and s['object_pose'][2,3]-self.initial[2,3]>=.02)
        self.carried |= bool(grasp and np.linalg.norm(s['object_pose'][:2,3]-self.initial[:2,3])>=.10)
        target,_=self.target(s)
        reached=bool(np.linalg.norm(target[:3,3]-s['tcp'][:3,3])<=P['position_stage_tolerance_m'])
        new=None
        if self.stage=='approach' and reached:new='align'
        elif self.stage=='align' and reached:new='close'
        elif self.stage=='close' and grasp:new='verify_grasp'
        elif self.stage=='verify_grasp' and self.grasp_since is not None and t-self.grasp_since>=.3:new='lift'
        elif self.stage=='lift' and reached and self.lifted:new='carry'
        elif self.stage=='carry' and reached and self.carried:new='lower'
        elif self.stage=='lower' and reached and abs(s['object_pose'][2,3]-(P['table_top']+.02))<=.004:
            new='release';self.release_tcp=s['tcp'].copy()
        elif self.stage=='release' and not grasp and s['gripper_fraction']>=.85:
            new='withdraw';self.final_tcp=self.release_tcp.copy();self.final_tcp[2,3]+=P['withdraw_clearance']
        elif self.stage=='withdraw' and reached:new='hold'
        if new:
            self.events.append(dict(time_s=t,previous=self.stage,stage=new,physical_grasp=grasp,actual_lift=self.lifted,actual_carry=self.carried))
            self.stage=new;self.stage_since=t

    def plan(self,s):
        initial=s['tcp'];target,opening=self.target(s);offset=np.arange(40)/20
        delta=target[:3,3]-initial[:3,3];distance=np.linalg.norm(delta)
        poses=np.repeat(initial[None],40,axis=0)
        poses[:,:3,3]+=np.minimum(1.,P['reference_speed_m_s']*offset/max(distance,1e-12))[:,None]*delta
        angle=float(geodesic_angle(initial[:3,:3],target[:3,:3]))
        for i,t in enumerate(offset):poses[i,:3,:3]=quat_to_matrix(quat_slerp(matrix_to_quat(initial[:3,:3]),matrix_to_quat(target[:3,:3]),min(1.,P['reference_angular_speed_rad_s']*t/max(angle,1e-12))))
        return dict(poses=poses,opening=np.full(40,opening),offset_s=offset,stage=self.stage,anchor_pose=initial.copy(),decision_time=s['time_s'])
