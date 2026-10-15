"""Frozen candidate selection plus original actual object/contact task events."""
import itertools
import numpy as np
from .mp_teacher import Teacher as OriginalTeacher
from .mps2_geometry import packages,alignment
from .mps2_client import Client,PlannerFailure
from .mps2_runtime import VERSION,cards

class Teacher(OriginalTeacher):
    def __init__(self,card,state,engine,base,*,root,condition=None,cpu=False):
        super().__init__(card,state,engine,base,root=root,condition=condition,cpu=cpu)
        self.client=Client(root,engine);self.packages=packages(engine,state,card['task'],card);self.selected=None
        self.initial_tcp=state['tcp'].copy()

    def target(self,state):
        package=self.selected if self.selected is not None else self.packages[0]
        if self.card['task']=='T1':
            if self.stage in ('approach','align'):
                return np.asarray(package['approach' if self.stage=='approach' else 'contact']).copy(),1.
            return super().target(state)
        target=np.asarray(package['contact']).copy();d=package['direction'];opening=0.
        if self.stage in ('approach','close'):
            target=np.asarray(package['approach']).copy();opening=1. if self.stage=='approach' else 0.
        elif self.stage=='align':target[:2,3]-=.013*d
        elif self.stage=='push':target[:2,3]+=np.asarray(self.card['goal_xy'])-self.initial[:2,3]
        elif self.stage in ('detach','withdraw','hold'):
            target=self.final_tcp.copy();target[:2,3]-=.04*d
            if self.stage in ('withdraw','hold'):target[2,3]+=.13;opening=1.
        return target,opening

    def observe(self,state,base):
        task=self.card['task'];t=state['time_s'];contact=bool(state['robot_object_contact']);grasp=bool(state['native_grasp'])
        self.geometry=alignment(self.engine,state,self.selected if self.selected is not None else self.packages[0])
        self.milestones['safe_precontact']|=bool(self.geometry['precontact_geometry_ready'] and state['safety_valid'])
        self.milestones['actual_contact']|=contact
        target,_=self.target(state);distance=float(np.linalg.norm(target[:3,3]-state['tcp'][:3,3]))
        if self.initial_distance is None:self.initial_distance=distance
        if self.stage=='approach':self.milestones['position_progress']|=self.initial_distance-distance>=.02
        if grasp:
            if self.grasp_since is None:self.grasp_since=t
        else:self.grasp_since=None
        displacement=float(np.linalg.norm(state['object_pose'][:2,3]-self.initial[:2,3]));rise=float(state['object_pose'][2,3]-self.initial[2,3])
        if task=='T1':self.milestones['manipulation']|=self.grasp_since is not None and t-self.grasp_since>=.3-1e-8
        else:self.milestones['manipulation']|=bool(contact and not grasp and displacement>=.01 and rise<=.005)
        corners=np.array(list(itertools.product((-.02,.02),repeat=3)))@state['object_pose'][:3,:3].T+state['object_pose'][:3,3]
        goal=bool((np.abs(corners[:,:2]-self.card['goal_xy'])<=.07).all())
        supported=bool(abs(state['object_pose'][2,3]-.46)<=.005 and np.linalg.norm(state['object_linear_velocity'])<=.02 and np.linalg.norm(state['object_angular_velocity'])<=.10)
        new=None
        if self.stage=='approach' and self.selected is not None and self.geometry['precontact_geometry_ready']:new='align' if task=='T1' else 'close'
        elif task=='T1':
            if self.stage=='align' and self.geometry['actual_contact_alignment']:new='close';self.release_tcp=state['tcp'].copy()
            elif self.stage=='close' and self.grasp_since is not None and t-self.grasp_since>=.3-1e-8:
                new='lift';self.grasp_relative=np.linalg.inv(state['object_pose'])@state['tcp']
            elif self.stage=='lift':
                if grasp and rise>=.02:
                    if self.lift_since is None:self.lift_since=t
                    if t-self.lift_since>=.3-1e-8:new='carry'
                else:self.lift_since=None
            elif self.stage=='carry' and grasp and displacement>=.10 and goal:new='lower'
            elif self.stage=='lower' and goal and supported:new='release';self.release_tcp=state['tcp'].copy()
            elif self.stage=='release' and not grasp and state['gripper_fraction']>=.85:
                new='withdraw';self.final_tcp=self.release_tcp.copy();self.final_tcp[2,3]+=.13
            elif self.stage=='withdraw' and np.linalg.norm(state['tcp'][:3,3]-state['object_pose'][:3,3])>=.08:new='hold'
        else:
            if self.stage=='close' and state['gripper_fraction']<=.05:new='align'
            elif self.stage=='align' and self.geometry['actual_contact_alignment']:new='contact'
            elif self.stage=='contact' and contact:new='push'
            elif self.stage=='push' and displacement>=.10 and goal:new='detach';self.final_tcp=state['tcp'].copy()
            elif self.stage=='detach' and not contact:new='withdraw'
            elif self.stage=='withdraw' and not contact and np.linalg.norm(state['tcp'][:3,3]-state['object_pose'][:3,3])>=.08:new='hold'
        if new:
            self.events.append(dict(time_s=t,previous=self.stage,stage=new,basis='ACTUAL_GEOMETRY_GRASP_OBJECT_TASK_EVENT',
                actual_contact=contact,actual_grasp=grasp,actual_object_displacement_m=displacement,actual_object_rise_m=rise));self.stage=new

    def plan(self,state,base):
        target,opening=self.target(state);hold=self.stage in ('close','release','hold')
        choices=self.packages if self.selected is None else [self.selected]
        if self.stage!='approach':choices=[dict(p,target=state['tcp'].copy() if hold else target) for p in choices]
        attached=bool(self.card['task']=='T1' and state['native_grasp'] and self.stage in ('lift','carry','lower'))
        poses,certificate,selected=self.client.bundle(state,choices,self.stage,self.condition,self.query_index,attached=attached,hold=hold)
        if self.selected is None:
            self.selected=next(p for p in self.packages if p['index']==selected)
            if self.card['task']=='T1':self.rotation=self.selected['contact'][:3,:3].copy()
            self.initial_distance=float(np.linalg.norm(self.selected['approach'][:3,3]-self.initial_tcp[:3,3]))
        self.geometry=alignment(self.engine,state,self.selected)
        self.query_index+=1
        self.last_plan=dict(schema='MPS2_ACTUAL_STATE_PLAN',selected_candidate=selected,stage=self.stage,
            target=self.target(state)[0].tolist(),query_index=self.query_index-1,certificate=certificate,
            physics_steps_in_planning=0,selected_MPC_policy_consumption='UNKNOWN')
        return dict(poses=poses,opening=np.full(40,opening),BASE_soft_velocity=np.zeros((40,3)),stage=self.stage,
            persistent=self.last_plan,geometry=self.geometry,milestones=dict(self.milestones),stop_reason=None,version=VERSION)
