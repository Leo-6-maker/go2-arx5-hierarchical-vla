"""Object skills with actual geometric/contact events and MPlib planning.

OBB-axis selection below is adapted from ManiSkill's compute_grasp_info_by_obb,
62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3 (Apache-2.0). SAPIEN access and Panda
finger constants are replaced by native object and measured ARX5 pad geometry.
See MANISKILL_COMPONENT_NOTICE.md. No joint command or controller correction.
"""
import itertools
import numpy as np
from .mp_runtime import VERSION,cards
from .mp_client import Client,PlannerFailure
from .component_teacher import geometry,pad_corners
from .teacher import PushTeacher


def obb_closing(object_pose,target_closing):
    approaching=np.array([0.,0.,-1.]);axes=np.asarray(object_pose)[:3,:3];extents=np.full(3,.04)
    inds0=np.argsort(np.abs(approaching@axes));inds1=np.argsort(extents[inds0[:-1]])
    first,second=inds0[:-1][inds1]
    if .99<extents[first]/extents[second]<1.01 and abs(target_closing@axes[:,first])<abs(target_closing@axes[:,second]):
        first=second
    closing=axes[:,first].copy()
    if target_closing@closing<0:closing=-closing
    closing-=float(approaching@closing)*approaching;closing/=np.linalg.norm(closing)
    return approaching,closing


class Teacher:
    def __init__(self,card,state,engine,base,*,root,condition=None,cpu=False):
        self.card,self.engine,self.root=card,engine,root;self.condition=condition or card['id'];self.cpu=cpu
        self.client=Client(root,engine);self.initial=np.asarray(state['object_pose']).copy()
        self.stage='approach';self.events=[];self.stop_reason=None;self.station=np.asarray(base).copy()
        self.query_index=0;self.geometry=None;self.grasp_since=None;self.lift_since=None;self.grasp_relative=None
        self.release_tcp=None;self.final_tcp=None;self.initial_distance=None;self.initial_grasp_rotation=None
        self.milestones=dict(position_progress=False,safe_precontact=False,actual_contact=False,manipulation=False)
        if card['task']=='T2':self.push=PushTeacher(card,state['object_pose'],engine)
        else:
            approach,closing=obb_closing(state['object_pose'],state['tcp'][:3,1]);self.rotation=np.column_stack([approach,closing,np.cross(approach,closing)])
            local=[];support=[]
            for side,g in enumerate(engine.pad):
                center=state['tcp'][:3,:3].T@(engine.data.geom_xpos[g]-state['tcp'][:3,3])
                center[1]-=(1 if side==0 else -1)*state['finger_q'][side];local.append(center)
                axes=state['tcp'][:3,:3].T@engine.data.geom_xmat[g].reshape(3,3)
                support.append(float(np.abs(approach@self.rotation@axes)@engine.model.geom_size[g]))
            self.pad_center_tcp=np.mean(local,axis=0);self.grasp_depth=float(min(support))
            self.initial_grasp_rotation=self.rotation.copy()
        self.last_plan=None

    def target(self,state):
        if self.card['task']=='T2':
            self.push.stage=self.stage
            if self.final_tcp is not None:self.push.final_tcp=self.final_tcp
            return self.push.target(state)
        obj=np.asarray(state['object_pose']);target=np.eye(4);target[:3,:3]=self.rotation
        center=obj[:3,3]+np.array([0,0,1])*(.02-min(self.grasp_depth,.02))
        target[:3,3]=center-self.rotation@self.pad_center_tcp
        opening=1.
        if self.stage=='approach':target[2,3]+=.08
        elif self.stage in ('close','verify_grasp'):
            if self.release_tcp is not None:target=self.release_tcp.copy()
            else:target=state['tcp'].copy()
            opening=0.
        elif self.stage in ('lift','carry','lower'):
            destination=self.initial.copy()
            destination[2,3]=.46+(.08 if self.stage!='lower' else 0.)
            if self.stage!='lift':destination[:2,3]=self.card['goal_xy']
            target=destination@self.grasp_relative;opening=0.
        elif self.stage=='release':target=self.release_tcp.copy()
        elif self.stage in ('withdraw','hold'):target=self.final_tcp.copy()
        return target,opening

    def observe(self,state,base):
        task=self.card['task'];t=state['time_s'];contact=bool(state['robot_object_contact']);grasp=bool(state['native_grasp'])
        self.geometry=geometry(self.engine,state,task,None if task=='T1' else self.push.direction)
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
        if self.stage=='approach' and self.geometry['precontact_geometry_ready']:new='align' if task=='T1' else 'close'
        elif task=='T1':
            if self.stage=='align' and self.geometry['actual_contact_alignment']:
                new='close';self.release_tcp=state['tcp'].copy()
            elif self.stage=='close' and self.grasp_since is not None and t-self.grasp_since>=.3-1e-8:
                new='lift';self.grasp_relative=np.linalg.inv(state['object_pose'])@state['tcp']
            elif self.stage=='lift':
                if grasp and rise>=.02:
                    if self.lift_since is None:self.lift_since=t
                    if t-self.lift_since>=.3-1e-8:new='carry'
                else:self.lift_since=None
            elif self.stage=='carry' and grasp and displacement>=.10 and goal:new='lower'
            elif self.stage=='lower' and goal and supported:
                new='release';self.release_tcp=state['tcp'].copy()
            elif self.stage=='release' and not grasp and state['gripper_fraction']>=.85:
                new='withdraw';self.final_tcp=self.release_tcp.copy();self.final_tcp[2,3]+=.13
            elif self.stage=='withdraw' and np.linalg.norm(state['tcp'][:3,3]-state['object_pose'][:3,3])>=.08:new='hold'
        else:
            if self.stage=='close' and state['gripper_fraction']<=.05:new='align'
            elif self.stage=='align' and self.geometry['actual_contact_alignment']:new='contact'
            elif self.stage=='contact' and contact:new='push'
            elif self.stage=='push' and displacement>=.10 and goal:
                new='detach';self.final_tcp=state['tcp'].copy()
            elif self.stage=='detach' and not contact:new='withdraw'
            elif self.stage=='withdraw' and not contact and np.linalg.norm(state['tcp'][:3,3]-state['object_pose'][:3,3])>=.08:new='hold'
        if new:
            self.events.append(dict(time_s=t,previous=self.stage,stage=new,basis='ACTUAL_GEOMETRY_GRASP_OBJECT_TASK_EVENT',
                actual_contact=contact,actual_grasp=grasp,actual_object_displacement_m=displacement,actual_object_rise_m=rise))
            self.stage=new

    def plan(self,state,base):
        target,opening=self.target(state);speed=.04 if self.card['task']=='T2' and self.stage in ('contact','push','detach') else .06
        hold=self.stage in ('close','release','hold')
        if hold:target=state['tcp'].copy()
        attached=bool(self.card['task']=='T1' and state['native_grasp'] and self.stage in ('lift','carry','lower'))
        response=self.client.call(state,target,self.stage,self.condition,self.query_index,cpu=self.cpu,attached=attached,hold=hold)
        poses,certificate=self.client.cartesian(state,response,speed=speed,attached=attached,stage=self.stage)
        self.query_index+=1
        self.last_plan=dict(schema='MANISKILL_ACTUAL_STATE_PLAN_V1',target=target.tolist(),stage=self.stage,
            query_index=self.query_index-1,certificate=certificate,backend_status=response['status'],
            physics_steps_in_planning=0,selected_MPC_policy_consumption='UNKNOWN')
        return dict(poses=poses,opening=np.full(40,opening),BASE_soft_velocity=np.zeros((40,3)),stage=self.stage,
            persistent=self.last_plan,geometry=self.geometry,milestones=dict(self.milestones),stop_reason=None,version=VERSION)
