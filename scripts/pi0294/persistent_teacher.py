"""Explicitly adopted full-body Teacher, separate from historical CPU proposal.

Persistent reference progress and actual pad/object geometry; no WBC changes,
joint command, measured-state clipping, timed forced contact or task-score edits.
"""
import itertools
import numpy as np
from .persistent_approach_draft import PersistentApproachDraft,precontact_geometry
from .teacher import PushTeacher
from scripts.pi0291_arx5_data.teacher import Teacher as PickTeacher
from .task_plan import pilot_cards
from go2arx5_vla.trajectory.so3 import geodesic_angle

VERSION='FULLBODY_PERSISTENT_TEACHER_QUALIFICATION_V1'


def cards():
    result=[]
    for old in pilot_cards():
        c=dict(old,id='NEW_TEACHER_DEV_'+old['scene_class']+'_'+old['task'],
            duration_s=12.,split='TEACHER_DEVELOPMENT',teacher=VERSION)
        result.append(c)
    for task in ('T1','T2'):
        c=next(c for c in pilot_cards() if c['scene_class']=='W1' and c['task']==task)
        result.append(dict(c,id='NEW_TEACHER_FULL_W1_'+task,split='TEACHER_CONFIRMATION',teacher=VERSION))
    return result


def pad_corners(engine):
    signs=np.asarray(list(itertools.product((-1.,1.),repeat=3)))
    return np.asarray([engine.data.geom_xpos[g]+(signs*engine.model.geom_size[g])@
        engine.data.geom_xmat[g].reshape(3,3).T for g in engine.pad])


def geometry(engine,state,task,direction=None):
    pads=pad_corners(engine);obj=np.asarray(state['object_pose'])
    if task=='T1':
        pre=precontact_geometry(pad_corners_world=pads,object_pose=obj)
        local=(pads-obj[:3,3])@obj[:3,:3];lo=local.min(axis=1);hi=local.max(axis=1)
        order=np.argsort(local.mean(axis=1)[:,1]);neg,pos=order
        # A real open gap enclosing the actual cube, not a TCP distance gate.
        aligned=bool(hi[neg,1]<=-.02 and lo[pos,1]>=.02 and
            np.all(lo[:,0]<=.02) and np.all(hi[:,0]>=-.02) and
            np.all(lo[:,2]<.02) and np.all(hi[:,2]>-.02))
        pre['actual_contact_alignment']=aligned
    else:
        direction=np.asarray(direction);side=np.r_[-direction[1],direction[0]]
        rel=pads-obj[:3,3];long=rel[:,:,:2]@direction;across=rel[:,:,:2]@side
        front=float(long.max());table_bottom=float(pads[:,:,2].min())
        lateral=bool(across.min()<=.02 and across.max()>=-.02)
        behind=bool(front<=-.02 and front>=-.02-.06)
        above=bool(table_bottom>=.44+.004)
        pre=dict(precontact_geometry_ready=bool(lateral and behind and above and table_bottom<=.44+.10),
            actual_contact_alignment=bool(lateral and behind and .44<=table_bottom<=.44+.02),
            pad_front_object_frame_m=front,pad_table_clearance_m=table_bottom-.44)
    pre.update(actual_pad_corners_world=pads.tolist(),actual_object_world=obj.tolist(),
        full_robot_clearance='ACTUAL_NATIVE_OBSERVER_REMAINS_AUTHORITATIVE')
    return pre


class Teacher:
    def __init__(self,card,state,engine,base):
        self.card,self.engine=card,engine
        self.legacy=PickTeacher(card,state['object_pose']) if card['task']=='T1' else PushTeacher(card,state['object_pose'],engine)
        self.events=[];self.stage='approach';self.stop_reason=None;self.core=None
        self.station=np.asarray(base).copy();self.station[:3,3]+=np.array([card['base_translation_preference_m'],0.,0.])
        self.radius=float(np.linalg.norm(pad_corners(engine)-state['tcp'][:3,3],axis=-1).max())
        self.initial=np.asarray(state['object_pose']).copy();self.initial_distance=None
        self.milestones=dict(position_progress=False,safe_precontact=False,actual_contact=False,manipulation=False)
        self.grasp_since=None;self.last_core_stage=None
        self.geometry=None;self.last_state=None;self.last_base=None

    def observe(self,state,base):
        self.last_state,self.last_base=state,np.asarray(base)
        self.geometry=geometry(self.engine,state,self.card['task'],getattr(self.legacy,'direction',None))
        contact=bool(state['robot_object_contact']);self.milestones['actual_contact']|=contact
        self.milestones['safe_precontact']|=bool(self.geometry['precontact_geometry_ready'] and state['safety_valid'])
        target,_=self.legacy.target(state)
        distance=float(np.linalg.norm(target[:3,3]-state['tcp'][:3,3]))
        if self.initial_distance is None:self.initial_distance=distance
        if self.stage=='approach':self.milestones['position_progress']|=self.initial_distance-distance>=.02
        if state['native_grasp']:
            if self.grasp_since is None:self.grasp_since=state['time_s']
        else:self.grasp_since=None
        if self.card['task']=='T1':
            self.milestones['manipulation']|=self.grasp_since is not None and state['time_s']-self.grasp_since>=.3
        else:
            self.milestones['manipulation']|=bool(contact and not state['native_grasp'] and
                np.linalg.norm(state['object_pose'][:2,3]-self.initial[:2,3])>=.01 and
                state['object_pose'][2,3]-self.initial[2,3]<=.005)
        old=self.stage
        if self.stage=='approach':
            if self.geometry['precontact_geometry_ready']:
                self.legacy.stage='align' if self.card['task']=='T1' else 'close'
                self.legacy.stage_since=state['time_s']
        elif self.stage=='align':
            if self.geometry['actual_contact_alignment']:
                self.legacy.stage='close' if self.card['task']=='T1' else 'contact'
                self.legacy.stage_since=state['time_s']
        else:self.legacy.observe(state)
        self.stage=self.legacy.stage
        if self.stage!=old:
            self.events.append(dict(time_s=state['time_s'],previous=old,stage=self.stage,
                basis='ACTUAL_GEOMETRY_OR_ORIGINAL_PHYSICAL_MANIPULATION_PREDICATE',geometry=self.geometry))
        if self.core is not None and self.stage==self.last_core_stage:
            self.core.observe(time_s=state['time_s'],actual_tcp=state['tcp'],actual_BASE=base,native_safe=state['safety_valid'])
            if self.stage=='approach':self.stop_reason=self.core.stop_reason

    def plan(self,state,base):
        target,opening=self.legacy.target(state)
        if self.core is None or self.stage!=self.last_core_stage:
            self.core=PersistentApproachDraft(task=self.card['task'],actual_tcp=state['tcp'],stage_target=target,
                actual_BASE=base,station_world=self.station,pad_radius_m=self.radius)
            # Only an explicitly entered NEW stage starts its persistent route.
            # No per-query reanchoring; manipulation targets use a direct path.
            if self.stage!='approach':self.core.route=[target.copy()]
            if self.stage in ('contact','push','detach') and self.card['task']=='T2':
                from dataclasses import replace
                self.core.config=replace(self.core.config,linear_speed_m_s=.04)
            self.core.observe(time_s=state['time_s'],actual_tcp=state['tcp'],actual_BASE=base,native_safe=state['safety_valid'])
            self.last_core_stage=self.stage
        result=self.core.plan(time_s=state['time_s'],actual_tcp=state['tcp'],actual_BASE=base)
        return dict(poses=result['H40_world_poses'],opening=np.full(40,opening),
            BASE_soft_velocity=result['BASE_soft_velocity'],stage=self.stage,
            persistent=self.core.snapshot(),geometry=self.geometry,milestones=dict(self.milestones),
            stop_reason=self.stop_reason,version=VERSION)
