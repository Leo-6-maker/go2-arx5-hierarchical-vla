"""Predeclared six-instruction scorer, privileged truth never a policy input.

This module does not admit a scene, parse a native ABI or integrate physics.
Success requires a complete 2ms state stream, not sparse atom snapshots.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np

INSTRUCTIONS=('HOLD-red','HOLD-blue','PLACE-red-left','PLACE-red-right','PLACE-blue-left','PLACE-blue-right')
DT=.002
LIFT=.020
HOLD_SECONDS=1.
BASE_MOTION=.080
STABLE_SPEED=.020
OPEN_FRACTION=.90
RETREAT=.050
TABLE_TOLERANCE=.005


@dataclass(frozen=True)
class Task:
    instruction:str
    operation:str
    color:str
    destination:str|None

    @classmethod
    def parse(cls,instruction):
        if instruction not in INSTRUCTIONS:raise ValueError('ONE_OF_SIX_FIXED_V6_INSTRUCTIONS_REQUIRED')
        parts=instruction.split('-')
        return cls(instruction,parts[0],parts[1],parts[2]if len(parts)==3 else None)


def footprint_inside(pose,half_size,zone):
    pose=np.asarray(pose,float);half_size=np.asarray(half_size,float)
    center=np.asarray(zone['center_world_XY'],float);half=np.asarray(zone['half_size_XY'],float)
    if pose.shape!=(4,4)or half_size.shape!=(3,)or center.shape!=(2,)or half.shape!=(2,)or(half_size<=0).any()or(half<=0).any():
        raise ValueError('PREDECLARED_BOX_AND_WORLD_RECTANGLE_REQUIRED')
    corners=np.asarray(list(product((-1.,1.),repeat=3)))*half_size
    world=corners@pose[:3,:3].T+pose[:3,3]
    return bool(np.all(abs(world[:,:2]-center)<=half+1e-12))


class Score:
    def __init__(self,task,initial,geometry):
        self.task=Task.parse(task);self.initial=initial;self.geometry=geometry
        if set(initial['objects'])!={'red','blue'}or set(geometry['zones'])!={'left','right'}:
            raise ValueError('BOTH_OBJECTS_AND_BOTH_DESTINATIONS_IN_EVERY_SCENE_REQUIRED')
        direction=np.asarray(geometry['direction_world_XY'],float)
        if direction.shape!=(2,)or not np.isclose(np.linalg.norm(direction),1.,rtol=0,atol=1e-12):raise ValueError('FIXED_UNIT_TASK_DIRECTION_REQUIRED')
        self.direction=direction;self.start=int(initial['native_step']);self.last=None;self.gaps=0;self.states=0
        self.ever_grasp=False;self.ever_lift=False;self.wrong_grasp=False;self.wrong_contact=False
        self.unsafe_states=0
        self.hold_since=None;self.release_since=None;self.longest_hold=0.;self.longest_release=0.;self.maximum_lift=0.
        self.events={};self.latest=None;self.maximum_base_motion=0.

    def update(self,sample):
        step=sample['native_step'];time_s=sample['time_s']
        if type(step)is not int or step<self.start or not np.isclose(time_s,step*DT,rtol=0,atol=1e-8):raise ValueError('ACTUAL_INTEGER_2MS_NATIVE_CLOCK_REQUIRED')
        if step-self.start>(30000 if self.task.operation=='HOLD'else 45000):raise ValueError('FIXED_HOLD60_OR_PLACE90_NATIVE_BUDGET_REQUIRED')
        if self.last is not None and step<=self.last:raise ValueError('NO_DUPLICATE_OR_REORDERED_NATIVE_SCORE_STATES')
        if(self.last is None and step!=self.start)or(self.last is not None and step!=self.last+1):
            self.gaps+=1;self.hold_since=self.release_since=None
        self.last=step;self.states+=1
        safe=sample.get('safety_valid') is True
        self.unsafe_states+=not safe
        if set(sample['objects'])!={'red','blue'}:raise ValueError('TWO_ACTUAL_OBJECT_STATES_REQUIRED')
        target=sample['objects'][self.task.color];other=sample['objects']['blue'if self.task.color=='red'else'red']
        pose=np.asarray(target['pose'],float);initial=np.asarray(self.initial['objects'][self.task.color]['pose'],float)
        tcp=np.asarray(sample['tcp'],float);base=np.asarray(sample['base_pose'],float)
        velocity=np.asarray(target['linear_velocity'],float);opening=float(sample['gripper_open_fraction'])
        if any(a.shape!=(4,4)for a in(pose,initial,tcp,base))or velocity.shape!=(3,)or not all(np.isfinite(x).all()for x in(pose,initial,tcp,base,velocity))or not np.isfinite(opening):
            raise ValueError('FINITE_ACTUAL_POSES_VELOCITY_AND_CALIBRATED_OPENING_REQUIRED')
        grasp=target['native_grasp']is True;other_grasp=other['native_grasp']is True
        lift=float(pose[2,3]-initial[2,3]);self.maximum_lift=max(self.maximum_lift,lift)
        self.ever_grasp|=grasp;self.ever_lift|=grasp and lift>=LIFT;self.wrong_grasp|=other_grasp;self.wrong_contact|=other['hand_contact']is True
        holding=grasp and lift>=LIFT
        if holding:
            if self.hold_since is None:self.hold_since=time_s
            self.longest_hold=max(self.longest_hold,time_s-self.hold_since)
        else:self.hold_since=None
        motion=float((base[:2,3]-np.asarray(self.initial['base_pose'])[:2,3])@self.direction)
        self.maximum_base_motion=max(self.maximum_base_motion,motion)
        retreat=float(np.linalg.norm(tcp[:3,3]-pose[:3,3]));inside=None;stable=False;released=not grasp and not target['hand_contact']
        if self.task.destination is not None:
            zone=self.geometry['zones'][self.task.destination];inside=footprint_inside(pose,self.geometry['object_half_sizes'][self.task.color],zone)
            # Actual surface contact is required; an airborne object passing
            # through the region does not become a successful PLACE.
            half=np.asarray(self.geometry['object_half_sizes'][self.task.color]);bottom=float(pose[2,3]-np.sum(abs(pose[2,:3])*half))
            surface=abs(bottom-float(self.geometry['table_top_world_z']))<=TABLE_TOLERANCE and target['table_contact']is True
            stable=bool(self.ever_lift and inside and surface and released and opening>=OPEN_FRACTION and retreat>=RETREAT and np.linalg.norm(velocity)<STABLE_SPEED)
            if stable:
                if self.release_since is None:self.release_since=time_s
                self.longest_release=max(self.longest_release,time_s-self.release_since)
            else:self.release_since=None
        for name,value in(('target_grasp',grasp),('target_lift20mm',grasp and lift>=LIFT),('wrong_object_grasp',other_grasp),('wrong_object_contact',other['hand_contact']),
            ('correct_region',inside is True),('released',released and self.ever_lift),('retreated50mm',retreat>=RETREAT and self.ever_lift),('base80mm',motion>=BASE_MOTION)):
            if value and name not in self.events:self.events[name]=dict(native_step=step,native_time_s=time_s)
        qualified_hold=holding and self.hold_since is not None and time_s-self.hold_since>=HOLD_SECONDS-1e-10
        qualified_place=stable and self.release_since is not None and time_s-self.release_since>=HOLD_SECONDS-1e-10
        success=bool(self.unsafe_states==0 and not self.wrong_grasp and self.gaps==0 and motion>=BASE_MOTION and
            (qualified_hold if self.task.operation=='HOLD'else qualified_place))
        self.latest=dict(task_success=success,native_step=step,base_directional_motion_m=motion,lift_m=lift,inside_selected_region=inside,
            target_grasp=grasp,wrong_object_grasp_ever=self.wrong_grasp,wrong_object_contact_ever=self.wrong_contact,released=released,
            gripper_open_fraction=opening,TCP_retreat_m=retreat,stable_release_condition=stable)
        return self.latest

    def result(self):
        if self.latest is None:raise ValueError('NO_NATIVE_SCORE_DENOMINATOR')
        return dict(schema='SLA_V6_MULTITASK_NATIVE_SCORE',instruction=self.task.instruction,**self.latest,
            states=self.states,missing_native_intervals=self.gaps,unsafe_states=self.unsafe_states,ever_target_grasp=self.ever_grasp,ever_target_lift20mm=self.ever_lift,
            maximum_lift_m=self.maximum_lift,longest_grasp_lift_hold_s=self.longest_hold,longest_stable_released_retreat_s=self.longest_release,
            events=self.events,task_privileged_truth_in_student_input=False,clock='ACTUAL_2MS_NATIVE_STREAM')
