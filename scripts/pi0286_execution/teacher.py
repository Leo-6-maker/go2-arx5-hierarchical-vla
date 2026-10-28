"""One frozen state-feedback teacher; privileged state never becomes policy input."""
import numpy as np
from scipy.spatial.transform import Rotation


class Teacher:
    phases={'approach':0,'align':0,'close':1,'lift':1,'carry':2,'lower':2,'release':3,'withdraw':3,'hold':4}

    def __init__(self,episode,task,zone):
        self.ep=episode;self.task=task;self.zone=np.asarray(zone);self.stage='approach';self.attempts=0
        self.stage_start=0.;self.ready_start=None;self.grasp_since=None;self.failed=False;self.events=[]
        self.command=episode.states[-1]['tcp'].copy();self.gripper=1.
        self.goal=None;self.place_center=None;self.terminal=None;self.grasp_pose=None
        self.prepare_grasp()

    def prepare_grasp(self):
        from mani_skill.examples.motionplanning.base_motionplanner.utils import get_actor_obb,compute_grasp_info_by_obb
        obb=get_actor_obb(self.ep.env.cubeA)
        closing=self.ep.states[-1]['tcp'][:3,1]
        info=compute_grasp_info_by_obb(obb,approaching=np.asarray([0,0,-1]),target_closing=closing,depth=.025)
        pose=self.ep.env.agent.build_grasp_pose(np.asarray([0,0,-1]),info['closing'],info['center'])
        self.grasp_pose=pose.to_transformation_matrix().copy()

    def transition(self,stage,t):
        self.events.append(dict(time_s=t,from_stage=self.stage,to_stage=stage,attempt=self.attempts))
        self.stage=stage;self.stage_start=t;self.ready_start=None

    def ready(self,flag,t,period=.1):
        if not flag:self.ready_start=None;return False
        if self.ready_start is None:self.ready_start=t
        return t-self.ready_start>=period-1e-8

    def query(self,t):
        row=self.ep.states[-1];tcp=row['tcp'];red=row['cubeA_pose'][:3];green=row['cubeB_pose'][:3]
        grasp=bool(row['is_cubeA_grasped']);target=self.grasp_pose.copy();g=1.
        if self.stage=='approach':target[2,3]+=.05
        elif self.stage=='align':pass
        elif self.stage=='close':g=0.
        elif self.stage=='lift':target[2,3]+=.10;g=0.
        elif self.stage in ('carry','lower'):
            g=0.;target[:3,3]=tcp[:3,3]+self.place_center-red
            if self.stage=='carry':target[2,3]+=.10
        elif self.stage=='release':target=self.goal.copy()
        elif self.stage=='withdraw':target=self.terminal.copy()
        elif self.stage=='hold':target=self.terminal.copy();g=1.
        xyz=np.linalg.norm(tcp[:3,3]-target[:3,3])
        angle=Rotation.from_matrix(tcp[:3,:3].T@target[:3,:3]).magnitude()
        arrived=xyz<.005 and angle<.06
        if self.stage in ('approach','align') and self.ready(arrived,t):
            self.transition('align' if self.stage=='approach' else 'close',t)
        elif self.stage=='close':
            if self.ready(grasp,t,.3):self.transition('lift',t)
            elif t-self.stage_start>1.5:
                if self.attempts<2:
                    self.attempts+=1;self.prepare_grasp();self.transition('approach',t)
                else:self.failed=True
        elif self.stage=='lift' and self.ready(grasp and red[2]>=self.ep.env.layout_spec.get('initial_red_z',.02)+.07 and arrived,t):
            self.place_center=np.r_[self.zone,.02] if self.task=='T1' else green+np.asarray([0,0,.04])
            self.transition('carry',t)
        elif self.stage=='carry' and self.ready(grasp and np.linalg.norm(red[:2]-self.place_center[:2])<.005 and arrived,t):self.transition('lower',t)
        elif self.stage=='lower' and self.ready(grasp and np.linalg.norm(red-self.place_center)<.004,t):
            self.goal=target.copy();self.transition('release',t)
        elif self.stage=='release' and self.ready(not grasp and bool(row['is_cubeA_static']),t,.3):
            self.terminal=self.goal.copy();self.terminal[2,3]+=.10;self.transition('withdraw',t)
        elif self.stage=='withdraw' and self.ready(arrived,t):self.transition('hold',t)
        if self.stage not in ('close','approach','align','hold','release','withdraw') and not grasp:
            if t-self.stage_start>.3 and self.attempts<2:
                self.attempts+=1;self.prepare_grasp();self.transition('approach',t)
            elif self.attempts>=2:self.failed=True
        if self.failed:
            target=self.command.copy();g=1.
        # Fixed reference-speed limits, same for both tasks, no physics/gain change.
        delta=target[:3,3]-self.command[:3,3];length=np.linalg.norm(delta)
        self.command[:3,3]+=delta*min(1.,.15*.05/max(length,1e-12))
        axis=Rotation.from_matrix(self.command[:3,:3].T@target[:3,:3]).as_rotvec();length=np.linalg.norm(axis)
        self.command[:3,:3]=self.command[:3,:3]@Rotation.from_rotvec(axis*min(1.,1.*.05/max(length,1e-12))).as_matrix()
        self.gripper=g
        return np.r_[self.command[:3,3],self.command[:2,:3].reshape(6),g].astype(np.float32),self.phases[self.stage]
