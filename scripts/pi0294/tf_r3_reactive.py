"""Actual-object task guidance, frozen native plant and original hard gates.

No integration, qpos write, compensation table, joint action or neural model.
Reference increments use the existing 2 mm working-margin length; phase changes
use actual contacts and the original object-task geometry.
"""
import hashlib,itertools,json
from pathlib import Path
import numpy as np
from .task_gripper_contract import MARGIN
from .tf_r3_source import sample

VERSION='TF_R3_GEOMETRY_REACTIVE_PUSH_V1'
CLEARANCE=.06*.05
STEP=MARGIN


def reference(state,previous,pad,card,initial_object,mode,*,local_time_factor=1.):
    """Pure deterministic reference; callers preserve the measured state."""
    previous=np.asarray(previous,dtype=float);pad=np.asarray(pad,dtype=float)
    obj=np.asarray(state['object_pose']);goal=np.asarray(card['goal_xy'],dtype=float)
    direction=np.r_[goal-np.asarray(card['source_xy']),0.];direction/=np.linalg.norm(direction)
    rotation=previous[:3,:3];pad_rotation=pad[:3,:3]
    size=np.array([.02,.003,.017]);cube_size=np.full(3,.02)
    cube_extent=float(abs(obj[:3,:3].T@direction)@cube_size)
    pad_extent=float(abs(pad_rotation.T@direction)@size)
    table=float(card['table_top_m']);pad_z_extent=float(abs(pad_rotation[2,:])@size)
    corners=np.asarray(list(itertools.product((-.02,.02),repeat=3)))@obj[:3,:3].T+obj[:3,3]
    in_zone=bool((abs(corners[:,:2]-goal)<=.07).all())
    moved=float(np.linalg.norm(obj[:2,3]-np.asarray(initial_object)[:2,3]))
    mode=dict(mode)
    if mode.get('withdraw_target') is None and in_zone and moved>=.10:
        mode['withdraw_target']=(pad[:3,3]+np.array([0.,0.,.08])-.04*direction).tolist()
    if mode.get('withdraw_target') is not None:
        desired=np.asarray(mode['withdraw_target']);stage='withdraw'
        if np.linalg.norm(state['tcp'][:3,3]-obj[:3,3])>=.08 and not state['robot_object_contact']:
            mode['hold']=True
    else:
        desired=obj[:3,3]-direction*(cube_extent+pad_extent)
        desired[2]=max(desired[2],table+pad_z_extent+CLEARANCE)
        stage='contact'
        if state['robot_object_contact']:
            desired[:2]=goal-direction[:2]*(cube_extent+pad_extent);stage='push'
    delta=desired-pad[:3,3];length=float(np.linalg.norm(delta))
    if local_time_factor not in (1.,1.25,1.5,2.):raise ValueError('ORIGINAL_FOUR_TIMING_FACTORS_ONLY')
    step=np.zeros(3) if mode.get('hold') else delta*min(1.,STEP/max(length,1e-15))/(local_time_factor**2)
    if mode.get('hold'):stage='hold'
    corrected=card.get('reference_grid')=='PREFIX8_FULL040_END_SAMPLES_V2'
    u=np.minimum((np.arange(40,dtype=float)+1)/8,1.) if corrected else np.minimum(np.arange(40,dtype=float)/7,1.)
    ramp=10*u**3-15*u**4+6*u**5
    poses=np.repeat(previous[None],40,axis=0);poses[:,:3,3]+=ramp[:,None]*step
    proof=dict(stage=stage,desired_pad_world=desired.tolist(),actual_pad_world=pad[:3,3].tolist(),
        reference_step_world=step.tolist(),reference_step_bound_m=STEP,object_all_corners_in_original_zone=in_zone,
        actual_object_horizontal_displacement_m=moved,future_tail='HOLD_AFTER_PREFIX8',
        actual_contact=bool(state['robot_object_contact']),local_time_factor=local_time_factor,reference_grid=card.get('reference_grid','PREFIX8_J0_OVER7_V1'),no_joint_bias_calibration=True)
    return poses,stage,mode,proof


def overhead(model,state,engine,card):
    """One measured orientation, pad envelopes and original 8cm TCP clearance."""
    import mujoco
    d=mujoco.MjData(model);d.qpos[:]=state['qpos'];d.qvel[:]=state['qvel'];d.ctrl[:]=state['ctrl']
    for joint in engine.fingers:d.qpos[model.jnt_qposadr[joint]]=.022
    mujoco.mj_forward(model,d);site=engine.tcp;tcp=np.eye(4);tcp[:3,:3]=d.site_xmat[site].reshape(3,3);tcp[:3,3]=d.site_xpos[site]
    pad=np.eye(4);g=engine.pad[0];pad[:3,:3]=d.geom_xmat[g].reshape(3,3);pad[:3,3]=d.geom_xpos[g]
    relative=np.linalg.inv(tcp)@pad;direction=np.r_[np.asarray(card['goal_xy'])-np.asarray(card['source_xy']),0.];direction/=np.linalg.norm(direction)
    center=np.asarray(state['object_pose'])[:3,3]-.04*direction+np.array([0.,0.,.08])
    pose=tcp.copy();pose[:3,3]=center-pose[:3,:3]@relative[:3,3]
    return dict(index=0,task='T2',approach=pose.copy(),contact=pose.copy(),direct_join=False,
        entry_contact_stage='approach',active_duration_s=0,source_id=VERSION,source_program_sha256=card['locked_source_program_sha256'])


class Teacher:
    def __init__(self,card,state,engine,base,*,root):
        from .tf_r3_client import Client
        self.root=Path(root);self.card=card;self.engine=engine;self.client=Client(root,engine);self.condition=card['id']
        self.initial=state['object_pose'].copy();self.station=base.copy();self.initial_tcp=state['tcp'].copy()
        self.entry=None;self.entry_program=None;self.mode={};self.stage='approach';self.geometry={};self.events=[];self.stop_reason=None;self.query_index=0
        self.milestones=dict(actual_contact=False,manipulation=False,position_progress=False)

    def note_executed(self,*args,**kwargs):
        from .tf_r3_teacher import Teacher as Parent
        return Parent.note_executed(self,*args,**kwargs)

    def observe(self,state,base):
        self.milestones['actual_contact']|=bool(state['robot_object_contact'])
        self.milestones['manipulation']|=bool(state['robot_object_contact'] and np.linalg.norm(state['object_pose'][:2,3]-self.initial[:2,3])>=.10)
        self.milestones['position_progress']|=bool(np.linalg.norm(state['tcp'][:3,3]-self.initial_tcp[:3,3])>=.02)
        self.geometry=dict(teacher=VERSION,actual_contact=bool(state['robot_object_contact']),actual_object_displacement_m=float(np.linalg.norm(state['object_pose'][:2,3]-self.initial[:2,3])),source_progress_basis='ACTUAL_OBJECT_AND_PAD_NO_EXPERT_PHASE')

    def plan(self,state,base):
        import mujoco
        from scripts.pi0292.durable import write
        from .tf_r3_entry import retime
        selected=overhead(self.engine.model,state,self.engine,self.card)
        if self.entry is None:
            _,certificate,_=self.client.bundle(state,[selected],'approach',self.condition+'_WARMUP',0)
            route=self.client.last_route;q=np.asarray(route['position']);self.entry=retime(dict(time=np.asarray(route['time'])*certificate['uniform_time_scale'],poses=self.client.fk_path(state,q)),q)
            self.entry_program=dict(time=self.entry['time'],poses=self.entry['poses'],gripper=np.full(len(self.entry['time']),.5))
            if self.entry['time'][-1]>32:raise ValueError('FIXED96_ENTRY_LEAVES_INSUFFICIENT_TASK_TIME')
            folder=self.root/'factory/episodes'/self.condition;np.savez_compressed(folder/'TASKFRAME_ENTRY.npz',**self.entry_program)
            write(folder/'TASKFRAME_ENTRY_BINDING.json',dict(version=VERSION,certificate=certificate,entry_end_s=float(self.entry['time'][-1]),overhead_world=selected['contact'].tolist(),nominal_gripper_m=.022,source='ACTUAL_STARTUP_GEOMETRY',source_EEF_changed=True,entry_timing=self.entry['entry_timing'],WBC_changed=False))
        from .mp_client import PlannerFailure
        corrected=self.card.get('reference_grid')=='PREFIX8_FULL040_END_SAMPLES_V2'
        factors=(1.,1.25,1.5,2.) if corrected else (1.,)
        previous_mode=dict(self.mode);rejections=[];last_error=None
        for attempt,factor in enumerate(factors):
            if state['time_s']<self.entry['time'][-1]-1e-12:
                if attempt:raise last_error
                poses,_=sample(self.entry_program,state['time_s']+np.arange(40)*.05);stage='approach';candidate_mode=previous_mode;proof=dict(stage=stage,entry_end_s=float(self.entry['time'][-1]),future_after_entry='HOLD_UNTIL_NEXT_ACTUAL_OBSERVATION')
            else:
                history=self.client.executed_reference_history;previous=np.asarray(history[-1]['pose']) if history else self.entry['poses'][-1]
                d=self.engine.data;pad=np.eye(4);g=self.engine.pad[0];pad[:3,:3]=d.geom_xmat[g].reshape(3,3);pad[:3,3]=d.geom_xpos[g]
                poses,stage,candidate_mode,proof=reference(state,previous,pad,self.card,self.initial,previous_mode,local_time_factor=factor)
            selected['target']=poses[0];bundle_index=self.query_index*4+attempt if corrected else self.query_index
            try:
                checked,certificate,_=self.client.bundle(state,[selected],stage,self.condition,bundle_index,frozen_poses=poses,frozen_stages=[stage]*40,attached=False)
            except PlannerFailure as exc:
                routes=exc.evidence.get('routes',[])
                if not corrected or not routes or not all(r['status'] in ('FROZEN_SOURCE_JOINT_ACCELERATION','FROZEN_SOURCE_JOINT_SPEED','FROZEN_SOURCE_RATE_REJECT') for r in routes):raise
                rejections.append(dict(attempt=attempt,factor=factor,bundle_index=bundle_index,reason=str(exc),routes=routes));last_error=exc
                if state['time_s']<self.entry['time'][-1]-1e-12 or not np.linalg.norm(proof['reference_step_world']):raise
                continue
            if not np.array_equal(checked,poses):raise ValueError('REACTIVE_ACTIVE_REFERENCE_CHANGED')
            self.mode=candidate_mode
            if self.stage!=stage:self.events.append(dict(time_s=float(state['time_s']),previous=self.stage,stage=stage,basis='ACTUAL_GEOMETRY_REFERENCE',actual_contact=bool(state['robot_object_contact'])))
            self.stage=stage;break
        else:raise last_error
        proof.update(planner_bundle_index=bundle_index,preinstallation_rate_rejections=rejections)
        proof.update(query_index=self.query_index,task_time_s=float(state['time_s']),mode=self.mode,previous_reference_world=poses[0].tolist(),reference_sha256=hashlib.sha256(poses.astype('<f8').tobytes()).hexdigest())
        folder=self.root/'factory/episodes'/self.condition
        with (folder/'REACTIVE_REFERENCE_PLAN.jsonl').open('a') as f:f.write(json.dumps(proof)+'\n')
        persistent=dict(schema=VERSION,source_id=VERSION,source_program_sha256=self.card['locked_source_program_sha256'],consumed_task_time_s=float(state['time_s']),certificate=certificate,query_index=self.query_index,planner_bundle_index=bundle_index,reference_proof=proof)
        self.query_index+=1
        return dict(poses=poses,opening=np.full(40,.5),BASE_soft_velocity=np.zeros((40,3)),stage=stage,persistent=persistent,geometry=self.geometry,milestones=dict(self.milestones),stop_reason=None,version=VERSION)
