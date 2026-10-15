"""Declared source path, actual progress, fresh full-body planning; zero BASE intent."""
from pathlib import Path
import numpy as np
from .night_r1_plan_runtime import VERSION,FOLDER,read,sha
from .night_r1_source import load,sample,with_entry,packages_for
from .night_r1_client import Client,PlannerFailure
from scripts.pi0292.durable import write


class Teacher:
    def __init__(self,card,state,engine,base,*,root):
        self.root=Path(root);self.card=card;self.engine=engine;self.client=Client(root,engine);self.condition=card['id']
        self.record=next(r for r in read(self.root/FOLDER/'PROGRAMS.json')['rows'] if r['id']==card['locked_source_id'])
        self.program=load(self.root,self.record);self.station=base.copy();self.initial=state['object_pose'].copy();self.initial_tcp=state['tcp'].copy()
        self.stage='approach';self.geometry={};self.entry=None;self.combined=None;self.query_index=0;self.events=[];self.stop_reason=None
        self.milestones=dict(actual_contact=False,manipulation=False,position_progress=False)

    def note_executed(self,intent,prefix,native_step,native_end_time):
        from scripts.pi0292.eval_contract import matrix10
        history=self.client.executed_reference_history
        expected_end=intent['source_mpc_s']+(prefix+1)*.05
        if not 0<=prefix<8 or abs(native_end_time-expected_end)>1e-8:raise ValueError('ACTUAL_EXECUTED_PREFIX_CLOCK')
        if history and native_step<=history[-1]['native_step']:raise ValueError('REFERENCE_PREFIX_REPLAY')
        history.append(dict(pose=matrix10(np.asarray(intent['action_EEF13'][prefix])[:10]),wire_sha256=intent['wire_sha256'],
            prefix=int(prefix),native_step=int(native_step),native_end_time=float(native_end_time)))

    def stage_at(self,at):
        if self.entry is None or at<self.entry['time'][-1]-1e-12:return 'approach' if self.card['task']=='T1' else 'contact'
        elapsed=at-self.entry['time'][-1];stage=self.record['events'][0]['stage']
        for event in self.record['events']:
            if event['program_time_s']<=elapsed+1e-12:stage=event['stage']
        return stage

    def observe(self,state,base):
        previous=self.stage;self.stage=self.stage_at(state['time_s'])
        self.milestones['actual_contact']|=bool(state['robot_object_contact'])
        distance=float(np.linalg.norm(state['object_pose'][:2,3]-self.initial[:2,3]));rise=float(state['object_pose'][2,3]-self.initial[2,3])
        self.milestones['manipulation']|=bool(state['native_grasp'] and rise>=.02) if self.card['task']=='T1' else bool(state['robot_object_contact'] and distance>=.10)
        self.milestones['position_progress']|=bool(np.linalg.norm(state['tcp'][:3,3]-self.initial_tcp[:3,3])>=.02)
        self.geometry=dict(source_id=self.record['id'],source_phase=self.stage,actual_contact=bool(state['robot_object_contact']),actual_grasp=bool(state['native_grasp']),
            actual_object_displacement_m=distance,actual_object_rise_m=rise,source_progress_basis='CONSUMED_PHYSICAL_PREFIX_TIME_ONLY')
        if previous!=self.stage:self.events.append(dict(time_s=float(state['time_s']),previous=previous,stage=self.stage,basis='DECLARED_SOURCE_CLOCK',actual_contact=bool(state['robot_object_contact'])))

    def plan(self,state,base):
        selected=next(p for p in packages_for(self.root,self.card['task']) if p['source_id']==self.record['id'])
        if self.entry is None:
            _,certificate,_=self.client.bundle(state,[selected],'approach',self.card['id']+'_WARMUP',0)
            route=self.client.last_route;t=np.asarray(route['time'])*certificate['uniform_time_scale'];q=np.asarray(route['position'])
            self.entry=dict(time=t,poses=self.client.fk_path(state,q));self.combined=with_entry(self.program,self.entry)
            if self.combined['time'][-1]>self.card['duration_s']-4:raise PlannerFailure('NIGHT_PREDECLARED_DURATION_DOES_NOT_FIT',certificate)
            folder=self.root/FOLDER/'episodes'/self.card['id']
            np.savez_compressed(folder/'ACTUAL_COMBINED_PROGRAM.npz',**self.combined)
            write(folder/'ACTUAL_ENTRY_BINDING.json',dict(source_id=self.record['id'],source_program_sha256=self.record['sha256'],actual_warmup_certificate=certificate,
                entry_duration_s=float(t[-1]),activity_duration_s=self.record['active_duration_s'],entry_and_activity_s=float(self.combined['time'][-1]),actual_packet_time_s=float(state['time_s']),base_intent=[0,0,0]))
        at=float(state['time_s'])+np.arange(40)*.05;poses,opening=sample(self.combined,at);stages=[self.stage_at(t) for t in at]
        attached=bool(self.card['task']=='T1' and state['native_grasp'] and self.stage in ('lift','carry','lower'))
        package=dict(selected,target=poses[0]);validated,cert,_=self.client.bundle(state,[package],self.stage,self.condition,self.query_index,
            attached=attached,frozen_poses=poses,frozen_stages=stages)
        if not np.array_equal(validated,poses):raise ValueError('NIGHT_ACTIVE_PATH_VALUES_CHANGED')
        self.query_index+=1
        persistent=dict(schema='NIGHT_DECLARED_SOURCE_AND_ENTRY',source_id=self.record['id'],source_program_sha256=self.record['sha256'],consumed_task_time_s=float(state['time_s']),
            entry_duration_s=float(self.entry['time'][-1]),source_elapsed_s=max(0.,float(state['time_s']-self.entry['time'][-1])),source_phase=self.stage,certificate=cert,query_index=self.query_index-1)
        return dict(poses=poses,opening=opening,BASE_soft_velocity=np.zeros((40,3)),stage=self.stage,persistent=persistent,geometry=self.geometry,milestones=dict(self.milestones),stop_reason=None,version=VERSION)
