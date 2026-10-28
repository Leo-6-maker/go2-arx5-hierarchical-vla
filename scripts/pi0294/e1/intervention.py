"""One-way student-to-qualified-expert switch at an original S1 rejection."""
from pathlib import Path
import uuid

import numpy as np

from scripts.pi0292.durable import read,write
from scripts.pi0292.relay_protocol import unseal
from scripts.pi0292.eval_contract import gate_action
from go2arx5_vla.task_data.codec import pose10
from scripts.pi0294.native_dataset import sha
from .artifacts import atomic_npz


class InterventionTeacher:
    def __init__(self,student,engine,root,folder,card):
        self.student,self.expert=student,student.teacher
        self.engine,self.root,self.folder,self.card=engine,Path(root),Path(folder),card
        qualification=read(self.root/'evidence/ORACLE_RECOVERY_RESULTS.json')['tasks'][card['task']]
        if qualification['status']!='RECOVERY_QUALIFIED_FOR_E1':raise PermissionError('E1_TASK_SPECIFIC_RECOVERY_REQUIRED')
        self.mode='STUDENT';self.anchor=None;self.expert_acks=[];self.invalid_expert=None

    def __getattr__(self,name):return getattr(self.student,name)
    def observe(self,state):self.expert.observe(state)
    def plan(self,state):
        if self.mode=='STUDENT':return self.student.plan(state)
        plan=self.expert.plan(state)
        action=np.stack([pose10(p,float(g)) for p,g in zip(plan['poses'],plan['opening'])])
        gate=gate_action(action,pose10(state['tcp'],state['gripper_fraction']),state['time_s'],str(uuid.uuid4()))
        if gate['decision']=='REJECT':
            self.invalid_expert=dict(reason='ORIGINAL_REFERENCE_REJECT',gate=gate,tick=self.engine.steps)
            write(self.folder/'INVALID_EXPERT.json',self.invalid_expert);raise PermissionError('E1_QUALIFIED_EXPERT_REFERENCE_REJECT')
        plan['source']='E1_ACTUAL_INTERVENTION_EXPERT';return plan
    def intervene(self,state,forecast):
        if self.mode!='STUDENT':
            self.invalid_expert=dict(reason='ORIGINAL_EXPERT_S1_REJECT',forecast=forecast,tick=self.engine.steps)
            write(self.folder/'INVALID_EXPERT.json',self.invalid_expert);return False
        pending=self.student.pending
        if pending is None or pending['command']['tick']!=self.engine.steps:raise ValueError('E1_INTERVENTION_STUDENT_PROGRESS_MISMATCH')
        sequence=self.engine.steps//200;packet=read(self.folder/'relay/observations'/f'{sequence:03d}.json')
        if not np.array_equal(unseal(packet)['payload']['state_eef10'],pose10(state['tcp'],state['gripper_fraction'])):
            raise ValueError('E1_INTERVENTION_ACTUAL_OBSERVATION_MISMATCH')
        self.anchor=dict(status='STUDENT_REJECTED_BEFORE_ANY_INTEGRATION_EXPERT_NOT_YET_EXECUTED',tick=self.engine.steps,
            sequence=sequence,student_actual_prefixes=self.student.acknowledged,observation=packet,student_command=pending,
            original_student_S1_rejection=forecast,mode_before='STUDENT',mode_after='EXPERT',switches=1)
        write(self.folder/'INTERVENTION.private.json',self.anchor)
        write(self.folder/'relay/INTERVENTION.json',{k:v for k,v in self.anchor.items() if k not in ['observation','student_command']})
        self.student.pending=None;self.mode='EXPERT';return True
    def guard_prefix(self):
        if self.mode=='STUDENT':self.student.guard_prefix()
    def acknowledge(self,engine,rows):
        if self.mode=='STUDENT':return self.student.acknowledge(engine,rows)
        if engine is not self.engine or len(rows)!=200:raise ValueError('E1_EXACT_EXPERT_PREFIX_ACK')
        p=self.folder/('expert-prefix-'+str(len(self.expert_acks)).zfill(3)+'.npz')
        atomic_npz(p,rows=np.asarray(rows),tick_after=engine.steps,tick_before=engine.steps-200)
        self.expert_acks.append(dict(tick_before=engine.steps-200,tick_after=engine.steps,integrations=200,
            expert_commands=8,physical_receipt_sha256=sha(p)))
        write(self.folder/'EXPERT_PREFIX_ACKS.json',self.expert_acks)
    def finish(self,status,error):
        self.student.finish(status,error)
        write(self.folder/'INTERVENTION_RESULT.json',dict(status=status,error=error,mode=self.mode,
            first_intervention_tick=self.anchor['tick'] if self.anchor else None,
            student_executed_prefixes=self.student.acknowledged,expert_executed_prefixes=len(self.expert_acks),
            expert_executed_commands=sum(a['expert_commands'] for a in self.expert_acks),invalid_expert=self.invalid_expert,
            data_class='STUDENT_VISITED_INTERVENTION' if self.anchor and self.student.acknowledged else
                ('EXPERT_ASSISTED_DATA' if self.anchor else 'NO_INTERVENTION_NO_CORRECTION_DATA'),
            subsequent_expert_states_are_student_states=False,switches=int(self.anchor is not None)))
