"""Pause at fresh observations and install each hash-bound prefix exactly once."""
import base64
from pathlib import Path
import time
import uuid

import numpy as np

from scripts.pi0292.durable import read,write
from scripts.pi0292.eval_contract import FRAME,TCP,matrix10
from scripts.pi0292.physics_relay import accept_prefix
from scripts.pi0292.relay_protocol import PROTOCOL,sealed,unseal,validate_identity
from scripts.pi0294.run_admission import sha,validate
from go2arx5_vla.task_data.codec import pose10


def admit_request(root,folder,card):
    root=Path(root);manifest=validate(root,profile='4080-simulation')
    path=root/card['relay_request_relative']
    if not path.resolve().is_relative_to(root.resolve()) or sha(path)!=card['relay_request_sha256']:
        raise PermissionError('EXACT_REGISTERED_RELAY_REQUEST_REQUIRED')
    request=read(path)
    if (request['run_uuid']!=root.name or request['attempt_id']!=card['id']
        or request['source_sha']!=manifest['code']['vla_commit']
        or request['run_manifest_sha256']!=sha(root/'RUN_MANIFEST.json')
        or request['initial_state_sha256']!=card['initial_state_sha256']
        or request['profile'] not in ('TEACHER_REFERENCE','B1')
        or request['category'] not in ('RELAY_QUALIFICATION','TRAIN_DEVELOPMENT','DEV','CONFIRM_CORE','CONFIRM_CHALLENGE','REGRESSION')):
        raise PermissionError('PI0294_RELAY_SOURCE_CASE_AND_INITIAL_STATE_REQUIRED')
    if request['profile']=='B1':
        proof=root/'evidence/LIVE_READINESS.json'
        if sha(proof)!=request['live_readiness_sha256'] or read(proof)['status']!='PASS_ACTUAL_TEACHER_AND_DUAL_SSH_ROUTE':
            raise PermissionError('ACTUAL_TEACHER_AND_DOUBLE_SSH_PROOF_REQUIRED')
    validate_identity(request['identity'])
    if request['identity']['profile']!=request['profile']:
        raise PermissionError('ONE_REGISTERED_PROFILE_REQUIRED')
    uuid.UUID(request['episode_uuid'])
    return request


class RelayTeacher:
    """Teacher state remains audit-only; B1 actions come solely from the wire."""
    def __init__(self,teacher,root,folder,card):
        self.teacher,self.root,self.folder,self.card=teacher,Path(root),Path(folder),card
        self.request=admit_request(root,folder,card)
        self.directory=self.folder/'relay';self.directory.mkdir(exist_ok=False)
        for name in ('observations','commands','intents','acks'):(self.directory/name).mkdir()
        self.pending=None;self.acknowledged=0
        self.state=dict(status='INITIAL_REAL_OBSERVATION_PENDING',run_uuid=self.root.name,
            episode_uuid=self.request['episode_uuid'],profile=self.request['profile'],sequence=0,tick=0,
            acknowledged_commands=0,teacher_controls_model=False)
        write(self.directory/'STATE.json',self.state)

    def __getattr__(self,name):return getattr(self.teacher,name)

    def observe(self,state):self.teacher.observe(state)

    def plan(self,state):
        tick=int(round(state['time_s']/.002));sequence=tick//200
        if tick!=self.acknowledged*200 or sequence>=80 or self.pending is not None:
            raise ValueError('FRESH_OBSERVATION_ONLY_AFTER_PREVIOUS_ACTUAL_ACK')
        images=[]
        for wire,camera in (('external_rgb','body_rgb'),('wrist_rgb','wrist_rgb')):
            path=self.folder/camera/f'{tick//25:04d}.png'
            raw=path.read_bytes()
            images.append(dict(camera=wire,png_base64=base64.b64encode(raw).decode(),sha256=sha(path)))
        payload=dict(snapshot_uuid=str(uuid.uuid4()),language=self.card['instruction'],
            state_eef10=pose10(state['tcp'],state['gripper_fraction']).tolist(),images=images,frame=FRAME,tcp=TCP)
        observation=sealed(dict(protocol=PROTOCOL,kind='observation',run_uuid=self.root.name,
            episode_uuid=self.request['episode_uuid'],sequence=sequence,tick=tick,payload=payload))
        write(self.directory/'observations'/f'{sequence:03d}.json',observation)
        if self.request['profile']=='TEACHER_REFERENCE':
            actual=self.teacher.plan(state)
            write(self.directory/f'teacher-{sequence:03d}.json',dict(observation_sha256=observation['sha256'],
                trajectory_eef10=np.stack([pose10(p,g) for p,g in zip(actual['poses'],actual['opening'])]).tolist(),
                source='ACTUAL_CURRENT_FEEDBACK_TEACHER_NOT_MODEL'))
        self.state.update(status='WAIT_FRESH_ACTION_PHYSICS_PAUSED',sequence=sequence,tick=tick,
            observation_sha256=observation['sha256'],published_epoch=time.time())
        write(self.directory/'STATE.json',self.state)
        path=self.directory/'commands'/f'{sequence:03d}.json';deadline=time.time()+60
        while not path.exists():
            if (self.directory/'STOP.json').exists():raise RuntimeError('OWNER_STOP_NO_PHYSICS_REPLAY')
            if time.time()>=min(deadline,self.request['deadline_epoch']):
                raise TimeoutError('FRESH_RELAY_ACTION_WAIT_EXPIRED')
            time.sleep(.02)
        bundle=read(path)
        if set(bundle)!= {'command','action','request'}:raise ValueError('EXACT_FRESH_ACTION_BUNDLE_REQUIRED')
        command,gate=accept_prefix(bundle['command'],bundle['action'],observation,
            request=bundle['request'],identity=self.request['identity'],now=time.time())
        query=unseal(bundle['request'])
        previous=read(self.directory/'acks'/f'{sequence-1:03d}.json') if sequence else None
        if query['previous_ack']!=previous:raise ValueError('PREVIOUS_ACK_MUST_BIND_FRESH_QUERY')
        intent=self.directory/'intents'/f'{sequence:03d}.json'
        if intent.exists():raise FileExistsError('PHYSICAL_PREFIX_ALREADY_RESERVED_NO_REPLAY')
        write(intent,dict(status='RESERVED_BEFORE_FIRST_INTEGRATION',command_sha256=bundle['command']['sha256'],
            requested_integrations=200,tick_before=tick,query_uuid=command['query_uuid'],gate=gate))
        self.pending=dict(command=command,command_sha256=bundle['command']['sha256'],
            command_packet=bundle['command'],action_packet=bundle['action'])
        trajectory=np.asarray(unseal(bundle['action'])['payload']['trajectory_eef10'],dtype=float)
        return dict(poses=matrix10(trajectory),opening=trajectory[:,9],offset_s=np.arange(40)/20,
            stage='AUDIT_ONLY_'+self.teacher.stage,
            source='ACTUAL_FRESH_MODEL_OBSERVATION' if self.request['profile']=='B1' else 'ACTUAL_FEEDBACK_TEACHER_VIA_DOUBLE_SSH',
            reference_uuid=command['reference_uuid'])

    def guard_prefix(self):
        if self.pending is not None and time.time()>=self.pending['command']['expires']:
            raise TimeoutError('PREFIX_EXPIRED_NO_ADDITIONAL_INTEGRATION')

    def acknowledge(self,engine,rows):
        if self.pending is None:raise ValueError('NO_PENDING_PHYSICAL_PREFIX')
        command=self.pending['command'];sequence=command['sequence']
        if engine.steps!=command['tick']+200 or len(rows)!=200:
            raise ValueError('EXACT_ACTUAL_PREFIX_PROGRESS_REQUIRED')
        self.guard_prefix()
        path=self.directory/f'physics-{sequence:03d}.npz'
        np.savez_compressed(path,rows=np.asarray(rows),tick_before=command['tick'],tick_after=engine.steps)
        with path.open('rb') as stream:__import__('os').fsync(stream.fileno())
        ack=sealed(dict(protocol=PROTOCOL,kind='ack',run_uuid=self.root.name,episode_uuid=self.request['episode_uuid'],
            sequence=sequence,tick=command['tick'],payload={k:command[k] for k in ('query_uuid','reference_uuid','prefix_sha256')}|
                dict(command_sha256=self.pending['command_sha256'],tick_after=engine.steps,integrations=200,physics_receipt_sha256=sha(path))))
        write(self.directory/'acks'/f'{sequence:03d}.json',ack)
        self.acknowledged+=1;self.pending=None
        self.state.update(status='PREFIX_ACKNOWLEDGED_NEXT_OBSERVATION_ALLOWED',tick=engine.steps,
            acknowledged_commands=self.acknowledged)
        write(self.directory/'STATE.json',self.state)

    def finish(self,status,error):
        self.state.update(status=status,error=error,unknown_prefix_progress=self.pending is not None,
                          acknowledged_commands=self.acknowledged)
        write(self.directory/'STATE.json',self.state)
