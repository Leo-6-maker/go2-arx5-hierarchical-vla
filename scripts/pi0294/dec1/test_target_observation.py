"""Synthetic measured-state fixtures; no MuJoCo/model/ROS or native capture.

The real binary parser and PNG/relay validators run. Only frozen ActualState's
read-only kinematics update is replaced with a deterministic CPU test double.
This does not claim actual native state, camera, prefix or task qualification.
"""
from copy import deepcopy
import hashlib
import io
import json
import struct
import uuid
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from scripts.pi0292.relay_protocol import unseal,validate_observation
from scripts.pi0294.task_state import ActualState
from scripts.pi0292.durable import read,write
from scripts.pi0292.relay_protocol import sealed
from scripts.pi0294.dec1.target_relay_exchange import TargetRelayExchange
from scripts.pi0294.dec1 import target_observation as unit
from scripts.pi0294.dec1.test_target_mapped_ack import fixture as ack_fixture


def png(value):
    out=io.BytesIO();Image.fromarray(np.full((256,256,3),value,dtype=np.uint8)).save(out,format='PNG');return out.getvalue()


def fixture(monkeypatch,*,step=2500):
    rx=np.array([[1,0,0],[0,np.cos(.3),-np.sin(.3)],[0,np.sin(.3),np.cos(.3)]])
    rz=np.array([[np.cos(.2),-np.sin(.2),0],[np.sin(.2),np.cos(.2),0],[0,0,1]])
    transform=np.eye(4);transform[:3,:3]=rz;transform[:3,3]=[.02,.01,.03]
    policy=np.eye(4);policy[:3,:3]=rx;policy[:3,3]=[.4,-.02,.4]
    tcp=transform@policy;seen=[]
    def update(self,packet):
        seen.append(packet)
        return dict(native_step=packet['step'],time_s=packet['time_s'],tcp=tcp.copy(),gripper_fraction=.5,
            safety_valid=True,gripper_state_valid=True,native_abort=False,
            object_pose='PRIVILEGED_TASK_TRUTH_MUST_NOT_BE_READ',teacher_stage='MUST_NOT_BE_READ')
    monkeypatch.setattr(ActualState,'update',update)
    values=np.zeros(86,dtype='<f8');values[3]=1.
    raw=struct.pack('<8sQdIII',b'GRTSTATE',step,step/500,34,32,20)+values.tobytes()+struct.pack('<I',0)
    c=unit.FixedTargetObservationBinding(run_uuid='CPU-existing-target',episode_uuid=str(uuid.uuid4()),
        target_runtime_sha256='1'*64,target_world_frame='CPU_EXPLICIT_FROZEN_TARGET_WORLD',
        T_target_world_from_policy_world=transform,expected_transform_sha256=unit.transform_hash(transform))
    cal={name:dict(timestamp_s=step/500,T_world_optical=np.eye(4).tolist(),K=[[128.,0.,128.],[0.,128.,128.],[0.,0.,1.]]) for name in ('body_rgb','wrist_rgb')}
    args=dict(actual_state=object.__new__(ActualState),native_state_bytes=raw,expected_state_sha256=hashlib.sha256(raw).hexdigest(),
        native_step=step,MPC_source_time_s=step/500,predecessor_ack_id=None if step==2500 else 'previous-actual-ACK',
        language='Original task instruction',snapshot_uuid=str(uuid.uuid4()),
        capture=dict(index=(step-2500)//25,native_step=step,time_s=step/500),calibration_bytes=json.dumps(cal).encode(),
        body_png_bytes=png(10),wrist_png_bytes=png(20))
    return c,args,seen,policy,tcp,transform


@pytest.mark.parametrize('step',[2500,2700,18300])
def test_actual_binary_clock_and_explicit_inverse_frame_preserve_model_input_and_target_feedback(monkeypatch,step):
    c,kw,seen,policy,tcp,transform=fixture(monkeypatch,step=step);raw=kw['native_state_bytes'];r=c.bind(**kw)
    p=unseal(r.packet);validate_observation(p['payload'])
    assert p['tick']==step-2500 and p['sequence']==(step-2500)//200
    np.testing.assert_allclose(p['payload']['state_eef10'][:3],policy[:3,3],rtol=0,atol=1e-14)
    np.testing.assert_allclose(p['payload']['state_eef10'][3:9],unit.rotation_matrix_to_rot6d_rows(policy[:3,:3]),rtol=0,atol=1e-14)
    assert p['payload']['state_eef10'][9]==.5 and len(p['payload']['state_eef10'])==10
    np.testing.assert_allclose(r.actual_feedback.eef_position,tcp[:3,3],rtol=0,atol=0)
    assert r.actual_feedback.timestamp==step/500 and r.actual_feedback.source_frame==c.target_frame
    assert r.actual_feedback.predecessor_ack_id==kw['predecessor_ack_id']
    assert r.raw_native_time_s==step/500 and r.raw_MPC_source_time_s==step/500
    assert r.native_state_sha256==hashlib.sha256(raw).hexdigest() and len(seen)==1
    assert seen[0]['step']==step and kw['native_state_bytes']==raw
    assert not r.physics_advance_authorized and not r.target_task_native_qualified
    assert set(p['payload'])=={'snapshot_uuid','language','state_eef10','images','frame','tcp'}
    assert 'PRIVILEGED' not in json.dumps(p)


@pytest.mark.parametrize('fault',['state_hash','native_step','relative_time','MPC_stale','MPC_future','missing_previous_ACK',
    'unexpected_initial_ACK','capture_step','capture_index','capture_time','calibration_time','calibration_nan',
    'calibration_rigid','intrinsics','missing_camera','PNG_size','PNG_order','fake_reader','measured_time','unsafe','gripper','native_abort','desired_pose_invalid'])
def test_bad_or_retimed_state_camera_and_non_native_feedback_cannot_be_a_model_observation(monkeypatch,fault):
    c,kw,seen,*_=fixture(monkeypatch)
    if fault=='state_hash':kw['expected_state_sha256']='0'*64
    elif fault=='native_step':kw['native_step']=2525
    elif fault=='relative_time':
        raw=bytearray(kw['native_state_bytes']);struct.pack_into('<d',raw,16,0.);kw['native_state_bytes']=bytes(raw);kw['expected_state_sha256']=hashlib.sha256(raw).hexdigest()
    elif fault=='MPC_stale':kw['MPC_source_time_s']-=.025001
    elif fault=='MPC_future':kw['MPC_source_time_s']+=.001
    elif fault=='missing_previous_ACK':kw['native_step']=2700;kw['predecessor_ack_id']=None
    elif fault=='unexpected_initial_ACK':kw['predecessor_ack_id']='unrelated'
    elif fault=='capture_step':kw['capture']['native_step']+=1
    elif fault=='capture_index':kw['capture']['index']=True
    elif fault=='capture_time':kw['capture']['time_s']=0.
    elif fault in ('calibration_time','calibration_nan','calibration_rigid','intrinsics','missing_camera'):
        d=json.loads(kw['calibration_bytes'])
        if fault=='calibration_time':d['body_rgb']['timestamp_s']=0.
        elif fault=='calibration_nan':d['body_rgb']['K'][0][0]=float('nan')
        elif fault=='calibration_rigid':d['body_rgb']['T_world_optical'][0][0]=2.
        elif fault=='intrinsics':d['body_rgb']['K'][0][0]=-1.
        else:del d['wrist_rgb']
        kw['calibration_bytes']=json.dumps(d).encode()
    elif fault=='PNG_size':kw['body_png_bytes']=b'not original PNG'
    elif fault=='PNG_order':kw['wrist_png_bytes']=b''
    elif fault=='fake_reader':kw['actual_state']=object()
    else:
        original=ActualState.update
        def wrong(self,packet):
            d=original(self,packet)
            if fault=='measured_time':d['time_s']=0.
            elif fault=='unsafe':d['safety_valid']=False
            elif fault=='gripper':d['gripper_fraction']=1.001
            elif fault=='native_abort':d['native_abort']=True
            elif fault=='desired_pose_invalid':d['tcp'][0,0]=2.
            return d
        monkeypatch.setattr(ActualState,'update',wrong)
    with pytest.raises((ValueError,PermissionError,OSError)):c.bind(**kw)


@pytest.mark.parametrize('fault',['hash','reflection','scale','translation_nan'])
def test_explicit_rigid_transform_cannot_be_fitted_or_silently_changed(monkeypatch,fault):
    c,kw,seen,policy,tcp,tf=fixture(monkeypatch);sha=unit.transform_hash(tf)
    if fault=='hash':tf[0,3]+=.001
    elif fault=='reflection':tf[:3,0]*=-1
    elif fault=='scale':tf[0,0]=2.
    else:tf[0,3]=float('nan')
    with pytest.raises(ValueError):unit.FixedTargetObservationBinding(run_uuid=c.run,episode_uuid=c.episode,
        target_runtime_sha256=c.runtime_sha256,target_world_frame=c.target_frame,
        T_target_world_from_policy_world=tf,expected_transform_sha256=sha)
    assert seen==[]


def test_measured_target_observation_enters_existing_e1_model_loop_and_original_raw_action_gate(tmp_path,monkeypatch):
    from scripts.pi0294.e1 import neural
    from scripts.pi0294.dec1.authority import PARENT
    _,contract,*_=ack_fixture();c,kw,seen,policy,tcp,tf=fixture(monkeypatch)
    c=unit.FixedTargetObservationBinding(run_uuid=contract.condition['run_uuid'],episode_uuid=contract.condition['episode_uuid'],
        target_runtime_sha256=contract.condition['target_runtime_sha256'],target_world_frame=c.target_frame,
        T_target_world_from_policy_world=tf,expected_transform_sha256=unit.transform_hash(tf))
    bound=c.bind(**kw);root=tmp_path/c.run;root.mkdir();label='CPU_MEASURED_TARGET_INPUT'
    exchange=TargetRelayExchange(root/'route',run_uuid=c.run,episode_uuid=c.episode,identity=contract.condition['identity'],now=100.,mapped_ack_contract=contract)
    try:
        request=exchange.observe(bound.packet,100.);body=unseal(request)
        plan=root/'plans'/label;plan.mkdir(parents=True);(root/'pbs').mkdir()
        identity=dict(relay_identity=contract.condition['identity'],global_step=2048,checkpoint_manifest_sha256=PARENT)
        write(plan/'REGISTRY.json',dict(run_uuid=root.name,checkpoint_identity=identity,cases={c.episode:dict(layout_id='paired-geometry')}))
        directory=root/'evidence'/label/'resident';calls=[];charges=[]
        def generate(payload,seed):
            assert payload==unseal(bound.packet)['payload'] and 'object_pose' not in payload
            calls.append(dict(seed=seed,state=payload['state_eef10']))
            return dict(physical_H40=np.tile(np.asarray(bound.raw_actual_state10),(40,1)),normalized=np.zeros((1,40,10)),
                noise={'CPU_fixture_only':True},inference_precision='CPU_FIXTURE_ONLY')
        def guard():
            path=directory/'inbox'/f"{body['query_uuid']}.json"
            if not path.exists():write(path,request)
            if (directory/'outbox'/path.name).exists():
                write(directory/'STAGE_DONE.json',dict(run_uuid=root.name,job_id='CPU-fixture-job',label=label))
        monkeypatch.setenv('PBS_JOBID','CPU-fixture-job')
        monkeypatch.setattr(neural.time,'time',lambda:100.)
        monkeypatch.setattr(neural.time,'sleep',lambda _:None)
        monkeypatch.setattr(neural.signal,'signal',lambda *a:None)
        monkeypatch.setattr(neural.signal,'alarm',lambda *a:None)
        neural.serve(root,label,SimpleNamespace(identity=identity,generate=generate),dict(allocation_end_epoch=1000.),
            SimpleNamespace(charge=lambda k,n:charges.append((k,n))),guard,max_queries=80,mapped_ack_contracts={c.episode:contract})
        result=read(directory/'outbox'/f"{body['query_uuid']}.json");command=exchange.action(result,100.)
        state=read(directory/'QUERY_STATE.json');binding=state['queries'][0]['mapped_ack_binding']
        assert binding['response_sha256']==result['sha256'] and binding['command_sha256']==command['sha256']
        assert len(calls)==1 and charges==[('generation_queries',1)]
        assert exchange.state['phase']=='WAIT_ACK' and exchange.state['tick']==0
        assert bound.raw_native_time_s==5. and unseal(request)['tick']==0
        # No ACK, native safety receipt or physical progress is synthesized.
        assert not bound.physics_advance_authorized and not bound.target_task_native_qualified
    finally:exchange.close()
