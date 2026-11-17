"""Original binary/pause ABI with a nonintegrating fake adapter, no native run."""
import copy
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from go2arx5_vla.world_model.close_reference_input_v1 import (
    SHIFT7_METHOD, SHIFT7_NATIVE_VERSION, remember, prediction, invalidate,
)
from go2arx5_vla.world_model.shift7_native_input_v5 import capture, verify_query_return
from go2arx5_vla.world_model.sim_paused_contract_v5 import SELECTED_SHIFT7_MANIFEST_SHA256
from scripts.pi0294.task_state import read_state


@pytest.fixture
def setup(tmp_path):
    root = tmp_path/'scope'; root.mkdir()
    output = root/'native-01'; output.mkdir()
    scene = tmp_path/'scene.mjb'
    spec = dict(registered_root=str(root), model_path=str(scene), frozen_files={str(scene):'c'*64},
        mobile_condition=dict(method=SHIFT7_METHOD, mode='M_FREE', family='F1'), max_ticks=30000,
        mobile_experiment_version=SHIFT7_NATIVE_VERSION,
        execution_clock_profile='SIM_PAUSED_NATIVE',
        SHIFT7_package=dict(sha256=SELECTED_SHIFT7_MANIFEST_SHA256),
        policy_from_world=np.eye(4).tolist())
    (root/'EXECUTION_SPEC.json').write_text(json.dumps(spec))
    step = 5753
    raw = (struct.pack('<8sQdIII', b'GRTSTATE', step, step*.002, 34,32,20)
           + np.zeros(86,dtype='<f8').tobytes() + struct.pack('<I',0))
    (output/'state.bin').write_bytes(raw)
    (output/'step_quota').write_text(str(step))
    ack = dict(step=step,time_s=step*.002,motiontime=5000,leg_history_steps=5000,
               gait_indices=[0,1,2,3],clock_paused=True,running=False,ack_sequence=1)
    (output/'pause_ack.json').write_text(json.dumps(ack))
    class Adapter:
        def __init__(self):
            self.spec=spec; self.output=output; self.shift7_fixed_task_frame=np.eye(4)
            self.on_spin=None; self.integrations=0
        def guard(self):
            if (self.output/'STOP.json').exists():raise RuntimeError('OPERATOR_STOP')
        def observe(self):return read_state(self.output/'state.bin')
        def spin(self,seconds,predicate):
            self.guard()
            value=json.loads((self.output/'pause_ack.json').read_text())
            value['ack_sequence']+=1
            (self.output/'pause_ack.json').write_text(json.dumps(value))
            if self.on_spin:self.on_spin()
            return predicate()
    adapter=Adapter()
    row=np.zeros(10);row[9]=.012/.044
    remember(adapter,row,dict(source_native_s=step*.002-.05),
             dict(step=step,time_s=step*.002))
    tcp=np.eye(4);tcp[:3,3]=[.4,0,.3]
    sample=dict(native_step=step,time_s=step*.002,tcp=tcp,gripper_fraction=.9,
                object_pose=np.full((4,4),999.),expert_stage='LIFT',teacher_future_H40='FORBIDDEN')
    sample.update({k:v.copy() for k,v in adapter.observe().items() if k in ('qpos','qvel','ctrl')})
    class Camera:
        model_path=scene;model_sha='c'*64
        def render_memory(self,packet):
            return (dict(wrist_rgb=np.ones((256,256,3),np.uint8)),
                    np.zeros((256,256,3),np.uint8),{},dict(body='CPU_TEST_FIXTURE'))
    return adapter,sample,Camera(),tcp


def prepare(f, index=0):
    adapter,sample,camera,reference=f
    return capture(adapter,sample,camera,np.eye(4),reference,index)


def test_SHIFT7_transports_actual_command_and_whitelisted_RGB_without_truth(setup):
    adapter,sample,_,_=setup
    result=prepare(setup)
    record=result['metadata'];pin=record['input'];context=result['previous_reference']
    assert context['width_m']==pytest.approx(.012)
    assert context['width_m'] != sample['gripper_fraction']*.044
    assert record['pause_after']['ack_sequence']>record['pause_before']['ack_sequence']
    assert record['native_execution_admitted'] is False
    assert record['physics_integrations_requested']==0 and adapter.integrations==0
    with np.load(pin['path'])as z:
        assert set(z.files)=={'state_eef10','body_rgb','wrist_rgb','T_policy_from_mount',
            'policy_from_world','accepted_reference_TCP','previous_width_m',
            'previous_source_native_s','query_native_s'}
        data={k:z[k]for k in z.files}
    class Provider:
        def predict(self,*args,**kw):return kw
    assert prediction(Provider(),SHIFT7_METHOD,data,'pick',7)==dict(noise_id=7,previous_reference=context)
    checked=verify_query_return(adapter,result)
    assert checked['unchanged_native_step']==sample['native_step']
    assert checked['native_execution_admitted'] is False


@pytest.mark.parametrize('fault',['method','version','mode','profile','package','frame','scene','sample_clock','STOP'])
def test_capture_rejects_wrong_identity_or_observation(setup,fault):
    adapter,sample,camera,_=setup
    if fault=='method':adapter.spec['mobile_condition']['method']='CLOSE_TIME500'
    elif fault=='version':adapter.spec['mobile_experiment_version']='MFREE_STUDENT_60S_V1'
    elif fault=='mode':adapter.spec['mobile_condition']['mode']='M_ASSIST'
    elif fault=='profile':adapter.spec['execution_clock_profile']='WALL_CLOCK_NATIVE'
    elif fault=='package':adapter.spec['SHIFT7_package']['sha256']='a'*64
    elif fault=='frame':adapter.shift7_fixed_task_frame[0,3]=.1
    elif fault=='scene':camera.model_sha='b'*64
    elif fault=='sample_clock':sample['native_step']+=25
    else:(adapter.output/'STOP.json').write_text('STOP')
    with pytest.raises((ValueError,RuntimeError)):prepare(setup)
    assert adapter.integrations==0


@pytest.mark.parametrize('fault',['packet','quota','leg_history','ACK','STOP','command','input','episode'])
def test_query_return_rejects_changed_control_context(setup,fault):
    adapter,_,_,_=setup
    result=prepare(setup)
    if fault=='packet':
        p=adapter.output/'state.bin';raw=bytearray(p.read_bytes());raw[36:44]=struct.pack('<d',.001);p.write_bytes(raw)
    elif fault=='quota':(adapter.output/'step_quota').write_text('5778')
    elif fault=='leg_history':
        p=adapter.output/'pause_ack.json';v=json.loads(p.read_text());v['leg_history_steps']+=1;p.write_text(json.dumps(v))
    elif fault=='ACK':
        p=adapter.output/'pause_ack.json';v=json.loads(p.read_text());v['ack_sequence']=0;p.write_text(json.dumps(v))
    elif fault=='STOP':(adapter.output/'STOP.json').write_text('STOP')
    elif fault=='command':adapter.close_previous_reference['width_m']=.013
    elif fault=='input':
        with Path(result['metadata']['input']['path']).open('ab')as f:f.write(b'changed')
    else:
        adapter.output=adapter.output.parent/'native-02';adapter.output.mkdir()
    with pytest.raises((ValueError,RuntimeError)):verify_query_return(adapter,result)
    assert adapter.integrations==0


def test_partial_or_wrong_tick_cannot_become_SHIFT7_history(setup):
    adapter,_,_,_=setup
    before=copy.deepcopy(adapter.close_previous_reference)
    row=np.zeros(10);row[9]=.2
    with pytest.raises(ValueError,match='25_TICK'):
        remember(adapter,row,dict(source_native_s=11.506),dict(step=5763,time_s=11.526))
    assert adapter.close_previous_reference==before


def test_serialization_then_native_advance_cannot_publish_ready_input(setup):
    adapter,_,_,_=setup
    def move():
        p=adapter.output/'state.bin';raw=bytearray(p.read_bytes());raw[8:16]=struct.pack('<Q',5754);p.write_bytes(raw)
    adapter.on_spin=move
    with pytest.raises(ValueError):prepare(setup)
    assert not list(adapter.output.glob('shift7-queries/*/INPUT_META.private.json'))


def test_existing_capture_directory_cannot_be_reused(setup):
    prepare(setup)
    with pytest.raises(FileExistsError):prepare(setup)


def test_cancel_invalidates_old_SHIFT7_command_without_changing_TIME_history(setup):
    adapter,sample,_,_=setup
    invalidate(adapter)
    assert adapter.close_previous_reference is None
    with pytest.raises(ValueError,match='HISTORY'):prepare(setup)
    row=np.zeros(10);row[9]=.2
    with pytest.raises(ValueError):
        remember(adapter,row,dict(source_native_s=11.506),dict(step=5778,time_s=11.556))
    old=SimpleNamespace(spec=dict(mobile_condition=dict(method='CLOSE_TIME500')),
                        close_previous_reference={'unchanged':True})
    invalidate(old)
    assert old.close_previous_reference=={'unchanged':True}


def test_actual_adapter_cancel_invalidates_even_when_publication_fails(setup):
    from go2arx5_vla.world_model.target_sequential import TargetSequentialAdapter
    adapter=setup[0]
    # Stop before any ROS write; the real cancel entry must invalidate first.
    with pytest.raises(AttributeError):TargetSequentialAdapter.cancel_reference(adapter)
    assert adapter.shift7_history_invalidated
    assert adapter.close_previous_reference is None


def test_out_of_order_full_atom_cannot_replace_SHIFT7_history(setup):
    adapter=setup[0];before=copy.deepcopy(adapter.close_previous_reference)
    row=np.zeros(10);row[9]=.2
    with pytest.raises(ValueError,match='ONE_ATOM'):
        remember(adapter,row,dict(source_native_s=11.556),dict(step=5803,time_s=11.606))
    assert adapter.close_previous_reference==before


def test_total_native_cap_includes_startup_and_partial_P10_is_explicit(setup):
    adapter,sample,_,_=setup
    adapter.spec['max_ticks']=sample['native_step']+75
    p=Path(adapter.spec['registered_root'])/'EXECUTION_SPEC.json'
    p.write_text(json.dumps(adapter.spec))
    result=prepare(setup)
    assert result['metadata']['available_prefix_atoms']==3
    assert result['metadata']['max_total_episode_ticks']==5828
    # The entire input/physical H40 convention is retained; this is only the
    # remaining permitted consumption, never a shortened prediction horizon.
    adapter.spec['max_ticks']=sample['native_step']+24
    p.write_text(json.dumps(adapter.spec))
    with pytest.raises(ValueError,match='NO_COMPLETE_NATIVE_ATOM'):prepare(setup,1)


def test_stale_sample_state_cannot_be_retagged_with_current_time(setup):
    setup[1]['qpos'][2]=.1
    with pytest.raises(ValueError,match='CURRENT_SAMPLE'):prepare(setup)


def test_V51_keeps_full60s_task_budget_beside_initialization(setup):
    adapter=setup[0]
    adapter.spec['mobile_experiment_version']='MFREE_SHIFT7_SIM_ONLY_V51'
    adapter.spec['max_ticks']=35703
    (Path(adapter.spec['registered_root'])/'EXECUTION_SPEC.json').write_text(json.dumps(adapter.spec))
    result=prepare(setup)
    assert result['metadata']['max_total_episode_ticks']==35703
    assert result['metadata']['available_prefix_atoms']==10


def test_V52_successor_has_distinct_pinned_arm_and_unchanged_full_horizon(setup):
    from go2arx5_vla.sla_adapter.mobile_successor_v52 import PLAN,PARENT
    adapter=setup[0];package=dict(path='immutable-successor-package',sha256='1'*64)
    adapter.spec.update(mobile_experiment_version='MFREE_SHIFT7_SIM_ONLY_V52',
        scope='SLA_MOBILE_F2_V52',plan_commit=PLAN,arm='B1',max_ticks=35703,
        physics_dt_s=.002,SHIFT7_package=package,packages=dict(B0=dict(sha256=PARENT),B1=package,B2=dict(sha256='2'*64)))
    path=Path(adapter.spec['registered_root'])/'EXECUTION_SPEC.json';path.write_text(json.dumps(adapter.spec))
    result=prepare(setup)
    assert result['metadata']['available_prefix_atoms']==10
    assert result['metadata']['max_total_episode_ticks']==35703
    assert result['previous_reference']['width_m']==.012
    assert result['previous_reference']['query_native_s']==setup[1]['time_s']
    adapter.spec['arm']='B2';path.write_text(json.dumps(adapter.spec))
    with pytest.raises(ValueError,match='V52_SINGLE_MODEL_IDENTITY'):prepare(setup,1)


@pytest.mark.parametrize('wrong_identity',[False,True])
def test_real_protocol_handoff_does_not_install_or_mock_consume_the_model_reply(setup,monkeypatch,wrong_identity):
    from go2arx5_vla.world_model import sim_only_mobile_v51 as runtime
    from go2arx5_vla.world_model.storage import digest
    adapter=setup[0];root=Path(adapter.spec['registered_root']);server=root/'server';server.mkdir()
    (server/'control').mkdir();(server/'bus').mkdir()
    (server/'control/LEASE.json').write_text(json.dumps(dict(status='ACTIVE')))
    spec=dict(registered_root=str(server),ipc_root=str(server/'bus'),
        SHIFT7_package=dict(sha256=SELECTED_SHIFT7_MANIFEST_SHA256),
        request_bindings=dict(policy_package_sha256=SELECTED_SHIFT7_MANIFEST_SHA256,selected_combo='SHIFT7_SINGLE'))
    sp=server/'EXECUTION_SPEC.json';sp.write_text(json.dumps(spec))
    adapter.spec.update(resident_server_spec=dict(path=str(sp),sha256=digest(sp)),language='pick',
        model_service_timeout_s=30)
    adapter.spec['mobile_condition']['seed']=7
    (root/'EXECUTION_SPEC.json').write_text(json.dumps(adapter.spec))
    prepared=prepare(setup);previous=copy.deepcopy(adapter.close_previous_reference)
    original=runtime.write;seen=[]
    def deliver(path,value):
        original(path,value)
        if path==server/'bus/REQUEST.json':
            seen.append(value);q=value['request'];out=Path(value['response_root']);action=out/'ACTION.npz'
            raw=np.repeat(prepared['observation'].values_for(prepared['metadata']['input'])['state_eef10'][None],40,axis=0)
            np.savez(action,physical_H40=raw,normalized=np.zeros((1,40,32)))
            original(out/'RESPONSE.json',dict(status='GENERATED_NOT_INSTALLED',
                request_id='wrong'if wrong_identity else q['request_id'],bindings=q['bindings'],
                action=dict(path=str(action),sha256=digest(action)),new_generations_this_request=1))
    monkeypatch.setattr(runtime,'write',deliver)
    if wrong_identity:
        with pytest.raises(ValueError,match='REAL_SHIFT7_RESPONSE'):runtime.request_policy(adapter,prepared,0)
    else:
        assert runtime.request_policy(adapter,prepared,0)['status']=='GENERATED_NOT_INSTALLED'
    assert seen[0]['request']['previous_reference']['width_m']==pytest.approx(.012)
    assert adapter.close_previous_reference==previous and adapter.integrations==0
    folder=Path(prepared['metadata']['input']['path']).parent/'inference-control'
    assert not list(folder.glob('install-*'))and not list(folder.glob('emitted-*'))
    assert json.loads((folder/'live.json').read_text())['cancelled']
