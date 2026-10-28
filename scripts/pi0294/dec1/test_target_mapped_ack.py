"""CPU integration evidence only: no model, solver, simulator or native proof.

Use the existing mapper, receiver and native-policy transport fixtures. Their
synthetic native verdict is NOT a qualified target chain or dispatch permit.
"""
from copy import deepcopy
from dataclasses import asdict,replace
import base64
import hashlib
import json
import uuid
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.pi0292.relay_protocol import digest,sealed,unseal
from scripts.pi0294.resident import QueryReservations
from scripts.pi0292.durable import read,write
from scripts.pi0294.dec1 import target_mapped_ack as unit
from . import target_time_transaction as wire
from .authority import R0,PARENT
from .test_mapped_ack import fixture as source_fixture
from .test_resident_mapped_seam import packets
from .test_target_mpc_prediction import case
from .test_target_time_transaction import fixture as time_fixture,events


def target_condition(condition):
    c=deepcopy(condition);c.update(schema=unit.CONDITION_SCHEMA,execution_semantics=unit.SEMANTICS,
        checkpoint_manifest_sha256=PARENT,target_runtime_sha256='1'*64,
        native_safety_chain_sha256='2'*64,task_clock_sha256=unit.FROZEN_CLOCK,native_tick_origin=2500)
    return c


def contract(c,**kw):
    raw=json.dumps(sealed(c),sort_keys=True,allow_nan=False).encode()
    options=dict(expected_sha256=hashlib.sha256(raw).hexdigest(),expected_condition_id=c['condition_id'],
        expected_owner_release=c['owner_release'],expected_plan_sha256=c['scientific_plan_sha256'],
        expected_identity=c['identity'],expected_target_runtime_sha256='1'*64,expected_native_safety_chain_sha256='2'*64)
    options.update(kw)
    return unit.FixedTargetTimeAckContract(raw,**options)


def fixture(mode='I_PROGRESS',sequence=0,initial_ack=None,generation=None):
    if generation is None:
        body,_,_,action,command,_=packets();body['identity']['statistics_sha256']=R0
        a=unseal(action);a['sequence']=sequence;a['tick']=sequence*200;a['payload']['identity']=body['identity']
        raw,*_=time_fixture(mode);a['payload']['trajectory_eef10']=raw.tolist()
        cmd=unseal(command);cmd.update(sequence=sequence,tick=sequence*200,prefix_eef10=raw[:8].tolist(),prefix_sha256=digest(raw[:8].tolist()))
        action=sealed(a);command=sealed(cmd)
    else:
        request,action,command=generation;body=unseal(request)
        assert body['sequence']==sequence and body['identity']['statistics_sha256']==R0
    s,*_=source_fixture(mode=mode,run=body['run_uuid'],episode=body['episode_uuid'],identity=body['identity'])
    c=contract(target_condition(s.condition));expected=c.bind_action(action,command,initial_adapter_ack_id=initial_ack)
    refs=[];prior=initial_ack;start=2500+sequence*200
    for i in range(8):
        # ROS timestamps use the frozen nanosecond conversion. Multiplying a
        # Python float .002 can instead introduce a negative sub-ns age.
        step=start+i*25;v,_,obs,p=case(mode,clock_origin=step/500)
        doc=json.loads(v.wire_json);doc['_pi0294'].update(episode=body['episode_uuid'],sequence=sequence*8+i)
        serialized=json.dumps(doc,separators=(',',':'),allow_nan=False)
        v=replace(v,episode=body['episode_uuid'],sequence=sequence*8+i,wire_json=serialized,
            wire_sha256=hashlib.sha256(serialized.encode()).hexdigest())
        trace=events(v);row=trace[1]
        p['plan_target_trajectories']=dict(time_trajectory=row['times'],state_trajectory=[{'value':x} for x in row['poses']],input_trajectory=[{'value':x} for x in row['inputs']])
        policy=json.dumps(p,separators=(',',':'),allow_nan=False).encode()
        refs.append(dict(reference=i,predecessor_ack_id=prior,prepared=asdict(v),installation_events=trace,
            policy_bytes_base64=base64.b64encode(policy).decode(),policy_sha256=hashlib.sha256(policy).hexdigest(),
            actual_MPC_observation=obs,tick_before=sequence*200+25*i,tick_after=sequence*200+25*(i+1),
            native_step_before=step,native_step_after=step+25,actual_feedback_sha256='3'*64,
            pause_before_sha256='4'*64,pause_after_sha256='5'*64));prior=v.command_id
    mapping=dict(schema=unit.SCHEMA,condition_sha256=c.condition_sha256,condition_id=c.condition['condition_id'],
        interface_mode=mode,identity=body['identity'],executor_id=c.condition['executor_id'],
        response_sha256=action['sha256'],raw_h40_sha256=expected['raw_h40_sha256'],initial_adapter_ack_id=initial_ack,
        target_runtime_sha256='1'*64,native_safety_chain_sha256='2'*64,task_clock_sha256=unit.FROZEN_CLOCK,references=refs,
        native_prefix_receipt=dict(schema='DEC1_FROZEN_TARGET_NATIVE_PREFIX8_VERDICT_V1',decision='PASS',
            target_runtime_sha256='1'*64,native_safety_chain_sha256='2'*64,task_clock_sha256=unit.FROZEN_CLOCK,
            raw_h40_sha256=expected['raw_h40_sha256'],wire_sha256s=[r['prepared']['wire_sha256'] for r in refs],
            native_step_before=start,native_step_after=start+200,evidence_sha256='6'*64,physics_receipt_sha256='7'*64))
    packet=c.make_packet(expected=expected,mapping=mapping,physics_receipt_sha256='7'*64)
    return body,c,expected,action,command,packet


@pytest.mark.parametrize('mode',['I_TIME','I_PROGRESS'])
@pytest.mark.parametrize('sequence',[0,1,79])
def test_existing_receiver_and_original_float32_MPC_bind_eight_target_references(mode,sequence):
    body,c,e,a,cmd,p=fixture(mode,sequence,initial_ack='previous-real-command' if sequence else None);before=deepcopy(p)
    checked=c.validate(p,expected=e)
    assert checked.packet_sha256==p['sha256'] and p==before
    assert checked.final_adapter_ack_id==p['body']['payload']['mapped_execution']['references'][-1]['prepared']['command_id']
    assert e['response_sha256']==a['sha256'] and e['command_sha256']==cmd['sha256']
    assert not isinstance(c,unit.source.FixedMappedAckContract)


@pytest.mark.parametrize('fault',['missing','reordered','repeat','predecessor','raw_tail','wire_metadata','install',
    'policy','independent_observation','tick','warmup_clock','phase','pending_candidate','fullbody_verdict',
    'source_s1','physics_hash','clock_pin','response','mode','model','prefix_wire','bool_tick'])
def test_stale_partial_source_or_resealed_wrong_target_evidence_stops_next_generation(fault):
    _,c,e,_,_,p=fixture();b=unseal(p);m=b['payload']['mapped_execution'];r=m['references'][3]
    if fault=='missing':m['references'].pop()
    elif fault=='reordered':m['references'][2],m['references'][3]=m['references'][3],m['references'][2]
    elif fault=='repeat':m['references'][3]=deepcopy(m['references'][2])
    elif fault=='predecessor':r['predecessor_ack_id']='unrelated'
    elif fault=='raw_tail':r['prepared']['raw_h40_sha256']='0'*64
    elif fault=='wire_metadata':
        d=json.loads(r['prepared']['wire_json']);d['_pi0294']['sequence']=0
        r['prepared']['wire_json']=json.dumps(d)
        r['prepared']['wire_sha256']=hashlib.sha256(r['prepared']['wire_json'].encode()).hexdigest()
    elif fault=='install':r['installation_events'].pop()
    elif fault=='policy':r['policy_sha256']='0'*64
    elif fault=='independent_observation':r['actual_MPC_observation']['time']+=.026
    elif fault=='tick':r['tick_after']+=1
    elif fault=='warmup_clock':r['native_step_before']-=2500
    elif fault=='phase':r['prepared']['phase']=-.1
    elif fault=='pending_candidate':r['prepared'].update(pause_anchor_sha256='8'*64,native_prefix_validation_pending=True,physics_advance_authorized=False)
    elif fault=='fullbody_verdict':del m['native_prefix_receipt']
    elif fault=='source_s1':m['native_prefix_receipt']={'source_S1':'PASS','shadow_integrations':200}
    elif fault=='physics_hash':b['payload']['physics_receipt_sha256']='8'*64
    elif fault=='clock_pin':m['task_clock_sha256']='9'*64
    elif fault=='response':m['response_sha256']='0'*64
    elif fault=='mode':m['interface_mode']='I_TIME'
    elif fault=='model':m['identity']['model_revision']='0'*40
    elif fault=='prefix_wire':m['native_prefix_receipt']['wire_sha256s'][3]='0'*64
    elif fault=='bool_tick':r['native_step_after']=True
    with pytest.raises((ValueError,PermissionError,KeyError)):c.validate(sealed(b),expected=e)


@pytest.mark.parametrize('fault',['A_weights','R1','clock_HEAD','runtime','native_chain','owner'])
def test_cold_condition_cannot_switch_science_runtime_or_owner(fault):
    _,c,*_=fixture();d=c.condition;kw={}
    if fault=='A_weights':d['checkpoint_manifest_sha256']='a'*64
    elif fault=='R1':d['identity']['statistics_sha256']='a'*64
    elif fault=='clock_HEAD':d['task_clock_sha256']='9f4286bdc6ddf5f3931d776e08102a42dc22951bd2a683785bd27df753c1c8f7'
    elif fault=='runtime':d['target_runtime_sha256']='a'*64
    elif fault=='native_chain':d['native_safety_chain_sha256']='a'*64
    elif fault=='owner':kw['expected_owner_release']=None
    with pytest.raises(ValueError):contract(d,**kw)


def test_source_and_target_ack_ABIs_cannot_accept_each_others_receipts():
    _,c,e,action,cmd,p=fixture()
    s,source_e,source_ack,*_=source_fixture(run=c.condition['run_uuid'],episode=c.condition['episode_uuid'],
        identity=c.condition['identity'],action=action,command=cmd)
    with pytest.raises(ValueError):c.validate(source_ack,expected=e)
    with pytest.raises(ValueError):s.validate(p,expected=source_e)


@pytest.mark.parametrize('fault',['none','stripped','replaced_contract','altered_response','unknown_generation'])
def test_actual_resident_roundtrip_retains_target_ack_and_unknown_generation(tmp_path,fault):
    body,c,e,a,cmd,p=fixture();q=QueryReservations(tmp_path,body['run_uuid'],160,mapped_ack_contracts={body['episode_uuid']:c})
    q.begin(body,100.)
    if fault=='unknown_generation':
        q.fail(body,'UNKNOWN_LOST_GENERATION_RETURN')
        with pytest.raises(ValueError):q.begin(body,101.)
        assert q.state['pending']['query_uuid']==body['query_uuid']
        return
    q.complete(body,a['sha256'],prefix_sha256=e['prefix_sha256'],command_sha256=cmd['sha256'],action_packet=a,command_packet=cmd)
    nxt=deepcopy(body);nxt.update(sequence=1,tick=200,query_uuid=str(uuid.uuid4()),previous_ack=p)
    nxt['payload']['snapshot_uuid']=str(uuid.uuid4())
    if fault=='stripped':
        b=unseal(p);del b['payload']['mapped_execution'];del b['payload']['execution_semantics'];nxt['previous_ack']=sealed(b)
    elif fault=='replaced_contract':q.mapped_ack_contracts={}
    elif fault=='altered_response':q.state['queries'][-1]['response_sha256']='0'*64
    if fault=='none':
        q.begin(nxt,102.)
        assert q.state['pending']['initial_adapter_ack_id']==c.validate(p,expected=e).final_adapter_ack_id
        assert nxt['previous_ack']==p
    else:
        with pytest.raises(ValueError):q.begin(nxt,102.)
        assert q.state['pending'] is None


@pytest.mark.parametrize('mode',['I_TIME','I_PROGRESS'])
def test_original_e1_query_loop_accepts_explicit_target_contract_and_draws_once(tmp_path,monkeypatch,mode):
    from scripts.pi0294.e1 import neural
    body,c,e,action,_,_=fixture(mode);root=tmp_path/body['run_uuid'];root.mkdir()
    label='CPU_TARGET_SEAM_ONLY';plan=root/'plans'/label;plan.mkdir(parents=True)
    identity=dict(relay_identity=body['identity'],global_step=2048,checkpoint_manifest_sha256=PARENT)
    write(plan/'REGISTRY.json',dict(run_uuid=root.name,checkpoint_identity=identity,
        cases={body['episode_uuid']:dict(layout_id='same-paired-geometry')}))
    (root/'pbs').mkdir();directory=root/'evidence'/label/'resident';draws=[];charges=[]
    def generate(payload,seed):
        draws.append(seed)
        return dict(physical_H40=np.asarray(unseal(action)['payload']['trajectory_eef10']),
            normalized=np.zeros((1,40,10)),noise={'CPU_fixture':True},inference_precision='CPU_FIXTURE_ONLY')
    def guard():
        path=directory/'inbox'/(body['query_uuid']+'.json')
        if not path.exists():write(path,sealed(body))
        if (directory/'outbox'/path.name).exists():
            write(directory/'STAGE_DONE.json',dict(run_uuid=root.name,job_id='CPU-fixture-job',label=label))
    monkeypatch.setenv('PBS_JOBID','CPU-fixture-job')
    monkeypatch.setattr(neural.time,'time',lambda:100.)
    monkeypatch.setattr(neural.time,'sleep',lambda _:None)
    monkeypatch.setattr(neural.signal,'signal',lambda *a:None)
    monkeypatch.setattr(neural.signal,'alarm',lambda *a:None)
    neural.serve(root,label,SimpleNamespace(identity=identity,generate=generate),dict(allocation_end_epoch=1000.),
        SimpleNamespace(charge=lambda key,n:charges.append((key,n))),guard,max_queries=80,
        mapped_ack_contracts={body['episode_uuid']:c})
    state=read(directory/'QUERY_STATE.json');result=read(directory/'outbox'/(body['query_uuid']+'.json'))
    expected_seed=int.from_bytes(hashlib.sha256(b'PI0294/same-paired-geometry/0').digest()[:8],'big')%(2**63)
    assert draws==[expected_seed] and charges==[('generation_queries',1)]
    assert state['mapped_ack_conditions']=={body['episode_uuid']:c.condition_sha256}
    assert state['queries'][0]['mapped_ack_binding']['raw_h40_sha256']==e['raw_h40_sha256']
    assert state['queries'][0]['response_sha256']==result['sha256']


def test_target_contract_cannot_enter_source_physics_receipt_writer():
    from scripts.pi0294.dec1.mapped_relay_ack import make_mapped_relay_acknowledger
    _,c,*_=fixture()
    with pytest.raises(TypeError,match='FIXED_OWNER_CONTRACT_REQUIRED'):
        make_mapped_relay_acknowledger(None,c)
