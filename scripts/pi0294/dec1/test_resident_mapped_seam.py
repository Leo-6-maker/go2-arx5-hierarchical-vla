"""CPU fixtures for actual mature-resident seams, no native evidence.

Apply the three optional patches before running. Instances/models/physics are
synthetic. Existing RAW tests remain the default-path regression checks.
"""
from copy import deepcopy
from types import SimpleNamespace
import uuid
import numpy as np
import pytest
from scripts.pi0292.durable import read,write
from scripts.pi0292.relay_protocol import sealed,unseal
from scripts.pi0294.dec1.test_mapped_ack import fixture
from scripts.pi0294.resident import QueryReservations


def packets():
    # Existing CPU-only fixture supplies a complete real-format observation/query.
    from scripts.pi0292.test_neural_relay import TestNeuralRelayCPUOnly
    helper=TestNeuralRelayCPUOnly();helper.setUp()
    try:
        body,_=helper.inspect()
        c,_,_,action,command=fixture(run=helper.run,episode=helper.episode,identity=helper.identity)
        a=unseal(action);cmd=unseal(command)
        for key in ('query_uuid','reference_uuid','observation_sha256','anchor_sha256'):
            a['payload'][key]=body[key];cmd[key]=body[key]
        action=sealed(a);command=sealed(cmd)
        c,expected,ack,_,_=fixture(run=helper.run,episode=helper.episode,identity=helper.identity,
            action=action,command=command)
        return body,c,expected,action,command,ack
    finally:helper.tearDown()


@pytest.mark.parametrize('mutation',['none','noop','response-sha','stripped','wrong-mode','removed-contract'])
def test_actual_resident_consumes_full_mapped_ACK_and_original_completed_response(tmp_path,monkeypatch,mutation):
    body,c,e,action,command,ack=packets()
    q=QueryReservations(tmp_path,body['run_uuid'],max_queries=160,
        mapped_ack_contracts={body['episode_uuid']:c})
    q.begin(body,100.)
    q.complete(body,action['sha256'],prefix_sha256=e['prefix_sha256'],command_sha256=command['sha256'],
        action_packet=action,command_packet=command)
    assert q.state['queries'][-1]['response_sha256']==action['sha256']
    next_body=deepcopy(body);next_body.update(sequence=1,tick=200,query_uuid=str(uuid.uuid4()),previous_ack=ack)
    next_body['payload']['snapshot_uuid']=str(uuid.uuid4())
    if mutation=='noop':monkeypatch.setattr(c,'validate',lambda *a,**k:None)
    if mutation=='response-sha':q.state['queries'][-1]['response_sha256']='0'*64
    if mutation=='stripped':
        raw=unseal(ack);del raw['payload']['mapped_execution'];del raw['payload']['execution_semantics']
        next_body['previous_ack']=sealed(raw)
    if mutation=='wrong-mode':
        raw=unseal(ack);raw['payload']['mapped_execution']['interface_mode']='I_TIME'
        next_body['previous_ack']=sealed(raw)
    if mutation=='removed-contract':q.mapped_ack_contracts={}
    if mutation=='none':
        q.begin(next_body,102.)
        assert q.state['pending']['initial_adapter_ack_id']==ack['body']['payload']['mapped_execution']['references'][-1]['ack_id']
        assert next_body['previous_ack']==ack  # Never stripped or resealed.
    else:
        with pytest.raises(ValueError):q.begin(next_body,102.)
        assert q.state['pending'] is None and len(q.state['queries'])==1


def test_default_actual_resident_still_rejects_mapped_ACK_and_accepts_raw_only(tmp_path):
    body,c,e,action,command,ack=packets();q=QueryReservations(tmp_path,body['run_uuid'],160)
    q.begin(body,100.);q.complete(body,action['sha256'],prefix_sha256=e['prefix_sha256'],command_sha256=command['sha256'])
    later=deepcopy(body);later.update(sequence=1,tick=200,query_uuid=str(uuid.uuid4()),previous_ack=ack)
    later['payload']['snapshot_uuid']=str(uuid.uuid4())
    with pytest.raises(ValueError):q.begin(later,102.)
    raw=unseal(ack);del raw['payload']['mapped_execution'];del raw['payload']['execution_semantics']
    later['previous_ack']=sealed(raw);q.begin(later,102.)
    assert q.state['pending']['sequence']==1


def test_actual_resident_requires_full_response_packets_only_in_opted_in_completion(tmp_path):
    body,c,e,action,command,ack=packets()
    q=QueryReservations(tmp_path,body['run_uuid'],160,mapped_ack_contracts={body['episode_uuid']:c})
    q.begin(body,100.)
    with pytest.raises((ValueError,TypeError)):
        q.complete(body,action['sha256'],prefix_sha256=e['prefix_sha256'],command_sha256=command['sha256'])
    assert q.state['pending'] is not None and q.state['episodes'][body['episode_uuid']]['next_sequence']==0


@pytest.mark.parametrize('mapped',[False,True])
def test_e1_serve_passes_fixed_contract_and_original_sealed_response_to_actual_resident(tmp_path,monkeypatch,mapped):
    from scripts.pi0294.e1 import neural as unit
    body,c,e,action,command,ack=packets();root=tmp_path/body['run_uuid'];root.mkdir()
    label='CPU_SEAM_ONLY';plan=root/'plans'/label;plan.mkdir(parents=True)
    checkpoint_identity=dict(relay_identity=body['identity'],global_step=2048,
        checkpoint_manifest_sha256='94ad4d0eaaa5bf7df0c55a7b0c9c06701b4039fa2a32a98a29453a1974399c38')
    write(plan/'REGISTRY.json',dict(run_uuid=root.name,checkpoint_identity=checkpoint_identity,
        cases={body['episode_uuid']:dict(layout_id='synthetic-layout')}))
    (root/'pbs').mkdir();directory=root/'evidence'/label/'resident';calls=[];captured=[]
    instance=SimpleNamespace(identity=checkpoint_identity)
    def generate(payload,seed):
        calls.append(seed)
        return dict(physical_H40=np.asarray(unseal(action)['payload']['trajectory_eef10']),
            normalized=np.zeros((1,40,10)),noise={'synthetic':True},inference_precision='CPU_FIXTURE_ONLY')
    instance.generate=generate
    class Ledger:
        def charge(self,key,count):captured.append((key,count))
    def guard():
        path=directory/'inbox'/(body['query_uuid']+'.json')
        if not path.exists():write(path,sealed(body))
        if (directory/'outbox'/(body['query_uuid']+'.json')).exists():
            write(directory/'STAGE_DONE.json',dict(run_uuid=root.name,job_id='cpu-fixture-job',label=label))
    monkeypatch.setenv('PBS_JOBID','cpu-fixture-job')
    monkeypatch.setattr(unit.time,'time',lambda:100.)
    monkeypatch.setattr(unit.time,'sleep',lambda _:None)
    monkeypatch.setattr(unit.signal,'signal',lambda *a:None)
    monkeypatch.setattr(unit.signal,'alarm',lambda *a:None)
    contracts={body['episode_uuid']:c} if mapped else None
    options={} if contracts is None else dict(mapped_ack_contracts=contracts)
    unit.serve(root,label,instance,dict(allocation_end_epoch=1000.),Ledger(),guard,
        max_queries=80,**options)
    state=read(directory/'QUERY_STATE.json');result=read(directory/'outbox'/(body['query_uuid']+'.json'))
    assert len(calls)==1 and captured==[('generation_queries',1)]
    assert state['queries'][0]['response_sha256']==result['sha256']
    if mapped:
        binding=state['queries'][0]['mapped_ack_binding']
        assert binding['response_sha256']==result['sha256']
        assert binding['raw_h40_sha256']==e['raw_h40_sha256']
        assert state['mapped_ack_conditions']=={body['episode_uuid']:c.condition_sha256}
    else:assert 'mapped_ack_binding' not in state['queries'][0]


def test_owner_route_explicitly_injects_contract_but_never_from_case_flag(tmp_path):
    from scripts.pi0294 import relay_transport as unit
    from scripts.pi0294.test_relay_transport import RelayTransportTests
    helper=RelayTransportTests();case,core,hopper,now,events,installs=helper.fixture()
    model_identity=packets()[0]['identity']
    def model_response(cas,query,teacher):
        assert teacher is None
        body=unseal(query)
        return sealed(dict(protocol=body['protocol'],kind='action',run_uuid=body['run_uuid'],
            episode_uuid=body['episode_uuid'],sequence=body['sequence'],tick=body['tick'],
            payload={k:body[k] for k in ('query_uuid','reference_uuid','observation_sha256','anchor_sha256','identity')}|
                dict(trajectory_eef10=[body['payload']['state_eef10']]*40,offsets_ns=[i*50000000 for i in range(40)])))
    case['identity']=model_identity;hopper.query=model_response
    c,_,_,_,_=fixture(run=case['run_uuid'],episode=case['episode_uuid'],identity=case['identity'])
    original=core.ack;received=[]
    def read_ack(cas,sequence):
        if sequence==1:raise RuntimeError('BOUNDED_FIXTURE_STOP_AFTER_ONE_FULL_NEW_ACK')
        return received[-1]
    def install(cas,sequence,bundle):
        query=unseal(bundle['request'])
        # Compose a fresh synthetic full mapped ACK from the SAME raw packets.
        _,e,p,_,_=fixture(run=case['run_uuid'],episode=case['episode_uuid'],identity=case['identity'],
            action=bundle['action'],command=bundle['command'])
        received.append(p);installs.append(sequence)
    core.install=install;core.ack=read_ack
    # First mapped ACK validates; next query reaches route without resealing it.
    observation=core.observation
    def stop_at_second(cas,sequence):
        if sequence==1:raise RuntimeError('BOUNDED_FIXTURE_STOP_AFTER_ONE_FULL_NEW_ACK')
        return observation(cas,sequence)
    core.observation=stop_at_second
    with pytest.raises(RuntimeError,match='BOUNDED_FIXTURE_STOP'):
        unit.run_episode(tmp_path/'mapped',case,core,hopper,now=now,mapped_ack_contract=c)
    route=read(tmp_path/'mapped/STATE.json')
    assert route['sequence']==1 and route['tick']==200 and route['history'][0]['ack_sha256']==received[0]['sha256']
    # A case dictionary flag does not opt the default route in.
    case,core,hopper,now,events,installs=helper.fixture();case['mapped_ack_contract']=c
    case['identity']=model_identity;hopper.query=model_response
    original_install=core.install
    def wrong_mapped(cas,sequence,bundle):
        original_install(cas,sequence,bundle)
        _,_,p,_,_=fixture(run=case['run_uuid'],episode=case['episode_uuid'],identity=case['identity'],
            action=bundle['action'],command=bundle['command'])
        received.append(p)
    core.install=wrong_mapped;core.ack=lambda cas,s:received[-1]
    with pytest.raises(ValueError):unit.run_episode(tmp_path/'default',case,core,hopper,now=now)
    assert read(tmp_path/'default/STATE.json')['tick']==0
