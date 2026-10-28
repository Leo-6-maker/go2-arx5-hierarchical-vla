"""CPU binding fixtures. Hash binding examples are NOT native evidence.

Consumer cases require applying the optional consumer patches first. No model,
MuJoCo, network, scheduler or physics primitive is invoked by these tests.
"""
from copy import deepcopy
import hashlib
import json
import uuid
import numpy as np
import pytest
from scripts.pi0292.relay_protocol import PROTOCOL,FRAME,TCP,digest,sealed,unseal
from scripts.pi0294.dec1 import mapped_ack as unit


def fixture(*,mode='I_PROGRESS',run='synthetic-run',episode=None,identity=None,action=None,command=None):
    episode=episode or str(uuid.uuid4())
    identity=identity or dict(profile='B1',model_revision='a'*40,source_sha='b'*40,
        config_sha256='c'*64,processor_sha256='d'*64,statistics_sha256='e'*64)
    condition=dict(schema=unit.CONDITION_SCHEMA,condition_id='sealed-condition-0',run_uuid=run,
        episode_uuid=episode,executor_id='synthetic-executor',interface_mode=mode,policy='CORR2048',
        noise_bank='R0',identity=identity,execution_semantics=unit.SEMANTICS,frame=FRAME,tcp=TCP,
        owner_release='f'*40,scientific_plan_sha256='a'*64)
    raw=(json.dumps(sealed(condition),sort_keys=True)+'\n').encode()
    contract=unit.FixedMappedAckContract(raw,expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_condition_id=condition['condition_id'],expected_owner_release='f'*40,
        expected_plan_sha256='a'*64,expected_identity=identity)
    if action is None:
        query=str(uuid.uuid4());reference=str(uuid.uuid4())
        rows=[[.4,0.,.4,1.,0.,0.,0.,1.,0.,.5] for _ in range(40)]
        action=sealed(dict(protocol=PROTOCOL,kind='action',run_uuid=run,episode_uuid=episode,sequence=0,tick=0,
            payload=dict(query_uuid=query,reference_uuid=reference,observation_sha256='a'*64,
                anchor_sha256='b'*64,identity=identity,trajectory_eef10=rows,
                offsets_ns=[i*50000000 for i in range(40)])))
        command=sealed(dict(protocol=PROTOCOL,kind='install_prefix8',run_uuid=run,episode_uuid=episode,
            sequence=0,tick=0,query_uuid=query,reference_uuid=reference,observation_sha256='a'*64,
            anchor_sha256='b'*64,prefix_eef10=rows[:8],prefix_sha256=digest(rows[:8]),
            advance_integrations=200,frame=FRAME,tcp=TCP,expires=160.))
    expected=contract.bind_action(action,command,initial_adapter_ack_id=None)
    refs=[];prior=None;poses=[];grips=[]
    for i in range(8):
        target=np.eye(4);target[:3,3]=[.4+i*.002,0.,.4];poses.append(target);grips.append(.5)
        name=f'{episode}:{i}';proposal=digest(dict(reference=i,source='synthetic-only'))
        refs.append(dict(reference=i,command_id=name,proposal_sha256=proposal,phase=i*.05,
            predecessor_ack_id=prior,ack_id=name,tick_before=i*25,tick_after=(i+1)*25,
            execution_end_timestamp=(i+1)*.05,target=target.tolist(),opening=.5));prior=name
    action_hash=hashlib.sha256(np.asarray(poses,dtype=float).tobytes()+np.asarray(grips,dtype=float).tobytes()).hexdigest()
    nonce=str(uuid.uuid4())
    mapping=dict(schema=unit.SCHEMA,condition_sha256=contract.condition_sha256,
        condition_id=condition['condition_id'],interface_mode=mode,identity=identity,
        executor_id=condition['executor_id'],response_sha256=action['sha256'],
        raw_h40_sha256=expected['raw_h40_sha256'],mapped_prefix_sha256=action_hash,nonce=nonce,
        initial_adapter_ack_id=None,frame=FRAME,tcp=TCP,
        s1=dict(version=unit.S1_VERSION,decision='PASS',nonce=nonce,state_sha256='b'*64,
            action_sha256=action_hash,shadow_integrations=200),
        shadow_proposal_sha256=[r['proposal_sha256'] for r in refs],shadow_phases=[r['phase'] for r in refs],
        references=refs,maximum_live_shadow_difference=0.)
    packet=contract.make_packet(expected=expected,mapping=mapping,physics_receipt_sha256='e'*64)
    return contract,expected,packet,action,command


@pytest.mark.parametrize('mode',['I_TIME','I_PROGRESS'])
def test_fixed_condition_binds_raw_and_mapped_identity_without_resealing_old_ack(mode):
    c,e,p,a,command=fixture(mode=mode)
    checked=c.validate(p,expected=e)
    assert isinstance(checked,unit.ValidatedMappedAck)
    assert checked.packet_sha256==p['sha256'] and checked.body==unseal(p)
    assert p['body']['payload']['prefix_sha256']==unseal(command)['prefix_sha256']
    assert p['body']['payload']['mapped_execution']['response_sha256']==a['sha256']
    assert p['body']['payload']['mapped_execution']['mapped_prefix_sha256']!=e['prefix_sha256']


@pytest.mark.parametrize('path,value',[
    (('payload','command_sha256'),'f'*64),
    (('payload','prefix_sha256'),'f'*64),
    (('payload','tick_after'),199),
    (('payload','integrations'),True),
    (('payload','physics_receipt_sha256'),''),
    (('payload','execution_semantics'),'RAW_PREFIX8'),
    (('payload','mapped_execution','response_sha256'),'f'*64),
    (('payload','mapped_execution','raw_h40_sha256'),'f'*64),
    (('payload','mapped_execution','interface_mode'),'I_TIME'),
    (('payload','mapped_execution','identity','model_revision'),'f'*40),
    (('payload','mapped_execution','s1','decision'),'REJECT'),
    (('payload','mapped_execution','s1','shadow_integrations'),199),
    (('payload','mapped_execution','s1','nonce'),str(uuid.uuid4())),
    (('payload','mapped_execution','s1','action_sha256'),'f'*64),
    (('payload','mapped_execution','s1','state_sha256'),'invalid'),
    (('payload','mapped_execution','maximum_live_shadow_difference'),1e-9),
    (('payload','mapped_execution','initial_adapter_ack_id'),'unrelated'),
])
def test_resealed_wrong_identity_hash_clock_or_safety_chain_rejected(path,value):
    c,e,p,_,_=fixture();body=unseal(p);node=body
    for key in path[:-1]:node=node[key]
    node[path[-1]]=value
    with pytest.raises((ValueError,TypeError,KeyError)):c.validate(sealed(body),expected=e)


@pytest.mark.parametrize('mutation',['duplicate','missing','reorder','clock','phase','target','opening','predecessor','ack','proposal'])
def test_eight_actual_reference_ack_chain_and_same_mapped_action_required(mutation):
    c,e,p,_,_=fixture();body=unseal(p);refs=body['payload']['mapped_execution']['references']
    if mutation=='duplicate':refs[3]=deepcopy(refs[2])
    if mutation=='missing':refs.pop()
    if mutation=='reorder':refs[2],refs[3]=refs[3],refs[2]
    if mutation=='clock':refs[3]['execution_end_timestamp']+=.002
    if mutation=='phase':refs[3]['phase']+=.01
    if mutation=='target':refs[3]['target'][0][3]+=.001
    if mutation=='opening':refs[3]['opening']=.1
    if mutation=='predecessor':refs[3]['predecessor_ack_id']='unrelated'
    if mutation=='ack':refs[3]['ack_id']='unrelated'
    if mutation=='proposal':refs[3]['proposal_sha256']='f'*64
    with pytest.raises(ValueError):c.validate(sealed(body),expected=e)


def test_old_raw_ack_is_not_mapped_execution_and_no_arbitrary_request_flag_allowed():
    c,e,p,_,_=fixture();body=unseal(p)
    del body['payload']['mapped_execution'];del body['payload']['execution_semantics']
    with pytest.raises(ValueError):c.validate(sealed(body),expected=e)
    body=unseal(p);body['enable_mapped_ack']=True
    with pytest.raises(ValueError):c.validate(sealed(body),expected=e)


def test_raw_h40_hash_binds_tail_beyond_first8_and_native_float64_encoding():
    c,e,p,a,command=fixture();body=unseal(a);rows=body['payload']['trajectory_eef10']
    assert e['raw_h40_sha256']==hashlib.sha256(np.asarray(rows,dtype=float).tobytes()).hexdigest()
    rows[39][0]+=.001
    changed=c.bind_action(sealed(body),command,initial_adapter_ack_id=None)
    assert changed['prefix_sha256']==e['prefix_sha256']
    assert changed['raw_h40_sha256']!=e['raw_h40_sha256']
    assert changed['response_sha256']!=e['response_sha256']
    with pytest.raises(ValueError):c.validate(p,expected=changed)


def test_condition_byte_hash_and_owner_pins_required():
    c,e,p,_,_=fixture();condition=c.condition
    raw=(json.dumps(sealed(condition))+'\n').encode()
    pins=dict(expected_sha256=hashlib.sha256(raw).hexdigest(),expected_condition_id=condition['condition_id'],
        expected_owner_release='f'*40,expected_plan_sha256='a'*64,expected_identity=condition['identity'])
    with pytest.raises(ValueError):unit.FixedMappedAckContract(raw+b' ',**pins)
    condition['policy']='A4096';other=(json.dumps(sealed(condition))+'\n').encode()
    pins['expected_sha256']=hashlib.sha256(other).hexdigest()
    with pytest.raises(ValueError):unit.FixedMappedAckContract(other,**pins)


def test_optional_exchange_keeps_whole_mapped_ack_as_next_query_previous_ack(tmp_path):
    # Reuse only the existing pure-CPU RGB packet fixture. No native engine.
    from scripts.pi0292.test_relay_protocol import TestPausedRelayProtocol
    from scripts.pi0292.relay_protocol import RelayExchange
    helper=TestPausedRelayProtocol();helper.setUp()
    try:
        c,_,_,_,_=fixture(run=helper.run,episode=helper.episode,identity=helper.identity)
        exchange=RelayExchange(tmp_path/'mapped',run_uuid=helper.run,episode_uuid=helper.episode,
            identity=helper.identity,now=100.,mapped_ack_contract=c)
        try:
            request=exchange.observe(helper.observation(),101.);action=helper.response(request)
            command=exchange.action(action,102.)
            _,expected,packet,_,_=fixture(run=helper.run,episode=helper.episode,identity=helper.identity,
                action=action,command=command)
            exchange.acknowledge(packet,103.)
            fresh=exchange.observe(helper.observation(1,200),104.)
            assert unseal(fresh)['previous_ack']==packet
            assert exchange.state['history'][0]['ack_sha256']==packet['sha256']
        finally:exchange.close()
    finally:helper.tearDown()


@pytest.mark.parametrize('mutation',['none','noop','response-sha'])
def test_optional_query_consumer_checks_original_response_sha_and_complete_packet(tmp_path,monkeypatch,mutation):
    from scripts.pi0292.neural_relay import QueryReservations
    from scripts.pi0292.test_neural_relay import TestNeuralRelayCPUOnly
    helper=TestNeuralRelayCPUOnly();helper.setUp()
    try:
        body,_=helper.inspect();c,_,_,_,_=fixture(run=helper.run,episode=helper.episode,identity=helper.identity)
        q=QueryReservations(tmp_path,helper.run,mapped_ack_contracts={helper.episode:c})
        q.begin(body,100.)
        # Build the same response/command identities the original service seals.
        _,_,_,a,cmd=fixture(run=helper.run,episode=helper.episode,identity=helper.identity)
        ab=unseal(a);cb=unseal(cmd)
        for key in ('query_uuid','reference_uuid','observation_sha256','anchor_sha256'):
            ab['payload'][key]=body[key];cb[key]=body[key]
        a=sealed(ab);cmd=sealed(cb)
        _,e,p,_,_=fixture(run=helper.run,episode=helper.episode,identity=helper.identity,action=a,command=cmd)
        q.complete(body,a['sha256'],prefix_sha256=e['prefix_sha256'],command_sha256=cmd['sha256'],
            action_packet=a,command_packet=cmd)
        following=deepcopy(body);following.update(sequence=1,tick=200,query_uuid=str(uuid.uuid4()),previous_ack=p)
        following['payload']['snapshot_uuid']=str(uuid.uuid4())
        if mutation=='noop':monkeypatch.setattr(c,'validate',lambda *a,**k:None)
        if mutation=='response-sha':q.state['queries'][-1]['response_sha256']='0'*64
        if mutation=='none':
            q.begin(following,102.)
            assert q.state['pending']['initial_adapter_ack_id']==p['body']['payload']['mapped_execution']['references'][-1]['ack_id']
            assert following['previous_ack']==p
        else:
            with pytest.raises(ValueError):q.begin(following,102.)
            assert q.state['pending'] is None
    finally:helper.tearDown()


def test_default_raw_exchange_still_rejects_mapped_packet():
    from scripts.pi0292.test_relay_protocol import TestPausedRelayProtocol
    helper=TestPausedRelayProtocol();helper.setUp()
    try:
        request=helper.exchange.observe(helper.observation(),101.);action=helper.response(request)
        command=helper.exchange.action(action,102.)
        _,_,packet,_,_=fixture(run=helper.run,episode=helper.episode,identity=helper.identity,
            action=action,command=command)
        with pytest.raises(ValueError):helper.exchange.acknowledge(packet,103.)
        assert helper.exchange.state['phase']=='STOPPED' and helper.exchange.state['tick']==0
    finally:helper.tearDown()


def test_optional_exchange_rejects_noop_validator_without_progress(tmp_path,monkeypatch):
    from scripts.pi0292.test_relay_protocol import TestPausedRelayProtocol
    from scripts.pi0292.relay_protocol import RelayExchange
    helper=TestPausedRelayProtocol();helper.setUp()
    try:
        c,_,_,_,_=fixture(run=helper.run,episode=helper.episode,identity=helper.identity)
        exchange=RelayExchange(tmp_path/'noop',run_uuid=helper.run,episode_uuid=helper.episode,
            identity=helper.identity,now=100.,mapped_ack_contract=c)
        try:
            request=exchange.observe(helper.observation(),101.);action=helper.response(request)
            command=exchange.action(action,102.)
            _,_,packet,_,_=fixture(run=helper.run,episode=helper.episode,identity=helper.identity,
                action=action,command=command)
            monkeypatch.setattr(c,'validate',lambda *a,**k:None)
            with pytest.raises(ValueError,match='COMPLETE_MAPPED_ACK_VALIDATION_REQUIRED'):
                exchange.acknowledge(packet,103.)
            assert exchange.state['tick']==0 and exchange.state['history']==[]
        finally:exchange.close()
    finally:helper.tearDown()


def producer_fixture(tmp_path):
    from types import SimpleNamespace
    from scripts.pi0294.dec1.mapped_relay_ack import make_mapped_relay_acknowledger
    c,e,p,action,command=fixture();condition=c.condition;m=p['body']['payload']['mapped_execution']
    directory=tmp_path/'relay';directory.mkdir();(directory/'acks').mkdir()
    relay=SimpleNamespace(acknowledged=0,pending=None,request=dict(run_uuid=condition['run_uuid'],
        episode_uuid=condition['episode_uuid'],identity=condition['identity']),directory=directory,
        state={},guard_prefix=lambda:None)
    callback=make_mapped_relay_acknowledger(relay,c)
    relay.pending=dict(command=unseal(command),command_sha256=command['sha256'],
        command_packet=command,action_packet=action)
    proof=dict(episode_id=condition['episode_uuid'],executor_id=condition['executor_id'],nonce=m['nonce'],
        raw_chunk_id=e['reference_uuid'],actual_ticks=200,tick_before=0,tick_after=200,
        raw_h40_sha256=m['raw_h40_sha256'],action_sha256=m['mapped_prefix_sha256'],
        initial_adapter_ack_id=None,s1=m['s1'],proposal_digests=m['shadow_proposal_sha256'],
        phases=m['shadow_phases'],actual_reference_acks=m['references'],maximum_live_shadow_difference=0.,
        ABI='SINGLE_4x4_POSE_PLUS_GRIPPER_NOT_PREVIEW9')
    engine=SimpleNamespace(steps=200,safety=[],model=SimpleNamespace(nq=0,nv=0,nu=0))
    rows=np.zeros((200,8));rows[:,0]=np.arange(1,201)*.002
    return callback,relay,engine,rows,proof,c,e


def test_owner_callback_readbacks_original_physics_and_full_mapped_packet(tmp_path):
    from scripts.pi0292.durable import read
    from scripts.pi0294.dec1.source_episode_wire import NativeMappedPrefixAck
    callback,relay,engine,rows,proof,c,e=producer_fixture(tmp_path)
    ack=callback(engine,rows,proof)
    assert isinstance(ack,NativeMappedPrefixAck) and relay.pending is None and relay.acknowledged==1
    packet=read(relay.directory/'acks/000.json');c.validate(packet,expected=e)
    assert ack.acknowledgement_sha256==packet['sha256']
    assert packet['body']['payload']['mapped_execution']['references']==proof['actual_reference_acks']
    assert (relay.directory/'physics-000.npz').exists()
    with pytest.raises(ValueError):callback(engine,rows,proof)


@pytest.mark.parametrize('mutation',['unknown-proposal','unsafe-rows','wrong-clock','S1-reject','partial200'])
def test_owner_callback_rejects_missing_or_wrong_actual_safety_clock_binding_without_ACK(tmp_path,mutation):
    callback,relay,engine,rows,proof,c,e=producer_fixture(tmp_path)
    if mutation=='unknown-proposal':proof['actual_reference_acks'][3]['proposal_sha256']='f'*64
    if mutation=='unsafe-rows':rows[37,-1]=1.
    if mutation=='wrong-clock':rows[37,0]+=.002
    if mutation=='S1-reject':proof['s1']['decision']='REJECT'
    if mutation=='partial200':engine.steps=199
    with pytest.raises(ValueError):callback(engine,rows,proof)
    assert relay.acknowledged==0 and not (relay.directory/'acks/000.json').exists()
    # Even malformed proof after a real200 effect is never fixed and retried.
    with pytest.raises(ValueError,match='UNKNOWN_NO_RETRY'):callback(engine,rows,proof)
