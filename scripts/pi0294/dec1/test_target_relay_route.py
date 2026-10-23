"""Original durable relay + target typed ACK integration; entirely CPU fixtures.

No network, physics, model, solver, task-native proof or quota is produced.
Existing default RAW/source paths retain their own regression tests.
"""
from copy import deepcopy
import uuid
import pytest
from scripts.pi0292.durable import read
from scripts.pi0292.relay_protocol import RelayExchange,PROTOCOL,sealed,unseal,digest
from scripts.pi0294.relay_transport import run_episode
from scripts.pi0294.resident import QueryReservations
from scripts.pi0294.dec1.test_target_mapped_ack import fixture
from scripts.pi0294.dec1.target_relay_exchange import TargetRelayExchange


class BoundedCPUStop(RuntimeError):pass


def endpoints(tmp_path,mode,fault):
    body,c,_,base_action,_,_=fixture(mode);rows=unseal(base_action)['payload']['trajectory_eef10']
    cas=dict(run_uuid=body['run_uuid'],episode_uuid=body['episode_uuid'],identity=body['identity'],deadline_epoch=1000.)
    qdir=tmp_path/'neural';qdir.mkdir();resident=QueryReservations(qdir,cas['run_uuid'],80,mapped_ack_contracts={cas['episode_uuid']:c})
    seen=dict(observations=[],queries=[],installs=[],acks={},last_adapter=None,draws=0);clock=[100.]
    def now():clock[0]+=.01;return clock[0]
    class Core:
        def observation(self,case,sequence):
            if sequence==2:raise BoundedCPUStop('TWO_REAL_FORMAT_PREFIXES_ONLY_CPU_FIXTURE')
            if sequence and sequence-1 not in seen['acks']:raise AssertionError('NO_OBSERVATION_BEFORE_ACK')
            seen['observations'].append(sequence);payload=deepcopy(body['payload'])
            payload.update(snapshot_uuid=str(uuid.uuid4()),state_eef10=rows[0])
            return sealed(dict(protocol=PROTOCOL,kind='observation',run_uuid=cas['run_uuid'],episode_uuid=cas['episode_uuid'],
                sequence=sequence,tick=sequence*200,payload=payload))
        def teacher(self,*_):raise AssertionError('MODEL_ROUTE_CANNOT_ASK_TEACHER_FOR_ACTION')
        def install(self,case,sequence,bundle):
            seen['installs'].append(sequence)
            if fault=='unknown_install':raise ConnectionError('CPU_UNKNOWN_SEND_RETURN')
            _,ack_contract,expected,action,command,ack=fixture(mode,sequence,seen['last_adapter'],
                generation=(bundle['request'],bundle['action'],bundle['command']))
            assert ack_contract.condition_sha256==c.condition_sha256
            assert resident.state['queries'][-1]['expected_command_sha256']==command['sha256']
            if fault=='bad_ack':
                b=unseal(ack);b['payload']['mapped_execution']['references'][3]['installation_events'].pop();ack=sealed(b)
            seen['acks'][sequence]=ack
            if fault!='bad_ack':seen['last_adapter']=c.validate(ack,expected=expected).final_adapter_ack_id
        def ack(self,case,sequence):return seen['acks'].get(sequence)
    class Hopper:
        def query(self,case,request,teacher):
            assert teacher is None;query=unseal(request);seen['queries'].append(query['query_uuid']);resident.begin(query,now())
            seen['draws']+=1
            if fault=='unknown_generation':raise ConnectionError('CPU_UNKNOWN_DRAW_RETURN')
            action=sealed(dict(protocol=PROTOCOL,kind='action',run_uuid=case['run_uuid'],episode_uuid=case['episode_uuid'],
                sequence=query['sequence'],tick=query['tick'],payload={k:query[k] for k in
                ('query_uuid','reference_uuid','observation_sha256','anchor_sha256','identity')}|
                dict(trajectory_eef10=rows,offsets_ns=[i*50000000 for i in range(40)])))
            # Mirror the mature e1 consumer: a known generation completes on
            # Hopper before the independently owned physical send can occur.
            command=sealed(dict(protocol=PROTOCOL,kind='install_prefix8',run_uuid=case['run_uuid'],episode_uuid=case['episode_uuid'],
                sequence=query['sequence'],tick=query['tick'],**{k:query[k] for k in
                ('query_uuid','reference_uuid','observation_sha256','anchor_sha256')},
                prefix_eef10=rows[:8],prefix_sha256=digest(rows[:8]),advance_integrations=200,
                frame=query['payload']['frame'],tcp=query['payload']['tcp'],expires=query['expires']))
            resident.complete(query,action['sha256'],prefix_sha256=digest(rows[:8]),command_sha256=command['sha256'],
                action_packet=action,command_packet=command)
            return action
        def response(self,*_):raise AssertionError('NO_SECOND_GENERATION_ATTEMPT')
    return cas,c,Core(),Hopper(),now,seen,resident


@pytest.mark.parametrize('mode',['I_TIME','I_PROGRESS'])
@pytest.mark.parametrize('fault',['none','bad_ack','unknown_install','unknown_generation'])
def test_actual_route_and_resident_share_the_unmodified_target_ack_and_preserve_failures(tmp_path,mode,fault):
    cas,c,core,hopper,now,seen,resident=endpoints(tmp_path,mode,fault);route=tmp_path/'route'
    expected_error=BoundedCPUStop if fault=='none' else PermissionError if fault=='bad_ack' else ConnectionError
    with pytest.raises(expected_error):run_episode(route,cas,core,hopper,now=now,mapped_ack_contract=c)
    state=read(route/'STATE.json');routing=read(route/'ROUTE.json')
    if fault=='none':
        assert state['sequence']==2 and state['tick']==400 and routing['acknowledgements']==2
        assert seen['observations']==[0,1] and seen['installs']==[0,1] and seen['draws']==2
        assert len(set(seen['queries']))==2
        assert read(route/'ack-001.json')==seen['acks'][1]
        assert state['mapped_last_adapter_ack_id']==seen['last_adapter']
        assert resident.state['queries'][1]['mapped_ack_binding']['initial_adapter_ack_id']==seen['acks'][0]['body']['payload']['mapped_execution']['references'][-1]['prepared']['command_id']
    else:
        assert state['sequence']==0 and state['tick']==0 and routing['acknowledgements']==0
        assert seen['observations']==[0] and seen['draws']==1
        if fault in ('unknown_install','bad_ack'):
            assert seen['installs']==[0] and state['pending']['command_sha256']
            assert resident.state['pending'] is None  # Neural known, physical outcome unknown.
        if fault=='bad_ack':assert state['phase']=='STOPPED' and state['unacknowledged_physics']=='UNKNOWN_DO_NOT_RESEND_COMMAND'
        if fault=='unknown_generation':assert resident.state['pending'] is not None and seen['installs']==[]
        with pytest.raises(FileExistsError):run_episode(route,cas,core,hopper,now=now,mapped_ack_contract=c)
        assert seen['draws']==1


@pytest.mark.parametrize('change',['omit','other_condition'])
def test_reopening_original_relay_cannot_drop_or_change_cold_target_contract(tmp_path,change):
    body,c,*_=fixture();x=TargetRelayExchange(tmp_path,run_uuid=body['run_uuid'],episode_uuid=body['episode_uuid'],identity=body['identity'],now=100.,mapped_ack_contract=c);x.close()
    if change=='other_condition':
        from scripts.pi0294.dec1.test_target_mapped_ack import contract
        d=c.condition;d['condition_id']='different-condition';c=contract(d)
    else:c=None
    if c is None:
        with pytest.raises(TypeError,match='EXPLICIT_COLD_TARGET_ACK_CONTRACT_REQUIRED'):
            TargetRelayExchange(tmp_path,mapped_ack_contract=c)
    else:
        with pytest.raises(FileExistsError,match='NEVER_REOPEN_OR_RESTART'):
            TargetRelayExchange(tmp_path,mapped_ack_contract=c)


def test_original_sealed_source_glue_still_rejects_target_contract(tmp_path):
    import hashlib
    from pathlib import Path
    from scripts.pi0294.dec1.source_cpu_intake import GLUE_2
    path='scripts/pi0292/relay_protocol.py';root=Path(__file__).resolve().parents[3]
    assert hashlib.sha256((root/path).read_bytes()).hexdigest()==GLUE_2[path]['current_sha256']
    body,c,*_=fixture()
    with pytest.raises(TypeError,match='OWNER_PINNED_MAPPED_ACK_CONTRACT_REQUIRED'):
        RelayExchange(tmp_path,run_uuid=body['run_uuid'],episode_uuid=body['episode_uuid'],identity=body['identity'],now=100.,mapped_ack_contract=c)


def test_target_overlay_inherits_original_action_intents_ACK_and_stop_implementation():
    for method in ('observe','action','acknowledge','_clock','_fail','check_timeout','close'):
        assert getattr(TargetRelayExchange,method) is getattr(RelayExchange,method)


@pytest.mark.parametrize('fault',['owner_identity','source_contract'])
def test_target_factory_rejects_wrong_cold_binding_before_creating_route(tmp_path,fault):
    body,c,*_=fixture();folder=tmp_path/'not-created';identity=deepcopy(body['identity'])
    if fault=='owner_identity':identity['source_sha']='0'*40
    else:
        from scripts.pi0294.dec1.test_mapped_ack import fixture as source_fixture
        c,*_=source_fixture()
    with pytest.raises((TypeError,ValueError)):
        TargetRelayExchange(folder,run_uuid=body['run_uuid'],episode_uuid=body['episode_uuid'],identity=identity,now=100.,mapped_ack_contract=c)
    assert not folder.exists()


def test_first_durable_target_state_is_pinned_even_if_write_return_is_lost(tmp_path,monkeypatch):
    from scripts.pi0294.dec1 import target_relay_exchange as module
    body,c,*_=fixture();original_write=module.write;captured=[]
    def lose_return(path,value):
        captured.append(deepcopy(value));original_write(path,value)
        raise ConnectionError('CPU_LOST_COLD_STATE_WRITE_RETURN')
    monkeypatch.setattr(module,'write',lose_return)
    with pytest.raises(ConnectionError):TargetRelayExchange(tmp_path,run_uuid=body['run_uuid'],
        episode_uuid=body['episode_uuid'],identity=body['identity'],now=100.,mapped_ack_contract=c)
    assert len(captured)==1 and captured[0]['mapped_ack_condition_sha256']==c.condition_sha256
    assert read(tmp_path/'STATE.json')==captured[0]
    with pytest.raises(ValueError,match='COLD_ACK_CONTRACT_CHANGED_OR_OMITTED'):
        RelayExchange(tmp_path)
    with pytest.raises(FileExistsError):TargetRelayExchange(tmp_path,mapped_ack_contract=c)
    assert len(captured)==1


@pytest.mark.parametrize('mode',['I_TIME','I_PROGRESS'])
def test_case_flag_cannot_select_target_contract_in_default_raw_route(tmp_path,mode):
    cas,c,core,hopper,now,seen,_=endpoints(tmp_path,mode,'none')
    cas['mapped_ack_contract']=c
    with pytest.raises(ValueError):run_episode(tmp_path/'route',cas,core,hopper,now=now)
    state=read(tmp_path/'route/STATE.json')
    assert state['phase']=='STOPPED' and state['tick']==0 and seen['draws']==1
    assert state['unacknowledged_physics']=='UNKNOWN_DO_NOT_RESEND_COMMAND'
