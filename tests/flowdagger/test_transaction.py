import copy

import pytest

from go2arx5_vla.flowdagger.abi import SOURCE_ABI
from go2arx5_vla.flowdagger.contracts import Candidate
from go2arx5_vla.flowdagger.transaction import ChunkTransaction
from .test_codec import action
from .test_contracts import proof


def transaction():
    return ChunkTransaction('synthetic-episode',max_chunks=2,context=dict(source_domain='FIXED_BASE_ARX5_TASK_SIM',
        task='T1',split='TRAIN',scene_group='fixture-only',language='pick and place',model_sha256='b'*64,statistics_sha256='c'*64))


def observe(t,identifier,tick):
    return t.observe(observation_id=identifier,tick=tick,timestamp_ns=tick*2_000_000,
        measured_state10=action()[0],rgb_sha256=['a'*64,'b'*64],binding=SOURCE_ABI)


def choose(t,identifier='obs-1',student_safe=True,expert_safe=True,intervene=False):
    s,e=Candidate(identifier,'STUDENT',action(),40),Candidate(identifier,'EXPERT',action(),40)
    return t.choose(s,proof(s,student_safe),e,proof(e,expert_safe),expert_valid=True,
        request_intervention=intervene,student_noise_id='SYNTHETIC_DRAW_0')


def ack(t,intent,**change):
    values=dict(intent_sha256=intent['intent_sha256'],installed_prefix=intent['prefix'],
        tick_before=t.tick,tick_after=t.tick+200,native_safe=True,object_contact=False)
    values.update(change);return t.acknowledge(**values)


def test_cpu_chunk_handshake_advances_without_contact_not_fake_success():
    t=transaction();observe(t,'obs-1',0);choose(t);intent=t.dispatch_intent()
    row=ack(t,intent)
    assert t.tick==200 and row['execution']['integrations']==200
    assert not row['execution']['object_contact'] and not row['successful_BC_demo_admitted']
    observe(t,'obs-2',200);choose(t,'obs-2');ack(t,t.dispatch_intent())
    assert t.phase=='WAIT_OBSERVATION'
    observe(t,'final',400);assert t.phase=='COMPLETE'
    with pytest.raises(ValueError):choose(t,'final')


def test_expert_takeover_and_all_rejected_stop():
    t=transaction();observe(t,'obs-1',0)
    assert choose(t,student_safe=False)['selected_executor']=='EXPERT'
    row=ack(t,t.dispatch_intent());assert row['decision']['selected_executor']=='EXPERT'
    t=transaction();observe(t,'obs-1',0)
    assert choose(t,student_safe=False,expert_safe=False)['selected_executor']=='STOP'
    assert t.phase=='STOPPED' and t.records[0]['bank']=='FAILURE_DIAGNOSTIC'
    with pytest.raises(ValueError):t.dispatch_intent()


def test_repeated_chunk_clock_ack_and_unknown_dispatch_are_not_replayed():
    t=transaction();observe(t,'obs-1',0);choose(t);intent=t.dispatch_intent()
    with pytest.raises(ValueError):t.dispatch_intent()
    with pytest.raises(ValueError):observe(t,'obs-2',200)
    ack(t,intent)
    with pytest.raises(ValueError):ack(t,intent)
    with pytest.raises(ValueError):observe(t,'obs-1',200)
    with pytest.raises(ValueError):observe(t,'obs-2',0)
    t=transaction();observe(t,'obs-1',0);choose(t);t.dispatch_intent();t.unknown_dispatch()
    assert t.phase=='UNKNOWN_STOPPED_NO_RETRY'
    with pytest.raises(ValueError):t.dispatch_intent()


def test_partial_and_safety_end_remain_explicit_not_success_bank():
    t=transaction();observe(t,'obs-1',0);choose(t);intent=t.dispatch_intent()
    with pytest.raises(ValueError):ack(t,intent,tick_after=25)
    row=ack(t,intent,tick_after=25,early_end=True,native_safe=False)
    assert t.phase=='STOPPED' and row['bank']=='FAILURE_DIAGNOSTIC'
    assert row['execution']['integrations']==25 and not row['successful_BC_demo_admitted']


def test_early_end_before_any_dispatch_does_not_install_action():
    t=transaction();observe(t,'obs-1',0);t.early_end('ORACLE_UNRECOVERABLE')
    assert t.records[0]['execution'] is None and t.phase=='STOPPED'


def test_altered_prefix_is_not_accepted_as_original_command():
    t=transaction();observe(t,'obs-1',0);choose(t);intent=t.dispatch_intent()
    changed=copy.deepcopy(intent['prefix']);changed[0][0]+=.01
    with pytest.raises(ValueError):ack(t,intent,installed_prefix=changed)
