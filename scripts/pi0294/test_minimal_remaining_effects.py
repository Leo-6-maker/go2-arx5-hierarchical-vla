"""Actual saved-prefix bytes + real temp original effects/query transactions.

New request admission is explicitly a CPU fixture, not a registered request.
No GR00T, optimizer, CUDA, PBS, host or physics operation occurs here.
"""
import json,uuid
from copy import deepcopy
from types import SimpleNamespace
import pytest
from scripts.pi0294.test_minimal_remaining_contract import actual_prefix
from scripts.pi0294.test_minimal_effects import draw,finish
from scripts.pi0294.test_pv1_recipe import old_recipe
from scripts.pi0294.generalization import minimal_remaining_contract as r,minimal_training as t,minimal_execution as e
from scripts.pi0294.generalization import minimal_execution_adapter as adapters
from scripts.pi0294.generalization.minimal_remaining_effects import RemainingEffects,SCHEMA,SUMMARY
from scripts.pi0294.e1.journal import MutationOutcomeUnknown
from scripts.pi0292.durable import read,write

@pytest.fixture
def remaining(actual_prefix,tmp_path,monkeypatch):
    request=dict(request_id=str(uuid.uuid4()),submission_nonce=str(uuid.uuid4()),source_sha='a'*40,
        platform_training_version=r.VERSION,profile=r.PROFILE,optimizer_update_limit=1026,query_limit=76,
        model_load_limit=7,CPU_upper_s=165600,CPU_FIXTURE_NOT_REGISTERED=True)
    rp=tmp_path/'pbs/alloc-39/REQUEST.json';rp.parent.mkdir(parents=True);rp.write_bytes(e.raw(request));rp.chmod(0o400)
    inputs={}
    for name,body in zip((r.PREFIX,r.CLOSURE),actual_prefix):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body);path.chmod(0o400);inputs[name]=e.sha(body)
    (tmp_path/'RUN_MANIFEST.json').write_bytes(e.raw(dict(execution=dict(sealed_inputs=inputs))))
    write(tmp_path/'plans/FROZEN_SOURCE_PAIR_RECIPE.json',old_recipe())
    # Central adapter selection is deliberately not implemented by this unit.
    # Native dispatch continues to reject the unregistered new type.
    original=adapters.request
    monkeypatch.setattr(adapters,'request',lambda value,**kw:SimpleNamespace(VERSION=r.VERSION)if value.get('CPU_FIXTURE_NOT_REGISTERED')else original(value,**kw))
    allocation=dict(job_id='123456.stdct-mgmt-02',request_id=request['request_id'],request_sha256=e.sha(rp.read_bytes()),source_sha=request['source_sha'])
    ledger=RemainingEffects(tmp_path,allocation,request_path=rp,guard=lambda:None);ledger.initialize()
    return ledger,original

def test_saved_prefix_remains_metadata_not_new_effects_or_query_links(remaining):
    ledger,original=remaining;v=ledger.guard();summary=ledger.summary()
    assert v['schema']==SCHEMA and summary['schema']==SUMMARY
    assert (v['optimizer_updates'],v['generation_queries'],v['model_loads'])==(0,0,0)and v['query_intent_links']==[]
    assert len(v['sessions'])==3 and summary['retained_old_native']==r.OLD_PAID
    assert summary['carried_saved_E2_zero_queries']==15 and not summary['all_remaining_sessions_runtime_complete']
    assert not summary['scientific_endpoints_independently_audited']
    with pytest.raises(PermissionError):original(ledger.request)

def test_E2_own_pending_commit_then_fresh_formal_zero_uses_original_transactions(remaining):
    ledger,_=remaining;sid=r.sessions()[0]['id'];ledger.start_session(sid);ledger.charge('model_loads',1)
    recipe=t.recipe(old_recipe(),t.ARMS[1],raw_parent_manifest_sha256=t.RAW_PARENT_FILES_ID)
    for step in range(2):
        ledger.begin_update(sid,step,recipe)
        pending=ledger.guard();assert pending['generation_queries']==0 and len(pending['pending_updates'])==1
        with pytest.raises(PermissionError):ledger.begin_update(sid,step,recipe)
        ledger.commit_update(sid,step+1)
    for i in range(2):draw(ledger,'REMAINDER_CPU_RESTORE_'+str(i))
    ledger.charge('model_loads',1);ledger.charge('model_loads',1);finish(ledger,sid)
    summary=ledger.summary();assert (summary['optimizer_updates_committed'],summary['query_attempts'],summary['model_load_attempts'])==(2,2,3)
    assert ledger.logical_session_counters(sid)==dict(optimizer_updates=2,generation_queries=17,model_loads=3)
    formal=r.sessions()[1]['id'];ledger.start_session(formal)
    assert ledger.known_effects(formal)['optimizer_updates']==0
    assert ledger.guard()['sessions'][formal]['generation_queries']==0

def test_own_unknown_query_stays_unknown_and_cannot_be_credited_as_carried(remaining):
    ledger,_=remaining;sid=r.sessions()[0]['id'];ledger.start_session(sid);ledger.charge('model_loads',1)
    with pytest.raises(RuntimeError):draw(ledger,'REMAINDER_CPU_LOST',fail=True)
    before=ledger.path.read_bytes();assert ledger.summary()['unknown_query_intents']==[0]
    with pytest.raises(MutationOutcomeUnknown):draw(ledger,'REMAINDER_CPU_LOST')
    with pytest.raises(PermissionError):ledger.charge('model_loads',1)
    recipe=t.recipe(old_recipe(),t.ARMS[1],raw_parent_manifest_sha256=t.RAW_PARENT_FILES_ID)
    with pytest.raises(PermissionError):ledger.begin_update(sid,0,recipe)
    assert ledger.path.read_bytes()==before

@pytest.mark.parametrize('kind',['old_E0','old_E1','prefix_rewrite','carried_as_new','incomplete_loads'])
def test_replay_prefix_tampering_and_fake_completion_are_refused(remaining,kind):
    ledger,_=remaining;v=ledger.guard();before=deepcopy(v);sid=r.sessions()[0]['id']
    if kind=='incomplete_loads':
        ledger.start_session(sid);ledger.charge('model_loads',1)
        recipe=t.recipe(old_recipe(),t.ARMS[1],raw_parent_manifest_sha256=t.RAW_PARENT_FILES_ID)
        for step in range(2):ledger.begin_update(sid,step,recipe);ledger.commit_update(sid,step+1)
        for i in range(2):draw(ledger,'MISSING_RELOAD_CPU_'+str(i))
        before=ledger.path.read_bytes()
        with pytest.raises(PermissionError,match='ALL_REAL_RAW_RESTORE_AND_EXPORT_LOADS'):finish(ledger,sid)
        assert ledger.path.read_bytes()==before
        return
    with pytest.raises(PermissionError):
        if kind=='old_E0':ledger.start_session(e.native_sessions()[0]['id'])
        elif kind=='old_E1':ledger.start_session(e.native_sessions()[1]['id'])
        elif kind=='prefix_rewrite':
            v['audited_saved_prefix']['old_paid_effects']['generation_queries']-=1;ledger.check(v)
        elif kind=='carried_as_new':
            v['sessions'][sid].update(status='ACTIVE',generation_queries=15);v.update(active_session=sid,generation_queries=15);ledger.check(v)
    assert read(ledger.path)==before
