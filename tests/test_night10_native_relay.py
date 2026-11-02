"""Custody/admission failures at the existing Night10/native boundary."""
import hashlib
import json
from pathlib import Path

import pytest
from scripts.pi0294.night10 import native_relay as relay


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_unavailable_producer_is_not_replaced_with_a_qualification_or_zero(tmp_path):
    assert relay.checkpoint(tmp_path/'future-model')is None
    root=tmp_path/'model';put(root/'control/LEASE.json',{'status':'ACTIVE'})
    assert relay.checkpoint(root)is None


def test_completed_updates_require_own_full42_restore_and_same_checkpoint(tmp_path,monkeypatch):
    monkeypatch.setattr(relay,'observe',lambda r:dict(MainPID='0'))
    put(tmp_path/'control/LEASE.json',{'status':'CLOSED'})
    put(tmp_path/'RESULT.json',dict(optimizer_updates=4096,error=None,status='COMPLETE_BOOK5_FINITE_TARGET_LEARNING'))
    cp=tmp_path/'training/checkpoint-4096';put(cp/'CHECKPOINT_COMPLETE.json',{'global_step':4096})
    proof=dict(checkpoint_manifest_sha256=relay.sha(cp/'CHECKPOINT_COMPLETE.json'),optimizer_updates=4096,
        sampler_cursor=131072,optimizer_state_entries=42,model_reconstruction=True,
        scheduler_restored=True,RNG_restored=True,frozen_parent_unchanged=True)
    put(tmp_path/'ACTUAL_DELTA_CHECKPOINT_RESTORE.json',proof)
    assert relay.checkpoint(tmp_path)==cp
    for field,value in [('optimizer_state_entries',7),('sampler_cursor',320),
            ('checkpoint_manifest_sha256','f'*64),('RNG_restored',False)]:
        changed=dict(proof,**{field:value});put(tmp_path/'ACTUAL_DELTA_CHECKPOINT_RESTORE.json',changed)
        with pytest.raises(PermissionError,match='ACTUAL_FULL_CLOSED'):relay.checkpoint(tmp_path)


def test_unknown_counts_and_counts_over_original_limits_never_close(tmp_path):
    m=dict(limits=dict(model_loads=1,optimizer_updates=10,generation_queries=32,physics_integrations=0,RGB_pairs=0))
    result=dict(model_load_attempts=1,optimizer_updates=10,generation_attempts=32,physics_integrations=0)
    assert relay.audit_terminal(tmp_path,m,result)['optimizer_updates']==10
    for field,value in [('physics_integrations',None),('physics_integrations',1),
            ('generation_attempts',33),('model_load_attempts',2),('actual_RGB_pairs',1)]:
        with pytest.raises(PermissionError,match='UNKNOWN_OR_EXCESS'):
            relay.audit_terminal(tmp_path,m,dict(result,**{field:value}))


def test_native_record_changes_are_unknown_even_if_report_says_valid(tmp_path):
    folder=tmp_path/'conditions/BOOK_DEV000';folder.mkdir(parents=True)
    (folder/'ACTUAL_RECORDS.jsonl').write_text('changed actual records\n')
    result=dict(model_load_attempts=1,optimizer_updates=0,model_queries=1,integrations=20,RGB_pairs=1,
        conditions=[dict(id='BOOK_DEV000',raw_sha256='f'*64,valid_model_condition=True)])
    m=dict(limits=dict(model_loads=1,optimizer_updates=0,generation_queries=80,physics_integrations=18500,RGB_pairs=642))
    with pytest.raises(PermissionError,match='ACTUAL_NATIVE_RECORDS'):relay.audit_terminal(tmp_path,m,result)


def test_missing_unclosed_or_unregistered_actor_is_not_a_known_closed_failure(tmp_path):
    assert not relay.effects_closed({})
    c={'_actors':[str(tmp_path/'actor')]};assert not relay.effects_closed(c)
    put(tmp_path/'actor/control/LEASE.json',{'status':'ACTIVE'});assert not relay.effects_closed(c)
    put(tmp_path/'actor/control/LEASE.json',{'status':'CLOSED'});assert relay.effects_closed(c)
    c['_actors'].append(str(tmp_path/'second'));assert not relay.effects_closed(c)


def test_existing_run_identity_never_replayed_or_overwritten(tmp_path):
    root=tmp_path/'actor';root.mkdir();(root/'sentinel').write_text('original')
    with pytest.raises(PermissionError,match='NO_REPLAY'):
        relay.stage(dict(run_parent=str(tmp_path)),'actor','unused',{}, {})
    assert(root/'sentinel').read_text()=='original'and len(list(root.iterdir()))==1


@pytest.mark.parametrize('native_live',[False,True])
def test_owner_availability_uses_closure_not_scientific_success(tmp_path,monkeypatch,native_live):
    root=tmp_path/'new';root.mkdir();(root/'receipts').mkdir()
    previous=tmp_path/'M13';put(previous/'STATE.json',dict(status='BLOCKED',reason='CLOSED_KNOWN_FAILURE'))
    actor=tmp_path/'actor';put(actor/'control/LEASE.json',dict(status='ACTIVE'if native_live else'CLOSED'))
    put(previous/'jobs/D2/ACTUAL_STAGED.private.json',dict(root=str(actor)))
    account=tmp_path/'LOCAL.json';put(account,dict(unknown_effects=[],pending_updates=[],pending_queries=[]))
    clock=[0]
    monkeypatch.setattr(relay.time,'time',lambda:clock[0]);monkeypatch.setattr(relay.time,'sleep',lambda n:clock.__setitem__(0,1000))
    monkeypatch.setattr(relay,'observe',lambda r:dict(MainPID='8'if native_live else'0'))
    monkeypatch.setattr(relay,'checkpoint',lambda p:None)
    config=dict(root=str(root),existing_M13=str(previous),existing_neural_owner_lock=str(tmp_path/'owner.lock'),
        dispatch_deadline_epoch=300,LOCAL=str(account),book_producers={'D2':str(actor)})
    result=relay.monitor(config)
    assert result['effects_closed']
    assert relay.read(root/'NEURAL_SLOT.json')['available']is(not native_live)
    assert relay.read(root/'NIGHT_SUMMARY.json')['current_M13']['status']=='BLOCKED'
