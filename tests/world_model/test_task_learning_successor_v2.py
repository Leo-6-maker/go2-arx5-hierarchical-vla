"""Cross-site decision fixtures never establish a real DEV or trained policy."""
import copy,json
from pathlib import Path
import pytest
from go2arx5_vla.world_model import task_learning_successor_v2 as successor
from go2arx5_vla.world_model import task_learning_endpoint_decision_v2 as decision
from go2arx5_vla.world_model import task_learning_training_profile_v1 as training
from go2arx5_vla.world_model.storage import digest
from scripts.pi0294.e1.journal import digest as object_digest
from tests.world_model.test_task_learning_endpoint_decision_v2 import evidence as scientific_fixture


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def sealed(tmp_path,*,partial=False):
    root=tmp_path/'core-decision';root.mkdir();e=scientific_fixture();e['TEST_results_read']=False
    if partial:e['native_outcomes']=e['native_outcomes'][:2]
    pop=dict(summary=dict(complete=not partial),planned_conditions=80,
        attempted_conditions=len(e['native_outcomes']),unrun_conditions=80-len(e['native_outcomes']),
        outcomes=e['native_outcomes'],groups=[dict(root='/fixture-original-DEV-group')])
    pp=save(root/'NATIVE_POPULATION.private.json',pop)
    e['provenance']=dict(population=decision.descriptor(pp),actual_native_groups=pop['groups'])
    ep=save(root/'SCIENTIFIC_EVIDENCE.private.json',e);cp=save(root/'DECISION_CONTRACT.json',decision.contract());r=decision.evaluate(e)
    doc=dict(schema='TASK_LEARNING_ENDPOINT_DECISION_V1',scientific_evidence=decision.descriptor(ep),
        decision_contract=decision.descriptor(cp),scientific_result=r,action=r['action'],
        allows_execution=r['allows_execution'],next_endpoint=r['next_endpoint'],TEST_results_read=False,
        **{k:e[k]for k in ('generation_id','dataset_sha256','checkpoint_sha256','global_step')})
    dp=save(root/'ENDPOINT_DECISION.private.json',doc)
    response=dict(status='ACTUAL_COMPLETE_DEV_ENDPOINT_SCIENTIFIC_DECISION_SEALED',
        decision=decision.descriptor(dp),scientific_result=r,training_started=False,TEST_opened=False)
    sp=save(tmp_path/'owner-journal.json',dict(run_uuid='fixture-owner',message_id=root.name.upper()+'-SEAL',
        status='COMPLETE',result=response,result_sha256=object_digest(response)))
    return root,sp


def test_decision_transport_preserves_original_evidence_and_only_rebinds_locations(tmp_path):
    root,seal=sealed(tmp_path);bundle=successor.export_bundle(root,seal)
    target=tmp_path/'hopper-import';raw,doc,result=successor.localized_decision(bundle,target)
    for name,data in raw.items():
        p=target/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        original=seal if name=='OWNER_SEAL.private.json'else root/name
        assert p.read_bytes()==original.read_bytes()
    original=json.loads((root/'ENDPOINT_DECISION.private.json').read_bytes())
    assert {k:v for k,v in doc.items()if k not in ('scientific_evidence','decision_contract')}=={
        k:v for k,v in original.items()if k not in ('scientific_evidence','decision_contract')}
    assert doc['scientific_evidence']['sha256']==original['scientific_evidence']['sha256']
    assert result['next_endpoint']==4000 and bundle['model_or_optimizer_bytes']==0


@pytest.mark.parametrize('fault',['partial','open_seal','changed_bytes','missing_member','altered_decision'])
def test_partial_unsealed_or_changed_decisions_never_grant_successor_execution(tmp_path,fault):
    root,seal=sealed(tmp_path,partial=fault=='partial')
    if fault=='open_seal':
        x=json.loads(seal.read_bytes());x['status']='INTENT';save(seal,x)
    if fault in ('partial','open_seal'):
        with pytest.raises(ValueError):successor.export_bundle(root,seal)
        return
    b=successor.export_bundle(root,seal)
    if fault=='changed_bytes':b['files']['SCIENTIFIC_EVIDENCE.private.json']['body']='e30='
    elif fault=='missing_member':b['files'].pop('OWNER_SEAL.private.json')
    else:b['scientific_result']['next_endpoint']=8000
    with pytest.raises(ValueError):successor.unpack(b)


def test_continuation_retains_full_checkpoint_generation_and_original_schedule(tmp_path):
    root,seal=sealed(tmp_path);bundle=successor.export_bundle(root,seal);target=tmp_path/'import'
    raw,doc,_=successor.localized_decision(bundle,target)
    for n,b in raw.items():p=target/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    prior=dict(root=str(tmp_path/'prior'),recipe=training.recipe('a'*64,'b'*64,endpoint=2000),generation_id='generation')
    cp=Path(prior['root'])/'training/checkpoint-2000';save(cp/'CHECKPOINT_COMPLETE.json',dict(global_step=2000,
        cursor=64000,generation_id=prior['generation_id'],recipe=prior['recipe']))
    pin=digest(cp/'CHECKPOINT_COMPLETE.json')
    # Bind the synthetic endpoint evidence and original seal to this synthetic checkpoint.
    e=json.loads((target/'SCIENTIFIC_EVIDENCE.private.json').read_bytes());e['checkpoint_sha256']=pin
    save(target/'SCIENTIFIC_EVIDENCE.private.json',e);doc['scientific_evidence']=decision.descriptor(target/'SCIENTIFIC_EVIDENCE.private.json');doc['checkpoint_sha256']=pin
    dp=save(target/'LOCAL_ENDPOINT_DECISION.private.json',doc)
    producer=dict(status='CHECKPOINT_SEALED',profile=training.PROFILE,global_step=2000,generation_id='generation',
        checkpoint=str(cp),restore=dict(checkpoint_sha256=pin))
    fields=successor.continuation_fields(prior,producer,decision.descriptor(dp))
    assert fields['start_step']==2000 and fields['recipe']['endpoint']==4000
    assert fields['resume']==dict(path=str(cp),sha256=pin)and fields['generation_id']==prior['generation_id']
    assert fields['recipe']['scheduler_horizon']==8000 and fields['recipe']['warmup_updates']==200
    assert all(fields['recipe'][k]==prior['recipe'][k]for k in prior['recipe']if k!='endpoint')
    save(cp/'CHECKPOINT_COMPLETE.json',dict(global_step=2000,cursor=0,generation_id='generation',recipe=prior['recipe']))
    with pytest.raises(ValueError,match='FULL_STATE_PREDECESSOR'):successor.continuation_fields(prior,producer,decision.descriptor(dp))
