"""Private handoff/decision assembly fixtures; no real candidate is asserted."""
import json,shutil
from pathlib import Path
import pytest
from go2arx5_vla.world_model import task_learning_endpoint_evidence_v2 as evidence
from go2arx5_vla.world_model import task_learning_delivery_handoff_v2 as handoff
from go2arx5_vla.world_model import task_learning_student_population_v2 as population
from go2arx5_vla.world_model import task_learning_endpoint_decision_v2 as decision
from go2arx5_vla.world_model.storage import digest
from tests.world_model.test_task_learning_delivery_handoff_v2 import fixture,write
from tests.world_model.test_task_learning_endpoint_decision_v2 import evidence as scientific_fixture


def delivered(tmp_path):
    source,job=fixture(tmp_path);inventory=handoff.audit_training_root(source,job,job_id='fixture.pbs')
    root=tmp_path/'delivery'
    for name,member in inventory['files'].items():
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(member['source'],target)
    result=dict(status='PASS_PRIVATE_TASK_POLICY_PACKAGE_TRANSFER_V2',exit_code=0,
        package=dict(path=str(root/'package'),sha256=inventory['package_sha256']),
        all_delivered_files_sha256={n:v['sha256']for n,v in inventory['files'].items()},
        **{k:inventory[k]for k in('global_step','generation_id','dataset_sha256','statistics_sha256','checkpoint_sha256')})
    write(root/'RESULT.private.json',result)
    write(root/'control/LEASE.json',dict(status='CLOSED',worker_exit_code=0,result_sha256=digest(root/'RESULT.private.json')))
    return root,result


def test_delivery_uses_actual_producer_spec_recipe_and_rechecks_retained_arrays(tmp_path):
    root,r=delivered(tmp_path)
    # The actual training RESULT has no recipe field; the pinned spec does.
    assert 'recipe'not in json.loads((root/'producer/RESULT.json').read_bytes())
    assert evidence.delivery(root)['checkpoint_sha256']==r['checkpoint_sha256']
    artifact=next((root/'producer/restored').glob('*.npz'));artifact.write_bytes(b'corrupted fixture')
    with pytest.raises(ValueError,match='SEALED_TRANSFERRED_ENDPOINT_EVIDENCE'):evidence.delivery(root)


def test_partial_or_other_candidate_dev_cannot_be_turned_into_a_scientific_decision(tmp_path,monkeypatch):
    root,r=delivered(tmp_path);e=scientific_fixture();p=e['native_panel']
    path=tmp_path/'panel.json';write(path,p);pin=decision.descriptor(path)
    monkeypatch.setattr(population,'aggregate',lambda *a:dict(summary=dict(complete=False)))
    calls=[];monkeypatch.setattr(decision,'probe_summary',lambda *a,**kw:calls.append(kw))
    with pytest.raises(ValueError,match='PARTIAL_DEV_ONLY'):evidence.assemble(root,pin,[])
    assert not calls
    monkeypatch.setattr(population,'aggregate',lambda *a:dict(summary=dict(complete=True),outcomes=e['native_outcomes']))
    with pytest.raises(ValueError,match='ACTUAL_DELIVERED_CANDIDATE'):evidence.assemble(root,pin,[])
    assert not calls


def test_first_endpoint_recomputes_both_splits_from_its_own_zero_step_and_restored_arrays(tmp_path,monkeypatch):
    root,r=delivered(tmp_path);e=scientific_fixture();p=e['native_panel'];p['policies']['B1']['package']=r['package']
    path=tmp_path/'panel.json';write(path,p);pin=decision.descriptor(path)
    pop=dict(summary=dict(complete=True),outcomes=e['native_outcomes'],groups=[])
    monkeypatch.setattr(population,'aggregate',lambda *a:pop)
    calls=[]
    def probe(folder,*,split,global_step,**identity):
        calls.append((Path(folder).name,split,global_step,identity))
        summary=e['open_before'if global_step==0 else'open_after'][split].copy()
        summary.update(identity);return summary
    monkeypatch.setattr(decision,'probe_summary',probe)
    actual,_,result=evidence.assemble(root,pin,[])
    assert {(n,s,t)for n,s,t,_ in calls}=={('before','TRAIN',0),('before-DEV','DEV',0),('restored','TRAIN',2000),('restored-DEV','DEV',2000)}
    assert actual['checkpoint_sha256']==r['checkpoint_sha256']and result['next_endpoint']==4000
    assert actual['provenance']['delivery']==decision.descriptor(root/'RESULT.private.json')
    with pytest.raises(ValueError,match='OWN_STEP0'):
        evidence.assemble(root,pin,[],previous_evidence=dict(path='/never-read',sha256='f'*64))
