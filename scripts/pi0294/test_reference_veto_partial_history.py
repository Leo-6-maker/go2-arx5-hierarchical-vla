"""Strict existing-panel continuation; CPU fixtures are not runtime grants."""
from copy import deepcopy
import base64,hashlib,json,os
from pathlib import Path

import pytest
from scripts.pi0294.pv1 import baseline as b
from scripts.pi0294.generalization import minimal_evaluation as e
from scripts.pi0294.generalization.minimal_core_accounting import CHECKPOINT
from scripts.pi0294.test_pv1_evaluation_plan import groups

raw=lambda v:(json.dumps(v,indent=2)+'\n').encode()
sha=lambda v:hashlib.sha256(v).hexdigest()


def fixture():
    population=raw(dict(groups=groups()))
    plan=e.candidate_plan(groups(),'E1',checkpoint_manifest_sha256=CHECKPOINT)
    card=plan['panels']['E1-DEV-N294101']['cards'][0]
    result=raw(dict(id=card['id'],source_sha='a'*40,physics_steps=0,rgb_pairs=1))
    row={k:card[k]for k in ('condition_id','task','split','scene_group','noise_bank','family')}
    row.update(attempt_id=card['id'],model='E1',status='VALID_TASK_FAIL',pipeline_trustworthy=True,
        failure_stage='ORIGINAL_REFERENCE_GATE_REJECT',live_native_failures=0,main_integrations=0,
        teacher_takeover=False,duration_s=0.,RGB_pairs=1,S1_rejections=0,result_sha256=sha(result))
    audit=dict(attempt_id=row['attempt_id'],status='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT',
        adjudication=row,errors=[],checks={k:True for k in b.REFERENCE_VETO_CHECKS},
        actual_native_safety_flag_rows=0,model_autonomous_evaluation=True,execution_source_sha='a'*40,
        video=dict(sha256='c'*64,initial_frame_only=True),new_integrations=0,new_native_constructors=0,
        new_model_queries=0,physical_integrations=0,reference_veto_audit_revision='ZERO_REFERENCE_VETO_V2',
        original_reference_veto=True,missing_physics_not_synthesized=True,task_result_available=False,score=None)
    ready=dict(status='PV1_NATIVE_MODEL_READY',label=card['pv1_session_label'],job_id='CPU_FIXTURE',run_uuid='CPU_FIXTURE',
        identity=dict(label='E1',global_step=512,checkpoint_manifest_sha256=CHECKPOINT,statistics_sha256=e.R0,
            relay_identity=dict(source_sha='a'*40)),
        native_pre_model_qualification=dict(status='PASS_SAME_JOB_NATIVE_QUALIFICATION_READBACK',
            heartbeat_elapsed_s=600.1,transport_cycles=80,new_model_queries=0,new_physics=0,receipt_sha256='e'*64))
    chain=dict(status='PASS_INDEPENDENT_ACTUAL_TWO_SITE_CAUSAL_CHAIN',attempt_id=row['attempt_id'],
        adjudication_sha256=sha(raw(row)),identity=ready['identity'],new_integrations=0,new_model_queries=0,
        executed_ACKs=0,draws=1,core_proof_sha256='a'*64,hopper_proof_sha256='b'*64)
    return population,dict(adjudication=raw(row),result=result,audit=raw(audit),model_ready=raw(ready),causal_chain=raw(chain))


def parse(population,item):
    return b.partial_history(population,candidate='E1',evidence=[item],checkpoint_manifest_sha256=CHECKPOINT)


def test_veto_stays_valid_and_is_removed_from_original_remaining_panel(monkeypatch):
    population,item=fixture();monkeypatch.setattr(b,'POPULATION',sha(population))
    before=deepcopy(item);value=parse(population,item)
    assert value['valid_conditions']==1 and value['remaining_eligible_conditions']==23
    done=json.loads(item['adjudication'])['condition_id']
    assert all(c['condition_id']!=done for cards in value['remaining_panels'].values()for c in cards)
    assert value['successes']==dict(T1=0,T2=0)and item==before


@pytest.mark.parametrize('field,value',[
    ('reference_veto_audit_revision','OLD'),('original_reference_veto',False),
    ('missing_physics_not_synthesized',False),('task_result_available',True),
    ('physical_integrations',1),('score',{}),('checks',{'no_physics_or_task_score_fabricated':True})])
def test_audit_claim_alone_does_not_replace_zero_step_evidence(monkeypatch,field,value):
    population,item=fixture();monkeypatch.setattr(b,'POPULATION',sha(population))
    audit=json.loads(item['audit']);audit[field]=value;item['audit']=raw(audit)
    with pytest.raises(PermissionError):parse(population,item)


@pytest.mark.parametrize('field,value',[('draws',0),('draws',2),('executed_ACKs',1)])
def test_same_actual_single_draw_and_no_executed_ACK_required(monkeypatch,field,value):
    population,item=fixture();monkeypatch.setattr(b,'POPULATION',sha(population))
    chain=json.loads(item['causal_chain']);chain[field]=value;item['causal_chain']=raw(chain)
    with pytest.raises(PermissionError):parse(population,item)


def test_full_actual_private_reaudit_bytes_keep_one_valid_twenty_three_remaining():
    path=os.environ.get('E1_REFERENCE_VETO_HISTORY_CAPTURE')
    if not path:pytest.skip('Explicit actual private native history capture required')
    packet=json.loads(Path(path).read_bytes());files={k:base64.b64decode(v,validate=True)for k,v in packet['files'].items()}
    population=files.pop('population');value=parse(population,files)
    assert value['valid_conditions']==1 and value['remaining_eligible_conditions']==23
    assert value['all_actual_attempts'][json.loads(files['adjudication'])['attempt_id']]['independent_audit_sha256']==sha(files['audit'])


def test_existing_request_accepts_only_v7_partial_count_and_existing_labels_check_history(monkeypatch):
    from scripts.pi0294.generalization import minimal_core_accounting as c
    from scripts.pi0294.test_minimal_core_v7_successor import v7_fixture
    population,item=fixture();monkeypatch.setattr(b,'POPULATION',sha(population))
    history=parse(population,item);_,request=v7_fixture();request['remaining_conditions']=23
    assert c.request(request)==request
    plan=e.candidate_plan(groups(),'E1',checkpoint_manifest_sha256=CHECKPOINT)
    assert e.labels(request,plan,history['prior_rows'])==('E1-DEV-N294101',)
    for count in [0,25,True]:
        with pytest.raises(PermissionError):c.request(dict(request,remaining_conditions=count))
    for count in [22,24]:
        with pytest.raises(PermissionError):e.labels(dict(request,remaining_conditions=count),plan,history['prior_rows'])
    request.pop('execution_policy_sha256')
    with pytest.raises(PermissionError):c.request(request)
