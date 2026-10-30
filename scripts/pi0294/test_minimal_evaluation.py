"""CPU contract tests; no runtime, model, scene, outcome or native admission."""
from copy import deepcopy
import json
import hashlib

import pytest
from scripts.pi0294.generalization import minimal_evaluation as e
from scripts.pi0294.pv1 import mix_evaluation as original
from scripts.pi0294.pv1 import evaluation_plan as routing
from scripts.pi0294.pv1.plan import original_group, slots


@pytest.fixture
def groups():
    return [original_group(**slot,candidate=0) for slot in slots()]


def request(plan):
    return dict(platform_evaluation_version=e.VERSION,stage=e.STAGE,
        evaluation_candidate=plan['candidate'],checkpoint_step=512,
        noise_bank=294101,optimizer_updates=0,statistics_sha256=e.R0,
        raw_parent_files_identity=e.RAW_PARENT_FILES_ID,
        checkpoint_manifest_sha256=plan['checkpoint_manifest_sha256'],remaining_conditions=24)


@pytest.mark.parametrize('candidate',e.ARMS)
def test_exact_original_geometry_task_camera_safety_score_and_noise_population(groups,candidate):
    got=e.panel(groups,candidate);old=original.panel(groups,'C50')
    assert got['noise_namespace']==old['noise_namespace']
    assert got['checkpoint_step']==512 and old['checkpoint_step']==1024
    label_fields={'id','policy_candidate','pv1_model_label','pv1_session_label'}
    for new_card,old_card in zip(got['cards'],old['cards']):
        assert {k:v for k,v in new_card.items()if k not in label_fields}=={
            k:v for k,v in old_card.items()if k not in label_fields}
    assert got['complete_task_duration_s']==32 and got['Teacher_takeover'] is False
    assert len(got['cards'])==24


def test_real_service_label_router_accepts512_without_coercing_old1024(groups):
    from scripts.pi0294.pv1.neural import evaluation_labels
    plan=e.candidate_plan(groups,'E1',checkpoint_manifest_sha256='a'*64)
    assert evaluation_labels(request(plan),plan,[])==('E1-DEV-N294101',)
    with pytest.raises(PermissionError):
        evaluation_labels(dict(request(plan),platform_evaluation_version='V5_CD_CORE_MATCHED_DEV24_V1'),plan,[])
    old=original.candidate_plan(groups,'C50')
    assert evaluation_labels(dict(platform_evaluation_version='V5_CD_CORE_MATCHED_DEV24_V1',
        stage='V5_MATCHED_DEV24',evaluation_candidate='C50',checkpoint_step=1024,
        noise_bank=294101,remaining_conditions=24,optimizer_updates=0),old,[])==('C50-DEV-N294101',)


@pytest.mark.parametrize('field,value',[
    ('checkpoint_step',1024),('checkpoint_step',True),('evaluation_candidate','D25'),
    ('noise_bank',294102),('optimizer_updates',1),('optimizer_updates',False),('statistics_sha256','b'*64),
    ('raw_parent_files_identity','CORR2048'),('checkpoint_manifest_sha256','b'*64),
    ('remaining_conditions',1),('stage','V5_MATCHED_DEV24')])
def test_old_or_changed_science_is_not_aliased_to_fixed512(groups,field,value):
    plan=e.candidate_plan(groups,'E1',checkpoint_manifest_sha256='a'*64)
    r=request(plan);r[field]=value
    with pytest.raises(PermissionError):e.labels(r,plan,[])


def test_valid_failure_is_kept_and_safety_failure_blocks_replacement(groups):
    plan=e.candidate_plan(groups,'E1',checkpoint_manifest_sha256='a'*64)
    card=plan['panels']['E1-DEV-N294101']['cards'][0]
    prior=[dict(condition_id=card['condition_id'],model='E1',attempt_id=card['id'],
        pipeline_trustworthy=True,status='VALID_TASK_FAIL',failure_stage='ACTION_GATE')]
    with pytest.raises(PermissionError):e.labels(request(plan),plan,prior)
    assert e.labels(dict(request(plan),remaining_conditions=23),plan,prior)==('E1-DEV-N294101',)
    prior[0].update(status='ACTUAL_SAFETY_FAIL',pipeline_trustworthy=False)
    with pytest.raises(PermissionError):e.labels(dict(request(plan),remaining_conditions=23),plan,prior)


def test_actual_sealed_session_router_regenerates_cards_before_serving(groups,tmp_path,monkeypatch):
    population=(json.dumps(dict(groups=groups))+'\n').encode()
    digest=hashlib.sha256(population).hexdigest()
    # Explicit CPU population fixture, not an actual campaign population seal.
    monkeypatch.setattr(e,'POPULATION',digest)
    plans=tmp_path/'plans';plans.mkdir()
    (tmp_path/e.POPULATION_FILE).write_bytes(population)
    plan=e.candidate_plan(groups,'E1',checkpoint_manifest_sha256='a'*64)
    p=tmp_path/e.PLAN;p.write_text(json.dumps(plan))
    manifest=dict(platform_evaluation_version=e.VERSION,execution=dict(sealed_inputs={
        e.POPULATION_FILE:digest,e.PLAN:hashlib.sha256(p.read_bytes()).hexdigest()}))
    m=tmp_path/'RUN_MANIFEST.json';m.write_text(json.dumps(manifest))
    assert routing.session_spec(tmp_path,'E1-DEV-N294101')==plan['panels']['E1-DEV-N294101']
    # Even a freshly resealed altered card must fail deterministic regeneration.
    changed=deepcopy(plan);changed['panels']['E1-DEV-N294101']['cards'][0]['source_xy'][0]+=.001
    p.write_text(json.dumps(changed));manifest['execution']['sealed_inputs'][e.PLAN]=hashlib.sha256(p.read_bytes()).hexdigest()
    m.write_text(json.dumps(manifest))
    with pytest.raises(PermissionError):routing.session_spec(tmp_path,'E1-DEV-N294101')


def test_planning_contract_never_claims_admission(groups):
    plan=e.candidate_plan(groups,'E2',checkpoint_manifest_sha256='a'*64)
    assert plan['raw_parent_files_identity']==e.RAW_PARENT_FILES_ID
    assert 'parent_checkpoint_manifest_sha256' not in plan
    assert plan['policy_selection']=='NONE'
    assert not any(k in plan for k in ['native_admission','native_handle','model_loaded'])
    with pytest.raises(PermissionError):e.panel(groups,'GEN1')
