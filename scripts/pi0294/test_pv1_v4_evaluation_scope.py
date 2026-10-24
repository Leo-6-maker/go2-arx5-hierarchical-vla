"""Authorization isolation and original-panel dispatch; CPU only."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json
import pytest
from scripts.pi0294.pv1 import evaluation_scope as s
from scripts.pi0294.pv1 import resources,evaluation_admission as a,neural
from scripts.pi0294.pv1.evaluation_plan import development,remaining_panel
from scripts.pi0294.test_pv1_v4_evaluation_work import synthetic_population_only,evidence,POP
from scripts.pi0294.pv1.baseline import partial_history
from scripts.pi0294.pv1.evaluation_work import remaining
from scripts.pi0294.pv1 import evaluation_platform as e
from scripts.pi0294.dec1 import b_precuda_retry as prior

GRANT=Path(__file__).resolve().parents[2]/'reports/PI0294/design/parallel-science-20261022/V4_S2_GLOBAL_RESOURCE_AND_SAVED_B4096_REUSE_AUTHORIZATION_20261024.json'


def test_exact_direct_answer_bytes_and_proposal_is_not_grant():
    data=GRANT.read_bytes();g=s.grant(data)
    assert g['human_answer']=='批准整个工作包' and g['native_adoption_at_publication'] is False
    with pytest.raises(PermissionError):s.grant(data+b' ')
    with pytest.raises(PermissionError):s.grant(GRANT.with_name('V4_S2_COMPLETE_RESOURCE_AND_B4096_REUSE_SCOPE_DRAFT_20261024.json').read_bytes())


def test_no_marker_keeps_four_old_caps_and_cannot_mutate_constant():
    old=s.caps({});assert old==s.OLD
    old['H200_s']=999999
    assert s.caps({})['H200_s']==96*3600


def ledger():
    return dict(campaign_run_uuid=s.CAMPAIGN,training_runs={},
        PBS_submissions=[dict(status='NATIVE_TERMINAL',job_id=str(i)) for i in range(25)])


def test_real_validator_never_accepts_marker_on_empty_or_incomplete_history():
    for g in ({},{s.FIELD:s.marker(s.APPROVED,dict(GLOBAL='a'*64,LOCAL='b'*64))}):
        with pytest.raises((PermissionError,KeyError)):s.validate_global(g)


@pytest.mark.parametrize('fault',['grant','release','PBS','GPU','CPU','output','DEC1','training','owner_exit','checkpoint','closure','before','early','cutoff','nan','extra','history','progress'])
def test_tampered_marker_cannot_supply_extra_allowance(fault):
    g=ledger();m=s.marker(s.APPROVED,dict(GLOBAL='a'*64,LOCAL='b'*64));g[s.FIELD]=m
    changes=dict(grant=('grant_sha256','a'*64),release=('authority_release','a'*40),
        DEC1=('DEC1_and_other_caps_unchanged',False),training=('new_training_authorized',True),
        owner_exit=('outer_B_owner_exit',0),checkpoint=('saved_B4096_checkpoint_sha256','a'*64),
        closure=('saved_B4096_closed_proof_sha256','a'*64),before=('before_sha256',{}),
        early=('adopted_epoch',s.APPROVED-1),cutoff=('adopted_epoch',s.STOP_NEW),nan=('adopted_epoch',float('nan')))
    if fault in changes:k,v=changes[fault];m[k]=v
    elif fault in ('PBS','GPU','CPU','output'):
        m['authorized_global_caps']=dict(m['authorized_global_caps']);k=dict(PBS='PBS_submissions',GPU='H200_s',CPU='Hopper_CPU_s',output='Hopper_storage_bytes')[fault];m['authorized_global_caps'][k]+=1
    elif fault=='extra':m['native_GO']=True
    elif fault=='history':g['PBS_submissions'][0]['status']='RUNNING'
    else:g['training_runs']['new_training']=1
    h=s.digest(ledger()['PBS_submissions']);r=s.digest({})
    with patch.object(prior,'validate_global'),patch.object(s,'HISTORY_SHA',h),patch.object(s,'RUNS_SHA',r):
        with pytest.raises(PermissionError):s.validate_global(g)


def test_pair_requires_both_markers_and_original_clock():
    g=ledger();h=s.digest(g['PBS_submissions']);r=s.digest(g['training_runs'])
    with patch.object(prior,'validate_global'),patch.object(s,'HISTORY_SHA',h),patch.object(s,'RUNS_SHA',r):
        g[s.FIELD]=s.marker(s.APPROVED,dict(GLOBAL='a'*64,LOCAL='b'*64))
        l=dict(run_uuid=s.CAMPAIGN,T0_epoch=s.T0,hard_closeout_epoch=s.HARD_CLOSE,PBS_submissions=deepcopy(g['PBS_submissions']))
        with pytest.raises(PermissionError):s.validate_pair(g,l)
        l[s.FIELD]=deepcopy(g[s.FIELD]);assert s.validate_pair(g,l)==g[s.FIELD]
        l['T0_epoch']+=1
        with pytest.raises(PermissionError):s.validate_pair(g,l)


def test_B_reuse_does_not_accept_hash_assertion_or_grant_physics():
    with patch.object(s,'validate_pair'):
        with pytest.raises(PermissionError):s.require_B_reuse({}, {},checkpoint_sha256=s.B_CHECKPOINT,closure_bytes=s.B_CLOSURE)
        with pytest.raises(PermissionError):s.require_B_reuse({}, {},checkpoint_sha256='a'*64,closure_bytes=b'{}')


def test_typed_request_executes_only_one_original_panel_with_valid_negatives_skipped(synthetic_population_only):
    population=json.loads(POP);plan=development(population['groups'],route='COVERAGE_FIRST',candidate='B0_CORR',new_bank_manifest_sha256='a'*64)
    hist=partial_history(POP,candidate='B0_CORR',evidence=evidence(count=14))
    prior_rows=hist['prior_rows'];count=len(remaining_panel(prior_rows,plan['panels']['B0_CORR-DEV-N294101']))
    request=dict(platform_evaluation_version=e.VERSION,stage='B0_DEV',evaluation_candidate='B0_CORR',noise_bank=294101,remaining_conditions=count)
    assert count==10 and neural.evaluation_labels(request,plan,prior_rows)==('B0_CORR-DEV-N294101',)
    assert neural.evaluation_labels({},plan,prior_rows)==tuple(plan['panels'])
    for bad in (dict(request,remaining_conditions=24),dict(request,noise_bank=0),dict(request,stage='P2_PRIMARY')):
        with pytest.raises(PermissionError):neural.evaluation_labels(bad,plan,prior_rows)
