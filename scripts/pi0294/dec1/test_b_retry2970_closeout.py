"""CPU evidence joins, not native settlement or permission to retry."""
from copy import deepcopy
from unittest.mock import patch
import pytest

from . import b_retry2970_closeout as c
from .test_b_retry2970_failure import fixture


def saved():
    expected,value,digest=fixture()
    value['native_proof'].update(observed_epoch=10.,fresh_native_scheduler_check=False)
    value.update(financial_settlement_ready=False,prior_failed_audit_CPU_upper_s=34,
                 CPU_audit_actual_s=20.)
    return expected,value,digest


def test_new_actual_native_read_is_joined_without_changing_audit_or_historical_epoch():
    expected,value,digest=saved();original=deepcopy(value)
    observation=dict(job=deepcopy(value['raw_native']),observed_epoch=100.)
    with patch.object(c.failed.prior.b,'PROBE_SHA',digest):
        result=c.financial_evidence(value,observation,expected,now_epoch=102.)
    assert value==original
    assert result['native_proof']['observed_epoch']==100.
    assert result['historical_native_observed_epoch']==10.
    assert result['financial_settlement_ready'] is True
    assert result['checkpoints']==value['checkpoints']
    assert result['effects']==value['effects']
    assert result['prior_failed_audit_CPU_upper_s']==34


@pytest.mark.parametrize('fault',['stale','future','bool','nan','R','success','other_request',
    'cost_correction','historical_only','claimed_fresh','no_prior_cost','no_actual_cost',
    'missing_optimizer','lost_updates_refund'])
def test_stale_native_or_mutated_state_is_never_a_financial_join(fault):
    expected,value,digest=saved()
    observation=dict(job=deepcopy(value['raw_native']),observed_epoch=100.)
    if fault=='stale':observation['observed_epoch']=69.
    elif fault=='future':observation['observed_epoch']=103.
    elif fault=='bool':observation['observed_epoch']=True
    elif fault=='nan':observation['observed_epoch']=float('nan')
    elif fault=='R':observation['job']['job_state']='R'
    elif fault=='success':observation['job']['Exit_status']=0
    elif fault=='other_request':observation['job']['Variable_List']['PV1_REQUEST_ID']='different'
    elif fault=='cost_correction':observation['job']['resources_used']={'cput':'03:00:00'}
    elif fault=='historical_only':value.pop('financial_settlement_ready')
    elif fault=='claimed_fresh':value['native_proof']['fresh_native_scheduler_check']=True
    elif fault=='no_prior_cost':value.pop('prior_failed_audit_CPU_upper_s')
    elif fault=='no_actual_cost':value.pop('CPU_audit_actual_s')
    elif fault=='missing_optimizer':value['checkpoint_members']['2048'].pop('optimizer.pt')
    else:value['effects']['optimizer_updates_committed']=0
    with patch.object(c.failed.prior.b,'PROBE_SHA',digest),pytest.raises(PermissionError):
        c.financial_evidence(value,observation,expected,now_epoch=102.)


def test_saved_state_only_cannot_enter_original_settlement_arithmetic():
    expected,value,digest=saved()
    with patch.object(c.failed.prior.b,'PROBE_SHA',digest),pytest.raises(PermissionError):
        c.settle_once({}, {}, evidence=value, expected=expected, now_epoch=102.)


@pytest.mark.parametrize('cost',[True,float('nan'),float('inf'),-1,500])
def test_missing_or_unbounded_audit_cost_cannot_be_dropped(cost):
    expected,value,digest=saved();value['CPU_audit_actual_s']=cost
    with patch.object(c.failed.prior.b,'PROBE_SHA',digest),pytest.raises(PermissionError):
        c.financial_evidence(value,dict(job=value['raw_native'],observed_epoch=100.),expected,now_epoch=102.)


def reserved_readonly():
    e,v,digest=saved();e['request']['CPU_upper_s']=54000
    marker=dict(grant_sha256=c.handoff.GRANT_SHA,old_parent_mutations_permitted=False,
        origin_is_fresh_native_read=False,one_mutation_owner=True)
    h=dict(CPU_reserved_s=54000,optimizer_updates=8744,unknown_effects=[],pending_updates=[],
        PBS_submissions=[dict(request_id=c.failed.REQUEST,job_id=c.failed.JOB,status='SUBMITTED_PENDING_NATIVE')])
    d=dict(CPU_reserved_s=54000,optimizer_updates=8744,unknown_effects=[])
    h[c.handoff.FIELD]=deepcopy(marker);d[c.handoff.FIELD]=deepcopy(marker)
    allocation=dict(schema=c.READONLY_CPU_SCOPE,job_id=c.failed.JOB,request_sha256=c.failed.REQUEST_SHA,
        previous_failed_audit_receipt_sha256=c.FAILED_AUDIT_SHA,previous_audit_actor_confirmed_absent=True,
        prior_all_audit_and_admin_CPU_upper_s=480,new_finite_audit_CPU_upper_s=600,
        observer_CPU_upper_s=1500,remaining_closeout_CPU_upper_s=120,native_PBS_CPU_s=8323,
        original_job_CPU_reserved_s=54000,new_CPU_prepayment=0)
    v.update(prior_failed_audit_CPU_upper_s=480,CPU_audit_actual_s=550.)
    v['native_proof']['conservative_CPU_s']=11023
    return e,v,digest,h,d,allocation


def test_failed480_group_stays_spent_separate600_reads_fit_same_54000_job_hold():
    e,v,digest,h,d,a=reserved_readonly();before=deepcopy((v,h,d,a))
    with patch.object(c.failed.prior.b,'PROBE_SHA',digest):
        result=c.financial_evidence(v,dict(job=v['raw_native'],observed_epoch=100.),e,now_epoch=102.,readonly_allocation=a,budget_before=(h,d))
    assert (v,h,d,a)==before
    assert result['readonly_audit_allocation']==a and result['native_proof']['conservative_CPU_s']==11023
    assert result['CPU_audit_actual_s']==550. and result['prior_failed_audit_CPU_upper_s']==480
    assert result['effects']['optimizer_updates_committed']==922


@pytest.mark.parametrize('fault',['missing_budget','released_hold','different_hold','other_job','prior_refund','newlimit',
    'observer_dropped','unknown','new_prepay','bool_prepay','floatlimit','missing_failure','active_actor','source_grant',
    'CPU_bill_dropped','overrun','wrong_request_CPU'])
def test_readonly_reallocation_never_adds_cpu_or_erases_failed_attempts(fault):
    e,v,digest,h,d,a=reserved_readonly();budget=(h,d)
    if fault=='missing_budget':budget=None
    elif fault=='released_hold':h['CPU_reserved_s']=0
    elif fault=='different_hold':d['CPU_reserved_s']=60000
    elif fault=='other_job':h['PBS_submissions'][0]['job_id']='other'
    elif fault=='prior_refund':a['prior_all_audit_and_admin_CPU_upper_s']=0
    elif fault=='newlimit':a['new_finite_audit_CPU_upper_s']=601
    elif fault=='observer_dropped':a['observer_CPU_upper_s']=0
    elif fault=='unknown':h['unknown_effects']=[{}]
    elif fault=='new_prepay':a['new_CPU_prepayment']=1
    elif fault=='bool_prepay':a['new_CPU_prepayment']=False
    elif fault=='floatlimit':a['new_finite_audit_CPU_upper_s']=600.
    elif fault=='missing_failure':a['previous_failed_audit_receipt_sha256']='0'*64
    elif fault=='active_actor':a['previous_audit_actor_confirmed_absent']=False
    elif fault=='source_grant':h[c.handoff.FIELD]['grant_sha256']='0'*64
    elif fault=='CPU_bill_dropped':v['native_proof']['conservative_CPU_s']=10423
    elif fault=='overrun':v['CPU_audit_actual_s']=601.
    else:e['request']['CPU_upper_s']=60000
    with patch.object(c.failed.prior.b,'PROBE_SHA',digest),pytest.raises(PermissionError):
        c.financial_evidence(v,dict(job=v['raw_native'],observed_epoch=100.),e,now_epoch=102.,readonly_allocation=a,budget_before=budget)
