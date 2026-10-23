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
