"""Once-paid CPU arithmetic only, synthetic ledgers; no writes or dispatch."""
from copy import deepcopy
import json
from unittest.mock import patch
import pytest
from scripts.pi0294.pv1 import evaluation_control as c,evaluation_scope as s,evaluation_admission as a
from scripts.pi0294.test_pv1_a40_evaluation_identity import case
from scripts.pi0294.test_pv1_v4_evaluation_admission import inputs,rebuild_snapshot,raw
from scripts.pi0294.test_pv1_v4_evaluation_work import synthetic_population_only

PID='22345678-1234-4234-8234-123456789abc'


@pytest.fixture
def paid(case):
    i=inputs(case);r=json.loads(i['request_bytes']);r['control_prepayment_id']=PID;i['request_bytes']=raw(r)
    with patch.object(s,'validate_pair'),patch.object(s,'APPROVED',s.T0),patch.object(c,'BASE',dict.fromkeys(c.CPU_KEYS,0)):
        before=a.recompute(**i,purpose='PREPARE')
        p=c.prepay_proposal(i['global_bytes'],i['local_bytes'],recompute_inputs=i,payment_id=PID,now_epoch=i['now_epoch'])
        yield i,p,before


def after(i,p):
    i=deepcopy(i);i['global_bytes']=raw(p['proposed_global']);i['local_bytes']=raw(p['proposed_LOCAL']);rebuild_snapshot(i);return i


def test_each_site_is_debited_and_exact_current_control_credit_is_once(paid):
    i,p,before=paid;original=deepcopy(i);new=after(i,p);proof=a.recompute(**new,purpose='PREPARE')
    assert i==original and p['IO_performed'] is False and p['native_dispatch_admitted'] is False
    assert p['proposed_global']['CPU_s']==7800
    assert p['proposed_LOCAL']['CPU_s']==dict(CoreLab=7800,Hopper=7800,local_PC=7800)
    assert proof['control_prepayment_credit_applied'] is True
    for key in c.CPU_KEYS:
        assert proof['dimensions'][key]['projected']==before['dimensions'][key]['projected']
        assert proof['current_credit'][key]==7800
    assert p['proposed_global']['used']==json.loads(i['global_bytes'])['used']
    assert p['proposed_global']['PBS_submissions']==[]


def test_repeat_is_rejected_even_if_no_registration_or_GPU(paid):
    i,p,_=paid;new=after(i,p)
    with pytest.raises(PermissionError):c.prepay_proposal(new['global_bytes'],new['local_bytes'],recompute_inputs=new,payment_id=PID,now_epoch=new['now_epoch'])


@pytest.mark.parametrize('fault',['missing_LOCAL','changed_receipt','missing_Core_debit','missing_LOCAL_debit','missing_site_debit','other_request','other_source','other_panel'])
def test_credit_requires_exact_actual_dual_index_and_all_paid_counters(paid,fault):
    i,p,_=paid;new=after(i,p);g=json.loads(new['global_bytes']);l=json.loads(new['local_bytes']);r=json.loads(new['request_bytes'])
    if fault=='missing_LOCAL':l.pop(c.FIELD)
    elif fault=='changed_receipt':l[c.FIELD][PID]['control_CPU_upper_s']+=1
    elif fault=='missing_Core_debit':l['CPU_s']['CoreLab']-=1
    elif fault=='missing_LOCAL_debit':l['CPU_s']['local_PC']-=1
    elif fault=='missing_site_debit':g['CPU_s']-=1
    elif fault=='other_request':r['request_id']='32345678-1234-4234-8234-123456789abc'
    elif fault=='other_source':r['source_sha']='b'*40
    else:r['noise_bank']=294101
    new.update(global_bytes=raw(g),local_bytes=raw(l),request_bytes=raw(r));rebuild_snapshot(new)
    with pytest.raises(PermissionError):a.recompute(**new,purpose='PREPARE')


def test_past_other_control_stays_paid_but_does_not_finance_new_control(paid):
    i,p,_=paid;new=after(i,p);r=json.loads(new['request_bytes']);r.pop('control_prepayment_id');new['request_bytes']=raw(r)
    proof=a.recompute(**new,purpose='PREPARE')
    assert all(proof['current_credit'][key]==600 for key in c.CPU_KEYS)
    assert all(proof['dimensions'][key]['used']==7800 for key in c.CPU_KEYS)


def test_second_payment_does_not_pay_shared_preparation_twice(paid):
    i,p,_=paid;new=after(i,p);r=json.loads(new['request_bytes']);r.update(request_id='32345678-1234-4234-8234-123456789abc',control_prepayment_id='42345678-1234-4234-8234-123456789abc');new['request_bytes']=raw(r)
    q=c.prepay_proposal(new['global_bytes'],new['local_bytes'],recompute_inputs=new,payment_id=r['control_prepayment_id'],now_epoch=new['now_epoch'])
    assert q['payment']['shared_preparation_CPU_upper_s']==0 and q['proposed_global']['CPU_s']==15000
    new=after(new,q);proof=a.recompute(**new,purpose='PREPARE')
    assert all(proof['current_credit'][key]==7800 for key in c.CPU_KEYS)
