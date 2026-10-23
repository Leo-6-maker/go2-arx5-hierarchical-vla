"""Exact new authorization and full preserved runway; no native effects."""
from copy import deepcopy
from pathlib import Path
import pytest
from . import b_precuda_retry as r
from .test_b_vanda_recovery2 import inputs as prior_inputs
from .test_b_recovery_forecast import forecast
from .test_a40_allocation import raw

GRANT=Path(__file__).resolve().parents[3]/'reports/PI0294/design/parallel-science-20261022/B_PRECUDA1481489_RETRY_PBS_AUTHORIZATION_20261024.json'

def inputs():
    a,p=prior_inputs();now=r.APPROVED+100
    a.update(now_epoch=now,precuda_bytes=GRANT.read_bytes())
    for v in (a['snapshot'],a['plan'],p):v.update(r.scope())
    for v in (a['snapshot'],p):v['observed_epoch']=now
    a['snapshot']['used']['PBS_submissions']=24
    a['snapshot']['dec1_used']['PBS_submissions']=13
    a['plan']['training_jobs'][0].update(r.scope())
    a['progress_bytes']=raw(p)
    return a,p

def test_only_two_pbs_caps_change_and_full_original_work_is_retained():
    a,p=inputs();prior=deepcopy(a);prior.pop('precuda_bytes')
    for v in (prior['snapshot'],prior['plan'],p,prior['plan']['training_jobs'][0]):
        for k in r.SCOPE_KEYS:v.pop(k,None)
    prior['progress_bytes']=raw(p)
    old=forecast(prior);new=forecast(a)
    assert old['global_shortfalls']==old['DEC1_shortfalls']=={'PBS_submissions':1}
    assert new['admissible_forecast']
    for field,cap in [('effective_global_caps',28),('effective_DEC1_VAP1_caps',17)]:
        assert new[field]['PBS_submissions']==cap
        assert {k:v for k,v in new[field].items() if k!='PBS_submissions'}=={k:v for k,v in old[field].items() if k!='PBS_submissions'}
    assert old['future_required_additional']==new['future_required_additional']
    assert new['future_required_additional']['PBS_submissions']==4
    assert new['future_required_additional']['optimizer_updates']==2048
    assert old['original_final_protection']==new['original_final_protection']
    assert new['original_final_protection']['H200_s']==72000

@pytest.mark.parametrize('fault',['bytes','snapshot','plan','progress','block','partial','old_lost922','old_lost805','deadline','GPU','CPU','PBS'])
def test_no_scope_drop_replay_or_extra_budget(fault):
    a,p=inputs()
    if fault=='bytes':a['precuda_bytes']+=b' '
    elif fault in ('snapshot','plan'):a[fault].pop(r.SCOPE_KEYS[1])
    elif fault=='progress':p.pop(r.SCOPE_KEYS[1])
    elif fault=='block':a['plan']['training_jobs'][0].pop(r.SCOPE_KEYS[0])
    elif fault=='partial':a['snapshot'][r.SCOPE_KEYS[0]]='wrong'
    elif fault=='old_lost922':p['B_recovery1_paid_lost_updates']=0
    elif fault=='old_lost805':p['B_failed_paid_lost_updates']=0
    elif fault=='deadline':a['now_epoch']=r.prior.b.CUTOFF
    elif fault=='GPU':a['snapshot']['dec1_used']['H200_s']+=200000
    elif fault=='CPU':a['snapshot']['dec1_used']['local_CPU_s']+=60000
    else:a['snapshot']['dec1_used']['PBS_submissions']+=1
    a['progress_bytes']=raw(p)
    try:result=forecast(a)
    except PermissionError:return
    assert not result['admissible_forecast']

def test_old_scope_keeps_old_caps_and_new_request_keeps_same_lineage():
    a,_=prior_inputs();v=forecast(a)
    assert v['effective_global_caps']['PBS_submissions']==27
    assert v['effective_DEC1_VAP1_caps']['PBS_submissions']==16
    from .test_b_heartbeat_recovery import scope
    from . import b_saved_state_reuse as saved,b_system_startup_retry as retry
    row=dict(scope(),**saved.scope(),**retry.scope(),**r.prior.scope(),**r.scope(),
        training_block=dict(arm='B_COVERAGE',stop_after_step=4096),expected_native_start_step=2048,
        training_lineage_key=r.prior.LINEAGE,request_id='new-unique-request')
    assert r.request(row)
    for key,value in [('request_id',r.REQUEST),('training_lineage_key',r.prior.b.LINEAGE),('expected_native_start_step',2970)]:
        changed=deepcopy(row);changed[key]=value
        with pytest.raises(PermissionError):r.request(changed)
    assert not r.request({})
