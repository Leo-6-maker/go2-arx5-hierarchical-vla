"""Real grant and full preserved runway fixtures; no native dispatch."""
from copy import deepcopy
from pathlib import Path
import pytest
from . import b_vanda_recovery2 as r,b_saved_state_reuse as saved,b_system_startup_retry as retry
from .test_b_heartbeat_recovery import scope
from .test_b_system_startup_retry import full_retry
from .test_b_recovery_forecast import forecast
from .test_a40_allocation import raw

GRANT=Path(__file__).resolve().parents[3]/'reports/PI0294/design/parallel-science-20261022/B2970_RECOVERY2_FIVE_CAP_AUTHORIZATION_20261023.json'

def inputs():
    a,p=full_retry();now=r.APPROVED+100;a['now_epoch']=now
    a['vanda2_bytes']=GRANT.read_bytes()
    for v in (a['snapshot'],a['plan'],p):v.update(r.scope())
    a['snapshot']['observed_epoch']=p['observed_epoch']=now
    for v in (a['snapshot']['used'],a['snapshot']['dec1_used']):v['optimizer_updates']=9666
    a['snapshot']['used']['PBS_submissions']=23;a['snapshot']['dec1_used']['PBS_submissions']=12
    p.update(B_recovery1_paid_lost_updates=922,B_recovery1_preserved_step=2970)
    p['candidates']['B_COVERAGE']['training_lineage_key']=r.LINEAGE
    a['plan']['training_jobs'][0].update(r.scope(),training_lineage_key=r.LINEAGE)
    a['progress_bytes']=raw(p)
    return a,p

def test_exact_public_grant_and_only_five_caps_change():
    assert r.grant(GRANT.read_bytes())['narrow_changes']==r.CHANGES
    a,p=inputs();v=forecast(a)
    assert v['admissible_forecast'] and v['effective_DEC1_VAP1_caps']['optimizer_updates']==11714
    assert v['effective_DEC1_VAP1_caps']['PBS_submissions']==16
    assert v['effective_global_caps']['PBS_submissions']==27
    assert v['effective_global_caps']['Hopper_storage_bytes']==680*1024**3
    assert v['original_final_protection']['H200_s']==72000
    assert v['future_required_additional']['PBS_submissions']==4
    assert v['future_required_additional']['optimizer_updates']==2048
    old,_=full_retry();prior=forecast(old)
    for field in ('effective_global_caps','effective_DEC1_VAP1_caps'):
        allowed={'PBS_submissions','Hopper_storage_bytes','optimizer_updates'}
        assert {k:z for k,z in v[field].items() if k not in allowed}=={k:z for k,z in prior[field].items() if k not in allowed}

@pytest.mark.parametrize('fault',['bytes','partial','plan','progress','lost922','old2970','old_lineage','GPU','CPU','deadline'])
def test_no_scope_drop_loss_refund_or_other_cap_extension(fault):
    a,p=inputs()
    if fault=='bytes':a['vanda2_bytes']+=b' '
    elif fault=='partial':a['snapshot'].pop(r.SCOPE_KEYS[1])
    elif fault=='plan':a['plan']['training_jobs'][0].pop(r.SCOPE_KEYS[1])
    elif fault=='progress':p.pop(r.SCOPE_KEYS[1])
    elif fault=='lost922':p['B_recovery1_paid_lost_updates']=0
    elif fault=='old2970':p['B_recovery1_preserved_step']=2048
    elif fault=='old_lineage':p['candidates']['B_COVERAGE']['training_lineage_key']=r.b.LINEAGE
    elif fault=='GPU':a['snapshot']['dec1_used']['H200_s']+=200000
    elif fault=='CPU':a['snapshot']['dec1_used']['local_CPU_s']+=60000
    else:a['now_epoch']=r.b.CUTOFF
    a['progress_bytes']=raw(p)
    try:v=forecast(a)
    except PermissionError:return
    assert not v['admissible_forecast']

def test_new_request_has_all_prior_scopes_and_distinct_lineage():
    row=dict(scope(),**saved.scope(),**retry.scope(),**r.scope(),training_block=dict(arm='B_COVERAGE',stop_after_step=4096),expected_native_start_step=2048,request_id='new-unique-request')
    row['training_lineage_key']=r.LINEAGE
    assert r.request(row) and saved.request(row) and retry.request(row)
    assert r.b.caps(row)==(680*1024**3,614*1024**3)
    for key,value in [('training_lineage_key',r.b.LINEAGE),('request_id',r.REQUEST),('expected_native_start_step',2970),('heartbeat_recovery_release','wrong')]:
        wrong=deepcopy(row);wrong[key]=value
        with pytest.raises(PermissionError):r.request(wrong)
    assert not r.request({})
