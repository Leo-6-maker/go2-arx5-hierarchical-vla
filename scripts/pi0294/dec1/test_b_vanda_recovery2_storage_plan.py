"""Cold storage lifecycle regression; no IO/dispatch or native evidence."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from . import b_vanda_recovery2_storage_plan as c
from .test_b_vanda_recovery2 import inputs
from .test_b_recovery_forecast import forecast

ROOT=Path(__file__).resolve().parents[3]
DIR=ROOT/'reports/PI0294/design/parallel-science-20261022'
GRANT=(DIR/'B2970_RECOVERY2_FIVE_CAP_AUTHORIZATION_20261023.json').read_bytes()
DRAFT=(DIR/'B2970_VANDA_ACTUAL_SETTLED_GPU_CLOSED_AND_RECOVERY2_SCOPE_DRAFT_20261023.json').read_bytes()

def pair():
    h=dict(unknown_effects=[],pending_updates=[],reserved={},CPU_reserved_s=0,storage_reserved_bytes=0,
        PBS_submissions=[],new_storage_current_bytes=607644695364)
    d=dict(unknown_effects=[],new_GPU_site_output_allocated_upper_bytes=579316935492)
    return h,d

def prepare(plan,h,d):return c.cold_plan(plan,h,d,grant_bytes=GRANT,draft_bytes=DRAFT)

def test_correct_full_peak_adds_actual_closed_costs_once_without_changing_other_work(monkeypatch):
    monkeypatch.setattr(c.r,'validate_adopted',lambda _h,_d:None)
    a,_=inputs();old=deepcopy(a['plan']);h,d=pair();new=prepare(old,h,d)
    assert new['storage_lifecycle']['new_output_peak_upper_bytes']['Hopper_storage_bytes']==658759907328
    assert prepare(new,h,d)==new and a['plan']==old
    repaired=deepcopy(new);repaired['storage_lifecycle']=old['storage_lifecycle']
    assert repaired==old

@pytest.mark.parametrize('fault',['draft','scope','global_hold','CPU_hold','pending','unknown','unclosed','rollback','cap'])
def test_no_history_refund_live_owner_or_silent_new_cap(monkeypatch,fault):
    monkeypatch.setattr(c.r,'validate_adopted',lambda _h,_d:None)
    a,_=inputs();p=a['plan'];h,d=pair();draft=DRAFT
    if fault=='draft':draft+=b' '
    elif fault=='scope':p.pop(c.r.SCOPE_KEYS[1])
    elif fault=='global_hold':h['reserved']['H200_s']=1
    elif fault=='CPU_hold':d['CPU_reserved_s']=1
    elif fault=='pending':h['pending_updates']=[{}]
    elif fault=='unknown':d['unknown_effects']=[{}]
    elif fault=='unclosed':h['PBS_submissions']=[dict(status='SUBMITTED_PENDING_NATIVE')]
    elif fault=='rollback':d['new_GPU_site_output_allocated_upper_bytes']=0
    else:d['new_GPU_site_output_allocated_upper_bytes']=614*1024**3
    with pytest.raises(PermissionError):c.cold_plan(p,h,d,grant_bytes=GRANT,draft_bytes=draft)

@pytest.mark.parametrize('reserved',[False,True])
def test_same_complete_forecast_before_qsub_and_after_actual_reservation(monkeypatch,reserved):
    monkeypatch.setattr(c.r,'validate_adopted',lambda _h,_d:None)
    a,_=inputs();s=a['snapshot'];h,d=pair()
    for vector,amount in ((s['used'],h['new_storage_current_bytes']),(s['dec1_used'],d['new_GPU_site_output_allocated_upper_bytes'])):
        vector['Hopper_storage_bytes']=amount
    s['storage_peaks']['Hopper_storage_bytes']=s['used']['Hopper_storage_bytes']
    s['dec1_storage_peaks']['Hopper_storage_bytes']=s['dec1_used']['Hopper_storage_bytes']
    if reserved:
        block=a['plan']['training_jobs'][0]
        held={k:v for k,v in block['resources'].items() if k in ('H200_s','Hopper_CPU_s','optimizer_updates','neural_queries','Hopper_storage_bytes')}
        s['reserved']=held;s['dec1_reserved']=deepcopy(held)
        s['used']['PBS_submissions']+=1;s['dec1_used']['PBS_submissions']+=1
        s['work_reservations']=[dict(work_id=block['work_id'],job_id='CPU_ONLY_FIXTURE',kind='TRAIN',submission_already_charged=True,resources=deepcopy(held))]
        s['current_native_request_work_id']=block['work_id']
    with pytest.raises(PermissionError,match='RECOVERY2_PREQUEUE_PEAK_COVERS_ALL_REMAINING_OUTPUT_WITHOUT_RETIREMENT'):
        forecast(a)
    a['plan']=prepare(a['plan'],h,d)
    result=forecast(a)
    assert result['admissible_forecast'] and result['future_required_additional']['optimizer_updates']==(0 if reserved else 2048)
    assert result['original_final_protection']['H200_s']==72000
