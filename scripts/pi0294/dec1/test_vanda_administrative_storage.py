from copy import deepcopy
import pytest
from . import vanda_administrative_storage as c
from .test_vanda_control_handoff import fixture,prepare

def inputs():
    h,d,kw=fixture();h,d,_=prepare(h,d,kw)
    h.update(new_storage_current_bytes=537502089028,new_storage_peak_bytes=537502089028)
    d.update(new_GPU_site_output_allocated_upper_bytes=509174329156,GPU_site_output_reserved_bytes=70000000000)
    r=dict(status='INDEPENDENT_MIGRATED_AUTHORITY_LEDGER_AND_STORAGE_BYTES',metadata_allocated_upper_bytes=5885952,
        combined_upper_bytes=70881280,fits_prepaid_output=False)
    return h,d,r

def test_full_gap_and_future_audit_charged_once_without_changing_science_or_caps():
    h,d,r=inputs();old=deepcopy((h,d,r));nh,nd,receipt=c.prepare(h,d,readback=r,now_epoch=1792762000.)
    assert (h,d,r)==old
    assert nh['new_storage_current_bytes']==h['new_storage_current_bytes']+8*1024**2
    assert nd['new_GPU_site_output_allocated_upper_bytes']==d['new_GPU_site_output_allocated_upper_bytes']+8*1024**2
    assert {k:v for k,v in nh.items() if k not in (c.FIELD,'new_storage_current_bytes','new_storage_peak_bytes')}=={k:v for k,v in h.items() if k not in ('new_storage_current_bytes','new_storage_peak_bytes')}
    assert {k:v for k,v in nd.items() if k not in (c.FIELD,'new_GPU_site_output_allocated_upper_bytes')}=={k:v for k,v in d.items() if k!='new_GPU_site_output_allocated_upper_bytes'}
    assert receipt['cap_or_clock_increase'] is False
    with pytest.raises(PermissionError):c.prepare(nh,nd,readback=r,now_epoch=1792762001.)

@pytest.mark.parametrize('fault',['clock','boolclock','nan','replay','unknown','dropreserve','refund',
    'unfenced','Hopperfresh','receipt','changedsize','boolsize','claimedfit','globalcap','DEC1cap','negative','float'])
def test_missing_native_evidence_or_room_cannot_hide_or_charge_a_second_cost(fault):
    h,d,r=inputs();now=1792762000.
    if fault=='clock':now=1792952996
    elif fault=='boolclock':now=True
    elif fault=='nan':now=float('nan')
    elif fault=='replay':d[c.FIELD]={}
    elif fault=='unknown':h['unknown_effects']=[{}]
    elif fault=='dropreserve':h['reserved']['H200_s']=0
    elif fault=='refund':h['optimizer_updates']=7822
    elif fault=='unfenced':h[c.handoff.FIELD]['old_parent_mutations_permitted']=True
    elif fault=='Hopperfresh':h[c.handoff.FIELD]['origin_is_fresh_native_read']=True
    elif fault=='receipt':r['status']='TEMP_FIXTURE'
    elif fault=='changedsize':r['metadata_allocated_upper_bytes']+=1
    elif fault=='boolsize':r['metadata_allocated_upper_bytes']=True
    elif fault=='claimedfit':r['fits_prepaid_output']=True
    elif fault=='globalcap':h['new_storage_current_bytes']=h['new_storage_peak_bytes']=615*1024**3
    elif fault=='DEC1cap':d['new_GPU_site_output_allocated_upper_bytes']=549*1024**3
    elif fault=='negative':h['new_storage_current_bytes']=-1
    else:d['new_GPU_site_output_allocated_upper_bytes']=1.
    with pytest.raises(PermissionError):c.prepare(h,d,readback=r,now_epoch=now)
