"""Captured real paired ledgers; no transfer, model or native permission."""
import os,json,base64
from pathlib import Path
from copy import deepcopy
import pytest
from scripts.pi0294.pv1 import mix_model_delivery_worker as w
from scripts.pi0294.generalization import minimal_scope,minimal_package
from scripts.pi0294.pv1.mix_vanda_first import CORE_STORAGE,LOCAL_STORAGE

@pytest.fixture
def captured():
    name=os.environ.get('ARX5_D25_DELIVERY_CAPTURE')
    if not name:pytest.skip('explicit actual D25 once-paid dual ledger receipt required')
    v=json.loads(Path(name).read_bytes());assert v['status']=='INDEPENDENT_ACTUAL_D25_ORIGINAL_MODEL_CONTROL_DUAL_PREPAID_ONCE_READBACK'
    return [json.loads(base64.b64decode(v[k]['ledger_base64']))for k in('vanda','core')]

def test_current_actual_package_uses_exact_adopted_caps_and_all14_dimensions(captured):
    g,l=captured;before=deepcopy(captured);p=w.full_remaining(g,l,Core_storage_upper_bytes=CORE_STORAGE,LOCAL_storage_upper_bytes=LOCAL_STORAGE)
    baseline=minimal_scope.first512_forecast(w.a.raw(g),w.a.raw(l))
    assert p['shortfalls']=={}and len(p['dimensions'])==14 and before==captured
    assert p['dimensions']['PBS_submissions']['cap']==39
    assert p['dimensions']['Hopper_storage_bytes']['cap']==1399*1024**3
    assert {k:r['projected']for k,r in p['dimensions'].items()}=={k:r['projected']for k,r in baseline['dimensions'].items()}
    assert p['native_model_or_physics_admission']is False and p['new_reservation']=={}

@pytest.mark.parametrize('field',['unknown_effects','pending_updates'])
def test_unknown_or_pending_is_not_bypassed_by_new_authority(captured,field):
    g,l=captured;g[field]=['PRESERVED_NOT_CLEARED']
    with pytest.raises((ValueError,PermissionError,TypeError)):w.full_remaining(g,l,Core_storage_upper_bytes=CORE_STORAGE,LOCAL_storage_upper_bytes=LOCAL_STORAGE)

def test_full_old_storage_cannot_be_credited_back(captured):
    with pytest.raises(PermissionError):w.full_remaining(*captured,Core_storage_upper_bytes=CORE_STORAGE-1,LOCAL_storage_upper_bytes=LOCAL_STORAGE)
    p=w.full_remaining(*captured,Core_storage_upper_bytes=1024**5,LOCAL_storage_upper_bytes=LOCAL_STORAGE)
    assert p['shortfalls']['Core_storage_bytes']>0

def test_old_delivery_path_keeps_original_forecast(monkeypatch):
    result=object();calls=[]
    monkeypatch.setattr(w.scope,'recompute',lambda *args,**kwargs:calls.append((args,kwargs))or result)
    assert w.full_remaining({}, {},Core_storage_upper_bytes=17,LOCAL_storage_upper_bytes=19)is result
    assert calls==[(({},{}),dict(Core_storage_upper_bytes=17,local_storage_upper_bytes=19))]
