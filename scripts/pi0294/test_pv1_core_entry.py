"""Actual private config capture, additive metadata and strict once semantics."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.pi0294.pv1 import evaluation_core_entry as e
from scripts.pi0294.dec1 import source_cpu_once_io as once


@pytest.fixture
def captured():
    folder=Path(__file__).resolve().parents[4]/'work/private/science-next/wdr1-target-identity/v4a-core-eval-preparation-01'
    p=folder/'ACTUAL_CORE_EXISTING_STANDARD_ENTRY_FOR_GPU_PROFILE_1792840426467566678.private.json'
    if not p.is_file():pytest.skip('private exact deployed entry metadata absent')
    return base64.b64decode(json.loads(p.read_bytes())['value']['members']['ENVIRONMENTS.json']['base64'])


def test_adds_one_actual_environment_profile_preserving_every_old_setting(captured):
    result=e.proposal(captured,now_epoch=e.STOP_NEW-100)
    before=json.loads(captured);after=json.loads(result['proposed_bytes'])
    assert before['profiles']==after['profiles'][:-1]
    assert len(after['profiles'])==len(before['profiles'])+1
    assert all(after[k]==v for k,v in before.items() if k!='profiles')
    old=next(r for r in before['profiles'] if r['name']=='4080-cpu-data');new=after['profiles'][-1]
    assert new['python']==old['python'] and new['source']==old['source']
    assert new['native_GPU_allocation_required'] and new['run_manifest_and_active_lease_required']
    assert not new['scientific_GO'] and not result['native_GPU_allocation']
    assert not result['IO_performed'] and not result['entry_shell_or_doctor_modified']
    for data in [captured+b' ',result['proposed_bytes']]:
        with pytest.raises(PermissionError):e.proposal(data,now_epoch=e.STOP_NEW-100)


@pytest.mark.parametrize('now',[True,e.T0-1,e.STOP_NEW,e.HARD_CLOSE,float('nan')])
def test_config_change_or_expired_clock_is_not_new_authority(captured,now):
    with pytest.raises(PermissionError):e.proposal(captured,now_epoch=now)


def test_original_once_IO_cannot_repeat_additive_entry_write(captured,tmp_path):
    value=e.proposal(captured,now_epoch=e.STOP_NEW-100)
    path=tmp_path/'ENVIRONMENTS.json';path.write_bytes(captured)
    intent=tmp_path/'intent';receipt=tmp_path/'receipt';lock=tmp_path/'lock'
    once.commit(path,expected_before_sha256=value['before_sha256'],proposed_bytes=value['proposed_bytes'],
        request_id='CORE-GPU-ENTRY-ONCE',intent_path=intent,receipt_path=receipt,lock_path=lock)
    assert path.read_bytes()==value['proposed_bytes']
    assert once.reconcile(path,intent_path=intent,receipt_path=receipt)['status']=='CONFIRMED_CURRENT_PROPOSED_BYTES_WITH_RECEIPT'
    with pytest.raises(Exception):once.commit(path,expected_before_sha256=value['before_sha256'],
        proposed_bytes=value['proposed_bytes'],request_id='CORE-GPU-ENTRY-ONCE',intent_path=intent,receipt_path=receipt,lock_path=lock)
