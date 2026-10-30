"""Exact approved successor window and real closed costs; no runtime effects."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
from scripts.pi0294.multitask import night_scope as scope,night_source_stage


@pytest.fixture(scope='module')
def actual():
    directory=os.environ.get('NIGHT_SOURCE_PRIVATE_CAPTURE_DIR')
    if not directory:pytest.skip('Explicit actual closed PC custody and direct authority required')
    root=Path(directory)
    parent=(root/'NIGHT_SOURCE30_PREPARATION_ACCOUNT.private.json').read_bytes()
    supplement=(root/'NIGHT_SOURCE30_PC_READONLY_SUPPLEMENT.private.json').read_bytes()
    authority=(root.parent/'minimal-adaptation-20261027/MAINTENANCE30_CONTROL_IO/DIRECT_AUTHORITY.json').read_bytes()
    old=(root.parent.parent/'teacher-progress-s0/reports/PI0294/design/parallel-science-20261022/NIGHT_BOUNDED_AUTONOMY_DIRECT_AUTHORITY_20261030.json').read_bytes()
    return parent,supplement,authority,old


def prepared(actual):
    parent,supplement,authority,_=actual
    return scope.successor_account(parent,supplement,authority,
        destination='/node-local/PI0294/new-night-source-four01',now=scope.DEADLINE+3600)


def test_actual_both_closed_records_are_carried_once_with_original_caps(actual):
    parent,supplement,_,_=actual;v=prepared(actual);p,s=map(json.loads,(parent,supplement))
    assert v['caps']==s['caps']==scope.CAPS
    assert v['charged_upper']==s['charged_upper']
    assert v['charged_upper']['PC_CPU_s']==1040 and p['charged_upper']['PC_CPU_s']==830
    assert v['operations']==s['operations'] and not v['native_adopted']
    assert scope.validate_account(v)is v
    assert json.loads(parent)==p and json.loads(supplement)==s
    for fault in ('refund','history','parent'):
        bad=deepcopy(v)
        if fault=='refund':bad['charged_upper']['PC_CPU_s']-=1
        elif fault=='history':bad['operations'][0]['status']='REFUNDED'
        else:bad['section5_successor']['closed_parent_base64']=bad['section5_successor']['closed_PC_supplement_base64']
        with pytest.raises(PermissionError):scope.validate_account(bad)


def test_old_authority_stays_expired_and_only_exact_approved_new_window_works(actual,tmp_path):
    _,_,authority,old=actual;now=scope.DEADLINE+3600
    with pytest.raises(PermissionError):scope.authority(old,now=now)
    scope.authority(old,now=now,successor_body=authority)
    with pytest.raises(PermissionError):scope.authority(old,now=now,successor_body=authority+b'\n')
    with pytest.raises(PermissionError):scope.authority(old,now=scope.SUCCESSOR_DEADLINE,successor_body=authority)
    (tmp_path/'SOURCE_SUCCESSOR_AUTHORITY.json').write_bytes(authority)
    active=scope.for_root(tmp_path);active.authority(old,now=now)
    assert active.STOP_NEW==scope.SUCCESSOR_STOP and active.DEADLINE==scope.SUCCESSOR_DEADLINE
    assert active.APPROVAL_SHA==scope.APPROVAL_SHA


def test_same_remaining_pool_reserves_before_effect_and_stop_new_is_enforced(actual):
    v=prepared(actual);v.update(native_adopted=True,qualification_sha256='a'*64)
    upper=dict(attempts=1,physics_integrations=50000,RGB_pairs=2000,Core_CPU_s=2800,render_device_s=600,Core_peak_bytes=512*1024**2)
    args=dict(operation='LIBERO_LEFT_FIRST01',kind='ATTEMPT',upper=upper,task='LIBERO_BOOK_LEFT',qualification='a'*64)
    after=scope.reserve(v,now=scope.DEADLINE+3600,**args)
    assert after['charged_upper']['attempts']==1 and after['charged_upper']['PC_CPU_s']==1040
    assert after['charged_upper']['Core_CPU_s']==3220
    with pytest.raises(PermissionError):scope.reserve(after,now=scope.DEADLINE+3600,**args)
    with pytest.raises(PermissionError):scope.reserve(v,now=scope.SUCCESSOR_STOP,**args)
    scope.reserve(after,operation='CLOSE',kind='CLOSEOUT',upper={},now=scope.SUCCESSOR_DEADLINE-1)
    with pytest.raises(PermissionError):scope.reserve(after,operation='CLOSE',kind='CLOSEOUT',upper={},now=scope.SUCCESSOR_DEADLINE)


def test_four_task_storage_bound_and_all_native_programs_compile():
    assert 4*512*1024**2+2*128*1024**2+768*1024**2==scope.FOUR_TASK_CORE_STORAGE
    assert scope.FOUR_TASK_CORE_STORAGE<scope.CAPS['Core_peak_bytes']
    for name in('PROGRAM','READBACK','LAUNCH'):compile(getattr(night_source_stage,name),name,'exec')
