"""Captured-ledger proposals and real temp onceIO; no native site mutation."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from scripts.pi0294.generalization import minimal_adoption as a
from scripts.pi0294.dec1 import source_cpu_once_io as once


@pytest.fixture
def inputs():
    d = os.environ.get('ARX5_FIRST512_FORECAST_INPUTS')
    if not d:
        pytest.skip('Actual private captured ledgers required, not native fixtures')
    root = Path(d)
    grant = Path(__file__).resolve().parents[1]/'reports/PI0294/design/parallel-science-20261022/ARX5_MINIMAL_FIRST512_SEVEN_CAP_DIRECT_HUMAN_GRANT_20261027.json'
    return ((root/'FIRST512_GLOBAL_CAPTURED_BYTES.private.json').read_bytes(),
            (root/'FIRST512_HISTORICAL_LOCAL_BYTES.private.json').read_bytes(), grant.read_bytes())


def propose(inputs):
    return a.propose(*inputs, now_epoch=1793092400.)


def test_exact_seven_caps_preserve_every_old_ledger_field_and_reservation(inputs):
    before = json.loads(inputs[0]); p = propose(inputs); out = p['proposed_global']
    assert out['revision'] == before['revision']+1
    assert set(out)-set(before) == {a.FIELD}
    assert all(out[k] == v for k,v in before.items() if k != 'revision')
    assert p['marker']['caps'] == a.CAPS
    assert p['marker']['LOCAL_sync'] == 'PENDING_CANONICAL_4080_NODE_LOCAL_NO_COPY'
    assert p['marker']['actual_LOCAL_written'] is False
    assert p['marker']['package_reservation_state'] == 'NOT_YET_RESERVED_NO_NATIVE_DISPATCH'
    assert p['full_future_proof']['cap_changes_required'] == {}
    assert all(p['full_future_proof']['dimensions'][k]['cap'] == v for k,v in a.CAPS.items())
    assert not p['native_admission'] and p['new_requests'] == p['new_PBS'] == p['new_GPU'] == 0


def test_adoption_is_once_not_a_repeated_budget_reset(inputs):
    p=propose(inputs)
    with pytest.raises(PermissionError, match='ONCE_NO_READOPTION'):
        a.propose(a.raw(p['proposed_global']),*inputs[1:],now_epoch=1793092401.)


@pytest.mark.parametrize('change', ['grant','LOCAL','window','unknown','active'])
def test_wrong_authority_source_window_or_live_effects_refused(inputs,change):
    gb,lb,hb=inputs;now=1793092400.
    if change=='grant':hb+=b' '
    elif change=='LOCAL':lb+=b' '
    elif change=='window':now=1793451149.8133287
    else:
        g=json.loads(gb)
        if change=='unknown':g['unknown_effects']=['CPU_CAPTURE_CHANGE_NOT_A_REAL_NATIVE_EFFECT']
        else:g['PBS_submissions'][-1]['status']='NATIVE_RUNNING_QUALIFICATION'
        gb=a.raw(g)
    with pytest.raises((PermissionError,KeyError,ValueError)):
        a.propose(gb,lb,hb,now_epoch=now)


@pytest.mark.parametrize('field,value', [('native_admission',True),('actual_LOCAL_written',True),
                                      ('deadline_epoch',1793537550.),('caps',{'PBS_submissions':999})])
def test_marker_cannot_grant_GPU_fresh_LOCAL_new_clock_or_arbitrary_cap(inputs,field,value):
    g=propose(inputs)['proposed_global'];g[a.FIELD][field]=value
    with pytest.raises(PermissionError):a.validate(g)


def test_real_original_onceIO_temp_CAS_and_duplicate_refusal(inputs,tmp_path):
    p=propose(inputs);path=tmp_path/'GLOBAL.json';path.write_bytes(inputs[0])
    kwargs=dict(expected_before_sha256=a.sha(inputs[0]),proposed_bytes=a.raw(p['proposed_global']),
        request_id='CPU-FIRST512-SEVEN-CAPS',intent_path=tmp_path/'I.json',receipt_path=tmp_path/'R.json',lock_path=tmp_path/'L')
    once.commit(path,**kwargs)
    assert once.reconcile(path,intent_path=kwargs['intent_path'],receipt_path=kwargs['receipt_path'])['receipt_binding_valid']
    with pytest.raises(FileExistsError):once.commit(path,**kwargs)
    assert path.read_bytes()==kwargs['proposed_bytes']


def test_real_CAS_lost_receipt_keeps_applied_bytes_and_never_replays(inputs,tmp_path,monkeypatch):
    path=tmp_path/'GLOBAL.json';path.write_bytes(inputs[0]);p=propose(inputs)
    kwargs=dict(expected_before_sha256=a.sha(inputs[0]),proposed_bytes=a.raw(p['proposed_global']),
        request_id='CPU-FIRST512-LOST-REPLY',intent_path=tmp_path/'I.json',receipt_path=tmp_path/'R.json',lock_path=tmp_path/'L')
    original=once._exclusive
    def lost(path,body,*args):
        if path==kwargs['receipt_path']:raise OSError('CPU_INJECTED_RECEIPT_LOSS_AFTER_CAS')
        return original(path,body,*args)
    monkeypatch.setattr(once,'_exclusive',lost)
    with pytest.raises(OSError):once.commit(path,**kwargs)
    proof=once.reconcile(path,intent_path=kwargs['intent_path'],receipt_path=kwargs['receipt_path'])
    assert proof['status']=='CURRENT_PROPOSED_BYTES_RECEIPT_UNCONFIRMED'
    assert not proof['automatic_retry_allowed']
    with pytest.raises(FileExistsError):once.commit(path,**kwargs)
