"""New successor arithmetic/scope only; no GPU or native-admission claims."""
from copy import deepcopy
import math
import pytest
from scripts.pi0294.pv1 import successor_cd as s


def test_single_window_and_frozen_order():
    a=s.scope(1792920000)
    assert a['deadline_epoch']==1792920000+168*3600
    assert a['stop_new_training_epoch']==a['deadline_epoch']-24*3600
    assert a['order']==['TRAIN:C50:1024','EVAL:C50:N294101','TRAIN:D25:1024','EVAL:D25:N294101']
    assert a['engineering_replacements_total']==4 and a['old_clocks_unchanged']


@pytest.mark.parametrize('epoch',[True,0,-1,float('nan'),float('inf')])
def test_no_invented_registration_time(epoch):
    with pytest.raises(PermissionError):s.scope(epoch)


def test_whole_pair_not_one_arm_or_valid_failure_replay():
    p=s.whole_package()
    assert p['optimizer_updates']==2048 and p['PBS_submissions']==2
    assert p['episode_starts']==48+4 and p['physics_ticks']==52*32000
    assert p['neural_queries']==2*29+52*80+2*17
    assert p['H200_s']==2*22560+2*28860
    assert p['Hopper_storage_bytes']>=2*70000000000+2*s.MODEL_BYTES
    assert p['Core_storage_bytes']>=52*768*1024**2+2*s.MODEL_BYTES


def test_existing_projection_preserves_inputs_and_all_protection(monkeypatch):
    used=s._vector({});held=s._vector({});caps={k:10**15 for k in used}
    used['Hopper_storage_bytes']=100;caps['Hopper_storage_bytes']=100
    monkeypatch.setattr(s,'resource_floor',lambda g,l:(deepcopy(used),deepcopy(held)))
    monkeypatch.setattr(s,'legacy_caps',lambda g:deepcopy(caps))
    g=dict(pending_updates=[],unknown_effects=[],old_T0=1792348196);before=deepcopy(g)
    r=s.recompute(g,{},Core_storage_upper_bytes=0,local_storage_upper_bytes=0)
    assert g==before and not r['native_admission'] and r['cap_changes']=={}
    row=r['dimensions']['Hopper_storage_bytes']
    assert row['matrix']==s.MATRIX['Hopper_storage_bytes']
    assert row['final']==40*1024**3
    assert row['shortfall']==s.whole_package()['Hopper_storage_bytes']+row['matrix']+row['final']
    assert set(r['shortfalls'])=={'Hopper_storage_bytes'}


def test_pending_effects_and_historical_storage_not_discarded(monkeypatch):
    used=s._vector(dict(Core_storage_bytes=10));held=s._vector({})
    monkeypatch.setattr(s,'resource_floor',lambda g,l:(deepcopy(used),deepcopy(held)))
    monkeypatch.setattr(s,'legacy_caps',lambda g:{k:10**15 for k in used})
    with pytest.raises(PermissionError):s.recompute(dict(pending_updates=[],unknown_effects=[]),{},Core_storage_upper_bytes=9,local_storage_upper_bytes=0)
    with pytest.raises(PermissionError):s.recompute(dict(pending_updates=['u'],unknown_effects=[]),{},Core_storage_upper_bytes=10,local_storage_upper_bytes=0)


def test_exact_grant_not_a_cap_inference():
    from pathlib import Path
    p=Path(__file__).parents[2]/'reports/PI0294/design/parallel-science-20261022/V5_CD_TWO_STORAGE_CAPS_AND_START_AUTHORIZATION_20261025.json'
    b=p.read_bytes()
    assert s.grant(b)['other_caps_unchanged']
    with pytest.raises(PermissionError):s.grant(b+b' ')


def test_marker_has_one_clock_and_exact_two_caps():
    from pathlib import Path
    b=(Path(__file__).parents[2]/'reports/PI0294/design/parallel-science-20261022/V5_CD_TWO_STORAGE_CAPS_AND_START_AUTHORIZATION_20261025.json').read_bytes()
    m=s.marker(1792933000,dict(GLOBAL='a'*64,LOCAL='b'*64),b)
    assert s.validated({s.FIELD:m})==m
    bad=deepcopy(m);bad['deadline_epoch']+=1
    with pytest.raises(PermissionError):s.validated({s.FIELD:bad})
    bad=deepcopy(m);bad['authorized_storage_caps']['Hopper_storage_bytes']+=1
    with pytest.raises(PermissionError):s.validated({s.FIELD:bad})
