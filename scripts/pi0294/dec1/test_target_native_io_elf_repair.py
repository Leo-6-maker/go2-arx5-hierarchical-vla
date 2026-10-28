"""Pure cap-delta and immutable-history tests; no native transition."""
from copy import deepcopy
from pathlib import Path
import pytest
from . import target_native_io_elf_repair as q
from . import target_native_io_budget as budget

def grant_bytes():
    return (Path(__file__).resolve().parents[3]/'reports/PI0294/design/WDR1_ONE_LOCAL_CONTROL_MINUTE_AUTHORIZATION_20261024.json').read_bytes()

def test_actual_published_grant_is_one_local_minute_only():
    raw=grant_bytes();v=q.grant(raw)
    assert v['exact_cap_change']==dict(DEC1_LOCAL_CPU_s=dict(before=57600,after=57660,delta=60))
    with pytest.raises(PermissionError):q.grant(raw+b' ')

@pytest.mark.parametrize('fault',('missing','one_side','different_cap','cost_reset','clock_reset'))
def test_exact_dual_marker_required(fault):
    p={q.MARKER:q.marker()};c=deepcopy(p)
    if fault=='missing':p={};c={}
    elif fault=='one_side':c={}
    elif fault=='different_cap':p[q.MARKER]['after_LOCAL_CPU_s']=60000;c=deepcopy(p)
    elif fault=='cost_reset':p[q.MARKER]['all_other_caps_clocks_and_costs_unchanged']=False;c=deepcopy(p)
    elif fault=='clock_reset':p[q.MARKER]['original_T0_epoch']=0;c=deepcopy(p)
    with pytest.raises(PermissionError):q.adopted(p,c)

@pytest.mark.parametrize('shortfall',(0,58.97468250899692,70))
def test_only_local_shortfall_changes_no_other_credit(shortfall):
    p={q.MARKER:q.marker()};c=deepcopy(p)
    v=dict(effective_DEC1_VAP1_caps=dict(local_CPU_s=57600,Core_CPU_s=576000),
        DEC1_shortfalls=dict(local_CPU_s=shortfall,physics_ticks=7),global_shortfalls={'Core_CPU_s':9},clock_failures=['original'])
    before=deepcopy(v);q.extend_forecast(v,p,c)
    assert v['effective_DEC1_VAP1_caps']==dict(local_CPU_s=57660,Core_CPU_s=576000)
    assert v['DEC1_shortfalls'].get('local_CPU_s',0)==max(0,shortfall-60)
    assert v['DEC1_shortfalls']['physics_ticks']==7 and v['global_shortfalls']==before['global_shortfalls']
    assert v['clock_failures']==before['clock_failures']
    with pytest.raises(PermissionError):q.extend_forecast(v,p,c)

def test_changed_closed_packet_cannot_enter_verifier():
    packet=q.pack({name:b'CPU fixture cannot be an actual closure' for name in q.ANCHORS})
    with pytest.raises(PermissionError):q.verify(packet,prior_row={},prior_settlement={})

def test_no_fourth_io_pool_and_fixed_full_request():
    assert budget.REQUESTED['local_CPU_s']==60 and budget.REQUESTED['CoreLab_CPU_s']==3600
    assert budget.REQUESTED['physics_ticks']==budget.REQUESTED['episode_starts']==0
