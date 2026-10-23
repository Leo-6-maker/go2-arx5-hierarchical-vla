"""Original target identities and fail-closed scope; no native execution."""
from copy import deepcopy
import pytest
from .test_source_commissioning_plan import population
from scripts.pi0294.dec1 import source_commissioning_plan as source,target_commissioning_plan as unit
from scripts.pi0294.scene_cards import digest

def test_original_four_target_slots_preserve_static_geometry_and_fixed_student(population):
    value,encode=population;raw=encode(value);plan=unit.build(raw,owner_release='e'*40)
    old=source.build(raw,owner_release='e'*40)
    assert plan['original_source_slots']==plan['remaining_target_slots']==4
    assert plan['total_original_commissioning_upper']==8 and len(plan['conditions'])==4
    assert plan['source_outcomes_are_not_an_input'] and not plan['target_task_native_qualified']
    assert not plan['new_GPU_or_PBS_authorized'] and plan['core64_cells']==0
    for c,s in zip(plan['conditions'],old['conditions']):
        a,b=deepcopy(c['card']),deepcopy(s['card']);a.pop('id');b.pop('id')
        assert a==b and c['card_sha256']==digest(c['card'])
        assert c['policy']=='CORR2048' and c['checkpoint_manifest_sha256']==unit.PARENT
        assert c['statistics_sha256']==unit.R0 and c['executor']=='E_TARGET'
        assert c['native_tick_origin']==2500 and c['task_ticks']==16000 and c['physical_tick_upper_including_warmup']==18500
    assert unit.validate(plan,raw,owner_release='e'*40)==plan

@pytest.mark.parametrize('fault',['new_pool','A_substitution','new_clock','source_success_gate','SDK_as_safety','geometry','zero_warmup','old_pilot','implicit_GPU'])
def test_changes_cannot_silently_admit_target_or_reset_original_scope(population,fault):
    value,encode=population;raw=encode(value);p=unit.build(raw,owner_release='e'*40)
    if fault=='new_pool':p['remaining_target_slots']=8
    elif fault=='A_substitution':p['conditions'][0]['policy']='A_CONTROL'
    elif fault=='new_clock':p['conditions'][0]['task_clock_sha256']='f'*64
    elif fault=='source_success_gate':p['source_outcomes_are_not_an_input']=False
    elif fault=='SDK_as_safety':p['SDK12_or_CPU_PASS_is_not_wholebody_prefix_safety']=False
    elif fault=='geometry':p['conditions'][0]['card']['source_xy'][0]+=.01
    elif fault=='zero_warmup':p['upper_counts_requiring_fresh_full_resource_admission']['warmup_ticks']=0
    elif fault=='old_pilot':p['old_R4_run_and_expired_pilots_must_not_execute']=False
    else:p['new_GPU_or_PBS_authorized']=True
    with pytest.raises(PermissionError):unit.validate(p,raw,owner_release='e'*40)
