"""Captured historical budget plus synthetic closed C50; never native adoption."""
import base64,json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.pi0294.test_pv1_core_evaluation_accounting import captured_global
from scripts.pi0294.test_v5_closed_training_handoff import fixture
from scripts.pi0294.pv1 import mix_accounting as a,successor_cd as s,mix_stage_handoff as h
from scripts.pi0294.pv1 import mix_core_accounting as c,mix_core_cpu_projection as p


@pytest.fixture
def dual(captured_global):
    path=Path(__file__).resolve().parents[4]/'work/private/v5-execution-20261025/C50_CORE_CONTROL_PREPAY.private.json'
    if not path.exists():pytest.skip('private captured LOCAL budget absent')
    small,local,capture=fixture();pool=s.whole_package()
    local['CPU_reserved_s'].update(CoreLab=pool['Core_CPU_s'],local_PC=pool['local_CPU_s'])
    local['CPU_s'].update(CoreLab=0.,local_PC=0.)
    local=h.project_training_to_LOCAL(local_bytes=a.raw(local),global_bytes=a.raw(small),terminal_capture=capture,arm='C50')['proposed_LOCAL']
    g=deepcopy(captured_global)
    for k in (s.FIELD,a.PACKAGE):g[k]=deepcopy(small[k])
    g['PBS_submissions']+=small['PBS_submissions']
    for key in ('used','reserved'):
        for k,v in small[key].items():g[key][k]+=v
    for key in ('CPU_s','CPU_reserved_s','storage_reserved_bytes','new_storage_current_bytes','optimizer_updates','generation_queries','model_loads'):
        g[key]=g.get(key,0)+small[key]
    g['new_storage_peak_bytes']=max(g['new_storage_peak_bytes'],g['new_storage_current_bytes'])
    g['PBS_submissions_count']=len(g['PBS_submissions'])
    l=json.loads(base64.b64decode(json.loads(path.read_bytes())['value']['LOCAL_base64']));old=deepcopy(l)
    for k,v in local.items():l[k]=deepcopy(v)
    for site,v in old['CPU_s'].items():l['CPU_s'][site]+=v
    l['GPU_allocated_s']+=old['GPU_allocated_s'];l['PBS_submissions']=old['PBS_submissions']+local['PBS_submissions']
    l['optimizer_updates']+=old['optimizer_updates'];l['neural_queries']+=old['neural_queries']
    return dict(global_bytes=a.raw(g),local_bytes=a.raw(l),Core_storage_upper_bytes=335287617560,
        local_storage_upper_bytes=9126805504,now_epoch=1200.)


def test_once_actual_Core_dimension_keeps_original_package_history_and_all_caps(dual):
    before=deepcopy(dual);g,l=map(json.loads,(dual['global_bytes'],dual['local_bytes']))
    v=p.propose(**dual);after,local=v['proposed_global'],v['proposed_LOCAL']
    assert dual==before and not v['IO_performed']and not v['native_model_or_physics_admission']
    assert after['PBS_submissions']==g['PBS_submissions']and after['used']==g['used']
    assert after['CPU_s']==g['CPU_s']and after['CPU_reserved_s']==g['CPU_reserved_s']
    assert local['CPU_s']==l['CPU_s']and local['CPU_reserved_s']['CoreLab']==l['CPU_reserved_s']['CoreLab']+160000
    assert after[a.PACKAGE]['original_resources']==g[a.PACKAGE]['original_resources']
    assert after[a.PACKAGE]['remaining_resources']['Core_CPU_s']==g[a.PACKAGE]['remaining_resources']['Core_CPU_s']+160000
    assert not v['full_remaining_proof']['shortfalls']and p.validate(after,local)
    dims=v['full_remaining_proof']['dimensions'];assert dims['H200_s']['final']==72000 and dims['PBS_submissions']['matrix']==3
    assert dims['Core_CPU_s']['cap']==320*3600 and dims['Core_CPU_s']['projected']<=320*3600
    with pytest.raises(PermissionError):p.propose(**dict(dual,global_bytes=a.raw(after),local_bytes=a.raw(local)))


@pytest.mark.parametrize('fault',range(5))
def test_current_GPU_missing_dual_unknown_or_real_cap_gap_never_get_adoption(dual,fault):
    g,l=map(json.loads,(dual['global_bytes'],dual['local_bytes']))
    if fault==0:g['PBS_submissions'][-1]['status']='SUBMITTED_PENDING_NATIVE'
    if fault==1:l['unknown_effects']=[dict(kind='UNKNOWN')]
    if fault==2:l[a.PACKAGE]['remaining_resources']['Core_CPU_s']-=1
    if fault==3:l['CPU_s']['CoreLab']=320*3600
    if fault==4:g[a.PACKAGE]['completed_stages']=[];l[a.PACKAGE]['completed_stages']=[]
    with pytest.raises(PermissionError):p.propose(**dict(dual,global_bytes=a.raw(g),local_bytes=a.raw(l)))


def test_each_stage_needs_full_neural_and_physical_CPU_without_changing_legacy_vector():
    assert a.stage_vector('EVAL')['Core_CPU_s']==93900
    assert c.remaining_stage()['Core_CPU_s']==93900+80000
    assert c.remaining_stage()['Hopper_CPU_s']==80000
