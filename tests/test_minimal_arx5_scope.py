"""Full-vector forecast tests; never exercise an admission or ledger writer."""
import base64
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path

import pytest

from scripts.pi0294.generalization import minimal_scope as s


def test_fixed_pair_update_query_and_independent_PBS_counts():
    p = s.components()
    assert sum(x['optimizer_updates'] for x in p.values()) == 1026
    assert sum(x['PBS_submissions'] for x in p.values()) == 2
    assert sum(x['neural_queries'] for x in p.values()) == 147 + 2*2097
    assert sum(x['H200_s'] for x in p.values()) == 120840
    assert all(p[a+'_QUALIFICATION_AND_FORMAL512']['optimizer_updates'] == 513 for a in ('E1','E2'))


def test_large_raw_inference_output_adjusted_once_per_copy():
    from scripts.pi0294.pv1.mix_accounting import stage_vector
    from scripts.pi0294.pv1.evaluation_core_accounting import MODEL_BYTES
    old = stage_vector('EVAL', Core_GPU_projection=True)
    new = s.components()['E1_CORE_ORIGINAL24']
    delta = 16*1024**3 - MODEL_BYTES
    for key, copies in [('Hopper_storage_bytes',1),('Core_storage_bytes',1),('transfer_bytes',2)]:
        assert new[key] - old[key] == copies*delta
    assert new['Core_CPU_s'] == old['Core_CPU_s'] == 173900
    assert new['episode_starts'] == 26


def test_all_prior_dimensions_and_protections_survive(monkeypatch):
    from scripts.pi0294.pv1.future_budget import _vector
    old = dict(dimensions={k:dict(cap=1000000000000, projected=800000000000,
                                reserved=8, used=7) for k in _vector({})},
               T_CD=1, deadline=3, stop_new_training=2)
    pristine = deepcopy(old)
    monkeypatch.setattr(s, 'full_forecast', lambda a,b: old)
    p = s.first512_forecast(b'global',b'local')
    assert old == pristine
    assert p['scope_adopted'] is p['native_admission'] is False
    assert p['LOCAL_is_historical'] is True
    for key, row in p['dimensions'].items():
        assert row['projected'] == pristine['dimensions'][key]['projected'] + p['additional'][key]
        assert row['reserved'] == 8 and row['used'] == 7
    assert p['existing_C_D_GEN_Evals_and_old_costs_retained']
    assert p['matrix3_and_final20GPUh40GiB_retained']
    assert p['conditional1024_not_reserved_or_authorized']


def test_resource_rounding_never_lowers_an_adopted_cap(monkeypatch):
    from scripts.pi0294.pv1.future_budget import _vector
    old=dict(dimensions={k:dict(cap=1,projected=1)for k in _vector({})},
             T_CD=1,deadline=3,stop_new_training=2)
    monkeypatch.setattr(s,'full_forecast',lambda a,b:old)
    p=s.first512_forecast(b'g',b'l')
    for key,row in p['dimensions'].items():
        assert row['proposed_cap'] >= row['cap']
        assert row['headroom_after_proposed_cap'] >= 0
        if row['shortfall']:
            quantum=s.GIB if key.endswith('bytes')else 3600 if key.endswith('_s')else 1
            assert row['proposed_cap'] % quantum == 0


def test_unknown_or_incomplete_ledger_does_not_yield_admission():
    with pytest.raises((PermissionError, KeyError)):
        s.first512_forecast(b'{}',b'{}')


def test_actual_captured_whole_future_preserves_inputs_and_requires_new_caps():
    d=os.environ.get('ARX5_FIRST512_FORECAST_INPUTS')
    if not d: pytest.skip('Actual private captures explicitly required; no fixture called native')
    root=Path(d)
    gb=(root/'FIRST512_GLOBAL_CAPTURED_BYTES.private.json').read_bytes()
    lb=(root/'FIRST512_HISTORICAL_LOCAL_BYTES.private.json').read_bytes()
    before=(gb,lb)
    p=s.first512_forecast(gb,lb)
    assert before==(gb,lb)
    assert p['GLOBAL_sha256']=='993ea3f729935a47847fb7419d0c3cdfe5400e25ec491549c7b43bd97e2ef458'
    assert p['historical_LOCAL_sha256']=='7cea556c4c80765582644ce1ed47d71f3c0c3084b445d02dd16a55854b386f1a'
    assert p['baseline']['cap_changes_required']=={}
    assert p['dimensions']['optimizer_updates']['projected']==15812
    assert p['dimensions']['PBS_submissions']['proposed_cap']==39
    assert p['dimensions']['H200_s']['proposed_cap']==147*3600
    assert set(p['cap_changes_required'])=={'H200_s','PBS_submissions','Hopper_CPU_s','Core_CPU_s',
        'transfer_bytes','Hopper_storage_bytes','Core_storage_bytes'}
    assert p['scope_adopted'] is False
