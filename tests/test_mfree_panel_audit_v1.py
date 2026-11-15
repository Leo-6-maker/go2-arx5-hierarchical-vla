import numpy as np
from scripts.wm02.audit_mfree_panel_v1 import per_knot
from go2arx5_vla.world_model.fusion_trajectory_adapter import _geometry
from scripts.pi0292.eval_contract import matrix10


def path():return np.tile([.2,0,.7,1,0,0,0,1,0,.5],(40,1))


def test_original_rest_endpoints_are_reported_without_relaxing_full40():
    p=path();p[:,0]+=np.arange(40)*.001
    metrics=per_knot(p);a=metrics['translation_acceleration']
    assert np.isclose(a['maximum'],.4) and a['first_violation_k']==0
    assert np.isclose(a['prefix_maximum'],.4) and np.isclose(a['suffix_maximum'],.4)
    assert a['endpoint_rest_padding_in_acceleration']
    assert np.isclose(a['maximum'],_geometry(p,matrix10(p[0]))['maximum_discrete_translation_acceleration_m_s2'])


def test_suffix_failure_cannot_disappear_behind_quiet_prefix():
    p=path();p[-1,0]+=.001
    a=per_knot(p)['translation_acceleration']
    assert a['prefix_maximum']==0 and a['first_violation_k']==38
    assert np.isclose(a['ratio'],1.6) and a['suffix_maximum']>a['limit']
