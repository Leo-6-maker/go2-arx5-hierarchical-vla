import numpy as np
from scripts.sla.analyze_joint_suffix_v7 import partition


def test_prefix_boundary_uses_rightmost_knot_including_original_history():
    q=np.zeros((42,6));q[11,2]=.003
    p=partition(q)
    assert p['first_exceedance_k']==9 and p['first_exceedance_partition']=='PREFIX'
    assert p['joint3_prefix_max_abs_acceleration_rad_s2']>1
    q=np.zeros((42,6));q[12,2]=.003
    p=partition(q)
    assert p['first_exceedance_k']==10 and p['first_exceedance_partition']=='SUFFIX'
    assert p['prefix_max_abs_acceleration_rad_s2']==0
    assert p['prefix_only_execution_admitted']is False


def test_history_join_is_reported_and_not_silently_removed():
    q=np.zeros((42,6));q[2:,2]=.003
    p=partition(q)
    assert p['first_exceedance_k']==0
    assert np.isclose(p['reference_join_max_abs_acceleration_rad_s2'],1.2)
