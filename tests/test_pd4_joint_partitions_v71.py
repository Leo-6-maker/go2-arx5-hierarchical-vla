import numpy as np

from scripts.sla.diagnose_pd4_native_rejects_v71 import acceleration_partitions


def test_triple_minus1_0_1_is_history_join_even_with_center_at_H40_zero():
    values=np.zeros((40,6))
    values[0,3]=.7       # [-2,-1,0]
    values[1,3]=1.1      # [-1,0,1], center0 crosses the reference boundary.
    values[2,3]=.3       # [0,1,2]
    values[11,2]=2.      # [9,10,11], center10 belongs to the suffix.
    result=acceleration_partitions(dict(acceleration_rad_s2=values,history_points=2))
    assert result==dict(reference_join_triples_with_history=1.1,
        H40_prefix_center1_9_without_history=.3,H40_suffix_center10_38=2.)
