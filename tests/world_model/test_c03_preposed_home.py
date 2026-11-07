import numpy as np
import pytest
from scripts.wm02.prepare_c03_preposed_home import replace_arm_home


def test_only_declared_arm_initial_angles_change():
    q=np.arange(34,dtype=float)/100;indices=np.arange(19,25);v=np.ones(6);b=np.tile([-2.,2.],(6,1))
    out=replace_arm_home(q,indices,v,b);mask=np.ones(34,dtype=bool);mask[indices]=False
    np.testing.assert_array_equal(q[mask],out[mask]);np.testing.assert_array_equal(out[indices],v)
    assert not np.shares_memory(q,out)


@pytest.mark.parametrize('indices,values',[(np.arange(6),np.zeros(6)),(np.array([19]*6),np.zeros(6)),(np.arange(19,25),np.full(6,1.99)),(np.arange(19,25),np.full(6,np.nan))])
def test_base_changes_duplicates_and_unqualified_angles_reject(indices,values):
    with pytest.raises(ValueError,match='SIX_ARM_INITIAL'):
        replace_arm_home(np.zeros(34),indices,values,np.tile([-2.,2.],(6,1)))
