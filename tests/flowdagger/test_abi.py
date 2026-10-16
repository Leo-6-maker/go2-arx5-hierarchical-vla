import numpy as np
import pytest

from go2arx5_vla.flowdagger.abi import SOURCE_ABI,validate_source_abi,validate_label_mask
from .test_codec import action


@pytest.mark.parametrize('change',[dict(position_units='mm'),dict(tcp='OTHER_TOOL'),dict(gripper='1_CLOSED_MINUS1_OPEN'),
    dict(disk_action='ALREADY_RELATIVE'),dict(observation_hz=10)])
def test_five_declared_abi_errors_rejected(change):
    with pytest.raises(ValueError):validate_source_abi(dict(SOURCE_ABI,**change))


def test_source_cannot_acquire_fake_zero_base_labels():
    a=action()
    assert validate_label_mask(a,np.ones_like(a,dtype=bool),domain='FIXED_BASE_ARX5_TASK_SIM',base_preference_valid=False).all()
    fake=np.pad(a,((0,0),(0,3)))
    with pytest.raises(ValueError):validate_label_mask(fake,np.ones_like(fake,dtype=bool),domain='FIXED_BASE_ARX5_TASK_SIM',base_preference_valid=True)


def test_target_missing_base_and_soft_zero_have_different_masks():
    a=np.pad(action(),((0,0),(0,3)));mask=np.ones_like(a,dtype=bool)
    domain='FULL_GO2_ARX5_TASK_SIM_EEF13_STATE19_V1'
    with pytest.raises(ValueError):validate_label_mask(a,mask,domain=domain,base_preference_valid=False)
    mask[:,10:]=False
    assert not validate_label_mask(a,mask,domain=domain,base_preference_valid=False)[:,10:].any()
    assert validate_label_mask(a,np.ones_like(a,dtype=bool),domain=domain,base_preference_valid=True).all()
