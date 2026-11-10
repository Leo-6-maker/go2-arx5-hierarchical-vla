import numpy as np
from go2arx5_vla.world_model.task_learning_executability_v3 import path_series


def path():
    rows=np.zeros((40,10));rows[:,3:9]=[1,0,0,0,1,0];rows[:,9]=1.
    return rows


def test_linear_path_has_only_zero_padded_start_and_end_acceleration():
    x=path();x[:,0]=np.arange(40)*.001
    s=path_series(x)['linear_acceleration']
    assert s['exceedance_indices']==[0,39]
    assert s['exceedances_by_part']==dict(first=1,interior=0,last=1)
    assert np.isclose(s['peak_value'],.4)


def test_submillimeter_alternation_is_an_internal_acceleration_failure():
    x=path();x[:,0]=np.where(np.arange(40)%2,.0002,-.0002)
    s=path_series(x)['linear_acceleration']
    assert s['exceedances_by_part']['interior']==38
    assert np.isclose(s['peak_value'],.32)


def test_constant_offset_does_not_invent_internal_motion_or_grip_acceleration_gate():
    x=path();x[:,0]=.1
    s=path_series(x)
    assert s['linear_velocity']['peak_value']==s['linear_acceleration']['peak_value']==0.
    assert s['first_state_to_point_gap_included']is False
    assert s['finger_acceleration']['limit']is None


def test_working_grip_affine_is_not_mechanical_range_or_repeated_normalization():
    x=path();x[:,9]=np.linspace(1.,0.,40);s=path_series(x)
    assert np.isclose(s['working_gripper_width_m'][0],.042)
    assert np.isclose(s['working_gripper_width_m'][-1],.002)
    assert np.isclose(s['finger_velocity']['peak_value'],.04/1.95)
