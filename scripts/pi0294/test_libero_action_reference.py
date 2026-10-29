"""Causality, world rotation and stateful goal checks; no simulator."""
import numpy as np
import pytest
from scripts.pi0294.multitask.libero_action_reference import extract,rotation


def config():
    return dict(env_kwargs=dict(robots=['Panda'],control_freq=20,controller_configs=dict(type='OSC_POSE',
        impedance_mode='fixed',control_delta=True,input_min=-1,input_max=1,
        output_max=[.05,.05,.05,.5,.5,.5],output_min=[-.05,-.05,-.05,-.5,-.5,-.5],
        position_limits=None,orientation_limits=None,interpolation=None)))


def test_lagged_actual_measurement_world_delta_and_zero_goal_memory():
    actions=np.array([[0,0,0,0,0,0,-1],[1,0,0,.4,0,0,-1],[0,0,0,0,0,0,0.]])
    states=np.array([[.1,.2,.3,0,.7,0],[8,9,10,0,0,1.],[90,90,90,1,1,1.]])
    rows=extract(actions,states,config())['rows']
    assert len(rows)==2
    np.testing.assert_allclose(rows[0]['controller_target_world'][:3,3],[.15,.2,.3])
    np.testing.assert_allclose(rows[0]['controller_target_world'][:3,:3],rotation([.2,0,0])@rotation([0,.7,0]))
    assert not np.allclose(rows[0]['controller_target_world'][:3,:3],rotation([0,.7,0])@rotation([.2,0,0]))
    np.testing.assert_array_equal(rows[1]['controller_target_world'][:3,:3],rows[0]['controller_target_world'][:3,:3])
    assert rows[1]['gripper_direction']==-1
    assert rows[0]['pre_action_observation_index']==0


def test_unrecoverable_initial_orientation_is_not_fabricated_and_unknown_config_refused():
    actions=np.array([[0,0,0,0,0,0,-1],[0,0,0,0,0,0,-1.]])
    v=extract(actions,np.zeros((2,6)),config())
    assert v['rows']==[]and v['unknown_stateful_goal_indices']==[1]
    c=config();c['env_kwargs']['control_freq']=10
    with pytest.raises(ValueError):extract(actions,np.zeros((2,6)),c)
