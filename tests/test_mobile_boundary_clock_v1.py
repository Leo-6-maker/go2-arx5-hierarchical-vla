import numpy as np
import pytest
from go2arx5_vla.world_model.mobile_boundary_clock_v1 import accepted_boundary


def evidence():
    event=dict(kind='installed',ref_id='a',times=[11.406+k*.05 for k in range(41)],
        poses=[[k*.001,0,0,0,0,0,1] for k in range(41)],inputs=[[] for k in range(41)],
        monotonic_s=1.,bound_start=11.406,observation_time_s=11.406)
    policy=dict(topic_key='policy',received_monotonic_s=2.,message=dict(plan_target_trajectories=dict(
        time_trajectory=event['times'],state_trajectory=[dict(value=p) for p in event['poses']],
        input_trajectory=[dict(value=p) for p in event['inputs']])))
    return event,policy


def test_boundary_uses_true_050_receiver_knot_and_solver_witness():
    event,policy=evidence()
    pose,w=accepted_boundary([event],'a',11.906,[policy])
    assert pose[0,3]==.010 and w['receiver_knot_index']==10
    assert w['receiver_knot_time_s']==11.906 and np.allclose(pose[:3,:3],np.eye(3))
    with pytest.raises(ValueError,match='SOLVER_TARGET_WITNESS'):
        accepted_boundary([event],'a',11.906,[])
    with pytest.raises(ValueError,match='NO_EXACT'):
        accepted_boundary([event],'a',11.907,[policy])


def test_initial_constant_base_hold_is_explicitly_separate():
    event,policy=evidence()
    for row in event['poses']:row.extend([0,0,.4,0,0,0,1])
    with pytest.raises(ValueError,match='EEF_ONLY'):
        accepted_boundary([event],'a',11.906,[policy])
    pose,w=accepted_boundary([event],'a',11.906,[policy],initial_hold=True)
    assert pose[0,3]==.010 and w['initial_constant_BASE_hold']


def test_immutable_callback_identity_cache_preserves_exact_boundary(monkeypatch):
    from go2arx5_vla.world_model import mobile_boundary_clock_v1 as clock
    event,policy=evidence();cache={};calls=[];original=clock.trajectory_identity
    def measured(*args):calls.append(1);return original(*args)
    monkeypatch.setattr(clock,'trajectory_identity',measured)
    a,w=accepted_boundary([event],'a',11.906,[policy],policy_identity_cache=cache)
    b,z=accepted_boundary([event],'a',11.956,[policy],policy_identity_cache=cache)
    c,q=accepted_boundary([event],'a',11.956,[policy])
    assert np.array_equal(b,c) and z==q and len(calls)==2
    assert w['receiver_knot_index']==10 and z['receiver_knot_index']==11
