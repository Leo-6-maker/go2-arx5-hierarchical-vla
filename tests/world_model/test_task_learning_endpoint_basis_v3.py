import numpy as np
import pytest
from go2arx5_vla.world_model.task_learning_endpoint_basis_v3 import decode
from go2arx5_vla.world_model.task_learning_local_analysis_v1 import geometry


def test_global_curve_preserves_inputs_end_goal_current_pose_and_nominal_grip():
    state=np.array([0.,0.,0.,1.,0.,0.,0.,1.,0.,.95454545])
    raw=np.repeat(state[None],40,axis=0);raw[:,9]=1.;raw[:,0]=np.linspace(.0006,.01,40)
    original=raw.copy();value,audit=decode(raw,state,consumed_nominal_grip=1.)
    assert np.array_equal(raw,original)and np.array_equal(value[0,:9],state[:9])and value[0,9]==1.
    assert np.array_equal(value[-1],raw[-1])and geometry(value,state)['original_envelope_pass']
    assert audit['command_clips']==0 and not audit['native_execution_authority']and audit['last_knot_time_s']==pytest.approx(1.95)


def test_full_grip_close_obeys_original_finger_rate_without_clipping():
    state=np.array([0.,0.,0.,1.,0.,0.,0.,1.,0.,.95]);raw=np.repeat(state[None],40,axis=0);raw[:,9]=0.
    value,_=decode(raw,state,consumed_nominal_grip=1.)
    assert geometry(value,state)['original_envelope_pass']and value[-1,9]==0.
    with pytest.raises(ValueError):decode(raw,state,consumed_nominal_grip=1.001)
    raw[3,9]=-1e-9
    with pytest.raises(ValueError):decode(raw,state,consumed_nominal_grip=1.)
