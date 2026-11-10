import numpy as np

from go2arx5_vla.world_model.task_learning_feasible_decoder_v3 import difference_operators,fit,summarize_change
from go2arx5_vla.world_model.task_learning_local_analysis_v1 import geometry


def raw():
    x=np.zeros((40,10));x[:,3]=x[:,7]=1.;x[:,9]=1.
    return x


def test_operators_include_original_rest_boundary_acceleration():
    velocity,acceleration=difference_operators();x=np.arange(40.)[:,None]*np.array([[1.,2.,3.]])
    assert np.array_equal(velocity@x,np.diff(x,axis=0))
    assert np.array_equal(acceleration@x,np.diff(np.vstack((np.zeros(3),np.diff(x,axis=0),np.zeros(3))),axis=0))


def test_jitter_fit_retains_intent_endpoints_and_original_envelope():
    x=raw();u=np.linspace(0,1,40);x[:,0]=.012*u**3*(10+u*(-15+6*u));x[1:-1,1]=.0008*(-1.)**np.arange(38)
    before=x.copy();decoded,info=fit(x)
    assert info['solver_success']and info['endpoints_exact_unchanged']and info['rotations_exact_unchanged']
    assert np.array_equal(x,before)
    state=np.r_[np.zeros(3),[1,0,0,0,1,0],1.]
    assert not geometry(x,state)['original_envelope_pass']
    assert geometry(decoded,state)['original_envelope_pass']


def test_static_legal_output_stays_static_and_is_never_task_success():
    x=raw();decoded,info=fit(x);summary=summarize_change(x,decoded,x)
    assert info['solver_success']and np.allclose(x,decoded)
    assert summary['endpoint_motion_m']==0. and summary['numerical_feasibility_is_task_success']is False


def test_impossible_fixed_gripper_endpoint_is_rejected_without_clamping():
    x=raw();x[-1,9]=1.1;decoded,info=fit(x)
    assert not info['solver_success']and decoded[-1,9]==1.1
