import numpy as np
from go2arx5_vla.world_model.task_learning_conic_decoder_v3 import fit
from go2arx5_vla.world_model.task_learning_local_analysis_v1 import geometry


def raw():
    x=np.zeros((40,10));x[:,3]=x[:,7]=1.;x[:,9]=.8
    return x


def test_conic_removes_internal_zigzag_with_exact_endpoints_and_rotation():
    x=raw();x[1:-1,0]=np.tile([.0008,-.0008],19)
    y,a=fit(x)
    assert a['solver_success']and geometry(y,x[0])['original_envelope_pass']
    assert np.array_equal(y[[0,-1]],x[[0,-1]])
    assert np.array_equal(y[:,3:9],x[:,3:9])
    assert a['translation']['status']=='Solved'and a['gripper']['status']=='Solved'


def test_infeasible_endpoints_are_explicit_failure_without_raw_fallback():
    x=raw();x[-1,0]=1.
    y,a=fit(x)
    assert y is None and not a['solver_success']
    assert a['translation']['status']=='PrimalInfeasible'


def test_nominal_gripper_fit_keeps_endpoints_and_original_speed_bound():
    x=raw();x[15:25,9]=0.
    y,a=fit(x)
    assert a['solver_success']and geometry(y,x[0])['original_envelope_pass']
    assert np.array_equal(y[[0,-1]],x[[0,-1]])
    assert np.all((y[:,9]>=0)&(y[:,9]<=1))
