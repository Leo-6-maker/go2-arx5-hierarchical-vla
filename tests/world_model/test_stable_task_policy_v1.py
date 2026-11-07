"""Synthetic candidate previews, not actual native qualification."""
import numpy as np
import pytest
from go2arx5_vla.world_model import stable_task_policy_v1 as policy
from go2arx5_vla.world_model import stable_task_adapter_v1 as candidate
from go2arx5_vla.world_model.storage import digest


def test_no_privileged_argument_or_silent_command_scaling():
    actual=np.eye(4);base=np.array([0.,0.,.4,0.,0.,0.,1.])
    delta=np.array([.002,-.001,0.])
    program,proof=policy.build_for_delta(actual,actual,base,delta)
    np.testing.assert_array_equal(program.decoded_H40[-1,:3],delta)
    np.testing.assert_array_equal(program.decoded_H40[0,:3],np.zeros(3))
    assert proof['candidate_command_clipped']is False
    assert proof['raw_reference_is_GR00T']is False
    assert proof['native_execution_authorized']is False
    with pytest.raises(ValueError,match='BOUNDED_ACTUAL'):
        policy.build_for_delta(actual,actual,base,np.array([.15,0.,0.]))


def test_export_is_pinned_and_has_only_finite_trained_arrays(tmp_path):
    model=candidate.make_model(np.zeros(116,np.float32),np.ones(116,np.float32))
    path=tmp_path/'export.npz';arrays=candidate.export_arrays(model);np.savez(path,**arrays)
    loaded=policy.load_export(dict(path=str(path),sha256=digest(path)))
    for key in arrays:np.testing.assert_array_equal(loaded[key],arrays[key])
    arrays['scale'][0]=np.nan;bad=tmp_path/'bad.npz';np.savez(bad,**arrays)
    with pytest.raises(ValueError,match='FINITE_TRAINED'):
        policy.load_export(dict(path=str(bad),sha256=digest(bad)))
    with pytest.raises(ValueError,match='PINNED'):
        policy.load_export(dict(path=str(path),sha256='0'*64))
