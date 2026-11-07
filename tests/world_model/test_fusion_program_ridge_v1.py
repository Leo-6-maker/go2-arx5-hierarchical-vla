import numpy as np
import pytest
from go2arx5_vla.world_model import fusion_program_ridge_v1 as ridge
from go2arx5_vla.world_model.fusion_program_response_data_v1 import FEATURE_DIM,FUTURE_SLICE


def data():
    rng=np.random.default_rng(13);n=48
    x=np.zeros((n,FEATURE_DIM));x[:,FUTURE_SLICE.start]=rng.normal(size=n)
    y=np.zeros((n,16,8));y[:,:,0]=x[:,FUTURE_SLICE.start,None]*np.arange(1,17)*.001
    return dict(X=x,Y=y,valid=np.ones((n,16),bool),split=np.array(['TRAIN']*32+['DEV']*16),
        parent=np.array(['T']*32+['D']*16),run_id=np.array(['T']*32+['D']*16),program_atom=np.arange(n))


def test_train_only_action_response_generalizes_and_beats_no_future_control():
    d=data();ids=np.arange(32)
    action=ridge.fit(d,ids);blind=ridge.fit(d,ids,no_future=True)
    prediction=ridge.predict(action,d['X']);noaction=ridge.predict(blind,d['X'])
    assert np.mean((prediction[32:]-d['Y'][32:])**2)<np.mean((noaction[32:]-d['Y'][32:])**2)*.01
    changed=d['X'].copy();changed[:,FUTURE_SLICE]+=100
    np.testing.assert_array_equal(ridge.predict(blind,changed),noaction)
    d['Y'][32:]+=1000;again=ridge.fit(d,ids)
    for name in action:np.testing.assert_array_equal(again[name],action[name])


def test_dev_labels_cannot_fit_a_linear_control():
    d=data()
    with pytest.raises(ValueError,match='TRAIN_ONLY'):ridge.fit(d,np.arange(48))
