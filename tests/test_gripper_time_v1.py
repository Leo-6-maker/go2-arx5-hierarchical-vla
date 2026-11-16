import numpy as np
import pytest
from go2arx5_vla.sla_adapter.gripper_time_v1 import decode,label_parameters

LOW,HIGH=-9.21024036697585,9.21024036697596

def expert(t):
    u=np.clip(np.asarray(t)/2,0,1)
    return .042-.040*(10*u**3-15*u**4+6*u**5)

def context(w):return dict(width_m=w,source_native_s=10.,query_native_s=10.05)


def test_reference_continuation_preserves_seen_event_and_opening():
    for elapsed in(0.,.5,1.,1.5,1.95,1.99,2.):
        target=expert(elapsed+np.arange(40)*.05)
        previous=float(expert(elapsed-.05))
        for opening in(False,True):
            labels=.044-target if opening else target
            ctx=context(.044-previous if opening else previous)
            normalized,_=label_parameters(labels,ctx,LOW,HIGH)
            actual,info=decode(normalized,ctx,LOW,HIGH)
            assert np.max(abs(actual-labels))<1e-5
            assert info['analytical_speed_bound_m_s']<=.04+1e-12
            assert np.max(abs(np.diff(np.r_[ctx['width_m'],actual])/.05))<=.04+1e-12


def test_hold_and_arbitrary_predictions_respect_original_width_and_rate_bounds():
    for width in(.002,.042):
        normalized,_=label_parameters(np.full(40,width),context(width),LOW,HIGH)
        actual,_=decode(normalized,context(width),LOW,HIGH)
        assert max(abs(actual-width))<1e-5
    rng=np.random.default_rng(7264)
    for _ in range(100):
        ctx=context(float(rng.uniform(.002,.042)))
        actual,_=decode(rng.normal(0,3,40),ctx,LOW,HIGH)
        assert actual.min()>=.002-1e-12 and actual.max()<=.042+1e-12
        assert np.max(abs(np.diff(np.r_[ctx['width_m'],actual])/.05))<=.04+1e-12
    with pytest.raises(ValueError):decode(np.zeros(40),dict(context(.042),query_native_s=10.06),LOW,HIGH)
    with pytest.raises(ValueError):decode(np.full(40,np.nan),context(.042),LOW,HIGH)
