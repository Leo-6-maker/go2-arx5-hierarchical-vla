import numpy as np
from go2arx5_vla.world_model.mobile_observation_v2 import ObservedGrasp,AfterGraspRoute


def test_failed_initialization_cannot_fabricate_grasp_or_base_preview():
    s=dict(native_step=5703,native_grasp=False,object_pose=np.eye(4),tcp=np.eye(4))
    observed=ObservedGrasp(0);route=AfterGraspRoute('+x')
    for k in range(600):
        s['native_step']+=25
        assert not observed.observe(s)
        assert not route.trigger(s,np.eye(4),observed.qualified)
        assert route.preview(s['native_step'])==('M_FREE',None)
    assert not observed.ever_grasp and route.evidence()is None


def test_contiguous_real_grasp_lift_and_measured_trigger_origin():
    s=dict(native_step=5703,native_grasp=True,object_pose=np.eye(4),tcp=np.eye(4));s['object_pose'][2,3]=.021
    observed=ObservedGrasp(0);route=AfterGraspRoute('+y');mount=np.eye(4);mount[0,3]=.123
    for k in range(20):
        assert not observed.observe(s);s['native_step']+=25
    assert observed.observe(s)
    assert route.trigger(s,mount,observed.qualified)
    mode,base=route.preview(s['native_step']);assert mode=='M_ASSIST'
    assert np.array_equal(base[0],mount)
    mount[0,3]=.9
    assert route.evidence()['initial_mount'][0][3]==.123
    s['native_step']+=100
    assert not observed.observe(s) # A recording gap cannot count toward 1 second.
