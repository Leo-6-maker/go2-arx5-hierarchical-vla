import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from go2arx5_vla.sla_adapter.source_native_v6 import relative_arrays,pad32


def logged(frames=100):
    x=np.arange(frames)/1000
    rot=Rotation.from_rotvec(np.c_[np.zeros(frames),np.zeros(frames),x]).as_quat()
    state=np.c_[x,np.zeros(frames),np.zeros(frames),rot[:,[3,0,1,2]]]
    action=np.vstack([state[1:],state[-1:]])
    gs=np.linspace(.1,.9,frames).reshape(-1,1);ga=np.vstack([gs[1:],gs[-1:]])
    return [state,state.copy()],[action,action.copy()],[gs,gs.copy()],[ga,ga.copy()]


def test_native_h50_preserves_both_arms_one_offset_and_excludes_copied_tail():
    s,a,g,h=logged();state,actions,audit=relative_arrays(s,a,g,h)
    assert state.shape==(50,14)and actions.shape==(50,50,14)
    assert actions[0,0,0]==pytest.approx(.001)and actions[0,49,0]==pytest.approx(.05)
    assert actions[-1,-1,6]==pytest.approx(g[0][-1,0])
    assert audit['last_copied_action_excluded']and audit['first_future_state_offset_frames']==1
    assert actions[0,0,3:6]==pytest.approx([0,0,.001])
    assert pad32(actions).shape==(50,50,32)and not pad32(actions)[...,14:].any()
    for row in audit['roundtrip']:assert row['SO3_max_rad']<1e-6


def test_broken_relationship_and_nonunit_rotation_are_rejected_not_repaired():
    s,a,g,h=logged();a[0][0,0]+=1
    with pytest.raises(ValueError,match='RELATION'):relative_arrays(s,a,g,h)
    s,a,g,h=logged();s[0][:,3:]*=2;a[0][:,3:]*=2
    with pytest.raises(ValueError,match='QUATERNION'):relative_arrays(s,a,g,h)


def test_anchor_static_is_not_misreported_as_a_static_future_window():
    from go2arx5_vla.sla_adapter.source_native_v6 import anchor_strata
    s,a,g,h=logged();state,actions,_=relative_arrays(s,a,g,h)
    # Adjacent translations/rotations .001 are below a .01 native threshold,
    # while the unchanged complete future moves .05. Do not relabel that
    # future as motionless or feed the previous frame to the student.
    phase=anchor_strata(state,actions,np.full(14,1.))
    assert phase[0]==-1
    assert not (phase[1:]==1).any()
    assert actions[0,-1,0]>.01
