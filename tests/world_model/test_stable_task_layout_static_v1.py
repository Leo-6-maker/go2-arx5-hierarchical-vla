"""Synthetic model-array tests only, not target static/native qualification."""
import numpy as np
import pytest
from go2arx5_vla.world_model import stable_task_layout_static_v1 as s


def model():
    import mujoco
    bodies=''.join(f'<body name="link{i}" pos="{i*.03} 0 0"><joint name="j{i}"/>'
        '<geom type="sphere" size=".01" mass=".01"/></body>' for i in range(20))
    motors=''.join(f'<motor joint="j{i}"/>' for i in range(20))
    home=' '.join(map(str,[0,0,1,1,0,0,0]+[0]*20+[.4549,-.076,.46,1,0,0,0]))
    return mujoco.MjModel.from_xml_string(f'<mujoco><option timestep=".002"/>'
        '<worldbody><body name="base_link"><freejoint/><geom type="sphere" size=".05" mass="1"/>'
        f'{bodies}</body><body name="task_cube" pos=".45 -.08 .46"><freejoint name="task_cube_free"/>'
        '<geom type="box" size=".02 .02 .02" mass=".05"/></body></worldbody>'
        f'<actuator>{motors}</actuator><keyframe><key name="home" qpos="{home}"/></keyframe></mujoco>')


def test_only_cube_home_xy_changes_with_all_robot_arrays_retained():
    m=model();before=s.arrays(m);r=s.move_cube_home_only(m,s.LAYOUTS['TRAIN01'])
    expected=before['key_qpos'].copy();expected[0,27:29]+=[-.03,-.02]
    np.testing.assert_array_equal(m.key_qpos,expected)
    for name,v in before.items():
        if name!='key_qpos':np.testing.assert_array_equal(getattr(m,name),v)
    assert r['robot_startup_joints_changed'] is False and r['robot_dynamics_changed'] is False


def test_robot_mass_change_is_rejected_by_original_array_comparison():
    m=model();before=s.arrays(m);s.move_cube_home_only(m,s.LAYOUTS['DEV01'])
    m.body_mass[1]+=.1
    with pytest.raises(ValueError,match='ONLY_HOME_CUBE_XY_CHANGED_body_mass'):
        s.verify_only_cube_home_change(before,m,s.LAYOUTS['DEV01'])


@pytest.mark.parametrize('delta',[(0.,0.),(.041,0.),(np.nan,0.)])
def test_unregistered_large_or_invalid_offsets_are_rejected(delta):
    with pytest.raises(ValueError,match='PREREGISTERED_SMALL_XY'):
        s.move_cube_home_only(model(),delta)
