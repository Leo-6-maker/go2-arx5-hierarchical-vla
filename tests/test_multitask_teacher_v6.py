import numpy as np

from go2arx5_vla.world_model.multitask_teacher_v6 import MultitaskTeacher, TRANSFER_CLEARANCE_M


def pose(x, y, z):
    transform = np.eye(4)
    transform[:3, 3] = [x, y, z]
    return transform


def test_carry_raises_before_crossing_a_detected_box_and_preserves_the_clear_height():
    geometry = dict(direction_world_XY=[1., 0.], object_half_sizes={'red': [.02] * 3, 'blue': [.02] * 3})
    teacher = MultitaskTeacher(
        dict(object_pose=pose(0., 0., .02), tcp=pose(0., 0., .04), base_pose=np.eye(4)),
        dict(family='F2', mode='M_FREE'), geometry, 'PLACE-blue-right',
    )
    teacher.lift_origin = np.array([0., 0., .04])
    sample = dict(tcp=pose(0., 0., .066), native_grasp=True,
                  objects={'blue': {'pose': pose(0., 0., .044)}, 'red': {'pose': pose(.08, 0., .02)}})
    zone = np.array([.15, 0.])

    raise_goal, lower_clear = teacher.carry_goal(sample, zone)
    np.testing.assert_allclose(raise_goal, [0., 0., .082 + TRANSFER_CLEARANCE_M])
    assert not lower_clear

    sample['tcp'] = pose(.005, 0., raise_goal[2])
    sample['objects']['blue']['pose'] = pose(.005, 0., .061 + TRANSFER_CLEARANCE_M)
    transit, lower_clear = teacher.carry_goal(sample, zone)
    np.testing.assert_allclose(transit, [.15, 0., raise_goal[2]])
    assert lower_clear

    sample['objects']['blue']['pose'] = pose(.08, 0., .061 + TRANSFER_CLEARANCE_M)
    _, lower_clear = teacher.carry_goal(sample, zone)
    assert not lower_clear

    sample['objects']['blue']['pose'] = pose(.15, 0., .061 + TRANSFER_CLEARANCE_M)
    goal, lower_clear = teacher.carry_goal(sample, zone)
    assert lower_clear and np.isclose(goal[2], raise_goal[2])


def test_clear_route_retains_the_original_carry_height():
    geometry = dict(direction_world_XY=[1., 0.], object_half_sizes={'red': [.02] * 3, 'blue': [.02] * 3})
    teacher = MultitaskTeacher(
        dict(object_pose=pose(0., 0., .02), tcp=pose(0., 0., .04), base_pose=np.eye(4)),
        dict(family='F1', mode='M_FREE'), geometry, 'PLACE-blue-left',
    )
    teacher.lift_origin = np.array([0., 0., .04])
    sample = dict(tcp=pose(0., 0., .066), native_grasp=True,
                  objects={'blue': {'pose': pose(0., 0., .044)}, 'red': {'pose': pose(.08, .05, .02)}})

    goal, lower_clear = teacher.carry_goal(sample, np.array([.15, 0.]))

    np.testing.assert_allclose(goal, [.15, 0., .066])
    assert lower_clear
