import numpy as np

from go2arx5_vla.world_model.multitask_teacher_v6 import (
    F1_RED_GRASP_RAISE_M, MultitaskTeacher, TRANSFER_CLEARANCE_M,
)


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



def test_f1_red_grasp_raise_is_teacher_only_and_scoped_to_selected_instructions():
    initial = dict(object_pose=pose(0., 0., .02), tcp=pose(0., 0., .04), base_pose=np.eye(4))
    geometry = dict(direction_world_XY=[1., 0.])

    def grasp(family, instruction):
        teacher = MultitaskTeacher(initial, dict(family=family, mode='M_FREE'), geometry, instruction)
        return teacher.grasp_goal()

    other_f1 = grasp('F1', 'HOLD-blue')
    np.testing.assert_allclose(grasp('F1', 'HOLD-red') - other_f1, [0., 0., F1_RED_GRASP_RAISE_M])
    np.testing.assert_allclose(grasp('F1', 'PLACE-red-right') - other_f1, [0., 0., F1_RED_GRASP_RAISE_M])
    np.testing.assert_array_equal(grasp('F1', 'PLACE-red-left'), other_f1)
    np.testing.assert_allclose(grasp('F2', 'HOLD-red') - other_f1, [0., 0., .008])




def test_distal_carry_waits_for_measured_clearance_and_checks_the_lowering_corridor():
    geometry = dict(direction_world_XY=[1., 0.], table_top_world_z=0.,
                    object_half_sizes={'red': [.02] * 3, 'blue': [.02] * 3})
    teacher = MultitaskTeacher(
        dict(object_pose=pose(0., 0., .02), tcp=pose(0., 0., .04), base_pose=np.eye(4)),
        dict(family='F1', mode='M_FREE'), geometry, 'PLACE-red-left',
    )
    teacher.lift_origin = np.array([0., 0., .04])
    zone = np.array([.15, 0.])
    sample = dict(tcp=pose(0., 0., .066), native_grasp=True,
                  objects={'red': {'pose': pose(0., 0., .044)},
                           'blue': {'pose': pose(.08, .08, .02)}})
    gaps = iter([.001, .009, .012, -.001, .012, .003])
    teacher.landing = None
    teacher.distal_clearance = lambda sample, zone, *, support_z=None: next(gaps)

    raise_goal, can_lower = teacher.carry_goal(sample, zone)
    np.testing.assert_allclose(raise_goal, [0., 0., .102])
    assert not can_lower and teacher.distal_clearance_evidence['waiting_for_measured_clearance']

    sample['tcp'] = pose(0., 0., .102)
    transit_goal, can_lower = teacher.carry_goal(sample, zone)
    np.testing.assert_allclose(transit_goal, [.15, 0., .102])
    assert not can_lower

    sample['tcp'] = pose(.15, 0., .102)
    sample['objects']['red']['pose'] = pose(.15, 0., .080)
    assert not teacher.carry_goal(sample, zone)[1]
    assert teacher.carry_goal(sample, zone)[1]



def test_oriented_landing_preserves_measured_yaw_until_the_turn_corridor_is_clear():
    geometry = dict(direction_world_XY=[1., 0.], table_top_world_z=0.,
        object_half_sizes={'red': [.02] * 3, 'blue': [.02] * 3},
        zones={'left': dict(center_world_XY=[.15, 0.], half_size_XY=[.035, .035])})
    teacher = MultitaskTeacher(
        dict(object_pose=pose(0., 0., .02), tcp=pose(0., 0., .04), base_pose=np.eye(4)),
        dict(family='F1', mode='M_FREE'), geometry, 'PLACE-red-left',
    )
    teacher.lift_origin = np.array([0., 0., .04])
    zone = np.array([.158, .008])
    sample = dict(tcp=pose(0., 0., .066), native_grasp=True,
                  objects={'red': {'pose': pose(0., 0., .044)},
                           'blue': {'pose': pose(.08, .08, .02)}},
                  object_pose=pose(0., 0., .044))
    gaps = iter([.001, .010, .010, .004, .010, .004])
    teacher.distal_clearance = lambda sample, zone, **kwargs: next(gaps)

    raised, can_lower = teacher.carry_goal(sample, zone)
    assert not can_lower and teacher.landing_rotation is None
    np.testing.assert_allclose(raised, [0., 0., .102])

    assert not teacher.carry_goal(sample, zone)[1]
    assert np.isclose(np.arctan2(teacher.rotation[1, 0], teacher.rotation[0, 0]), np.deg2rad(-20.))

    sample.update(tcp=pose(*zone, .102), object_pose=pose(*zone, .08))
    sample['objects']['red']['pose'] = sample['object_pose']
    assert not teacher.carry_goal(sample, zone)[1]
    sample['tcp'][:3, :3] = teacher.landing_rotation
    assert teacher.carry_goal(sample, zone)[1]
