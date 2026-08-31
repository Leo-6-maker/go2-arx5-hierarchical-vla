import numpy as np

from go2arx5_vla.contracts import (
    EpisodeReferenceBindingV1,
    PlacementProfileV1,
    TrajectoryChunkV1,
    TrajectoryShapeV1,
)
from go2arx5_vla.trajectory.relative_se3 import pose_from_position_quaternion
from go2arx5_vla.wbc.native.reference_binding import (
    NativeAnchorInputs,
    native_translation_anchor,
)


def _chunk():
    return TrajectoryChunkV1(
        chunk_id="shape-test",
        source="unit-test",
        observation_timestamp=10.0,
        point_timestamps=np.array([10.0, 10.1, 10.2]),
        eef_position=np.array([[0.5, 0.0, 0.4], [0.55, 0.0, 0.4], [0.55, 0.1, 0.4]]),
        eef_orientation=np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)),
        source_frame="canonical_world",
        tcp_semantic_id="test_tcp",
        validity_start=10.0,
        validity_end=10.2,
    )


def test_shape_is_relative_to_initial_eef_and_preserves_timing():
    shape = TrajectoryShapeV1.from_chunk(_chunk(), parent_id="e1-parent", family="translation")
    assert shape.point_count == 3
    assert np.allclose(shape.point_timestamps, [0.0, 0.1, 0.2])
    assert np.allclose(shape.task_poses[0], np.eye(4))
    assert np.allclose(shape.position[1], [0.05, 0.0, 0.0])
    assert shape.parent_id == "e1-parent"


def test_episode_binding_composes_anchor_and_task_transform():
    shape = TrajectoryShapeV1.from_chunk(_chunk())
    world_anchor = pose_from_position_quaternion([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
    anchor_task = pose_from_position_quaternion([0.1, 0.2, 0.3], [0.0, 0.0, 0.0, 1.0])
    identity = np.eye(4)
    binding = EpisodeReferenceBindingV1(
        episode_id="reset-0",
        reset_timestamp=20.0,
        native_base_pose=identity,
        native_shoulder_pose=identity,
        native_reach_anchor_pose=world_anchor,
        native_eef_semantic_id="x5_link6",
        tcp_semantic_id="test_tcp",
        T_world_anchor=world_anchor,
        T_anchor_task=anchor_task,
        T_link_tcp=identity,
        source_shape=shape,
        placement_profile=PlacementProfileV1("p0", "CENTERED"),
        native_generator_reference_semantics="TEST",
        policy_reference_frame="native_heading_frame",
    )
    expected = world_anchor @ anchor_task @ shape.task_poses
    assert np.allclose(binding.T_world_eef, expected)
    assert np.allclose(binding.point_timestamps, [20.0, 20.1, 20.2])
    assert binding.bound_chunk().point_count == 3


def test_native_translation_anchor_matches_reset_formula():
    result = native_translation_anchor(
        NativeAnchorInputs(
            base_position=np.array([1.0, 2.0, 0.5]),
            base_quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            arm_mount_translation_body=np.array([0.2, 0.0, 0.4]),
            anchor_offset_body=np.array([0.3, 0.0, 0.4]),
            rho_star=0.5,
            reach_radius_m=1.0,
        )
    )
    direction = np.array([0.3, 0.0, 0.4]) / 0.5
    expected = np.array([1.0, 2.0, 0.5]) + np.array([0.2, 0.0, 0.4]) + 0.5 * direction
    assert np.allclose(result["shoulder_position"], [1.2, 2.0, 0.9])
    assert np.allclose(result["anchor_position"], expected)
    assert result["anchor_rotation_semantics"] == "WORLD_ALIGNED_TRANSLATION_ONLY"


def test_native_binding_keeps_world_aligned_shape_translation_explicit():
    shape = TrajectoryShapeV1.from_task_poses(
        trajectory_id="world-aligned",
        parent_id="fixture",
        point_timestamps=[0.0, 1.0],
        task_poses=pose_from_position_quaternion(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
        ),
        task_frame_semantics="NATIVE_WORLD_ALIGNED_TRANSLATION_ONLY",
        family="fixture",
    )
    anchor = pose_from_position_quaternion([1.0, 2.0, 3.0], [0.0, 0.0, 0.7071067811865476, 0.7071067811865476])
    task = pose_from_position_quaternion([0.1, 0.2, 0.3], [0.0, 0.0, 0.7071067811865476, 0.7071067811865476])
    binding = EpisodeReferenceBindingV1(
        episode_id="reset-world-aligned",
        reset_timestamp=0.0,
        native_base_pose=np.eye(4),
        native_shoulder_pose=np.eye(4),
        native_reach_anchor_pose=anchor,
        native_eef_semantic_id="x5_link6",
        tcp_semantic_id="gripper_center",
        T_world_anchor=anchor,
        T_anchor_task=task,
        T_link_tcp=np.eye(4),
        source_shape=shape,
        placement_profile=PlacementProfileV1("p-world", "CENTERED"),
        native_generator_reference_semantics="NATIVE_TRANSLATION_ONLY",
        policy_reference_frame="NATIVE_WORLD_ALIGNED",
        binding_semantics="NATIVE_TRANSLATION_ONLY_WORLD_ALIGNED",
    )
    # The native loader adds the shape translation in world coordinates; it
    # does not rotate it by the anchor/task orientation.
    assert np.allclose(binding.T_world_eef[1, :3, 3], [2.1, 2.2, 3.3])
