import numpy as np
import pytest

from go2arx5_vla.contracts import TrajectoryChunkV1
from go2arx5_vla.data import StateV1
from go2arx5_vla.runtime import (
    ChunkLifecycleConfig,
    ChunkLifecycleStatus,
    RecedingHorizonExecutorV1,
    ReferenceUpdateV1,
)
from go2arx5_vla.simulation import (
    NativeSceneObservationV1,
    SameSceneTargetV1,
    compare_counterfactuals,
)
from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d_rows


def _chunk(chunk_id: str, observation: float = 0.0, offset: float = 0.0) -> TrajectoryChunkV1:
    timestamps = observation + np.arange(40, dtype=np.float64) * 0.05
    position = np.stack(
        [0.2 + offset + np.linspace(0.0, 0.08, 40), np.zeros(40), np.full(40, 0.4)],
        axis=-1,
    )
    quaternion = np.tile([0.0, 0.0, 0.0, 1.0], (40, 1))
    return TrajectoryChunkV1(
        chunk_id=chunk_id,
        source="stage21-test",
        observation_timestamp=observation,
        point_timestamps=timestamps,
        eef_position=position,
        eef_orientation=quaternion,
        source_frame="sim_world",
        tcp_semantic_id="x5_link6_gripper_center",
        validity_start=float(timestamps[0]),
        validity_end=float(timestamps[-1]),
        gripper=np.full(40, 0.5),
    )


def _state(value: float = 0.0) -> StateV1:
    return StateV1(
        eef_xyz=np.array([0.2 + value, 0.0, 0.4]),
        eef_rot6d_rows=rotation_matrix_to_rot6d_rows(np.eye(3)),
        gripper=np.array([0.5]),
    )


def _observation(observation_id: str, target_position: list[float], image_value: int) -> NativeSceneObservationV1:
    image = np.full((8, 8, 3), image_value, dtype=np.uint8)
    target = SameSceneTargetV1(
        target_id=observation_id,
        simulator_instance_id="sim-1",
        position_world_m=np.asarray(target_position),
    )
    return NativeSceneObservationV1(
        observation_id=observation_id,
        simulator_instance_id="sim-1",
        simulation_time_s=0.0,
        body_rgb=image,
        wrist_rgb=image.copy(),
        state=_state(),
        language="move the end effector to the red target",
        target=target,
        source_frame="sim_world",
        tcp_semantic_id="x5_link6_gripper_center",
        camera_provenance={"same_scene": True},
    )


def test_stage21_executor_freezes_prefix_and_preserves_physics_contract():
    executor = RecedingHorizonExecutorV1(prefix_steps=4, provenance={"mode": "GT"})
    initial = executor.submit_initial(
        _chunk("c0"),
        receive_timestamp=0.0,
        current_achieved_position=[0.2, 0.0, 0.4],
        current_achieved_orientation=[0.0, 0.0, 0.0, 1.0],
    )
    assert initial.lifecycle_status is ChunkLifecycleStatus.FRESH
    update = executor.submit_update(
        _chunk("c1", observation=0.2, offset=0.01),
        receive_timestamp=0.2,
        execute_timestamp=0.2,
        current_achieved_position=[0.21, 0.0, 0.4],
        current_achieved_orientation=[0.0, 0.0, 0.0, 1.0],
        previous_predicted_position=[0.208, 0.0, 0.4],
        previous_predicted_orientation=[0.0, 0.0, 0.0, 1.0],
    )
    assert update.prefix_steps == 4
    assert update.handoff.executed_prefix_steps == 4
    assert update.physics_reset is False
    assert update.physics_teleport is False
    assert update.policy_history_reset is False
    assert update.handoff.achieved_to_new_first_position_m is not None
    assert executor.current_chunk.chunk_id == "c1"


def test_stage21_executor_rejects_invalid_prefix_and_lifecycle_order():
    with pytest.raises(ValueError, match="prefix_steps"):
        RecedingHorizonExecutorV1(prefix_steps=5)

    executor = RecedingHorizonExecutorV1(
        prefix_steps=4,
        lifecycle_config=ChunkLifecycleConfig(fresh_age_s=0.1, stale_age_s=0.2),
    )
    executor.submit_initial(_chunk("c0"), receive_timestamp=0.0)
    with pytest.raises(ValueError, match="STALE"):
        executor.submit_update(
            _chunk("c1", observation=0.0),
            receive_timestamp=0.3,
            current_achieved_position=[0.2, 0.0, 0.4],
            current_achieved_orientation=[0.0, 0.0, 0.0, 1.0],
            previous_predicted_position=[0.2, 0.0, 0.4],
            previous_predicted_orientation=[0.0, 0.0, 0.0, 1.0],
        )
    with pytest.raises(ValueError, match="OUT_OF_ORDER"):
        executor.submit_update(
            _chunk("older", observation=0.0),
            receive_timestamp=-0.1,
            current_achieved_position=[0.2, 0.0, 0.4],
            current_achieved_orientation=[0.0, 0.0, 0.0, 1.0],
            previous_predicted_position=[0.2, 0.0, 0.4],
            previous_predicted_orientation=[0.0, 0.0, 0.0, 1.0],
        )


def test_stage21_lifecycle_exposes_fresh_aging_and_stale_states():
    tracker = ChunkLifecycleConfig(fresh_age_s=0.1, stale_age_s=0.2)
    executor = RecedingHorizonExecutorV1(prefix_steps=4, lifecycle_config=tracker)
    executor.submit_initial(_chunk("c0"), receive_timestamp=0.0)
    assert executor.sample_lifecycle(execute_timestamp=0.1).status is ChunkLifecycleStatus.FRESH
    assert executor.sample_lifecycle(execute_timestamp=0.15).status is ChunkLifecycleStatus.AGING
    assert executor.sample_lifecycle(execute_timestamp=0.2).status is ChunkLifecycleStatus.AGING
    assert executor.sample_lifecycle(execute_timestamp=0.21).status is ChunkLifecycleStatus.STALE


def test_reference_update_cannot_assert_a_physics_or_history_reset():
    executor = RecedingHorizonExecutorV1(prefix_steps=4)
    executor.submit_initial(_chunk("c0"), receive_timestamp=0.0)
    update = executor.submit_update(
        _chunk("c1", observation=0.05),
        receive_timestamp=0.05,
        current_achieved_position=[0.2, 0.0, 0.4],
        current_achieved_orientation=[0.0, 0.0, 0.0, 1.0],
        previous_predicted_position=[0.2, 0.0, 0.4],
        previous_predicted_orientation=[0.0, 0.0, 0.0, 1.0],
    )
    with pytest.raises(ValueError, match="preserve physics"):
        ReferenceUpdateV1(
            update_id=update.update_id,
            chunk=update.chunk,
            previous_chunk_id="c0",
            observation_timestamp=update.observation_timestamp,
            receive_timestamp=update.receive_timestamp,
            execute_timestamp=update.execute_timestamp,
            lifecycle_status=update.lifecycle_status,
            prefix_steps=4,
            handoff=update.handoff,
            physics_reset=True,
        )


def test_same_scene_counterfactual_has_same_state_and_changed_native_image():
    a = _observation("a", [0.30, 0.10, 0.58], 10)
    b = _observation("b", [0.30, -0.20, 0.58], 80)
    result = compare_counterfactuals(a, b)
    assert result.same_state is True
    assert result.same_language is True
    assert result.different_target_location is True
    assert result.different_image is True
    assert result.body_image_changed is True
    assert result.wrist_image_changed is True
    assert result.state_max_abs_error == 0.0


def test_same_scene_contract_rejects_cross_simulator_target():
    with pytest.raises(ValueError, match="share simulator_instance_id"):
        NativeSceneObservationV1(
            observation_id="bad",
            simulator_instance_id="sim-1",
            simulation_time_s=0.0,
            body_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            wrist_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
            state=_state(),
            language="reach",
            target=SameSceneTargetV1("target", "sim-2", [0.3, 0.0, 0.5]),
            source_frame="sim_world",
            tcp_semantic_id="tcp",
        )
