"""Dynamic CPU regression for the Stage40-local handoff schedule."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from go2arx5_vla.contracts import TrajectoryChunkV1  # noqa: E402
from stage40_live_one_handoff import (  # noqa: E402
    HANDOFF_S,
    _build_stage40_executor,
)


def _chunk(chunk_id: str, observation_timestamp: float, offset: float = 0.0) -> TrajectoryChunkV1:
    timestamps = observation_timestamp + np.arange(40, dtype=np.float64) * 0.05
    position = np.stack(
        [0.2 + offset + np.linspace(0.0, 0.08, 40), np.zeros(40), np.full(40, 0.4)],
        axis=-1,
    )
    quaternion = np.tile([0.0, 0.0, 0.0, 1.0], (40, 1))
    return TrajectoryChunkV1(
        chunk_id=chunk_id,
        source="stage40-prefix-regression",
        observation_timestamp=observation_timestamp,
        point_timestamps=timestamps,
        eef_position=position,
        eef_orientation=quaternion,
        source_frame="NATIVE_WBC_RELATIVE_REFERENCE",
        tcp_semantic_id="x5_link6_gripper_center",
        validity_start=float(timestamps[0]),
        validity_end=float(timestamps[-1]),
        gripper=np.full(40, 0.5),
    )


def test_stage40_executor_uses_actual_native_prefix_and_one_handoff() -> None:
    native_dt_s = 0.05
    executor, actual_native_steps = _build_stage40_executor(native_dt_s=native_dt_s)
    assert actual_native_steps == 20
    assert executor.prefix_steps == actual_native_steps
    assert executor.allowed_prefix_steps == (actual_native_steps,)

    events: list[str] = []
    initial = _chunk("stage40-q0", observation_timestamp=0.0)
    updated = _chunk("stage40-q1", observation_timestamp=HANDOFF_S, offset=0.01)

    events.append("initial_bind")
    initial_update = executor.submit_initial(
        initial,
        receive_timestamp=0.0,
        execute_timestamp=0.0,
        current_achieved_position=[0.2, 0.0, 0.4],
        current_achieved_orientation=[0.0, 0.0, 0.0, 1.0],
    )
    assert initial_update.physics_reset is False
    assert initial_update.physics_teleport is False
    assert initial_update.policy_history_reset is False

    events.append("first_wbc_forward")
    executed_steps = sum(1 for _ in range(actual_native_steps))
    assert executed_steps == actual_native_steps

    events.append("q1_observation")
    handoff = executor.submit_update(
        updated,
        receive_timestamp=HANDOFF_S,
        execute_timestamp=HANDOFF_S,
        current_achieved_position=[0.2 + 0.01, 0.0, 0.4],
        current_achieved_orientation=[0.0, 0.0, 0.0, 1.0],
        previous_predicted_position=[0.2 + 0.08, 0.0, 0.4],
        previous_predicted_orientation=[0.0, 0.0, 0.0, 1.0],
        executed_prefix_steps=executed_steps,
    )
    events.append("reference_handoff")

    assert events.index("initial_bind") < events.index("first_wbc_forward")
    assert events.count("reference_handoff") == 1
    assert len(executor.updates) == 2  # initial bind plus exactly one update
    assert handoff.prefix_steps == actual_native_steps
    assert handoff.handoff.executed_prefix_steps == actual_native_steps
    assert handoff.provenance["handoff_target_s"] == HANDOFF_S
    assert handoff.physics_reset is False
    assert handoff.physics_teleport is False
    assert handoff.policy_history_reset is False

