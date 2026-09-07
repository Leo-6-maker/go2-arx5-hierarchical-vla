"""CPU regressions for the Stage40-local handoff schedule and call order."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from go2arx5_vla.contracts import TrajectoryChunkV1  # noqa: E402
import stage40_live_one_handoff as stage40  # noqa: E402
from stage40_live_one_handoff import HANDOFF_S, _build_stage40_executor  # noqa: E402


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

    initial = _chunk("stage40-q0", observation_timestamp=0.0)
    updated = _chunk("stage40-q1", observation_timestamp=HANDOFF_S, offset=0.01)

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

    handoff = executor.submit_update(
        updated,
        receive_timestamp=HANDOFF_S,
        execute_timestamp=HANDOFF_S,
        current_achieved_position=[0.21, 0.0, 0.4],
        current_achieved_orientation=[0.0, 0.0, 0.0, 1.0],
        previous_predicted_position=[0.2 + 0.08, 0.0, 0.4],
        previous_predicted_orientation=[0.0, 0.0, 0.0, 1.0],
        executed_prefix_steps=actual_native_steps,
    )

    assert len(executor.updates) == 2
    assert handoff.prefix_steps == actual_native_steps
    assert handoff.handoff.executed_prefix_steps == actual_native_steps
    assert handoff.provenance["handoff_target_s"] == HANDOFF_S
    assert handoff.physics_reset is False
    assert handoff.physics_teleport is False
    assert handoff.policy_history_reset is False


class _FakeTensor:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, _index):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.value)

    def item(self):
        return float(self.value)


class _FakeTrajectoryBatch:
    def __init__(self):
        self.L = _FakeTensor(1.95)

    def s_ref(self, _sim_time):
        return _FakeTensor(1.95)


class _FakeNative:
    def __init__(self, events):
        self.events = events
        self.dt = 0.05
        self.stage21_anchor_world = _FakeTensor(np.zeros(3, dtype=np.float64))
        self.traj_sim_time = _FakeTensor(0.0)
        self.traj_s = _FakeTensor(0.0)
        self.traj_batch = _FakeTrajectoryBatch()
        self.root_states = object()
        self.dof_state = object()
        self.episode_length_buf = _FakeTensor(0.0)
        self.state_hash = "same-physics-state"
        self.stage21_wbc_forward_count = 0

    def initial_bind_reference(self, chunk):
        self.events.append(f"initial_bind:{chunk.chunk_id}")

    def load_reference_chunk(self, chunk):
        self.events.append(f"load_reference:{chunk.chunk_id}")
        return {"native_reference_update": "REFERENCE_ONLY"}


def test_stage40_run_live_orders_initial_bind_before_first_wbc_and_one_handoff() -> None:
    """Exercise the real run_live call order with a fake native/runtime boundary."""

    events: list[str] = []
    q0 = _chunk("stage40-q0", observation_timestamp=0.0)
    q1 = _chunk("stage40-q1", observation_timestamp=HANDOFF_S, offset=0.01)
    native = _FakeNative(events)
    actors = [None, None, SimpleNamespace(__version__="cpu-test"), "cpu"]
    request_index = {"value": 0}

    def fake_build_native(_args, _trajectory):
        return native, SimpleNamespace(close=lambda: events.append("env_close")), actors, [], None

    def fake_request_model(**kwargs):
        request_id = kwargs["request_id"]
        events.append(f"model_request:{request_id.rsplit('-', 1)[-1]}")
        case = {
            "request_id": request_id,
            "safety_gate": {"decision": "ACCEPT"},
            "binding": {"preview_finite": True},
        }
        chunk = q0 if request_index["value"] == 0 else q1
        request_index["value"] += 1
        return chunk, np.zeros((40, 10), dtype=np.float32), case

    def fake_step_rows(native_arg, _env, _actors, *, chunk, sim_time, steps, update_count):
        del _env, _actors
        events.append(f"wbc_forward:{update_count}")
        native_arg.traj_sim_time = _FakeTensor(sim_time + float(steps) * native_arg.dt)
        return [{"done": False, "reset_buf": False, "termination_code": "NONE"}], float(
            native_arg.traj_sim_time.item()
        ), None

    def fake_current_eef(_native):
        return np.asarray([0.2, 0.0, 0.4]), np.asarray([0.0, 0.0, 0.0, 1.0])

    args = SimpleNamespace(
        mode="live-seed",
        seed=4242,
        output_dir=ROOT / "_stage40_prefix_test_output",
        mailbox=ROOT / "_stage40_prefix_test_mailbox",
        response_timeout_s=1.0,
        structured_stats=ROOT / "unused_stats.json",
    )

    with patch.object(stage40, "_build_native", fake_build_native), patch.object(
        stage40, "_request_model", fake_request_model
    ), patch.object(stage40, "_step_rows", fake_step_rows), patch.object(
        stage40, "_current_eef", fake_current_eef
    ), patch.object(stage40, "_tensor_hash", lambda _value: native.state_hash), patch.object(
        stage40, "_write_json", lambda _path, _value: None
    ):
        result = stage40.run_live(args, trajectory={})

    assert result["status"] == "PASS"
    episode = result["episodes"][0]
    assert episode["status"] == "PASS"
    handoff = episode["handoff"]
    assert handoff["single_reference_handoff"] is True
    assert handoff["actual_native_control_steps"] == 20
    assert handoff["executor"]["handoff"]["executed_prefix_steps"] == 20
    assert handoff["physics_state_unchanged"] is True
    assert handoff["physics_reset"] is False
    assert handoff["policy_history_reset"] is False
    assert events.index("initial_bind:stage40-q0") < events.index("wbc_forward:0")
    assert events.index("wbc_forward:0") < events.index("model_request:q1")
    assert events.index("model_request:q1") < events.index("load_reference:stage40-q1")
    assert events.index("load_reference:stage40-q1") < events.index("wbc_forward:1")
    assert events.count("load_reference:stage40-q1") == 1
    assert native.stage21_wbc_forward_count == 0
