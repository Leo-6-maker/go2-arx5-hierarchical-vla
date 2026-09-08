"""Transport-independent reference hot-swap contracts for Stage21.

The executor in this module does not step a simulator and does not own a
network transport.  It records the boundary at which a new external EEF
trajectory replaces the previous one.  A native runner may use the resulting
``ReferenceUpdateV1`` to update only its reference buffers while preserving
physics state, policy history, and episode time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from ..contracts.trajectory import TrajectoryChunkV1
from ..trajectory.so3 import geodesic_angle, quat_to_matrix
from .chunk_lifecycle import ChunkLifecycleConfig, ChunkLifecycleStatus, ChunkLifecycleTracker


def _pose_vector(name: str, value: Any, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape [{width}]")
    return array.copy()


def _rotation_discontinuity(q0: np.ndarray, q1: np.ndarray) -> float:
    return float(geodesic_angle(quat_to_matrix(q0), quat_to_matrix(q1)))


@dataclass(frozen=True)
class ChunkHandoffV1:
    """Measured discontinuities at one reference replacement boundary."""

    previous_chunk_id: Optional[str]
    new_chunk_id: str
    executed_prefix_steps: int
    previous_predicted_position: Optional[np.ndarray]
    previous_predicted_orientation: Optional[np.ndarray]
    new_first_position: np.ndarray
    new_first_orientation: np.ndarray
    current_achieved_position: Optional[np.ndarray]
    current_achieved_orientation: Optional[np.ndarray]
    previous_prediction_to_new_first_position_m: Optional[float]
    previous_prediction_to_new_first_rotation_rad: Optional[float]
    achieved_to_new_first_position_m: Optional[float]
    achieved_to_new_first_rotation_rad: Optional[float]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.new_chunk_id:
            raise ValueError("new_chunk_id is required")
        if int(self.executed_prefix_steps) < 0:
            raise ValueError("executed_prefix_steps must be non-negative")
        new_position = _pose_vector("new_first_position", self.new_first_position, 3)
        new_orientation = _pose_vector("new_first_orientation", self.new_first_orientation, 4)
        new_orientation /= max(float(np.linalg.norm(new_orientation)), 1e-12)
        object.__setattr__(self, "new_first_position", new_position)
        object.__setattr__(self, "new_first_orientation", new_orientation)
        for prefix, width in (("previous_predicted", 3), ("current_achieved", 3)):
            position = getattr(self, f"{prefix}_position")
            if position is not None:
                object.__setattr__(self, f"{prefix}_position", _pose_vector(f"{prefix}_position", position, width))
        for prefix in ("previous_predicted", "current_achieved"):
            orientation = getattr(self, f"{prefix}_orientation")
            if orientation is not None:
                normalized = _pose_vector(f"{prefix}_orientation", orientation, 4)
                normalized /= max(float(np.linalg.norm(normalized)), 1e-12)
                object.__setattr__(self, f"{prefix}_orientation", normalized)
        for name in (
            "previous_prediction_to_new_first_position_m",
            "previous_prediction_to_new_first_rotation_rad",
            "achieved_to_new_first_position_m",
            "achieved_to_new_first_rotation_rad",
        ):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be finite and non-negative when present")
        object.__setattr__(self, "provenance", dict(self.provenance))

    def as_dict(self) -> dict[str, Any]:
        def optional(value: Optional[np.ndarray]) -> Any:
            return None if value is None else value.tolist()

        return {
            "schema": "ChunkHandoffV1",
            "previous_chunk_id": self.previous_chunk_id,
            "new_chunk_id": self.new_chunk_id,
            "executed_prefix_steps": int(self.executed_prefix_steps),
            "previous_predicted_position": optional(self.previous_predicted_position),
            "previous_predicted_orientation": optional(self.previous_predicted_orientation),
            "new_first_position": self.new_first_position.tolist(),
            "new_first_orientation": self.new_first_orientation.tolist(),
            "current_achieved_position": optional(self.current_achieved_position),
            "current_achieved_orientation": optional(self.current_achieved_orientation),
            "previous_prediction_to_new_first_position_m": self.previous_prediction_to_new_first_position_m,
            "previous_prediction_to_new_first_rotation_rad": self.previous_prediction_to_new_first_rotation_rad,
            "achieved_to_new_first_position_m": self.achieved_to_new_first_position_m,
            "achieved_to_new_first_rotation_rad": self.achieved_to_new_first_rotation_rad,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class ReferenceUpdateV1:
    """One accepted external-reference replacement.

    ``physics_reset`` and ``physics_teleport`` are explicit negative
    assertions.  They make it difficult for a simulator adapter to silently
    turn a reference update into a new episode or a state teleport.
    """

    update_id: int
    chunk: TrajectoryChunkV1
    previous_chunk_id: Optional[str]
    observation_timestamp: float
    receive_timestamp: float
    execute_timestamp: float
    lifecycle_status: ChunkLifecycleStatus
    prefix_steps: int
    handoff: ChunkHandoffV1
    physics_reset: bool = False
    physics_teleport: bool = False
    policy_history_reset: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.update_id) < 0:
            raise ValueError("update_id must be non-negative")
        for name in ("observation_timestamp", "receive_timestamp", "execute_timestamp"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if int(self.prefix_steps) < 1:
            raise ValueError("prefix_steps must be positive")
        if self.lifecycle_status in {
            ChunkLifecycleStatus.INVALID,
            ChunkLifecycleStatus.OUT_OF_ORDER,
            ChunkLifecycleStatus.STALE,
        }:
            raise ValueError(f"cannot create an accepted update with status {self.lifecycle_status.value}")
        if self.physics_reset or self.physics_teleport or self.policy_history_reset:
            raise ValueError("accepted Stage21 updates must preserve physics and policy history")
        if self.previous_chunk_id == self.chunk.chunk_id:
            raise ValueError("new update must have a new chunk_id")
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "ReferenceUpdateV1",
            "update_id": int(self.update_id),
            "chunk_id": self.chunk_id,
            "previous_chunk_id": self.previous_chunk_id,
            "observation_timestamp": float(self.observation_timestamp),
            "receive_timestamp": float(self.receive_timestamp),
            "execute_timestamp": float(self.execute_timestamp),
            "lifecycle_status": self.lifecycle_status.value,
            "prefix_steps": int(self.prefix_steps),
            "handoff": self.handoff.as_dict(),
            "physics_reset": bool(self.physics_reset),
            "physics_teleport": bool(self.physics_teleport),
            "policy_history_reset": bool(self.policy_history_reset),
            "provenance": dict(self.provenance),
        }


class RecedingHorizonExecutorV1:
    """Validate and record a fixed-prefix reference-update schedule.

    This class is intentionally simulator-agnostic.  It accepts FRESH or
    AGING chunks from the abstract lifecycle tracker, rejects stale and
    out-of-order chunks, and measures handoff discontinuities without
    modifying any trajectory values.
    """

    DEFAULT_PREFIX_STEPS = (4, 6, 8, 10)

    def __init__(
        self,
        *,
        prefix_steps: int,
        allowed_prefix_steps: tuple[int, ...] = DEFAULT_PREFIX_STEPS,
        lifecycle_config: ChunkLifecycleConfig | None = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> None:
        allowed = tuple(int(value) for value in allowed_prefix_steps)
        if int(prefix_steps) not in allowed:
            raise ValueError(f"prefix_steps must be one of {allowed}")
        self.prefix_steps = int(prefix_steps)
        self.allowed_prefix_steps = allowed
        self.lifecycle = ChunkLifecycleTracker(lifecycle_config)
        self._updates: list[ReferenceUpdateV1] = []
        self._current_chunk: Optional[TrajectoryChunkV1] = None
        self._provenance = {} if provenance is None else dict(provenance)

    @property
    def current_chunk(self) -> Optional[TrajectoryChunkV1]:
        return self._current_chunk

    @property
    def updates(self) -> tuple[ReferenceUpdateV1, ...]:
        return tuple(self._updates)

    @property
    def latest_update(self) -> Optional[ReferenceUpdateV1]:
        return self._updates[-1] if self._updates else None

    def submit_initial(
        self,
        chunk: TrajectoryChunkV1,
        *,
        receive_timestamp: float,
        execute_timestamp: Optional[float] = None,
        current_achieved_position: Any | None = None,
        current_achieved_orientation: Any | None = None,
    ) -> ReferenceUpdateV1:
        if self._current_chunk is not None:
            raise RuntimeError("initial chunk has already been accepted")
        return self._submit(
            chunk,
            receive_timestamp=receive_timestamp,
            execute_timestamp=execute_timestamp,
            current_achieved_position=current_achieved_position,
            current_achieved_orientation=current_achieved_orientation,
            previous_predicted_position=None,
            previous_predicted_orientation=None,
            executed_prefix_steps=0,
            previous_chunk_id=None,
        )

    def submit_update(
        self,
        chunk: TrajectoryChunkV1,
        *,
        receive_timestamp: float,
        execute_timestamp: Optional[float] = None,
        current_achieved_position: Any,
        current_achieved_orientation: Any,
        previous_predicted_position: Any,
        previous_predicted_orientation: Any,
        executed_prefix_steps: Optional[int] = None,
    ) -> ReferenceUpdateV1:
        if self._current_chunk is None or self.latest_update is None:
            raise RuntimeError("submit_initial must succeed before submit_update")
        prefix = self.prefix_steps if executed_prefix_steps is None else int(executed_prefix_steps)
        if prefix < 1 or prefix > self._current_chunk.point_count:
            raise ValueError("executed_prefix_steps must be inside the previous chunk")
        previous_id = self._current_chunk.chunk_id
        return self._submit(
            chunk,
            receive_timestamp=receive_timestamp,
            execute_timestamp=execute_timestamp,
            current_achieved_position=current_achieved_position,
            current_achieved_orientation=current_achieved_orientation,
            previous_predicted_position=previous_predicted_position,
            previous_predicted_orientation=previous_predicted_orientation,
            executed_prefix_steps=prefix,
            previous_chunk_id=previous_id,
        )

    def sample_lifecycle(self, *, execute_timestamp: float):
        """Sample age for the current chunk without accepting a new one."""

        return self.lifecycle.sample(execute_timestamp=execute_timestamp)

    def _submit(
        self,
        chunk: TrajectoryChunkV1,
        *,
        receive_timestamp: float,
        execute_timestamp: Optional[float],
        current_achieved_position: Any | None,
        current_achieved_orientation: Any | None,
        previous_predicted_position: Any | None,
        previous_predicted_orientation: Any | None,
        executed_prefix_steps: int,
        previous_chunk_id: Optional[str],
    ) -> ReferenceUpdateV1:
        event = self.lifecycle.submit(chunk, receive_timestamp=float(receive_timestamp))
        if event.status in {
            ChunkLifecycleStatus.INVALID,
            ChunkLifecycleStatus.OUT_OF_ORDER,
            ChunkLifecycleStatus.STALE,
        }:
            raise ValueError(f"chunk lifecycle rejected update: {event.status.value}: {event.message}")
        execute = float(receive_timestamp if execute_timestamp is None else execute_timestamp)
        if not np.isfinite(execute):
            raise ValueError("execute_timestamp must be finite")
        if event.timing is None:
            raise ValueError("accepted lifecycle event has no timing record")

        new_position = np.asarray(chunk.eef_position[0], dtype=np.float64)
        new_orientation = np.asarray(chunk.eef_orientation[0], dtype=np.float64)
        achieved_position = None
        achieved_orientation = None
        achieved_position_error = None
        achieved_rotation_error = None
        if current_achieved_position is not None or current_achieved_orientation is not None:
            if current_achieved_position is None or current_achieved_orientation is None:
                raise ValueError("current achieved position and orientation must be supplied together")
            achieved_position = _pose_vector("current_achieved_position", current_achieved_position, 3)
            achieved_orientation = _pose_vector("current_achieved_orientation", current_achieved_orientation, 4)
            achieved_orientation /= max(float(np.linalg.norm(achieved_orientation)), 1e-12)
            achieved_position_error = float(np.linalg.norm(new_position - achieved_position))
            achieved_rotation_error = _rotation_discontinuity(achieved_orientation, new_orientation)

        previous_position = None
        previous_orientation = None
        previous_position_error = None
        previous_rotation_error = None
        if previous_predicted_position is not None or previous_predicted_orientation is not None:
            if previous_predicted_position is None or previous_predicted_orientation is None:
                raise ValueError("previous predicted position and orientation must be supplied together")
            previous_position = _pose_vector("previous_predicted_position", previous_predicted_position, 3)
            previous_orientation = _pose_vector("previous_predicted_orientation", previous_predicted_orientation, 4)
            previous_orientation /= max(float(np.linalg.norm(previous_orientation)), 1e-12)
            previous_position_error = float(np.linalg.norm(new_position - previous_position))
            previous_rotation_error = _rotation_discontinuity(previous_orientation, new_orientation)

        handoff = ChunkHandoffV1(
            previous_chunk_id=previous_chunk_id,
            new_chunk_id=chunk.chunk_id,
            executed_prefix_steps=int(executed_prefix_steps),
            previous_predicted_position=previous_position,
            previous_predicted_orientation=previous_orientation,
            new_first_position=new_position,
            new_first_orientation=new_orientation,
            current_achieved_position=achieved_position,
            current_achieved_orientation=achieved_orientation,
            previous_prediction_to_new_first_position_m=previous_position_error,
            previous_prediction_to_new_first_rotation_rad=previous_rotation_error,
            achieved_to_new_first_position_m=achieved_position_error,
            achieved_to_new_first_rotation_rad=achieved_rotation_error,
            provenance={
                "metric": "raw_pose_difference_no_smoothing_or_correction",
                "same_physics_episode": True,
            },
        )
        update = ReferenceUpdateV1(
            update_id=len(self._updates),
            chunk=chunk,
            previous_chunk_id=previous_chunk_id,
            observation_timestamp=float(chunk.observation_timestamp),
            receive_timestamp=float(receive_timestamp),
            execute_timestamp=execute,
            lifecycle_status=event.status,
            prefix_steps=self.prefix_steps,
            handoff=handoff,
            provenance={
                **self._provenance,
                "executor": "RecedingHorizonExecutorV1",
                "reference_only_update": True,
                "physics_reset": False,
                "physics_teleport": False,
                "policy_history_reset": False,
            },
        )
        self._updates.append(update)
        self._current_chunk = chunk
        return update
