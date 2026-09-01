"""Explicit chunk timing/lifecycle semantics without a transport dependency."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from ..contracts.trajectory import TrajectoryChunkV1


class ChunkLifecycleStatus(str, Enum):
    FRESH = "FRESH"
    AGING = "AGING"
    STALE = "STALE"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ChunkLifecycleConfig:
    """Age thresholds; they are interface timing contracts, not robot limits."""

    fresh_age_s: float = 0.20
    stale_age_s: float = 0.75

    def __post_init__(self) -> None:
        if not np.isfinite(self.fresh_age_s) or not np.isfinite(self.stale_age_s):
            raise ValueError("chunk age thresholds must be finite")
        if self.fresh_age_s < 0.0 or self.stale_age_s <= self.fresh_age_s:
            raise ValueError("require 0 <= fresh_age_s < stale_age_s")


@dataclass(frozen=True)
class ChunkTimingV1:
    chunk_id: str
    observation_timestamp: float
    receive_timestamp: float
    execute_timestamp: Optional[float]
    age_s: float
    status: ChunkLifecycleStatus

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "observation_timestamp": self.observation_timestamp,
            "receive_timestamp": self.receive_timestamp,
            "execute_timestamp": self.execute_timestamp,
            "age_s": self.age_s,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class ChunkLifecycleEvent:
    status: ChunkLifecycleStatus
    timing: Optional[ChunkTimingV1]
    message: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "timing": None if self.timing is None else self.timing.as_dict(),
            "message": self.message,
            "provenance": dict(self.provenance),
        }


class ChunkLifecycleTracker:
    """Track one latest chunk and make freshness/order decisions explicitly."""

    def __init__(self, config: ChunkLifecycleConfig | None = None) -> None:
        self.config = ChunkLifecycleConfig() if config is None else config
        self._latest: Optional[ChunkTimingV1] = None
        self._latest_chunk: Optional[TrajectoryChunkV1] = None

    @property
    def latest_chunk(self) -> Optional[TrajectoryChunkV1]:
        return self._latest_chunk

    @property
    def latest_timing(self) -> Optional[ChunkTimingV1]:
        return self._latest

    def submit(self, chunk: TrajectoryChunkV1, *, receive_timestamp: float) -> ChunkLifecycleEvent:
        receive = float(receive_timestamp)
        if not np.isfinite(receive) or not np.isfinite(chunk.observation_timestamp):
            return ChunkLifecycleEvent(
                ChunkLifecycleStatus.INVALID,
                None,
                "chunk or receive timestamp is non-finite",
                {"contract": "ChunkLifecycleV1"},
            )
        if self._latest is not None:
            if receive < self._latest.receive_timestamp or chunk.observation_timestamp < self._latest.observation_timestamp:
                return ChunkLifecycleEvent(
                    ChunkLifecycleStatus.OUT_OF_ORDER,
                    None,
                    "received chunk is older than the latest accepted chunk",
                    {"latest_chunk_id": self._latest.chunk_id, "incoming_chunk_id": chunk.chunk_id},
                )
            if chunk.chunk_id == self._latest.chunk_id:
                return ChunkLifecycleEvent(
                    ChunkLifecycleStatus.OUT_OF_ORDER,
                    None,
                    "duplicate chunk_id is not accepted as a new chunk",
                    {"latest_chunk_id": self._latest.chunk_id},
                )
        status = self._age_status(receive - float(chunk.observation_timestamp))
        if status is ChunkLifecycleStatus.INVALID:
            return ChunkLifecycleEvent(
                status,
                None,
                "chunk observation time is after its receive time",
                {"event": "RECEIVE", "transport": "ABSTRACT_NO_NETWORK"},
            )
        timing = ChunkTimingV1(
            chunk_id=str(chunk.chunk_id),
            observation_timestamp=float(chunk.observation_timestamp),
            receive_timestamp=receive,
            execute_timestamp=None,
            age_s=receive - float(chunk.observation_timestamp),
            status=status,
        )
        self._latest = timing
        self._latest_chunk = chunk
        return ChunkLifecycleEvent(
            status,
            timing,
            "chunk accepted",
            {"event": "RECEIVE", "transport": "ABSTRACT_NO_NETWORK"},
        )

    def sample(self, *, execute_timestamp: float) -> ChunkLifecycleEvent:
        execute = float(execute_timestamp)
        if self._latest is None or self._latest_chunk is None:
            return ChunkLifecycleEvent(
                ChunkLifecycleStatus.INVALID,
                None,
                "no chunk has been accepted",
                {"event": "EXECUTE"},
            )
        if not np.isfinite(execute):
            return ChunkLifecycleEvent(
                ChunkLifecycleStatus.INVALID,
                None,
                "execute timestamp is non-finite",
                {"event": "EXECUTE"},
            )
        age = execute - self._latest.observation_timestamp
        status = self._age_status(age)
        if status is ChunkLifecycleStatus.INVALID:
            return ChunkLifecycleEvent(
                status,
                None,
                "execute timestamp precedes chunk observation timestamp",
                {"event": "EXECUTE", "transport": "ABSTRACT_NO_NETWORK"},
            )
        timing = ChunkTimingV1(
            chunk_id=self._latest.chunk_id,
            observation_timestamp=self._latest.observation_timestamp,
            receive_timestamp=self._latest.receive_timestamp,
            execute_timestamp=execute,
            age_s=age,
            status=status,
        )
        self._latest = timing
        return ChunkLifecycleEvent(
            status,
            timing,
            "chunk sampled for execution" if status is not ChunkLifecycleStatus.STALE else "chunk is stale at execution",
            {"event": "EXECUTE", "transport": "ABSTRACT_NO_NETWORK"},
        )

    def _age_status(self, age: float) -> ChunkLifecycleStatus:
        if not np.isfinite(age) or age < 0.0:
            return ChunkLifecycleStatus.INVALID
        if age <= self.config.fresh_age_s:
            return ChunkLifecycleStatus.FRESH
        if age <= self.config.stale_age_s:
            return ChunkLifecycleStatus.AGING
        return ChunkLifecycleStatus.STALE
