from __future__ import annotations

from typing import Callable, Optional

from ..contracts.trajectory import TrajectoryChunkV1
from .base import TrajectoryProvider


class ReplayTrajectoryProvider(TrajectoryProvider):
    """Replay provider backed by a caller-owned chunk factory."""

    def __init__(self, factory: Callable[[float], Optional[TrajectoryChunkV1]]):
        self._factory = factory

    def next_chunk(self, observation_timestamp: float) -> Optional[TrajectoryChunkV1]:
        return self._factory(float(observation_timestamp))
