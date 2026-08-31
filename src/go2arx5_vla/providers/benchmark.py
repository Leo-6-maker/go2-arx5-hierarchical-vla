from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Optional

from ..contracts.trajectory import TrajectoryChunkV1
from .base import TrajectoryProvider


class BenchmarkTrajectoryProvider(TrajectoryProvider):
    """Finite provider over already validated benchmark chunks."""

    def __init__(self, chunks: Iterable[TrajectoryChunkV1]):
        self._chunks: Iterator[TrajectoryChunkV1] = iter(chunks)

    def next_chunk(self, observation_timestamp: float) -> Optional[TrajectoryChunkV1]:
        del observation_timestamp
        try:
            return next(self._chunks)
        except StopIteration:
            return None
