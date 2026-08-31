from __future__ import annotations

from typing import Optional

from ..contracts.trajectory import TrajectoryChunkV1
from .base import TrajectoryProvider


class GR00TTrajectoryProvider(TrajectoryProvider):
    """Interface placeholder for the future GR00T N1.7 runtime provider."""

    def next_chunk(self, observation_timestamp: float) -> Optional[TrajectoryChunkV1]:
        del observation_timestamp
        raise NotImplementedError("GR00T model execution is intentionally outside Stage14 CPU tests")
