from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..contracts.trajectory import TrajectoryChunkV1


class TrajectoryProvider(ABC):
    """Downstream code consumes chunks, never model-specific tensors."""

    @abstractmethod
    def next_chunk(self, observation_timestamp: float) -> Optional[TrajectoryChunkV1]:
        raise NotImplementedError
