"""Reference-source boundary used by the native WBC runner.

The actual lab runtime remains outside this repository. This module makes the
three Stage14 source modes explicit and ensures every mode returns the same
``WBCPreviewV1`` object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..adapters.groot_to_wbc import GR00TToWBCAdapter
from ..contracts.trajectory import TrajectoryChunkV1, WBCPreviewV1
from .reference_modes import ReferenceMode


@dataclass
class ReferenceSource:
    mode: ReferenceMode
    adapter: GR00TToWBCAdapter
    chunk_factory: Callable[[float], Optional[TrajectoryChunkV1]]

    def preview(self, controller_timestamp: float) -> WBCPreviewV1:
        chunk = self.chunk_factory(float(controller_timestamp))
        if chunk is None:
            raise RuntimeError(f"no reference chunk available for mode {self.mode.value}")
        preview, _ = self.adapter.adapt(chunk, controller_timestamp=controller_timestamp)
        return preview
