"""Frame-corrected reference conversion; historical V1 bytes remain frozen.

DEC1 I_TIME and I_PROGRESS must share this revision. This is not a WBC change,
a progress estimator, or permission to execute an unchecked reference.
"""
from dataclasses import replace
from typing import Optional

from ..contracts.trajectory import TrajectoryChunkV1
from .groot_to_wbc import GR00TToWBCAdapter


class GR00TToWBCAdapterV2(GR00TToWBCAdapter):
    def _prepare_chunk(self, chunk: TrajectoryChunkV1) -> TrajectoryChunkV1:
        prepared = super()._prepare_chunk(chunk)
        if prepared.source_frame != self.config.wbc_frame:
            # V1 already checked that a transform is present in this branch,
            # but omitted its application for the expected source-frame name.
            prepared = prepared.transformed(self.config.frame_transform, target_frame=self.config.wbc_frame)
        return prepared

    def adapt(self, chunk: TrajectoryChunkV1, *, controller_timestamp: Optional[float] = None):
        preview, grip = super().adapt(chunk, controller_timestamp=controller_timestamp)
        preview = replace(preview, provenance={**preview.provenance,
            'adapter': 'GR00TToWBCAdapterV2',
            'reference_adapter_version': 'FRAME_CORRECTED_REFERENCE_V2'})
        return preview, grip
