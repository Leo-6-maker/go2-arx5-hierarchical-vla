"""Robot-specific data-readiness contracts.

The data layer is intentionally separate from the VLA model and the native
WBC.  Raw episodes preserve source-rate observations; training examples are
derived, versioned views whose state/action ABI matches the Stage17
VLA-to-WBC contract.
"""

from .quality import (
    EpisodeGateDecision,
    EpisodeQualityGateConfig,
    EpisodeQualityGateV1,
    EpisodeQualityIssue,
    EpisodeQualityResult,
)
from .readiness import (
    CAMERA_TIMESTAMP_LAYOUT,
    GROOT_EEF_ROT6D_LAYOUT,
    CalibrationBundleV1,
    CameraStreamV1,
    GR00TTrainingExampleV1,
    RawEpisodeV1,
    StateV1,
    SynchronizationResultV1,
    SynchronizerConfigV1,
    SynchronizerV1,
    interpolate_pose_rot6d_rows,
)

__all__ = [
    "CAMERA_TIMESTAMP_LAYOUT",
    "GROOT_EEF_ROT6D_LAYOUT",
    "CalibrationBundleV1",
    "CameraStreamV1",
    "EpisodeGateDecision",
    "EpisodeQualityGateConfig",
    "EpisodeQualityGateV1",
    "EpisodeQualityIssue",
    "EpisodeQualityResult",
    "GR00TTrainingExampleV1",
    "RawEpisodeV1",
    "StateV1",
    "SynchronizationResultV1",
    "SynchronizerConfigV1",
    "SynchronizerV1",
    "interpolate_pose_rot6d_rows",
]
