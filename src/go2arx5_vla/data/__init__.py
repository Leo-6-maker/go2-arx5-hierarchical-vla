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
from .materialization import (
    ACTION_DIM,
    BODY_VIDEO_KEY,
    DEFAULT_HORIZON,
    MaterializationConfigV1,
    MaterializationInputV1,
    MaterializedDatasetV1,
    WRIST_VIDEO_KEY,
    materialize_lerobot_v2,
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
from .recorder import RecorderConfigV1, RecorderV1, RecordedEpisodeV1, replay_raw_episode
from .teleop import TeleopCommandV1
from .stress import run_synchronization_stress

__all__ = [
    "CAMERA_TIMESTAMP_LAYOUT",
    "ACTION_DIM",
    "BODY_VIDEO_KEY",
    "DEFAULT_HORIZON",
    "GROOT_EEF_ROT6D_LAYOUT",
    "CalibrationBundleV1",
    "CameraStreamV1",
    "EpisodeGateDecision",
    "EpisodeQualityGateConfig",
    "EpisodeQualityGateV1",
    "EpisodeQualityIssue",
    "EpisodeQualityResult",
    "GR00TTrainingExampleV1",
    "MaterializationConfigV1",
    "MaterializationInputV1",
    "MaterializedDatasetV1",
    "RawEpisodeV1",
    "RecorderConfigV1",
    "RecorderV1",
    "RecordedEpisodeV1",
    "StateV1",
    "SynchronizationResultV1",
    "SynchronizerConfigV1",
    "SynchronizerV1",
    "TeleopCommandV1",
    "WRIST_VIDEO_KEY",
    "interpolate_pose_rot6d_rows",
    "materialize_lerobot_v2",
    "replay_raw_episode",
    "run_synchronization_stress",
]
