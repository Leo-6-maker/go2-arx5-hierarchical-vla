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
from .processor import dataset_statistics
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
from .stage22_expert import (
    SameSceneExpertSnapshotV1,
    load_snapshot,
    snapshot_to_raw_episode,
    snapshot_to_training_example,
)
from .stage22_split import (
    FINAL_TEST_TARGET_IDS,
    TRAIN_TARGET_IDS,
    VALIDATION_TARGET_IDS,
    locked_target_ids,
    split_lock_payload,
    validate_locked_target_universe,
)
from .stage22_snapshot_lock import (
    EXPECTED_INFERENCE_SEEDS,
    EXPECTED_VALIDATION_SNAPSHOT_IDS,
    load_snapshot_lock,
    snapshot_ids_sha256,
    snapshot_lock_file_sha256,
    validate_snapshot_lock,
)

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
    "dataset_statistics",
    "replay_raw_episode",
    "run_synchronization_stress",
    "SameSceneExpertSnapshotV1",
    "load_snapshot",
    "snapshot_to_raw_episode",
    "snapshot_to_training_example",
    "TRAIN_TARGET_IDS",
    "VALIDATION_TARGET_IDS",
    "FINAL_TEST_TARGET_IDS",
    "locked_target_ids",
    "split_lock_payload",
    "validate_locked_target_universe",
    "EXPECTED_INFERENCE_SEEDS",
    "EXPECTED_VALIDATION_SNAPSHOT_IDS",
    "load_snapshot_lock",
    "snapshot_ids_sha256",
    "snapshot_lock_file_sha256",
    "validate_snapshot_lock",
]
