"""Runtime-independent Stage45C evaluation primitives.

The package intentionally contains no GR00T, Torch, CUDA, simulator, or PBS
imports.  Runtime adapters may provide raw predictions later; all scientific
metric calculations live here so the H200 path cannot grow a second metric
implementation.
"""

from .checkpoint_eval import (
    EVALUATION_MANIFEST_SCHEMA,
    EvaluationManifestError,
    aggregate_evaluation_rows,
    evaluate_raw_prediction,
    load_evaluation_manifest,
    validate_evaluation_manifest,
)
from .structured_action_eval import (
    RAW_ACTION_SHAPE,
    STATE_V1_SHAPE,
    PHYSICAL_ACTION_SHAPE,
    decode_raw_prediction,
)
from .trajectory_metrics import (
    ACTION_HORIZON,
    ACTION_WIDTH,
    TARGET_PROGRESS_TOLERANCE_M,
    EvaluationContractError,
    comparison_metrics,
    trajectory_metrics,
)

__all__ = [
    "ACTION_HORIZON",
    "ACTION_WIDTH",
    "EVALUATION_MANIFEST_SCHEMA",
    "EvaluationContractError",
    "EvaluationManifestError",
    "PHYSICAL_ACTION_SHAPE",
    "RAW_ACTION_SHAPE",
    "STATE_V1_SHAPE",
    "TARGET_PROGRESS_TOLERANCE_M",
    "aggregate_evaluation_rows",
    "comparison_metrics",
    "decode_raw_prediction",
    "evaluate_raw_prediction",
    "load_evaluation_manifest",
    "trajectory_metrics",
    "validate_evaluation_manifest",
]
