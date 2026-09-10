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
from .reusable import (
    GENERIC_EVALUATION_MANIFEST_SCHEMA,
    GENERIC_EVALUATION_OUTPUT_SCHEMA,
    GENERIC_EVALUATION_PLAN_SCHEMA,
    GENERIC_EVALUATION_ROW_SCHEMA,
    GENERIC_EVAL_STATUS,
    GENERIC_PREDICTION_RECORD_SCHEMA,
    ReusableEvaluationError,
    build_evaluation_plan,
    load_prediction_records,
    run_evaluation,
    verify_evaluation_artifacts,
)
from .analyzer import (
    ANALYSIS_SCHEMA,
    CheckpointAnalysisError,
    analyze_rows,
    load_rows,
    write_analysis,
)
from .h200_admission import (
    H200_INFERENCE_ADMISSION_SCHEMA,
    H200_INFERENCE_ADMISSION_STATUS,
    H200AdmissionError,
    validate_h200_inference_admission,
)
from .h200_binding import (
    ADAPTER_RUNTIME_BINDING_SCHEMA,
    AdapterRuntimeBindingError,
    AdapterRuntimeBindingV1,
    build_adapter_runtime_binding,
    validate_adapter_runtime_binding,
)
from .stage45c_compatibility import (
    STAGE45C_COMPATIBILITY_PROOF_SCHEMA,
    Stage45CCompatibilityError,
    build_stage45c_compatibility_proof,
    project_stage45c_window,
    validate_stage45c_compatibility_proof,
)
from .stage45c_one_window_fixture import (
    PI0102_FIXTURE_STATUS,
    PI0102_ONE_WINDOW_FIXTURE_SCHEMA,
    PI0102_ONE_WINDOW_SCOPE,
    PI0102QualificationFixtureError,
    build_pi0102_one_window_fixture,
    validate_pi0102_one_window_fixture,
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
    "ANALYSIS_SCHEMA",
    "CheckpointAnalysisError",
    "GENERIC_EVALUATION_MANIFEST_SCHEMA",
    "GENERIC_EVALUATION_OUTPUT_SCHEMA",
    "GENERIC_EVALUATION_PLAN_SCHEMA",
    "GENERIC_EVALUATION_ROW_SCHEMA",
    "GENERIC_EVAL_STATUS",
    "GENERIC_PREDICTION_RECORD_SCHEMA",
    "ReusableEvaluationError",
    "analyze_rows",
    "build_evaluation_plan",
    "load_prediction_records",
    "load_rows",
    "run_evaluation",
    "verify_evaluation_artifacts",
    "write_analysis",
    "H200_INFERENCE_ADMISSION_SCHEMA",
    "H200_INFERENCE_ADMISSION_STATUS",
    "H200AdmissionError",
    "validate_h200_inference_admission",
    "ADAPTER_RUNTIME_BINDING_SCHEMA",
    "AdapterRuntimeBindingError",
    "AdapterRuntimeBindingV1",
    "build_adapter_runtime_binding",
    "validate_adapter_runtime_binding",
    "STAGE45C_COMPATIBILITY_PROOF_SCHEMA",
    "Stage45CCompatibilityError",
    "build_stage45c_compatibility_proof",
    "project_stage45c_window",
    "validate_stage45c_compatibility_proof",
    "PI0102_FIXTURE_STATUS",
    "PI0102_ONE_WINDOW_FIXTURE_SCHEMA",
    "PI0102_ONE_WINDOW_SCOPE",
    "PI0102QualificationFixtureError",
    "build_pi0102_one_window_fixture",
    "validate_pi0102_one_window_fixture",
]
