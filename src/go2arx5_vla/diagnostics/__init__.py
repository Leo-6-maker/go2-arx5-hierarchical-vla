"""CPU-only diagnostics for the robot-specific GR00T data boundary."""

from .stage23_action_pipeline import (
    ACTION_HORIZON,
    PHYSICAL_ACTION_DIM,
    PADDING_ACTION_DIM,
    UNIVERSAL_ACTION_DIM,
    action_contract_summary,
    action_mask_partition,
    build_universal_action_and_mask,
    label_as_prediction_metrics,
    relative_eef_action,
    relative_eef_roundtrip,
    split_physical_action,
)

__all__ = [
    "ACTION_HORIZON",
    "PHYSICAL_ACTION_DIM",
    "PADDING_ACTION_DIM",
    "UNIVERSAL_ACTION_DIM",
    "action_contract_summary",
    "action_mask_partition",
    "build_universal_action_and_mask",
    "label_as_prediction_metrics",
    "relative_eef_action",
    "relative_eef_roundtrip",
    "split_physical_action",
]
