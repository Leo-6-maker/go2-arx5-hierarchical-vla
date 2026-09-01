"""Dependency-light bridge to LeRobot/official GR00T processor inputs.

The project does not vendor LeRobot or the private/model runtime.  These
helpers therefore expose a stable logical record and a lazy official
``VLAStepData`` constructor.  Importing this module remains CPU-only.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from .readiness import GR00TTrainingExampleV1


def example_to_lerobot_record(example: GR00TTrainingExampleV1) -> dict[str, Any]:
    """Return one processor-facing record without importing LeRobot."""

    return example.to_lerobot_record()


def examples_to_lerobot_records(
    examples: Iterable[GR00TTrainingExampleV1],
) -> list[dict[str, Any]]:
    return [example_to_lerobot_record(example) for example in examples]


def dataset_statistics(examples: Iterable[GR00TTrainingExampleV1]) -> dict[str, Any]:
    """Compute finite 10D state/action statistics for fixture auditing."""

    materialized = list(examples)
    if not materialized:
        raise ValueError("at least one example is required")
    states = np.stack([example.state_v1.as_array() for example in materialized], axis=0)
    actions = np.concatenate([example.physical_action for example in materialized], axis=0)
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("state/action statistics contain non-finite values")

    def stats(array: np.ndarray) -> dict[str, list[float]]:
        return {
            "min": np.min(array, axis=0).tolist(),
            "max": np.max(array, axis=0).tolist(),
            "mean": np.mean(array, axis=0).tolist(),
            "std": np.std(array, axis=0).tolist(),
        }

    return {
        "state": stats(states),
        "action": stats(actions),
        "state_shape": list(states.shape),
        "action_shape": list(actions.shape),
        "state_dim": int(states.shape[-1]),
        "action_dim": int(actions.shape[-1]),
        "rot6d_layout": "GR00T_FIRST_TWO_ROWS",
        "action_label": "future_desired_eef_plus_gripper",
    }


def build_official_vla_step_data(
    example: GR00TTrainingExampleV1,
    *,
    body_rgb: np.ndarray,
    wrist_rgb: np.ndarray | None,
    embodiment: Any,
) -> Any:
    """Build official GR00T ``VLAStepData`` lazily.

    ``body_rgb`` and ``wrist_rgb`` must already be decoded RGB arrays.  This
    helper never opens a camera or a video file and never invokes a model.
    """

    from gr00t.data.types import VLAStepData  # type: ignore[import-not-found]

    body = np.asarray(body_rgb)
    if body.ndim == 3:
        body = body[None, ...]
    if body.ndim != 4:
        raise ValueError("body_rgb must have shape [T,H,W,C] or [H,W,C]")
    images: dict[str, np.ndarray] = {"external_rgb": body}
    if wrist_rgb is not None:
        wrist = np.asarray(wrist_rgb)
        if wrist.ndim == 3:
            wrist = wrist[None, ...]
        if wrist.ndim != 4:
            raise ValueError("wrist_rgb must have shape [T,H,W,C] or [H,W,C]")
        images["wrist_rgb"] = wrist
    state = example.state_v1.as_array()
    return VLAStepData(
        images=images,
        states={
            # Official VLAStepData expects one observation step as [T,D].
            # The extra batch dimension belongs to model decode/state input,
            # not to the processor's per-message state payload.
            "eef_9d": state[None, :9],
            "gripper_position": state[None, 9:],
        },
        actions={},
        text=example.task_instruction,
        embodiment=embodiment,
    )


def assert_processor_physical_abi(
    example: GR00TTrainingExampleV1,
    *,
    decoded_physical_action: Any,
) -> dict[str, Any]:
    """Check the model-facing physical shape before official decoding."""

    decoded = np.asarray(decoded_physical_action, dtype=np.float64)
    if decoded.ndim == 3:
        if decoded.shape[0] != 1:
            raise ValueError("only a single processor sample is supported by this audit")
        decoded = decoded[0]
    if decoded.shape != (example.horizon, 10):
        raise ValueError(
            f"decoded physical action must be [{example.horizon},10], got {decoded.shape}"
        )
    if not np.isfinite(decoded).all():
        raise ValueError("decoded physical action is non-finite")
    return {
        "decoded_shape": list(decoded.shape),
        "finite": True,
        "state_dim": 10,
        "action_dim": 10,
        "eef_dim": 9,
        "gripper_dim": 1,
        "target_label_source": "GR00TTrainingExampleV1.future_desired_eef_9d",
    }
