"""CPU-only tests for the Stage38-D label-side normalization boundary."""

from __future__ import annotations

import numpy as np

from go2arx5_vla.stage38_structured_training import StructuredActionNormalizer
from go2arx5_vla.structured_action import (
    STRUCTURED_ACTION_DIM,
    STRUCTURED_ACTION_HORIZON,
    encode_structured_action,
    structured_action_mask,
)
from go2arx5_vla.trajectory.so3 import rotation_matrix_to_rot6d_rows


def _label(offset: float):
    timestamps = np.arange(STRUCTURED_ACTION_HORIZON, dtype=np.float64) * 0.05
    position = np.zeros((STRUCTURED_ACTION_HORIZON, 3), dtype=np.float64)
    position[:, 0] = offset + np.linspace(0.0, 0.1, STRUCTURED_ACTION_HORIZON)
    rotation = np.broadcast_to(
        rotation_matrix_to_rot6d_rows(np.eye(3)),
        (STRUCTURED_ACTION_HORIZON, 6),
    ).copy()
    eef = np.concatenate([position, rotation], axis=-1)
    physical = np.concatenate(
        [eef, np.linspace(0.1, 0.9, STRUCTURED_ACTION_HORIZON)[:, None]], axis=-1
    )
    state = np.concatenate([eef[0], [0.1]])
    return encode_structured_action(physical, timestamps, state, timing_mode="T16")


def test_structured_stats_ignore_inactive_dimensions_and_normalize_to_unit_interval() -> None:
    labels = [_label(0.0), _label(0.02)]
    from go2arx5_vla.stage38_structured_training import _structured_stats

    metadata = _structured_stats(labels, use_percentiles=False)
    normalizer = StructuredActionNormalizer(metadata)
    normalized = normalizer.normalize(labels[0])

    assert normalized.shape == (STRUCTURED_ACTION_HORIZON, STRUCTURED_ACTION_DIM)
    assert np.isfinite(normalized).all()
    assert np.array_equal(normalized[labels[0].mask == 0.0], np.zeros(np.count_nonzero(labels[0].mask == 0.0), dtype=np.float32))
    assert np.max(np.abs(normalized[labels[0].mask == 1.0])) <= 1.0
    assert int(np.count_nonzero(structured_action_mask("T16"))) == 190


def test_constant_active_dimension_is_zero_without_division_failure() -> None:
    labels = [_label(0.0)]
    from go2arx5_vla.stage38_structured_training import _structured_stats

    metadata = _structured_stats(labels, use_percentiles=True)
    normalizer = StructuredActionNormalizer(metadata)
    normalized = normalizer.normalize(labels[0])
    assert np.isfinite(normalized).all()
    constant_dims = np.isclose(
        np.asarray(metadata["selected_lower"]),
        np.asarray(metadata["selected_upper"]),
    )
    assert np.all(normalized[:, constant_dims] == 0.0)
