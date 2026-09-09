"""CPU contract tests for Stage38 StructuredActionV1."""

from __future__ import annotations

import numpy as np
import pytest

from go2arx5_vla.structured_action import (
    STRUCTURED_ACTION_DIM,
    STRUCTURED_ACTION_HORIZON,
    StructuredActionV1,
    decode_structured_action,
    encode_structured_action,
    materialize_structured_action,
    reconcile_stage45b_label_anchor,
    structured_action_contract_summary,
    structured_action_mask,
    structured_active_scalar_count,
)
from go2arx5_vla.trajectory.so3 import (
    geodesic_angle,
    rot6d_rows_to_matrix,
    rotation_matrix_to_rot6d_rows,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.arange(STRUCTURED_ACTION_HORIZON, dtype=np.float64) * 0.05
    phase = np.linspace(0.0, 1.0, STRUCTURED_ACTION_HORIZON)
    smooth = phase * phase * (3.0 - 2.0 * phase)
    position = np.stack(
        [0.1 * smooth, -0.03 * smooth, 0.02 * smooth], axis=-1
    )
    identity_rows = np.broadcast_to(
        np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        (STRUCTURED_ACTION_HORIZON, 6),
    ).copy()
    eef = np.concatenate([position, identity_rows], axis=-1)
    physical = np.concatenate(
        [eef, np.linspace(0.1, 0.9, STRUCTURED_ACTION_HORIZON)[:, None]], axis=-1
    )
    state = np.concatenate([eef[0], [0.25]])
    return timestamps, physical, state


def test_layout_and_mask_are_frozen() -> None:
    summary = structured_action_contract_summary("T16")
    assert summary["action_horizon"] == 40
    assert summary["universal_action_dim"] == 132
    assert summary["geometry_k"] == 16
    assert summary["predicted_future_geometry_controls"] == 15
    assert summary["mask_active_scalar_count"] == 190
    assert summary["dense_h40_active_scalar_count"] == 400
    mask = structured_action_mask("T16")
    assert mask.shape == (40, STRUCTURED_ACTION_DIM)
    assert int(np.count_nonzero(mask)) == structured_active_scalar_count("T16")
    assert np.count_nonzero(mask[15:, :9]) == 0
    assert np.count_nonzero(mask[:, 11:]) == 0


def test_encode_decode_materialize_preserves_contract_and_h40_gripper() -> None:
    timestamps, physical, state = _fixture()
    label = encode_structured_action(
        physical,
        timestamps,
        state,
        timing_mode="T16",
        source_frame="NATIVE_WBC_RELATIVE_REFERENCE",
        tcp_semantic_id="x5_link6_gripper_center",
    )
    assert isinstance(label, StructuredActionV1)
    assert label.values.shape == (40, 132)
    assert label.mask.shape == (40, 132)
    assert label.active_scalar_count == 190
    assert np.isclose(label.source_duration_s, 1.95, atol=1e-12)
    assert np.all(np.diff(label.timing_timestamps) > 0.0)
    assert np.isclose(label.timing_timestamps[-1], 1.95, atol=1e-12)
    assert np.isclose(np.mean(label.values[:15, 9]), 0.0, atol=1e-12)
    assert np.allclose(label.timing_interval_fractions, np.diff(label.timing_timestamps) / 1.95)

    intent = decode_structured_action(label, intent_id="roundtrip")
    materialized = materialize_structured_action(label, intent)
    assert materialized.source_frame == label.source_frame
    assert materialized.tcp_semantic_id == label.tcp_semantic_id
    assert np.array_equal(materialized.point_timestamps, timestamps)
    assert np.array_equal(materialized.gripper, physical[:, 9])
    assert np.array_equal(materialized.eef_position[0], physical[0, :3])
    assert np.array_equal(materialized.eef_position[-1], physical[-1, :3])
    assert np.isfinite(materialized.eef_position).all()
    assert np.isfinite(materialized.eef_orientation).all()
    assert np.isclose(intent.validity_end - intent.validity_start, 1.95, atol=1e-12)


def test_decode_rejects_anchor_mismatch_and_inactive_values() -> None:
    timestamps, physical, state = _fixture()
    with pytest.raises(ValueError, match="runtime anchor"):
        encode_structured_action(physical, timestamps, state + np.r_[0.01, np.zeros(9)])

    label = encode_structured_action(physical, timestamps, state)
    bad_values = label.values.copy()
    bad_values[20, 20] = 1.0
    with pytest.raises(ValueError, match="inactive"):
        StructuredActionV1(
            values=bad_values,
            mask=label.mask,
            source_timestamps=label.source_timestamps,
            source_duration_s=label.source_duration_s,
            anchor_state_v1=label.anchor_state_v1,
            geometry_u=label.geometry_u,
            timing_u=label.timing_u,
            timing_timestamps=label.timing_timestamps,
            timing_mode=label.timing_mode,
            source_frame=label.source_frame,
            tcp_semantic_id=label.tcp_semantic_id,
            observation_timestamp=label.observation_timestamp,
            provenance=label.provenance,
        )


def test_canonical_model_rot6d_is_distinct_boundary_from_native_rows() -> None:
    # The fixture is identity in both conventions, so this assertion focuses
    # on the explicit contract labels rather than relying on an ambiguous
    # numerical equality for the identity rotation.
    _, physical, state = _fixture()
    label = encode_structured_action(physical, np.arange(40) * 0.05, state)
    assert label.provenance["rot6d_layout"] == "CANONICAL_FIRST_TWO_COLUMNS"
    assert label.provenance["native_rot6d_layout"] == (
        "NATIVE_WBC_FIRST_TWO_ROWS_AT_ADAPTER_BOUNDARY"
    )
    assert rotation_matrix_to_rot6d_rows(np.eye(3)).shape == (6,)


def test_stage45b_label_anchor_reconciliation_preserves_physical_action() -> None:
    # This finite native-row representation exercises the arccos one-ulp floor
    # in the strict self-comparison while remaining the same physical pose.
    native_rot6d = np.asarray(
        [0.94230586, -0.11466519, 0.31450209, 0.33371529, 0.39569730, -0.85560375],
        dtype=np.float64,
    )
    timestamps = np.arange(STRUCTURED_ACTION_HORIZON, dtype=np.float64) * 0.05
    physical = np.zeros((STRUCTURED_ACTION_HORIZON, 10), dtype=np.float64)
    physical[:, 3:9] = native_rot6d
    physical[:, 9] = 0.5
    state = physical[0].copy()
    physical_before = physical.copy()
    state_before = state.copy()

    raw_error = float(
        geodesic_angle(
            rot6d_rows_to_matrix(state[3:9]),
            rot6d_rows_to_matrix(physical[0, 3:9]),
        )
    )
    assert raw_error > 1e-9
    reconciled, metadata = reconcile_stage45b_label_anchor(physical, state)
    label = encode_structured_action(physical, timestamps, reconciled)

    assert np.array_equal(physical, physical_before)
    assert np.array_equal(state, state_before)
    assert np.array_equal(reconciled[:3], physical[0, :3])
    assert reconciled[9] == state[9]
    assert metadata["label_only_state_copy"] is True
    assert metadata["physical_action_untouched"] is True
    assert metadata["original_observation_state_untouched"] is True
    assert metadata["reconciled_state_action_rotation_error_rad"] <= 1e-9
    assert label.anchor_state_v1.shape == (10,)


def test_stage45b_label_anchor_reconciliation_rejects_material_mismatch() -> None:
    _, physical, state = _fixture()
    state = state.copy()
    state[0] += 2e-6
    with pytest.raises(ValueError, match="same physical pose"):
        reconcile_stage45b_label_anchor(physical, state)
