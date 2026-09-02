from __future__ import annotations

import numpy as np
import pytest

from go2arx5_vla.diagnostics import (
    action_contract_summary,
    action_mask_partition,
    build_universal_action_and_mask,
    label_as_prediction_metrics,
    relative_eef_action,
    relative_eef_roundtrip,
    split_physical_action,
)
from go2arx5_vla.trajectory.so3 import (
    quat_to_matrix,
    rotation_matrix_to_rot6d_rows,
)


def _identity_row_rot6d() -> np.ndarray:
    return np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def _smooth_physical_action() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.arange(40, dtype=np.float64) * 0.05
    phase = np.linspace(0.0, 1.0, 40)
    smooth = phase * phase * (3.0 - 2.0 * phase)
    position = np.stack([0.1 * smooth, -0.03 * smooth, 0.02 * smooth], axis=-1)
    rot6d = np.broadcast_to(_identity_row_rot6d(), (40, 6)).copy()
    eef = np.concatenate([position, rot6d], axis=-1)
    physical = np.concatenate([eef, np.full((40, 1), 0.5)], axis=-1)
    state = np.concatenate([eef[0], [0.5]])
    return timestamps, physical, state


def test_split_and_universal_mask_keep_only_ten_physical_dimensions() -> None:
    _, physical, _ = _smooth_physical_action()
    eef, gripper = split_physical_action(physical)
    assert eef.shape == (40, 9)
    assert gripper.shape == (40, 1)

    universal, mask = build_universal_action_and_mask(physical)
    assert universal.shape == (40, 132)
    assert mask.shape == (40, 132)
    assert np.array_equal(universal[:, :10], physical.astype(np.float32))
    summary = action_mask_partition(mask)
    assert summary["physical_dimensions"] == 10
    assert summary["padding_dimensions"] == 122
    assert summary["physical_active_elements"] == 400
    assert summary["padding_active_elements"] == 0
    assert summary["padding_contributes_to_loss"] is False


def test_relative_action_uses_one_observation_reference_not_frame_deltas() -> None:
    reference_rotation = quat_to_matrix(np.asarray([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]))
    local_rotation = quat_to_matrix(np.asarray([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]))
    reference = np.concatenate([[1.0, 2.0, 3.0], rotation_matrix_to_rot6d_rows(reference_rotation)])
    absolute_rotation = reference_rotation @ local_rotation
    # The world displacement is the reference rotation applied to a local
    # displacement, so the relative result should recover [0.2, 0, 0].
    absolute_position = reference[:3] + reference_rotation @ np.asarray([0.2, 0.0, 0.0])
    absolute = np.concatenate(
        [absolute_position, rotation_matrix_to_rot6d_rows(absolute_rotation)], axis=0
    )[None, :]
    relative = relative_eef_action(absolute, reference)
    assert np.allclose(relative[0, :3], [0.2, 0.0, 0.0], atol=1e-10)
    assert np.allclose(
        relative[0, 3:9], rotation_matrix_to_rot6d_rows(local_rotation), atol=1e-10
    )
    roundtrip = relative_eef_roundtrip(absolute, reference)
    assert roundtrip["max_position_error_m"] <= 1e-10
    assert roundtrip["max_rotation_error_rad"] <= 1e-10


def test_label_as_prediction_is_a_strict_evaluator_sanity_pass() -> None:
    timestamps, physical, state = _smooth_physical_action()
    result = label_as_prediction_metrics(
        physical_action=physical,
        timestamps=timestamps,
        state_v1=state,
        source_frame="NATIVE_WBC_RELATIVE_REFERENCE",
        tcp_semantic_id="x5_link6_gripper_center",
    )
    assert result["identity_pass"] is True
    assert result["classification"] == "LABEL_AS_PREDICTION_PASS"
    assert result["prediction_error"]["position_m"]["max"] == pytest.approx(0.0, abs=1e-12)
    assert result["prediction_error"]["rotation_rad"]["max"] == pytest.approx(0.0, abs=1e-12)
    assert result["profile_ratios"] == {
        "path_length_ratio": pytest.approx(1.0),
        "mean_speed_ratio": pytest.approx(1.0),
        "max_speed_ratio": pytest.approx(1.0),
        "roughness_ratio": pytest.approx(1.0),
    }
    assert result["safety_gate"]["decision"] == "ACCEPT"


def test_contract_marks_sim_period_as_test_only() -> None:
    summary = action_contract_summary()
    assert summary["action_horizon"] == 40
    assert summary["action_sample_period_s"] == pytest.approx(0.05)
    assert summary["action_sample_period_provenance"] == "TEST_ONLY_STAGE22_SIMULATION"
    assert summary["production_action_timing"] == "TBD_BY_ADAPTATION_DATASET"
    assert summary["eef_transform_is_frame_to_frame_delta"] is False


def test_universal_builder_rejects_wrong_horizon() -> None:
    with pytest.raises(ValueError, match="expected horizon"):
        build_universal_action_and_mask(np.zeros((39, 10), dtype=np.float64))
