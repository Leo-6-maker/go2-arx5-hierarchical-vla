"""CPU-only contract tests for the Stage38-E offline evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import stage38_structured_offline_evaluation as stage38e  # noqa: E402
from go2arx5_vla.contracts import TrajectoryChunkV1  # noqa: E402
from go2arx5_vla.structured_action import encode_structured_action  # noqa: E402


def _chunk() -> TrajectoryChunkV1:
    return TrajectoryChunkV1(
        chunk_id="stage38e-test",
        source="stage38e-test",
        observation_timestamp=0.0,
        point_timestamps=np.linspace(0.0, 1.95, stage38e.HORIZON),
        eef_position=np.zeros((stage38e.HORIZON, 3), dtype=np.float64),
        eef_orientation=np.tile(np.asarray([0.0, 0.0, 0.0, 1.0]), (stage38e.HORIZON, 1)),
        source_frame="test-frame",
        tcp_semantic_id="test-tcp",
        validity_start=0.0,
        validity_end=1.95,
        gripper=np.zeros(stage38e.HORIZON, dtype=np.float64),
    )


def _metric_row(seed: int, values: dict[str, float], decision: str = "REJECT") -> dict[str, object]:
    return {
        "seed": seed,
        "finite": True,
        "decode_valid": True,
        "metrics_to_expert": {
            "preview": {
                "preview_position_error_m": {"p95": values["p1"]},
                "preview_rotation_error_rad": {"p95": values["p2"]},
                "se3_arc_length_ratio": 1.0 + values["p3"],
                "time_to_arc_progress_error": {"mean": values["p4"]},
            },
            "candidate_profile": {"roughness_position_second_difference_m": {"mean": values["p5"]}},
            "high_frequency_xyz_fraction": values["hf"],
            "endpoint_position_error_m": values["endpoint_xyz"],
            "endpoint_rotation_error_rad": values["endpoint_so3"],
            "safety_gate": {"decision": decision},
            "all_finite": True,
        },
    }


def test_static_guard_is_ast_based_and_passes_its_own_source() -> None:
    guard = stage38e.static_guard()
    assert guard["status"] == "PASS"
    assert guard["static_no_model_or_training_path"] is True
    assert guard["optimizer_or_scheduler_step_literal"] is False
    assert guard["checkpoint_write_literal"] is False


def test_chunk_physical_action_uses_native_row_packed_rotation_and_shape() -> None:
    physical = stage38e._chunk_physical_action(_chunk())
    assert physical.shape == (1, stage38e.HORIZON, stage38e.PHYSICAL_DIM)
    assert physical.dtype == np.float32
    assert np.array_equal(physical[0, 0, 3:9], np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32))
    assert np.isfinite(physical).all()


def test_codec_anchor_uses_official_float32_materialization_boundary() -> None:
    native_rows = np.asarray(
        [
            0.5496097999785584,
            -0.3069952625745831,
            0.7769703832993198,
            0.8182666570490404,
            0.3853249473212521,
            -0.42657281082326653,
        ],
        dtype=np.float64,
    )
    eef = np.zeros((stage38e.HORIZON, 9), dtype=np.float64)
    eef[:, 3:9] = native_rows
    physical = np.concatenate([eef, np.zeros((stage38e.HORIZON, 1))], axis=-1).astype(np.float32)
    state = np.concatenate([eef[0], [0.0]])
    timestamps = np.linspace(0.0, 1.95, stage38e.HORIZON)
    with pytest.raises(ValueError, match="runtime anchor"):
        encode_structured_action(physical, timestamps, state, timing_mode="T16")
    label = encode_structured_action(physical, timestamps, state.astype(np.float32), timing_mode="T16")
    assert label.active_scalar_count == stage38e.STRUCTURED_ACTIVE_SCALARS


def test_high_frequency_fraction_is_finite_and_distinguishes_pathology() -> None:
    smooth = np.linspace(0.0, 1.0, stage38e.HORIZON)[:, None] * np.asarray([1.0, 0.0, 0.0])[None]
    alternating = smooth.copy()
    alternating[:, 1] = np.asarray([0.02 if index % 2 else -0.02 for index in range(stage38e.HORIZON)])
    assert stage38e._high_frequency_xyz_fraction(smooth) == 0.0
    assert stage38e._high_frequency_xyz_fraction(alternating) > 0.0


def test_p5_requires_both_roughness_and_high_frequency_improvement() -> None:
    rows = [_metric_row(seed, {"p1": 1.0, "p2": 1.0, "p3": 1.0, "p4": 1.0, "p5": 0.5, "hf": 0.5, "endpoint_xyz": 0.1, "endpoint_so3": 0.1}) for seed in range(2223, 2239)]
    dense = {seed: {"primary_values": {"P1_WBC_PREVIEW_XYZ": 2.0, "P2_WBC_PREVIEW_SO3": 2.0, "P3_SE3_ARC_INFLATION": 2.0, "P4_TIMING_PROGRESS": 2.0, "P5_ROUGHNESS_HF": 1.0, "P5_HIGH_FREQUENCY_XYZ": 1.0}, "endpoint_xyz": 0.1, "endpoint_so3": 0.1} for seed in range(2223, 2239)}
    aggregate = stage38e._aggregate_primary(rows, dense)
    assert aggregate["pass_count"] == 5
    assert aggregate["families"]["P5_ROUGHNESS_HF"]["primary_metric_improved"] is True
    assert aggregate["families"]["P5_ROUGHNESS_HF"]["high_frequency_strict_seed_improvement_count"] == 16


def test_primary_classification_has_fail_closed_regression_and_mutually_exclusive_a_b() -> None:
    families = {name: {"structured_median": 1.0, "dense_median": 2.0} for name in ("P1_WBC_PREVIEW_XYZ", "P2_WBC_PREVIEW_SO3", "P3_SE3_ARC_INFLATION", "P4_TIMING_PROGRESS", "P5_ROUGHNESS_HF")}
    primary = {"pass_count": 5, "passed_families": list(families), "families": families, "endpoint_median": {"jointly_not_worse": True}}
    safety_warn = {"structured": {"ACCEPT": 0, "WARN": 1, "REJECT": 15}}
    safety_reject = {"structured": {"ACCEPT": 0, "WARN": 0, "REJECT": 16}}
    rows = [_metric_row(seed, {"p1": 1.0, "p2": 1.0, "p3": 1.0, "p4": 1.0, "p5": 1.0, "hf": 1.0, "endpoint_xyz": 0.1, "endpoint_so3": 0.1}) for seed in range(2223, 2239)]
    assert stage38e._classify_primary(primary, rows, safety_warn) == "STRUCTURED_OFFLINE_CLEAR_PROFILE_IMPROVEMENT"
    assert stage38e._classify_primary(primary, rows, safety_reject) == "STRUCTURED_OFFLINE_PROFILE_IMPROVED_BUT_GATE_STILL_REJECTS"

    broad_regression = {name: {"structured_median": 3.0, "dense_median": 2.0} for name in list(families)[:4]}
    broad_regression["P5_ROUGHNESS_HF"] = {"structured_median": 1.0, "dense_median": 2.0}
    regression_primary = {"pass_count": 1, "passed_families": ["P5_ROUGHNESS_HF"], "families": broad_regression, "endpoint_median": {"jointly_not_worse": True}}
    assert stage38e._classify_primary(regression_primary, rows, safety_warn) == "STRUCTURED_OFFLINE_REGRESSION"
