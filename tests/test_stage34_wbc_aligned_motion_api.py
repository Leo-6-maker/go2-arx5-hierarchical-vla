from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from go2arx5_vla.contracts import MotionIntentV2, TrajectoryChunkV1, WBCPreviewV1
from go2arx5_vla.motion_api import (
    GEOMETRY_K_VALUES,
    TIMING_KNOTS,
    TIMING_MODES,
    build_geometry_candidate,
    build_timing_candidate,
    evaluate_preview_equivalence,
)
from go2arx5_vla.trajectory.so3 import geodesic_angle, matrix_to_quat, quat_to_matrix


REPO = Path(__file__).resolve().parents[1]
QUALIFICATION = REPO / "src/go2arx5_vla/motion_api/qualification.py"
ORCHESTRATOR = REPO / "scripts/stage34_wbc_aligned_motion_api.py"


def _z_rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _chunk(*, straight: bool = False, count: int = 41) -> TrajectoryChunkV1:
    timestamps = np.linspace(0.0, 2.0, count)
    progress = timestamps / timestamps[-1]
    if straight:
        position = np.c_[0.5 * progress, np.zeros(count), np.zeros(count)]
        rotations = np.repeat(np.eye(3)[None], count, axis=0)
    else:
        position = np.c_[
            0.45 * progress,
            0.08 * np.sin(2.0 * np.pi * progress),
            0.03 * np.cos(2.0 * np.pi * progress),
        ]
        rotations = np.asarray([_z_rotation(0.9 * x * x) for x in progress])
    return TrajectoryChunkV1(
        chunk_id="synthetic-stage34",
        source="stage34-test",
        observation_timestamp=0.0,
        point_timestamps=timestamps,
        eef_position=position,
        eef_orientation=matrix_to_quat(rotations),
        source_frame="NATIVE_WBC_RELATIVE_REFERENCE",
        tcp_semantic_id="x5_link6_gripper_center",
        validity_start=0.0,
        validity_end=2.0,
        gripper=np.linspace(0.0, 1.0, count),
    )


def test_geometry_uses_uniform_arc_knots_and_exact_endpoints() -> None:
    dense = _chunk()
    first = build_geometry_candidate(dense, 6)
    second = build_geometry_candidate(dense, 6)
    assert first.point_count == 6
    assert np.allclose(first.timing_arc_fraction, np.linspace(0.0, 1.0, 6))
    assert np.allclose(first.point_timestamps, second.point_timestamps)
    assert np.allclose(first.eef_position, second.eef_position)
    assert np.allclose(first.eef_orientation, second.eef_orientation)
    assert np.allclose(first.eef_position[0], dense.eef_position[0], atol=1e-12)
    assert np.allclose(first.eef_position[-1], dense.eef_position[-1], atol=1e-12)
    assert geodesic_angle(
        quat_to_matrix(first.eef_orientation[0]), quat_to_matrix(dense.eef_orientation[0])
    ) < 1e-12
    assert geodesic_angle(
        quat_to_matrix(first.eef_orientation[-1]), quat_to_matrix(dense.eef_orientation[-1])
    ) < 1e-12


def test_all_geometry_sizes_are_supported_and_line_preview_is_exact() -> None:
    dense = _chunk(straight=True)
    for k in GEOMETRY_K_VALUES:
        candidate = build_geometry_candidate(dense, k)
        result = evaluate_preview_equivalence(dense, candidate, anchor_count=101)
        assert candidate.point_count == k
        assert result["anchor_count"] == 101
        assert result["finite"] is True
        assert result["start_end_pose_exact"] is True
        assert result["duration_exact"] is True
        assert result["preview_position_error_m"]["max"] < 1e-12
        assert result["preview_rotation_error_rad"]["max"] < 1e-12


def test_timing_laws_are_monotonic_and_keep_geometry_fixed() -> None:
    dense = _chunk()
    geometry = build_geometry_candidate(dense, 6)
    for mode in TIMING_MODES:
        candidate = build_timing_candidate(dense, geometry, mode)
        assert np.all(np.diff(candidate.point_timestamps) > 0.0)
        assert np.allclose(candidate.eef_position, geometry.eef_position)
        assert np.allclose(candidate.eef_orientation, geometry.eef_orientation)
        assert np.allclose(candidate.timing_arc_fraction, TIMING_KNOTS[mode])
        assert candidate.to_trajectory_chunk().point_count == 6


def test_motion_intent_v2_rejects_identity_timing_and_nonfinite_drift() -> None:
    dense = _chunk()
    candidate = build_timing_candidate(dense, build_geometry_candidate(dense, 4), "T1")
    with pytest.raises(ValueError, match="source_frame mismatch"):
        candidate.to_trajectory_chunk(expected_source_frame="wrong")
    with pytest.raises(ValueError, match="tcp_semantic_id mismatch"):
        candidate.to_trajectory_chunk(expected_tcp_semantic_id="wrong")

    fields = dict(candidate.__dict__)
    fields["timing_arc_fraction"] = np.asarray([0.0, 0.5, 0.5])
    fields["timing_timestamps"] = np.asarray([0.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        MotionIntentV2(**fields)

    fields = dict(candidate.__dict__)
    position = candidate.eef_position.copy()
    position[0, 0] = np.nan
    fields["eef_position"] = position
    with pytest.raises(ValueError, match="non-finite"):
        MotionIntentV2(**fields)


def test_v1_contracts_remain_importable_and_separate() -> None:
    assert TrajectoryChunkV1.__name__ == "TrajectoryChunkV1"
    assert WBCPreviewV1.__name__ == "WBCPreviewV1"
    assert MotionIntentV2.__name__ == "MotionIntentV2"
    assert _chunk().rot6d.shape == (41, 6)


def test_stage34_modules_have_no_eager_model_or_scheduler_execution() -> None:
    for path in (QUALIFICATION, ORCHESTRATOR):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        assert all(name not in {"torch", "transformers", "isaacgym", "omni"} for name in imports)
        source = path.read_text(encoding="utf-8")
        assert "subprocess.run([\"qsub\"" not in source
        assert "import torch" not in source
        assert "import transformers" not in source


def test_geometry_invalid_k_fails_closed() -> None:
    with pytest.raises(ValueError):
        build_geometry_candidate(_chunk(), 5)
