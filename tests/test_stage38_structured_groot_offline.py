"""Regression tests for the Stage38-A timing-capacity boundary."""

from __future__ import annotations

import ast
import sys

import numpy as np
import pytest


REPO = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import stage38_structured_groot_offline as stage38  # noqa: E402
from go2arx5_vla.contracts import TrajectoryChunkV1  # noqa: E402
from go2arx5_vla.motion_api import (  # noqa: E402
    STAGE38_TIMING_KNOTS,
    STAGE38_TIMING_MODES,
    build_geometry_candidate,
    build_timing_candidate_with_knots,
)
from go2arx5_vla.oracle import build_case  # noqa: E402


def _chunk() -> TrajectoryChunkV1:
    payload = build_case(0, "N")
    return TrajectoryChunkV1(
        chunk_id=payload["case_id"],
        source="stage38-test",
        observation_timestamp=0.0,
        point_timestamps=np.asarray(payload["timestamps"], dtype=np.float64),
        eef_position=np.asarray(payload["eef_position_m"], dtype=np.float64),
        eef_orientation=np.asarray(payload["eef_orientation_xyzw"], dtype=np.float64),
        source_frame=payload["source_frame"],
        tcp_semantic_id=payload["tcp_semantic_id"],
        validity_start=0.0,
        validity_end=2.0,
        gripper=np.asarray(payload["gripper"], dtype=np.float64),
    )


def test_stage38_timing_scope_is_preregistered_and_distinct_from_stage37() -> None:
    assert tuple(sorted(STAGE38_TIMING_MODES, key=lambda value: int(value[1:]))) == (
        "T16",
        "T20",
        "T24",
        "T32",
        "T40",
    )
    assert stage38.STAGE38_K_CAPACITY_STAR == 16
    assert stage38.STAGE37_LIVE_HEAD == stage38.EXPECTED_PARENT_HEAD
    assert stage38.STAGE37_SEAL_SHA256 == "4a606636d4769367a97b2386f4e202be3ae97ac7d63750631919fa9710fdeb87"


def test_stage38_timing_builder_has_uniform_knots_monotonic_law_and_exact_duration() -> None:
    dense = _chunk()
    geometry = build_geometry_candidate(dense, 16, allowed_k_values=(16,))
    for mode, knots in STAGE38_TIMING_KNOTS.items():
        candidate = build_timing_candidate_with_knots(dense, geometry, mode, knots)
        assert candidate.timing_mode == mode
        assert candidate.timing_arc_fraction.size == int(mode[1:])
        assert np.all(np.diff(candidate.timing_timestamps) > 0.0)
        assert np.isclose(candidate.timing_timestamps[0], 0.0)
        assert np.isclose(candidate.timing_timestamps[-1], 2.0)
        assert candidate.provenance["timing_interpolation"] == "MONOTONIC_PIECEWISE_LINEAR"


def test_stage38_timing_builder_rejects_nonuniform_or_wrong_count_knots() -> None:
    dense = _chunk()
    geometry = build_geometry_candidate(dense, 16, allowed_k_values=(16,))
    with pytest.raises(ValueError, match="uniformly spaced"):
        build_timing_candidate_with_knots(
            dense, geometry, "T16", np.linspace(0.0, 1.0, 16) ** 2
        )
    with pytest.raises(ValueError, match="exactly 16"):
        build_timing_candidate_with_knots(
            dense, geometry, "T16", np.linspace(0.0, 1.0, 15)
        )


def test_stage38_static_guard_has_no_model_or_allocation_path() -> None:
    guard = stage38.static_guard()
    assert guard["status"] == "PASS"
    assert guard["cpu_zero_model_no_scheduler_path"] is True
    assert guard["allocation_requested"] is False


def test_stage38_script_ast_has_no_forbidden_import_roots() -> None:
    tree = ast.parse(stage38.Path(stage38.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert not roots.intersection({"torch", "transformers", "isaacgym", "omni"})
