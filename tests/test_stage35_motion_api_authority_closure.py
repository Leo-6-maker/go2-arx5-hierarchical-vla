from __future__ import annotations

import ast
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest

from go2arx5_vla.contracts import MotionIntentV2, TrajectoryChunkV1
from go2arx5_vla.adapters import GR00TToWBCAdapter, ReferenceAdapterConfig
from go2arx5_vla.motion_api import build_geometry_candidate, build_timing_candidate
from go2arx5_vla.oracle import build_case


REPO = Path(__file__).resolve().parents[1]
BASE_HEAD = "6037c392966715d516035507ddef6c1e8b07de6d"
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_ID = "x5_link6_gripper_center"


def _chunk(family: str = "T", index: int = 0) -> TrajectoryChunkV1:
    payload = build_case(index, family)
    return TrajectoryChunkV1(
        chunk_id=payload["case_id"],
        source="stage35-test",
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


def test_v2_materializes_u_then_gamma_and_preserves_endpoints() -> None:
    dense = _chunk()
    geometry = build_geometry_candidate(dense, 6)
    timing = build_timing_candidate(dense, geometry, "T9")
    query = np.linspace(0.0, 2.0, 37)
    u = np.asarray(timing.progress_at(query))
    position, orientation = timing.pose_at_progress(u)
    chunk = timing.to_trajectory_chunk(
        expected_source_frame=SOURCE_FRAME,
        expected_tcp_semantic_id=TCP_ID,
        query_timestamps=query,
    )
    assert np.allclose(chunk.eef_position, position, atol=1e-12, rtol=0.0)
    assert np.allclose(chunk.eef_orientation, orientation, atol=1e-12, rtol=0.0)
    assert np.allclose(timing.pose_at_progress(0.0)[0], timing.eef_position[0], atol=1e-12)
    assert np.allclose(timing.pose_at_progress(1.0)[0], timing.eef_position[-1], atol=1e-12)
    assert float(timing.progress_at(-1.0)) == 0.0
    assert float(timing.progress_at(3.0)) == 1.0
    assert chunk.provenance["materialization"] == "u(t)_then_Gamma(u)"


def test_full_density_t0_and_existing_adapter_path_are_exact() -> None:
    dense = _chunk("N", 0)
    intent = MotionIntentV2.from_trajectory_chunk(dense, timing_mode="T0")
    rebuilt = intent.to_trajectory_chunk(
        expected_source_frame=SOURCE_FRAME,
        expected_tcp_semantic_id=TCP_ID,
        query_timestamps=dense.point_timestamps,
    )
    assert np.allclose(rebuilt.point_timestamps, dense.point_timestamps, atol=1e-12, rtol=0.0)
    assert np.allclose(rebuilt.eef_position, dense.eef_position, atol=1e-12, rtol=0.0)
    assert np.allclose(rebuilt.eef_orientation, dense.eef_orientation, atol=1e-12, rtol=0.0)
    adapter = GR00TToWBCAdapter(
        ReferenceAdapterConfig(
            expected_source_frame=SOURCE_FRAME,
            wbc_frame=SOURCE_FRAME,
            tcp_semantic_id=TCP_ID,
        )
    )
    preview, gripper = adapter.adapt(rebuilt, controller_timestamp=0.25)
    assert np.isfinite(preview.eef_position).all()
    assert np.isfinite(preview.eef_rot6d).all()
    assert gripper is not None


def test_same_geometry_different_timing_is_causally_separate() -> None:
    dense = _chunk()
    geometry = build_geometry_candidate(dense, 9)
    t1 = build_timing_candidate(dense, geometry, "T1")
    t9 = build_timing_candidate(dense, geometry, "T9")
    u = np.linspace(0.0, 1.0, 101)
    p1, q1 = t1.pose_at_progress(u)
    p9, q9 = t9.pose_at_progress(u)
    times = np.linspace(0.0, 2.0, 101)
    assert np.allclose(p1, p9, atol=1e-12, rtol=0.0)
    assert np.allclose(q1, q9, atol=1e-12, rtol=0.0)
    assert np.max(np.abs(t1.progress_at(times) - t9.progress_at(times))) > 1e-6


def test_same_timing_different_geometry_keeps_progress_law() -> None:
    dense = _chunk()
    k4 = build_timing_candidate(dense, build_geometry_candidate(dense, 4), "T9")
    k12 = build_timing_candidate(dense, build_geometry_candidate(dense, 12), "T9")
    times = np.linspace(0.0, 2.0, 101)
    assert np.allclose(k4.progress_at(times), k12.progress_at(times), atol=1e-12, rtol=0.0)
    p4, _ = k4.pose_at_progress(np.linspace(0.0, 1.0, 101))
    p12, _ = k12.pose_at_progress(np.linspace(0.0, 1.0, 101))
    assert np.max(np.linalg.norm(p4 - p12, axis=-1)) > 1e-6


def test_geometry_arc_fraction_is_explicit_and_round_trips() -> None:
    dense = _chunk("N", 1)
    intent = MotionIntentV2.from_trajectory_chunk(dense)
    assert np.allclose(intent.geometry_arc_fraction, intent.timing_arc_fraction)
    rebuilt = MotionIntentV2(**intent.__dict__)
    assert np.allclose(rebuilt.geometry_arc_fraction, intent.geometry_arc_fraction)
    assert np.allclose(rebuilt.as_dict()["geometry_arc_fraction"], intent.geometry_arc_fraction)


def test_v2_rejects_invalid_geometry_progress_and_query_grid() -> None:
    dense = _chunk()
    candidate = build_geometry_candidate(dense, 4)
    fields = dict(candidate.__dict__)
    fields["geometry_arc_fraction"] = np.asarray([0.0, 0.5, 0.5, 1.0])
    with pytest.raises(ValueError, match="geometry arc fraction"):
        MotionIntentV2(**fields)
    with pytest.raises(ValueError, match="query_timestamps"):
        candidate.to_trajectory_chunk(query_timestamps=np.asarray([0.0, 1.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="finite"):
        candidate.pose_at_progress(np.asarray([0.0, np.nan, 1.0]))


def test_identity_guards_fail_closed() -> None:
    candidate = build_geometry_candidate(_chunk(), 4)
    with pytest.raises(ValueError, match="source_frame mismatch"):
        candidate.to_trajectory_chunk(expected_source_frame="wrong")
    with pytest.raises(ValueError, match="tcp_semantic_id mismatch"):
        candidate.to_trajectory_chunk(expected_tcp_semantic_id="wrong")


def test_v1_contract_files_are_unchanged_from_stage34() -> None:
    v1_files = (
        "src/go2arx5_vla/contracts/trajectory.py",
        "src/go2arx5_vla/contracts/trajectory_shape.py",
        "src/go2arx5_vla/trajectory/preview.py",
        "src/go2arx5_vla/adapters/groot_to_wbc.py",
        "src/go2arx5_vla/diagnostics/stage23_action_pipeline.py",
    )
    for relative in v1_files:
        current = hashlib.sha256((REPO / relative).read_bytes()).hexdigest()
        base = subprocess.run(
            ["git", "show", f"{BASE_HEAD}:{relative}"],
            cwd=REPO,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        assert current == hashlib.sha256(base).hexdigest(), relative


def test_stage35_runner_and_corpus_are_zero_model_static_code() -> None:
    paths = (
        REPO / "scripts/stage35_motion_api_authority_closure.py",
        REPO / "src/go2arx5_vla/oracle/stage35_corpus.py",
    )
    forbidden_imports = {"torch", "transformers", "isaacgym", "omni"}
    forbidden_calls = {"qsub", "train", "backward", "step", "resume", "get_action", "from_pretrained"}
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert not imports & forbidden_imports
        assert not calls & forbidden_calls
