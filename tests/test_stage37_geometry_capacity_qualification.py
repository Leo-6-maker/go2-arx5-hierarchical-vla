"""Regression tests for the CPU-only Stage37 admission and plan boundary."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import stage36_wbc_oracle_campaign as campaign  # noqa: E402
import stage37_geometry_capacity_qualification as stage37  # noqa: E402


def test_stage37_scope_is_frozen() -> None:
    assert stage37.STAGE37_K_VALUES == (16, 20, 24, 32, 40)
    assert stage37.STAGE37_TIMING_MODES == ("T1", "T3", "T5", "T9")
    assert stage37.NATIVE_SAMPLE_COUNT == 121
    assert stage37.NATIVE_DT_S == 1.0 / 60.0
    assert stage37.NATIVE_DURATION_S == 2.0


def test_g0_predicates_report_exact_frozen_limits() -> None:
    row = {
        "finite": True,
        "monotonic_timestamps": True,
        "preview_position_error_m": {"p95": stage37.DEFAULT_PREVIEW_POSITION_P95_LIMIT_M},
        "preview_rotation_error_rad": {"p95": 0.0},
        "endpoint_clamp_agreement": 1.0,
        "start_end_pose_exact": True,
        "duration_exact": True,
    }
    assert stage37.predicate_failures(row) == ["POSITION_P95"]
    margins = stage37.predicate_margins(row)
    assert margins["position_p95_margin_m"] == 0.0
    assert margins["rotation_p95_margin_rad"] == stage37.DEFAULT_PREVIEW_ROTATION_P95_LIMIT_RAD


def test_stage37_structured_plan_accepts_k16_without_dense_rows() -> None:
    with tempfile.TemporaryDirectory(dir="/scratch/e1192548") as directory:
        root = Path(directory)
        trajectory = root / "k16_t0_N01_seed13_primary.npz"
        metadata = root / "k16_t0_N01_seed13_primary.json"
        trajectory.write_bytes(b"stage37-test-trajectory")
        metadata.write_text(json.dumps({"schema": "test"}) + "\n", encoding="utf-8")
        row = {
            "row_id": "k16_t0_N01_seed13_primary",
            "case_id": "N01",
            "family": "N",
            "representation": "K16+T0",
            "geometry_k": 16,
            "timing_mode": "T0",
            "phase": "PRIMARY",
            "seed": 13,
            "reference_mode": "BENCHMARK_CONTINUOUS",
            "seconds": 2.0,
            "trajectory_path": str(trajectory),
            "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
            "metadata_path": str(metadata),
            "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        }
        plan = {
            "schema": "STAGE36_WBC_ORACLE_CAMPAIGN_PLAN_V3",
            "campaign_kind": "STAGE37_STRUCTURED_ORACLE",
            "pi_gate": "PI_GATE_STATE_0025",
            "plan_status": "READY_FOR_E",
            "project": "CFP03-CF-004",
            "queue": "auto",
            "one_pbs_allocation_only": True,
            "representations": ["K16+T0"],
            "rows": [row],
            "row_count": 1,
            "primary_row_count": 1,
            "robustness_row_count": 0,
            "sparse_rows": 1,
            "native_transport": {
                "sample_count": 121,
                "duration_s": 2.0,
                "shared_across_representations": True,
            },
        }
        assert campaign.validate_plan(plan, root) == []


def test_stage37_cpu_ladder_has_no_model_or_scheduler_import_path() -> None:
    guard = stage37.static_guard()
    assert guard["status"] == "PASS"
    assert guard["cpu_ladder_has_no_model_or_scheduler_path"] is True
    assert guard["queue_auto_only"] is True
