from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from scripts import stage36_wbc_authority_oracle_qualification as stage36


REPO = Path(__file__).resolve().parents[1]


def test_stage36_gate_and_identity_constants_are_current() -> None:
    assert stage36.PI_GATE_ID == "PI_GATE_STATE_0023"
    assert stage36.BASE_HEAD == "7c1e864ee975e4e94196792ed8408f326fb4dfb9"
    assert stage36.NEXT_GATE == "GPT_PI_REVIEW_AFTER_STAGE36_LONG_HORIZON_LADDER"
    assert stage36.CASE_IDS == tuple(
        f"{family}{index:02d}"
        for family in ("N", "T", "P", "C")
        for index in range(1, 9)
    )


def test_zip_member_safety_is_fail_closed() -> None:
    safe = (
        "WBCTrackingShare/modules/trajectory.py",
        "WBCTrackingShare/",
    )
    unsafe = (
        "",
        "/tmp/escape",
        "../escape",
        "WBCTrackingShare/../../escape",
        "WBCTrackingShare\\escape",
    )
    assert all(stage36.is_safe_zip_member(value) for value in safe)
    assert not any(stage36.is_safe_zip_member(value) for value in unsafe)


def test_stage36_campaign_has_no_scheduler_or_qsub_path() -> None:
    source = (REPO / "scripts/stage36_wbc_oracle_campaign.py").read_text()
    tree = ast.parse(source)
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "qsub"
        for node in ast.walk(tree)
    )
    assert "subprocess.run" in source
    assert "BENCHMARK_CONTINUOUS" in source
    assert "optimizer.step(" not in source
    assert "scheduler.step(" not in source
    assert ".backward(" not in source


def test_pbs_is_auto_only_and_requests_one_h200_slot() -> None:
    source = (REPO / "hpc/pbs/stage36_wbc_oracle_qualification.pbs").read_text()
    assert "#PBS -q auto" in source
    assert "#PBS -P CFP03-CF-004" in source
    assert "select=1:ncpus=12:ngpus=1" in source
    for queue in ("small", "medium", "large", "special"):
        assert "#PBS -q %s" % queue not in source


def test_campaign_plan_rows_are_dense_only_when_validated(tmp_path: Path) -> None:
    plan = {
        "schema": "STAGE36_WBC_ORACLE_CAMPAIGN_PLAN_V3",
        "pi_gate": "PI_GATE_STATE_0023",
        "plan_status": "READY_FOR_E",
        "project": "CFP03-CF-004",
        "queue": "auto",
        "one_pbs_allocation_only": True,
        "maximum_rows": 96,
        "row_count": 1,
        "primary_row_count": 1,
        "robustness_row_count": 0,
        "representations": ["DENSE"],
        "sparse_rows": 0,
        "native_transport": {
            "sample_count": 121,
            "duration_s": 2.0,
            "dt_s": 1.0 / 60.0,
            "shared_across_representations": True,
        },
        "rows": [
            {
                "row_id": "dense_N01_seed13",
                "case_id": "N01",
                "family": "N",
                "representation": "DENSE",
                "reference_mode": "BENCHMARK_CONTINUOUS",
                "seed": 13,
                "phase": "PRIMARY",
                "seconds": 2.0,
                "trajectory_path": str(tmp_path / "N01.npz"),
                "metadata_path": str(tmp_path / "N01.json"),
            }
        ],
    }
    (tmp_path / "N01.npz").write_bytes(b"input")
    (tmp_path / "N01.json").write_text("{}")
    plan["rows"][0]["trajectory_sha256"] = hashlib.sha256(
        (tmp_path / "N01.npz").read_bytes()
    ).hexdigest()
    plan["rows"][0]["metadata_sha256"] = hashlib.sha256(
        (tmp_path / "N01.json").read_bytes()
    ).hexdigest()
    from scripts import stage36_wbc_oracle_campaign as campaign

    assert campaign.validate_plan(plan, tmp_path) == ["non_scratch_input:dense_N01_seed13"]
