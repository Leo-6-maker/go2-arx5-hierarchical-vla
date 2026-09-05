from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

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


def test_pbs_accepts_auto_route_destination_without_changing_submit_queue() -> None:
    source = (REPO / "hpc/pbs/stage36_wbc_oracle_qualification.pbs").read_text()
    assert 'test "${PBS_O_QUEUE:-}" = "auto"' in source
    assert 'test "${PBS_QUEUE:-auto}" = "auto"' not in source
    assert "STAGE36_PBS_REQUEST_QUEUE" in source
    assert "STAGE36_PBS_EXECUTION_QUEUE" in source


def test_pbs_binds_an_explicit_container_runtime_path() -> None:
    source = (REPO / "hpc/pbs/stage36_wbc_oracle_qualification.pbs").read_text()
    assert "STAGE36_APPTAINER_BIN" in source
    assert "/cm/shared/apps/singularity/1.3.1/bin/apptainer" in source
    assert '"${APPTAINER_BIN}" exec --nv --cleanenv' in source


def test_pbs_binds_exact_authority_root_and_preserves_job_id() -> None:
    source = (REPO / "hpc/pbs/stage36_wbc_oracle_qualification.pbs").read_text()
    assert '"${STAGE36_AUTHORITY_EXTRACTED_ROOT}:${CONTAINER_AUTHORITY}:ro"' in source
    assert 'PBS_JOBID="${PBS_JOBID}"' in source
    assert '"${AUTHORITY_PARENT}:${CONTAINER_AUTHORITY}:ro"' not in source


def test_pbs_preserves_apptainer_nvidia_driver_paths() -> None:
    source = (REPO / "hpc/pbs/stage36_wbc_oracle_qualification.pbs").read_text()
    assert 'PATH="/usr/local/nvidia/bin:${CONTAINER_NINJA%/ninja}:${PATH}"' in source
    assert "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs" in source


def test_campaign_accepts_only_fresh_wrapper_scaffold(tmp_path: Path) -> None:
    from scripts import stage36_wbc_oracle_campaign as campaign

    output_root = tmp_path / "stage36_eval_123"
    (output_root / "runtime" / "home").mkdir(parents=True)
    (output_root / "runtime" / "tmp").mkdir()
    (output_root / "stage36_job.log").write_text("wrapper\n")

    campaign.ensure_output_root_scaffold(output_root)

    (output_root / "stage36_result.json").write_text("{}\n")
    with pytest.raises(SystemExit, match="scaffold is not fresh"):
        campaign.ensure_output_root_scaffold(output_root)


def test_campaign_return_does_not_claim_model_before_runtime_qualification() -> None:
    from scripts import stage36_wbc_oracle_campaign as campaign

    result = {"runtime_qualification": None, "scientific_execution_begun": False}
    payload = campaign.make_return(
        Path("/scratch/stage36-test-output"),
        result,
        Path("/scratch/stage36-test-output/stage36_artifact_seal.json"),
        None,
        "BLOCKED",
    )
    assert payload["MODEL_CONSTRUCTED"] is False
    assert payload["GR00T_MODEL_CONSTRUCTED"] is False


def test_complete_native_output_with_teardown_signal_is_immutable_result(tmp_path: Path) -> None:
    from scripts import stage36_wbc_oracle_campaign as campaign

    output_dir = tmp_path / "row"
    output_dir.mkdir()
    summary = {
        "status": "NATIVE_BENCHMARK_CONTINUOUS_PASS",
        "termination": "REFERENCE_COMPLETE",
        "steps": 1,
        "reference_completed": True,
        "error": None,
        "metrics": {"position_m": {"rmse": 0.0}},
    }
    raw_row = {
        "step": 0,
        "termination_code": "REFERENCE_COMPLETE",
        "reference_complete": True,
        "fall_flag": False,
        "invalid_state": False,
        "joint_limit_violation": True,
        "joint_limit_termination": False,
        "base_pos": [0.0, 0.0, 0.0],
    }
    (output_dir / "native_external_summary.json").write_text(
        __import__("json").dumps(summary) + "\n"
    )
    (output_dir / "native_external_rows.jsonl").write_text(
        __import__("json").dumps(raw_row) + "\n"
    )
    plan_row = {
        "row_id": "dense_N01_seed13_primary",
        "case_id": "N01",
        "family": "N",
        "representation": "DENSE",
        "geometry_k": None,
        "timing_mode": None,
        "phase": "PRIMARY",
        "seed": 13,
        "reference_mode": "BENCHMARK_CONTINUOUS",
        "seconds": 2.0,
        "trajectory_path": "/scratch/input.npz",
        "trajectory_sha256": "a" * 64,
    }

    row = campaign.row_result(plan_row, output_dir, -11, False)

    assert row["scientific_row_finalized"] is True
    assert row["post_run_completion_signal"] is True
    assert row["runner_pass"] is True
    assert row["tracking_pass"] is False
    assert row["safety_gate"]["safety_violation"] is True


def test_resume_projection_ignores_allocation_identity_but_keeps_science() -> None:
    from scripts import stage36_wbc_oracle_campaign as campaign

    first = {
        "timestamp_utc": "old",
        "execution_head": "old-head",
        "native_runner_sha256": "old-runner",
        "authority_manifest_path": "/old/manifest.json",
        "row_count": 32,
        "native_transport": {"sample_count": 121},
    }
    second = dict(first)
    second.update(
        {
            "timestamp_utc": "new",
            "execution_head": "new-head",
            "native_runner_sha256": "new-runner",
            "authority_manifest_path": "/new/manifest.json",
        }
    )

    assert campaign.resume_plan_fingerprint(first) == campaign.resume_plan_fingerprint(second)


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
