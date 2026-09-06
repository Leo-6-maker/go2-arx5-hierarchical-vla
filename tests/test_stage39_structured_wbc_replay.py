from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

import stage39_structured_wbc_replay as stage39


def _snapshot() -> dict:
    timestamps = np.linspace(0.0, stage39.EXPECTED_DURATION_S, stage39.EXPECTED_POINT_COUNT)
    physical = np.zeros((stage39.EXPECTED_POINT_COUNT, 10), dtype=np.float64)
    physical[:, 0] = np.linspace(0.0, 0.2, stage39.EXPECTED_POINT_COUNT)
    physical[:, 3] = 1.0
    physical[:, 7] = 1.0
    physical[:, 9] = 0.5
    return {
        "example_id": "test",
        "observation_timestamp": 0.0,
        "future_timestamps": timestamps.tolist(),
        "state_v1": {"values": physical[0].tolist()},
        "source_frame": stage39.EXPECTED_FRAME,
        "tcp_semantic_id": stage39.EXPECTED_TCP,
    }, physical


def test_structured_oracle_preserves_source_duration_and_shape() -> None:
    snapshot, physical = _snapshot()
    chunk, metadata = stage39.reconstruct_c1_oracle(physical, snapshot)
    assert chunk.point_count == 40
    assert np.array_equal(chunk.point_timestamps, np.asarray(snapshot["future_timestamps"]))
    assert chunk.validity_end - chunk.validity_start == stage39.EXPECTED_DURATION_S
    assert metadata["geometry_k"] == 16
    assert metadata["timing_mode"] == "T16"
    assert metadata["roundtrip_finite"] is True
    assert chunk.source_frame == stage39.EXPECTED_FRAME
    assert chunk.tcp_semantic_id == stage39.EXPECTED_TCP


def test_issue_comparison_requires_identity_and_keeps_warning_details() -> None:
    issue = {
        "code": "LINEAR_SPEED_WARN",
        "severity": "WARNING",
        "message": "speed",
        "provenance": "TEST_ONLY",
        "observed": 1.0,
        "threshold": 1.0,
    }
    comparison = stage39.compare_issue_lists([issue], [dict(issue)])
    assert comparison["match"] is True
    changed = dict(issue, provenance="CONTROLLER_CONTRACT")
    assert stage39.compare_issue_lists([issue], [changed])["match"] is False


def test_native_finalized_does_not_accept_empty_or_timeout() -> None:
    summary = {
        "status": "NATIVE_BENCHMARK_40POINT_RECONSTRUCTED_PASS",
        "error": None,
        "termination": "REFERENCE_COMPLETE",
        "steps": 1,
        "reference_completed": True,
    }
    row = {"step": 0, "finite": True}
    assert stage39.native_finalized(summary, [row], False) is True
    assert stage39.native_finalized(summary, [row], True) is False
    assert stage39.native_finalized(summary, [], False) is False


def test_campaign_command_is_external_40point_mode_without_retries() -> None:
    command = stage39.campaign_command(
        Path("native.py"),
        Path("/authority/WBCTrackingShare"),
        Path("arm.pt"),
        Path("dog.pt"),
        {
            "row_id": "C0_dense_expert_tiny1",
            "reference_class": "DENSE_EXPERT_TINY1",
            "seed": None,
            "trajectory_path": "/scratch/input.npz",
            "metadata_path": "/scratch/input.json",
        },
        Path("/scratch/out"),
    )
    text = " ".join(command)
    assert "--reference-mode BENCHMARK_40POINT_RECONSTRUCTED" in text
    assert "--seconds %s" % stage39.EXPECTED_DURATION_S in text
    assert "--resume" not in text
    assert "--no-stop-at-reference-completion" not in text
    assert "--disable-trajectory-termination" not in text


def test_severity_counts_is_explicit() -> None:
    assert stage39.severity_counts(
        [{"decision": "ACCEPT"}, {"decision": "WARN"}, {"decision": "WARN"}]
    ) == {"ACCEPT": 1, "WARN": 2, "REJECT": 0}


def test_execution_authority_is_separate_from_scientific_protocol() -> None:
    records = [
        stage39.validate_execution_provenance(
            "PI_GATE_STATE_0028",
            authority,
            "/scratch/stage39-pbs.log",
            stage39.REPO,
        )
        for authority in ("PI_GATE_STATE_0030", "PI_GATE_STATE_0031")
    ]
    assert [record["execution_authority"] for record in records] == [
        "PI_GATE_STATE_0030",
        "PI_GATE_STATE_0031",
    ]
    assert all(record["scientific_protocol_id"] == "PI_GATE_STATE_0028" for record in records)
    assert all(record["protocol_identity_immutable"] is True for record in records)


def test_execution_provenance_rejects_wrong_protocol_and_internal_stdout() -> None:
    with pytest.raises(stage39.Stage39Error, match="scientific protocol identity mismatch"):
        stage39.validate_execution_provenance(
            "PI_GATE_STATE_0030",
            "PI_GATE_STATE_0030",
            "/scratch/stage39-pbs.log",
            stage39.REPO,
        )
    with pytest.raises(stage39.Stage39Error, match="inside the checkout"):
        stage39.validate_execution_provenance(
            "PI_GATE_STATE_0028",
            "PI_GATE_STATE_0030",
            str(stage39.REPO / "stage39-pbs.log"),
            stage39.REPO,
        )


def test_wrapper_emits_named_fail_closed_protocol_diagnostic() -> None:
    wrapper = stage39.REPO / "hpc/pbs/stage39_structured_wbc_replay.pbs"
    environment = os.environ.copy()
    environment.update(
        {
            "STAGE39_REPO": str(stage39.REPO),
            "STAGE39_ADMISSION_ROOT": "/scratch/unused-admission",
            "STAGE39_OUTPUT_PARENT": "/scratch/unused-output",
            "STAGE39_AUTHORITY_EXTRACTED_ROOT": "/scratch/unused-authority",
            "STAGE39_RUNTIME_SITE": "/scratch/unused-site",
            "STAGE39_RUNTIME_NINJA": "/scratch/unused-ninja",
            "STAGE39_SCIENTIFIC_PROTOCOL_ID": "PI_GATE_STATE_0030",
            "STAGE39_EXECUTION_AUTHORITY": "PI_GATE_STATE_0030",
            "STAGE39_PBS_STDOUT_PATH": "/scratch/stage39-pbs.log",
            "STAGE39_EXPECTED_HEAD": "c88d32f6d9887916a873e292a0e1ed5c216be136",
            "STAGE39_EXPECTED_ADMISSION_SHA256": "0" * 64,
            "STAGE39_EXPECTED_MANIFEST_SHA256": "0" * 64,
            "STAGE39_EXPECTED_WARN_SHA256": "0" * 64,
            "PBS_O_QUEUE": "auto",
        }
    )
    completed = subprocess.run(
        ["bash", str(wrapper)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 1
    output = completed.stdout + completed.stderr
    assert "FAIL_CLOSED_PRE_SCIENCE" in output
    assert "check=scientific_protocol_id" in output
    assert "expected=PI_GATE_STATE_0028" in output
    assert "observed=PI_GATE_STATE_0030" in output


def test_wrapper_has_no_ambiguous_gate_variable_and_requires_external_stdout() -> None:
    wrapper_text = (stage39.REPO / "hpc/pbs/stage39_structured_wbc_replay.pbs").read_text(
        encoding="utf-8"
    )
    assert "STAGE39_PI_GATE" not in wrapper_text
    assert "STAGE39_SCIENTIFIC_PROTOCOL_ID" in wrapper_text
    assert "STAGE39_EXECUTION_AUTHORITY" in wrapper_text
    assert "STAGE39_PBS_STDOUT_PATH" in wrapper_text
    assert "FAIL_CLOSED_PRE_SCIENCE" in wrapper_text
    assert "CHECK stdout_external PASS" in wrapper_text
