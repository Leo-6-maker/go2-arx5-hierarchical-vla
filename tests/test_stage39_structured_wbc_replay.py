from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import zipfile

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
            "STAGE39_EXECUTION_BUNDLE": "/scratch/unused-bundle.tar.gz",
            "STAGE39_EXPECTED_BUNDLE_SHA256": "0" * 64,
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



def _write_v2_zip(path: Path, *, changed: bool = False) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("WBCTrackingShare/", "")
        archive.writestr("WBCTrackingShare/empty/", "")
        archive.writestr("WBCTrackingShare/a.txt", "changed" if changed else "alpha")
        archive.writestr("WBCTrackingShare/empty/b.txt", "beta")


def test_v2_tab_size_sha_lf_digest_and_live_vector(tmp_path: Path) -> None:
    archive_path = tmp_path / "authority.zip"
    _write_v2_zip(archive_path)
    archive = stage39.audit_wbc_archive(archive_path)
    live = tmp_path / "WBCTrackingShare"
    (live / "empty").mkdir(parents=True)
    (live / "a.txt").write_text("alpha")
    (live / "empty" / "b.txt").write_text("beta")
    live_audit = stage39.audit_wbc_live_tree(live)
    comparison = stage39.compare_wbc_vectors(
        archive["file_records"],
        live_audit["file_records"],
        archive["directory_names"],
        live_audit["directory_names"],
    )
    expected_payload = "".join(
        "%s\t%s\t%s\n" % (item["path"], item["size_bytes"], item["sha256"])
        for item in archive["file_records"]
    ).encode("utf-8")
    assert archive["v2_tree_sha256"] == stage39.sha256_bytes(expected_payload)
    assert live_audit["v2_tree_sha256"] == archive["v2_tree_sha256"]
    assert comparison["identical"] is True


def test_v2_live_changed_missing_extra_and_directory_deltas_are_visible(tmp_path: Path) -> None:
    live = tmp_path / "WBCTrackingShare"
    live.mkdir()
    (live / "a.txt").write_text("alpha")
    archive_files = [{"path": "WBCTrackingShare/a.txt", "size_bytes": 5, "sha256": stage39.sha256_bytes(b"alpha")}]
    changed = live / "a.txt"
    changed.write_text("delta")
    (live / "extra.txt").write_text("extra")
    audit = stage39.audit_wbc_live_tree(live)
    comparison = stage39.compare_wbc_vectors(
        archive_files,
        audit["file_records"],
        ["WBCTrackingShare", "WBCTrackingShare/empty"],
        audit["directory_names"],
    )
    assert comparison["changed_files"] == ["WBCTrackingShare/a.txt"]
    assert comparison["extra_files"] == ["WBCTrackingShare/extra.txt"]
    assert comparison["missing_directories"] == ["WBCTrackingShare/empty"]


def test_v2_live_symlink_is_rejected(tmp_path: Path) -> None:
    live = tmp_path / "WBCTrackingShare"
    live.mkdir()
    target = tmp_path / "target.txt"
    target.write_text("target")
    (live / "link.txt").symlink_to(target)
    with pytest.raises(stage39.Stage39Error, match="symlink"):
        stage39.audit_wbc_live_tree(live)


def test_zip_unsafe_path_and_symlink_members_are_rejected(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../escape", "x")
    with pytest.raises(stage39.Stage39Error, match="unsafe WBC archive members"):
        stage39.audit_wbc_archive(unsafe)

    symlink_zip = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("WBCTrackingShare/link")
    info.create_system = 3
    info.external_attr = (0o120777 << 16)
    with zipfile.ZipFile(symlink_zip, "w") as archive:
        archive.writestr(info, "target")
    with pytest.raises(stage39.Stage39Error, match="unsafe WBC archive members"):
        stage39.audit_wbc_archive(symlink_zip)


def test_authority_source_identity_is_exact_and_alias_configured() -> None:
    host = stage39.authority_source_identity(stage39.EXPECTED_WBC_SOURCE_ROOT)
    alias = stage39.authority_source_identity(Path("/authority/WBCTrackingShare"))
    assert host["source_root_identity_class"] == "HOST_EXACT_PATH"
    assert alias["source_root_identity_class"] == "CONTAINER_READONLY_BIND_ALIAS"
    with pytest.raises(stage39.Stage39Error, match="accepted host path or container alias"):
        stage39.authority_source_identity(Path("/tmp/third-authority-root"))


def test_historical_digest_is_diagnostic_not_current_v2_predicate() -> None:
    assert stage39.HISTORICAL_REPORTED_WBC_TREE_SHA256 != stage39.EXPECTED_WBC_V2_TREE_SHA256
    assert stage39.HISTORICAL_REPORTED_WBC_TREE_SHA256.startswith("63311df")
    assert stage39.EXPECTED_WBC_V2_TREE_SHA256.startswith("9d653c39")
