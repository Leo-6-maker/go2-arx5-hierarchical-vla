#!/usr/bin/env python3
"""Build and verify the immutable PI0086 Stage45B B2 production release.

The release is an external, scratch-only manifest.  It binds the final
post-merge Stage45B checkout, the production entrypoint bytes, the exact
runtime evidence, the external admission bundle, and every scientific input
identity needed by the one authorized H200 job.  This module is deliberately
stdlib-only and performs no network access or model imports.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Mapping

from stage45b_external_admission import (
    BASE_TREE_SHA256,
    B0_PAYLOAD_SHA256,
    B0_STATS_SHA256,
    BRANCH,
    CONFIG_SHA256,
    COSMOS_REVISION,
    COSMOS_TREE_SHA256,
    FULL24_TREE_SHA256,
    GROOT_ARCHIVE_SHA256,
    GROOT_COMMIT,
    GROOT_MANIFEST_FILE_SHA256,
    GROOT_MANIFEST_PAYLOAD_SHA256,
    GROOT_PAYLOAD_TREE_SHA256,
    GROOT_TREE,
    GROOT_SNAPSHOT_SCHEMA,
    MODALITY_SHA256,
    PI_GATE,
    RUNTIME_PAYLOAD_SHA256,
    RUNTIME_PROTOCOL_REVISION,
    RUNTIME_STATS_SCHEMA,
    RUNTIME_STATS_SHA256,
    SCHEMA as ADMISSION_SCHEMA,
    assert_bundle,
)
from stage45b_groot_source_snapshot import sha256_file, verify_snapshot


SCHEMA = "STAGE45B_B2_TRAINING_RELEASE_V1"
STATUS = "SEALED"
TOKEN = "VANDA_STAGE45B_B2_PRODUCTION_TRAINING_RELEASE_V1_QUALIFIED"
NEXT_GATE = "GPT_PI_REVIEW_AFTER_STAGE45B_B2_PI0086_RETURN"
EXPECTED_PYTHON = "/scratch/e1192548/stage22_h200_runtime_deb920_20260902/runtime/venv/bin/python3"
EXPECTED_PYTHON_REALPATH = "/app1/Hopper/shared/apps/python/3.12.3/bin/python3.12"
EXPECTED_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}
EXECUTION_PATHS = {
    "pbs": "hpc/pbs/stage45b_full24_fresh_base.pbs",
    "runner": "scripts/stage45b_full24_fresh_base_train.py",
    "premodel": "scripts/stage45b_premodel_runtime_smoke.py",
    "source_verifier": "scripts/stage45b_groot_source_snapshot.py",
    "external_admission": "scripts/stage45b_external_admission.py",
    "release_verifier": "scripts/stage45b_production_release.py",
    "runtime_qualification": "scripts/stage45b_full24_runtime_parquet_qualification.py",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise RuntimeError(f"required release JSON is unavailable: {path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise RuntimeError(f"required release JSON is not a regular file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def regular_file(path: Path, label: str, *, immutable: bool = False) -> None:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise RuntimeError(f"{label} is not a regular file: {path}")
    if immutable and mode & 0o222:
        raise RuntimeError(f"{label} is writable: {path}")


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not value or ".." in path.parts:
        raise RuntimeError(f"release execution path is not repository-relative: {value!r}")
    return path


def _equal(mapping: Mapping[str, Any], key: str, expected: Any, label: str) -> None:
    if mapping.get(key) != expected:
        raise RuntimeError(f"{label}.{key} mismatch: {mapping.get(key)!r} != {expected!r}")


def _expected_checkpoint_policy() -> dict[str, Any]:
    return {
        "adaptive_extension": False,
        "checkpoint_1000": "required descriptive checkpoint with full trainer state",
        "checkpoint_2000": "required fixed primary endpoint with full trainer state",
        "loss_based_selection": False,
        "required_state_files": ["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"],
    }


def _entrypoint_identity(repo: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, relative in EXECUTION_PATHS.items():
        path = repo / relative
        regular_file(path, f"release entrypoint {key}")
        result[key] = {"path": relative, "sha256": sha256_file(path)}
    return result


def _runtime_from_evidence(evidence: Mapping[str, Any], *, expected_head: str | None = None) -> dict[str, Any]:
    if evidence.get("status") != "PASS":
        raise RuntimeError("release runtime evidence is not PASS")
    if expected_head is not None and evidence.get("execution_head") != expected_head:
        raise RuntimeError("release runtime evidence is bound to a different Stage45B HEAD")
    if evidence.get("branch") != BRANCH:
        raise RuntimeError("release runtime evidence is bound to a different Stage45B branch")
    runtime = evidence.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("release runtime evidence has no runtime record")
    requested = runtime.get("requested_interpreter")
    realpath = runtime.get("sys_executable_realpath") or runtime.get("readlink_f")
    for label, value in (("requested_interpreter", requested), ("sys_executable_realpath", realpath)):
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"release runtime evidence lacks {label}")
    if requested != EXPECTED_PYTHON or realpath != EXPECTED_PYTHON_REALPATH:
        raise RuntimeError("release runtime interpreter identity differs from PI0086")
    for key in ("python_version", "torch_version", "torch_cuda_build"):
        if not isinstance(runtime.get(key), str) or not runtime[key]:
            raise RuntimeError(f"release runtime evidence lacks {key}")
    return {
        "python_path": requested,
        "python_realpath": realpath,
        "python_version": runtime["python_version"],
        "torch_version": runtime["torch_version"],
        "cuda_build": runtime["torch_cuda_build"],
        "offline_env": dict(EXPECTED_OFFLINE_ENV),
    }


def _assert_runtime_evidence(release: Mapping[str, Any]) -> dict[str, Any]:
    qualification = release.get("qualification")
    if not isinstance(qualification, Mapping):
        raise RuntimeError("production release qualification evidence is missing")
    evidence_path = Path(str(qualification.get("runtime_evidence_path", "")))
    regular_file(evidence_path, "release runtime qualification evidence", immutable=True)
    observed = sha256_file(evidence_path)
    _equal(qualification, "runtime_evidence_sha256", observed, "qualification")
    evidence = read_json(evidence_path)
    runtime = _runtime_from_evidence(
        evidence, expected_head=str(release.get("stage45b_execution_head", ""))
    )
    if evidence.get("token") not in {
        "VANDA_HOPPER_STAGE45B_PREMODEL_CPU_PBS_QUALIFIED",
        "VANDA_STAGE45B_SCIENTIFIC_PREMODEL_RUNTIME_PASS",
    }:
        raise RuntimeError("release runtime evidence token is not an approved login-plane qualification")
    release_runtime = release.get("runtime")
    if not isinstance(release_runtime, Mapping) or dict(release_runtime) != runtime:
        raise RuntimeError("release runtime identity differs from sealed qualification evidence")
    exposure = evidence.get("scientific_exposure", {})
    if not isinstance(exposure, Mapping):
        raise RuntimeError("release runtime exposure record is missing")
    for key in ("model_construction", "forward", "backward", "optimizer", "trainer", "checkpoint_creation", "wbc", "simulator", "new_data"):
        if exposure.get(key) is not False:
            raise RuntimeError(f"release qualification crossed forbidden boundary: {key}")
    return {"path": str(evidence_path.resolve()), "sha256": observed, "token": evidence["token"]}


def _snapshot_from_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = bundle.get("source_snapshot")
    if not isinstance(snapshot, Mapping):
        raise RuntimeError("external admission has no source snapshot")
    for key, expected in (
        ("schema", GROOT_SNAPSHOT_SCHEMA),
        ("origin_repository", "NVIDIA/Isaac-GR00T"),
        ("origin_commit", GROOT_COMMIT),
        ("origin_tree", GROOT_TREE),
        ("source_route", "HISTORICAL_LOCAL"),
        ("archive_sha256", GROOT_ARCHIVE_SHA256),
        ("manifest_file_sha256", GROOT_MANIFEST_FILE_SHA256),
        ("manifest_payload_sha256", GROOT_MANIFEST_PAYLOAD_SHA256),
        ("payload_tree_sha256", GROOT_PAYLOAD_TREE_SHA256),
    ):
        _equal(snapshot, key, expected, "source_snapshot")
    archive = Path(str(snapshot["archive_path"]))
    manifest = Path(str(snapshot["manifest_path"]))
    verification = verify_snapshot(archive, manifest)
    for key in ("archive_sha256", "manifest_file_sha256", "manifest_payload_sha256", "payload_tree_sha256"):
        _equal(verification, key, snapshot[key], "source_snapshot_verification")
    return {
        "schema": GROOT_SNAPSHOT_SCHEMA,
        "origin_repository": "NVIDIA/Isaac-GR00T",
        "origin_commit": GROOT_COMMIT,
        "origin_tree": GROOT_TREE,
        "source_route": "HISTORICAL_LOCAL",
        "archive_path": str(archive.resolve()),
        "manifest_path": str(manifest.resolve()),
        "archive_sha256": verification["archive_sha256"],
        "manifest_file_sha256": verification["manifest_file_sha256"],
        "manifest_payload_sha256": verification["manifest_payload_sha256"],
        "payload_tree_sha256": verification["payload_tree_sha256"],
    }


def build_release(
    *,
    repo: Path,
    output: Path,
    expected_head: str,
    bundle_path: Path,
    runtime_evidence: Path,
) -> dict[str, Any]:
    if len(expected_head) != 40 or any(character not in "0123456789abcdef" for character in expected_head):
        raise RuntimeError("release expected HEAD is not a lowercase commit SHA")
    regular_file(bundle_path, "external admission bundle", immutable=True)
    bundle_sha = sha256_file(bundle_path)
    bundle = read_json(bundle_path)
    assert_bundle(bundle, repo=repo, expected_head=expected_head, bundle_path=bundle_path)
    regular_file(runtime_evidence, "runtime qualification evidence", immutable=True)
    evidence = read_json(runtime_evidence)
    runtime = _runtime_from_evidence(evidence, expected_head=expected_head)
    snapshot = _snapshot_from_bundle(bundle)
    manifest_path = repo / "manifests/STAGE45B_FULL24_FRESH_BASE_EXECUTION_MANIFEST_V1.json"
    manifest = read_json(manifest_path)
    if manifest.get("checkpoint_policy") != _expected_checkpoint_policy():
        raise RuntimeError("frozen checkpoint policy changed")
    entrypoints = _entrypoint_identity(repo)
    release = {
        "schema": SCHEMA,
        "status": STATUS,
        "token": TOKEN,
        "pi_gate": PI_GATE,
        "stage": "Stage45B",
        "version": "V1",
        "scientific_protocol": "B_FRESH_BASE_FULL24",
        "stage45b_execution_head": expected_head,
        "branch": BRANCH,
        "created_utc": now_utc(),
        "entrypoint": {
            "pbs_path": entrypoints["pbs"]["path"],
            "pbs_sha256": entrypoints["pbs"]["sha256"],
            "runner_path": entrypoints["runner"]["path"],
            "runner_sha256": entrypoints["runner"]["sha256"],
            "premodel_path": entrypoints["premodel"]["path"],
            "premodel_sha256": entrypoints["premodel"]["sha256"],
            "source_verifier_path": entrypoints["source_verifier"]["path"],
            "source_verifier_sha256": entrypoints["source_verifier"]["sha256"],
            "external_admission_path": entrypoints["external_admission"]["path"],
            "external_admission_sha256": entrypoints["external_admission"]["sha256"],
            "release_verifier_path": entrypoints["release_verifier"]["path"],
            "release_verifier_sha256": entrypoints["release_verifier"]["sha256"],
            "runtime_qualification_path": entrypoints["runtime_qualification"]["path"],
            "runtime_qualification_sha256": entrypoints["runtime_qualification"]["sha256"],
        },
        "runtime": runtime,
        "inputs": {
            "groot_origin_commit": GROOT_COMMIT,
            "groot_origin_tree": GROOT_TREE,
            "groot_archive_sha256": GROOT_ARCHIVE_SHA256,
            "groot_manifest_file_sha256": GROOT_MANIFEST_FILE_SHA256,
            "groot_manifest_payload_sha256": GROOT_MANIFEST_PAYLOAD_SHA256,
            "groot_payload_tree_sha256": GROOT_PAYLOAD_TREE_SHA256,
            "base_tree_sha256": BASE_TREE_SHA256,
            "full24_tree_sha256": FULL24_TREE_SHA256,
            "runtime_v2_stats_schema": RUNTIME_STATS_SCHEMA,
            "runtime_v2_protocol_revision": RUNTIME_PROTOCOL_REVISION,
            "runtime_v2_stats_sha256": RUNTIME_STATS_SHA256,
            "runtime_v2_stats_payload_sha256": RUNTIME_PAYLOAD_SHA256,
            "historical_b0_stats_sha256": B0_STATS_SHA256,
            "historical_b0_stats_payload_sha256": B0_PAYLOAD_SHA256,
            "cosmos_revision": COSMOS_REVISION,
            "cosmos_tree_sha256": COSMOS_TREE_SHA256,
            "modality_sha256": MODALITY_SHA256,
            "training_config_sha256": CONFIG_SHA256,
        },
        "source_snapshot": snapshot,
        "external_admission": {
            "schema": ADMISSION_SCHEMA,
            "path": str(bundle_path.resolve()),
            "sha256": bundle_sha,
        },
        "contract": dict(bundle["contract"]),
        "recipe": {
            **dict(bundle["recipe"]),
            "structured_action": "K16/T16/H40",
            "universal_to_physical": "D132->D10",
        },
        "checkpoint_policy": _expected_checkpoint_policy(),
        "scientific_lock": {
            "tracked_manifest_sha256": bundle["scientific_lock"]["tracked_manifest"]["sha256"],
            "projection_sha256": bundle["scientific_lock"]["projection_sha256"],
            "scientific_contract_changed": False,
        },
        "qualification": {
            "runtime_evidence_path": str(runtime_evidence.resolve()),
            "runtime_evidence_sha256": sha256_file(runtime_evidence),
            "login_plane_no_scientific_execution": True,
        },
        "execution_boundary": {
            "model_constructed": False,
            "forward": False,
            "backward": False,
            "optimizer": False,
            "trainer": False,
            "checkpoint_creation": False,
            "pbs_submissions": 0,
            "validation_e_read": False,
            "holdout_f_read": False,
            "wbc": False,
            "simulator": False,
            "hardware": False,
        },
        "required_tokens": sorted(set(bundle["required_tokens"]) | {TOKEN}),
        "h200_budget": {"authorized_real_b2_jobs": 1, "reserve": 0, "qsub_submissions_before_release": 0},
        "no_network_fetch_inside_h200": True,
        "next_gate": NEXT_GATE,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"production release output already exists: {output}")
    output.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(output, 0o444)
    return release


def _assert_entrypoints(repo: Path, entrypoint: Mapping[str, Any]) -> None:
    mapping = {
        "pbs": ("pbs_path", "pbs_sha256"),
        "runner": ("runner_path", "runner_sha256"),
        "premodel": ("premodel_path", "premodel_sha256"),
        "source_verifier": ("source_verifier_path", "source_verifier_sha256"),
        "external_admission": ("external_admission_path", "external_admission_sha256"),
        "release_verifier": ("release_verifier_path", "release_verifier_sha256"),
        "runtime_qualification": ("runtime_qualification_path", "runtime_qualification_sha256"),
    }
    for key, (path_key, sha_key) in mapping.items():
        relative = _relative(str(entrypoint.get(path_key, "")))
        path = repo / relative
        regular_file(path, f"release entrypoint {key}")
        _equal(entrypoint, sha_key, sha256_file(path), "entrypoint")


def assert_release(
    release: Mapping[str, Any],
    *,
    repo: Path,
    expected_head: str,
    release_path: Path,
    expected_release_sha256: str,
    bundle_path: Path,
) -> dict[str, Any]:
    regular_file(release_path, "production release", immutable=True)
    _equal(release, "schema", SCHEMA, "release")
    _equal(release, "status", STATUS, "release")
    _equal(release, "token", TOKEN, "release")
    _equal(release, "pi_gate", PI_GATE, "release")
    _equal(release, "scientific_protocol", "B_FRESH_BASE_FULL24", "release")
    _equal(release, "stage45b_execution_head", expected_head, "release")
    _equal(release, "branch", BRANCH, "release")
    if sha256_file(release_path) != expected_release_sha256:
        raise RuntimeError("production release file SHA differs from qsub admission")
    if git(repo, "branch", "--show-current") != BRANCH or git(repo, "rev-parse", "HEAD") != expected_head:
        raise RuntimeError("production release Stage45B checkout identity mismatch")
    if git(repo, "status", "--porcelain=v1"):
        raise RuntimeError("production release Stage45B checkout is dirty")

    entrypoint = release.get("entrypoint")
    if not isinstance(entrypoint, Mapping):
        raise RuntimeError("production release entrypoint identity is missing")
    _assert_entrypoints(repo, entrypoint)

    external = release.get("external_admission")
    if not isinstance(external, Mapping):
        raise RuntimeError("production release external admission identity is missing")
    external_path = Path(str(external.get("path", "")))
    if external_path.resolve() != bundle_path.resolve():
        raise RuntimeError("production release points to a different external admission bundle")
    regular_file(external_path, "production release external admission", immutable=True)
    _equal(external, "sha256", sha256_file(external_path), "external_admission")
    bundle = read_json(external_path)
    assert_bundle(bundle, repo=repo, expected_head=expected_head, bundle_path=external_path)

    inputs = release.get("inputs")
    expected_inputs = {
        "groot_origin_commit": GROOT_COMMIT,
        "groot_origin_tree": GROOT_TREE,
        "groot_archive_sha256": GROOT_ARCHIVE_SHA256,
        "groot_manifest_file_sha256": GROOT_MANIFEST_FILE_SHA256,
        "groot_manifest_payload_sha256": GROOT_MANIFEST_PAYLOAD_SHA256,
        "groot_payload_tree_sha256": GROOT_PAYLOAD_TREE_SHA256,
        "base_tree_sha256": BASE_TREE_SHA256,
        "full24_tree_sha256": FULL24_TREE_SHA256,
        "runtime_v2_stats_schema": RUNTIME_STATS_SCHEMA,
        "runtime_v2_protocol_revision": RUNTIME_PROTOCOL_REVISION,
        "runtime_v2_stats_sha256": RUNTIME_STATS_SHA256,
        "runtime_v2_stats_payload_sha256": RUNTIME_PAYLOAD_SHA256,
        "historical_b0_stats_sha256": B0_STATS_SHA256,
        "historical_b0_stats_payload_sha256": B0_PAYLOAD_SHA256,
        "cosmos_revision": COSMOS_REVISION,
        "cosmos_tree_sha256": COSMOS_TREE_SHA256,
        "modality_sha256": MODALITY_SHA256,
        "training_config_sha256": CONFIG_SHA256,
    }
    if not isinstance(inputs, Mapping):
        raise RuntimeError("production release input identities are missing")
    for key, expected in expected_inputs.items():
        _equal(inputs, key, expected, "inputs")
    snapshot = release.get("source_snapshot")
    if not isinstance(snapshot, Mapping):
        raise RuntimeError("production release source snapshot is missing")
    _snapshot_from_bundle({"source_snapshot": snapshot})

    runtime = release.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("production release runtime identity is missing")
    _equal(runtime, "python_path", EXPECTED_PYTHON, "runtime")
    _equal(runtime, "python_realpath", EXPECTED_PYTHON_REALPATH, "runtime")
    _equal(runtime, "offline_env", EXPECTED_OFFLINE_ENV, "runtime")
    _assert_runtime_evidence(release)

    if release.get("contract") != bundle.get("contract"):
        raise RuntimeError("production release contract differs from external admission")
    expected_recipe = dict(bundle["recipe"])
    expected_recipe.update({"structured_action": "K16/T16/H40", "universal_to_physical": "D132->D10"})
    if release.get("recipe") != expected_recipe:
        raise RuntimeError("production release recipe differs from frozen Stage45B recipe")
    if release.get("checkpoint_policy") != _expected_checkpoint_policy():
        raise RuntimeError("production release checkpoint policy changed")
    lock = release.get("scientific_lock")
    if not isinstance(lock, Mapping) or lock.get("scientific_contract_changed") is not False:
        raise RuntimeError("production release scientific lock is not closed")
    if lock.get("tracked_manifest_sha256") != bundle["scientific_lock"]["tracked_manifest"]["sha256"] or lock.get("projection_sha256") != bundle["scientific_lock"]["projection_sha256"]:
        raise RuntimeError("production release scientific lock differs from external admission")
    boundary = release.get("execution_boundary")
    if not isinstance(boundary, Mapping):
        raise RuntimeError("production release execution boundary is missing")
    for key in ("model_constructed", "forward", "backward", "optimizer", "trainer", "checkpoint_creation", "pbs_submissions", "validation_e_read", "holdout_f_read", "wbc", "simulator", "hardware"):
        if boundary.get(key) not in {False, 0}:
            raise RuntimeError(f"production release execution boundary is non-zero: {key}")
    tokens = release.get("required_tokens")
    if not isinstance(tokens, list) or TOKEN not in tokens:
        raise RuntimeError("production release token is not sealed")
    if release.get("h200_budget") != {"authorized_real_b2_jobs": 1, "reserve": 0, "qsub_submissions_before_release": 0}:
        raise RuntimeError("production release H200 budget changed")
    return dict(release)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "verify"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-release-sha256")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runtime-evidence", type=Path)
    parser.add_argument("--evidence-path", type=Path)
    args = parser.parse_args()
    try:
        repo = args.repo.resolve()
        release_path = args.release.resolve()
        if args.mode == "build":
            if args.runtime_evidence is None:
                parser.error("--runtime-evidence is required for build mode")
            release = build_release(
                repo=repo,
                output=release_path,
                expected_head=args.expected_head,
                bundle_path=args.bundle.resolve(),
                runtime_evidence=args.runtime_evidence.resolve(),
            )
            if args.evidence_path:
                args.evidence_path.parent.mkdir(parents=True, exist_ok=True)
                args.evidence_path.write_text(json.dumps({"schema": SCHEMA, "status": "PASS", "token": TOKEN, "release_sha256": sha256_file(release_path), "timestamp_utc": now_utc()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(TOKEN)
            return 0
        release = read_json(release_path)
        if not args.expected_release_sha256:
            parser.error("--expected-release-sha256 is required for verify mode")
        result = assert_release(
            release,
            repo=repo,
            expected_head=args.expected_head,
            release_path=release_path,
            expected_release_sha256=args.expected_release_sha256,
            bundle_path=args.bundle.resolve(),
        )
        if args.evidence_path:
            args.evidence_path.parent.mkdir(parents=True, exist_ok=True)
            args.evidence_path.write_text(json.dumps({"schema": SCHEMA, "status": "PASS", "token": TOKEN, "release_sha256": sha256_file(release_path), "timestamp_utc": now_utc()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(TOKEN)
        return 0
    except Exception as exc:
        failure = {"schema": SCHEMA, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "timestamp_utc": now_utc()}
        if args.evidence_path:
            try:
                args.evidence_path.parent.mkdir(parents=True, exist_ok=True)
                args.evidence_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            except Exception:
                pass
        print(f"STAGE45B_PRODUCTION_RELEASE_BLOCKED: {failure['error']}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
