#!/usr/bin/env python3
"""Run the CPU-only, fail-closed portion of Stage36.

Stage36 starts by binding a new delivered-WBC authority identity.  This
program only performs read-only filesystem, hash, Git, and corpus checks.  It
has no PBS submission, simulator, WBC rollout, GR00T import, model loading,
or checkpoint-writing path.  Missing or ambiguous delivered inputs produce a
sealed return and stop before native-support admission or WBC execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[1]
PI_GATE_ID = "PI_GATE_STATE_0019"
STAGE = "Stage36"
PR_NUMBER = 22
BASE_HEAD = "3a3db87bbbad64d73d9013493ad5779a519e021a"
STAGE35_BRANCH = "stage35/motion-api-authority-closure"
STAGE36_BRANCH = "stage36/wbc-authority-oracle-qualification"
STAGE35_CORPUS_TREE_SHA256 = (
    "b98960f9dc4bfd04b5a0de0f693ddea4f50c8849ef80da83428280c1e2e42414"
)
STAGE35_MANIFEST_SHA256 = (
    "c1e4021a77817af851ec8c1ba5d0c4b5f116558c16f03ff0e24457e1fd6e0642"
)
STAGE35_SEAL_SHA256 = (
    "5ec9dc293546ea33c39c225d184921e77f1ae6f86756ec1bd0304cb19aa63e89"
)
STAGE35_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage35_motion_api_authority_closure_20260905/"
    "stage35_cpu_20260905_v2"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage36_wbc_authority_oracle_qualification_20260905/"
    "stage36_cpu_20260905_v1"
)
NEXT_PI_GATE = "GPT_PI_REVIEW_BEFORE_STRUCTURED_GR00T_TRAINING"
REDELIVERY_STOP = "GPT_PI_REVIEW_WBC_ARTIFACT_REDELIVERY_REQUIRED"
CORE_STOP = "GPT_PI_REVIEW_ORACLE_CORE_CORPUS_ADMISSION"

EXPECTED_OLD_HASHES = {
    "delivered_wbc_archive": "b52a53bef87340e6244ead352fa434390faee65246cd32ab8e6648336a018167",
    "native_urdf": "96c1a81e3dd42af5f1bf87ac8958a0aebd7e798ce06408462e0e49a8a2067e25",
    "arm_checkpoint": "99a3a9bda30f7372c7c31240cc3584757ba46cb3b6b9881f00b371906d077a0a",
    "dog_checkpoint": "51776ddfd7177213444c0bf0c8f017b66ffd28a8674458aa0a16c3b9f77cc670",
    "native_config": "56132d4ff9b866ce196bbc1b1353900b967c54f0f48d2d4e4b82c3555769a0bb",
}

LAB_ROOT = Path("/var/cache/users/liuyu-nfs")
WBC_SOURCE_ROOT = LAB_ROOT / "go2arx5-wbc-stage13/extracted/WBCTrackingShare"
WBC_STAGE16_ROOT = LAB_ROOT / "go2arx5-wbc-stage16"
WBC_EXPERIMENT_ROOT = LAB_ROOT / "go2arx5-experiment-e2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_capture(argv: list[str], *, cwd: Path = REPO) -> tuple[int, str, str]:
    try:
        result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False)
    except OSError as exc:
        return 127, "", str(exc)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def git_output(*args: str) -> str | None:
    code, stdout, _ = run_capture(["git", *args])
    return stdout if code == 0 else None


def finite_json(value: Any, path: str = "root") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite value at {path}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_json(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_json(item, f"{path}.{key}")


def deterministic_tree_hash(path: Path) -> tuple[str | None, list[dict[str, Any]]]:
    """Hash a directory without following directory symlinks."""

    if not path.is_dir():
        return None, []
    records: list[dict[str, Any]] = []
    symlink_seen = False
    for current, dirs, files in os.walk(path, topdown=True, followlinks=False):
        current_path = Path(current)
        kept_dirs: list[str] = []
        for dirname in sorted(dirs):
            candidate = current_path / dirname
            if candidate.is_symlink():
                symlink_seen = True
                records.append(
                    {
                        "relative_path": str(candidate.relative_to(path)),
                        "kind": "symlink_directory",
                        "target": os.readlink(candidate),
                    }
                )
            else:
                kept_dirs.append(dirname)
        dirs[:] = kept_dirs
        for filename in sorted(files):
            candidate = current_path / filename
            relative = str(candidate.relative_to(path))
            if candidate.is_symlink():
                symlink_seen = True
                records.append(
                    {
                        "relative_path": relative,
                        "kind": "symlink_file",
                        "target": os.readlink(candidate),
                    }
                )
            elif candidate.is_file():
                records.append(
                    {
                        "relative_path": relative,
                        "kind": "file",
                        "size_bytes": candidate.stat().st_size,
                        "sha256": sha256_file(candidate),
                    }
                )
    records.sort(key=lambda row: (row["relative_path"], row["kind"]))
    if symlink_seen:
        return None, records
    return sha256_bytes(canonical_json(records)), records


def path_record(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    text_path = str(path)
    if not os.path.lexists(text_path):
        return {
            "path": text_path,
            "exists": False,
            "expected_sha256": expected_sha256,
            "hash_status": "MISSING",
            "is_symlink": False,
            "is_file": False,
            "is_dir": False,
            "sha256": None,
            "tree_sha256": None,
        }

    is_symlink = path.is_symlink()
    resolved = Path(os.path.realpath(text_path))
    record: dict[str, Any] = {
        "path": text_path,
        "exists": True,
        "expected_sha256": expected_sha256,
        "is_symlink": is_symlink,
        "symlink_target": os.readlink(text_path) if is_symlink else None,
        "resolved_path": str(resolved),
        "is_file": resolved.is_file(),
        "is_dir": resolved.is_dir(),
        "size_bytes": None,
        "sha256": None,
        "tree_sha256": None,
        "tree_records": [],
    }
    if resolved.is_file():
        record["size_bytes"] = resolved.stat().st_size
        record["sha256"] = sha256_file(resolved)
        actual = record["sha256"]
    elif resolved.is_dir():
        tree_sha, tree_records = deterministic_tree_hash(resolved)
        record["tree_sha256"] = tree_sha
        record["tree_records"] = tree_records
        actual = tree_sha
    else:
        actual = None
    if is_symlink:
        record["hash_status"] = "SYMLINK_REQUIRES_EXPLICIT_BINDING"
    elif actual is None:
        record["hash_status"] = "NOT_REGULAR_FILE_OR_TREE"
    elif expected_sha256 is None:
        record["hash_status"] = "HASHED_NEW_CANDIDATE"
    elif actual == expected_sha256:
        record["hash_status"] = "MATCH_OLD"
    else:
        record["hash_status"] = "DIFFERENT_NEW_BASELINE"
    return record


def discover_paths(names: Iterable[str]) -> list[Path]:
    """Discover only names under the previously referenced lab/project roots."""

    wanted = set(names)
    roots = (
        LAB_ROOT,
        Path("/scratch/e1192548/stage23_o1_20260903/phase_a_seal_20260903/source"),
        Path("/scratch/e1192548/stage22_repo_deb920a07014"),
    )
    found: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
            dirs[:] = sorted(
                dirname for dirname in dirs if dirname not in {".git", "__pycache__"}
            )
            current_path = Path(current)
            for filename in files:
                if filename in wanted:
                    candidate = current_path / filename
                    if not candidate.is_symlink():
                        found.add(candidate)
    return sorted(found, key=str)


def component(
    name: str,
    candidates: Iterable[Path],
    *,
    expected_sha256: str | None = None,
    discover_names: Iterable[str] = (),
) -> dict[str, Any]:
    all_candidates = {Path(path) for path in candidates}
    all_candidates.update(discover_paths(discover_names))
    records = [path_record(path, expected_sha256) for path in sorted(all_candidates, key=str)]
    existing = [
        row
        for row in records
        if row["exists"]
        and (row["is_file"] or row["is_dir"])
        and (row.get("sha256") or row.get("tree_sha256"))
    ]
    identities = {row.get("sha256") or row.get("tree_sha256") for row in existing}
    non_symlink = [row for row in existing if not row["is_symlink"]]
    if not existing:
        status = "MISSING"
    elif len(identities) > 1:
        status = "AMBIGUOUS_NON_IDENTICAL_DUPLICATES"
    elif len(non_symlink) == 0:
        status = "SYMLINK_REQUIRES_EXPLICIT_BINDING"
    elif expected_sha256 is not None and next(iter(identities)) != expected_sha256:
        status = "DIFFERENT_NEW_BASELINE"
    else:
        status = "PASS_CANDIDATE"
    selected = non_symlink[0] if len(identities) == 1 and non_symlink else None
    selected_hash = None
    if selected is not None:
        selected_hash = selected.get("sha256") or selected.get("tree_sha256")
    historical = "UNKNOWN"
    if expected_sha256 is not None:
        if not existing:
            historical = "UNKNOWN"
        elif expected_sha256 in identities:
            historical = "MATCH_OLD"
        else:
            historical = "DIFFERENT_NEW_BASELINE"
    return {
        "logical_name": name,
        "expected_old_sha256": expected_sha256,
        "status": status,
        "historical_comparison": historical,
        "selected_path": selected["path"] if selected is not None else None,
        "selected_sha256_or_tree_sha256": selected_hash,
        "existing_count": len(existing),
        "distinct_identity_count": len(identities),
        "duplicate_identity_byte_equal": len(identities) == 1 and len(existing) > 1,
        "candidate_records": records,
    }


def authority_components() -> list[dict[str, Any]]:
    source_tree = component(
        "delivered_wbc_source_or_archive",
        [
            WBC_SOURCE_ROOT,
            WBC_STAGE16_ROOT,
            WBC_EXPERIMENT_ROOT / "intake/incoming/WBCTrackingShare_20260831.zip",
        ],
        discover_names=("WBCTrackingShare_20260831.zip",),
    )
    arm = component(
        "arm_policy_checkpoint",
        [
            WBC_SOURCE_ROOT / "checkpoints/arm/ac_weights_039999.pt",
            WBC_SOURCE_ROOT / "resources/checkpoints/arm/ac_weights_039999.pt",
            WBC_STAGE16_ROOT / "checkpoints/arm/ac_weights_039999.pt",
        ],
        expected_sha256=EXPECTED_OLD_HASHES["arm_checkpoint"],
        discover_names=("ac_weights_039999.pt",),
    )
    dog = component(
        "dog_policy_checkpoint",
        [
            WBC_SOURCE_ROOT / "checkpoints/dog/ac_weights_039999.pt",
            WBC_SOURCE_ROOT / "resources/checkpoints/dog/ac_weights_039999.pt",
            WBC_STAGE16_ROOT / "checkpoints/dog/ac_weights_039999.pt",
        ],
        expected_sha256=EXPECTED_OLD_HASHES["dog_checkpoint"],
        discover_names=("ac_weights_039999.pt",),
    )
    config = component(
        "native_wbc_config",
        [
            WBC_SOURCE_ROOT / "config/wbc.py",
            WBC_SOURCE_ROOT / "go1_gym/envs/roboduet/wbc.py",
            WBC_STAGE16_ROOT / "config/wbc.py",
        ],
        expected_sha256=EXPECTED_OLD_HASHES["native_config"],
        discover_names=("wbc.py",),
    )
    urdf = component(
        "native_go2_arx5_urdf",
        [WBC_SOURCE_ROOT / "resources/robots/go2_x5_v3/urdf/go2_x5.urdf"],
        expected_sha256=EXPECTED_OLD_HASHES["native_urdf"],
        discover_names=("go2_x5.urdf",),
    )
    reach = component(
        "native_reach_table",
        [
            WBC_SOURCE_ROOT / "resources/reach_tables/go2_x5_2d.pt",
            WBC_STAGE16_ROOT / "resources/reach_tables/go2_x5_2d.pt",
        ],
        discover_names=("go2_x5_2d.pt",),
    )
    trajectory = component(
        "trajectory_canonicalization_module",
        [
            WBC_SOURCE_ROOT / "modules/trajectory.py",
            WBC_STAGE16_ROOT / "modules/trajectory.py",
        ],
        # Do not accept a project contract file or any unrelated trajectory.py
        # as the private delivered-WBC runtime module.  Only the explicitly
        # bound delivery-tree paths can satisfy this identity.
        discover_names=(),
    )
    preview = component(
        "native_preview_and_controller_launcher",
        [
            WBC_SOURCE_ROOT / "runs/2026-08-05/stage2_v3_trajtracking_rl_e2e_gait_wo_postproc_192018/scripts/wbc_env.py",
            WBC_STAGE16_ROOT / "runs/2026-08-05/stage2_v3_trajtracking_rl_e2e_gait_wo_postproc_192018/scripts/wbc_env.py",
        ],
        discover_names=("wbc_env.py",),
    )
    stepping = component(
        "native_controller_step_module",
        [
            WBC_SOURCE_ROOT / "runs/2026-08-05/stage2_v3_trajtracking_rl_e2e_gait_wo_postproc_192018/scripts/legged_robot.py",
            WBC_STAGE16_ROOT / "runs/2026-08-05/stage2_v3_trajtracking_rl_e2e_gait_wo_postproc_192018/scripts/legged_robot.py",
        ],
        discover_names=("legged_robot.py",),
    )
    runtime = component(
        "simulator_runtime_identity",
        [
            WBC_SOURCE_ROOT / "environment.yml",
            WBC_SOURCE_ROOT / "requirements.txt",
            WBC_SOURCE_ROOT / "runtime_manifest.json",
            WBC_STAGE16_ROOT / "environment.yml",
            WBC_STAGE16_ROOT / "requirements.txt",
            WBC_STAGE16_ROOT / "runtime_manifest.json",
        ],
        discover_names=("environment.yml", "requirements.txt", "runtime_manifest.json"),
    )
    return [source_tree, arm, dog, config, urdf, reach, trajectory, preview, stepping, runtime]


def authority_pass(components: list[dict[str, Any]]) -> bool:
    return all(
        row["status"] == "PASS_CANDIDATE"
        and row["selected_path"] is not None
        and row["distinct_identity_count"] == 1
        for row in components
    )


def load_json(path: Path) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    finite_json(payload, str(path))
    return payload


def validate_case(case: dict[str, Any], expected_family: str) -> dict[str, Any]:
    errors: list[str] = []
    if case.get("case_id") is None or case.get("family") != expected_family:
        errors.append("case_id_or_family")
    if case.get("schema") != "STAGE35_ORACLE_TRAJECTORY_V1":
        errors.append("schema")
    if case.get("source_frame") != "NATIVE_WBC_RELATIVE_REFERENCE":
        errors.append("source_frame")
    if case.get("tcp_semantic_id") != "x5_link6_gripper_center":
        errors.append("tcp_semantic_id")
    timestamps = case.get("timestamps")
    positions = case.get("eef_position_m")
    orientations = case.get("eef_orientation_xyzw")
    sample_count = case.get("sample_count")
    if not isinstance(timestamps, list) or len(timestamps) != sample_count:
        errors.append("timestamps_shape")
    elif any(float(b) <= float(a) for a, b in zip(timestamps, timestamps[1:])):
        errors.append("timestamps_not_strictly_monotonic")
    if not isinstance(positions, list) or len(positions) != sample_count:
        errors.append("position_shape")
    elif any(not isinstance(row, list) or len(row) != 3 for row in positions):
        errors.append("position_dimension")
    if not isinstance(orientations, list) or len(orientations) != sample_count:
        errors.append("orientation_shape")
    elif any(not isinstance(row, list) or len(row) != 4 for row in orientations):
        errors.append("orientation_dimension")
    if case.get("validity_start") != 0.0:
        errors.append("validity_start")
    if case.get("validity_end") != case.get("duration_s"):
        errors.append("validity_end")
    provenance = case.get("provenance", {})
    if provenance.get("authoritative_stage16_source") is not False:
        errors.append("historical_source_flag")
    if provenance.get("native_wbc_inputs_bound") is not False:
        errors.append("native_binding_flag")
    if provenance.get("outcome_independent") is not True:
        errors.append("outcome_independence")
    return {
        "case_id": case.get("case_id"),
        "family": expected_family,
        "structural_pass": not errors,
        "errors": errors,
    }


def validate_stage35_corpus() -> dict[str, Any]:
    manifest_path = STAGE35_ROOT / "stage35_oracle_corpus_manifest.json"
    seal_path = STAGE35_ROOT / "stage35_artifact_seal.json"
    manifest_exists = manifest_path.is_file()
    manifest_sha = sha256_file(manifest_path) if manifest_exists else None
    manifest: dict[str, Any] = load_json(manifest_path) if manifest_exists else {}
    rows = manifest.get("case_records", [])
    per_case: list[dict[str, Any]] = []
    for row in rows:
        case_path = STAGE35_ROOT / row.get("relative_path", "")
        if not case_path.is_file():
            per_case.append({"case_id": row.get("case_id"), "structural_pass": False, "errors": ["missing_case"]})
            continue
        if sha256_file(case_path) != row.get("sha256") or case_path.stat().st_size != row.get("size_bytes"):
            per_case.append({"case_id": row.get("case_id"), "structural_pass": False, "errors": ["case_hash_or_size"]})
            continue
        per_case.append(validate_case(load_json(case_path), str(row.get("family"))))
    families = {
        family: [row for row in per_case if row.get("family") == family]
        for family in ("N", "T", "P", "C")
    }
    core = families["N"] + families["T"] + families["P"]
    continuous = families["C"]
    structure_core_pass = len(core) == 24 and all(row["structural_pass"] for row in core)
    structure_continuous_pass = len(continuous) == 8 and all(
        row["structural_pass"] for row in continuous
    )
    return {
        "schema": "STAGE36_STAGE35_CORPUS_INPUT_AUDIT_V1",
        "input_root": str(STAGE35_ROOT),
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_exists,
        "manifest_sha256": manifest_sha,
        "manifest_sha256_expected": STAGE35_MANIFEST_SHA256,
        "manifest_hash_match": manifest_sha == STAGE35_MANIFEST_SHA256,
        "stage35_seal_path": str(seal_path),
        "stage35_seal_sha256": sha256_file(seal_path) if seal_path.is_file() else None,
        "stage35_seal_hash_match": seal_path.is_file() and sha256_file(seal_path) == STAGE35_SEAL_SHA256,
        "tree_sha256": manifest.get("corpus_tree_sha256"),
        "tree_sha256_expected": STAGE35_CORPUS_TREE_SHA256,
        "tree_hash_match": manifest.get("corpus_tree_sha256") == STAGE35_CORPUS_TREE_SHA256,
        "case_count": manifest.get("case_count"),
        "class_counts": manifest.get("class_counts"),
        "core24_structural_pass": structure_core_pass,
        "continuous8_structural_pass": structure_continuous_pass,
        "core24_authority_pass": None,
        "continuous8_authority_pass": None,
        "core24_admission_status": "NOT_EVALUATED_MISSING_NATIVE_AUTHORITY",
        "continuous8_admission_status": "NOT_EVALUATED_MISSING_NATIVE_AUTHORITY",
        "case_results": per_case,
        "regeneration_performed": False,
        "cases_edited": False,
        "analytic_stage35_support_promoted": False,
    }


def qstat_snapshot() -> dict[str, Any]:
    code, stdout, stderr = run_capture(["qstat", "-u", os.environ.get("USER", "e1192548")])
    return {
        "command_available": code != 127,
        "returncode": code,
        "stdout": stdout,
        "stderr": stderr,
        "stage36_jobs_visible": any(
            token in (stdout or "").lower() for token in ("stage36", "wbc-authority", "oracle")
        ),
    }


def make_preflight(corpus: dict[str, Any]) -> dict[str, Any]:
    status = git_output("status", "--porcelain")
    branch = git_output("branch", "--show-current")
    head = git_output("rev-parse", "HEAD")
    remote_head = git_output("rev-parse", f"origin/{STAGE36_BRANCH}")
    merge_base = git_output("merge-base", "HEAD", BASE_HEAD)
    qstat = qstat_snapshot()
    return {
        "schema": "STAGE36_ZERO_MODEL_PREFLIGHT_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pi_gate": PI_GATE_ID,
        "expected_base_head": BASE_HEAD,
        "expected_branch": STAGE36_BRANCH,
        "branch": branch,
        "head": head,
        "origin_stage36_head": remote_head,
        "base_is_ancestor": merge_base == BASE_HEAD,
        "worktree_clean": status == "",
        "status_porcelain": status,
        "qstat": qstat,
        "project": "CFP03-CF-004",
        "selection_queue": "auto_only_if_E_predicates_pass",
        "corpus_input_exact": corpus["manifest_hash_match"] and corpus["tree_hash_match"],
        "execution_boundary": {
            "pbs_submitted": False,
            "gpu_used": False,
            "model_constructed": False,
            "gr00t_model_forward": False,
            "wbc_executed": False,
            "training": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "hardware_executed": False,
            "pr_merge": False,
        },
    }


def sealed_entries(output_root: Path, authority: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paths: set[Path] = {
        REPO / "scripts/stage36_wbc_authority_oracle_qualification.py",
        REPO / "reports/STAGE36_WBC_AUTHORITY_ORACLE_QUALIFICATION.md",
        STAGE35_ROOT / "stage35_artifact_seal.json",
        STAGE35_ROOT / "stage35_oracle_corpus_manifest.json",
        STAGE35_ROOT / "stage35_oracle_corpus_protocol.json",
    }
    for row in authority:
        for candidate in row["candidate_records"]:
            path = Path(candidate["path"])
            if candidate["exists"] and not candidate["is_symlink"] and (candidate["is_file"] or candidate["is_dir"]):
                paths.add(path)
    manifest = load_json(STAGE35_ROOT / "stage35_oracle_corpus_manifest.json")
    for row in manifest["case_records"]:
        paths.add(STAGE35_ROOT / row["relative_path"])
    for path in output_root.rglob("*"):
        if path.is_file() and path.name not in {"stage36_artifact_seal.json", "stage36_return.json"}:
            paths.add(path)
    entries: list[dict[str, Any]] = []
    for path in sorted(paths, key=str):
        if not path.is_file():
            continue
        entries.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-repo", type=Path, default=REPO)
    parser.add_argument("--stage35-root", type=Path, default=STAGE35_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    if args.project_repo.resolve() != REPO:
        raise SystemExit(f"FAIL_CLOSED wrong repository: {args.project_repo}")
    if args.stage35_root.resolve() != STAGE35_ROOT:
        raise SystemExit(f"FAIL_CLOSED wrong Stage35 input root: {args.stage35_root}")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"FAIL_CLOSED output already exists: {output_root}")
    if git_output("branch", "--show-current") != STAGE36_BRANCH:
        raise SystemExit("FAIL_CLOSED wrong Stage36 branch")
    if git_output("status", "--porcelain") != "":
        raise SystemExit("FAIL_CLOSED dirty worktree before Stage36")
    if git_output("merge-base", "HEAD", BASE_HEAD) != BASE_HEAD:
        raise SystemExit("FAIL_CLOSED Stage35 base is not an ancestor of Stage36 HEAD")

    output_root.mkdir(parents=True)
    authority = authority_components()
    authority_ok = authority_pass(authority)
    audit = {
        "schema": "STAGE36_DELIVERED_WBC_AUTHORITY_V2_AUDIT_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pi_gate": PI_GATE_ID,
        "stage": STAGE,
        "authority_id": "DELIVERED_WBC_AUTHORITY_V2",
        "historical_stage16_claim": False,
        "status": "PASS" if authority_ok else "BLOCKED_MISSING_OR_AMBIGUOUS_COMPONENT",
        "hard_stop": not authority_ok,
        "components": authority,
        "required_component_names": [row["logical_name"] for row in authority],
        "missing_components": [row["logical_name"] for row in authority if row["status"] == "MISSING"],
        "ambiguous_components": [
            row["logical_name"]
            for row in authority
            if row["status"] in {"AMBIGUOUS_NON_IDENTICAL_DUPLICATES", "SYMLINK_REQUIRES_EXPLICIT_BINDING"}
        ],
        "shared_codex_modified": False,
        "read_only": True,
    }
    write_json(output_root / "stage36_a_authority_audit.json", audit)

    selected = [
        {
            "logical_name": row["logical_name"],
            "selected_path": row["selected_path"],
            "selected_sha256_or_tree_sha256": row["selected_sha256_or_tree_sha256"],
            "historical_comparison": row["historical_comparison"],
        }
        for row in authority
        if row["selected_path"] is not None
    ]
    authority_manifest = {
        "schema": "DELIVERED_WBC_AUTHORITY_V2_MANIFEST_V1",
        "authority_id": "DELIVERED_WBC_AUTHORITY_V2",
        "status": "PASS" if authority_ok else "BLOCKED_MISSING_OR_AMBIGUOUS_COMPONENT",
        "historical_stage16_equivalence_claim": False,
        "components": selected,
        "component_audit_path": str(output_root / "stage36_a_authority_audit.json"),
        "source_provenance": "HOPPER_READ_ONLY_LOCAL_CANDIDATE_AUDIT",
        "no_mutation_performed": True,
    }
    write_json(output_root / "wbc_authority_v2_manifest.json", authority_manifest)
    tree_manifest = {
        "schema": "DELIVERED_WBC_AUTHORITY_V2_TREE_MANIFEST_V1",
        "authority_id": "DELIVERED_WBC_AUTHORITY_V2",
        "status": "PASS" if authority_ok else "BLOCKED_MISSING_OR_AMBIGUOUS_COMPONENT",
        "selected_tree_identities": [
            row for row in selected if row["selected_sha256_or_tree_sha256"] is not None
        ],
        "deterministic_tree_hash_available": authority_ok,
        "no_mutation_performed": True,
    }
    write_json(output_root / "wbc_authority_v2_tree_manifest.json", tree_manifest)
    runtime_contract = {
        "schema": "DELIVERED_WBC_AUTHORITY_V2_RUNTIME_CONTRACT_V1",
        "authority_id": "DELIVERED_WBC_AUTHORITY_V2",
        "status": "PASS" if authority_ok else "BLOCKED_MISSING_OR_AMBIGUOUS_COMPONENT",
        "runtime_inputs": [
            "arm_policy_checkpoint",
            "dog_policy_checkpoint",
            "native_wbc_config",
            "native_go2_arx5_urdf",
            "native_reach_table",
            "trajectory_canonicalization_module",
            "native_preview_and_controller_launcher",
            "native_controller_step_module",
            "simulator_runtime_identity",
        ],
        "frame_tcp_contract": "x5_link6_gripper_center / native delivered WBC frame",
        "controller_semantics": "native source and runtime only; no V2 special WBC code",
        "model_calls": 0,
        "wbc_rollouts": 0,
        "pbs_submitted": False,
        "no_mutation_performed": True,
    }
    write_json(output_root / "wbc_authority_v2_runtime_contract.json", runtime_contract)

    corpus = validate_stage35_corpus()
    if authority_ok:
        corpus["core24_authority_pass"] = bool(corpus["core24_structural_pass"])
        corpus["continuous8_authority_pass"] = bool(corpus["continuous8_structural_pass"])
        corpus["core24_admission_status"] = "PASS" if corpus["core24_authority_pass"] else "FAIL"
        corpus["continuous8_admission_status"] = "PASS" if corpus["continuous8_authority_pass"] else "FAIL"
    write_json(output_root / "stage36_b_corpus_admission.json", corpus)

    preflight = make_preflight(corpus)
    write_json(output_root / "stage36_preflight.json", preflight)

    if not authority_ok:
        result = "INFRASTRUCTURE_PROVENANCE_BLOCKER"
        stop_reason = REDELIVERY_STOP
    elif not corpus["core24_authority_pass"]:
        result = "CORE_CORPUS_NOT_ADMITTED"
        stop_reason = CORE_STOP
    elif not corpus["continuous8_authority_pass"]:
        result = "CONTINUOUS_BASE_CORPUS_NOT_ADMITTED"
        stop_reason = "CONTINUOUS_BASE_AUTHORITY_UNRESOLVED"
    else:
        result = "AUTHORITY_AND_CORPUS_ADMISSION_PASS_REQUIRES_STAGE36_CD"
        stop_reason = NEXT_PI_GATE
    result_payload = {
        "schema": "STAGE36_RESULT_SUMMARY_V1",
        "pi_gate": PI_GATE_ID,
        "stage": STAGE,
        "authority_id": "DELIVERED_WBC_AUTHORITY_V2",
        "authority_status": audit["status"],
        "core24_authority_pass": corpus["core24_authority_pass"],
        "continuous8_authority_pass": corpus["continuous8_authority_pass"],
        "k_star": None,
        "t_star": None,
        "motion_api_parity_pass": None,
        "wbc_core_qualified": None,
        "wbc_continuous_base_qualified": None,
        "pbs_job_id": None,
        "wbc_rollouts": 0,
        "gr00t_model_forward": False,
        "training": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "hardware_executed": False,
        "result": result,
        "stop_reason": stop_reason,
        "no_stage36_cd_or_e_execution": not authority_ok,
    }
    result_path = output_root / "stage36_result.json"
    write_json(result_path, result_payload)

    entries = sealed_entries(output_root, authority)
    seal_payload = {
        "schema": "STAGE36_ARTIFACT_SEAL_V1",
        "stage": STAGE,
        "pi_gate": PI_GATE_ID,
        "base_head": BASE_HEAD,
        "stage36_head": git_output("rev-parse", "HEAD"),
        "execution_class": "CPU_READ_ONLY_WBC_AUTHORITY_AND_CORPUS_ADMISSION",
        "pbs_job_id": None,
        "pbs_submitted": False,
        "gpu_used": False,
        "model_constructed": False,
        "wbc_executed": False,
        "sealed_files": entries,
        "forbidden_actions": {
            "gr00t_model_constructed": False,
            "gr00t_forward": False,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "checkpoint_write": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "hardware": False,
            "pr_merge": False,
        },
    }
    seal_path = output_root / "stage36_artifact_seal.json"
    write_json(seal_path, seal_payload)
    seal_sha = sha256_file(seal_path)

    return_payload = {
        "schema": "STAGE36_HOPPER_RETURN_V1",
        "ROLE": "CODEX_HOPPER_SERVER_LONG_HORIZON",
        "PI_GATE": PI_GATE_ID,
        "STAGE": STAGE,
        "PR": PR_NUMBER,
        "BASE_HEAD": BASE_HEAD,
        "LIVE_HEAD": git_output("rev-parse", "HEAD"),
        "BRANCH": git_output("branch", "--show-current"),
        "WORKTREE_CLEAN": git_output("status", "--porcelain") == "",
        "WBC_AUTHORITY_V2_STATUS": audit["status"],
        "WBC_AUTHORITY_V2_MANIFEST_PATH": str(output_root / "wbc_authority_v2_manifest.json"),
        "WBC_AUTHORITY_V2_MANIFEST_SHA256": sha256_file(output_root / "wbc_authority_v2_manifest.json"),
        "CORE24_AUTHORITY_PASS": corpus["core24_authority_pass"],
        "CONTINUOUS8_AUTHORITY_PASS": corpus["continuous8_authority_pass"],
        "K_STAR": None,
        "T_STAR": None,
        "PBS_JOB_ID": None,
        "WBC_ROLLOUTS": 0,
        "MOTION_API_PARITY_PASS": None,
        "WBC_CORE_QUALIFIED": None,
        "WBC_CONTINUOUS_BASE_QUALIFIED": None,
        "ARTIFACT_ROOT": str(output_root),
        "RESULT_SHA256": sha256_file(result_path),
        "MANIFEST_SHA256": sha256_file(output_root / "wbc_authority_v2_manifest.json"),
        "SEAL_SHA256": seal_sha,
        "GR00T_MODEL_FORWARD": False,
        "TRAINING": False,
        "VALIDATION_E_READ": False,
        "HOLDOUT_F_READ": False,
        "HARDWARE_EXECUTED": False,
        "RESULT": result,
        "STOP_REASON": stop_reason,
        "NEXT_STAGE_OR_STOP": stop_reason,
        "STAGE35_CORPUS_TREE_SHA256": STAGE35_CORPUS_TREE_SHA256,
        "STAGE35_CORPUS_REGENERATED": False,
        "STAGE35_CORPUS_EDITED": False,
        "SCIENTIFIC_CONTRACT_CHANGED": False,
        "ARTIFACT_SEAL_PATH": str(seal_path),
        "DETAIL_OUTPUTS": {
            "authority_audit": str(output_root / "stage36_a_authority_audit.json"),
            "authority_manifest": str(output_root / "wbc_authority_v2_manifest.json"),
            "authority_tree_manifest": str(output_root / "wbc_authority_v2_tree_manifest.json"),
            "runtime_contract": str(output_root / "wbc_authority_v2_runtime_contract.json"),
            "corpus_admission": str(output_root / "stage36_b_corpus_admission.json"),
            "preflight": str(output_root / "stage36_preflight.json"),
            "result": str(result_path),
        },
    }
    write_json(output_root / "stage36_return.json", return_payload)
    print(json.dumps(return_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
