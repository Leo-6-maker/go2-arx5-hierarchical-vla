#!/usr/bin/env python3
"""Freeze a Stage41 LH10 execution bundle from a clean pushed branch."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import stage40_corelab_evidence as stage40
import stage41_corelab_evidence as evidence


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _artifact_identities(path: Path) -> Dict[str, Any]:
    source = stage40.read_json(path)
    if source.get("schema") != "CORELAB_STAGE40_CT1_PROVENANCE_V1":
        raise evidence.Stage41EvidenceError("CT1 provenance manifest schema mismatch")
    result: Dict[str, Any] = {}
    for section in ("transport", "runnable_identities"):
        for key, value in source.get(section, {}).items():
            if isinstance(value, (str, int)) and ("sha" in key or key == "cosmos_revision"):
                result[key] = value
    for key in ("archive_sha256", "archive_v2_tree_sha256", "live_v2_tree_sha256"):
        result["wbc_" + key] = source.get("wbc_authority", {}).get(key)
    evidence._require(all(value is not None for value in result.values()), "CT1 artifact identity is missing")
    return result


def prepare_bundle(
    repo: Path,
    runner: Path,
    provenance_manifest: Path,
    output: Path,
    wbc_step_budget: int,
    predecessor_seal: Path,
    predecessor_evidence_root: Path,
    baseline_profile_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    repo = Path(repo).resolve(strict=True)
    head = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "branch", "--show-current")
    evidence._require(branch == "stage41-dev/corelab-long-horizon", "Stage41 bundle must be frozen on the authorized branch")
    evidence._require(len(head) == 40 and bool(branch), "detached or invalid repository identity")
    evidence._require(not _git(repo, "status", "--porcelain", "--untracked-files=all"), "working tree is not clean")
    remote_line = _git(repo, "ls-remote", "--heads", "origin", branch)
    evidence._require(remote_line.split()[0] == head, "local HEAD does not equal remote branch HEAD")

    runner_path = Path(runner).resolve(strict=True)
    source_path = Path(evidence.__file__).resolve(strict=True)
    try:
        runner_relative = runner_path.relative_to(repo)
        source_relative = source_path.relative_to(repo)
    except ValueError as exc:
        raise evidence.Stage41EvidenceError("Stage41 executable is outside repository") from exc
    _git(repo, "ls-files", "--error-unmatch", runner_relative.as_posix())
    _git(repo, "ls-files", "--error-unmatch", source_relative.as_posix())

    predecessor_info = stage40.verify_stage_seal(
        Path(predecessor_seal), Path(predecessor_evidence_root), expected_stage="M2", _admit_without_predecessor=True
    )
    predecessor = {"stage_id": "M2", "seal_sha256": predecessor_info["seal_sha256"]}
    evidence._require(wbc_step_budget > 0, "Stage41 WBC budget must be positive")
    bundle = {
        "schema": evidence.BUNDLE_SCHEMA,
        "stage_id": evidence.STAGE_ID,
        "pi_gate": evidence.PI_GATE,
        "issue_comment": evidence.ISSUE_COMMENT,
        "classification": evidence.CLASSIFICATION,
        "execution_class": evidence.EXECUTION_CLASS,
        "dev_seed": evidence.DEV_SEED,
        "repository_branch": branch,
        "repository_head": head,
        "executable_lineage": {
            "runner_relative_path": runner_relative.as_posix(),
            "runner_sha256": evidence.sha256_file(runner_path),
            "source_relative_path": source_relative.as_posix(),
            "source_sha256": evidence.sha256_file(source_path),
        },
        "artifact_identities": _artifact_identities(Path(provenance_manifest)),
        "baseline_profile_sha256": baseline_profile_sha256,
        "protocol": {
            "handoff_count": evidence.HANDOFF_COUNT,
            "gr00t_forward_count": evidence.GR00T_FORWARD_COUNT,
            "reference_update_count": evidence.REFERENCE_UPDATE_COUNT,
            "native_handoff_schedule": "ROUND_HANDOFF_S_OVER_OBSERVED_NATIVE_DT",
            "executor_prefix_schedule": "ROUND_HANDOFF_S_OVER_VERIFIED_ACTION_DT",
            "completion": "EXISTING_STAGE40_REFERENCE_ENDPOINT_CRITERION",
        },
        "budgets": {
            "gr00t_forward_count": evidence.GR00T_FORWARD_COUNT,
            "wbc_step_count": int(wbc_step_budget),
            "reference_update_count": evidence.REFERENCE_UPDATE_COUNT,
        },
        "forbidden_operations": evidence.expected_forbidden_operations(),
        "predecessor": predecessor,
    }
    evidence.validate_bundle(bundle)
    digest = evidence.write_canonical_json(Path(output), bundle)
    return {"stage_id": evidence.STAGE_ID, "bundle_sha256": digest, "path": str(output), "repository_head": head}


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--provenance-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wbc-step-budget", type=int, required=True)
    parser.add_argument("--predecessor-seal", type=Path, required=True)
    parser.add_argument("--predecessor-evidence-root", type=Path, required=True)
    parser.add_argument("--baseline-profile-sha256")
    args = parser.parse_args(argv)
    value = prepare_bundle(
        args.repo,
        args.runner,
        args.provenance_manifest,
        args.output,
        args.wbc_step_budget,
        args.predecessor_seal,
        args.predecessor_evidence_root,
        args.baseline_profile_sha256,
    )
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
