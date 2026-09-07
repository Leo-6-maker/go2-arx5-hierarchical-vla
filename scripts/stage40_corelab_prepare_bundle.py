#!/usr/bin/env python3
"""Freeze one Corelab Stage40 execution bundle from a clean pushed HEAD."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import stage40_corelab_evidence as evidence


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _artifact_identities(path: Path) -> Dict[str, Any]:
    source = evidence.read_json(path)
    if source.get("schema") != "CORELAB_STAGE40_CT1_PROVENANCE_V1":
        raise evidence.Stage40EvidenceError("CT1 provenance manifest schema mismatch")
    result = {}
    for section in ("transport", "runnable_identities"):
        for key, value in source.get(section, {}).items():
            if isinstance(value, (str, int)) and ("sha" in key or key == "cosmos_revision"):
                result[key] = value
    wbc = source.get("wbc_authority", {})
    for key in ("archive_sha256", "archive_v2_tree_sha256", "live_v2_tree_sha256"):
        result["wbc_" + key] = wbc.get(key)
    evidence._require(all(value is not None for value in result.values()), "CT1 artifact identity is missing")
    return result


def prepare_bundle(
    stage_id: str,
    repo: Path,
    runner: Path,
    provenance_manifest: Path,
    output: Path,
    wbc_step_budget: int,
    predecessor_seal: Optional[Path] = None,
    predecessor_evidence_root: Optional[Path] = None,
) -> Dict[str, Any]:
    repo = Path(repo).resolve(strict=True)
    evidence._validate_stage_id(stage_id)
    head = _git(repo, "rev-parse", "HEAD")
    branch = _git(repo, "branch", "--show-current")
    evidence._require(len(head) == 40 and bool(branch), "detached or invalid repository identity")
    evidence._require(not _git(repo, "status", "--porcelain", "--untracked-files=all"), "working tree is not clean")
    remote_line = _git(repo, "ls-remote", "--heads", "origin", branch)
    evidence._require(remote_line.split()[0] == head, "local HEAD does not equal remote branch HEAD")

    runner_path = Path(runner).resolve(strict=True)
    try:
        runner_relative = runner_path.relative_to(repo)
    except ValueError as exc:
        raise evidence.Stage40EvidenceError("runner is outside repository") from exc
    source_path = Path(evidence.__file__).resolve(strict=True)
    source_relative = source_path.relative_to(repo)
    _git(repo, "ls-files", "--error-unmatch", runner_relative.as_posix())
    _git(repo, "ls-files", "--error-unmatch", source_relative.as_posix())

    expected_predecessor = evidence.PREDECESSOR[stage_id]
    predecessor = None
    if expected_predecessor is None:
        evidence._require(predecessor_seal is None, "root stage does not accept a predecessor")
    else:
        evidence._require(predecessor_seal is not None and predecessor_evidence_root is not None, "predecessor seal and evidence root are required")
        info = evidence.verify_stage_seal(
            Path(predecessor_seal),
            Path(predecessor_evidence_root),
            expected_stage=expected_predecessor,
            _admit_without_predecessor=True,
        )
        predecessor = {"stage_id": expected_predecessor, "seal_sha256": info["seal_sha256"]}

    budget = {
        "gr00t_forward_count": evidence.GR00T_BUDGET[stage_id],
        "wbc_step_count": int(wbc_step_budget),
        "reference_update_count": evidence.REFERENCE_BUDGET[stage_id],
    }
    bundle = {
        "schema": "CORELAB_STAGE40_EXECUTION_BUNDLE_V1",
        "stage_id": stage_id,
        "pi_gate": evidence.PI_GATE,
        "issue_comment": evidence.ISSUE_COMMENT,
        "classification": evidence.CLASSIFICATION,
        "execution_class": evidence.EXECUTION_CLASS[stage_id],
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
        "budgets": budget,
        "forbidden_operations": evidence.expected_forbidden_operations(),
        "predecessor": predecessor,
    }
    evidence.validate_bundle(bundle, expected_stage=stage_id)
    digest = evidence.write_canonical_json(Path(output), bundle)
    return {"stage_id": stage_id, "bundle_sha256": digest, "path": str(output), "repository_head": head}


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=evidence.STAGES, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--provenance-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wbc-step-budget", type=int, default=0)
    parser.add_argument("--predecessor-seal", type=Path)
    parser.add_argument("--predecessor-evidence-root", type=Path)
    args = parser.parse_args(argv)
    value = prepare_bundle(
        args.stage,
        args.repo,
        args.runner,
        args.provenance_manifest,
        args.output,
        args.wbc_step_budget,
        args.predecessor_seal,
        args.predecessor_evidence_root,
    )
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
