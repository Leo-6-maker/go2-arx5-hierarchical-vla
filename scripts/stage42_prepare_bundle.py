#!/usr/bin/env python3
"""Freeze the Stage42 bounded feedback-causality execution bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import stage42_feedback_causality as evidence


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def prepare_bundle(
    repo: Path,
    audit_path: Path,
    output: Path,
    source_run_root: Optional[Path] = None,
    require_remote: bool = True,
) -> Dict[str, Any]:
    repo = Path(repo).resolve(strict=True)
    audit_path = Path(audit_path).resolve(strict=True)
    audit = evidence.read_json(audit_path)
    evidence._require(audit.get("schema") == "CORELAB_STAGE42_F0_F2_AUDIT_V1" and audit.get("status") == "PASS", "F0/F1/F2 audit is not PASS")
    evidence._require(audit["predecessor"]["stage41_head"] == evidence.STAGE41_HEAD, "Stage41 predecessor HEAD drift")
    evidence._require(audit["predecessor"]["stage41_bundle_sha256"] == evidence.STAGE41_BUNDLE_SHA256, "Stage41 predecessor bundle drift")
    evidence._require(audit["predecessor"]["lh10_a_seal_sha256"] == evidence.STAGE41_A_SEAL, "LH10-A predecessor seal drift")
    evidence._require(audit["predecessor"]["lh10_b_seal_sha256"] == evidence.STAGE41_B_SEAL, "LH10-B predecessor seal drift")
    branch = _git(repo, "branch", "--show-current")
    head = _git(repo, "rev-parse", "HEAD")
    evidence._require(branch == evidence.BRANCH, "Stage42 bundle must use the authorized branch")
    evidence._require(len(head) == 40, "invalid Stage42 repository HEAD")
    evidence._require(not _git(repo, "status", "--porcelain", "--untracked-files=all"), "Stage42 working tree is not clean")
    if require_remote:
        remote_line = _git(repo, "ls-remote", "--heads", "origin", branch)
        evidence._require(remote_line.split()[0] == head, "local Stage42 HEAD is not pushed")
    runner = (repo / "scripts" / "stage42_corelab_probes.py").resolve(strict=True)
    source = (repo / "scripts" / "stage42_feedback_causality.py").resolve(strict=True)
    evidence._require(_git(repo, "ls-files", "--error-unmatch", "scripts/stage42_corelab_probes.py"), "probe runner is not tracked")
    evidence._require(_git(repo, "ls-files", "--error-unmatch", "scripts/stage42_feedback_causality.py"), "evidence source is not tracked")
    within = audit["f0"]["runs"]["LH10-A"]["within_run"]
    selected = within.get("archived_request_files", {})
    evidence._require(set(selected) == set(evidence.SELECTED_IDS), "selected archived request identity is incomplete")
    source_root = Path(source_run_root or audit["f0"]["runs"]["LH10-A"]["run_root"]).resolve()
    frozen_selected = {}
    for request_id in evidence.SELECTED_IDS:
        item = dict(selected[request_id])
        frozen_selected[request_id] = {
            "request_relative_path": item["relative_path"],
            "request_file_sha256": item["sha256"],
            "observation_timestamp_s": item["observation_timestamp_s"],
            "components": dict(item["component_hashes"]),
        }
    plan = {
        "ids": list(evidence.CF_IDS),
        "language": evidence.LANGUAGE,
        "definitions": {
            "CF1": "body(S0)+wrist(S0)+StateV1(S10)",
            "CF2": "body(S10)+wrist(S10)+StateV1(S0)",
            "CF3": "body(S10)+wrist(S0)+StateV1(S0)",
            "CF4": "body(S0)+wrist(S10)+StateV1(S0)",
            "CF5": "body(S5)+wrist(S5)+StateV1(S0)",
            "CF6": "body(S0)+wrist(S0)+StateV1(S5)",
        },
        "single_modality_classification_probes": {
            "proprioception": "CF1_vs_REPLAY-S0",
            "body_vision": "CF3_vs_REPLAY-S0",
            "wrist_vision": "CF4_vs_REPLAY-S0",
        },
    }
    bundle = {
        "schema": "CORELAB_STAGE42_EXECUTION_BUNDLE_V1",
        "stage_id": evidence.STAGE_ID,
        "pi_gate": evidence.PI_GATE,
        "issue_comment": evidence.ISSUE_COMMENT,
        "repository_branch": branch,
        "repository_head": head,
        "dev_seed": evidence.DEV_SEED,
        "classification": "NON_SCIENTIFIC_DEVELOPMENT_EVIDENCE",
        "executable_lineage": {
            "runner_relative_path": "scripts/stage42_corelab_probes.py",
            "runner_sha256": evidence.sha256_file(runner),
            "source_relative_path": "scripts/stage42_feedback_causality.py",
            "source_sha256": evidence.sha256_file(source),
        },
        "predecessor": {
            "stage_id": "Stage41-LH10",
            "stage41_head": evidence.STAGE41_HEAD,
            "stage41_bundle_sha256": evidence.STAGE41_BUNDLE_SHA256,
            "lh10_a_seal_sha256": evidence.STAGE41_A_SEAL,
            "lh10_b_seal_sha256": evidence.STAGE41_B_SEAL,
            "m2_seal_sha256": evidence.M2_SEAL,
            "source_run": "LH10-A",
        },
        "f0_f2_audit": {
            "path": str(audit_path),
            "sha256": evidence.sha256_file(audit_path),
            "token": "CORELAB_STAGE42_WITHIN_RUN_CAUSAL_AUDIT_PASS",
            "success_semantics_token": "CORELAB_STAGE42_SUCCESS_SEMANTICS_AUDITED",
            "joint_limit_classification": "JOINT_LIMIT_METRIC_SEMANTICS_RESOLVED",
        },
        "frozen_inputs": {
            "source_run_root": str(source_root),
            "source_run_verified_by_sealed_audit": True,
            "selected_observations": frozen_selected,
            "accepted_raw_action_sha256": {key: within["accepted_raw_action_sha256"][key] for key in evidence.SELECTED_IDS},
            "accepted_decoded_action_sha256": {
                key: within["hashes"]["decoded_action_d10_h40"][int(key.rsplit("q", 1)[1])]
                for key in evidence.SELECTED_IDS
            },
            "observation_semantics": {
                "body_rgb": {"shape": [96, 96, 3], "dtype": "uint8"},
                "wrist_rgb": {"shape": [96, 96, 3], "dtype": "uint8"},
                "StateV1": {"shape": [10], "dtype": "float32", "abi": "EEF_XYZ_3+GR00T_FIRST_TWO_ROWS_ROT6D_6+GRIPPER_1"},
            },
        },
        "counterfactual_plan": plan,
        "budgets": {
            "replay_forwards": 3,
            "counterfactual_forwards": 6,
            "total_gr00t_forwards": 9,
            "wbc_steps": 0,
            "reference_updates": 0,
        },
        "forbidden_operations": evidence.expected_forbidden_operations(),
    }
    evidence._verify_bundle(bundle)
    digest = evidence.write_canonical_json(Path(output), bundle)
    return {"status": "PASS", "stage": evidence.STAGE_ID, "bundle_sha256": digest, "path": str(output), "repository_head": head}


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run-root", type=Path)
    parser.add_argument("--no-remote-check", action="store_true")
    args = parser.parse_args(argv)
    value = prepare_bundle(args.repo, args.audit, args.output, args.source_run_root, not args.no_remote_check)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
