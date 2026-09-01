#!/usr/bin/env python3
"""Create a credential-free, node-local Stage22 H200 handoff package.

The package contains a Git archive of the tracked Stage22 source, the locked
train-only fixture, the validated base model directory, the custom modality
module, and non-submittable PBS templates.  It never copies a runtime
credential, private key, WBC artifact, or Git working-tree untracked file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPECTED_BRANCH = "stage22/action-profile-adaptation"
EXPECTED_TRAIN_TARGETS = [
    "red_target_train_a",
    "red_target_train_b",
    "red_target_train_c",
    "red_target_train_d",
]
EXPECTED_VALIDATION_TARGETS = ["red_target_holdout_e"]
EXPECTED_FINAL_TEST_TARGETS = ["red_target_holdout_f"]
SENSITIVE_NAMES = {
    ".env",
    ".env.local",
    ".envrc",
    ".ssh",
    "authorized_keys",
    "credentials",
    "secrets",
    "tokens",
}
SENSITIVE_PREFIXES = ("id_rsa", "id_ecdsa", "id_ed25519", "access_token", "hf_token")
SENSITIVE_SUFFIXES = (".pem", ".key", ".token")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(path: Path) -> bool:
    for component in path.parts:
        lowered = component.lower()
        if lowered in SENSITIVE_NAMES:
            return False
        if lowered.startswith(SENSITIVE_PREFIXES):
            return False
        if lowered.endswith(SENSITIVE_SUFFIXES):
            return False
    return True


def _assert_safe_tree(root: Path) -> None:
    if not root.is_dir():
        raise RuntimeError(f"required directory is missing: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"refusing symlink in handoff input: {path}")
        if not _safe_name(path.relative_to(root)):
            raise RuntimeError(f"refusing sensitive-looking handoff input: {path}")


def _tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        entries.append(
            {
                "path": f"{prefix}/{relative}",
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return entries


def _tree_digest(entries: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"git command failed: {' '.join(args)}")
    return completed.stdout.strip()


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"handoff destination already exists: {destination}")
    shutil.copytree(source, destination, symlinks=False)


def _copy_file(source: Path, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"handoff destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _archive_tracked_repo(repo: Path, destination: Path, git_sha: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "archive",
                "--format=tar.gz",
                f"--prefix=stage22_repo_{git_sha[:12]}/",
                "HEAD",
            ],
            stdout=handle,
            stderr=subprocess.PIPE,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace").strip())
    with tarfile.open(destination, "r:gz") as archive:
        unsafe_members = [member.name for member in archive.getmembers() if not _safe_name(Path(member.name))]
    if unsafe_members:
        raise RuntimeError(f"tracked archive contains sensitive-looking names: {unsafe_members[:3]}")


def _validate_train_fixture(dataset_dir: Path) -> dict[str, Any]:
    split_path = dataset_dir / "meta" / "stage22_split_lock.json"
    info_path = dataset_dir / "meta" / "info.json"
    if not split_path.is_file() or not info_path.is_file():
        raise RuntimeError("train fixture must contain meta/info.json and meta/stage22_split_lock.json")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    expected = {
        "dataset_role": "TRAIN",
        "selected_split": "train",
        "selected_target_ids": EXPECTED_TRAIN_TARGETS,
        "training_allowed": True,
        "final_test_in_training": False,
    }
    for key, value in expected.items():
        if split.get(key) != value:
            raise RuntimeError(f"train fixture split lock mismatch for {key}: {split.get(key)!r}")
    return split


def build_handoff(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    dataset_dir = args.dataset_dir.resolve()
    base_model_dir = args.base_model_dir.resolve()
    modality_config = args.modality_config.resolve()
    stage10_root = args.stage10_root.resolve()
    groot_repo = args.groot_repo.resolve()
    pbs_dir = args.pbs_dir.resolve()
    output_root = args.output_root.resolve()

    if not repo.is_dir():
        raise RuntimeError(f"repository is missing: {repo}")
    if _git(repo, "branch", "--show-current") != args.expected_branch:
        raise RuntimeError(f"unexpected branch; expected {args.expected_branch}")
    status = _git(repo, "status", "--porcelain=v1")
    if status:
        raise RuntimeError("repository must be clean before handoff packaging")
    git_sha = _git(repo, "rev-parse", "HEAD")
    if args.expected_git_sha and git_sha != args.expected_git_sha:
        raise RuntimeError(f"unexpected Git SHA; expected {args.expected_git_sha}, got {git_sha}")
    if not stage10_root.is_dir() or not groot_repo.is_dir():
        raise RuntimeError("Stage10 root and GR00T repository must exist for provenance")
    if not modality_config.is_file():
        raise RuntimeError(f"modality config is missing: {modality_config}")
    _assert_safe_tree(dataset_dir)
    _assert_safe_tree(base_model_dir)
    if not _safe_name(modality_config.name):
        raise RuntimeError(f"refusing sensitive-looking modality config: {modality_config}")
    split_lock = _validate_train_fixture(dataset_dir)

    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    repo_archive = output_root / f"stage22_repo_{git_sha[:12]}.tar.gz"
    payload_root = output_root / "payload"
    pbs_output = output_root / "pbs"
    manifests_root = output_root / "manifests"
    payload_root.mkdir()
    pbs_output.mkdir()
    manifests_root.mkdir()

    _archive_tracked_repo(repo, repo_archive, git_sha)
    _copy_tree(dataset_dir, payload_root / "lerobot_train")
    _copy_tree(base_model_dir, payload_root / "GR00T-N1.7-3B")
    _copy_file(modality_config, payload_root / "modality" / modality_config.name)

    template_paths: list[Path] = []
    for template in sorted(pbs_dir.glob("stage22_h200_*.pbs")):
        if not template.is_file():
            continue
        destination = pbs_output / template.name
        _copy_file(template, destination)
        template_paths.append(destination)
    if not template_paths:
        raise RuntimeError(f"no Stage22 H200 PBS templates found in {pbs_dir}")

    payload_entries = _tree_entries(output_root / "payload", "payload")
    archive_entry = {
        "path": repo_archive.name,
        "bytes": repo_archive.stat().st_size,
        "sha256": _sha256_file(repo_archive),
    }
    pbs_entries = _tree_entries(output_root / "pbs", "pbs")
    transfer_entries = [archive_entry, *payload_entries, *pbs_entries]
    manifest = {
        "schema": "STAGE22_H200_HANDOFF_V1",
        "status": "CODE_DATA_INPUT_FROZEN_FOR_H200",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": "CODEX_CORELAB",
        "stage": "Stage22 H200 handoff packaging",
        "git": {
            "repository": str(repo),
            "branch": args.expected_branch,
            "sha": git_sha,
            "working_tree_clean": True,
        },
        "groot": {
            "repository": str(groot_repo),
            "sha": _git(groot_repo, "rev-parse", "HEAD"),
        },
        "stage10_runtime_provenance": {
            "root": str(stage10_root),
            "copied": False,
            "credentials_copied": False,
        },
        "input_artifacts": {
            "train_fixture": {
                "source": str(dataset_dir),
                "destination": "payload/lerobot_train",
                "tree_sha256": _tree_digest(_tree_entries(output_root / "payload" / "lerobot_train", "payload/lerobot_train")),
            },
            "base_model": {
                "source": str(base_model_dir),
                "destination": "payload/GR00T-N1.7-3B",
                "tree_sha256": _tree_digest(_tree_entries(output_root / "payload" / "GR00T-N1.7-3B", "payload/GR00T-N1.7-3B")),
            },
            "modality_config": {
                "source": str(modality_config),
                "destination": f"payload/modality/{modality_config.name}",
                "sha256": _sha256_file(payload_root / "modality" / modality_config.name),
            },
            "target_split_manifest": {
                "source": "manifests/STAGE22_TARGET_SPLIT_LOCK_V1.json",
                "sha256": _sha256_file(repo / "manifests" / "STAGE22_TARGET_SPLIT_LOCK_V1.json"),
            },
        },
        "dataset_split": {
            "train": EXPECTED_TRAIN_TARGETS,
            "validation": EXPECTED_VALIDATION_TARGETS,
            "locked_final_test": EXPECTED_FINAL_TEST_TARGETS,
            "materialized_dataset_role": split_lock.get("dataset_role"),
            "final_test_in_training": split_lock.get("final_test_in_training"),
        },
        "recipes": {
            "R0": {
                "tune_projector": True,
                "tune_diffusion_model": False,
                "tune_visual": False,
                "tune_llm": False,
            },
            "R1": {
                "tune_projector": True,
                "tune_diffusion_model": True,
                "tune_visual": False,
                "tune_llm": False,
            },
            "first_round_optimizer_steps": 50,
            "training_seed": args.training_seed,
            "inference_seeds": [2223, 2224, 2225],
            "action_horizon": 40,
            "action_sample_period_s": 0.05,
            "action_sample_period_provenance": "TEST_ONLY_SIM_VISUAL_REACH_V0",
        },
        "pbs_templates_prepared": [str(path.relative_to(output_root)) for path in template_paths],
        "transfer_files": transfer_entries,
        "what_was_executed": [
            "working-tree and split-lock validation",
            "tracked-source Git archive creation",
            "credential-free input copy",
            "SHA256 manifest generation",
            "PBS template preparation",
        ],
        "what_was_not_executed": [
            "Hopper login",
            "PBS submission",
            "GPU training",
            "R0/R1 optimizer steps",
            "GPU inference",
            "WBC execution",
            "hardware, ROS, ZMQ, P1M, and real-data paths",
        ],
        "safety": {
            "passwords_copied": False,
            "hf_tokens_copied": False,
            "private_ssh_keys_copied": False,
            "wbc_private_artifacts_copied": False,
            "untracked_worktree_files_packaged": False,
            "pbs_submitted": False,
        },
        "cpu_tests": args.cpu_tests,
        "handoff_root": str(output_root),
        "result": "CODE_DATA_INPUT_FROZEN_FOR_H200",
        "blocker": None,
        "next_gate_request": "CODEX_HOPPER_R1_1STEP_PREFLIGHT",
    }
    manifest_path = manifests_root / "STAGE22_H200_HANDOFF_V1.json"
    _write_json(manifest_path, manifest)

    checksum_lines = [
        f"{entry['sha256']}  {entry['path']}" for entry in [*transfer_entries, {
            "path": str(manifest_path.relative_to(output_root)),
            "sha256": _sha256_file(manifest_path),
        }]
    ]
    (manifests_root / "SHA256SUMS.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return {"manifest": manifest, "manifest_path": str(manifest_path), "checksums_path": str(manifests_root / "SHA256SUMS.sha256")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--stage10-root", type=Path, required=True)
    parser.add_argument("--groot-repo", type=Path, required=True)
    parser.add_argument("--pbs-dir", type=Path, default=Path(__file__).parents[1] / "hpc" / "pbs")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-branch", default=EXPECTED_BRANCH)
    parser.add_argument("--expected-git-sha")
    parser.add_argument("--training-seed", type=int, default=2222)
    parser.add_argument("--cpu-tests", default="UNSPECIFIED")
    args = parser.parse_args()
    try:
        result = build_handoff(args)
    except (OSError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        print(f"HANDOFF_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
