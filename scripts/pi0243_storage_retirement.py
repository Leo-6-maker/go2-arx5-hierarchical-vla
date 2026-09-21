#!/usr/bin/env python3
"""Manifest-first retirement of explicitly listed retired checkpoint replicas.

Storage-only utility.  It never scans a parent tree or follows symlinks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from urllib.request import Request, urlopen


EXPECTED_PI0241_TREES = {
    250: "746d498b0ff4e7ee2c935c588290d34b35d5662093e85abc5e7fed74de6e04cd",
    500: "5c693760152fe803638c0eb8da7f23011a3e0168e32762dfbbf51609a8297688",
    1000: "bbf0d48a39b7433f3a650ac3a34a69a7deb5058f89cb3eab65897aa6a94b6180",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_inventory(root: Path) -> dict:
    if not root.exists():
        return {"status": "NOT_PRESENT", "path": str(root)}
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"candidate is not a real directory: {root}")
    entries = []
    seen = set()
    logical_bytes = 0
    allocated_bytes = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink in candidate: {path}")
        if not path.is_file():
            continue
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": stat.st_size,
                "allocated_bytes": int(getattr(stat, "st_blocks", 0)) * 512,
                "sha256": sha256_file(path),
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "nlink": stat.st_nlink,
            }
        )
        logical_bytes += stat.st_size
        if identity not in seen:
            allocated_bytes += int(getattr(stat, "st_blocks", 0)) * 512
            seen.add(identity)
    stat = root.stat()
    return {
        "status": "PRESENT",
        "path": str(root),
        "file_count": len(entries),
        "logical_bytes": logical_bytes,
        "unique_allocated_bytes": allocated_bytes,
        "device": stat.st_dev,
        "entries": entries,
    }


def tree_hash(step: int, entries: list[dict]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item["path"]):
        digest.update(
            f"pi0241/checkpoint-{step}/{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode()
        )
    return digest.hexdigest()


def lfs_sha(entry) -> str | None:
    value = getattr(entry, "lfs", None)
    if isinstance(value, dict):
        return value.get("sha256")
    return getattr(value, "sha256", None)


def remote_files(api, repo_id: str, prefix: str, revision: str) -> dict[str, dict]:
    from huggingface_hub.errors import EntryNotFoundError

    result = {}
    try:
        entries = api.list_repo_tree(
            repo_id=repo_id,
            path_in_repo=prefix,
            recursive=True,
            revision=revision,
            repo_type="model",
        )
        for entry in entries:
            path = getattr(entry, "path", None)
            size = getattr(entry, "size", None)
            if path is not None and size is not None:
                result[path] = {"size": int(size), "sha256": lfs_sha(entry)}
    except EntryNotFoundError:
        return {}
    return result


def stream_remote_sha(repo_id: str, path: str, revision: str, token: str | None) -> str:
    from huggingface_hub import hf_hub_url

    request = Request(hf_hub_url(repo_id, filename=path, revision=revision, repo_type="model"))
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    digest = hashlib.sha256()
    with urlopen(request, timeout=120) as response:
        for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_remote_tree(api, repo_id: str, prefix: str, revision: str, expected: dict[str, dict], token: str | None) -> dict:
    observed = remote_files(api, repo_id, prefix, revision)
    expected_paths = {f"{prefix}/{item['path']}": item for item in expected.values()}
    if set(observed) != set(expected_paths):
        return {
            "status": "FAIL_FILE_SET",
            "missing": sorted(set(expected_paths) - set(observed)),
            "extra": sorted(set(observed) - set(expected_paths)),
        }
    hashes = {}
    for path, item in expected_paths.items():
        remote = observed[path]
        if remote["size"] != int(item["bytes"]):
            return {"status": "FAIL_SIZE", "path": path, "expected": item["bytes"], "observed": remote["size"]}
        method = "lfs_sha256"
        remote_hash = remote["sha256"]
        if remote_hash is None:
            method = "stream_sha256"
            remote_hash = stream_remote_sha(repo_id, path, revision, token)
        hashes[path] = {"method": method, "sha256": remote_hash}
        if remote_hash != item["sha256"]:
            return {"status": "FAIL_SHA256", "path": path, "expected": item["sha256"], "observed": remote_hash, "method": method}
    return {"status": "PASS", "file_count": len(hashes), "files": hashes}


def verify_manifest_file(api, repo_id: str, path: Path, remote_path: str, revision: str, token: str | None) -> dict:
    observed = remote_files(api, repo_id, remote_path, revision)
    if set(observed) != {remote_path}:
        return {"status": "FAIL_FILE_SET", "path": remote_path, "observed": sorted(observed)}
    local_size = path.stat().st_size
    if observed[remote_path]["size"] != local_size:
        return {"status": "FAIL_SIZE", "path": remote_path}
    remote_hash = observed[remote_path]["sha256"] or stream_remote_sha(repo_id, remote_path, revision, token)
    local_hash = sha256_file(path)
    return {"status": "PASS" if remote_hash == local_hash else "FAIL_SHA256", "sha256": remote_hash, "local_sha256": local_hash}


def checkpoint_expected(run_root: Path, step: int) -> dict:
    payload = json.loads((run_root / f"pi0241_checkpoint_{step}_manifest.json").read_text(encoding="utf-8"))
    if payload.get("tree_sha256") != EXPECTED_PI0241_TREES[step]:
        raise RuntimeError(f"canonical PI0241 tree hash drift at step {step}")
    return {entry["path"]: entry for entry in payload["entries"]}


def revision(api, repo_id: str) -> str:
    commits = api.list_repo_commits(repo_id=repo_id, repo_type="model")
    if not commits:
        raise RuntimeError(f"HF repository has no commits: {repo_id}")
    return commits[0].commit_id


def upload_failed(api, repo_id: str, local: Path, prefix: str) -> dict:
    expected = tree_inventory(local)
    if expected.get("status") != "PRESENT":
        return expected
    current = revision(api, repo_id)
    observed = remote_files(api, repo_id, prefix, current)
    expected_paths = {f"{prefix}/{item['path']}": item for item in expected["entries"]}
    if observed and set(observed) != set(expected_paths):
        raise RuntimeError(f"conflicting HF failed-attempt prefix: {prefix}")
    if observed:
        for path, item in expected_paths.items():
            remote_hash = observed[path]["sha256"] or stream_remote_sha(repo_id, path, current, None)
            if observed[path]["size"] != item["bytes"] or remote_hash != item["sha256"]:
                raise RuntimeError(f"conflicting HF failed-attempt file: {path}")
        return {"status": "ALREADY_MATCHED", "prefix": prefix, "tree": expected}
    api.upload_folder(folder_path=str(local), repo_id=repo_id, repo_type="model", path_in_repo=prefix, commit_message=f"PI0243 archive failed attempt {prefix}")
    return {"status": "UPLOADED", "prefix": prefix, "tree": expected}


def candidate(run_root: Path, relative: str, kind: str) -> dict:
    path = run_root / relative
    return {"id": relative.replace("/", "_"), "kind": kind, "path": str(path), "inventory": tree_inventory(path)}


def delete_exact(path: Path) -> None:
    if path.is_symlink() or not path.exists() or not path.is_dir():
        raise RuntimeError(f"refusing deletion target: {path}")
    shutil.rmtree(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--active-job", action="append", default=[])
    parser.add_argument("--protected-path", action="append", default=[])
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import HfApi, get_token

    api = HfApi()
    info = api.model_info(args.repo_id, files_metadata=True)
    token = get_token()
    rev = revision(api, args.repo_id)

    valid_steps = {}
    for step in (250, 500, 1000):
        expected = checkpoint_expected(args.run_root, step)
        valid_steps[str(step)] = {
            "canonical_tree_sha256": EXPECTED_PI0241_TREES[step],
            "remote_prefix": f"checkpoints/checkpoint-{step}",
            "checkpoint_manifest": str(args.run_root / f"pi0241_checkpoint_{step}_manifest.json"),
            "remote_checkpoint": verify_remote_tree(api, args.repo_id, f"checkpoints/checkpoint-{step}", rev, expected, token),
            "remote_manifest": verify_manifest_file(api, args.repo_id, args.run_root / f"pi0241_checkpoint_{step}_manifest.json", f"manifests/checkpoint-{step}.json", rev, token),
        }

    failed = {}
    for job, relative in (("625381", "recovery_audit.failed-625381"), ("625384", "recovery_audit.failed-625384")):
        failed[job] = candidate(args.run_root, relative, "failed_attempt")
        if failed[job]["inventory"].get("status") == "PRESENT":
            failed[job]["archive"] = upload_failed(api, args.repo_id, Path(failed[job]["path"]), f"failed-attempts/{job}")

    final_rev = revision(api, args.repo_id)
    if final_rev != rev:
        for record in valid_steps.values():
            step = int(Path(record["remote_prefix"]).name.split("-")[-1])
            expected = checkpoint_expected(args.run_root, step)
            record["remote_checkpoint_after_upload"] = verify_remote_tree(api, args.repo_id, record["remote_prefix"], final_rev, expected, token)
            record["remote_manifest_after_upload"] = verify_manifest_file(api, args.repo_id, args.run_root / f"pi0241_checkpoint_{step}_manifest.json", f"manifests/checkpoint-{step}.json", final_rev, token)

    candidates = [
        candidate(args.run_root, "screen_models/checkpoint-250", "verified_archive_screen_replica"),
        candidate(args.run_root, "screen_models/checkpoint-500", "verified_archive_screen_replica"),
        candidate(args.run_root, "training/checkpoint-1000", "verified_archive_full_replica"),
    ]
    all_verified = all(
        record.get("remote_checkpoint_after_upload", record["remote_checkpoint"]).get("status") == "PASS"
        and record.get("remote_manifest_after_upload", record["remote_manifest"]).get("status") == "PASS"
        for record in valid_steps.values()
    )
    failed_verified = all(
        record.get("inventory", {}).get("status") != "PRESENT"
        or record.get("archive", {}).get("status") in {"UPLOADED", "ALREADY_MATCHED"}
        for record in failed.values()
    )
    dependency = {"active_jobs": args.active_job, "protected_paths": args.protected_path, "status": "PASS"}
    for item in candidates:
        if any(Path(protected).resolve() == Path(item["path"]).resolve() for protected in args.protected_path):
            dependency["status"] = "FAIL_PROTECTED_PATH"
    manifest = {
        "schema": "PI0243_CHECKPOINT_RETIREMENT_MANIFEST_V1",
        "status": "VERIFIED_PENDING_DELETE" if all_verified and failed_verified and dependency["status"] == "PASS" else "BLOCKED",
        "repo_id": args.repo_id,
        "repo_name": getattr(info, "id", args.repo_id),
        "hf_revision": final_rev,
        "valid_checkpoints": valid_steps,
        "failed_attempts": failed,
        "candidates": candidates,
        "dependency": dependency,
        "protected_unknowns": [
            str(args.run_root / "training/checkpoint-1000.incomplete-625370.hopper-m-02"),
            "/scratch/e1192548/pi0243_stage_a_runtime_r1_20260921",
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".partial")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.manifest)

    if args.delete:
        if manifest["status"] != "VERIFIED_PENDING_DELETE":
            raise RuntimeError("retirement manifest is not verified; refusing deletion")
        for item in candidates:
            current = tree_inventory(Path(item["path"]))
            if current.get("status") == "PRESENT" and current["logical_bytes"] != item["inventory"]["logical_bytes"]:
                raise RuntimeError(f"candidate changed before deletion: {item['path']}")
        for item in candidates:
            if Path(item["path"]).exists():
                delete_exact(Path(item["path"]))
        manifest["status"] = "DELETED"
        manifest["post_delete"] = [candidate(args.run_root, relative, kind) for relative, kind in (
            ("screen_models/checkpoint-250", "verified_archive_screen_replica"),
            ("screen_models/checkpoint-500", "verified_archive_screen_replica"),
            ("training/checkpoint-1000", "verified_archive_full_replica"),
        )]
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, args.manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
