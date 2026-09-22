#!/usr/bin/env python3
"""Archive one immutable model tree on a public Hugging Face repository."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

from pi0243_checkpoint_persistence import inventory


def remote_inventory(api, repo_id: str, prefix: str) -> list[dict]:
    try:
        from huggingface_hub.errors import RemoteEntryNotFoundError
    except ImportError:  # sealed runtime uses the older public exception name
        from huggingface_hub.errors import EntryNotFoundError as RemoteEntryNotFoundError

    result = []
    try:
        entries = api.list_repo_tree(repo_id=repo_id, path_in_repo=prefix, recursive=True, repo_type="model")
        for entry in entries:
            path = getattr(entry, "path", None)
            size = getattr(entry, "size", None)
            if path is None or size is None:
                continue
            lfs = getattr(entry, "lfs", None)
            remote_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            result.append({"path": path, "bytes": int(size), "sha256": remote_sha})
    except RemoteEntryNotFoundError:
        return []
    return sorted(result, key=lambda item: item["path"])


def compare(local: dict, remote: list[dict], prefix: str) -> tuple[bool, str]:
    expected = {
        f"{prefix}/{entry['path']}": {"bytes": int(entry["bytes"]), "sha256": entry["sha256"]}
        for entry in local["entries"]
    }
    observed = {entry["path"]: {"bytes": entry["bytes"], "sha256": entry["sha256"]} for entry in remote}
    if set(expected) != set(observed):
        return False, "file set mismatch"
    for path, item in expected.items():
        if item["bytes"] != observed[path]["bytes"]:
            return False, f"size mismatch: {path}"
        remote_sha = observed[path]["sha256"]
        if remote_sha is not None and remote_sha != item["sha256"]:
            return False, f"sha256 mismatch: {path}"
    return True, "PASS"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--manifest-path", required=True)
    args = parser.parse_args()

    token = args.token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit("HF token file is empty")
    from huggingface_hub import HfApi, hf_hub_download

    local = inventory(args.checkpoint, label=f"pi0243/{args.path}")
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=False, exist_ok=True)
    prior = remote_inventory(api, args.repo_id, args.path)
    matches, reason = compare(local, prior, args.path)
    if prior and not matches:
        raise SystemExit(f"refusing to overwrite existing archive: {reason}")
    if not matches:
        commit = api.upload_folder(
            folder_path=str(args.checkpoint),
            repo_id=args.repo_id,
            repo_type="model",
            path_in_repo=args.path,
            commit_message=f"PI0243 archive {args.path}",
        )
        commit_id = getattr(commit, "oid", None)
    else:
        commit_id = None
    remote = remote_inventory(api, args.repo_id, args.path)
    matches, reason = compare(local, remote, args.path)
    if not matches:
        raise SystemExit(f"HF archive verification failed: {reason}")
    info = api.model_info(repo_id=args.repo_id)
    if getattr(info, "private", None) is not False:
        raise SystemExit("HF repository is not public")

    # Small control files prove authenticated consumer access without creating a
    # second 9 GiB local copy of the model shards.
    for name in ("config.json", "model.safetensors.index.json"):
        downloaded = Path(hf_hub_download(repo_id=args.repo_id, filename=f"{args.path}/{name}", repo_type="model", token=token))
        if hashlib.sha256(downloaded.read_bytes()).hexdigest() != next(item["sha256"] for item in local["entries"] if item["path"] == name):
            raise SystemExit(f"HF read-back hash mismatch: {name}")
    receipt = {
        "schema": "PI0243_HF_ARCHIVED_TREE_V1",
        "status": "DURABLE_VERIFIED",
        "repo_id": args.repo_id,
        "public": info.private is False,
        "path": args.path,
        "commit_id": commit_id,
        "local": local,
        "remote": remote,
        "verification": {"file_set_size_hash": "PASS", "consumer_control_readback": "PASS"},
    }
    payload = json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    manifest_commit = api.upload_file(
        path_or_fileobj=io.BytesIO(payload),
        path_in_repo=args.manifest_path,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="PI0243 archive receipt",
    )
    receipt["manifest_commit_id"] = getattr(manifest_commit, "oid", None)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
