#!/usr/bin/env python3
"""Online, no-model HF publisher for PI0243 checkpoint receipts."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from pi0243_checkpoint_persistence import checkpoint_record, write_json_durable


def stable_record(root: Path, step: int, max_steps: int, family: str, timeout: float, poll: float) -> dict:
    deadline = time.monotonic() + timeout
    previous = None
    while time.monotonic() < deadline:
        try:
            record = checkpoint_record(root, expected_step=step, expected_max_steps=max_steps, family=family)
        except (OSError, ValueError, RuntimeError):
            time.sleep(poll)
            continue
        signature = (record["inventory"]["file_count"], record["inventory"]["total_bytes"], record["inventory"]["tree_sha256"])
        if signature == previous:
            return record
        previous = signature
        time.sleep(poll)
    raise RuntimeError(f"checkpoint-{step} did not stabilize")


def publish(args: argparse.Namespace, step: int) -> dict:
    root = args.output_root / "training" / f"checkpoint-{step}"
    source = stable_record(root, step, args.max_steps, args.family, args.checkpoint_timeout, args.poll_seconds)
    prefix = f"{args.path_prefix.rstrip('/')}/checkpoint-{step}"
    manifest_path = f"{args.path_prefix.rstrip('/')}/manifests/checkpoint-{step}.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("pi0243_hf_archive_tree.py")),
        "--token-file", str(args.token_file),
        "--checkpoint", str(root),
        "--repo-id", args.repo_id,
        "--path", prefix,
        "--manifest-path", manifest_path,
    ]
    environment = dict(os.environ)
    environment["HF_HUB_OFFLINE"] = "0"
    environment.pop("TRANSFORMERS_OFFLINE", None)
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)
    if result.returncode:
        raise RuntimeError(f"HF archive failed for checkpoint-{step}: {result.stdout[-4000:]}")
    try:
        payload = result.stdout[result.stdout.rfind("\n{") + 1 :].strip()
        archive = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"HF archive emitted no JSON receipt for checkpoint-{step}") from exc
    if archive.get("status") != "DURABLE_VERIFIED":
        raise RuntimeError(f"HF archive is not verified for checkpoint-{step}")
    receipt = {
        "schema": "PI0243_DURABLE_CHECKPOINT_RECEIPT_V1",
        "status": "DURABLE_VERIFIED",
        "family": args.family,
        "checkpoint_step": step,
        "max_steps": args.max_steps,
        "job_id": os.environ.get("PBS_JOBID", "local"),
        "source": str(root.resolve()),
        "durable_host": "huggingface",
        "durable_path": f"{args.repo_id}/{prefix}",
        "source_record": source,
        "durable_inventory": archive.get("remote", []),
        "hf_manifest_commit_id": archive.get("manifest_commit_id"),
        "verified_at_epoch_s": time.time(),
    }
    receipt_path = args.output_root / "durable_receipts" / f"checkpoint-{step}.durable_receipt.json"
    write_json_durable(receipt_path, receipt)
    with args.state_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": step, "status": "DURABLE_VERIFIED", "prefix": prefix, "tree_sha256": source["inventory"]["tree_sha256"], "total_bytes": source["inventory"]["total_bytes"], "manifest_commit_id": archive.get("manifest_commit_id")}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--path-prefix", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--family", default="PI0243_STAGE_A_V1")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--checkpoint-timeout", type=float, default=21600.0)
    parser.add_argument("--state-path", type=Path, required=True)
    args = parser.parse_args()
    if not args.token_file.is_file() or not args.token_file.read_text(encoding="utf-8").strip():
        raise SystemExit("HF token file is missing or empty")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*", args.path_prefix):
        raise SystemExit("unsafe HF path prefix")
    args.state_path.parent.mkdir(parents=True, exist_ok=True)
    for step in args.steps:
        publish(args, int(step))
    print(json.dumps({"schema": "PI0243_HF_CHECKPOINT_SIDECAR_V1", "status": "COMPLETE", "steps": args.steps, "repo_id": args.repo_id, "path_prefix": args.path_prefix}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
