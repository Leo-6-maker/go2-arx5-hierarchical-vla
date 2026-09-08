#!/usr/bin/env python3
"""Create a strict, non-selecting Stage22 H2 R0/R1 paired comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from go2arx5_vla.data.stage22_snapshot_lock import load_snapshot_lock, snapshot_lock_file_sha256
from go2arx5_vla.metrics.h2_paired import compare_h2_profile_payloads


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(paths: list[Path]) -> list[dict[str, Any]]:
    payloads = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"profile manifest is not an object: {path}")
        payloads.append(payload)
    return payloads


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--r0",
        type=Path,
        nargs="+",
        required=True,
        help="one or more R0 per-inference-seed profile manifests",
    )
    parser.add_argument(
        "--r1",
        type=Path,
        nargs="+",
        required=True,
        help="one or more R1 per-inference-seed profile manifests",
    )
    parser.add_argument(
        "--expected-split",
        choices=("validation",),
        required=True,
        help="H2 profile selection is restricted to the locked validation split",
    )
    parser.add_argument(
        "--snapshot-manifest",
        type=Path,
        required=True,
        help="frozen Stage22 validation snapshot lock manifest",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    r0_payloads = _load(args.r0)
    r1_payloads = _load(args.r1)
    snapshot_manifest_path = args.snapshot_manifest.resolve(strict=True)
    snapshot_manifest = load_snapshot_lock(snapshot_manifest_path)
    snapshot_manifest_sha256 = snapshot_lock_file_sha256(snapshot_manifest_path)
    payload = compare_h2_profile_payloads(
        r0_payloads,
        r1_payloads,
        expected_split=args.expected_split,
        snapshot_manifest_path=str(snapshot_manifest_path),
        snapshot_manifest_sha256=snapshot_manifest_sha256,
        snapshot_manifest=snapshot_manifest,
        source_files={
            "r0": [f"{path}:{_sha256(path)}" for path in args.r0],
            "r1": [f"{path}:{_sha256(path)}" for path in args.r1],
            "snapshot_manifest": [f"{snapshot_manifest_path}:{snapshot_manifest_sha256}"],
        },
    )
    payload["cli_contract"] = {
        "r0_input_count": len(args.r0),
        "r1_input_count": len(args.r1),
        "validation_only": True,
        "required_inference_seeds": [2223, 2224, 2225],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
