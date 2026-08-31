#!/usr/bin/env python3
"""Write a small reproducibility manifest without reading private artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-sha", default="UNKNOWN")
    parser.add_argument("--config-sha", default="UNKNOWN")
    parser.add_argument("--reference-mode", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    manifest = {
        "schema": "RUN_MANIFEST_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "integration_repo_git_sha": git_sha(args.repo),
        "delivered_wbc_archive_sha256": "b52a53bef87340e6244ead352fa434390faee65246cd32ab8e6648336a018167",
        "arm_checkpoint_sha256": os.environ.get("WBC_ARM_CHECKPOINT_SHA256", "UNKNOWN"),
        "dog_checkpoint_sha256": os.environ.get("WBC_DOG_CHECKPOINT_SHA256", "UNKNOWN"),
        "trajectory_sha256": args.trajectory_sha,
        "experiment_config_sha256": args.config_sha,
        "runtime": os.environ.get("WBC_RUNTIME_VERSION", "UNKNOWN"),
        "simulator": "IsaacGym_native_delivered_runtime",
        "reference_mode": args.reference_mode,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
