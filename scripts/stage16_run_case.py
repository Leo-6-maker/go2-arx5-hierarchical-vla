#!/usr/bin/env python3
"""Run one node-local Stage16 native case with its manifest metadata."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--dog", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-mode", default="BENCHMARK_CONTINUOUS")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    cases = {str(case["trajectory_id"]): case for case in manifest["cases"]}
    if args.trajectory_id not in cases:
        raise KeyError("trajectory is absent from manifest: %s" % args.trajectory_id)
    case = cases[args.trajectory_id]
    output_dir = args.output_root / args.trajectory_id
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "case_metadata.json"
    metadata_path.write_text(json.dumps(case, indent=2, sort_keys=True) + "\n")
    seconds = (
        float(args.seconds)
        if args.seconds is not None
        else float(case["reference_interval"][1]) + 0.02
    )
    seed = int(args.seed if args.seed is not None else case["reset_seed"])
    command = [
        sys.executable,
        str(args.launcher),
        "--source-root",
        str(args.source_root),
        "--arm",
        str(args.arm),
        "--dog",
        str(args.dog),
        "--reference-mode",
        args.reference_mode,
        "--trajectory",
        str(case["trajectory_path"]),
        "--output-dir",
        str(output_dir),
        "--seconds",
        str(seconds),
        "--seed",
        str(seed),
        "--sim-device",
        args.sim_device,
        "--metadata-json",
        str(metadata_path),
    ]
    (output_dir / "command.json").write_text(
        json.dumps(
            {
                "command": command,
                "trajectory_id": args.trajectory_id,
                "reference_mode": args.reference_mode,
                "requested_seconds": seconds,
                "seed": seed,
            },
            indent=2,
        )
        + "\n"
    )
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
