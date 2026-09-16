#!/usr/bin/env python3
"""Construct and close the frozen native Stage43 environment without science."""

import argparse
import json
import os
from pathlib import Path

import stage21_native_receding_horizon as native_runner


def _write_marker(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError("refusing to overwrite native preflight marker: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()
    args.mode = "camera_smoke"
    args.arm = None
    args.dog = None
    args.output_dir = None
    args.prefix_steps = 4
    args.max_prefix_updates = 1

    trajectory = native_runner._load_pose_file(args.trajectory)
    native = env = None
    try:
        native, env, actors, loads, cfg = native_runner._build_native(args, trajectory)
        if actors is not None or loads:
            raise RuntimeError("zero-science preflight loaded WBC actors")
        marker = {
            "schema": "CORELAB_PI0203_NATIVE_PREFLIGHT_V1",
            "status": "PASS",
            "sim_dt_s": float(cfg.sim.dt),
            "model_forward": 0,
            "wbc_science_rows": 0,
            "raw": 0,
            "seed": int(args.seed),
            "sim_device": str(args.sim_device),
        }
        _write_marker(args.marker, marker)
        print(json.dumps(marker, sort_keys=True), flush=True)
    finally:
        if env is not None:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
