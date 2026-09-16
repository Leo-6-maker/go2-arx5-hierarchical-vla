#!/usr/bin/env python3
"""Construct and close the frozen native Stage43 environment without science."""

import argparse
import json
from pathlib import Path

import stage21_native_receding_horizon as native_runner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
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
        print(json.dumps({
            "status": "PASS",
            "sim_dt_s": float(cfg.sim.dt),
            "model_forward": 0,
            "wbc_science_rows": 0,
            "raw": 0,
        }, sort_keys=True))
    finally:
        if env is not None:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
