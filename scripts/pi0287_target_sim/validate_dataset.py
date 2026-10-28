"""Validate actual saved target episodes; absent data is NOT_READY."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from go2arx5_vla.target_sim.dataset import validate_manifest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--clock-tolerance-s", type=float, default=1e-3)
    args = parser.parse_args()
    try:
        report, _ = validate_manifest(args.root, clock_tolerance_s=args.clock_tolerance_s)
    except (ValueError, KeyError, OSError, ImportError) as exc:
        report = dict(status="NOT_READY", reasons=[f"{type(exc).__name__}: {exc}"], neural_forwards=0, optimizer_updates=0)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "PASS_SAVED_PHYSICS_PENDING_CODEC" else 2


if __name__ == "__main__":
    raise SystemExit(main())
