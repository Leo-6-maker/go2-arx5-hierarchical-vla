"""Validate real PI0288 records; absent or synthetic data never become READY."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from go2arx5_vla.target_mujoco.dataset import validate_manifest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report, _ = validate_manifest(args.root)
    except (ValueError, KeyError, OSError, ImportError) as exc:
        report = dict(status="DATA_NOT_ADMITTED", reasons=[f"{type(exc).__name__}: {exc}"])
    report.update(neural_forwards=0, optimizer_updates=0, physics_integrations=0)
    if args.output: write_json(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "MUJOCO_TARGET_DATA_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
