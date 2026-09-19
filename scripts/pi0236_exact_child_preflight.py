#!/usr/bin/env python3
"""CPU/I/O-only preflight for the exact PI0236 evaluator child."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locked(path: Path) -> None:
    lowered = str(path).lower()
    if any(token in lowered for token in ("dev-e", "dev_e", "holdout-f", "holdout_f", "final-f", "final_f")):
        raise RuntimeError(f"locked path: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage10-root", type=Path, required=True)
    parser.add_argument("--prep-root", type=Path, required=True)
    parser.add_argument("--vla-source", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    locked(args.calib)
    locked(args.authority)
    for path in (args.stage10_root, args.prep_root, args.vla_source, args.calib, args.authority, args.selector, args.runner):
        if not path.exists():
            raise RuntimeError(f"missing preflight path: {path}")

    sys.path.insert(0, str(args.prep_root / "scripts"))
    sys.path.insert(0, str(args.vla_source / "src"))
    sys.path.insert(0, str(args.stage10_root / "src" / "isaac-gr00t"))
    import stage20s_infer_groot as stage20
    import stage21_same_scene_groot_inference as stage21

    if stage20._run_one is not stage21._run_one:
        raise RuntimeError("Stage20S helper is not the pinned Stage21 implementation")
    selector_test = subprocess.run(
        [sys.executable, str(args.selector), "--mode", "self-test"],
        text=True, capture_output=True, check=False,
    )
    if selector_test.returncode != 0:
        raise RuntimeError(f"selector self-test failed: {selector_test.stderr[-1000:]}")

    episodes = [json.loads(line) for line in (args.calib / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line]
    episodes.sort(key=lambda row: int(row["episode_index"]))
    if len(episodes) != 1088 or [int(row["episode_index"]) for row in episodes] != list(range(1088)):
        raise RuntimeError("TRAIN-CALIB row index contract failed")
    rollouts = {str(row["source_episode_id"]).split(":obs-", 1)[0] for row in episodes}
    if len(rollouts) != 68:
        raise RuntimeError(f"TRAIN-CALIB source rollout count is {len(rollouts)}, expected 68")
    authority = json.loads(args.authority.read_text(encoding="utf-8"))
    authority_rollouts = {str(item["source_rollout_id"]) for item in authority["rollouts"]}
    if authority_rollouts != rollouts:
        raise RuntimeError("target authority/source rollout mismatch")

    boundary_rows: list[dict[str, Any]] = []
    for index in (999, 1000):
        chunk = index // 1000
        stem = f"episode_{index:06d}.mp4"
        parquet = args.calib / "data" / f"chunk-{chunk:03d}" / f"episode_{index:06d}.parquet"
        body = args.calib / "videos" / f"chunk-{chunk:03d}" / "observation.images.external_rgb" / stem
        wrist = args.calib / "videos" / f"chunk-{chunk:03d}" / "observation.images.wrist_rgb" / stem
        for path in (parquet, body, wrist):
            if not path.is_file():
                raise RuntimeError(f"boundary artifact missing: {path}")
        boundary_rows.append({"episode_index": index, "parquet": str(parquet), "body": str(body), "wrist": str(wrist)})

    payload = {
        "schema": "PI0236_EXACT_CHILD_PREFLIGHT_V1",
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "stage20_helper": {"path": str(Path(stage20.__file__).resolve()), "sha256": sha256_file(Path(stage20.__file__).resolve())},
        "stage21_helper": {"path": str(Path(stage21.__file__).resolve()), "sha256": sha256_file(Path(stage21.__file__).resolve())},
        "selector": {"path": str(args.selector), "sha256": sha256_file(args.selector), "self_test_returncode": selector_test.returncode},
        "runner": {"path": str(args.runner), "sha256": sha256_file(args.runner)},
        "calib": {"path": str(args.calib), "row_count": len(episodes), "source_rollout_count": len(rollouts)},
        "boundary_artifacts": boundary_rows,
        "forward_count": 0,
        "model_loaded": False,
        "wbc_execution": False,
        "target_xyz_in_model_input": False,
        "dev_e_source_read_count": 0,
        "final_f_source_read_count": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "sha256": sha256_file(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
