#!/usr/bin/env python3
"""Materialize and/or launch the explicit 10/25/50/100 episode ladder."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage10-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[10, 25, 50, 100])
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--run-training", action="store_true")
    args = parser.parse_args()
    if not args.build_only and not args.run_training:
        parser.error("choose --build-only or --run-training")
    if sorted(set(args.counts)) != list(args.counts) or any(count < 1 for count in args.counts):
        parser.error("--counts must be positive and sorted")

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    repo_root = Path(__file__).resolve().parents[1]
    builder = repo_root / "scripts" / "stage20s_build_visual_reach.py"
    qualifier = repo_root / "scripts" / "stage19_official_dataloader.py"
    trainer = repo_root / "scripts" / "stage20s_train_gr00t.py"
    for count in args.counts:
        case_root = args.output_root / f"easy_{count}"
        dataset_dir = case_root / "lerobot"
        if args.build_only:
            command = [
                sys.executable,
                str(builder),
                "--output-dir",
                str(case_root),
                "--episodes",
                str(count),
                "--train-episode-count",
                str(count),
                "--overwrite",
            ]
            subprocess.run(command, check=True)
        row = {
            "episode_count": count,
            "dataset_dir": str(dataset_dir),
            "output_dir": str(args.output_root / f"train_{count}"),
            "status": "MATERIALIZED" if args.build_only else "PLANNED",
        }
        if args.run_training:
            # The official GR00T training path consumes relative action
            # statistics.  A freshly materialized fixture has only the
            # project-owned stats.json until this read-only qualification pass
            # generates relative_stats.json.  Make that prerequisite explicit
            # instead of allowing the training command to fail later with an
            # opaque missing-file error.
            relative_stats = dataset_dir / "meta" / "relative_stats.json"
            if not relative_stats.is_file():
                qualification_dir = case_root / "official_dataloader"
                subprocess.run(
                    [
                        sys.executable,
                        str(qualifier),
                        "--stage10-root",
                        str(args.stage10_root),
                        "--dataset-dir",
                        str(dataset_dir),
                        "--output-dir",
                        str(qualification_dir),
                    ],
                    check=True,
                )
                row["official_dataloader_qualification"] = str(
                    qualification_dir / "stage19_official_dataloader.json"
                )
            else:
                row["official_dataloader_qualification"] = "pre-existing relative_stats.json"
            command = [
                sys.executable,
                str(trainer),
                "--stage10-root",
                str(args.stage10_root),
                "--base-model-path",
                str(args.base_model_path),
                "--dataset-dir",
                str(dataset_dir),
                "--modality-config",
                str(args.modality_config),
                "--output-dir",
                row["output_dir"],
                "--max-steps",
                str(args.max_steps),
                "--save-steps",
                str(args.save_steps),
            ]
            subprocess.run(command, check=True)
            row["status"] = "TRAINED"
        rows.append(row)
    result = {
        "schema": "STAGE20S_TRAINING_LADDER_V1",
        "simulation_only": True,
        "hardware_control": False,
        "counts": args.counts,
        "runs": rows,
        "training_recipe": "projector/state/action-encoder adaptation; diffusion transformer frozen by default",
        "action_period_provenance": "TEST_ONLY_SIM_VISUAL_REACH_V0",
    }
    path = args.output_root / "training_ladder_manifest.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
