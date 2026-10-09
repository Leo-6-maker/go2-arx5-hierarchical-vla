"""Real target-only official CPU processor admission; no neural model load."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from go2arx5_vla.target_mujoco.dataset import (build_processor_files, check_official,
    fit_statistics, validate_manifest, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official", type=Path)
    parser.add_argument("--tokenizer-cache", type=Path)
    parser.add_argument("--processor-template", type=Path)
    args = parser.parse_args()
    report = dict(status="DATA_NOT_ADMITTED", reasons=[], training_allowed=False,
        neural_forwards=0, optimizer_updates=0, physics_integrations=0)
    try:
        data, samples = validate_manifest(args.root)
        report["dataset"] = data
        if data["status"] != "MUJOCO_TARGET_DATA_READY":
            report["reasons"] = data["reasons"]
        elif not all((args.official, args.tokenizer_cache, args.processor_template)):
            report["reasons"] = ["Actual local official processor source/tokenizer/template paths required"]
        else:
            statistics, population = fit_statistics(samples, args.official, args.tokenizer_cache)
            build_processor_files(args.output, args.processor_template, statistics, args.tokenizer_cache)
            report["statistics"] = population
            report["processor"] = check_official(samples, args.output, args.official, args.tokenizer_cache)
            config = dict(schema="PI0288_TARGET_FINETUNE_DRAFT_V1", status="DRAFT_REQUIRES_NEW_TRAINING_AUTHORITY",
                dataset=str(args.root.resolve()), processor=str(args.output.resolve()), target_only=True,
                action_horizon=40, action_representation="RELATIVE_SE3_EEF9_PLUS_ABSOLUTE_GRIPPER",
                action_label_source=data["label_source"], rot6d_layout=data["rot6d_layout"],
                statistics_fit="TRAIN_ONLY", use_native_loss=True, official_pretrained_required=True,
                optimizer_launch_authorized=False, BASE_supervision=False)
            write_json(args.output / "FINETUNE_CONFIG_DRAFT.json", config)
            report.update(status="TARGET_FINETUNE_INPUT_READY", reasons=[])
    except (ValueError, KeyError, OSError, ImportError, RuntimeError, AssertionError, StopIteration) as exc:
        report["reasons"] = [f"{type(exc).__name__}: {exc}"]
    write_json(args.output / "TRAINING_READINESS.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "TARGET_FINETUNE_INPUT_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
