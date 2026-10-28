"""Target-only processor preparation. No neural model or optimizer is loaded."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from go2arx5_vla.target_sim.dataset import (build_processor_files, check_official, fit_statistics,
                                         validate_manifest, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official", type=Path, help="Actual local official GR00T source checkout")
    parser.add_argument("--tokenizer-cache", type=Path, help="Local Cosmos/Qwen tokenizer and image processor snapshot; no weights needed")
    parser.add_argument("--processor-template", type=Path, help="Local official processor_config.json and embodiment_id.json directory")
    args = parser.parse_args()
    report = dict(status="NOT_READY", neural_forwards=0, optimizer_updates=0, training_allowed=False,
                  required_cache_interface=dict(official="gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py",
                    tokenizer="tokenizer_config.json + tokenizer vocabulary + image processor config (local-only)",
                    template="processor_config.json + embodiment_id.json; pretrained neural weights unused"))
    try:
        data, samples = validate_manifest(args.root)
        report["dataset"] = data
        if data["status"] != "PASS_SAVED_PHYSICS_PENDING_CODEC":
            report["reasons"] = data["reasons"]
        elif not all((args.official, args.tokenizer_cache, args.processor_template)):
            report["reasons"] = ["Actual local official processor source/cache/template paths required"]
        else:
            stats, population = fit_statistics(samples, args.official, args.tokenizer_cache)
            build_processor_files(args.output, args.processor_template, stats, args.tokenizer_cache)
            report["statistics"] = population
            report["processor"] = check_official(samples, args.output, args.official, args.tokenizer_cache)
            report.update(status="TARGET_EXPERT_DATA_AND_FINETUNE_INPUT_READY", reasons=[],
                          next_protocol="Target-only official pretrained GR00T + declared EEF10 + native loss; frozen WBC. Source-to-Target and Panda mixing need a separately registered comparison.")
    except (ValueError, KeyError, OSError, ImportError, RuntimeError, AssertionError) as exc:
        report["reasons"] = [f"{type(exc).__name__}: {exc}"]
    write_json(args.output / "TRAINING_READINESS.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "TARGET_EXPERT_DATA_AND_FINETUNE_INPUT_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
