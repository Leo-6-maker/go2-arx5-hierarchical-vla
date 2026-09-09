#!/usr/bin/env python3
"""Freeze the CPU-only Stage45C evaluator protocol design.

This script writes only a schema/protocol manifest.  It does not enumerate
dataset files, open a checkpoint, construct a model, read locked data, or
submit a scheduler job.  Golden Release identities remain late-bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from go2arx5_vla.evaluation.checkpoint_eval import (  # noqa: E402
    EVALUATION_MANIFEST_SCHEMA,
    EXPECTED_WINDOWS,
    canonical_json_bytes,
    validate_evaluation_manifest,
)


GOLDEN_HEAD = "8197e4cd591b9c44cf9f43576d3020e9e6ff5f24"


def build_manifest() -> dict[str, Any]:
    return {
        "schema": EVALUATION_MANIFEST_SCHEMA,
        "stage": "Stage45C",
        "pi_gate": "PI_GATE_STATE_0088",
        "source_golden_head": GOLDEN_HEAD,
        "status": "PREPARED_NOT_EXECUTED",
        "owner": "corelab",
        "cohort": {
            "name": "FULL24_EXISTING_TRAIN",
            "split": "TRAIN",
            "expected_window_count": 24,
            "window_order": "target_major_then_snapshot_major",
            "window_ids": list(EXPECTED_WINDOWS),
        },
        "contract": {
            "structured_action_schema": "StructuredActionV1",
            "raw_action_shape": [1, 40, 132],
            "decoded_action_shape": [40, 10],
            "state_v1_shape": [10],
            "geometry": "K16",
            "timing": "T16",
            "horizon": "H40",
            "source_frame": "NATIVE_WBC_RELATIVE_REFERENCE",
            "tcp_semantic_id": "x5_link6_gripper_center",
            "model_input_keys": ["body_rgb", "wrist_rgb", "state"],
            "target_xyz_in_model_input": False,
            "decoder": "go2arx5_vla.stage40_structured_decode.decode_raw_structured_action",
            "normalization": "late_bound_to_Golden_Release_statistics_manifest",
        },
        "runtime_binding": {
            "golden_release_manifest": None,
            "groot_runtime_manifest": None,
            "full24_manifest": None,
            "checkpoint_manifest": None,
            "runtime_admission": None,
            "structured_statistics": None,
            "sampler_noise_seed": None,
            "output_root": None,
        },
        "late_bound": [
            "golden_release_manifest",
            "groot_runtime_manifest",
            "full24_manifest",
            "checkpoint_manifest",
            "runtime_admission",
            "structured_statistics",
            "sampler_noise_seed",
            "output_root",
        ],
        "locked_inputs": {
            "Validation-E": "LOCKED",
            "Holdout-F": "LOCKED",
            "WBC": "LOCKED",
            "new_data": "LOCKED",
            "hardware": "LOCKED",
        },
        "execution_budget": {
            "forwards_per_checkpoint": 24,
            "reference_updates": 0,
            "wbc_steps": 0,
            "checkpoints": "late_bound_to_Vanda_P0_P1_manifest",
        },
        "execution_policy": {
            "model_forward": "DEFERRED",
            "gpu_h200": "HOLD",
            "pbs_submission": "HOLD",
            "training": "HOLD",
            "safety_gate": "preserve_runtime_decision; ACCEPT_or_WARN_only",
            "raw_predictions": "preserve_separately_and_hash_bind",
            "threshold_used": None,
        },
        "evidence": {
            "metric_library": "src/go2arx5_vla/evaluation/",
            "raw_to_decode_to_metric_pipeline": True,
            "aggregate_is_cpu_only": True,
            "scientific_semantics_source": "Stage44R accepted _trajectory_metrics and _comparison_metrics",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "manifests" / "STAGE45C_TRAIN24_EVALUATION_PROTOCOL_V1.json",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest()
    validate_evaluation_manifest(manifest)
    if args.check_only:
        print(json.dumps({"schema": manifest["schema"], "status": manifest["status"], "window_count": 24}, sort_keys=True))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(manifest))
    print(json.dumps({"output": str(args.output.resolve()), "status": manifest["status"], "window_count": 24}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
