#!/usr/bin/env python3
"""Reconstruct the Stage23 O1 recipe and choose one future Hopper diagnostic.

This is a read-only CPU audit of sealed manifests, YAML configuration, and
already committed Stage23/24 JSON evidence.  It never constructs a model,
loads weights, opens CUDA, runs inference, submits PBS, or reads validation
or locked-test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


EXPECTED_BASE_INDEX_SHA256 = "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59"
EXPECTED_BASE_TREE_SHA256 = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
EXPECTED_GROOT_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
EXPECTED_H2_STATS_SHA256 = "fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23"
EXPECTED_H2_RELATIVE_STATS_SHA256 = "8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML object: {path}")
    return value


def source_record(path: Path, *, expected: str | None = None) -> dict[str, Any]:
    actual = sha256_file(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": actual,
        "expected_sha256": expected,
        "expected_match": expected is None or actual == expected,
    }


def pick(mapping: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {key: mapping.get(key) for key in keys}


def audit(
    run_manifest_path: Path,
    campaign_manifest_path: Path,
    config_path: Path,
    upstream_path: Path,
    checkpoint_path: Path,
    prefix_path: Path,
    normalization_path: Path,
) -> dict[str, Any]:
    run_manifest = load_json(run_manifest_path)
    campaign = load_json(campaign_manifest_path)
    config = load_yaml(config_path)
    upstream = load_json(upstream_path)
    checkpoint = load_json(checkpoint_path)
    prefix = load_json(prefix_path)
    normalization = load_json(normalization_path)

    model = config.get("model", {})
    data = config.get("data", {})
    training = config.get("training", {})
    datasets = data.get("datasets", [])
    dataset_entry = datasets[0] if datasets and isinstance(datasets[0], Mapping) else {}
    upstream_findings = upstream.get("findings", {})
    exposure = upstream_findings.get("exposure_scale", {})

    o1_steps = int(campaign.get("candidate_steps", [200])[-1])
    o1_batch = int(training.get("global_batch_size", 1))
    grad_accum = int(training.get("gradient_accumulation_steps", 1))
    o1_presentations = o1_steps * o1_batch * grad_accum
    official_presentations = exposure.get("official_nominal_presentations")

    prefix_experiments = prefix.get("experiments", {})
    o1_200 = prefix_experiments.get("O1_200", {}).get("horizons", {})
    h4 = o1_200.get("4", {})
    h8 = o1_200.get("8", {})
    h40 = o1_200.get("40", {})
    checkpoint_class = checkpoint.get("id10", {}).get("classification")
    normalization_class = normalization.get("classification", {})

    exact_identity = {
        "base_model_index_sha256": run_manifest.get("base_model_index_sha256"),
        "expected_base_model_index_sha256": EXPECTED_BASE_INDEX_SHA256,
        "base_model_index_match": run_manifest.get("base_model_index_sha256") == EXPECTED_BASE_INDEX_SHA256,
        "base_model_tree_sha256": EXPECTED_BASE_TREE_SHA256,
        "groot_sha": run_manifest.get("groot_repo_sha"),
        "expected_groot_sha": EXPECTED_GROOT_SHA,
        "groot_sha_match": run_manifest.get("groot_repo_sha") == EXPECTED_GROOT_SHA,
        "dataset_dir": run_manifest.get("dataset_dir"),
        "modality_config": run_manifest.get("modality_config"),
        "simulation_only": run_manifest.get("simulation_only"),
        "hardware_control": run_manifest.get("hardware_control"),
    }

    recipe = {
        "model": pick(
            model,
            "model_type",
            "model_dtype",
            "model_name",
            "backbone_model_type",
            "tune_projector",
            "tune_diffusion_model",
            "tune_visual",
            "tune_llm",
            "tune_vlln",
            "use_relative_action",
            "max_state_dim",
            "max_action_dim",
            "action_horizon",
            "num_inference_timesteps",
            "state_dropout_prob",
            "use_mean_std",
            "override_pretraining_statistics",
        ),
        "data": {
            **pick(
                data,
                "mode",
                "shuffle",
                "seed",
                "episode_sampling_rate",
                "num_shards_per_epoch",
                "allow_padding",
                "subsample_ratio",
            ),
            "dataset_entry": pick(dataset_entry, "embodiment_tag", "dataset_type", "mix_ratio", "val_dataset_path"),
            "dataset_path": dataset_entry.get("dataset_paths", [None])[0] if isinstance(dataset_entry.get("dataset_paths"), list) else None,
            "action_indices": data.get("modality_configs", {}).get("new_embodiment", {}).get("action", {}).get("delta_indices"),
        },
        "training": pick(
            training,
            "max_steps",
            "global_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "lr_scheduler_type",
            "weight_decay",
            "warmup_ratio",
            "warmup_steps",
            "max_grad_norm",
            "optim",
            "tf32",
            "fp16",
            "bf16",
            "eval_bf16",
            "gradient_checkpointing",
            "num_gpus",
            "dataloader_num_workers",
            "transformers_local_files_only",
        ),
        "run_manifest_contract": pick(
            run_manifest,
            "schema",
            "status",
            "max_steps",
            "save_steps",
            "action_horizon",
            "action_sample_period_s",
            "action_sample_period_provenance",
            "tune_projector",
            "tune_diffusion_model",
            "tune_vlln",
            "tune_llm",
            "tune_visual",
            "load_bf16",
        ),
    }

    o1_vs_official = {
        "o1_steps": o1_steps,
        "o1_batch": o1_batch,
        "o1_gradient_accumulation": grad_accum,
        "o1_nominal_example_presentations": o1_presentations,
        "o1_stochastic_noise_time_draws": o1_presentations,
        "official_so100_steps": exposure.get("official_steps"),
        "official_so100_batch": exposure.get("official_batch"),
        "official_nominal_example_presentations": official_presentations,
        "official_to_o1_nominal_ratio": (official_presentations / o1_presentations if official_presentations and o1_presentations else None),
        "presentation_count_caveat": exposure.get("caveat"),
    }

    prefix_observation = {
        "O1_200_H4": pick(h4, "position_error_m", "endpoint_position_error_m", "path_length_ratio", "mean_speed_ratio", "max_speed_ratio", "roughness_ratio", "linear_speed_gate_counts"),
        "O1_200_H8": pick(h8, "position_error_m", "endpoint_position_error_m", "path_length_ratio", "mean_speed_ratio", "max_speed_ratio", "roughness_ratio", "linear_speed_gate_counts"),
        "O1_200_H40": pick(h40, "position_error_m", "endpoint_position_error_m", "path_length_ratio", "mean_speed_ratio", "max_speed_ratio", "roughness_ratio", "linear_speed_gate_counts"),
        "interpretation": "H4 and H8 are truncated views of sealed H40 predictions; this is descriptive prefix evidence, not a causal horizon experiment.",
    }

    next_experiment = {
        "choice": "H2_LARGER_STOCHASTIC_FLOW_EXPOSURE_SAME_SAMPLE_H40_OVERFIT",
        "status": "PREPARED_NOT_EXECUTED",
        "proposed_single_change": "increase exposure from the exact O1 B1/N200 diagnostic while retaining H40, the same TINY1_H2STATS sample, the same StateV1/action ABI, and the same stochastic flow path",
        "proposed_protocol": {
            "dataset": "TINY1_H2STATS",
            "training_split_only": True,
            "validation_e_read": False,
            "holdout_f_read": False,
            "action_horizon": 40,
            "action_period_s": 0.05,
            "action_period_provenance": "TEST_ONLY_STAGE22_SIMULATION",
            "training_seed": 2222,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "proposed_max_steps": 2000,
            "tune_projector": True,
            "tune_diffusion_model": True,
            "tune_visual": False,
            "tune_llm": False,
            "tune_vlln": False,
            "num_inference_timesteps": 4,
            "inference_seeds": [2223, 2224, 2225],
            "no_smoothing_clipping_retiming": True,
        },
        "why_this_one": [
            "The exact ID10 checkpoint row is present but initializer-like across all five audited blocks, so custom-slot learning/exposure remains an unresolved direct confound.",
            "The O1 profile defect is already visible in the H4/H8 prefixes, so a shorter horizon is not yet the leading explanation.",
            "Physical translation statistics are nondegenerate after the structural first relative point and are not shown to be catastrophically out of scale; rotation/gripper zero-width cells are task-constant fixture facts.",
            "Pinned upstream source contains a dense H40 relative-EEF precedent, while the O1 nominal presentation budget is 320x below the pinned SO100 example.",
        ],
        "falsifier": "If a controlled higher-exposure same-sample H40 run still has a healthy teacher-forced/decoded target but the same free-running oscillation, exposure alone is not sufficient; do not silently pivot within this protocol.",
    }

    return {
        "schema": "STAGE24_RECIPE_UPSTREAM_AUDIT_V1",
        "stage": "Stage24-A4/A5 exact adaptation recipe, exposure, and upstream precedent audit",
        "source_files": {
            "phase_a_run_manifest": source_record(run_manifest_path),
            "phase_a_campaign_manifest": source_record(campaign_manifest_path),
            "phase_a_config": source_record(config_path),
            "upstream_architecture_review": source_record(upstream_path),
            "static_checkpoint_audit": source_record(checkpoint_path),
            "prefix_audit": source_record(prefix_path),
            "normalization_audit": source_record(normalization_path),
        },
        "exact_identity": exact_identity,
        "recipe": recipe,
        "exposure_comparison": o1_vs_official,
        "upstream_precedent": {
            "tag": upstream_findings.get("pretrained_h40_relative_eef", {}).get("tag"),
            "type": upstream_findings.get("pretrained_h40_relative_eef", {}).get("type"),
            "representation": upstream_findings.get("pretrained_h40_relative_eef", {}).get("representation"),
            "format": upstream_findings.get("pretrained_h40_relative_eef", {}).get("format"),
            "action_horizon": upstream_findings.get("pretrained_h40_relative_eef", {}).get("action_horizon"),
            "interpretation": upstream_findings.get("pretrained_h40_relative_eef", {}).get("interpretation"),
            "official_guidance": upstream_findings.get("official_guidance"),
        },
        "checkpoint_and_prefix_findings": {
            "id10_classification": checkpoint_class,
            "id10_claim_boundary": checkpoint.get("id10", {}).get("claim_boundary"),
            "normalization_classification": normalization_class,
            "prefix": prefix_observation,
        },
        "next_experiment": next_experiment,
        "stop_boundary": {
            "new_hopper_training": False,
            "new_model_forward": False,
            "new_inference": False,
            "pbs_submitted": False,
            "wbc": False,
            "hardware": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "future_candidate_requires_pi_review": True,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upstream-review", type=Path, required=True)
    parser.add_argument("--checkpoint-audit", type=Path, required=True)
    parser.add_argument("--prefix-audit", type=Path, required=True)
    parser.add_argument("--normalization-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = [
        args.run_manifest,
        args.campaign_manifest,
        args.config,
        args.upstream_review,
        args.checkpoint_audit,
        args.prefix_audit,
        args.normalization_audit,
    ]
    for path in paths:
        path.expanduser().resolve(strict=True)
    result = audit(*(path.expanduser().resolve(strict=True) for path in paths))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite existing output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
