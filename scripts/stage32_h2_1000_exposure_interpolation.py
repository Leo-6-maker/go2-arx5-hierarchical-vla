#!/usr/bin/env python3
"""Stage32 H2@1000 exposure interpolation, exact and bounded."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

SCHEMA = "STAGE32_H2_1000_EXPOSURE_INTERPOLATION_V1"
PI_GATE = "PI_GATE_STATE_0014"
NEXT_GATE = "GPT_PI_REVIEW_AFTER_STAGE32_EXPOSURE_CURVE"
PROJECT = "CFP03-CF-004"
HISTORICAL_TRAINING_SHA = "a361e5cb8a4be33f40a21c9bbf7ad742e25c7fd0"
GROOT_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
BASE_TREE = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
H2_1000_TREE = "29f3e8f46cf41441696bfc4a5e7601378648c26fa675a9685c003080da0c4114"
H2_1000_INDEX = "ff5a9ac57951658b775ef7e958a3469900f767cec60cbf487b3c8d0a3d787bc7"
H2_1000_BYTES = 20889508912
H2_2000_TREE = "3106f9b5993b390438b62594d044781d11b393452a49f80b5c8e9e910606d0ba"
H2_2000_SEAL = "1349896f7b31a930764dbd9d849c5368e48372d0dc6273449f42f19f82a69a82"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
STAGE28_RESULT = "aa0ab03b851ad1ad6581a71168d92d682deee546db1bb79b76acb2c1acfe3f26"
STAGE31_RESULT = "c1d9cb9312cada95b35515fb44f87520a2133c7948785d19248613d6c3baf982"
STAGE31_POSTHOC = "49615ebc5129bc45c66c09206320f0e65978a4d6a25eda6e20e363bcdcaecd14"
TINY1_TREE = "541bf4a1b9ea459637525ef11fceea5c7f192c0902f6013f0331a2a668038157"
MODALITY_SHA = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
STATS_SHA = "fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23"
RELATIVE_STATS_SHA = "8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1"
SEEDS = (2223, 2224, 2225)
CONTINUATION_SAMPLER_SEEDS = (2224, 2225)
R1_SOURCE_HEAD = "90df42dd1a18a7ee423c9ac448349b0919e8c9b0"
R1_PARTIAL_TRACE_SHA = "49fc754c5197b214b55b753640a2820d2427a38b4af8327ce3e098c12489769b"
R1_RESULT_SHA = "08962dbd607bec236b2edf3a5a59a811691cb4b0b7e7cd7b6089c4140477adda"
R1_MANIFEST_SHA = "eb1a240ac38f6f7fdfc7cc9e68483a1e5f94a2e6d63e9664251c7f04da77cacb"
R1_SEAL_SHA = "877a5603a7472062dedc64d94a9b6a564a203e89644ffb33b040f4c27818ef71"


class Stage32Error(RuntimeError):
    pass


def sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise Stage32Error(f"unsafe/missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    if path.is_symlink() or not path.is_file():
        raise Stage32Error(f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def update_execution_state(path: Path, fields: Mapping[str, object]) -> dict:
    """Atomically merge runtime state without allowing counters to regress."""

    current = read_json(path) if path.exists() else {}
    merged = dict(current)
    merged.update(fields)
    for key in ("teacher_forced_probes", "sampler_calls"):
        old_value = int(current.get(key, 0))
        new_value = int(fields.get(key, old_value))
        merged[key] = max(old_value, new_value)
    if current.get("scientific_execution_begun") is True:
        merged["scientific_execution_begun"] = True
    write_json(path, merged)
    return merged


def _teacher_record_key(row: Mapping[str, object]) -> tuple[str, str]:
    return str(row.get("checkpoint")), str(row.get("probe_id"))


def load_teacher_records(output_root: Path) -> list[dict]:
    record_root = output_root / "teacher_records"
    if not record_root.exists():
        return []
    paths = sorted(record_root.glob("record_*.json"))
    records = [read_json(path) for path in paths]
    keys = [_teacher_record_key(row) for row in records]
    if len(keys) != len(set(keys)):
        raise Stage32Error("duplicate persisted teacher record")
    return records


def persist_teacher_record(output_root: Path, row: Mapping[str, object]) -> int:
    """Persist one completed teacher probe before the next probe starts."""

    records = load_teacher_records(output_root)
    key = _teacher_record_key(row)
    if key in {_teacher_record_key(item) for item in records}:
        raise Stage32Error(f"duplicate teacher record callback: {key}")
    record_root = output_root / "teacher_records"
    record_root.mkdir(parents=True, exist_ok=True)
    record_path = record_root / f"record_{len(records) + 1:02d}.json"
    write_json(record_path, dict(row))
    records.append(dict(row))
    write_json(
        output_root / "stage32_teacher_forced_partial_records.json",
        {
            "schema": SCHEMA,
            "status": "PARTIAL_TEACHER_RECORDS",
            "record_count": len(records),
            "records": records,
        },
    )
    return len(records)


def require_profile_provider(module: object) -> object:
    profile = getattr(module, "_profile", None)
    if not callable(profile):
        raise Stage32Error(
            "Stage32 sampler metric packaging requires Phase-B provider._profile"
        )
    return module


def package_sampler_metrics(
    stage29_module: object,
    stage26_module: object,
    decoded: object,
    expert: object,
    profile_provider: object,
    processor: object,
    embodiment: object,
    snapshot: object,
    checkpoint: Path,
    label: str,
) -> object:
    """Call the frozen Stage29 package path with the correct profile role."""

    require_profile_provider(profile_provider)
    return stage29_module._trajectory_metrics(
        decoded,
        expert,
        profile_provider,
        stage26_module,
        processor,
        embodiment,
        snapshot,
        checkpoint,
        label,
    )


def assert_continuation_sampler_seed(seed: int) -> None:
    if int(seed) not in CONTINUATION_SAMPLER_SEEDS:
        raise Stage32Error(
            f"continuation sampler seed {seed} is forbidden; only {CONTINUATION_SAMPLER_SEEDS}"
        )


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def inventory(root: Path):
    if root.is_symlink() or not root.is_dir():
        raise Stage32Error(f"checkpoint root is unsafe: {root}")
    rows = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in dirnames:
            child = base / name
            if child.is_symlink() or not child.is_dir():
                raise Stage32Error(f"unsafe directory entry: {child}")
        for name in filenames:
            child = base / name
            rows.append({
                "path": child.relative_to(root).as_posix(),
                "bytes": child.stat().st_size,
                "sha256": sha256(child),
            })
    return sorted(rows, key=lambda row: row["path"])


def tree_digest(rows, prefix: str) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["path"]):
        digest.update(f"{prefix}/{row['path']}\0{row['bytes']}\0{row['sha256']}\n".encode())
    return digest.hexdigest()


def verify_r1_salvage(args: argparse.Namespace) -> dict:
    """Verify the immutable zero-model salvage layer before any new forward."""

    if sha256(args.stage32_r1_result) != R1_RESULT_SHA:
        raise Stage32Error("R1 salvage result SHA mismatch")
    if sha256(args.stage32_r1_manifest) != R1_MANIFEST_SHA:
        raise Stage32Error("R1 salvage manifest SHA mismatch")
    if sha256(args.stage32_r1_seal) != R1_SEAL_SHA:
        raise Stage32Error("R1 salvage seal SHA mismatch")
    if sha256(args.stage32_partial_trace) != R1_PARTIAL_TRACE_SHA:
        raise Stage32Error("historical Stage32 partial trace SHA mismatch")

    result = read_json(args.stage32_r1_result)
    if (
        result.get("status") != "PASS_R1_ZERO_MODEL_SALVAGE"
        or result.get("pi_gate") != PI_GATE
        or result.get("scientific_contract_changed") is not False
        or result.get("historical_files_modified") is not False
    ):
        raise Stage32Error("R1 salvage status or contract declaration drifted")
    execution = result.get("execution", {})
    forbidden_true = (
        "model_constructed", "model_forward", "checkpoint_loaded", "sampler_calls",
        "training", "backward", "checkpoint_write", "pbs_submitted",
        "validation_e_read", "holdout_f_read", "wbc_executed", "hardware_executed",
    )
    if any(execution.get(key) not in (False, 0) for key in forbidden_true):
        raise Stage32Error("R1 is not zero-model/no-forward evidence")
    source = result.get("source", {})
    if (
        source.get("partial_trace_sha256") != R1_PARTIAL_TRACE_SHA
        or source.get("x0_matches_stage31_noise") is not True
        or source.get("stage31_seed2223_initial_noise_float32_sha256")
        != source.get("x0_float32_sha256")
    ):
        raise Stage32Error("R1 trace/noise identity declaration drifted")
    trace = result.get("trace", {})
    if (
        trace.get("x_shape") != [5, 1, 40, 132]
        or trace.get("v_shape") != [4, 1, 40, 132]
        or trace.get("timestep_ids") != [0, 250, 500, 750]
        or trace.get("all_finite") is not True
        or trace.get("euler_recurrence_exact") is not True
    ):
        raise Stage32Error("R1 trace shape/timestep/recurrence declaration drifted")

    manifest = read_json(args.stage32_r1_manifest)
    if manifest.get("status") != "PASS" or manifest.get("pi_gate") != PI_GATE:
        raise Stage32Error("R1 manifest status or gate drifted")
    if manifest.get("source_repo_head") != R1_SOURCE_HEAD:
        raise Stage32Error("R1 source-head binding drifted")
    manifest_output = manifest.get("salvage_output", {})
    if (
        Path(manifest_output.get("path", "")) != args.stage32_r1_result
        or manifest_output.get("sha256") != R1_RESULT_SHA
    ):
        raise Stage32Error("R1 manifest output binding drifted")

    seal = read_json(args.stage32_r1_seal)
    if (
        seal.get("status") != "PASS"
        or seal.get("pi_gate") != PI_GATE
        or seal.get("scientific_contract_changed") is not False
        or seal.get("historical_files_modified") is not False
        or seal.get("artifact_count") != 6
    ):
        raise Stage32Error("R1 seal status or artifact count drifted")
    entries = seal.get("artifacts", [])
    if len(entries) != 6:
        raise Stage32Error("R1 seal artifact list drifted")
    expected_paths = {
        str(args.stage32_partial_trace): R1_PARTIAL_TRACE_SHA,
        str(args.stage32_r1_result): R1_RESULT_SHA,
        str(args.stage32_r1_manifest): R1_MANIFEST_SHA,
    }
    for entry in entries:
        path = Path(str(entry.get("path", "")))
        if path in (args.stage32_partial_trace, args.stage32_r1_result, args.stage32_r1_manifest):
            if entry.get("sha256") != expected_paths[str(path)]:
                raise Stage32Error(f"R1 sealed hash mismatch: {path}")
        if not path.is_file() or path.is_symlink():
            raise Stage32Error(f"R1 seal references unsafe/missing artifact: {path}")
        if sha256(path) != str(entry.get("sha256")):
            raise Stage32Error(f"R1 artifact changed after sealing: {path}")
    roles = {str(entry.get("role")) for entry in entries}
    if not {"historical_partial_trace", "stage31_result", "stage31_posthoc_result", "stage31_locked_target", "salvage_manifest"}.issubset(roles):
        raise Stage32Error("R1 seal is missing required provenance roles")

    # This is an offline trace check only; it cannot construct a model.
    try:
        import numpy as np

        with np.load(args.stage32_partial_trace, allow_pickle=False) as payload:
            x = np.asarray(payload["x_trajectory_float32"])
            v = np.asarray(payload["v_trajectory_float32"])
            times = np.asarray(payload["timestep_ids"])
        if x.shape != (5, 1, 40, 132) or v.shape != (4, 1, 40, 132) or times.tolist() != [0, 250, 500, 750]:
            raise Stage32Error("R1 partial trace arrays drifted")
        if not np.isfinite(x).all() or not np.isfinite(v).all():
            raise Stage32Error("R1 partial trace is non-finite")
        # The historical trace stores BF16 sampler states/velocities at a
        # float32 serialization boundary; the recurrence therefore includes
        # the bounded BF16 rounding of the update.
        if not np.allclose(x[1:], x[:-1] + np.float32(0.25) * v, rtol=0.0, atol=1e-2):
            raise Stage32Error("R1 partial trace Euler recurrence is not exact")
    except ImportError as exc:
        raise Stage32Error("R1 trace validation requires the pinned numpy runtime") from exc
    return {
        "status": "PASS_R1_ZERO_MODEL_SALVAGE_VERIFIED",
        "result_sha256": R1_RESULT_SHA,
        "manifest_sha256": R1_MANIFEST_SHA,
        "seal_sha256": R1_SEAL_SHA,
        "partial_trace_sha256": R1_PARTIAL_TRACE_SHA,
    }


def verify_p0(args: argparse.Namespace) -> dict:
    if args.authorization_token != PI_GATE or args.pbs_project != PROJECT:
        raise Stage32Error("PI gate or PBS project mismatch")
    if args.runner_sha256 and sha256(Path(__file__).resolve()) != args.runner_sha256:
        raise Stage32Error("runner SHA mismatch")
    if args.pbs_sha256 and sha256(args.pbs_template) != args.pbs_sha256:
        raise Stage32Error("PBS template SHA mismatch")
    if git(args.project_repo, "rev-parse", "HEAD") != args.expected_project_sha:
        raise Stage32Error("authorized project SHA mismatch")
    if git(args.project_repo, "status", "--short"):
        raise Stage32Error("authorized project worktree is dirty")
    ancestry = subprocess.run(
        ["git", "-C", str(args.project_repo), "merge-base", "--is-ancestor", HISTORICAL_TRAINING_SHA, args.expected_project_sha],
        check=False,
    )
    if ancestry.returncode != 0:
        raise Stage32Error("Stage32 recovery HEAD does not descend from historical training SHA")
    from scripts import stage28_h2_fresh_base_2000_fixed_warmup10 as stage28

    base_identity = stage28._verify_base_checkpoint(args.base_model_path)
    if base_identity.get("tree_sha256") != BASE_TREE:
        raise Stage32Error("BASE canonical tree SHA mismatch")
    dataset_identity = stage28._verify_tiny_dataset(args.dataset_dir)
    if dataset_identity.get("tree_sha256") != TINY1_TREE:
        raise Stage32Error("TINY1 canonical tree SHA mismatch")
    cosmos_identity = stage28._verify_cosmos_cache(args.cosmos_cache_root)
    if cosmos_identity.get("revision") != COSMOS_REVISION or cosmos_identity.get("tree_sha256") != COSMOS_TREE:
        raise Stage32Error("Cosmos revision/tree SHA mismatch")
    groot_identity = stage28._verify_groot_source(args.groot_source)
    if groot_identity.get("observed_head") != GROOT_SHA:
        raise Stage32Error("GR00T source SHA mismatch")
    r1_identity = verify_r1_salvage(args)
    rows = inventory(args.h2_checkpoint_1000)
    if len(rows) != 18 or sum(row["bytes"] for row in rows) != H2_1000_BYTES:
        raise Stage32Error("checkpoint-1000 file count/bytes drifted")
    if tree_digest(rows, "stage28/checkpoint-1000") != H2_1000_TREE:
        raise Stage32Error("checkpoint-1000 full tree SHA mismatch")
    by_name = {row["path"]: row for row in rows}
    if by_name["model.safetensors.index.json"]["sha256"] != H2_1000_INDEX:
        raise Stage32Error("checkpoint-1000 index SHA mismatch")
    trainer = read_json(args.h2_checkpoint_1000 / "trainer_state.json")
    if trainer.get("global_step") != 1000 or trainer.get("max_steps") != 2000:
        raise Stage32Error("checkpoint-1000 trainer state mismatch")
    seal = read_json(args.h2_checkpoint_2000_seal)
    if sha256(args.h2_checkpoint_2000_seal) != H2_2000_SEAL or seal.get("tree_sha256") != H2_2000_TREE:
        raise Stage32Error("checkpoint-2000 seal mismatch")
    run = read_json(args.stage28_execution_manifest)
    if run.get("status") != "PASS" or run.get("recipe") != "H2_FRESH_BASE_2000_FIXED_WARMUP10":
        raise Stage32Error("historical H2 run is not a PASS")
    for key, step, path, tree in (
        ("checkpoint-1000", 1000, args.h2_checkpoint_1000, H2_1000_TREE),
        ("checkpoint-2000", 2000, args.h2_checkpoint_2000, H2_2000_TREE),
    ):
        row = run.get(key, {})
        if row.get("global_step") != step or row.get("tree_sha256") != tree:
            raise Stage32Error(f"historical {key} identity mismatch")
        if Path(row.get("path", "")).resolve() != path.resolve():
            raise Stage32Error(f"historical {key} path mismatch")
    config = run.get("config", {})
    frozen = {
        "learning_rate": 1e-4, "weight_decay": 1e-5, "max_steps": 2000,
        "warmup_steps": 10, "warmup_ratio": 0.0, "seed": 2222,
        "bf16": True, "tune_projector": True, "tune_diffusion_model": True,
        "tune_visual": False, "tune_llm": False, "tune_vlln": False,
        "save_only_model": False,
    }
    if any(config.get(key) != value for key, value in frozen.items()):
        raise Stage32Error("historical H2 training configuration drifted")
    if sha256(args.stage28_result) != STAGE28_RESULT:
        raise Stage32Error("Stage28 result SHA mismatch")
    if sha256(args.stage31_result) != STAGE31_RESULT:
        raise Stage32Error("Stage31 result SHA mismatch")
    if sha256(args.stage31_posthoc) != STAGE31_POSTHOC:
        raise Stage32Error("Stage31 posthoc SHA mismatch")
    stage28_result = read_json(args.stage28_result)
    paired = stage28_result.get("paired_aggregates", {}).get("paired", [])
    if len(paired) != 9 or len({str(row.get("probe_id")) for row in paired}) != 9:
        raise Stage32Error("Stage28 fixed-probe count drifted")
    stage31 = read_json(args.stage31_result)
    records = stage31.get("records", [])
    if len(records) != 6 or not all(r.get("stage29_reproduction", {}).get("status") == "MATCH" for r in records):
        raise Stage32Error("Stage31 six-call historical evidence is not exact")
    if sha256(args.modality_config) != MODALITY_SHA:
        raise Stage32Error("modality identity mismatch")
    if sha256(args.dataset_dir / "meta" / "stats.json") != STATS_SHA or sha256(args.dataset_dir / "meta" / "relative_stats.json") != RELATIVE_STATS_SHA:
        raise Stage32Error("dataset statistics identity mismatch")
    return {
        "schema": SCHEMA,
        "status": "PASS_PREFLIGHT_NO_MODEL",
        "pi_gate": PI_GATE,
        "project_sha": args.expected_project_sha,
        "historical_training_sha": HISTORICAL_TRAINING_SHA,
        "base_identity": base_identity,
        "dataset_identity": dataset_identity,
        "cosmos_identity": cosmos_identity,
        "groot_identity": groot_identity,
        "r1_salvage": r1_identity,
        "checkpoint_1000": {"path": str(args.h2_checkpoint_1000), "tree_sha256": H2_1000_TREE, "file_count": len(rows), "total_bytes": sum(row["bytes"] for row in rows), "trainer_state": trainer},
        "checkpoint_2000": {"path": str(args.h2_checkpoint_2000), "tree_sha256": H2_2000_TREE, "seal_sha256": H2_2000_SEAL},
        "historical_run_manifest_sha256": sha256(args.stage28_execution_manifest),
        "historical_stage28_result_sha256": STAGE28_RESULT,
        "historical_stage31_result_sha256": STAGE31_RESULT,
        "historical_stage31_posthoc_sha256": STAGE31_POSTHOC,
        "identities": {"base_tree_sha256": BASE_TREE, "tiny1_tree_sha256": TINY1_TREE, "cosmos_revision": COSMOS_REVISION, "cosmos_tree_sha256": COSMOS_TREE, "modality_sha256": MODALITY_SHA, "stats_sha256": STATS_SHA, "relative_stats_sha256": RELATIVE_STATS_SHA, "groot_sha": GROOT_SHA, "stage32_r1_result_sha256": R1_RESULT_SHA, "stage32_r1_manifest_sha256": R1_MANIFEST_SHA, "stage32_r1_seal_sha256": R1_SEAL_SHA},
        "execution": {"model_constructed": False, "model_forward": False, "teacher_forced_probes": 0, "sampler_calls": 0, "training": False, "backward": False, "checkpoint_write": False, "validation_e_read": False, "holdout_f_read": False},
        "scientific_contract_changed": False,
    }


def prepare_inputs(args, stage26):
    from gr00t.data.types import MessageType, VLAStepData
    snapshot = stage26._load_snapshot(stage26._verify_snapshot_identity(args.snapshot_root)[1])
    processor, embodiment, modality = stage26._build_runtime_processor(
        groot_source=args.groot_source, base_checkpoint=args.base_model_path,
        dataset_dir=args.dataset_dir, modality_path=args.modality_config,
        cosmos_snapshot=args.cosmos_snapshot,
    )
    prepared = stage26._prepare_official_sample(
        processor=processor, embodiment=embodiment, modality_configs=modality, dataset_dir=args.dataset_dir,
    )
    step = prepared["step_data"]
    inf_step = VLAStepData(images=step.images, states=step.states, actions={}, text=step.text, embodiment=embodiment)
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": inf_step}])
    batch = processor.collator([processed])
    if "action" in batch.get("inputs", {}) or "action_mask" in batch.get("inputs", {}):
        raise Stage32Error("ground-truth action leaked into inference conditioning")
    return processor, embodiment, prepared, snapshot, batch


def salvaged_sampler_record(args, stage31, target) -> dict:
    """Package the already sealed seed2223 trace without a model call."""

    import numpy as np

    salvage = read_json(args.stage32_r1_result)
    source = salvage["source"]
    with np.load(args.stage32_partial_trace, allow_pickle=False) as payload:
        x = np.asarray(payload["x_trajectory_float32"], dtype=np.float32)
        v = np.asarray(payload["v_trajectory_float32"], dtype=np.float32)
        timestep_ids = np.asarray(payload["timestep_ids"], dtype=np.int64)
    metrics = stage31.trace_metrics(x, v, target)
    if metrics != salvage["stage31_compatible_trace_metrics"]:
        raise Stage32Error("R1 sealed Stage31-compatible trace metrics changed")
    return {
        "seed": 2223,
        "checkpoint": "H2@1000",
        "provenance": "601973_SALVAGED",
        "historical_partial_job": "601973.hopper-m-02",
        "new_sampler_call": False,
        "initial_noise": {
            "shape": [1, 40, 132],
            "float32_sha256": source["stage31_seed2223_initial_noise_float32_sha256"],
            "raw_sha256": source["stage31_seed2223_initial_noise_raw_sha256"],
            "finite": True,
        },
        "initial_noise_matches_stage31": True,
        "trace": {
            "state_count": 5,
            "velocity_count": 4,
            "timestep_ids": timestep_ids.tolist(),
            "x_shape": list(x.shape),
            "v_shape": list(v.shape),
            "euler_recurrence": True,
        },
        "trace_artifact": {
            "path": str(args.stage32_partial_trace),
            "bytes": args.stage32_partial_trace.stat().st_size,
            "sha256": R1_PARTIAL_TRACE_SHA,
        },
        "trace_metrics": metrics,
        "all_finite": bool(np.isfinite(x).all() and np.isfinite(v).all()),
        "scientific_execution_attribution": "historical_partial_trace_salvage_only",
    }


def persist_sampler_records(output_root: Path, records: list[Mapping[str, object]]) -> None:
    write_json(
        output_root / "stage32_sampler_partial_records.json",
        {
            "schema": SCHEMA,
            "status": "PARTIAL_SAMPLER_RECORDS",
            "record_count": len(records),
            "records": records,
        },
    )


def write_result_manifests(args, result: Mapping[str, object]) -> None:
    """Write machine-readable manifests before the post-job seal step."""

    output_root = args.output_root
    raw_entries = []
    for path in sorted((output_root / "teacher_records").glob("record_*.json")):
        raw_entries.append({"path": str(path), "role": "teacher_record", "bytes": path.stat().st_size, "sha256": sha256(path)})
    for row in result.get("sampler_records", []):
        artifact = row.get("trace_artifact", {})
        path = Path(str(artifact.get("path", "")))
        if path.is_file() and not path.is_symlink():
            raw_entries.append({"path": str(path), "role": f"sampler_trace_{row.get('provenance')}", "bytes": path.stat().st_size, "sha256": sha256(path)})
    raw_manifest = {
        "schema": "STAGE32_H2_1000_RAW_HASH_MANIFEST_V1",
        "status": "PASS",
        "pi_gate": PI_GATE,
        "pbs_job_id": args.pbs_job_id,
        "teacher_record_count": len(result.get("teacher_forced_records", [])),
        "sampler_record_count": len(result.get("sampler_records", [])),
        "entries": raw_entries,
        "scientific_contract_changed": False,
    }
    raw_path = output_root / "stage32_raw_hash_manifest.json"
    write_json(raw_path, raw_manifest)
    result_path = output_root / "stage32_h2_1000_result.json"
    result_manifest = {
        "schema": "STAGE32_H2_1000_RESULT_MANIFEST_V1",
        "status": "PASS_UNSEALED",
        "pi_gate": PI_GATE,
        "pbs_job_id": args.pbs_job_id,
        "project_sha": git(args.project_repo, "rev-parse", "HEAD"),
        "result": {"path": str(result_path), "bytes": result_path.stat().st_size, "sha256": sha256(result_path)},
        "raw_hash_manifest": {"path": str(raw_path), "bytes": raw_path.stat().st_size, "sha256": sha256(raw_path)},
        "preflight": {"path": str(output_root / "stage32_preflight.json"), "sha256": sha256(output_root / "stage32_preflight.json")},
        "execution_state": {"path": str(output_root / "stage32_execution_state.json"), "sha256": sha256(output_root / "stage32_execution_state.json")},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "pbs_wrapper": {"path": str(args.pbs_template), "sha256": sha256(args.pbs_template)},
        "scientific_contract_changed": False,
    }
    write_json(output_root / "stage32_result_manifest.json", result_manifest)


def continuation(args, p0):
    import numpy as np
    import torch
    from scripts import stage26_h1_c1_fixed_noise_local_flow as stage26
    from scripts import stage29_c2_free_running_transfer as stage29
    from scripts import stage31_on_trajectory_flow_field_audit as stage31
    from scripts import stage23_phase_b_attribution_audit as stage23_profile
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "H200" not in torch.cuda.get_device_name(0):
        raise Stage32Error("Stage32 requires exactly one visible H200")
    stage23_flow = stage26._load_stage23_protocol()
    stage23_flow.assert_protocol_static(stage23_flow.protocol_plan())
    require_profile_provider(stage23_profile)
    contract = stage26._static_probe_contract()
    probes = contract["probes"]
    if len(probes) != 9 or {int(p["noise_seed"]) for p in probes} != set(SEEDS) or {int(p["t_discretized"]) for p in probes} != {67, 369, 783}:
        raise Stage32Error("fixed Stage32 probe contract drifted")
    processor, embodiment, prepared, snapshot, batch = prepare_inputs(args, stage26)
    fixed = tuple(stage26._canonical_probe_inputs(stage23_flow, prepared["expert_action_cpu"], p) for p in probes)
    state_path = args.output_root / "stage32_execution_state.json"
    if state_path.exists() or load_teacher_records(args.output_root):
        raise Stage32Error("continuation output root is not fresh; replacement/resume is forbidden")
    state = {
        "schema": SCHEMA,
        "status": "READY_BEFORE_FIRST_SCIENTIFIC_CALL",
        "scientific_execution_begun": False,
        "model_constructed": False,
        "teacher_forced_probes": 0,
        "sampler_calls": 0,
        "historical_salvaged_sampler_calls": 1,
        "new_sampler_calls_allowed": 2,
        "pbs_job_id": args.pbs_job_id,
    }
    write_json(state_path, state)

    def record_teacher(row):
        count = persist_teacher_record(args.output_root, row)
        update_execution_state(
            state_path,
            {
                "status": "SCIENTIFIC_EXECUTION_IN_PROGRESS",
                "scientific_execution_begun": True,
                "model_constructed": True,
                "teacher_forced_probes": count,
                "teacher_forced_provenance": "RECOVERY_REPLAY",
            },
        )

    def before_teacher_forward(label, probe):
        update_execution_state(
            state_path,
            {
                "status": "SCIENTIFIC_EXECUTION_BEGUN",
                "scientific_execution_begun": True,
                "model_constructed": True,
                "current_checkpoint": label,
                "current_probe_id": probe["probe_id"],
                "teacher_forced_provenance": "RECOVERY_REPLAY",
            },
        )
    teacher = stage26._run_one_checkpoint(
        checkpoint_label="H2@1000", checkpoint=args.h2_checkpoint_1000, groot_source=args.groot_source,
        prepared=prepared, processor=processor, embodiment=embodiment, stage23=stage23_flow,
        probes=probes, fixed_probes=fixed, device=args.device, snapshot=snapshot,
        on_before_direct_probe_forward=before_teacher_forward,
        on_record=record_teacher,
    )
    persisted_teacher = load_teacher_records(args.output_root)
    if len(teacher) != 9 or len(persisted_teacher) != 9:
        raise Stage32Error("teacher-forced count drifted")
    if [_teacher_record_key(row) for row in teacher] != [_teacher_record_key(row) for row in persisted_teacher]:
        raise Stage32Error("persisted teacher record order/key set drifted")
    teacher = persisted_teacher
    historical = read_json(args.stage31_result)
    expected_noise = {int(r["seed"]): r["initial_noise"] for r in historical["records"] if r["checkpoint"] == "BASE"}
    if set(expected_noise) != set(SEEDS):
        raise Stage32Error("Stage31 initial-noise comparator set drifted")
    conditioning = stage29._conditioning_manifest(batch)
    expert = stage26._decode_physical_action(processor, embodiment, prepared["expert_action_cpu"].numpy(), snapshot)
    target = prepared["expert_action_cpu"].numpy().astype(np.float32)
    sampler = [salvaged_sampler_record(args, stage31, target)]
    persist_sampler_records(args.output_root, sampler)
    model = stage29._load_model(args.h2_checkpoint_1000, args.device, torch)
    update_execution_state(state_path, {"model_constructed": True, "status": "READY_FOR_CONTINUATION_SAMPLER", "teacher_forced_probes": 9, "sampler_calls": 0})
    new_sampler_calls = 0
    for seed in CONTINUATION_SAMPLER_SEEDS:
        assert_continuation_sampler_seed(seed)
        rng = stage29._reset_rng(seed, torch, np)
        observer = stage31.StockSamplerTrace(model.action_head)
        started = time.perf_counter()
        with stage29._capture_stock_initial_noise(torch) as calls:
            with observer:
                update_execution_state(
                    state_path,
                    {
                        "status": "SCIENTIFIC_EXECUTION_BEGUN",
                        "scientific_execution_begun": True,
                        "model_constructed": True,
                        "current_seed": seed,
                        "current_sampler_provenance": "CONTINUATION",
                        "teacher_forced_probes": 9,
                        "sampler_calls": new_sampler_calls,
                    },
                )
                with torch.inference_mode():
                    prediction = model.get_action(**copy.deepcopy(batch))
        torch.cuda.synchronize()
        new_sampler_calls += 1
        update_execution_state(
            state_path,
            {
                "status": "SCIENTIFIC_EXECUTION_IN_PROGRESS",
                "scientific_execution_begun": True,
                "model_constructed": True,
                "teacher_forced_probes": 9,
                "sampler_calls": new_sampler_calls,
                "current_seed": seed,
                "current_sampler_provenance": "CONTINUATION",
            },
        )
        noise = stage29._validate_noise_capture(calls, model)
        if noise["raw_sha256"] != expected_noise[seed]["raw_sha256"] or noise["float32_sha256"] != expected_noise[seed]["float32_sha256"]:
            raise Stage32Error(f"initial noise mismatch for seed {seed}")
        trace = stage31._finalize_trace(observer, prediction["action_pred"], torch)
        raw = prediction["action_pred"].detach().float().contiguous().cpu().numpy().astype(np.float32, copy=False)
        decoded = stage26._decode_physical_action(processor, embodiment, raw, snapshot)
        path = args.output_root / "traces" / f"seed_{seed}_h2_1000_stock_sampler_trace.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, x_trajectory_float32=trace["x_trajectory_float32"], v_trajectory_float32=trace["v_trajectory_float32"], timestep_ids=np.asarray(trace["timestep_ids"], dtype=np.int64))
        trace_sha = sha256(path)
        update_execution_state(
            state_path,
            {
                "status": "SCIENTIFIC_EXECUTION_IN_PROGRESS",
                "scientific_execution_begun": True,
                "model_constructed": True,
                "teacher_forced_probes": 9,
                "sampler_calls": new_sampler_calls,
                "last_trace_artifact": str(path),
                "last_trace_artifact_sha256": trace_sha,
                "current_seed": seed,
                "current_sampler_provenance": "CONTINUATION",
            },
        )
        physical_metrics = package_sampler_metrics(
            stage29,
            stage26,
            decoded,
            expert,
            stage23_profile,
            processor,
            embodiment,
            snapshot,
            args.h2_checkpoint_1000,
            "H2@1000",
        )
        sampler.append({
            "seed": seed, "checkpoint": "H2@1000", "provenance": "CONTINUATION", "new_sampler_call": True, "rng_before_call": rng,
            "initial_noise": noise, "initial_noise_matches_stage31": True,
            "conditioning_sha256": conditioning["sha256"],
            "trace": {"tensor_records": trace["tensor_records"], "timestep_ids": trace["timestep_ids"], "euler_recurrence": trace["euler_recurrence"]},
            "trace_artifact": {"path": str(path), "bytes": path.stat().st_size, "sha256": trace_sha},
            "trace_metrics": stage31.trace_metrics(trace["x_trajectory_float32"], trace["v_trajectory_float32"], target),
            "physical_metrics": physical_metrics,
            "raw_universal_action_sha256": hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
            "decoded_physical_action_sha256": hashlib.sha256(np.ascontiguousarray(decoded).tobytes()).hexdigest(),
            "latency_s": time.perf_counter() - started,
            "all_finite": bool(np.isfinite(raw).all() and np.isfinite(decoded).all()),
        })
        persist_sampler_records(args.output_root, sampler)
        update_execution_state(
            state_path,
            {
                "status": "SCIENTIFIC_EXECUTION_IN_PROGRESS",
                "scientific_execution_begun": True,
                "teacher_forced_probes": 9,
                "sampler_calls": new_sampler_calls,
                "total_sampler_records": len(sampler),
                "last_completed_seed": seed,
            },
        )
    if len(sampler) != 3:
        raise Stage32Error("sampler count drifted")
    historical28 = read_json(args.stage28_result)
    h2000_flow = {row["probe_id"]: float(row["h2_mse"]) for row in historical28["paired_aggregates"]["paired"]}
    h1000_flow = {row["probe_id"]: float(row["flow_metrics"]["active_masked_mse"]) for row in teacher}
    h2000_final = {int(row["seed"]): float(row["metrics"]["final_x4"]["scopes"]["active10"]["state_error_mse"]) for row in historical["records"] if row["checkpoint"] == "H2@2000"}
    h1000_final = {int(row["seed"]): float(row["trace_metrics"]["final_x4"]["scopes"]["active10"]["state_error_mse"]) for row in sampler}
    if set(h1000_flow) != set(h2000_flow) or set(h1000_final) != set(SEEDS) or set(h2000_final) != set(SEEDS):
        raise Stage32Error("Stage32 promotion comparison set drifted")
    flow = [{"probe_id": k, "h2_1000": h1000_flow[k], "h2_2000": h2000_flow[k], "strictly_lower": h2000_flow[k] < h1000_flow[k]} for k in sorted(h1000_flow)]
    free = [{"seed": s, "h2_1000_final_active10_mse": h1000_final[s], "h2_2000_final_active10_mse": h2000_final[s], "strictly_lower": h2000_final[s] < h1000_final[s]} for s in SEEDS]
    all_finite = all(bool(row.get("all_finite")) for row in sampler) and all(all(bool(value) for value in row.get("finite_flags", {}).values()) for row in teacher)
    promotion = {
        "identity_and_contract_pass": True,
        "all_finite": all_finite,
        "h2_2000_teacher_mse_strictly_lower_9_of_9": len(flow) == 9 and all(x["strictly_lower"] for x in flow),
        "h2_2000_free_running_final_active10_mse_strictly_lower_3_of_3": len(free) == 3 and all(x["strictly_lower"] for x in free),
        "teacher_forced_9_of_9": len(flow) == 9 and all(x["strictly_lower"] for x in flow),
        "sampler_3_of_3": len(free) == 3 and all(x["strictly_lower"] for x in free),
        "identity_checks": True,
        "scientific_contract_changed": False,
    }
    promotion["pass"] = all(promotion.values())
    result = {
        "schema": SCHEMA,
        "status": "PASS_EXECUTION_COMPLETE",
        "pi_gate": PI_GATE,
        "preflight": p0,
        "teacher_forced_records": teacher,
        "sampler_records": sampler,
        "promotion": {"flow": flow, "free_running": free, **promotion},
        "provenance": {
            "teacher_forced": "RECOVERY_REPLAY",
            "sampler_seed2223": "601973_SALVAGED",
            "sampler_seed2224": "CONTINUATION",
            "sampler_seed2225": "CONTINUATION",
            "h2_2000_comparator": "IMMUTABLE_STAGE31_EVIDENCE",
        },
        "execution": {
            "model_constructed": True,
            "model_forward": True,
            "checkpoint_loaded": True,
            "teacher_forced_probes": 9,
            "teacher_forced_provenance": "RECOVERY_REPLAY",
            "sampler_calls": new_sampler_calls,
            "new_sampler_calls": new_sampler_calls,
            "historical_salvaged_sampler_calls": 1,
            "total_sampler_records": len(sampler),
            "sampler_seed2223_new_calls": 0,
            "sampler_seed2224_new_calls": 1,
            "sampler_seed2225_new_calls": 1,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "scheduler_step": False,
            "resume": False,
            "checkpoint_write": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "wbc_executed": False,
            "hardware_executed": False,
        },
        "pbs_submitted": True,
        "pbs_job_id": args.pbs_job_id,
        "scientific_contract_changed": False,
        "next_gate": "STAGE33_CONDITIONAL_AUTONOMOUS" if promotion["pass"] else NEXT_GATE,
    }
    write_json(args.output_root / "stage32_h2_1000_result.json", result)
    update_execution_state(
        state_path,
        {
            "status": "PASS_EXECUTION_COMPLETE",
            "scientific_execution_begun": True,
            "model_constructed": True,
            "teacher_forced_probes": 9,
            "sampler_calls": new_sampler_calls,
            "new_sampler_calls": new_sampler_calls,
            "total_sampler_records": 3,
            "next_gate": result["next_gate"],
        },
    )
    write_result_manifests(args, result)
    return result


def _artifact(path: Path, role: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise Stage32Error(f"required seal artifact is missing/unsafe: {path}")
    return {"path": str(path), "role": role, "bytes": path.stat().st_size, "sha256": sha256(path)}


def seal_result(args) -> dict:
    """Seal the completed continuation after the PBS log is closed."""

    root = args.output_root
    result_path = root / "stage32_h2_1000_result.json"
    result = read_json(result_path)
    if result.get("status") != "PASS_EXECUTION_COMPLETE" or result.get("pi_gate") != PI_GATE:
        raise Stage32Error("cannot seal a non-complete Stage32 result")
    if result.get("scientific_contract_changed") is not False:
        raise Stage32Error("cannot seal a changed scientific contract")
    if git(args.project_repo, "rev-parse", "HEAD") != result.get("preflight", {}).get("project_sha"):
        raise Stage32Error("current project HEAD differs from completed continuation")
    if args.pbs_log is None:
        raise Stage32Error("post-job PBS log is required for the Stage32 seal")
    log_path = args.pbs_log
    entries = [
        _artifact(result_path, "result_json"),
        _artifact(root / "stage32_result_manifest.json", "result_manifest"),
        _artifact(root / "stage32_raw_hash_manifest.json", "raw_hash_manifest"),
        _artifact(root / "stage32_preflight.json", "preflight"),
        _artifact(root / "stage32_execution_state.json", "execution_state"),
        _artifact(root / "stage32_teacher_forced_partial_records.json", "teacher_partial_records"),
        _artifact(root / "stage32_sampler_partial_records.json", "sampler_partial_records"),
        _artifact(Path(__file__).resolve(), "evaluation_runner"),
        _artifact(args.pbs_template, "pbs_wrapper"),
        _artifact(log_path, "pbs_log"),
        _artifact(args.stage32_r1_result, "r1_salvage_result"),
        _artifact(args.stage32_r1_manifest, "r1_salvage_manifest"),
        _artifact(args.stage32_r1_seal, "r1_salvage_seal"),
    ]
    for path in sorted((root / "teacher_records").glob("record_*.json")):
        entries.append(_artifact(path, "teacher_record"))
    for row in result.get("sampler_records", []):
        artifact_path = Path(str(row.get("trace_artifact", {}).get("path", "")))
        role = "sampler_trace_" + str(row.get("provenance", "UNKNOWN"))
        if not any(item["path"] == str(artifact_path) for item in entries):
            entries.append(_artifact(artifact_path, role))
    if len({item["path"] for item in entries}) != len(entries):
        raise Stage32Error("duplicate artifact path in Stage32 seal")
    seal = {
        "schema": "STAGE32_H2_1000_RESULT_SEAL_V1",
        "status": "PASS",
        "pi_gate": PI_GATE,
        "pbs_job_id": args.pbs_job_id,
        "source_repo_head": git(args.project_repo, "rev-parse", "HEAD"),
        "artifact_count": len(entries),
        "artifacts": entries,
        "execution": result.get("execution", {}),
        "promotion": result.get("promotion", {}),
        "pbs_submitted": True,
        "gpu_scientific_execution": True,
        "scientific_contract_changed": False,
        "next_gate": result.get("next_gate", NEXT_GATE),
    }
    seal_path = root / "stage32_result_seal.json"
    write_json(seal_path, seal)
    return {"status": "PASS", "seal_path": str(seal_path), "seal_sha256": sha256(seal_path), "artifact_count": len(entries), "next_gate": seal["next_gate"]}


def run_packaging_self_test() -> dict[str, object]:
    """Exercise the R2 packaging and monotonic-counter path without a model."""

    import tempfile

    class FakeProfileProvider:
        def __init__(self):
            self.calls = 0

        def _profile(self, rows, period):
            self.calls += 1
            return {"rows": len(rows), "period": period}

    class FakeStage29:
        def _trajectory_metrics(self, decoded, expert, provider, stage26, processor, embodiment, snapshot, checkpoint, label):
            if not callable(getattr(provider, "_profile", None)):
                raise AssertionError("profile provider was not passed")
            return {"predicted_profile": provider._profile(decoded, 0.05), "expert_profile": provider._profile(expert, 0.05)}

    provider = FakeProfileProvider()
    packaged = package_sampler_metrics(FakeStage29(), object(), [[]], [[]], provider, object(), object(), object(), Path("/tmp/checkpoint"), "H2@1000")
    if provider.calls != 2 or packaged["predicted_profile"]["rows"] != 1:
        raise Stage32Error("synthetic profile packaging self-test failed")
    with tempfile.TemporaryDirectory(prefix="stage32_r2_self_test_") as temporary:
        root = Path(temporary)
        count = persist_teacher_record(root, {"checkpoint": "H2@1000", "probe_id": "p1", "finite_flags": {"all": True}})
        if count != 1 or len(load_teacher_records(root)) != 1:
            raise Stage32Error("synthetic teacher persistence self-test failed")
        state_path = root / "state.json"
        write_json(state_path, {"teacher_forced_probes": 0, "sampler_calls": 0, "scientific_execution_begun": False})
        update_execution_state(state_path, {"teacher_forced_probes": 4, "sampler_calls": 1})
        update_execution_state(state_path, {"teacher_forced_probes": 2, "sampler_calls": 0})
        state = read_json(state_path)
        if state["teacher_forced_probes"] != 4 or state["sampler_calls"] != 1:
            raise Stage32Error("synthetic monotonic-counter self-test failed")
    try:
        assert_continuation_sampler_seed(2223)
    except Stage32Error:
        pass
    else:
        raise Stage32Error("seed2223 continuation rejection self-test failed")
    for seed in CONTINUATION_SAMPLER_SEEDS:
        assert_continuation_sampler_seed(seed)
    return {"status": "PASS_PACKAGING_SELF_TEST", "profile_calls": provider.calls, "teacher_records": 1, "counter_monotonic": True, "seed2223_rejected": True, "continuation_seeds": list(CONTINUATION_SAMPLER_SEEDS)}


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("preflight", "continuation", "seal", "packaging-self-test"), required=True)
    for name in ("project_repo", "base_model_path", "dataset_dir", "groot_source", "cosmos_snapshot", "cosmos_cache_root", "modality_config", "snapshot_root", "h2_checkpoint_1000", "h2_checkpoint_2000", "h2_checkpoint_2000_seal", "stage28_execution_manifest", "stage28_result", "stage31_root", "stage31_result", "stage31_posthoc", "stage32_partial_trace", "stage32_r1_result", "stage32_r1_manifest", "stage32_r1_seal", "output_root", "pbs_template", "pbs_log"):
        p.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=False)
    p.add_argument("--expected-project-sha", required=False)
    p.add_argument("--authorization-token", required=False)
    p.add_argument("--pbs-project", default=PROJECT)
    p.add_argument("--pbs-job-id", default="PREFLIGHT_NO_PBS")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--runner-sha256")
    p.add_argument("--pbs-sha256")
    return p


REQUIRED_ARGUMENTS = (
    "project_repo", "base_model_path", "dataset_dir", "groot_source", "cosmos_snapshot", "cosmos_cache_root",
    "modality_config", "snapshot_root", "h2_checkpoint_1000", "h2_checkpoint_2000", "h2_checkpoint_2000_seal",
    "stage28_execution_manifest", "stage28_result", "stage31_root", "stage31_result", "stage31_posthoc",
    "stage32_partial_trace", "stage32_r1_result", "stage32_r1_manifest", "stage32_r1_seal", "output_root", "pbs_template",
)


def require_arguments(args) -> None:
    missing = [name for name in REQUIRED_ARGUMENTS if getattr(args, name) is None]
    if args.expected_project_sha is None:
        missing.append("expected_project_sha")
    if args.authorization_token is None:
        missing.append("authorization_token")
    if missing:
        raise Stage32Error("missing required arguments: " + ", ".join(missing))


def main():
    args = parser().parse_args()
    try:
        if args.mode == "packaging-self-test":
            print(json.dumps(run_packaging_self_test(), indent=2, sort_keys=True))
            return 0
        require_arguments(args)
        if args.mode == "seal":
            print(json.dumps(seal_result(args), indent=2, sort_keys=True))
            return 0
        args.output_root.mkdir(parents=True, exist_ok=True)
        p0 = verify_p0(args)
        write_json(args.output_root / "stage32_preflight.json", p0)
        if args.mode == "preflight":
            print(json.dumps(p0, indent=2, sort_keys=True))
        else:
            print(json.dumps(continuation(args, p0), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {"schema": SCHEMA, "status": "STOP_OR_BLOCKED", "error_type": type(exc).__name__, "error_summary": str(exc).splitlines()[0][:1000], "next_gate": NEXT_GATE, "scientific_contract_changed": False}
        if getattr(args, "output_root", None) is not None:
            args.output_root.mkdir(parents=True, exist_ok=True)
            write_json(args.output_root / "stage32_failure.json", failure)
        print(json.dumps(failure, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
