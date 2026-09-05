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

SCHEMA = "STAGE32_H2_1000_EXPOSURE_INTERPOLATION_V1"
PI_GATE = "PI_GATE_STATE_0013"
NEXT_GATE = "GPT_PI_REVIEW_AFTER_STAGE32_EXPOSURE_CURVE"
PROJECT = "CFP03-CF-004"
PROJECT_SHA = "b0b72073403e0bdb3025d2bb76a9e035cc4da933"
GROOT_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
H2_1000_TREE = "29f3e8f46cf41441696bfc4a5e7601378648c26fa675a9685c003080da0c4114"
H2_1000_INDEX = "ff5a9ac57951658b775ef7e958a3469900f767cec60cbf487b3c8d0a3d787bc7"
H2_1000_BYTES = 20889508912
H2_2000_TREE = "3106f9b5993b390438b62594d044781d11b393452a49f80b5c8e9e910606d0ba"
H2_2000_SEAL = "1349896f7b31a930764dbd9d849c5368e48372d0dc6273449f42f19f82a69a82"
STAGE28_RESULT = "aa0ab03b851ad1ad6581a71168d92d682deee546db1bb79b76acb2c1acfe3f26"
STAGE31_RESULT = "c1d9cb9312cada95b35515fb44f87520a2133c7948785d19248613d6c3baf982"
STAGE31_POSTHOC = "49615ebc5129bc45c66c09206320f0e65978a4d6a25eda6e20e363bcdcaecd14"
TINY1_TREE = "541bf4a1b9ea459637525ef11fceea5c7f192c0902f6013f0331a2a668038157"
MODALITY_SHA = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
STATS_SHA = "fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23"
RELATIVE_STATS_SHA = "8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1"
SEEDS = (2223, 2224, 2225)


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
    stage28 = read_json(args.stage28_result)
    if len(stage28.get("paired_aggregates", {}).get("paired", [])) != 9:
        raise Stage32Error("Stage28 fixed-probe count drifted")
    stage31 = read_json(args.stage31_result)
    records = stage31.get("records", [])
    if len(records) != 6 or not all(r.get("stage29_reproduction", {}).get("status") == "MATCH" for r in records):
        raise Stage32Error("Stage31 six-call historical evidence is not exact")
    if git(args.groot_source, "rev-parse", "HEAD") != GROOT_SHA:
        raise Stage32Error("GR00T source SHA mismatch")
    if sha256(args.modality_config) != MODALITY_SHA:
        raise Stage32Error("modality identity mismatch")
    if sha256(args.dataset_dir / "meta" / "stats.json") != STATS_SHA or sha256(args.dataset_dir / "meta" / "relative_stats.json") != RELATIVE_STATS_SHA:
        raise Stage32Error("dataset statistics identity mismatch")
    return {
        "schema": SCHEMA,
        "status": "PASS_PREFLIGHT_NO_MODEL",
        "pi_gate": PI_GATE,
        "project_sha": args.expected_project_sha,
        "checkpoint_1000": {"path": str(args.h2_checkpoint_1000), "tree_sha256": H2_1000_TREE, "file_count": len(rows), "total_bytes": sum(row["bytes"] for row in rows), "trainer_state": trainer},
        "checkpoint_2000": {"path": str(args.h2_checkpoint_2000), "tree_sha256": H2_2000_TREE, "seal_sha256": H2_2000_SEAL},
        "historical_run_manifest_sha256": sha256(args.stage28_execution_manifest),
        "historical_stage28_result_sha256": STAGE28_RESULT,
        "historical_stage31_result_sha256": STAGE31_RESULT,
        "historical_stage31_posthoc_sha256": STAGE31_POSTHOC,
        "identities": {"tiny1_tree_sha256": TINY1_TREE, "modality_sha256": MODALITY_SHA, "stats_sha256": STATS_SHA, "relative_stats_sha256": RELATIVE_STATS_SHA, "groot_sha": GROOT_SHA},
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


def execute(args, p0):
    import numpy as np
    import torch
    from scripts import stage26_h1_c1_fixed_noise_local_flow as stage26
    from scripts import stage29_c2_free_running_transfer as stage29
    from scripts import stage31_on_trajectory_flow_field_audit as stage31
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "H200" not in torch.cuda.get_device_name(0):
        raise Stage32Error("Stage32 requires exactly one visible H200")
    stage23 = stage26._load_stage23_protocol()
    stage23.assert_protocol_static(stage23.protocol_plan())
    contract = stage26._static_probe_contract()
    probes = contract["probes"]
    if len(probes) != 9 or {int(p["noise_seed"]) for p in probes} != set(SEEDS) or {int(p["t_discretized"]) for p in probes} != {67, 369, 783}:
        raise Stage32Error("fixed Stage32 probe contract drifted")
    processor, embodiment, prepared, snapshot, batch = prepare_inputs(args, stage26)
    fixed = tuple(stage26._canonical_probe_inputs(stage23, prepared["expert_action_cpu"], p) for p in probes)
    state_path = args.output_root / "stage32_execution_state.json"
    state = {"schema": SCHEMA, "status": "READY_BEFORE_FIRST_SCIENTIFIC_CALL", "scientific_execution_begun": False, "teacher_forced_probes": 0, "sampler_calls": 0}
    write_json(state_path, state)
    teacher_counter = [0]
    def record_teacher(_row):
        teacher_counter[0] += 1
        write_json(state_path, {**state, "status": "SCIENTIFIC_EXECUTION_IN_PROGRESS", "scientific_execution_begun": True, "teacher_forced_probes": teacher_counter[0]})
    teacher = stage26._run_one_checkpoint(
        checkpoint_label="H2@1000", checkpoint=args.h2_checkpoint_1000, groot_source=args.groot_source,
        prepared=prepared, processor=processor, embodiment=embodiment, stage23=stage23,
        probes=probes, fixed_probes=fixed, device=args.device, snapshot=snapshot,
        on_before_direct_probe_forward=lambda label, probe: write_json(state_path, {**state, "status": "SCIENTIFIC_EXECUTION_BEGUN", "scientific_execution_begun": True, "current_probe_id": probe["probe_id"]}),
        on_record=record_teacher,
    )
    if len(teacher) != 9:
        raise Stage32Error("teacher-forced count drifted")
    historical = read_json(args.stage31_result)
    expected_noise = {int(r["seed"]): r["initial_noise"] for r in historical["records"] if r["checkpoint"] == "BASE"}
    conditioning = stage29._conditioning_manifest(batch)
    expert = stage26._decode_physical_action(processor, embodiment, prepared["expert_action_cpu"].numpy(), snapshot)
    model = stage29._load_model(args.h2_checkpoint_1000, args.device, torch)
    sampler = []
    for seed in SEEDS:
        rng = stage29._reset_rng(seed, torch, np)
        observer = stage31.StockSamplerTrace(model.action_head)
        started = time.perf_counter()
        with stage29._capture_stock_initial_noise(torch) as calls:
            with observer:
                write_json(state_path, {**state, "status": "SCIENTIFIC_EXECUTION_BEGUN", "scientific_execution_begun": True, "current_seed": seed})
                with torch.inference_mode():
                    prediction = model.get_action(**copy.deepcopy(batch))
        torch.cuda.synchronize()
        noise = stage29._validate_noise_capture(calls, model)
        noise_tensor = noise.pop("tensor_float32")
        if noise["raw_sha256"] != expected_noise[seed]["raw_sha256"] or noise["float32_sha256"] != expected_noise[seed]["float32_sha256"]:
            raise Stage32Error(f"initial noise mismatch for seed {seed}")
        trace = stage31._finalize_trace(observer, prediction["action_pred"], torch)
        raw = prediction["action_pred"].detach().float().contiguous().cpu().numpy().astype(np.float32, copy=False)
        decoded = stage26._decode_physical_action(processor, embodiment, raw, snapshot)
        path = args.output_root / "traces" / f"seed_{seed}_h2_1000_stock_sampler_trace.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, x_trajectory_float32=trace["x_trajectory_float32"], v_trajectory_float32=trace["v_trajectory_float32"], timestep_ids=np.asarray(trace["timestep_ids"], dtype=np.int64))
        sampler.append({
            "seed": seed, "checkpoint": "H2@1000", "rng_before_call": rng,
            "initial_noise": noise, "initial_noise_matches_stage31": True,
            "conditioning_sha256": conditioning["sha256"],
            "trace": {"tensor_records": trace["tensor_records"], "timestep_ids": trace["timestep_ids"], "euler_recurrence": trace["euler_recurrence"]},
            "trace_artifact": {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)},
            "trace_metrics": stage31.trace_metrics(trace["x_trajectory_float32"], trace["v_trajectory_float32"], prepared["expert_action_cpu"].numpy().astype(np.float32)),
            "physical_metrics": stage29._trajectory_metrics(decoded, expert, stage23, stage26, processor, embodiment, snapshot, args.h2_checkpoint_1000, "H2@1000"),
            "raw_universal_action_sha256": hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
            "decoded_physical_action_sha256": hashlib.sha256(np.ascontiguousarray(decoded).tobytes()).hexdigest(),
            "latency_s": time.perf_counter() - started,
            "all_finite": True,
        })
    if len(sampler) != 3:
        raise Stage32Error("sampler count drifted")
    historical28 = read_json(args.stage28_result)
    h2000_flow = {row["probe_id"]: float(row["h2_mse"]) for row in historical28["paired_aggregates"]["paired"]}
    h1000_flow = {row["probe_id"]: float(row["flow_metrics"]["active_masked_mse"]) for row in teacher}
    h2000_final = {int(row["seed"]): float(row["metrics"]["final_x4"]["scopes"]["active10"]["state_mse_x4"]) for row in historical["records"] if row["checkpoint"] == "H2@2000"}
    h1000_final = {int(row["seed"]): float(row["trace_metrics"]["final_x4"]["scopes"]["active10"]["state_error_mse"]) for row in sampler}
    flow = [{"probe_id": k, "h2_1000": h1000_flow[k], "h2_2000": h2000_flow[k], "strictly_lower": h2000_flow[k] < h1000_flow[k]} for k in sorted(h1000_flow)]
    free = [{"seed": s, "h2_1000_final_active10_mse": h1000_final[s], "h2_2000_final_active10_mse": h2000_final[s], "strictly_lower": h2000_final[s] < h1000_final[s]} for s in SEEDS]
    promotion = {"teacher_forced_9_of_9": all(x["strictly_lower"] for x in flow), "sampler_3_of_3": all(x["strictly_lower"] for x in free), "all_finite": True, "identity_checks": True, "scientific_contract_changed": False}
    promotion["pass"] = all(promotion.values())
    result = {"schema": SCHEMA, "status": "PASS_EXECUTION_COMPLETE", "pi_gate": PI_GATE, "preflight": p0, "teacher_forced_records": teacher, "sampler_records": sampler, "promotion": {"flow": flow, "free_running": free, **promotion}, "execution": {"model_constructed": True, "model_forward": True, "teacher_forced_probes": 9, "sampler_calls": 3, "training": False, "backward": False, "checkpoint_write": False, "validation_e_read": False, "holdout_f_read": False, "wbc_executed": False, "hardware_executed": False}, "scientific_contract_changed": False, "next_gate": "STAGE33_CONDITIONAL_AUTONOMOUS" if promotion["pass"] else NEXT_GATE}
    write_json(args.output_root / "stage32_h2_1000_result.json", result)
    write_json(state_path, {**state, "status": "PASS_EXECUTION_COMPLETE", "scientific_execution_begun": True, "teacher_forced_probes": 9, "sampler_calls": 3})
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("preflight", "execute"), required=True)
    for name in ("project_repo", "base_model_path", "dataset_dir", "groot_source", "cosmos_snapshot", "modality_config", "snapshot_root", "h2_checkpoint_1000", "h2_checkpoint_2000", "h2_checkpoint_2000_seal", "stage28_execution_manifest", "stage28_result", "stage31_root", "stage31_result", "stage31_posthoc", "output_root", "pbs_template"):
        p.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=True)
    p.add_argument("--expected-project-sha", required=True)
    p.add_argument("--authorization-token", required=True)
    p.add_argument("--pbs-project", default=PROJECT)
    p.add_argument("--pbs-job-id", default="PREFLIGHT_NO_PBS")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--runner-sha256")
    p.add_argument("--pbs-sha256")
    return p


def main():
    args = parser().parse_args()
    try:
        args.output_root.mkdir(parents=True, exist_ok=True)
        p0 = verify_p0(args)
        write_json(args.output_root / "stage32_preflight.json", p0)
        if args.mode == "preflight":
            print(json.dumps(p0, indent=2, sort_keys=True))
        else:
            print(json.dumps(execute(args, p0), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {"schema": SCHEMA, "status": "STOP_OR_BLOCKED", "error_type": type(exc).__name__, "error_summary": str(exc).splitlines()[0][:1000], "next_gate": NEXT_GATE, "scientific_contract_changed": False}
        args.output_root.mkdir(parents=True, exist_ok=True)
        write_json(args.output_root / "stage32_failure.json", failure)
        print(json.dumps(failure, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
