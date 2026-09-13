#!/usr/bin/env python3
"""CPU-only recovery of a completed PI0159 raw local-field attempt.

PI0159's model attempt saved all twelve raw bundles and then failed in the
derived CPU aggregation because the batch dimension was not removed before
applying the [40,132] StructuredAction mask.  This companion intentionally
does not import or construct a model.  It verifies the sealed raw bundles,
replays the runtime-V2 decoder and state audit, and emits the scientific
packet from those raw files only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
SRC = REPO / "src"
for _path in (SCRIPTS, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


PI_GATE = "PI_GATE_STATE_0159"
PI_COMMENT = 5651409238
PR_COMMENT = 5651409688
RETURN_TOKEN = "CORELAB_PI0159_CONDITIONING_MECHANISM_FOR_PI_REVIEW"
BRANCH = "stage45f/corelab-local-initial-state-pilot"
STARTING_HEAD = "e14b5b587eafc8c2e50aa5da1375aa2428cbffaa"
HORIZON = 40
ACTION_DIM = 132
STATE_DIM = 10
PHYSICAL_DIM = 10
RAW_SHAPE = (1, HORIZON, ACTION_DIM)
TIMESTAMPS = np.arange(HORIZON, dtype=np.float64) * 0.05
T_VALUES = (0.06776241796562847, 0.36966943557751086, 0.7837719744658148)
T_BUCKETS = (67, 369, 783)
N2_STATS_SHA = "0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb"
N2_STATS_PAYLOAD_SHA = "508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572"
TRAIN_INPUT_FILE_SHA = "3b525a5a628d3a9ee71894fdd9c6c77b2e888406e87926594700314768181866"
C0_INPUT_FILE_SHA = "67ca605ca90e05633b22dd0447e4873ee29c6a4b31be983b94c11db35f9932be"
TRAIN_LABEL_ID = "stage22_expert_red_target_train_a:expert-0000"


class ReconstructionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReconstructionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: Any) -> str:
    return sha256_bytes(np.ascontiguousarray(np.asarray(value)).tobytes())


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> str:
    path = Path(path)
    require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return sha256_file(path)


def write_text(path: Path, value: str) -> str:
    path = Path(path)
    require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return sha256_file(path)


def write_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> str:
    path = Path(path)
    require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def _load_runner() -> Any:
    import importlib.util

    path = SCRIPTS / "stage45f_pi0159_conditioning_mechanism.py"
    spec = importlib.util.spec_from_file_location("pi0159_runner_for_reconstruct", path)
    require(spec is not None and spec.loader is not None, f"cannot import PI0159 runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_records(raw_root: Path, runner: Any) -> list[dict[str, Any]]:
    expected = {(condition, bucket) for condition in ("TRAIN_REFERENCE", "STATE_C0_VISION_TRAIN", "VISION_C0_STATE_TRAIN", "C0_FULL") for bucket in T_BUCKETS}
    records: list[dict[str, Any]] = []
    for path in sorted((Path(raw_root) / "raw").glob("*.npz")):
        require(path.is_file() and not path.is_symlink(), f"raw bundle is not a regular file: {path}")
        with np.load(path, allow_pickle=False) as data:
            required = {"x0_float32", "x_t_float32", "v_target_float32", "v_hat_float32", "x1_hat_float32", "active_mask_float64", "state_v1_float32", "t_value", "t_bucket"}
            require(required.issubset(data.files), f"raw fields missing: {path}")
            x0 = np.asarray(data["x0_float32"], dtype=np.float32)
            x_t = np.asarray(data["x_t_float32"], dtype=np.float32)
            v_target = np.asarray(data["v_target_float32"], dtype=np.float32)
            v_hat = np.asarray(data["v_hat_float32"], dtype=np.float32)
            x1_hat = np.asarray(data["x1_hat_float32"], dtype=np.float32)
            mask = np.asarray(data["active_mask_float64"], dtype=np.float64)
            state = np.asarray(data["state_v1_float32"], dtype=np.float32).reshape(-1)
            t_value = float(np.asarray(data["t_value"]).reshape(-1)[0])
            bucket = int(np.asarray(data["t_bucket"]).reshape(-1)[0])
        name = path.stem
        condition = next((candidate for candidate in ("TRAIN_REFERENCE", "STATE_C0_VISION_TRAIN", "VISION_C0_STATE_TRAIN", "C0_FULL") if f"_{candidate}_t" in name), None)
        require(condition is not None, f"unrecognized raw bundle name: {path.name}")
        require((condition, bucket) in expected, f"unexpected raw key: {condition} t{bucket}")
        expected.discard((condition, bucket))
        require(x0.shape == RAW_SHAPE and x_t.shape == RAW_SHAPE and v_target.shape == RAW_SHAPE and v_hat.shape == RAW_SHAPE and x1_hat.shape == RAW_SHAPE, f"raw ABI drift: {path}")
        require(mask.shape == (HORIZON, ACTION_DIM) and int(np.count_nonzero(mask)) == 190, f"raw mask drift: {path}")
        require(state.shape == (STATE_DIM,) and np.isfinite(state).all(), f"raw state drift: {path}")
        for label, value in (("x0", x0), ("x_t", x_t), ("v_target", v_target), ("v_hat", v_hat), ("x1_hat", x1_hat)):
            require(np.isfinite(value).all(), f"raw {label} nonfinite: {path}")
        require(any(abs(t_value - candidate) <= 1e-15 and bucket == expected_bucket for candidate, expected_bucket in zip(T_VALUES, T_BUCKETS)), f"raw flow grid drift: {path}")
        require(np.allclose(x_t, ((1.0 - t_value) * x0 + t_value * (x0 + v_target)).astype(np.float32), rtol=0.0, atol=2e-6), f"raw x_t algebra drift: {path}")
        require(np.allclose(x1_hat, (x_t + (1.0 - t_value) * v_hat).astype(np.float32), rtol=0.0, atol=2e-6), f"raw x1_hat algebra drift: {path}")
        records.append({"condition": condition, "bucket": bucket, "t_value": t_value, "path": str(path), "raw_file_sha256": sha256_file(path), "x0": x0, "x_t": x_t, "v_target": v_target, "v_hat": v_hat, "x1_hat": x1_hat, "mask": mask.astype(bool), "state": state, "v_hat_array_sha256": sha256_array(v_hat), "x1_hat_array_sha256": sha256_array(x1_hat)})
    require(not expected and len(records) == 12, f"raw bundle coverage incomplete: missing={sorted(expected)} count={len(records)}")
    return sorted(records, key=lambda item: (("TRAIN_REFERENCE", "STATE_C0_VISION_TRAIN", "VISION_C0_STATE_TRAIN", "C0_FULL").index(item["condition"]), T_BUCKETS.index(item["bucket"])))


def _delta(value: np.ndarray) -> dict[str, Any]:
    x = np.asarray(value, dtype=np.float64)
    return {"max_abs": float(np.max(np.abs(x))), "rmse": float(np.sqrt(np.mean(x * x))), "count": int(x.size), "finite": bool(np.isfinite(x).all())}


def _field_blocks(value: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    x = np.asarray(value, dtype=np.float64)
    if x.shape == RAW_SHAPE:
        x = x[0]
    require(x.shape == (HORIZON, ACTION_DIM), f"field block ABI drift: {x.shape}")
    return {"active_190": _delta(x[mask]), "xyz": _delta(x[:, :3]), "rot6d": _delta(x[:, 3:9]), "timing": _delta(x[:, 9]), "gripper": _delta(x[:, 10]), "padding": _delta(x[:, 11:])}


def _tail_metrics(physical: np.ndarray) -> dict[str, Any]:
    xyz = np.asarray(physical, dtype=np.float64)[:, :3]
    tail = xyz[20:]
    second = np.diff(tail, n=2, axis=0)
    return {"tail_start_xyz": tail[0].tolist(), "tail_end_xyz": tail[-1].tolist(), "tail_displacement_m": float(np.linalg.norm(tail[-1] - tail[0])), "tail_path_length_m": float(np.sum(np.linalg.norm(np.diff(tail, axis=0), axis=1))), "tail_roughness_mean": float(np.mean(np.linalg.norm(second, axis=1))) if len(second) else 0.0}


def _tail_delta(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    c = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    return {"tail_xyz": _delta(c[20:, :3] - b[20:, :3]), "tail_full_physical": _delta(c[20:] - b[20:])}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_root).resolve()
    require(not output.exists(), f"output root already exists: {output}")
    output.mkdir(parents=True)
    runner = _load_runner()
    raw_root = Path(args.raw_root).resolve()
    admission = read_json(raw_root / "admission.json")
    require(admission.get("authority", {}).get("pi_gate") == PI_GATE, "raw admission gate drift")
    lock_path = Path(args.control_lock).resolve()
    lock = read_json(lock_path)
    require(lock.get("schema") == "CORELAB_STAGE45F_PI0159_ATTEMPT_LOCK_V1", "control lock schema drift")
    require(lock.get("output_root") == str(raw_root), "control lock output-root drift")
    require(lock.get("status") == "BLOCKED_OR_FAILED", "raw attempt status drift")
    lock_counters = lock.get("counters", {})
    require(lock_counters.get("action_head_calls") == 12 and lock_counters.get("conditioning_completed") == 4, "raw control-lock counter drift")
    require(lock_counters.get("model_side_calls") == 16 and lock_counters.get("model_construction_attempts") == 1, "raw control-lock model budget drift")
    records = _raw_records(raw_root, runner)
    train = runner._load_observation(Path(args.train_input), TRAIN_INPUT_FILE_SHA, "canonical TRAIN/T")
    c0 = runner._load_observation(Path(args.archived_c0_input), C0_INPUT_FILE_SHA, "archived C0/C")
    stats_path = Path(args.n2_stats).resolve()
    require(sha256_file(stats_path) == N2_STATS_SHA, "N2 stats file drift during reconstruction")

    condition_specs = {
        "TRAIN_REFERENCE": (train["body_rgb"], train["wrist_rgb"], train["state"], "TRAIN/T"),
        "STATE_C0_VISION_TRAIN": (train["body_rgb"], train["wrist_rgb"], c0["state"], "V_T,S_C"),
        "VISION_C0_STATE_TRAIN": (c0["body_rgb"], c0["wrist_rgb"], train["state"], "V_C,S_T"),
        "C0_FULL": (c0["body_rgb"], c0["wrist_rgb"], c0["state"], "C0/C"),
    }

    # CPU-only processor audit: this reconstructs state clipping metadata but
    # never constructs GR00T weights or calls a model.
    old = runner._load_old_probe()
    cosmos_binding = old._bind_local_cosmos_runtime(Path(args.stage10_root), Path(args.cosmos_snapshot))
    from stage26_h1_c1_fixed_noise_local_flow import _build_runtime_processor, _load_snapshot
    from stage40_b0_model_control import _bind_local_cosmos_processor
    processor_binding = _bind_local_cosmos_processor(Path(args.stage10_root), Path(args.cosmos_snapshot))
    processor, embodiment, _ = _build_runtime_processor(groot_source=Path(args.stage10_root) / "src" / "isaac-gr00t", base_checkpoint=Path(args.base_model), dataset_dir=Path(args.dataset), modality_path=Path(args.modality), cosmos_snapshot=Path(args.cosmos_snapshot))
    audit_by_condition = {}
    for condition, (body, wrist, state, _) in condition_specs.items():
        inputs = old._build_c1_inputs(processor, embodiment, {"body": body, "wrist": wrist, "state": state}, train=True)
        runner._assert_boundary(inputs, condition)
        audit_by_condition[condition] = runner._state_audit(processor, embodiment, np.asarray(state, dtype=np.float32), inputs, condition)

    # Validate the same label-side anchor reconciliation used by the failed
    # model attempt.  This is CPU metadata only and does not alter the raw.
    native = _load_snapshot(Path(args.native_a))
    from go2arx5_vla.structured_action import reconcile_stage45b_label_anchor
    expert = np.asarray(native.physical_action, dtype=np.float64)
    label_state = np.asarray(native.state_v1.as_array(), dtype=np.float32).astype(np.float64)
    _, anchor_reconciliation = reconcile_stage45b_label_anchor(expert, label_state)

    import stage45c_h200_eval_runtime_adapter as runtime
    physical_by_condition: dict[str, dict[int, dict[str, Any]]] = {}
    for row in records:
        previous_sha, previous_payload = runtime.FULL24_STATS_SHA256, runtime.FULL24_STATS_PAYLOAD_SHA256
        runtime.FULL24_STATS_SHA256, runtime.FULL24_STATS_PAYLOAD_SHA256 = N2_STATS_SHA, N2_STATS_PAYLOAD_SHA
        try:
            actual, details_actual = runtime._decode_v2(row["x1_hat"], row["state"], stats_path, TIMESTAMPS, float(native.observation_timestamp), f"PI0159:{row['condition']}:t{row['bucket']}:actual")
            standardized, details_standardized = runtime._decode_v2(row["x1_hat"], train["state"], stats_path, TIMESTAMPS, float(native.observation_timestamp), f"PI0159:{row['condition']}:t{row['bucket']}:standardized-T")
        finally:
            runtime.FULL24_STATS_SHA256, runtime.FULL24_STATS_PAYLOAD_SHA256 = previous_sha, previous_payload
        physical_by_condition.setdefault(row["condition"], {})[row["bucket"]] = {"actual": np.asarray(actual, dtype=np.float64), "standardized_T": np.asarray(standardized, dtype=np.float64), "decode_actual": details_actual, "decode_standardized_T": details_standardized}

    base_by_bucket = {bucket: next(row for row in records if row["condition"] == "TRAIN_REFERENCE" and row["bucket"] == bucket) for bucket in T_BUCKETS}
    condition_rows: list[dict[str, Any]] = []
    differences: list[dict[str, Any]] = []
    for condition, (_, _, _, source_label) in condition_specs.items():
        for t_value, bucket in zip(T_VALUES, T_BUCKETS):
            row = next(item for item in records if item["condition"] == condition and item["bucket"] == bucket)
            baseline = base_by_bucket[bucket]
            local_error = _field_blocks(row["v_hat"] - row["v_target"], row["mask"])
            active_delta = _field_blocks(row["v_hat"] - baseline["v_hat"], row["mask"])
            physical = physical_by_condition[condition][bucket]["standardized_T"]
            base_physical = physical_by_condition["TRAIN_REFERENCE"][bucket]["standardized_T"]
            tail = _tail_metrics(physical)
            tail_delta = _tail_delta(physical, base_physical)
            condition_rows.append({"condition": condition, "source_label": source_label, "t_value": t_value, "t_bucket": bucket, "local_field_active_rmse": local_error["active_190"]["rmse"], "local_field_active_max_abs": local_error["active_190"]["max_abs"], "local_field_xyz_rmse": local_error["xyz"]["rmse"], "active_delta_vs_train_rmse": active_delta["active_190"]["rmse"], "active_delta_vs_train_max_abs": active_delta["active_190"]["max_abs"], "tail_path_length_m": tail["tail_path_length_m"], "tail_roughness_mean": tail["tail_roughness_mean"], "tail_displacement_m": tail["tail_displacement_m"], "tail_delta_xyz_rmse": tail_delta["tail_xyz"]["rmse"], "tail_delta_xyz_max_abs": tail_delta["tail_xyz"]["max_abs"], "processor_information_loss": audit_by_condition[condition]["information_loss_observed"], "processor_clip_dimensions": audit_by_condition[condition]["clipping"], "raw_file_sha256": row["raw_file_sha256"]})
            differences.append({"condition": condition, "t_bucket": bucket, "local_field_error_vs_teacher": local_error, "active_output_delta_vs_train": active_delta, "tail_delta_vs_train_standardized_anchor": tail_delta, "physical_actual_anchor_tail": _tail_metrics(physical_by_condition[condition][bucket]["actual"]), "physical_standardized_T_anchor_tail": tail})

    state_rows = [row for row in condition_rows if row["condition"] == "STATE_C0_VISION_TRAIN"]
    vision_rows = [row for row in condition_rows if row["condition"] == "VISION_C0_STATE_TRAIN"]
    state_effect = float(np.mean([row["active_delta_vs_train_rmse"] for row in state_rows]))
    vision_effect = float(np.mean([row["active_delta_vs_train_rmse"] for row in vision_rows]))
    state_tail_effect = float(np.mean([row["tail_delta_xyz_rmse"] for row in state_rows]))
    vision_tail_effect = float(np.mean([row["tail_delta_xyz_rmse"] for row in vision_rows]))
    if state_effect > 0.0 and vision_effect > 0.0:
        classification = "STATE_AND_VISION_REACH_LOCAL_FIELD"
        sentence = "Both single-modality substitutions changed the active teacher-forced local field at the sampled flow times; standardized-T tails show the corresponding downstream trajectory change without treating anchor p0 mechanics as learned state response."
    elif state_effect > 0.0:
        classification = "STATE_DOMINANT_LOCAL_FIELD_EFFECT"
        sentence = "State substitution changed the active teacher-forced local field while the vision-only response was null under exact comparison; this remains a fixed-policy diagnostic, not a generalization claim."
    elif vision_effect > 0.0:
        classification = "VISION_DOMINANT_LOCAL_FIELD_EFFECT"
        sentence = "Vision substitution changed the active teacher-forced local field while the state-only response was null under exact comparison; this remains a fixed-policy diagnostic, not a generalization claim."
    else:
        classification = "NO_DETECTABLE_LOCAL_FIELD_EFFECT"
        sentence = "Neither single-modality substitution changed the active teacher-forced local field at the sampled flow times."

    hashes = {}
    fields = ["condition", "source_label", "t_value", "t_bucket", "local_field_active_rmse", "local_field_active_max_abs", "local_field_xyz_rmse", "active_delta_vs_train_rmse", "active_delta_vs_train_max_abs", "tail_path_length_m", "tail_roughness_mean", "tail_displacement_m", "tail_delta_xyz_rmse", "tail_delta_xyz_max_abs", "processor_information_loss", "processor_clip_dimensions", "raw_file_sha256"]
    hashes["conditions_csv_sha256"] = write_csv(output / "conditions.csv", condition_rows, fields)
    hashes["condition_differences_json_sha256"] = write_json(output / "condition_differences.json", differences)
    hashes["processed_state_audit_json_sha256"] = write_json(output / "processed_state_audit.json", audit_by_condition)
    hashes["anchor_reconciliation_json_sha256"] = write_json(output / "anchor_reconciliation.json", anchor_reconciliation)
    hashes["raw_manifest_json_sha256"] = write_json(output / "raw_manifest.json", [{key: value for key, value in row.items() if key not in {"x0", "x_t", "v_target", "v_hat", "x1_hat", "mask", "state"}} for row in records])
    prior = read_json(Path(args.prior_c0_analysis))
    hashes["prior_c0_reference_json_sha256"] = write_json(output / "prior_c0_reference.json", {"status": "REUSED_SEALED_CPU_EVIDENCE", "path": str(Path(args.prior_c0_analysis).resolve()), "sha256": sha256_file(Path(args.prior_c0_analysis)), "content": prior})
    hashes["trajectory_svg_sha256"] = runner._make_svg({condition: physical_by_condition[condition][T_BUCKETS[-1]]["standardized_T"] for condition in condition_specs}, output / "trajectory_plot.svg")
    conclusion = f"""# PI0159 conditioning-mechanism result

Status: **COMPLETED (CPU reconstruction from sealed raw)**

Science classification: **{classification}**

{sentence}

The original model attempt completed one model construction, four conditioning
passes, and twelve teacher-forced action-head calls. Its raw bundles were
sealed before the aggregation bug. This packet performs no model import,
model forward, sampler call, training, WBC, simulator, or closed-loop retry.

The fixed local-flow algebra is `x_t=(1-t)x0+t*x1`, `v_target=x1-x0`, and
`x1_hat=x_t+(1-t)*v_hat`; the exact N2 runtime-V2 decoder is applied only
after raw verification. State clipping is reported per condition. The
standardized-T anchor is the primary cross-condition tail view, while the
actual-anchor view is retained as decoder mechanics. TRAIN world-target and
task-success metrics remain unverified.

## Single recommendation

Authorize one fixed-budget independent-expert / visited-state local-field
supervision diagnostic before any ICL or training decision, using the same
raw-first ABI on post-policy states. Do not start it automatically.
"""
    hashes["one_page_conclusion_sha256"] = write_text(output / "one_page_conclusion.md", conclusion)
    report = {"schema": "CORELAB_STAGE45F_PI0159_CONDITIONING_MECHANISM_RESULT_V1", "status": "COMPLETED", "token": RETURN_TOKEN, "science_classification": classification, "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_comment": PR_COMMENT}, "repository": {"branch": BRANCH, "protocol_starting_head": STARTING_HEAD, "model_attempt_repository_head": admission.get("repository_head") or admission.get("repository", {}).get("execution_head"), "reconstruction_script_head": str(args.repository_head)}, "raw_attempt": {"root": str(raw_root), "admission_sha256": sha256_file(raw_root / "admission.json"), "original_status": admission.get("status"), "original_counters": admission.get("counters"), "raw_rows": len(records)}, "identities": {"n2_stats_sha256": N2_STATS_SHA, "n2_stats_payload_sha256": N2_STATS_PAYLOAD_SHA, "train_input_sha256": TRAIN_INPUT_FILE_SHA, "c0_input_sha256": C0_INPUT_FILE_SHA, "native_expert_snapshot": str(Path(args.native_a).resolve()), "groot_commit": runner.GROOT_COMMIT, "cosmos_revision": runner.COSMOS_REVISION, "cosmos_tree_sha256": runner.COSMOS_TREE, "modality_sha256": runner.MODALITY_SHA, "cosmos_binding": cosmos_binding, "processor_binding": processor_binding}, "anchor_reconciliation": anchor_reconciliation, "conditions": condition_rows, "condition_differences": differences, "processed_state_audit": audit_by_condition, "prior_c0_reference": {"path": str(Path(args.prior_c0_analysis).resolve()), "sha256": sha256_file(Path(args.prior_c0_analysis))}, "flow": {"t_values": list(T_VALUES), "t_buckets": list(T_BUCKETS), "equations": ["x_t=(1-t)x0+t*x1", "v_target=x1-x0", "x1_hat=x_t+(1-t)*v_hat"], "active_mask_shape": [HORIZON, ACTION_DIM], "active_scalars": 190}, "outputs": hashes, "counters": {"original_model_construction_attempts": admission.get("counters", {}).get("model_construction_attempts"), "original_conditioning_completed": admission.get("counters", {}).get("conditioning_completed"), "original_action_head_calls": admission.get("counters", {}).get("action_head_calls"), "original_model_side_calls": admission.get("counters", {}).get("model_side_calls"), "cpu_reconstruction_model_calls": 0, "raw_rows": len(records)}, "budget": {"original_conditioning": 4, "original_action_head_calls": 12, "original_model_side_calls": 16, "cpu_reconstruction_model_calls": 0}, "effect_summary": {"state_mean_active_delta_rmse": state_effect, "vision_mean_active_delta_rmse": vision_effect, "state_mean_tail_delta_xyz_rmse": state_tail_effect, "vision_mean_tail_delta_xyz_rmse": vision_tail_effect}, "execution_boundary": {"original_model_forward": 16, "reconstruction_model_forward": 0, "model_construction": 0, "training": False, "backward": False, "optimizer_step": 0, "public_sampler": False, "free_running_inference": False, "icl": False, "new_data": False, "wbc": False, "simulator": False, "hardware": False, "pbs": False, "validation_e_read": False, "holdout_f_read": False, "safety_gate_relaxed": False, "retry": False}, "claim_boundary": "fixed-policy teacher-forced conditioning and decoded-tail characterization only; no task-success, WBC, closed-loop, world-frame TRAIN, ICL, training, or generalization claim", "raw_first": True}
    # The admission file intentionally records the pre-forward boundary; the
    # post-forward counters live in the immutable attempt lock.  Bind both so
    # the recovered packet cannot silently report null counters.
    report["repository"]["model_attempt_repository_head"] = lock.get("repository_head") or admission.get("repository", {}).get("execution_head")
    report["raw_attempt"].update({"control_lock": str(lock_path), "control_lock_sha256": sha256_file(lock_path), "original_status": lock.get("status"), "original_error_type": lock.get("error_type"), "original_counters": lock_counters})
    report["counters"] = {"original_model_construction_attempts": lock_counters.get("model_construction_attempts"), "original_conditioning_completed": lock_counters.get("conditioning_completed"), "original_action_head_calls": lock_counters.get("action_head_calls"), "original_model_side_calls": lock_counters.get("model_side_calls"), "cpu_reconstruction_model_calls": 0, "raw_rows": len(records)}
    report_sha = write_json(output / "pi0159_conditioning_mechanism.json", report)
    hashes["report_sha256"] = report_sha
    evidence_sha = write_json(output / "evidence_hashes.json", {**hashes, "raw_attempt_admission_sha256": sha256_file(raw_root / "admission.json"), "raw_sha256_by_row": {f"{row['condition']}:t{row['bucket']}": row["raw_file_sha256"] for row in records}})
    return {**report, "report_sha256": report_sha, "evidence_hashes_sha256": evidence_sha}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-head", required=True)
    p.add_argument("--raw-root", required=True, type=Path)
    p.add_argument("--control-lock", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--stage10-root", required=True, type=Path)
    p.add_argument("--base-model", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--modality", required=True, type=Path)
    p.add_argument("--cosmos-snapshot", required=True, type=Path)
    p.add_argument("--n2-stats", required=True, type=Path)
    p.add_argument("--train-input", required=True, type=Path)
    p.add_argument("--archived-c0-input", required=True, type=Path)
    p.add_argument("--native-a", required=True, type=Path)
    p.add_argument("--prior-c0-analysis", required=True, type=Path)
    return p


if __name__ == "__main__":
    try:
        result = run(parser().parse_args())
        print(json.dumps({"status": result["status"], "science_classification": result["science_classification"], "token": result["token"], "report_sha256": result["report_sha256"], "counters": result["counters"]}, sort_keys=True))
    except Exception as exc:
        print(f"BLOCKED_OR_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
