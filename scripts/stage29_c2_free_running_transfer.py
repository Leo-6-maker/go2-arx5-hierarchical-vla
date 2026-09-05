#!/usr/bin/env python3
"""Stage29 C2 paired official-sampler transfer diagnostic.

The plan/preflight paths are standard-library-only and perform no model
construction or forward.  The separately authorized execute path runs the
stock pinned GR00T N1.7 four-step sampler exactly six times: BASE and H2@2000
for seeds 2223, 2224, and 2225.  It never trains, writes a checkpoint, runs
WBC/hardware, or reads validation/holdout data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "STAGE29_C2_FREE_RUNNING_TRANSFER_V1"
PI_GATE_ID = "PI_GATE_STATE_0009"
NEXT_GATE = "GPT_PI_REVIEW_AFTER_STAGE29_C2_FREE_RUNNING_RESULT"
STAGE28_ACCEPTED_SHA = "116ada218e02c819f4a2d4bedac7fc3f88e09c58"
STAGE28_ARTIFACT_SEAL_SHA256 = "c895dd96f915c0a0f995116cf252f2a12764573be28d9a5d4edc1e0ecc3d022c"
STAGE28_RESULT_SHA256 = "aa0ab03b851ad1ad6581a71168d92d682deee546db1bb79b76acb2c1acfe3f26"
EXPECTED_PROJECT = "CFP03-CF-004"
EXPECTED_GROOT_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
HORIZON = 40
UNIVERSAL_DIM = 132
ACTIVE_DIM = 10
SAMPLER_STEPS = 4
SEEDS = (2223, 2224, 2225)
CHECKPOINTS = ("BASE", "H2@2000")
MAX_SAMPLER_CALLS = 6
ACTION_PERIOD_S = 0.05

SOURCE_FILES = {
    "gr00t/policy/policy.py": "b53326aee0c26ea34201d70510ad92608a96929d7f39039f578c3d865446de23",
    "gr00t/policy/gr00t_policy.py": "ad1183a3ce1ebed6cace4a9cb8c5a123e823a0760ebf46f3694639a5ad1521ec",
    "gr00t/model/gr00t_n1d7/gr00t_n1d7.py": "6bd66ef768f1484ac5a7889c40358a599f1862b1d554d47d357e113f2f96501c",
    "gr00t/configs/model/gr00t_n1d7.py": "379ff708496a73b25b12f4656850284a16fde0e205828e2a0c0be0af61463744",
    "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py": "77fd0e1f6972cdcf0334f18f2e84724629f6746612721b6b27135ccb23f00b3f",
    "gr00t/data/state_action/state_action_processor.py": "0151dd0bb6cb727fd401bf78121efc532e7184f2575e6ef25294085f629ac145",
    "gr00t/data/utils.py": "fa765b240295d9884ab62326c8a21465ae21e6f10133a4b68ac321e8101e4ab7",
    "gr00t/eval/open_loop_eval.py": "e15706fac4cd8caff3229a1538eb0288b487919b28c0e26ecf30c32d0f61245f",
}

SOURCE_ANCHORS = {
    "gr00t/policy/gr00t_policy.py": (
        "model_pred = self.model.get_action(**collated_inputs)",
        "self.processor.decode_action(",
    ),
    "gr00t/model/gr00t_n1d7/gr00t_n1d7.py": (
        "actions = torch.randn(",
        "dt = 1.0 / self.num_inference_timesteps",
        "for t in range(self.num_inference_timesteps):",
        "actions = actions + dt * pred_velocity * vel_strength",
        "action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, options)",
    ),
    "gr00t/configs/model/gr00t_n1d7.py": (
        "num_inference_timesteps: int = 4",
        "max_action_dim: int = 132",
        "action_horizon: int = 40",
    ),
    "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py": (
        "return self.state_action_processor.unapply_action(",
        "out_dict[key] = action[..., :action_horizon, start_idx : start_idx + joint_dim]",
    ),
    "gr00t/data/state_action/state_action_processor.py": (
        "reference_state=s[-1]",
        "unnormalize_values_minmax(group_values, params)",
    ),
    "gr00t/data/utils.py": ("np.clip(normalized_values, -1.0, 1.0)",),
}


class Stage29Error(RuntimeError):
    """Fail-closed Stage29 contract error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    return str(value)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise Stage29Error(f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


def protocol_plan() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "pi_gate": PI_GATE_ID,
        "stage": "Stage29 C2 teacher-forced to free-running transfer gate",
        "stage28_accepted_sha": STAGE28_ACCEPTED_SHA,
        "sampler": {
            "api": "Gr00tN1d7.get_action -> Gr00tN1d7ActionHead.get_action",
            "algorithm": "stock explicit Euler flow integration",
            "steps": SAMPLER_STEPS,
            "continuous_time_grid": [0.0, 0.25, 0.5, 0.75],
            "discrete_time_grid": [0, 250, 500, 750],
            "external_noise_supported": False,
            "instrumentation": "capture and return unchanged sole stock torch.randn result",
        },
        "experiment": {
            "dataset": "same locked TINY1 training sample",
            "seeds": list(SEEDS),
            "checkpoints": list(CHECKPOINTS),
            "maximum_sampler_calls": MAX_SAMPLER_CALLS,
            "paired_initial_noise_required": True,
            "shapes": {"universal": [1, HORIZON, UNIVERSAL_DIM], "physical": [1, HORIZON, ACTIVE_DIM]},
            "action_period_s": ACTION_PERIOD_S,
        },
        "execution": {
            "gpu": "1x NVIDIA H200",
            "project": EXPECTED_PROJECT,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "scheduler_step": False,
            "resume": False,
            "checkpoint_write": False,
            "wbc": False,
            "hardware": False,
            "validation_e": False,
            "holdout_f": False,
        },
        "next_gate": NEXT_GATE,
    }


def assert_protocol_static(plan: Mapping[str, Any] | None = None) -> None:
    candidate = protocol_plan() if plan is None else dict(plan)
    if candidate.get("schema") != SCHEMA or candidate.get("pi_gate") != PI_GATE_ID:
        raise Stage29Error("Stage29 schema/gate drifted")
    experiment = candidate.get("experiment", {})
    if experiment.get("seeds") != list(SEEDS):
        raise Stage29Error("Stage29 seed set drifted")
    if experiment.get("checkpoints") != list(CHECKPOINTS):
        raise Stage29Error("Stage29 checkpoint pair drifted")
    if experiment.get("maximum_sampler_calls") != MAX_SAMPLER_CALLS:
        raise Stage29Error("Stage29 sampler-call budget drifted")
    execution = candidate.get("execution", {})
    forbidden = (
        "training", "backward", "optimizer_step", "scheduler_step", "resume",
        "checkpoint_write", "wbc", "hardware", "validation_e", "holdout_f",
    )
    if any(execution.get(key) is not False for key in forbidden):
        raise Stage29Error("Stage29 forbidden execution boundary drifted")


def audit_groot_source(groot_source: Path) -> dict[str, Any]:
    if groot_source.is_symlink() or not groot_source.is_dir():
        raise Stage29Error(f"invalid GR00T source root: {groot_source}")
    head = _git(groot_source, "rev-parse", "HEAD")
    dirty = _git(groot_source, "status", "--short")
    if head != EXPECTED_GROOT_SHA or dirty:
        raise Stage29Error(f"GR00T source identity failed: head={head} dirty={bool(dirty)}")
    records = []
    for relative, expected in SOURCE_FILES.items():
        path = groot_source / relative
        observed = _sha256_file(path)
        if observed != expected:
            raise Stage29Error(f"GR00T source file drifted: {relative}: {observed}")
        text = path.read_text(encoding="utf-8")
        missing = [anchor for anchor in SOURCE_ANCHORS.get(relative, ()) if anchor not in text]
        if missing:
            raise Stage29Error(f"GR00T source anchors missing: {relative}: {missing}")
        records.append({"path": relative, "sha256": observed, "anchors": list(SOURCE_ANCHORS.get(relative, ()))})
    return {
        "status": "PASS_UNAMBIGUOUS_OFFICIAL_SAMPLER",
        "git_sha": head,
        "files": records,
        "canonical_call_path": [
            "BasePolicy.get_action", "Gr00tPolicy._get_action", "Gr00tN1d7.get_action",
            "Gr00tN1d7ActionHead.get_action", "Gr00tN1d7ActionHead.get_action_with_features",
        ],
        "competing_stock_pytorch_sampler_paths": 0,
        "model_forward": False,
    }


def _require_project_identity(project_repo: Path, expected_sha: str) -> str:
    head = _git(project_repo, "rev-parse", "HEAD")
    if head != expected_sha:
        raise Stage29Error(f"project HEAD mismatch: {head} != {expected_sha}")
    if _git(project_repo, "status", "--short"):
        raise Stage29Error("project worktree is dirty")
    ancestry = subprocess.run(
        ["git", "-C", str(project_repo), "merge-base", "--is-ancestor", STAGE28_ACCEPTED_SHA, head],
        check=False,
    )
    if ancestry.returncode != 0:
        raise Stage29Error("Stage29 HEAD does not descend from accepted Stage28 HEAD")
    return head


def _ensure_external_output(output_root: Path, immutable_roots: Iterable[Path]) -> None:
    output = output_root.resolve()
    for immutable in immutable_roots:
        source = immutable.resolve()
        if output == source or source in output.parents or output in source.parents:
            raise Stage29Error(f"output overlaps immutable input: {output} vs {source}")
    output.mkdir(parents=True, exist_ok=True)


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.authorization_token != PI_GATE_ID:
        raise Stage29Error("PI gate token mismatch")
    if args.pbs_project != EXPECTED_PROJECT:
        raise Stage29Error("PBS project mismatch")
    assert_protocol_static()
    project_head = _require_project_identity(args.project_repo, args.expected_project_sha)
    runner_sha = _sha256_file(Path(__file__).resolve())
    if runner_sha != args.expected_runner_sha256:
        raise Stage29Error(f"runner SHA mismatch: {runner_sha}")
    pbs_sha = _sha256_file(args.pbs_template)
    if pbs_sha != args.expected_pbs_sha256:
        raise Stage29Error(f"PBS template SHA mismatch: {pbs_sha}")
    audit_sha = _sha256_file(args.source_audit_report)
    if audit_sha != args.expected_source_audit_sha256:
        raise Stage29Error(f"source-audit report SHA mismatch: {audit_sha}")

    from scripts import stage28_h2_fresh_base_2000_fixed_warmup10 as stage28
    stage26 = stage28._load_stage26_module()

    base = stage28._verify_base_checkpoint(args.base_model_path)
    dataset = stage28._verify_tiny_dataset(args.dataset_dir)
    modality_sha = stage28._verify_modality(args.modality_config)
    groot = audit_groot_source(args.groot_source)
    cosmos = stage28._verify_cosmos_cache(args.cosmos_cache_root)
    training = stage28._verify_final_checkpoint_artifacts(
        args.training_output_root,
        expected_checkpoint_tree_sha256=stage28.EXPECTED_H2_CHECKPOINT_TREE_SHA256,
        expected_seal_sha256=stage28.EXPECTED_H2_CHECKPOINT_SEAL_SHA256,
    )
    snapshot, _ = stage26._verify_snapshot_identity(args.snapshot_root)
    stage28_seal_sha = _sha256_file(args.stage28_artifact_seal)
    stage28_result_sha = _sha256_file(args.stage28_result)
    if stage28_seal_sha != STAGE28_ARTIFACT_SEAL_SHA256:
        raise Stage29Error("accepted Stage28 artifact seal drifted")
    if stage28_result_sha != STAGE28_RESULT_SHA256:
        raise Stage29Error("accepted Stage28 result drifted")
    h2_checkpoint = args.training_output_root / "h2_training" / "checkpoint-2000"
    if not h2_checkpoint.is_dir():
        raise Stage29Error("sealed H2@2000 checkpoint is unavailable")
    _ensure_external_output(
        args.output_root,
        (args.base_model_path, args.dataset_dir, args.training_output_root, args.stage28_artifact_seal.parent),
    )
    return {
        "schema": SCHEMA,
        "status": "PASS_PREFLIGHT_NO_MODEL",
        "timestamp_utc": _utc_now(),
        "project": {"head": project_head, "clean": True, "stage28_ancestor": STAGE28_ACCEPTED_SHA},
        "source_audit": groot,
        "identities": {
            "base": base,
            "dataset": dataset,
            "modality_sha256": modality_sha,
            "cosmos": cosmos,
            "h2_training_artifacts": training,
            "h2_checkpoint": str(h2_checkpoint),
            "snapshot": snapshot,
            "stage28_artifact_seal": {"path": str(args.stage28_artifact_seal), "sha256": stage28_seal_sha},
            "stage28_result": {"path": str(args.stage28_result), "sha256": stage28_result_sha},
            "runner_sha256": runner_sha,
            "pbs_template_sha256": pbs_sha,
            "source_audit_report_sha256": audit_sha,
        },
        "execution": {"model_constructed": False, "model_forward": False, "sampler_calls": 0},
        "scientific_contract_changed": False,
    }


def _tensor_bytes(tensor: Any) -> bytes:
    import torch
    value = tensor.detach().contiguous().view(torch.uint8).cpu()
    return value.numpy().tobytes()


def _tensor_record(tensor: Any) -> dict[str, Any]:
    return {
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "raw_sha256": _sha256_bytes(_tensor_bytes(tensor)),
        "float32_sha256": _sha256_bytes(tensor.detach().float().contiguous().cpu().numpy().tobytes()),
        "finite": bool(tensor.isfinite().all().item()),
    }


def _describe_conditioning(value: Any) -> Any:
    try:
        import numpy as np
        import torch
    except ImportError:  # pragma: no cover
        np = None
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return {"kind": "tensor", **_tensor_record(value)}
    if np is not None and isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "kind": "ndarray", "shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": _sha256_bytes(array.tobytes()), "finite": bool(np.isfinite(array).all()) if np.issubdtype(array.dtype, np.number) else None,
        }
    if isinstance(value, Mapping):
        return {str(key): _describe_conditioning(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_describe_conditioning(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        if np is not None:
            array = np.asarray(value)
            if array.dtype != object:
                return _describe_conditioning(array)
    except Exception:
        pass
    return {"kind": type(value).__name__, "text": str(value)}


def _conditioning_manifest(value: Any) -> dict[str, Any]:
    description = _describe_conditioning(value)
    canonical = json.dumps(_jsonable(description), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": _sha256_bytes(canonical), "description": description}


def _reset_rng(seed: int, torch: Any, np: Any) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return {
        "seed": seed,
        "cpu_torch_state_sha256": _sha256_bytes(torch.get_rng_state().cpu().numpy().tobytes()),
        "cuda_torch_state_sha256": _sha256_bytes(torch.cuda.get_rng_state().cpu().numpy().tobytes()),
    }


@contextmanager
def _capture_stock_initial_noise(torch: Any):
    """Capture the sole stock torch.randn output without replacing its value."""

    original = torch.randn
    calls: list[dict[str, Any]] = []

    def wrapped(*positional: Any, **keywords: Any) -> Any:
        value = original(*positional, **keywords)
        record = _tensor_record(value)
        record["tensor_float32"] = value.detach().float().contiguous().cpu()
        calls.append(record)
        return value

    torch.randn = wrapped
    try:
        yield calls
    finally:
        torch.randn = original


def _validate_noise_capture(calls: Sequence[Mapping[str, Any]], model: Any) -> dict[str, Any]:
    if len(calls) != 1:
        raise Stage29Error(f"official sampler emitted {len(calls)} Python torch.randn calls; expected exactly one")
    record = dict(calls[0])
    expected = [1, HORIZON, UNIVERSAL_DIM]
    if record.get("shape") != expected:
        raise Stage29Error(f"initial-noise shape drifted: {record.get('shape')}")
    if record.get("dtype") != str(model.dtype) or record.get("device") != str(model.device):
        raise Stage29Error("initial-noise dtype/device differs from model")
    if record.get("finite") is not True:
        raise Stage29Error("initial noise is non-finite")
    return record


def validate_paired_noise(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = {(int(row["seed"]), str(row["checkpoint"])): row for row in records}
    pairs = []
    for seed in SEEDS:
        base = rows.get((seed, "BASE"))
        h2 = rows.get((seed, "H2@2000"))
        if base is None or h2 is None:
            raise Stage29Error(f"missing paired sampler record for seed {seed}")
        left = base["initial_noise"]["raw_sha256"]
        right = h2["initial_noise"]["raw_sha256"]
        if left != right:
            raise Stage29Error(f"paired initial-noise mismatch for seed {seed}")
        pairs.append({"seed": seed, "base_raw_sha256": left, "h2_raw_sha256": right, "exact_match": True})
    return {"status": "MATCH", "pairs": pairs, "all_exact": True}


def _summary(values: Iterable[float]) -> dict[str, Any]:
    rows = [float(value) for value in values]
    return {"count": len(rows), "min": min(rows), "median": median(rows), "mean": mean(rows), "max": max(rows)}


def _trajectory_metrics(decoded: Any, expert: Any, stage23: Any, stage26: Any, processor: Any, embodiment: Any, snapshot: Any, checkpoint: Path, label: str) -> dict[str, Any]:
    import numpy as np
    from go2arx5_vla.trajectory.so3 import geodesic_angle, rot6d_rows_to_matrix

    predicted = np.asarray(decoded, dtype=np.float32)
    target = np.asarray(expert, dtype=np.float32)
    if predicted.shape != (1, HORIZON, ACTIVE_DIM) or target.shape != predicted.shape:
        raise Stage29Error(f"physical trajectory shape mismatch: {predicted.shape} vs {target.shape}")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise Stage29Error("physical trajectory contains non-finite values")
    delta = predicted - target
    rotation = np.asarray(
        geodesic_angle(rot6d_rows_to_matrix(predicted[0, :, 3:9]), rot6d_rows_to_matrix(target[0, :, 3:9])),
        dtype=np.float64,
    ).reshape(-1)
    gripper = np.abs(delta[0, :, 9])
    profile = stage23._profile(predicted[0].tolist(), ACTION_PERIOD_S)
    expert_profile = stage23._profile(target[0].tolist(), ACTION_PERIOD_S)
    canonical = stage26._profile_result(
        decoded=predicted, snapshot=snapshot, model_checkpoint=checkpoint, source_label=label,
    )
    canonical.pop("decoded_physical_action", None)
    return {
        "physical_action_mse_active_10d": float(np.mean(delta.astype(np.float64) ** 2)),
        "eef_position_mse": float(np.mean(delta[..., :3].astype(np.float64) ** 2)),
        "endpoint_position_error_m": float(np.linalg.norm(delta[0, -1, :3].astype(np.float64))),
        "rotation_error_rad": _summary(rotation.tolist()),
        "absolute_gripper_error": _summary(gripper.astype(np.float64).tolist()),
        "predicted_profile": profile,
        "expert_profile": expert_profile,
        "canonical_chunk_profile": canonical["prediction_profile"],
        "safety_gate": canonical["safety_gate"],
        "all_finite": True,
    }


def _metric_value(row: Mapping[str, Any], name: str) -> float:
    metrics = row["metrics"]
    profile = metrics["predicted_profile"]
    choices = {
        "physical_action_mse_active_10d": metrics["physical_action_mse_active_10d"],
        "eef_position_mse": metrics["eef_position_mse"],
        "endpoint_position_error_m": metrics["endpoint_position_error_m"],
        "path_length_m": profile["path_length_m"],
        "linear_speed_mean_mps": profile["linear_speed_mps"]["mean"],
        "linear_speed_max_mps": profile["linear_speed_mps"]["max"],
        "roughness_mean_m": profile["second_difference_m"]["mean"],
        "adjacent_cosine_median": profile["adjacent_direction_cosine"]["median"],
        "back_and_forth_rate": profile["back_and_forth_rate"],
        "first_difference_dft_high_fraction": profile["first_difference_dft"]["high_fraction"],
        "rotation_error_mean_rad": metrics["rotation_error_rad"]["mean"],
        "gripper_abs_error_mean": metrics["absolute_gripper_error"]["mean"],
    }
    return float(choices[name])


def paired_aggregates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "physical_action_mse_active_10d", "eef_position_mse", "endpoint_position_error_m",
        "path_length_m", "linear_speed_mean_mps", "linear_speed_max_mps", "roughness_mean_m",
        "adjacent_cosine_median", "back_and_forth_rate", "first_difference_dft_high_fraction",
        "rotation_error_mean_rad", "gripper_abs_error_mean",
    )
    lookup = {(int(row["seed"]), str(row["checkpoint"])): row for row in records}
    per_seed = []
    for seed in SEEDS:
        base = lookup[(seed, "BASE")]
        h2 = lookup[(seed, "H2@2000")]
        metrics = {}
        for field in fields:
            base_value = _metric_value(base, field)
            h2_value = _metric_value(h2, field)
            metrics[field] = {
                "base": base_value,
                "h2": h2_value,
                "h2_minus_base": h2_value - base_value,
                "h2_over_base": None if abs(base_value) <= 1e-15 else h2_value / base_value,
            }
        per_seed.append({"seed": seed, "metrics": metrics})
    aggregate = {}
    for field in fields:
        base_values = [_metric_value(lookup[(seed, "BASE")], field) for seed in SEEDS]
        h2_values = [_metric_value(lookup[(seed, "H2@2000")], field) for seed in SEEDS]
        deltas = [right - left for left, right in zip(base_values, h2_values)]
        ratios = [right / left for left, right in zip(base_values, h2_values) if abs(left) > 1e-15]
        aggregate[field] = {
            "base": _summary(base_values), "h2": _summary(h2_values),
            "h2_minus_base": _summary(deltas),
            "h2_over_base": None if not ratios else _summary(ratios),
        }
    return {
        "per_seed": per_seed,
        "aggregate": aggregate,
        "mse_improved_count_over_3": sum(
            _metric_value(lookup[(seed, "H2@2000")], "physical_action_mse_active_10d")
            < _metric_value(lookup[(seed, "BASE")], "physical_action_mse_active_10d")
            for seed in SEEDS
        ),
        "descriptive_only": True,
        "scientific_interpretation": "DEFERRED_TO_GPT_PI",
    }


def stage28_historical_context(stage28_result: Mapping[str, Any]) -> dict[str, Any]:
    paired = stage28_result.get("paired_aggregates", {}).get("paired", [])
    if len(paired) != 9:
        raise Stage29Error("accepted Stage28 paired record count drifted")
    base_values = [float(row["base_mse"]) for row in paired]
    h2_values = [float(row["h2_mse"]) for row in paired]
    return {
        "result_sha256": STAGE28_RESULT_SHA256,
        "artifact_seal_sha256": STAGE28_ARTIFACT_SEAL_SHA256,
        "base_mse_mean": mean(base_values),
        "h2_mse_mean": mean(h2_values),
        "geometric_mean_ratio": float(stage28_result["paired_aggregates"]["geometric_mean_ratio"]),
        "safety_gate": "REJECT_TO_REJECT_9_OF_9",
        "rerun": False,
    }


def _load_model(checkpoint: Path, device: str, torch: Any) -> Any:
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: F401
    from transformers import AutoModel

    model = AutoModel.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=True)
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    if int(model.action_head.num_inference_timesteps) != SAMPLER_STEPS:
        raise Stage29Error("checkpoint sampler step count is not the stock four")
    if int(model.action_head.config.action_horizon) != HORIZON or int(model.action_head.action_dim) != UNIVERSAL_DIM:
        raise Stage29Error("checkpoint action shape contract drifted")
    return model


def execute(args: argparse.Namespace) -> dict[str, Any]:
    preflight = run_preflight(args)
    if args.device != "cuda:0":
        raise Stage29Error("Stage29 is frozen to cuda:0")
    import numpy as np
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Stage29Error("Stage29 requires exactly one visible CUDA GPU")
    if "H200" not in torch.cuda.get_device_name(0):
        raise Stage29Error(f"Stage29 requires H200, observed {torch.cuda.get_device_name(0)}")
    from gr00t.data.types import MessageType, VLAStepData
    from scripts import stage26_h1_c1_fixed_noise_local_flow as stage26
    from scripts import stage23_phase_b_attribution_audit as stage23

    snapshot_identity, snapshot_path = stage26._verify_snapshot_identity(args.snapshot_root)
    snapshot = stage26._load_snapshot(snapshot_path)
    cosmos_snapshot = Path(preflight["identities"]["cosmos"]["snapshot"])
    processor, embodiment, modality = stage26._build_runtime_processor(
        groot_source=args.groot_source,
        base_checkpoint=args.base_model_path,
        dataset_dir=args.dataset_dir,
        modality_path=args.modality_config,
        cosmos_snapshot=cosmos_snapshot,
    )
    prepared = stage26._prepare_official_sample(
        processor=processor, embodiment=embodiment, modality_configs=modality, dataset_dir=args.dataset_dir,
    )
    source_step = prepared["step_data"]
    inference_step = VLAStepData(
        images=source_step.images, states=source_step.states, actions={}, text=source_step.text, embodiment=embodiment,
    )
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": inference_step}])
    inference_batch = processor.collator([processed])
    model_inputs = inference_batch.get("inputs", {})
    if "action" in model_inputs or "action_mask" in model_inputs:
        raise Stage29Error("official inference conditioning unexpectedly contains ground-truth action")
    conditioning = _conditioning_manifest(inference_batch)
    expert = stage26._decode_physical_action(
        processor, embodiment, prepared["expert_action_cpu"].numpy(), snapshot,
    )
    snapshot_expert = np.concatenate(
        [snapshot.future_desired_eef_9d[None, :, :], snapshot.future_gripper.reshape(1, HORIZON, 1)], axis=-1,
    ).astype(np.float32)
    if not np.allclose(expert, snapshot_expert, rtol=1e-5, atol=1e-6):
        raise Stage29Error("official decoded TINY1 expert differs from sealed Stage22 expert snapshot")

    h2_checkpoint = args.training_output_root / "h2_training" / "checkpoint-2000"
    models = {
        "BASE": _load_model(args.base_model_path, args.device, torch),
        "H2@2000": _load_model(h2_checkpoint, args.device, torch),
    }
    state_path = args.output_root / "stage29_c2_execution_state.json"
    partial_path = args.output_root / "stage29_c2_partial_records.json"
    execution_state: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "READY_AFTER_MODEL_CONSTRUCTION_BEFORE_FIRST_SAMPLER_CALL",
        "scientific_execution_begun": False,
        "completed_sampler_calls": 0,
        "maximum_sampler_calls": MAX_SAMPLER_CALLS,
        "pbs_job_id": args.pbs_job_id,
        "updated_at_utc": _utc_now(),
    }
    _write_json_atomic(state_path, execution_state)
    records: list[dict[str, Any]] = []
    paths = {"BASE": args.base_model_path, "H2@2000": h2_checkpoint}
    expected_noise: dict[int, str] = {}
    for seed in SEEDS:
        for label in CHECKPOINTS:
            execution_state.update({
                "status": "SCIENTIFIC_EXECUTION_BEGUN",
                "scientific_execution_begun": True,
                "current_seed": seed,
                "current_checkpoint": label,
                "updated_at_utc": _utc_now(),
            })
            _write_json_atomic(state_path, execution_state)
            rng = _reset_rng(seed, torch, np)
            call_inputs = copy.deepcopy(inference_batch)
            start = time.perf_counter()
            with _capture_stock_initial_noise(torch) as noise_calls:
                with torch.inference_mode():
                    prediction = models[label].get_action(**call_inputs)
            torch.cuda.synchronize()
            latency = time.perf_counter() - start
            noise = _validate_noise_capture(noise_calls, models[label])
            noise_tensor = noise.pop("tensor_float32")
            if seed in expected_noise and noise["raw_sha256"] != expected_noise[seed]:
                raise Stage29Error(f"paired initial-noise mismatch immediately after {label} seed {seed}")
            expected_noise.setdefault(seed, noise["raw_sha256"])
            raw = prediction["action_pred"].detach().float().contiguous().cpu().numpy().astype(np.float32, copy=False)
            if raw.shape != (1, HORIZON, UNIVERSAL_DIM) or not np.isfinite(raw).all():
                raise Stage29Error(f"invalid official sampler output: {raw.shape}")
            decoded = stage26._decode_physical_action(processor, embodiment, raw, snapshot)
            artifact = args.output_root / f"seed_{seed}_{'base' if label == 'BASE' else 'h2_2000'}_sampler_output.npz"
            np.savez_compressed(
                artifact,
                initial_noise_float32=noise_tensor.numpy().astype(np.float32, copy=False),
                raw_universal_action=raw,
                decoded_physical_action=decoded,
                expert_physical_action=expert,
            )
            row = {
                "checkpoint": label,
                "checkpoint_path": str(paths[label]),
                "seed": seed,
                "sampler_call_index": len(records) + 1,
                "conditioning_sha256": conditioning["sha256"],
                "rng_before_call": rng,
                "initial_noise": noise,
                "sampler": {
                    "api": "Gr00tN1d7.get_action",
                    "steps": SAMPLER_STEPS,
                    "dt": 0.25,
                    "continuous_time_grid": [0.0, 0.25, 0.5, 0.75],
                    "discrete_time_grid": [0, 250, 500, 750],
                    "rtc": False,
                    "guidance": False,
                    "sampler_clipping": False,
                },
                "raw_universal_action": {
                    "shape": list(raw.shape), "dtype": str(raw.dtype),
                    "sha256": _sha256_bytes(np.ascontiguousarray(raw).tobytes()), "finite": True,
                },
                "decoded_physical_action": {
                    "shape": list(decoded.shape), "dtype": str(decoded.dtype),
                    "sha256": _sha256_bytes(np.ascontiguousarray(decoded).tobytes()), "finite": True,
                },
                "artifact": {"path": str(artifact), "bytes": artifact.stat().st_size, "sha256": _sha256_file(artifact)},
                "latency_s": latency,
                "metrics": _trajectory_metrics(decoded, expert, stage23, stage26, processor, embodiment, snapshot, paths[label], label),
            }
            records.append(row)
            execution_state.update({
                "status": "SCIENTIFIC_EXECUTION_IN_PROGRESS",
                "completed_sampler_calls": len(records),
                "last_completed_seed": seed,
                "last_completed_checkpoint": label,
                "updated_at_utc": _utc_now(),
            })
            _write_json_atomic(partial_path, {
                "schema": SCHEMA, "status": "PARTIAL_EVIDENCE", "record_count": len(records), "records": records,
            })
            _write_json_atomic(state_path, execution_state)

    if len(records) != MAX_SAMPLER_CALLS:
        raise Stage29Error(f"sampler call count drifted: {len(records)}")
    paired_noise = validate_paired_noise(records)
    if len({row["conditioning_sha256"] for row in records}) != 1:
        raise Stage29Error("conditioning hash changed across sampler calls")
    stage28_result = _read_json(args.stage28_result)
    historical = stage28_historical_context(stage28_result)
    result = {
        "schema": SCHEMA,
        "status": "PASS_EXECUTION_COMPLETE",
        "stage": "Stage29 C2 official free-running sampler transfer gate",
        "pi_gate": PI_GATE_ID,
        "project_sha": args.expected_project_sha,
        "pbs": {"job_id": args.pbs_job_id, "project": args.pbs_project, "gpu": torch.cuda.get_device_name(0)},
        "preflight": preflight,
        "conditioning": conditioning,
        "expert_physical_action": {
            "shape": list(expert.shape), "sha256": _sha256_bytes(np.ascontiguousarray(expert).tobytes()),
            "metrics": stage23._profile(expert[0].tolist(), ACTION_PERIOD_S), "finite": True,
        },
        "paired_initial_noise": paired_noise,
        "records": records,
        "paired_aggregates": paired_aggregates(records),
        "stage28_teacher_forced_historical_context": historical,
        "execution": {
            "scientific_execution_begun": True,
            "sampler_calls": len(records),
            "base_sampler_calls": 3,
            "h2_sampler_calls": 3,
            "maximum_sampler_calls": MAX_SAMPLER_CALLS,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "scheduler_step": False,
            "resume": False,
            "checkpoint_write": False,
            "o1_rerun": False,
            "stage28_rerun": False,
            "wbc": False,
            "hardware": False,
            "validation_e": False,
            "holdout_f": False,
        },
        "all_finite": all(row["metrics"]["all_finite"] for row in records),
        "scientific_contract_changed": False,
        "scientific_interpretation": "DEFERRED_TO_GPT_PI",
        "claim_boundary": "single locked TINY1 training sample; paired descriptive free-running transfer evidence only",
        "next_gate": NEXT_GATE,
    }
    result_path = args.output_root / "stage29_c2_free_running_result.json"
    _write_json_atomic(result_path, result)
    _write_json_atomic(args.output_root / "stage29_c2_raw_hash_manifest.json", {
        "schema": SCHEMA,
        "records": [
            {
                "checkpoint": row["checkpoint"], "seed": row["seed"],
                "initial_noise_raw_sha256": row["initial_noise"]["raw_sha256"],
                "raw_universal_sha256": row["raw_universal_action"]["sha256"],
                "decoded_physical_sha256": row["decoded_physical_action"]["sha256"],
                "artifact_sha256": row["artifact"]["sha256"],
            }
            for row in records
        ],
    })
    execution_state.update({
        "status": "PASS_EXECUTION_COMPLETE", "completed_sampler_calls": len(records), "updated_at_utc": _utc_now(),
    })
    _write_json_atomic(state_path, execution_state)
    return result


def seal_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    result = _read_json(args.output_root / "stage29_c2_free_running_result.json")
    if result.get("status") != "PASS_EXECUTION_COMPLETE" or result.get("next_gate") != NEXT_GATE:
        raise Stage29Error("cannot seal incomplete Stage29 result")
    paths = sorted(path for path in args.output_root.iterdir() if path.is_file() and path.name != "stage29_c2_artifact_seal.json")
    paths.extend((args.pbs_log, Path(__file__).resolve(), args.pbs_template, args.source_audit_report))
    records = []
    seen = set()
    for path in paths:
        resolved = path.resolve(strict=True)
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        records.append({"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": _sha256_file(resolved)})
    seal = {
        "schema": "STAGE29_C2_ARTIFACT_SEAL_V1",
        "status": "PASS",
        "created_at_utc": _utc_now(),
        "project_sha": result["project_sha"],
        "pbs_job_id": result["pbs"]["job_id"],
        "artifact_count": len(records),
        "artifacts": records,
        "scientific_contract_changed": False,
        "next_gate": NEXT_GATE,
    }
    path = args.output_root / "stage29_c2_artifact_seal.json"
    _write_json_atomic(path, seal)
    return {**seal, "path": str(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "preflight", "execute", "seal"), default="plan")
    parser.add_argument("--project-repo", type=Path)
    parser.add_argument("--expected-project-sha")
    parser.add_argument("--base-model-path", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--groot-source", type=Path)
    parser.add_argument("--cosmos-cache-root", type=Path)
    parser.add_argument("--modality-config", type=Path)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--training-output-root", type=Path)
    parser.add_argument("--stage28-artifact-seal", type=Path)
    parser.add_argument("--stage28-result", type=Path)
    parser.add_argument("--source-audit-report", type=Path)
    parser.add_argument("--pbs-template", type=Path)
    parser.add_argument("--expected-runner-sha256")
    parser.add_argument("--expected-pbs-sha256")
    parser.add_argument("--expected-source-audit-sha256")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--authorization-token")
    parser.add_argument("--pbs-job-id", default="PREFLIGHT_NO_PBS")
    parser.add_argument("--pbs-project", default=EXPECTED_PROJECT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pbs-log", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "plan":
        assert_protocol_static()
        print(json.dumps(protocol_plan(), indent=2, sort_keys=True))
        return 0
    required = (
        "project_repo", "expected_project_sha", "base_model_path", "dataset_dir", "groot_source",
        "cosmos_cache_root", "modality_config", "snapshot_root", "training_output_root",
        "stage28_artifact_seal", "stage28_result", "source_audit_report", "pbs_template",
        "expected_runner_sha256", "expected_pbs_sha256", "expected_source_audit_sha256",
        "output_root", "authorization_token",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit("missing required arguments: " + ", ".join(missing))
    try:
        if args.mode == "preflight":
            result = run_preflight(args)
            _write_json_atomic(args.output_root / "stage29_c2_preflight.json", result)
        elif args.mode == "execute":
            result = execute(args)
        else:
            if args.pbs_log is None:
                raise Stage29Error("--pbs-log is required in seal mode")
            result = seal_artifacts(args)
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "status": "STOP_OR_BLOCKED",
            "error_type": type(exc).__name__,
            "error_summary": str(exc).splitlines()[0][:500],
            "scientific_contract_changed": False,
            "next_gate": NEXT_GATE,
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
