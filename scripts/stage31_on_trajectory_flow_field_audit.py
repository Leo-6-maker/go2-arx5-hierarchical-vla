#!/usr/bin/env python3
"""Stage31 on-trajectory rectified-flow audit for the stock four-step sampler.

The preflight path is standard-library-only.  The authorized execute path
reproduces the six sealed Stage29 sampler calls and observes the sampler with
read-only module hooks.  It does not replace sampler math, train, write a
checkpoint, change the action ABI, or read validation/holdout data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import time
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "STAGE31_ON_TRAJECTORY_FLOW_FIELD_AUDIT_V1"
PI_GATE_ID = "PI_GATE_STATE_0011"
NEXT_GATE = "GPT_PI_REVIEW_AFTER_STAGE31_ON_TRAJECTORY_FLOW_RESULT"
STAGE30_ACCEPTED_SHA = "8ba410e7185e6508a59d7022fbba28da96941a40"
EXPECTED_PROJECT = "CFP03-CF-004"
EXPECTED_GROOT_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
STAGE29_EXECUTION_SHA = "07290d042c24996b93d046edf514f91a786f52ba"
STAGE29_CLOSURE_SHA = "ae10dbd44442bafd02bbe927f95ae271526e4970"
STAGE29_SEAL_SHA256 = "9c9c25e9106c4a3538d160e354a013fdf6fd4a5300b519ab8de6d98a8865247b"
STAGE29_RESULT_SHA256 = "1e800b69604a97f16cfc704b591ff2c710c14430282cbd1b0cc6e6da0c5dfd48"
STAGE30_SEAL_SHA256 = "2799b8ba36446e61a9487fd429d487f23f98420352cb8e28c0b3afe88cb7b681"
H2_TREE_SHA256 = "3106f9b5993b390438b62594d044781d11b393452a49f80b5c8e9e910606d0ba"
H2_SEAL_SHA256 = "1349896f7b31a930764dbd9d849c5368e48372d0dc6273449f42f19f82a69a82"
TINY1_TREE_SHA256 = "541bf4a1b9ea459637525ef11fceea5c7f192c0902f6013f0331a2a668038157"
HORIZON = 40
UNIVERSAL_DIM = 132
ACTIVE_DIM = 10
SAMPLER_STEPS = 4
DT = 0.25
SEEDS = (2223, 2224, 2225)
CHECKPOINTS = ("BASE", "H2@2000")
MAX_SAMPLER_CALLS = 6
TIMESTEP_IDS = (0, 250, 500, 750)
CONTINUOUS_TIMES = (0.0, 0.25, 0.5, 0.75)
TRACE_ARRAY_KEYS = {
    "x_trajectory_float32",
    "v_trajectory_float32",
    "timestep_ids",
}


class Stage31Error(RuntimeError):
    """Fail-closed Stage31 contract error."""


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
        raise Stage31Error(f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: Any) -> str:
    import numpy as np

    return _sha256_bytes(np.ascontiguousarray(array).tobytes())


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


def _git_bytes(repo: Path, revision: str, relative_path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{relative_path}"])


def _summary(values: Iterable[float]) -> dict[str, Any]:
    rows = [float(value) for value in values]
    if not rows:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(rows),
        "min": min(rows),
        "median": median(rows),
        "mean": mean(rows),
        "max": max(rows),
    }


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def protocol_plan() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "pi_gate": PI_GATE_ID,
        "stage": "Stage31 on-trajectory rectified-flow / sampler-state audit",
        "accepted_base_sha": STAGE30_ACCEPTED_SHA,
        "sampler": {
            "api": "Gr00tN1d7.get_action -> Gr00tN1d7ActionHead.get_action",
            "algorithm": "unchanged stock explicit Euler flow integration",
            "steps": SAMPLER_STEPS,
            "dt": DT,
            "continuous_times": list(CONTINUOUS_TIMES),
            "timestep_ids": list(TIMESTEP_IDS),
            "instrumentation": "read-only action_encoder pre-hook and action_decoder forward hook",
            "hook_replaces_tensor_or_output": False,
        },
        "experiment": {
            "dataset": "same locked TINY1 training sample",
            "seeds": list(SEEDS),
            "checkpoints": list(CHECKPOINTS),
            "maximum_sampler_calls": MAX_SAMPLER_CALLS,
            "reproduce_stage29_noise_and_final_output": True,
            "first_call_mismatch_stops": True,
            "shapes": {"universal": [1, HORIZON, UNIVERSAL_DIM], "active": [1, HORIZON, ACTIVE_DIM]},
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
            "sampler_step_change": False,
            "alternative_sampler": False,
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
        raise Stage31Error("Stage31 schema/gate drifted")
    sampler = candidate.get("sampler", {})
    if sampler.get("steps") != 4 or sampler.get("dt") != 0.25:
        raise Stage31Error("stock four-step sampler contract drifted")
    if sampler.get("timestep_ids") != list(TIMESTEP_IDS):
        raise Stage31Error("sampler timestep IDs drifted")
    experiment = candidate.get("experiment", {})
    if experiment.get("seeds") != list(SEEDS) or experiment.get("checkpoints") != list(CHECKPOINTS):
        raise Stage31Error("Stage31 seed/checkpoint contract drifted")
    if experiment.get("maximum_sampler_calls") != MAX_SAMPLER_CALLS:
        raise Stage31Error("Stage31 sampler-call budget drifted")
    forbidden = (
        "training", "backward", "optimizer_step", "scheduler_step", "resume",
        "checkpoint_write", "sampler_step_change", "alternative_sampler", "wbc",
        "hardware", "validation_e", "holdout_f",
    )
    execution = candidate.get("execution", {})
    if any(execution.get(key) is not False for key in forbidden):
        raise Stage31Error("Stage31 forbidden execution boundary drifted")


def _require_project_identity(project_repo: Path, expected_sha: str) -> str:
    head = _git(project_repo, "rev-parse", "HEAD")
    if head != expected_sha:
        raise Stage31Error(f"project HEAD mismatch: {head} != {expected_sha}")
    if _git(project_repo, "status", "--short"):
        raise Stage31Error("project worktree is dirty")
    ancestry = subprocess.run(
        ["git", "-C", str(project_repo), "merge-base", "--is-ancestor", STAGE30_ACCEPTED_SHA, head],
        check=False,
    )
    if ancestry.returncode != 0:
        raise Stage31Error("Stage31 HEAD does not descend from accepted Stage30 HEAD")
    return head


def _ensure_external_output(output_root: Path, immutable_roots: Iterable[Path]) -> None:
    output = output_root.resolve()
    for immutable in immutable_roots:
        source = immutable.resolve()
        if output == source or source in output.parents or output in source.parents:
            raise Stage31Error(f"output overlaps immutable input: {output} vs {source}")
    output.mkdir(parents=True, exist_ok=True)


def _verify_seal_entries(
    seal_path: Path,
    expected_sha256: str,
    project_repo: Path,
    *,
    expected_count: int,
    label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observed_seal_sha = _sha256_file(seal_path)
    if observed_seal_sha != expected_sha256:
        raise Stage31Error(f"{label} seal SHA mismatch: {observed_seal_sha}")
    seal = _read_json(seal_path)
    entries = seal.get("artifacts", [])
    if seal.get("status") != "PASS" or seal.get("artifact_count") != expected_count or len(entries) != expected_count:
        raise Stage31Error(f"{label} seal contract drifted")
    verified = []
    for entry in entries:
        location = str(entry["path"])
        if location.startswith("git:"):
            _, revision, relative_path = location.split(":", 2)
            content = _git_bytes(project_repo, revision, relative_path)
            observed_sha = _sha256_bytes(content)
            observed_bytes = len(content)
        else:
            path = Path(location)
            observed_sha = _sha256_file(path)
            observed_bytes = path.stat().st_size
        if observed_sha != entry["sha256"] or observed_bytes != int(entry["bytes"]):
            raise Stage31Error(f"{label} sealed artifact drifted: {location}")
        verified.append({"path": location, "bytes": observed_bytes, "sha256": observed_sha})
    return seal, verified


def _load_stage29_history(
    stage29_root: Path,
    stage29_result_path: Path,
    stage29_seal: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[tuple[int, str], dict[str, Any]], dict[tuple[int, str], dict[str, Any]]]:
    import numpy as np

    if _sha256_file(stage29_result_path) != STAGE29_RESULT_SHA256:
        raise Stage31Error("Stage29 result SHA mismatch")
    result = _read_json(stage29_result_path)
    records = result.get("records", [])
    lookup = {(int(row["seed"]), str(row["checkpoint"])): row for row in records}
    expected_keys = {(seed, checkpoint) for seed in SEEDS for checkpoint in CHECKPOINTS}
    if len(records) != MAX_SAMPLER_CALLS or set(lookup) != expected_keys:
        raise Stage31Error("Stage29 result does not contain the exact six frozen calls")
    sealed = {str(entry["path"]): entry for entry in stage29_seal["artifacts"]}
    arrays: dict[tuple[int, str], dict[str, Any]] = {}
    manifest: dict[tuple[int, str], dict[str, Any]] = {}
    for key, row in lookup.items():
        path = Path(row["artifact"]["path"]).resolve(strict=True)
        if path.parent != stage29_root.resolve(strict=True):
            raise Stage31Error(f"Stage29 NPZ escaped frozen root: {path}")
        file_sha = _sha256_file(path)
        sealed_entry = sealed.get(str(path))
        if file_sha != row["artifact"]["sha256"] or sealed_entry is None or sealed_entry["sha256"] != file_sha:
            raise Stage31Error(f"Stage29 NPZ hash/seal mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            expected_array_keys = {
                "initial_noise_float32", "raw_universal_action",
                "decoded_physical_action", "expert_physical_action",
            }
            if set(archive.files) != expected_array_keys:
                raise Stage31Error(f"Stage29 NPZ key drift: {path}")
            payload = {name: np.array(archive[name], copy=True) for name in archive.files}
        shapes = {
            "initial_noise_float32": (1, HORIZON, UNIVERSAL_DIM),
            "raw_universal_action": (1, HORIZON, UNIVERSAL_DIM),
            "decoded_physical_action": (1, HORIZON, ACTIVE_DIM),
            "expert_physical_action": (1, HORIZON, ACTIVE_DIM),
        }
        for name, array in payload.items():
            if array.shape != shapes[name] or array.dtype != np.float32 or not np.isfinite(array).all():
                raise Stage31Error(f"Stage29 NPZ array drift: {path}:{name}")
        if _array_sha256(payload["initial_noise_float32"]) != row["initial_noise"]["float32_sha256"]:
            raise Stage31Error(f"Stage29 initial-noise array hash drift: {path}")
        if _array_sha256(payload["raw_universal_action"]) != row["raw_universal_action"]["sha256"]:
            raise Stage31Error(f"Stage29 raw-output array hash drift: {path}")
        arrays[key] = payload
        manifest[key] = {
            "seed": key[0],
            "checkpoint": key[1],
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha,
            "initial_noise_raw_sha256": row["initial_noise"]["raw_sha256"],
            "initial_noise_float32_sha256": row["initial_noise"]["float32_sha256"],
            "final_raw_universal_sha256": row["raw_universal_action"]["sha256"],
            "conditioning_sha256": row["conditioning_sha256"],
        }
    for seed in SEEDS:
        if not np.array_equal(
            arrays[(seed, "BASE")]["initial_noise_float32"],
            arrays[(seed, "H2@2000")]["initial_noise_float32"],
        ):
            raise Stage31Error(f"Stage29 paired initial noise drifted for seed {seed}")
    conditioning_hashes = {entry["conditioning_sha256"] for entry in manifest.values()}
    if conditioning_hashes != {result["conditioning"]["sha256"]}:
        raise Stage31Error("Stage29 conditioning hashes are inconsistent")
    return result, arrays, manifest


def _retention_metrics(initial: Any, final: Any) -> dict[str, Any]:
    import numpy as np

    left = np.asarray(initial, dtype=np.float64).reshape(-1)
    right = np.asarray(final, dtype=np.float64).reshape(-1)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    cosine = None if left_norm <= 0.0 or right_norm <= 0.0 else float(np.dot(left, right) / (left_norm * right_norm))
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    centered_denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    centered_correlation = (
        None if centered_denominator <= 0.0
        else float(np.dot(left_centered, right_centered) / centered_denominator)
    )
    denominator = float(np.dot(left, left))
    coefficient = None if denominator <= 0.0 else float(np.dot(left, right) / denominator)
    if coefficient is None or float(np.dot(right, right)) <= 0.0:
        projected_output_energy_fraction = None
    else:
        projected = coefficient * left
        projected_output_energy_fraction = float(np.dot(projected, projected) / np.dot(right, right))
    return {
        "cosine": cosine,
        "centered_pearson": centered_correlation,
        "projection_coefficient_final_on_initial": coefficient,
        "projected_output_energy_fraction": projected_output_energy_fraction,
    }


def _zero_forward_retention(arrays: Mapping[tuple[int, str], Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for seed in SEEDS:
        for checkpoint in CHECKPOINTS:
            payload = arrays[(seed, checkpoint)]
            initial = payload["initial_noise_float32"][0]
            final = payload["raw_universal_action"][0]
            rows.append({
                "seed": seed,
                "checkpoint": checkpoint,
                "active10": _retention_metrics(initial[:, :ACTIVE_DIM], final[:, :ACTIVE_DIM]),
                "xyz": _retention_metrics(initial[:, :3], final[:, :3]),
            })
    return {"records": rows, "descriptive_only": True, "new_sampler_calls": 0}


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.authorization_token != PI_GATE_ID:
        raise Stage31Error("PI gate token mismatch")
    if args.pbs_project != EXPECTED_PROJECT:
        raise Stage31Error("PBS project mismatch")
    assert_protocol_static()
    project_head = _require_project_identity(args.project_repo, args.expected_project_sha)
    runner_sha = _sha256_file(Path(__file__).resolve())
    pbs_sha = _sha256_file(args.pbs_template)
    audit_sha = _sha256_file(args.source_audit_report)
    if runner_sha != args.expected_runner_sha256:
        raise Stage31Error(f"runner SHA mismatch: {runner_sha}")
    if pbs_sha != args.expected_pbs_sha256:
        raise Stage31Error(f"PBS template SHA mismatch: {pbs_sha}")
    if audit_sha != args.expected_source_audit_sha256:
        raise Stage31Error(f"source-audit report SHA mismatch: {audit_sha}")

    stage29_seal, stage29_verified = _verify_seal_entries(
        args.stage29_seal, STAGE29_SEAL_SHA256, args.project_repo,
        expected_count=16, label="Stage29",
    )
    stage30_seal, stage30_verified = _verify_seal_entries(
        args.stage30_seal, STAGE30_SEAL_SHA256, args.project_repo,
        expected_count=18, label="Stage30",
    )
    if (
        stage29_seal.get("execution_project_sha") != STAGE29_EXECUTION_SHA
        or stage29_seal.get("posthoc_repair_sha") != STAGE29_CLOSURE_SHA
    ):
        raise Stage31Error("Stage29 execution/closure source binding drifted")
    if stage30_seal.get("project_sha") != STAGE30_ACCEPTED_SHA:
        raise Stage31Error("Stage30 accepted source binding drifted")
    stage29_result, _, stage29_manifest = _load_stage29_history(
        args.stage29_root, args.stage29_result, stage29_seal,
    )

    from scripts import stage28_h2_fresh_base_2000_fixed_warmup10 as stage28
    from scripts import stage29_c2_free_running_transfer as stage29

    base = stage28._verify_base_checkpoint(args.base_model_path)
    dataset = stage28._verify_tiny_dataset(args.dataset_dir)
    if dataset.get("tree_sha256") != TINY1_TREE_SHA256:
        raise Stage31Error("TINY1 frozen tree mismatch")
    modality_sha = stage28._verify_modality(args.modality_config)
    groot = stage29.audit_groot_source(args.groot_source)
    if groot.get("git_sha") != EXPECTED_GROOT_SHA:
        raise Stage31Error("GR00T source identity mismatch")
    cosmos = stage28._verify_cosmos_cache(args.cosmos_cache_root)
    training = stage28._verify_final_checkpoint_artifacts(
        args.training_output_root,
        expected_checkpoint_tree_sha256=H2_TREE_SHA256,
        expected_seal_sha256=H2_SEAL_SHA256,
    )
    stage26 = stage28._load_stage26_module()
    snapshot, _ = stage26._verify_snapshot_identity(args.snapshot_root)
    h2_checkpoint = args.training_output_root / "h2_training" / "checkpoint-2000"
    if not h2_checkpoint.is_dir():
        raise Stage31Error("sealed H2@2000 checkpoint is unavailable")
    _ensure_external_output(
        args.output_root,
        (
            args.base_model_path, args.dataset_dir, args.training_output_root,
            args.stage29_root, args.stage30_root,
        ),
    )
    return {
        "schema": SCHEMA,
        "status": "PASS_PREFLIGHT_NO_MODEL",
        "timestamp_utc": _utc_now(),
        "project": {"head": project_head, "clean": True, "stage30_ancestor": STAGE30_ACCEPTED_SHA},
        "identities": {
            "base": base,
            "dataset": dataset,
            "modality_sha256": modality_sha,
            "groot": groot,
            "cosmos": cosmos,
            "h2_training_artifacts": training,
            "h2_checkpoint": str(h2_checkpoint),
            "snapshot": snapshot,
            "stage29_seal": {"path": str(args.stage29_seal), "sha256": STAGE29_SEAL_SHA256, "verified_entries": len(stage29_verified)},
            "stage29_result": {"path": str(args.stage29_result), "sha256": STAGE29_RESULT_SHA256},
            "stage29_calls": [_jsonable(stage29_manifest[key]) for key in sorted(stage29_manifest)],
            "stage30_seal": {"path": str(args.stage30_seal), "sha256": STAGE30_SEAL_SHA256, "verified_entries": len(stage30_verified)},
            "runner_sha256": runner_sha,
            "pbs_template_sha256": pbs_sha,
            "source_audit_report_sha256": audit_sha,
        },
        "historical_conditioning_sha256": stage29_result["conditioning"]["sha256"],
        "execution": {"model_constructed": False, "model_forward": False, "sampler_calls": 0},
        "scientific_contract_changed": False,
    }


class StockSamplerTrace(AbstractContextManager["StockSamplerTrace"]):
    """Observe stock action-state and decoder tensors without replacing outputs."""

    def __init__(self, action_head: Any) -> None:
        self.action_head = action_head
        self.states: list[Any] = []
        self.timesteps: list[Any] = []
        self.decoder_outputs: list[Any] = []
        self.handles: list[Any] = []

    def _on_action_encoder(self, module: Any, positional: Sequence[Any]) -> None:
        del module
        if len(positional) < 2:
            raise Stage31Error("action_encoder hook did not receive action/timestep tensors")
        self.states.append(positional[0])
        self.timesteps.append(positional[1])
        return None

    def _on_action_decoder(self, module: Any, positional: Sequence[Any], output: Any) -> None:
        del module, positional
        self.decoder_outputs.append(output)
        return None

    def __enter__(self) -> "StockSamplerTrace":
        self.handles = [
            self.action_head.action_encoder.register_forward_pre_hook(self._on_action_encoder),
            self.action_head.action_decoder.register_forward_hook(self._on_action_decoder),
        ]
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_type, exc_value, traceback
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        return None


def _finalize_trace(observer: StockSamplerTrace, final_actions: Any, torch: Any) -> dict[str, Any]:
    if len(observer.states) != SAMPLER_STEPS or len(observer.decoder_outputs) != SAMPLER_STEPS:
        raise Stage31Error(
            f"stock trace count drifted: states={len(observer.states)} velocities={len(observer.decoder_outputs)}"
        )
    if len(observer.timesteps) != SAMPLER_STEPS:
        raise Stage31Error("stock trace timestep count drifted")
    velocities = [output[:, -HORIZON:] for output in observer.decoder_outputs]
    states = [*observer.states, final_actions]
    expected_shape = (1, HORIZON, UNIVERSAL_DIM)
    for label, values in (("x", states), ("v", velocities)):
        for index, tensor in enumerate(values):
            if tuple(int(value) for value in tensor.shape) != expected_shape:
                raise Stage31Error(f"trace {label}_{index} shape drifted: {tuple(tensor.shape)}")
            if not bool(torch.isfinite(tensor).all().item()):
                raise Stage31Error(f"trace {label}_{index} is non-finite")
    timestep_ids = []
    for tensor in observer.timesteps:
        values = tensor.detach().contiguous().cpu().tolist()
        if len(values) != 1:
            raise Stage31Error(f"trace timestep batch drifted: {values}")
        timestep_ids.append(int(values[0]))
    if timestep_ids != list(TIMESTEP_IDS):
        raise Stage31Error(f"trace timestep IDs drifted: {timestep_ids}")
    recurrence = []
    for step, velocity in enumerate(velocities):
        reconstructed = states[step] + DT * velocity
        exact = bool(torch.equal(reconstructed, states[step + 1]))
        max_abs = float(
            torch.max(torch.abs(reconstructed.float() - states[step + 1].float())).item()
        )
        recurrence.append({"step": step, "exact_tensor_equality": exact, "max_abs_delta_float32": max_abs})
        if not exact:
            raise Stage31Error(f"captured tensors do not reproduce stock Euler update at step {step}")
    x = torch.stack([value.detach().float().contiguous().cpu() for value in states], dim=0).numpy()
    v = torch.stack([value.detach().float().contiguous().cpu() for value in velocities], dim=0).numpy()
    raw_records = {
        "x": [_tensor_metadata(value, torch) for value in states],
        "v": [_tensor_metadata(value, torch) for value in velocities],
    }
    return {
        "x_trajectory_float32": x,
        "v_trajectory_float32": v,
        "timestep_ids": timestep_ids,
        "tensor_records": raw_records,
        "euler_recurrence": recurrence,
    }


def _tensor_metadata(tensor: Any, torch: Any) -> dict[str, Any]:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    float32 = tensor.detach().float().contiguous().cpu().numpy().tobytes()
    return {
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "raw_sha256": _sha256_bytes(raw),
        "float32_sha256": _sha256_bytes(float32),
        "finite": bool(torch.isfinite(tensor).all().item()),
    }


def _scope_slices() -> dict[str, slice]:
    return {
        "active10": slice(0, 10),
        "xyz": slice(0, 3),
        "x": slice(0, 1),
        "y": slice(1, 2),
        "z": slice(2, 3),
    }


def _mse(value: Any) -> float:
    import numpy as np

    array = np.asarray(value, dtype=np.float64)
    return float(np.mean(np.square(array)))


def _alignment(predicted: Any, reference: Any) -> dict[str, Any]:
    import numpy as np

    left = np.asarray(predicted, dtype=np.float64).reshape(-1)
    right = np.asarray(reference, dtype=np.float64).reshape(-1)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    return {
        "cosine": None if left_norm <= 0.0 or right_norm <= 0.0 else float(np.dot(left, right) / (left_norm * right_norm)),
        "predicted_norm": left_norm,
        "reference_norm": right_norm,
        "predicted_over_reference_norm": None if right_norm <= 0.0 else left_norm / right_norm,
    }


def _frequency_energy(series: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(series, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != HORIZON or not np.isfinite(array).all():
        raise Stage31Error(f"invalid horizon series for DFT: {array.shape}")
    energy = np.square(np.abs(np.fft.fft(array, axis=0)))
    bins = np.arange(HORIZON)
    folded = np.minimum(bins, HORIZON - bins)
    high_cut = int(math.ceil(HORIZON / 4.0))
    low = float(np.sum(energy[folded < high_cut]))
    high = float(np.sum(energy[folded >= high_cut]))
    total = low + high
    return {
        "definition": "unscaled DFT over H40; low includes DC and folded bins <10; high folded bins >=10",
        "low_including_dc_energy": low,
        "high_energy": high,
        "total_energy": total,
        "low_fraction": None if total <= 0.0 else low / total,
        "high_fraction": None if total <= 0.0 else high / total,
        "high_over_low_energy": None if low <= 0.0 else high / low,
    }


def _occupancy(state: Any) -> float:
    import numpy as np

    return float(np.mean(np.abs(np.asarray(state, dtype=np.float64)) > 1.0))


def step_metrics(x: Any, x_next: Any, velocity: Any, target: Any, t_value: float) -> dict[str, Any]:
    import numpy as np

    current = np.asarray(x, dtype=np.float64)
    following = np.asarray(x_next, dtype=np.float64)
    predicted = np.asarray(velocity, dtype=np.float64)
    desired = np.asarray(target, dtype=np.float64)
    if current.shape != (HORIZON, UNIVERSAL_DIM) or following.shape != current.shape:
        raise Stage31Error("step state shape drifted")
    if predicted.shape != current.shape or desired.shape != current.shape:
        raise Stage31Error("step velocity/target shape drifted")
    reference_velocity = (desired - current) / (1.0 - t_value)
    reference_next = current + DT * reference_velocity
    scopes = {}
    for name, section in _scope_slices().items():
        state_error = current[:, section] - desired[:, section]
        next_error = following[:, section] - desired[:, section]
        velocity_error = predicted[:, section] - reference_velocity[:, section]
        before_mse = _mse(state_error)
        after_mse = _mse(next_error)
        scopes[name] = {
            "state_error_mse": before_mse,
            "velocity_reference_error_mse": _mse(velocity_error),
            "velocity_alignment": _alignment(predicted[:, section], reference_velocity[:, section]),
            "reference_next_deviation_mse": _mse(following[:, section] - reference_next[:, section]),
            "actual_error_contraction": None if before_mse <= 0.0 else after_mse / before_mse,
            "state_residual_frequency": _frequency_energy(state_error),
            "velocity_error_frequency": _frequency_energy(velocity_error),
            "raw_fraction_abs_gt_1": _occupancy(current[:, section]),
        }
    return {
        "continuous_time": t_value,
        "timestep_id": int(t_value * 1000),
        "scopes": scopes,
    }


def final_state_metrics(x_final: Any, target: Any) -> dict[str, Any]:
    import numpy as np

    state = np.asarray(x_final, dtype=np.float64)
    desired = np.asarray(target, dtype=np.float64)
    scopes = {}
    for name, section in _scope_slices().items():
        error = state[:, section] - desired[:, section]
        scopes[name] = {
            "state_error_mse": _mse(error),
            "state_residual_frequency": _frequency_energy(error),
            "raw_fraction_abs_gt_1": _occupancy(state[:, section]),
        }
    return {"state_index": 4, "scopes": scopes}


def trace_metrics(x_trajectory: Any, v_trajectory: Any, target: Any) -> dict[str, Any]:
    import numpy as np

    x = np.asarray(x_trajectory, dtype=np.float32)
    v = np.asarray(v_trajectory, dtype=np.float32)
    desired = np.asarray(target, dtype=np.float32)
    if x.shape != (5, 1, HORIZON, UNIVERSAL_DIM) or v.shape != (4, 1, HORIZON, UNIVERSAL_DIM):
        raise Stage31Error(f"trace metric shape drifted: x={x.shape} v={v.shape}")
    if desired.shape != (1, HORIZON, UNIVERSAL_DIM):
        raise Stage31Error(f"target shape drifted: {desired.shape}")
    steps = [
        {"step": index, **step_metrics(x[index, 0], x[index + 1, 0], v[index, 0], desired[0], CONTINUOUS_TIMES[index])}
        for index in range(SAMPLER_STEPS)
    ]
    return {"steps": steps, "final_x4": final_state_metrics(x[4, 0], desired[0]), "all_finite": True}


def assert_stage29_reproduction(reproduction: Mapping[str, Any], call_index: int) -> None:
    """Make every traced call, including the first, an immediate exact gate."""

    required = (
        "noise_raw_exact",
        "noise_float32_exact",
        "trace_x0_equals_captured_noise_raw",
        "final_raw_universal_exact",
        "conditioning_exact",
    )
    if reproduction.get("status") != "MATCH" or any(reproduction.get(key) is not True for key in required):
        raise Stage31Error(
            f"Stage29 reproduction mismatch after scientific call {call_index}; autonomous retry forbidden"
        )


def _cross_seed_decomposition(errors: Any) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(errors, dtype=np.float64)
    if values.shape[0] != len(SEEDS) or values.shape[1] != HORIZON:
        raise Stage31Error(f"cross-seed decomposition shape drifted: {values.shape}")
    common = np.mean(values, axis=0)
    deviations = values - common[None, ...]
    total = _mse(values)
    common_mse = _mse(common)
    seed_variance = _mse(deviations)
    return {
        "total_mse": total,
        "common_mode_mse": common_mse,
        "seed_variance_mse": seed_variance,
        "common_fraction": None if total <= 0.0 else common_mse / total,
        "seed_variance_fraction": None if total <= 0.0 else seed_variance / total,
        "identity_abs_delta": abs(total - (common_mse + seed_variance)),
    }


def aggregate_traces(records: Sequence[Mapping[str, Any]], target: Any) -> dict[str, Any]:
    import numpy as np

    lookup = {(int(row["seed"]), str(row["checkpoint"])): row for row in records}
    expected = {(seed, checkpoint) for seed in SEEDS for checkpoint in CHECKPOINTS}
    if set(lookup) != expected or len(records) != MAX_SAMPLER_CALLS:
        raise Stage31Error("aggregate requires the exact six Stage31 records")
    traces = {}
    for key, row in lookup.items():
        with np.load(row["trace_artifact"]["path"], allow_pickle=False) as archive:
            if set(archive.files) != TRACE_ARRAY_KEYS:
                raise Stage31Error(f"trace artifact keys drifted: {row['trace_artifact']['path']}")
            traces[key] = np.array(archive["x_trajectory_float32"], copy=True)
    desired = np.asarray(target, dtype=np.float64)[0]
    decomposition = {}
    for checkpoint in CHECKPOINTS:
        state_rows = []
        for state_index in range(SAMPLER_STEPS + 1):
            by_seed = np.stack([traces[(seed, checkpoint)][state_index, 0] - desired for seed in SEEDS], axis=0)
            scopes = {
                name: _cross_seed_decomposition(by_seed[..., section])
                for name, section in _scope_slices().items()
            }
            state_rows.append({"state_index": state_index, "scopes": scopes})
        decomposition[checkpoint] = state_rows

    comparisons = []
    for step in range(SAMPLER_STEPS):
        row = {"step": step, "continuous_time": CONTINUOUS_TIMES[step], "scopes": {}}
        for scope in _scope_slices():
            base_steps = [lookup[(seed, "BASE")]["metrics"]["steps"][step]["scopes"][scope] for seed in SEEDS]
            h2_steps = [lookup[(seed, "H2@2000")]["metrics"]["steps"][step]["scopes"][scope] for seed in SEEDS]
            fields = {}
            for field in (
                "state_error_mse", "velocity_reference_error_mse",
                "reference_next_deviation_mse", "actual_error_contraction",
                "raw_fraction_abs_gt_1",
            ):
                base_values = [float(item[field]) for item in base_steps]
                h2_values = [float(item[field]) for item in h2_steps]
                base_mean = mean(base_values)
                h2_mean = mean(h2_values)
                fields[field] = {
                    "base": _summary(base_values),
                    "h2": _summary(h2_values),
                    "h2_over_base_mean": None if abs(base_mean) <= 1e-15 else h2_mean / base_mean,
                }
            for frequency_name in ("state_residual_frequency", "velocity_error_frequency"):
                base_high = [float(item[frequency_name]["high_energy"]) for item in base_steps]
                h2_high = [float(item[frequency_name]["high_energy"]) for item in h2_steps]
                base_low = [float(item[frequency_name]["low_including_dc_energy"]) for item in base_steps]
                h2_low = [float(item[frequency_name]["low_including_dc_energy"]) for item in h2_steps]
                fields[frequency_name] = {
                    "base_high": _summary(base_high), "h2_high": _summary(h2_high),
                    "h2_over_base_high_mean": None if abs(mean(base_high)) <= 1e-15 else mean(h2_high) / mean(base_high),
                    "base_low": _summary(base_low), "h2_low": _summary(h2_low),
                    "h2_over_base_low_mean": None if abs(mean(base_low)) <= 1e-15 else mean(h2_low) / mean(base_low),
                }
            row["scopes"][scope] = fields
        comparisons.append(row)
    return {
        "cross_seed_state_residual": decomposition,
        "base_h2_step_comparison": comparisons,
        "descriptive_only": True,
        "scientific_interpretation": "DEFERRED_TO_GPT_PI",
    }


def _prepare_zero_forward_phase(
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
) -> tuple[Any, Any, Any, dict[str, Any], dict[tuple[int, str], dict[str, Any]], Any]:
    import numpy as np

    from gr00t.data.types import MessageType, VLAStepData
    from scripts import stage26_h1_c1_fixed_noise_local_flow as stage26
    from scripts import stage29_c2_free_running_transfer as stage29

    stage29_seal = _read_json(args.stage29_seal)
    stage29_result, historical_arrays, historical_manifest = _load_stage29_history(
        args.stage29_root, args.stage29_result, stage29_seal,
    )
    _, snapshot_path = stage26._verify_snapshot_identity(args.snapshot_root)
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
        processor=processor,
        embodiment=embodiment,
        modality_configs=modality,
        dataset_dir=args.dataset_dir,
    )
    target = prepared["expert_action_cpu"].numpy().astype(np.float32, copy=False)
    source_step = prepared["step_data"]
    inference_step = VLAStepData(
        images=source_step.images,
        states=source_step.states,
        actions={},
        text=source_step.text,
        embodiment=embodiment,
    )
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": inference_step}])
    inference_batch = processor.collator([processed])
    model_inputs = inference_batch.get("inputs", {})
    if "action" in model_inputs or "action_mask" in model_inputs:
        raise Stage31Error("inference conditioning unexpectedly contains ground-truth action")
    conditioning = stage29._conditioning_manifest(inference_batch)
    historical_conditioning = stage29_result["conditioning"]["sha256"]
    if conditioning["sha256"] != historical_conditioning:
        raise Stage31Error(
            f"locked conditioning reproduction failed before model construction: {conditioning['sha256']}"
        )
    target_record = {
        "shape": list(target.shape),
        "dtype": str(target.dtype),
        "sha256": _array_sha256(target),
        "active10_sha256": _array_sha256(target[..., :ACTIVE_DIM]),
        "xyz_sha256": _array_sha256(target[..., :3]),
        "finite": bool(np.isfinite(target).all()),
    }
    if target_record["shape"] != [1, HORIZON, UNIVERSAL_DIM] or not target_record["finite"]:
        raise Stage31Error("normalized universal target contract drifted")
    target_path = args.output_root / "stage31_locked_target.npz"
    np.savez_compressed(target_path, normalized_universal_target=target)
    planned = [
        {
            "seed": seed,
            "checkpoint": checkpoint,
            "target_sha256": target_record["sha256"],
            "conditioning_sha256": conditioning["sha256"],
            "exact_common_target_and_conditioning": True,
        }
        for seed in SEEDS for checkpoint in CHECKPOINTS
    ]
    z0 = {
        "schema": SCHEMA,
        "status": "PASS_Z0_NO_MODEL_CONSTRUCTION",
        "timestamp_utc": _utc_now(),
        "stage29_seal_revalidated": True,
        "stage30_seal_revalidated": True,
        "six_stage29_npz_revalidated": True,
        "target": target_record,
        "target_artifact": {"path": str(target_path), "bytes": target_path.stat().st_size, "sha256": _sha256_file(target_path)},
        "conditioning": {
            "observed_sha256": conditioning["sha256"],
            "stage29_expected_sha256": historical_conditioning,
            "exact_match": True,
        },
        "planned_pairs": planned,
        "stage29_call_identities": [_jsonable(historical_manifest[key]) for key in sorted(historical_manifest)],
        "initial_noise_to_final_output_retention": _zero_forward_retention(historical_arrays),
        "execution": {"model_constructed": False, "model_forward": False, "sampler_calls": 0},
        "scientific_contract_changed": False,
    }
    _write_json_atomic(args.output_root / "stage31_z0_zero_forward.json", z0)
    return processor, embodiment, snapshot, conditioning, historical_manifest, (inference_batch, target, z0)


def _save_trace_artifact(args: argparse.Namespace, seed: int, checkpoint: str, trace: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    trace_dir = args.output_root / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    suffix = "base" if checkpoint == "BASE" else "h2_2000"
    path = trace_dir / f"seed_{seed}_{suffix}_stock_sampler_trace.npz"
    np.savez_compressed(
        path,
        x_trajectory_float32=trace["x_trajectory_float32"],
        v_trajectory_float32=trace["v_trajectory_float32"],
        timestep_ids=np.asarray(trace["timestep_ids"], dtype=np.int64),
    )
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def execute(args: argparse.Namespace) -> dict[str, Any]:
    preflight = run_preflight(args)
    if args.device != "cuda:0":
        raise Stage31Error("Stage31 is frozen to cuda:0")
    import numpy as np
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Stage31Error("Stage31 requires exactly one visible CUDA GPU")
    if "H200" not in torch.cuda.get_device_name(0):
        raise Stage31Error(f"Stage31 requires H200, observed {torch.cuda.get_device_name(0)}")
    from scripts import stage29_c2_free_running_transfer as stage29

    processor, embodiment, snapshot, conditioning, historical_manifest, prepared = _prepare_zero_forward_phase(args, preflight)
    del processor, embodiment, snapshot
    inference_batch, target, z0 = prepared
    if z0["execution"]["model_constructed"] is not False:
        raise Stage31Error("Z0 model-construction boundary drifted")

    h2_checkpoint = args.training_output_root / "h2_training" / "checkpoint-2000"
    models = {
        "BASE": stage29._load_model(args.base_model_path, args.device, torch),
        "H2@2000": stage29._load_model(h2_checkpoint, args.device, torch),
    }
    state_path = args.output_root / "stage31_execution_state.json"
    partial_path = args.output_root / "stage31_partial_records.json"
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
    for seed in SEEDS:
        for checkpoint in CHECKPOINTS:
            expected = historical_manifest[(seed, checkpoint)]
            call_inputs = copy.deepcopy(inference_batch)
            call_conditioning = stage29._conditioning_manifest(call_inputs)
            if call_conditioning["sha256"] != conditioning["sha256"]:
                raise Stage31Error(f"conditioning changed before {checkpoint} seed {seed}")
            execution_state.update({
                "status": (
                    "READY_FOR_FIRST_SCIENTIFIC_CALL"
                    if not records else "BETWEEN_AUTHORIZED_CALLS_AFTER_SCIENCE_BEGUN"
                ),
                "scientific_execution_begun": bool(records),
                "current_seed": seed,
                "current_checkpoint": checkpoint,
                "updated_at_utc": _utc_now(),
            })
            _write_json_atomic(state_path, execution_state)
            rng = stage29._reset_rng(seed, torch, np)
            started = time.perf_counter()
            observer = StockSamplerTrace(models[checkpoint].action_head)
            with stage29._capture_stock_initial_noise(torch) as noise_calls:
                with observer:
                    execution_state.update({
                        "status": "SCIENTIFIC_EXECUTION_BEGUN",
                        "scientific_execution_begun": True,
                        "updated_at_utc": _utc_now(),
                    })
                    _write_json_atomic(state_path, execution_state)
                    with torch.inference_mode():
                        prediction = models[checkpoint].get_action(**call_inputs)
            torch.cuda.synchronize()
            latency = time.perf_counter() - started
            noise = stage29._validate_noise_capture(noise_calls, models[checkpoint])
            noise_tensor = noise.pop("tensor_float32")
            final_actions = prediction["action_pred"]
            trace = _finalize_trace(observer, final_actions, torch)
            trace_artifact = _save_trace_artifact(args, seed, checkpoint, trace)
            noise_raw_match = noise["raw_sha256"] == expected["initial_noise_raw_sha256"]
            noise_float32_match = noise["float32_sha256"] == expected["initial_noise_float32_sha256"]
            x0_noise_match = trace["tensor_records"]["x"][0]["raw_sha256"] == noise["raw_sha256"]
            final_float32_sha = trace["tensor_records"]["x"][4]["float32_sha256"]
            final_match = final_float32_sha == expected["final_raw_universal_sha256"]
            conditioning_match = call_conditioning["sha256"] == expected["conditioning_sha256"]
            reproduction = {
                "status": "MATCH" if all((noise_raw_match, noise_float32_match, x0_noise_match, final_match, conditioning_match)) else "MISMATCH",
                "noise_raw_expected": expected["initial_noise_raw_sha256"],
                "noise_raw_observed": noise["raw_sha256"],
                "noise_raw_exact": noise_raw_match,
                "noise_float32_expected": expected["initial_noise_float32_sha256"],
                "noise_float32_observed": noise["float32_sha256"],
                "noise_float32_exact": noise_float32_match,
                "trace_x0_equals_captured_noise_raw": x0_noise_match,
                "final_raw_universal_expected": expected["final_raw_universal_sha256"],
                "final_raw_universal_observed": final_float32_sha,
                "final_raw_universal_exact": final_match,
                "conditioning_expected": expected["conditioning_sha256"],
                "conditioning_observed": call_conditioning["sha256"],
                "conditioning_exact": conditioning_match,
            }
            row = {
                "sampler_call_index": len(records) + 1,
                "seed": seed,
                "checkpoint": checkpoint,
                "checkpoint_path": str(paths[checkpoint]),
                "rng_before_call": rng,
                "conditioning_sha256": call_conditioning["sha256"],
                "initial_noise": noise,
                "trace": {
                    "state_count": 5,
                    "velocity_count": 4,
                    "timestep_ids": trace["timestep_ids"],
                    "tensor_records": trace["tensor_records"],
                    "euler_recurrence": trace["euler_recurrence"],
                },
                "trace_artifact": trace_artifact,
                "stage29_reproduction": reproduction,
                "latency_s": latency,
                "metrics": trace_metrics(trace["x_trajectory_float32"], trace["v_trajectory_float32"], target),
            }
            records.append(row)
            execution_state.update({
                "status": "SCIENTIFIC_EXECUTION_IN_PROGRESS",
                "completed_sampler_calls": len(records),
                "last_completed_seed": seed,
                "last_completed_checkpoint": checkpoint,
                "last_reproduction_status": reproduction["status"],
                "updated_at_utc": _utc_now(),
            })
            _write_json_atomic(partial_path, {
                "schema": SCHEMA,
                "status": "PARTIAL_EVIDENCE",
                "record_count": len(records),
                "records": records,
            })
            _write_json_atomic(state_path, execution_state)
            assert_stage29_reproduction(reproduction, len(records))

    if len(records) != MAX_SAMPLER_CALLS:
        raise Stage31Error(f"sampler call count drifted: {len(records)}")
    aggregates = aggregate_traces(records, target)
    result = {
        "schema": SCHEMA,
        "status": "PASS_EXECUTION_COMPLETE",
        "stage": "Stage31 on-trajectory rectified-flow / sampler-state audit",
        "pi_gate": PI_GATE_ID,
        "project_sha": args.expected_project_sha,
        "pbs": {
            "job_id": args.pbs_job_id,
            "project": args.pbs_project,
            "gpu": torch.cuda.get_device_name(0),
        },
        "preflight": preflight,
        "z0": z0,
        "analytic_reference": {
            "definition": "v_ref(x,t)=(a-x)/(1-t)",
            "reference_next": "x_ref_next=x+0.25*v_ref",
            "single_locked_target_only": True,
            "general_multi_action_field_claim": False,
        },
        "records": records,
        "aggregates": aggregates,
        "reproduction": {
            "status": "MATCH_6_OF_6",
            "initial_noise_raw_exact": True,
            "final_raw_universal_exact": True,
            "conditioning_exact": True,
        },
        "execution": {
            "scientific_execution_begun": True,
            "sampler_calls": 6,
            "base_sampler_calls": 3,
            "h2_sampler_calls": 3,
            "maximum_sampler_calls": 6,
            "stock_sampler_steps": 4,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "scheduler_step": False,
            "resume": False,
            "checkpoint_write": False,
            "alternative_sampler": False,
            "new_seed": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "wbc_executed": False,
            "hardware_executed": False,
        },
        "all_finite": _all_finite(records) and _all_finite(aggregates),
        "scientific_contract_changed": False,
        "scientific_interpretation": "DEFERRED_TO_GPT_PI",
        "claim_boundary": "single locked TINY1 target and six reproduced stock Stage29 sampler trajectories",
        "next_gate": NEXT_GATE,
    }
    if result["all_finite"] is not True:
        raise Stage31Error("Stage31 result contains a non-finite numeric value")
    result_path = args.output_root / "stage31_on_trajectory_flow_result.json"
    _write_json_atomic(result_path, result)
    raw_manifest_path = args.output_root / "stage31_raw_hash_manifest.json"
    _write_json_atomic(raw_manifest_path, {
        "schema": SCHEMA,
        "project_sha": args.expected_project_sha,
        "target": z0["target"],
        "records": [
            {
                "sampler_call_index": row["sampler_call_index"],
                "seed": row["seed"],
                "checkpoint": row["checkpoint"],
                "trace_artifact": row["trace_artifact"],
                "stage29_reproduction": row["stage29_reproduction"],
                "x_raw_sha256": [item["raw_sha256"] for item in row["trace"]["tensor_records"]["x"]],
                "v_raw_sha256": [item["raw_sha256"] for item in row["trace"]["tensor_records"]["v"]],
            }
            for row in records
        ],
    })
    execution_state.update({
        "status": "PASS_EXECUTION_COMPLETE",
        "completed_sampler_calls": len(records),
        "updated_at_utc": _utc_now(),
    })
    _write_json_atomic(state_path, execution_state)
    return result


def _artifact_entry(path: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "role": role,
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def seal_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    result_path = args.output_root / "stage31_on_trajectory_flow_result.json"
    result = _read_json(result_path)
    if result.get("status") != "PASS_EXECUTION_COMPLETE" or result.get("next_gate") != NEXT_GATE:
        raise Stage31Error("cannot seal incomplete Stage31 result")
    if result.get("project_sha") != args.expected_project_sha:
        raise Stage31Error("Stage31 result/source binding drifted")
    _require_project_identity(args.project_repo, args.expected_project_sha)
    paths = [
        *sorted(path for path in args.output_root.iterdir() if path.is_file() and path.name != "stage31_artifact_seal.json"),
        *sorted((args.output_root / "traces").glob("*.npz")),
        args.pbs_log,
        Path(__file__).resolve(),
        args.pbs_template,
        args.source_audit_report,
        args.stage29_seal,
        args.stage30_seal,
        args.stage31_test,
    ]
    entries = []
    seen = set()
    for path in paths:
        resolved = path.resolve(strict=True)
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        entries.append(_artifact_entry(resolved, "stage31_evidence"))
    seal = {
        "schema": "STAGE31_ON_TRAJECTORY_ARTIFACT_SEAL_V1",
        "status": "PASS",
        "created_at_utc": _utc_now(),
        "pi_gate": PI_GATE_ID,
        "project_sha": result["project_sha"],
        "pbs_job_id": result["pbs"]["job_id"],
        "stage29_artifact_seal_sha256": STAGE29_SEAL_SHA256,
        "stage30_artifact_seal_sha256": STAGE30_SEAL_SHA256,
        "artifact_count": len(entries),
        "artifacts": entries,
        "sampler_calls": 6,
        "reproduction": "MATCH_6_OF_6",
        "scientific_contract_changed": False,
        "next_gate": NEXT_GATE,
    }
    seal_path = args.output_root / "stage31_artifact_seal.json"
    _write_json_atomic(seal_path, seal)
    return {**seal, "path": str(seal_path), "sha256": _sha256_file(seal_path), "bytes": seal_path.stat().st_size}


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
    parser.add_argument("--stage29-root", type=Path)
    parser.add_argument("--stage29-seal", type=Path)
    parser.add_argument("--stage29-result", type=Path)
    parser.add_argument("--stage30-root", type=Path)
    parser.add_argument("--stage30-seal", type=Path)
    parser.add_argument("--source-audit-report", type=Path)
    parser.add_argument("--pbs-template", type=Path)
    parser.add_argument("--stage31-test", type=Path)
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
    common_required = (
        "project_repo", "expected_project_sha", "base_model_path", "dataset_dir", "groot_source",
        "cosmos_cache_root", "modality_config", "snapshot_root", "training_output_root",
        "stage29_root", "stage29_seal", "stage29_result", "stage30_root", "stage30_seal",
        "source_audit_report", "pbs_template", "stage31_test", "expected_runner_sha256",
        "expected_pbs_sha256", "expected_source_audit_sha256", "output_root", "authorization_token",
    )
    missing = [name for name in common_required if getattr(args, name) is None]
    if args.mode == "seal" and args.pbs_log is None:
        missing.append("pbs_log")
    if missing:
        raise SystemExit("missing required arguments: " + ", ".join(missing))
    try:
        if args.mode == "preflight":
            result = run_preflight(args)
            _write_json_atomic(args.output_root / "stage31_preflight.json", result)
        elif args.mode == "execute":
            result = execute(args)
        else:
            result = seal_artifacts(args)
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "status": "STOP_OR_BLOCKED",
            "error_type": type(exc).__name__,
            "error_summary": str(exc).splitlines()[0][:1000],
            "scientific_contract_changed": False,
            "autonomous_retry_authorized": False,
            "next_gate": NEXT_GATE,
        }
        if args.output_root is not None and args.output_root.exists():
            _write_json_atomic(args.output_root / "stage31_failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
