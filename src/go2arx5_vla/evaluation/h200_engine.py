"""Manifest-driven inference execution boundary.

This module owns runtime admission, explicit manifest ordering, durable raw
handoff, and the existing CPU evaluator.  It has no model-framework imports.
The only model-specific code is selected through the exact allowlisted adapter
registry after admission has passed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
from typing import Any, Protocol

import numpy as np

from ..experiments.reusable import (
    INFERENCE_BACKEND_V1,
    RUNTIME_LATE_BOUND_FIELDS,
    ReusableManifestError,
    load_json,
    require_mapping,
    require_nonempty_string,
    sha256_file,
    validate_late_bound_runtime,
    verify_artifact_identity,
    write_json,
)
from .h200_admission import (
    H200_INFERENCE_ADMISSION_SCHEMA,
    H200AdmissionError,
    validate_h200_inference_admission,
)
from .h200_binding import (
    AdapterRuntimeBindingV1,
    build_adapter_runtime_binding,
    validate_adapter_runtime_binding,
)
from .reusable import (
    GENERIC_PREDICTION_RECORD_SCHEMA,
    load_evaluation_manifest,
    run_evaluation,
    verify_evaluation_artifacts,
)
from .structured_action_eval import (
    EXPECTED_MODEL_INPUT_KEYS,
    PHYSICAL_ACTION_SHAPE,
    RAW_ACTION_SHAPE,
    STATE_V1_SHAPE,
    DecodedStructuredPrediction,
    decode_raw_prediction,
)
from .trajectory_metrics import array_sha256


H200_ENGINE_SCHEMA = "GO2ARX5_REUSABLE_H200_INFERENCE_ENGINE_V1"
H200_RUNTIME_CONFIG_SCHEMA = "GO2ARX5_REUSABLE_H200_RUNTIME_CONFIG_V1"
H200_RAW_SEAL_SCHEMA = "GO2ARX5_REUSABLE_H200_RAW_SEAL_V1"
H200_ENGINE_RUN_SCHEMA = "GO2ARX5_REUSABLE_H200_ENGINE_RUN_V1"
H200_ENGINE_FAILURE_SCHEMA = "GO2ARX5_REUSABLE_H200_ENGINE_FAILURE_SEAL_V1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_DYNAMIC_RUNTIME_FIELDS = (
    "repository_execution_head",
    "python_runtime",
    "host",
    "offline_mode",
    "gpu_device",
    "runtime_admission_identity",
)


class H200EngineError(ReusableManifestError):
    """Raised when the execution boundary cannot be admitted safely."""


class H200EngineExecutionError(RuntimeError):
    """Raised after a failed run has persisted authoritative provenance."""

    def __init__(self, message: str, *, provenance: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.provenance = dict(provenance)


@dataclass(frozen=True)
class AdapterContext:
    """Read-only context handed to an adapter after admission."""

    manifest_path: Path
    manifest: Mapping[str, Any]
    admission: Mapping[str, Any]
    runtime_config: Mapping[str, Any]
    repo_root: Path
    verified_artifacts: Mapping[str, Any]
    adapter_binding: Mapping[str, Any]


@dataclass(frozen=True)
class AdapterInference:
    """One model result plus the sealed inputs needed by the CPU evaluator."""

    raw: Any
    sampler_calls: Any
    observations: Mapping[str, Any] = field(default_factory=dict)
    state_v1: Any = None
    expert_physical_action: Any = None
    model_input_keys: Sequence[str] = EXPECTED_MODEL_INPUT_KEYS
    target_xyz_in_model_input: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    noise_sha256: str | None = None


class InferenceAdapter(Protocol):
    """Minimal versioned adapter interface used by the generic engine."""

    def prepare(self, context: AdapterContext) -> None:
        """Import and construct model state after admission."""

    def infer(
        self,
        *,
        checkpoint: Mapping[str, Any],
        window: Mapping[str, Any],
        noise_bank: Path,
        context: AdapterContext,
    ) -> AdapterInference:
        """Produce one raw universal action and its sealed evaluation inputs."""

    def close(self) -> None:
        """Release model/runtime state after the evidence handoff."""


RuntimeProbe = Callable[[Path], Mapping[str, Any]]
Decoder = Callable[..., DecodedStructuredPrediction]


@dataclass
class _ExecutionState:
    """Monotonic engine-owned lifecycle and provenance counters."""

    phase: str = "PRE_ADMISSION"
    admission_passed: bool = False
    adapter_resolved: bool = False
    model_imported: bool = False
    model_construction: bool = False
    model_forward_started: bool = False
    model_forward: bool = False
    conditioning_backbone_passes: int = 0
    checkpoint_window_forwards_started: int = 0
    checkpoint_window_forwards_completed: int = 0
    sampler_calls_completed: int = 0
    raw_count: int = 0
    raw_seal_status: str = "NOT_STARTED"
    derived_phase: str = "NOT_STARTED"
    training_steps: int = 0
    wbc_steps: int = 0
    simulator_steps: int = 0
    hardware_steps: int = 0


def _execution_snapshot(state: _ExecutionState, adapter: Any) -> dict[str, Any]:
    """Merge adapter-observed flags into the authoritative engine state."""

    flags = _runtime_flags(adapter)
    for name in (
        "model_imported",
        "model_construction",
        "model_forward_started",
        "model_forward",
    ):
        setattr(state, name, bool(getattr(state, name) or flags.get(name, False)))
    for name in (
        "conditioning_backbone_passes",
        "checkpoint_window_forwards_started",
        "checkpoint_window_forwards_completed",
        "sampler_calls_completed",
    ):
        value = flags.get(name)
        if type(value) is int:
            setattr(state, name, max(getattr(state, name), value))
    state.raw_count = max(state.raw_count, 0)
    state.training_steps = 0
    state.wbc_steps = 0
    state.simulator_steps = 0
    state.hardware_steps = 0
    model_execution_started = bool(
        state.model_imported
        or state.model_construction
        or state.model_forward
        or state.checkpoint_window_forwards_started
        or state.sampler_calls_completed
        or state.raw_count
    )
    return {
        "phase": state.phase,
        "admission_passed": state.admission_passed,
        "adapter_resolved": state.adapter_resolved,
        "model_execution_started": model_execution_started,
        "model_imported": state.model_imported,
        "model_construction": state.model_construction,
        "model_forward_started": state.model_forward_started,
        "model_forward": state.model_forward,
        "conditioning_backbone_passes": state.conditioning_backbone_passes,
        "checkpoint_window_forwards_started": state.checkpoint_window_forwards_started,
        "checkpoint_window_forwards_completed": state.checkpoint_window_forwards_completed,
        "sampler_calls_completed": state.sampler_calls_completed,
        "raw_count": state.raw_count,
        "raw_seal_status": state.raw_seal_status,
        "derived_phase": state.derived_phase,
        "derived_started": state.derived_phase != "NOT_STARTED",
        "training_steps": state.training_steps,
        "training": state.training_steps > 0,
        "wbc_steps": state.wbc_steps,
        "simulator_steps": state.simulator_steps,
        "hardware_steps": state.hardware_steps,
        "adapter_runtime_flags": flags,
    }


def _execution_document(state: _ExecutionState, adapter: Any) -> dict[str, Any]:
    """Return nested provenance plus backward-compatible lifecycle fields."""

    snapshot = _execution_snapshot(state, adapter)
    return {
        "execution_state": snapshot,
        "model_execution_started": snapshot["model_execution_started"],
        "model_imported": snapshot["model_imported"],
        "model_construction": snapshot["model_construction"],
        "model_forward_started": snapshot["model_forward_started"],
        "model_forward": snapshot["model_forward"],
        "training": snapshot["training_steps"] > 0,
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise H200EngineError(message)


def _strict_integer(value: Any, *, field: str) -> int:
    _require(type(value) is int, f"{field} must be a JSON integer")
    return value


def _strict_number(value: Any, *, field: str) -> float:
    _require(
        type(value) in (int, float),
        f"{field} must be an explicit finite JSON number",
    )
    result = float(value)
    _require(math.isfinite(result), f"{field} must be an explicit finite JSON number")
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _durable_json(path: Path, value: Any) -> None:
    write_json(path, _jsonable(value))
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _durable_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _safe_segment(value: Any) -> str:
    result = _SAFE_SEGMENT_RE.sub("_", str(value)).strip("._")
    return result or "item"


def _manifest_file(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    _require(not candidate.is_symlink(), f"manifest may not be a symlink: {candidate}")
    try:
        candidate = candidate.resolve(strict=True)
    except OSError as exc:
        raise H200EngineError(f"manifest path cannot be resolved: {candidate}") from exc
    _require(candidate.is_file(), f"manifest must be a regular file: {candidate}")
    return candidate


def _runtime_config_file(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    _require(
        not candidate.is_symlink(),
        f"runtime config may not be a symlink: {candidate}",
    )
    try:
        candidate = candidate.resolve(strict=True)
    except OSError as exc:
        raise H200EngineError(
            f"runtime config path cannot be resolved: {candidate}"
        ) from exc
    _require(candidate.is_file(), f"runtime config must be a regular file: {candidate}")
    return candidate


def _load_runtime_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = _runtime_config_file(path)
    try:
        config = load_json(config_path)
    except ReusableManifestError as exc:
        raise H200EngineError(str(exc)) from exc
    _require(
        config.get("schema") == H200_RUNTIME_CONFIG_SCHEMA,
        "runtime config schema mismatch",
    )
    observed = config.get("observed_runtime")
    try:
        observed_mapping = validate_late_bound_runtime(
            observed,
            field="runtime_config.observed_runtime",
            allow_sentinel=False,
        )
    except ReusableManifestError as exc:
        raise H200EngineError(str(exc)) from exc
    _require(
        set(observed_mapping) == set(RUNTIME_LATE_BOUND_FIELDS),
        "runtime_config.observed_runtime must contain exactly the late-bound fields",
    )
    _require(
        type(observed_mapping["offline_mode"]) is bool,
        "runtime_config.observed_runtime.offline_mode must be a JSON boolean",
    )
    _require(
        "adapter_config" not in config,
        "runtime_config.adapter_config is deprecated; use immutable adapter_runtime_binding",
    )
    require_mapping(
        config.get("adapter_runtime_binding"),
        field="runtime_config.adapter_runtime_binding",
    )
    root_value = config.get("repository_root", config.get("repo_root"))
    _require(root_value is not None, "runtime config repository_root is required")
    root = Path(str(root_value)).expanduser()
    _require(root.is_absolute(), "runtime config repository_root must be absolute")
    _require(root.is_dir() and not root.is_symlink(), "runtime config repository_root is not a directory")
    _require(
        type(config.get("require_clean_checkout", False)) is bool,
        "runtime_config.require_clean_checkout must be a JSON boolean",
    )
    sampler = config.get("observed_sampler")
    sampler_mapping = require_mapping(sampler, field="runtime_config.observed_sampler")
    noise = sampler_mapping.get("noise_bank")
    require_mapping(noise, field="runtime_config.observed_sampler.noise_bank")
    return config, config_path


def _git_value(repo_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise H200EngineError(f"cannot inspect repository runtime: git {' '.join(arguments)}") from exc
    return result.stdout.strip()


def parse_git_status(status: str) -> tuple[list[str], list[str]]:
    """Return porcelain status lines and every explicitly reported untracked path."""

    status_lines = [line for line in status.splitlines() if line]
    untracked_paths = [
        line[3:].strip()
        for line in status_lines
        if line.startswith("?? ") and len(line) > 3
    ]
    return status_lines, untracked_paths


def observe_runtime(repo_root: str | Path) -> dict[str, Any]:
    """Observe dynamic facts from the process and checkout, without a model import."""

    root = Path(repo_root).expanduser().resolve()
    actual_root = Path(_git_value(root, "rev-parse", "--show-toplevel")).resolve()
    status = _git_value(
        root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    status_lines, untracked_paths = parse_git_status(status)
    return {
        "repository_root": str(actual_root),
        "repository_execution_head": _git_value(root, "rev-parse", "HEAD"),
        "repository_branch": _git_value(root, "branch", "--show-current"),
        "python_runtime": "|".join(
            (sys.executable, platform.python_implementation(), platform.python_version())
        ),
        "host": socket.gethostname(),
        "offline_mode": os.environ.get("GO2ARX5_OFFLINE_MODE") == "1",
        "gpu_device": os.environ.get("GO2ARX5_GPU_DEVICE")
        or os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "runtime_admission_identity": os.environ.get(
            "GO2ARX5_RUNTIME_ADMISSION_IDENTITY", ""
        ),
        "worktree_clean": not bool(status_lines),
        "untracked_paths": untracked_paths,
    }


def _observed_artifact(
    value: Any,
    *,
    base_dir: Path,
    field: str,
) -> dict[str, Any]:
    artifact = require_mapping(value, field=field)
    raw_path = require_nonempty_string(artifact.get("path"), field=f"{field}.path")
    try:
        actual = verify_artifact_identity(artifact, base_dir=base_dir, field=field)
    except ReusableManifestError as exc:
        raise H200EngineError(str(exc)) from exc
    kind = str(artifact.get("kind", "file"))
    _require(kind in {"file", "tree"}, f"{field}.kind must be file or tree")
    # Keep the observed path representation stable for h200_admission's exact
    # cross-binding check; the digest is always recomputed from the file/tree.
    return {"path": raw_path, "sha256": actual["sha256"], "kind": kind}


def _observed_artifact_map(
    value: Any,
    *,
    labels: Sequence[str],
    base_dir: Path,
    field: str,
) -> dict[str, dict[str, Any]]:
    mapping = require_mapping(value, field=field)
    normalized = {str(key): child for key, child in mapping.items()}
    _require(
        set(normalized) == set(labels),
        f"{field} must contain exactly the manifest checkpoint labels",
    )
    return {
        label: _observed_artifact(
            normalized[label],
            base_dir=base_dir,
            field=f"{field}.{label}",
        )
        for label in labels
    }


def _runtime_output_root(value: Any, *, field: str) -> str:
    result = require_nonempty_string(value, field=field)
    return result


def _resolve_runtime_config(
    runtime_config: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_file: Path,
    repo_root: Path | None,
    runtime_probe: RuntimeProbe | None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Bind observed paths and dynamic process facts into a new runtime map."""

    config_root_value = runtime_config.get("repository_root", runtime_config.get("repo_root"))
    _require(config_root_value is not None, "runtime config repository_root is required")
    config_root = Path(str(config_root_value)).expanduser().resolve()
    selected_root = (repo_root or config_root).expanduser().resolve()
    _require(
        selected_root == config_root,
        "runtime config repository_root differs from selected repository root",
    )
    if runtime_probe is None:
        observed_process = observe_runtime(selected_root)
    else:
        observed_process = dict(runtime_probe(selected_root))
    untracked_paths = observed_process.get("untracked_paths", [])
    _require(
        isinstance(untracked_paths, list)
        and all(type(path) is str and bool(path.strip()) for path in untracked_paths),
        "runtime probe untracked_paths must be a list of non-empty strings",
    )
    _require(
        not untracked_paths,
        "observed checkout contains untracked files; execution admission is fail-closed",
    )
    actual_root_value = observed_process.get("repository_root")
    _require(actual_root_value is not None, "runtime probe did not report repository_root")
    actual_root = Path(str(actual_root_value)).expanduser().resolve()
    _require(
        actual_root == selected_root,
        "observed repository root differs from runtime config repository_root",
    )

    expected_head = str(manifest["experiment"]["execution_head"])
    actual_head = require_nonempty_string(
        observed_process.get("repository_execution_head"),
        field="observed.repository_execution_head",
    )
    _require(
        actual_head.lower() == expected_head.lower(),
        "observed repository execution HEAD differs from manifest",
    )
    configured_head = runtime_config.get("expected_execution_head")
    if configured_head is not None:
        _require(
            str(configured_head).lower() == expected_head.lower(),
            "runtime config expected_execution_head differs from manifest",
        )
    configured_branch = runtime_config.get("expected_branch")
    if configured_branch is not None:
        _require(
            str(observed_process.get("repository_branch", "")) == str(configured_branch),
            "observed repository branch differs from runtime config",
        )
    if runtime_config.get("require_clean_checkout", False):
        _require(
            observed_process.get("worktree_clean") is True,
            "observed checkout is not clean",
        )

    expected_runtime = dict(runtime_config["observed_runtime"])
    for field in _DYNAMIC_RUNTIME_FIELDS:
        _require(field in observed_process, f"runtime probe did not report {field}")
        _require(
            observed_process[field] == expected_runtime[field],
            f"observed runtime {field} differs from runtime config",
        )
    _require(
        expected_runtime["repository_execution_head"].lower() == expected_head.lower(),
        "runtime config observed HEAD differs from manifest",
    )
    _require(
        expected_runtime["offline_mode"] is True,
        "H200 inference requires observed offline_mode=true",
    )

    labels = [str(item["label"]) for item in manifest["checkpoints"]]
    observed = dict(expected_runtime)
    observed["checkpoint_paths"] = _observed_artifact_map(
        expected_runtime["checkpoint_paths"],
        labels=labels,
        base_dir=manifest_file.parent,
        field="runtime_config.observed_runtime.checkpoint_paths",
    )
    observed["statistics_path"] = _observed_artifact_map(
        expected_runtime["statistics_path"],
        labels=labels,
        base_dir=manifest_file.parent,
        field="runtime_config.observed_runtime.statistics_path",
    )
    observed["dataset_path"] = _observed_artifact(
        expected_runtime["dataset_path"],
        base_dir=manifest_file.parent,
        field="runtime_config.observed_runtime.dataset_path",
    )
    observed["groot_source"] = _observed_artifact(
        expected_runtime["groot_source"],
        base_dir=manifest_file.parent,
        field="runtime_config.observed_runtime.groot_source",
    )
    observed["cosmos_cache_root"] = _observed_artifact(
        expected_runtime["cosmos_cache_root"],
        base_dir=manifest_file.parent,
        field="runtime_config.observed_runtime.cosmos_cache_root",
    )
    observed["output_root"] = _runtime_output_root(
        expected_runtime["output_root"],
        field="runtime_config.observed_runtime.output_root",
    )

    sampler = require_mapping(
        runtime_config["observed_sampler"], field="runtime_config.observed_sampler"
    )
    observed_noise = _observed_artifact(
        sampler["noise_bank"],
        base_dir=manifest_file.parent,
        field="runtime_config.observed_sampler.noise_bank",
    )
    _require(
        observed_noise["sha256"] == str(manifest["sampler"]["noise_bank_sha256"]),
        "observed noise bank differs from manifest",
    )
    observed_sampler = {
        "noise_bank": observed_noise,
        "source": "observed_runtime_artifact",
    }
    return observed, observed_sampler, selected_root


def build_observed_admission(
    manifest: Mapping[str, Any],
    manifest_path: str | Path,
    runtime_config: Mapping[str, Any] | str | Path,
    *,
    repo_root: str | Path | None = None,
    runtime_probe: RuntimeProbe | None = None,
    verified_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an admission document from observed runtime facts and artifacts."""

    manifest_file = _manifest_file(manifest_path)
    _require(
        load_json(manifest_file) == dict(manifest),
        "manifest mapping differs from the concrete manifest file",
    )
    if isinstance(runtime_config, Mapping):
        config = dict(runtime_config)
    else:
        config, _ = _load_runtime_config(runtime_config)
    _require(
        config.get("schema") == H200_RUNTIME_CONFIG_SCHEMA,
        "runtime config schema mismatch",
    )
    _require(
        "adapter_config" not in config,
        "runtime_config.adapter_config is deprecated; use immutable adapter_runtime_binding",
    )
    configured_adapter = config.get("adapter")
    if configured_adapter is not None:
        _require(
            configured_adapter == manifest["runtime"]["adapter"],
            "runtime config adapter differs from manifest",
        )
    try:
        validate_late_bound_runtime(
            config.get("observed_runtime"),
            field="runtime_config.observed_runtime",
            allow_sentinel=False,
        )
    except ReusableManifestError as exc:
        raise H200EngineError(str(exc)) from exc
    observed_runtime, observed_sampler, selected_root = _resolve_runtime_config(
        config,
        manifest=manifest,
        manifest_file=manifest_file,
        repo_root=Path(repo_root).expanduser().resolve() if repo_root is not None else None,
        runtime_probe=runtime_probe,
    )
    sampler_values = require_mapping(manifest.get("sampler"), field="manifest.sampler")
    timesteps = sampler_values.get("timesteps")
    _require(isinstance(timesteps, list), "manifest.sampler.timesteps must be an explicit list")
    for index, value in enumerate(timesteps):
        _strict_integer(value, field=f"manifest.sampler.timesteps[{index}]")
    _strict_number(sampler_values.get("dt"), field="manifest.sampler.dt")
    _strict_integer(
        sampler_values.get("expected_calls_per_window"),
        field="manifest.sampler.expected_calls_per_window",
    )
    if verified_artifacts is None:
        try:
            verified_artifacts = verify_evaluation_artifacts(
                manifest, manifest_path=manifest_file
            )
        except ReusableManifestError as exc:
            raise H200EngineError(f"mandatory artifact verification failed: {exc}") from exc
    adapter_binding = build_adapter_runtime_binding(
        manifest,
        config,
        observed_runtime=observed_runtime,
        observed_sampler=observed_sampler,
        verified_artifacts=verified_artifacts,
        manifest_path=manifest_file,
    )
    labels = [str(item["label"]) for item in manifest["checkpoints"]]
    windows = [str(item["id"]) for item in manifest["cohort"]["windows"]]
    targets = [str(item["target_id"]) for item in manifest["cohort"]["windows"]]
    admission = {
        "schema": H200_INFERENCE_ADMISSION_SCHEMA,
        "backend": INFERENCE_BACKEND_V1,
        "runtime_identity": manifest["runtime"]["identity"],
        "timestamp_profile": manifest["timestamp_profile"],
        "runtime_late_bound": observed_runtime,
        "adapter_runtime_binding": adapter_binding.to_dict(),
        "cohort": {
            "name": manifest["cohort"]["name"],
            "split": manifest["cohort"]["split"],
            "window_ids": windows,
            "target_ids": targets,
        },
        "sampler": {
            "profile": manifest["sampler"]["profile"],
            "timesteps": list(manifest["sampler"]["timesteps"]),
            "dt": manifest["sampler"]["dt"],
            "expected_calls_per_window": manifest["sampler"][
                "expected_calls_per_window"
            ],
            "same_noise_across_checkpoints": True,
            "noise_bank": observed_sampler["noise_bank"],
        },
        "output_fresh": True,
        "locked_boundaries": dict(manifest["locked_boundaries"]),
        "observed_runtime_provenance": {
            "source": "runtime_probe_plus_verified_artifacts",
            "repository_root": str(selected_root),
            "manifest_sha256": sha256_file(manifest_file),
            "runtime_config_sha256": (
                sha256_file(_runtime_config_file(runtime_config))
                if not isinstance(runtime_config, Mapping)
                else None
            ),
            "checkpoint_order": labels,
            "window_order": windows,
        },
    }
    return admission


def _fresh_directory(path: Path) -> None:
    if path.exists():
        _require(path.is_dir() and not path.is_symlink(), f"evidence root is unsafe: {path}")
        _require(not any(path.iterdir()), f"evidence root must be fresh and empty: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)


def _normalise_adapter_result(value: Any) -> AdapterInference:
    if isinstance(value, AdapterInference):
        return value
    if isinstance(value, Mapping):
        required = {
            "raw",
            "sampler_calls",
            "observations",
            "state_v1",
            "expert_physical_action",
        }
        missing = sorted(required.difference(value))
        _require(not missing, f"adapter result is missing fields: {', '.join(missing)}")
        return AdapterInference(
            raw=value["raw"],
            sampler_calls=value["sampler_calls"],
            observations=value["observations"],
            state_v1=value["state_v1"],
            expert_physical_action=value["expert_physical_action"],
            model_input_keys=value.get("model_input_keys", EXPECTED_MODEL_INPUT_KEYS),
            target_xyz_in_model_input=value.get("target_xyz_in_model_input", False),
            metadata=value.get("metadata", {}),
            noise_sha256=value.get("noise_sha256"),
        )
    _require(False, "adapter result must be AdapterInference or an object mapping")
    raise AssertionError("unreachable")


def _finite_array(value: Any, *, shape: tuple[int, ...], field: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise H200EngineError(f"{field} is not an array") from exc
    _require(tuple(array.shape) == shape, f"{field} must have shape {shape}")
    _require(
        np.issubdtype(array.dtype, np.number) and bool(np.isfinite(array).all()),
        f"{field} must be finite numeric data",
    )
    return np.ascontiguousarray(array)


def _observation_descriptor(
    value: Any,
    *,
    field: str,
    staging_root: Path,
    index: int,
) -> dict[str, Any]:
    array_value: Any = None
    artifact_value: Mapping[str, Any] | None = None
    if isinstance(value, Mapping) and "array" in value:
        array_value = value["array"]
    elif isinstance(value, Mapping) and "path" in value:
        artifact_value = value
    else:
        array_value = value

    if artifact_value is not None:
        try:
            verified = verify_artifact_identity(
                artifact_value,
                base_dir=staging_root,
                field=field,
            )
        except ReusableManifestError as exc:
            raise H200EngineError(str(exc)) from exc
        shape = list(artifact_value.get("shape", []))
        dtype = str(artifact_value.get("dtype", ""))
        _require(shape == [96, 96, 3], f"{field}.shape must be [96,96,3]")
        _require(dtype == "uint8", f"{field}.dtype must be uint8")
        return {
            "path": verified["resolved_path"],
            "sha256": verified["sha256"],
            "kind": str(artifact_value.get("kind", "file")),
            "shape": shape,
            "dtype": dtype,
        }

    array = _finite_array(array_value, shape=(96, 96, 3), field=field)
    _require(
        np.issubdtype(array.dtype, np.integer)
        and str(array.dtype) == "uint8",
        f"{field} must retain uint8 storage",
    )
    path = staging_root / "observations" / f"{index:08d}_{_safe_segment(field)}.npy"
    _durable_array(path, array)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "kind": "file",
        "shape": [96, 96, 3],
        "dtype": "uint8",
    }


def _normalise_observations(
    value: Any,
    *,
    staging_root: Path,
    index: int,
) -> dict[str, Any]:
    observations = require_mapping(value, field="adapter_result.observations")
    _require(
        set(observations) == set(EXPECTED_MODEL_INPUT_KEYS[:2]),
        "adapter observations must contain exactly body_rgb and wrist_rgb",
    )
    return {
        key: _observation_descriptor(
            observations[key],
            field=f"adapter_result.observations.{key}",
            staging_root=staging_root,
            index=index,
        )
        for key in EXPECTED_MODEL_INPUT_KEYS[:2]
    }


def _write_raw(
    raw: np.ndarray,
    *,
    staging_root: Path,
    checkpoint_index: int,
    window_index: int,
    checkpoint_label: str,
    window_id: str,
) -> tuple[Path, str, str]:
    path = (
        staging_root
        / "raw"
        / f"{checkpoint_index:08d}_{_safe_segment(checkpoint_label)}"
        / f"{window_index:08d}_{_safe_segment(window_id)}.npy"
    )
    _durable_array(path, raw)
    return path, sha256_file(path), array_sha256(raw)


def _write_inventory(output_root: Path) -> tuple[Path, str]:
    inventory = output_root / "engine_artifact_inventory_sha256.txt"
    lines: list[str] = []
    for candidate in sorted(output_root.rglob("*")):
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or candidate == inventory
            or candidate == output_root / "engine_run_manifest.json"
        ):
            continue
        lines.append(
            f"{sha256_file(candidate)}  {candidate.relative_to(output_root).as_posix()}\n"
        )
    inventory.write_text("".join(lines), encoding="utf-8")
    with inventory.open("rb") as handle:
        os.fsync(handle.fileno())
    return inventory, sha256_file(inventory)


def _ensure_failure_root(path: Path) -> None:
    if path.exists():
        _require(path.is_dir() and not path.is_symlink(), f"failure root is unsafe: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)


def _runtime_flags(adapter: Any) -> dict[str, Any]:
    value = getattr(adapter, "runtime_flags", {})
    if not isinstance(value, Mapping):
        return {}
    return {str(key): _jsonable(child) for key, child in value.items()}


def run_h200_inference_engine(
    manifest_path: str | Path,
    runtime_config_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    raw_staging_dir: str | Path | None = None,
    adapter: InferenceAdapter | None = None,
    runtime_probe: RuntimeProbe | None = None,
    decoder: Decoder = decode_raw_prediction,
) -> dict[str, Any]:
    """Run the complete manifest-driven raw-to-evidence lifecycle.

    The optional ``adapter`` and ``decoder`` injection points are for CPU
    fixtures only.  The command-line entrypoint supplies neither, so a real
    invocation resolves only the allowlisted adapter and uses the accepted
    decoder after admission.
    """

    manifest_file = _manifest_file(manifest_path)
    manifest = load_evaluation_manifest(manifest_file)
    runtime_config, runtime_config_file = _load_runtime_config(runtime_config_path)
    selected_root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(str(runtime_config.get("repository_root", runtime_config.get("repo_root")))).expanduser().resolve()
    )
    output_from_manifest = Path(str(manifest["output"]["root"])).expanduser()
    if not output_from_manifest.is_absolute():
        output_from_manifest = (manifest_file.parent / output_from_manifest).resolve()
    else:
        output_from_manifest = output_from_manifest.resolve()
    output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else output_from_manifest
    )
    _require(
        output == output_from_manifest,
        "selected output root differs from manifest output root",
    )

    staging: Path | None = None
    adapter_object: Any = adapter
    adapter_closed = False
    raw_count = 0
    total_sampler_calls = 0
    raw_status = "NOT_STARTED"
    derived_started = False
    records: list[dict[str, Any]] = []
    validated_admission: dict[str, Any] | None = None
    admission: dict[str, Any] | None = None
    adapter_binding: AdapterRuntimeBindingV1 | None = None
    state = _ExecutionState()
    try:
        # This verifies the concrete cohort/checkpoint/mapping/statistics/noise
        # inputs before resolving or preparing any adapter.
        try:
            verified = verify_evaluation_artifacts(manifest, manifest_path=manifest_file)
        except ReusableManifestError as exc:
            raise H200EngineError(f"mandatory artifact verification failed: {exc}") from exc
        admission = build_observed_admission(
            manifest,
            manifest_file,
            runtime_config,
            repo_root=selected_root,
            runtime_probe=runtime_probe,
            verified_artifacts=verified,
        )
        try:
            validated_admission = validate_h200_inference_admission(
                manifest,
                admission,
                manifest_path=manifest_file,
                require_adapter_binding=True,
            )
        except H200AdmissionError:
            # Do not resolve/import an adapter when the cross-binding fails.
            raise
        adapter_binding = validate_adapter_runtime_binding(
            manifest,
            validated_admission,
            manifest_path=manifest_file,
        )
        validated_admission["adapter_runtime_binding"] = adapter_binding.to_dict()
        state.admission_passed = True
        state.phase = "ADAPTER_RESOLVE"

        if raw_staging_dir is None:
            staging = output.parent / f".{_safe_segment(output.name)}.h200-raw"
        else:
            staging = Path(raw_staging_dir).expanduser().resolve()
        _fresh_directory(staging)
        _durable_json(
            staging / "admission.json",
            {
                **admission,
                "engine_schema": H200_ENGINE_SCHEMA,
                "admission_status": validated_admission["status"],
                **_execution_document(state, adapter_object),
            },
        )

        from .h200_adapters import adapter_spec, resolve_adapter

        # Validate the exact adapter ID after admission even for CPU-injected
        # fixtures; only the approved factory is ever used by the CLI.
        adapter_spec(str(manifest["runtime"]["adapter"]))
        if adapter_object is None:

            adapter_object = resolve_adapter(
                str(manifest["runtime"]["adapter"]),
                AdapterContext(
                    manifest_path=manifest_file,
                    manifest=manifest,
                    admission=validated_admission,
                    runtime_config=runtime_config,
                    repo_root=selected_root,
                    verified_artifacts=verified,
                    adapter_binding=adapter_binding,
                ),
            )
        state.adapter_resolved = True
        _require(callable(getattr(adapter_object, "infer", None)), "adapter does not provide infer()")
        prepare = getattr(adapter_object, "prepare", None)
        if callable(prepare):
            state.phase = "ADAPTER_PREPARE"
            prepare(
                AdapterContext(
                    manifest_path=manifest_file,
                    manifest=manifest,
                    admission=validated_admission,
                    runtime_config=runtime_config,
                    repo_root=selected_root,
                    verified_artifacts=verified,
                    adapter_binding=adapter_binding,
                )
            )
            _execution_snapshot(state, adapter_object)

        checkpoint_verified = {
            str(item["label"]): item for item in verified["checkpoints"]
        }
        noise_path = Path(str(verified["noise_bank"]["path"])).resolve()
        noise_sha = str(verified["noise_bank"]["sha256"])
        checkpoints = list(manifest["checkpoints"])
        windows = list(manifest["cohort"]["windows"])
        expected_calls = _strict_integer(
            manifest["sampler"]["expected_calls_per_window"],
            field="manifest.sampler.expected_calls_per_window",
        )
        raw_manifest_path = staging / "raw_predictions.jsonl"
        with raw_manifest_path.open("w", encoding="utf-8") as raw_handle:
            for checkpoint_index, checkpoint in enumerate(checkpoints):
                label = str(checkpoint["label"])
                checkpoint_runtime = checkpoint_verified[label]
                for window_index, window in enumerate(windows):
                    state.phase = "RAW_INFERENCE"
                    state.checkpoint_window_forwards_started += 1
                    result = _normalise_adapter_result(
                        adapter_object.infer(
                            checkpoint=checkpoint_runtime,
                            window=window,
                            noise_bank=noise_path,
                            context=AdapterContext(
                                manifest_path=manifest_file,
                                manifest=manifest,
                                admission=validated_admission,
                                runtime_config=runtime_config,
                                repo_root=selected_root,
                                verified_artifacts=verified,
                                adapter_binding=adapter_binding,
                            ),
                        )
                    )
                    _require(
                        tuple(result.model_input_keys) == EXPECTED_MODEL_INPUT_KEYS,
                        "adapter model input ABI differs from body_rgb,wrist_rgb,state",
                    )
                    _require(
                        result.target_xyz_in_model_input is False,
                        "adapter attempted to include target XYZ in model input",
                    )
                    observed_calls = _strict_integer(
                        result.sampler_calls,
                        field="adapter_result.sampler_calls",
                    )
                    _require(
                        observed_calls == expected_calls,
                        "adapter sampler call count differs from manifest profile",
                    )
                    if result.noise_sha256 is not None:
                        _require(
                            str(result.noise_sha256) == noise_sha,
                            "adapter noise identity differs from admitted noise bank",
                        )
                    raw = _finite_array(
                        result.raw,
                        shape=RAW_ACTION_SHAPE,
                        field="adapter_result.raw",
                    )
                    state_v1 = _finite_array(
                        result.state_v1,
                        shape=STATE_V1_SHAPE,
                        field="adapter_result.state_v1",
                    )
                    expert = _finite_array(
                        result.expert_physical_action,
                        shape=PHYSICAL_ACTION_SHAPE,
                        field="adapter_result.expert_physical_action",
                    )
                    observations = _normalise_observations(
                        result.observations,
                        staging_root=staging,
                        index=raw_count,
                    )
                    raw_path, raw_sha, raw_array_sha = _write_raw(
                        raw,
                        staging_root=staging,
                        checkpoint_index=checkpoint_index,
                        window_index=window_index,
                        checkpoint_label=label,
                        window_id=str(window["id"]),
                    )
                    record = {
                        "schema": GENERIC_PREDICTION_RECORD_SCHEMA,
                        "checkpoint_label": label,
                        "checkpoint_sha256": str(checkpoint["identity"]["sha256"]),
                        "window_id": str(window["id"]),
                        "raw_path": str(raw_path),
                        "raw_sha256": raw_sha,
                        "raw_array_sha256": raw_array_sha,
                        "noise_sha256": noise_sha,
                        "model_input_keys": list(EXPECTED_MODEL_INPUT_KEYS),
                        "target_xyz_in_model_input": False,
                        "observations": observations,
                        "state_v1": state_v1.tolist(),
                        "expert_physical_action": expert.tolist(),
                        "sampler_calls_observed": observed_calls,
                        "adapter_metadata": _jsonable(dict(result.metadata)),
                        "raw_persisted_before_decode": True,
                    }
                    raw_handle.write(
                        json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                    )
                    raw_handle.flush()
                    os.fsync(raw_handle.fileno())
                    records.append(record)
                    raw_count += 1
                    total_sampler_calls += observed_calls
                    state.checkpoint_window_forwards_completed += 1
                    state.sampler_calls_completed += observed_calls
                    state.raw_count = raw_count
                    _execution_snapshot(state, adapter_object)
                    _durable_json(
                        staging / "raw_progress.json",
                        {
                            "schema": H200_RAW_SEAL_SCHEMA,
                            "status": "IN_PROGRESS",
                            "raw_count": raw_count,
                            "sampler_calls": total_sampler_calls,
                            "raw_manifest": str(raw_manifest_path),
                            "derived_started": False,
                            "execution_state": _execution_snapshot(state, adapter_object),
                        },
                    )

        state.raw_seal_status = "PASS"
        state.raw_count = raw_count
        state.sampler_calls_completed = total_sampler_calls
        raw_seal = {
            "schema": H200_RAW_SEAL_SCHEMA,
            "status": "PASS",
            "raw_count": raw_count,
            "sampler_calls": total_sampler_calls,
            "checkpoint_order": [str(item["label"]) for item in checkpoints],
            "window_order": [str(item["id"]) for item in windows],
            "raw_prediction_shape": list(RAW_ACTION_SHAPE),
            "raw_manifest": str(raw_manifest_path),
            "raw_manifest_sha256": sha256_file(raw_manifest_path),
            "noise_bank_sha256": noise_sha,
            "same_noise_across_checkpoints": True,
            "raw_persisted_before_derived": True,
            "derived_started": False,
            "execution_state": _execution_snapshot(state, adapter_object),
        }
        _durable_json(staging / "raw_seal.json", raw_seal)
        raw_status = "PASS"
        state.derived_phase = "RUNNING"
        state.phase = "DERIVED_EVALUATION"
        derived_started = True
        generic_summary = run_evaluation(
            manifest,
            records,
            output_dir=output,
            manifest_path=manifest_file,
            records_base_dir=staging,
            decoder=decoder,
        )
        state.derived_phase = "COMPLETE"
        state.phase = "FINALIZATION"
        close = getattr(adapter_object, "close", None)
        if callable(close):
            close()
            adapter_closed = True

        output_admission = {
            **admission,
            "engine_schema": H200_ENGINE_SCHEMA,
            "admission_status": validated_admission["status"],
            **_execution_document(state, adapter_object),
        }
        _durable_json(output / "engine_admission.json", output_admission)
        _durable_json(output / "engine_raw_seal.json", raw_seal)
        inventory_path, inventory_sha = _write_inventory(output)
        engine_summary = {
            "schema": H200_ENGINE_RUN_SCHEMA,
            "status": "PASS",
            "engine": H200_ENGINE_SCHEMA,
            "manifest_sha256": sha256_file(manifest_file),
            "runtime_config_sha256": sha256_file(runtime_config_file),
            "execution_head": manifest["experiment"]["execution_head"],
            "checkpoint_order": [str(item["label"]) for item in checkpoints],
            "window_order": [str(item["id"]) for item in windows],
            "checkpoint_count": len(checkpoints),
            "window_count": len(windows),
            "raw_prediction_count": raw_count,
            "sampler_calls": total_sampler_calls,
            "admission": {
                "path": "engine_admission.json",
                "sha256": sha256_file(output / "engine_admission.json"),
                "status": validated_admission["status"],
            },
            "raw_seal": {
                "path": "engine_raw_seal.json",
                "sha256": sha256_file(output / "engine_raw_seal.json"),
            },
            "generic_evaluation": {
                "path": "run_manifest.json",
                "sha256": sha256_file(output / "run_manifest.json"),
                "raw_prediction_count": generic_summary["raw_prediction_count"],
            },
            "artifact_inventory": {
                "path": inventory_path.relative_to(output).as_posix(),
                "sha256": inventory_sha,
            },
            "runtime_counters": {
                "checkpoint_count": len(checkpoints),
                "window_count": len(windows),
                "checkpoint_window_forwards": raw_count,
                "sampler_calls": total_sampler_calls,
                "reference_updates": 0,
                "training_steps": 0,
                "wbc_steps": 0,
                "simulator_steps": 0,
                "hardware_steps": 0,
                **_execution_snapshot(state, adapter_object),
            },
            "boundaries": {
                "model_import_after_admission": True,
                "raw_before_derived": True,
                "training": False,
                "Validation-E": "LOCKED",
                "Holdout-F": "LOCKED",
                "WBC": "LOCKED",
                "simulator": "LOCKED",
                "hardware": "LOCKED",
                "pbs_submission": False,
            },
        }
        _durable_json(output / "engine_run_manifest.json", engine_summary)
        return engine_summary
    except Exception as exc:
        state.raw_count = raw_count
        state.sampler_calls_completed = max(
            state.sampler_calls_completed, total_sampler_calls
        )
        state.raw_seal_status = (
            "PASS"
            if raw_status == "PASS"
            else ("FAILED_PARTIAL" if raw_count else "NOT_STARTED")
        )
        if derived_started and state.derived_phase == "NOT_STARTED":
            state.derived_phase = "RUNNING"
        failure_phase = state.phase
        state.phase = "FAILED"
        provenance = _execution_snapshot(state, adapter_object)
        provenance["failure_phase"] = failure_phase
        if staging is not None:
            if raw_status != "PASS":
                partial_seal = {
                    "schema": H200_RAW_SEAL_SCHEMA,
                    "status": "FAILED_PARTIAL",
                    "raw_count": raw_count,
                    "sampler_calls": total_sampler_calls,
                    "raw_manifest": str(staging / "raw_predictions.jsonl"),
                    "raw_persisted_before_derived": True,
                    "derived_started": provenance["derived_started"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "execution_state": provenance,
                }
                _durable_json(staging / "raw_seal.json", partial_seal)
        failure = {
            "schema": H200_ENGINE_FAILURE_SCHEMA,
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "raw_sealed": raw_status == "PASS",
            "staging_root": str(staging) if staging is not None else None,
            "raw_manifest": (
                str(staging / "raw_predictions.jsonl")
                if staging is not None
                else None
            ),
            "no_retry": True,
            **provenance,
        }
        if staging is not None:
            _durable_json(staging / "engine_failure_seal.json", failure)
        try:
            _ensure_failure_root(output)
            _durable_json(output / "engine_failure_seal.json", failure)
        except Exception:
            pass
        if isinstance(exc, H200EngineExecutionError):
            raise
        if isinstance(exc, (H200EngineError, H200AdmissionError, ReusableManifestError)):
            raise
        raise H200EngineExecutionError(str(exc), provenance=failure) from exc
    finally:
        if adapter_object is not None and not adapter_closed:
            close = getattr(adapter_object, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


__all__ = [
    "AdapterContext",
    "AdapterInference",
    "H200_ENGINE_RUN_SCHEMA",
    "H200_ENGINE_SCHEMA",
    "H200_ENGINE_FAILURE_SCHEMA",
    "H200_RAW_SEAL_SCHEMA",
    "H200_RUNTIME_CONFIG_SCHEMA",
    "H200EngineError",
    "H200EngineExecutionError",
    "InferenceAdapter",
    "build_observed_admission",
    "parse_git_status",
    "observe_runtime",
    "run_h200_inference_engine",
]
