"""Static H200 inference-admission cross-binding preflight.

This module is the stage-specific boundary immediately before a future H200
inference adapter.  It verifies the materialized inputs and runtime admission
without importing a model package, constructing a model, or performing a
forward pass.  A future backend must consume the returned preflight only after
its own PI-approved execution gate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..experiments.profiles import ProfileError, sampler_profile
from ..experiments.reusable import (
    INFERENCE_BACKEND_V1,
    RUNTIME_LATE_BOUND_FIELDS,
    ReusableManifestError,
    load_json,
    reject_locked_data,
    require_mapping,
    require_nonempty_string,
    validate_late_bound_runtime,
    verify_artifact_identity,
)
from .reusable import (
    GENERIC_EVAL_STATUS,
    validate_evaluation_manifest,
    verify_evaluation_artifacts,
)

H200_INFERENCE_ADMISSION_SCHEMA = (
    "GO2ARX5_REUSABLE_H200_INFERENCE_ADMISSION_PREFLIGHT_V1"
)
H200_INFERENCE_ADMISSION_STATUS = "PASS_STATIC_PREFLIGHT"
_LOCKED_FIELDS = (
    "Validation-E",
    "Holdout-F",
    "WBC",
    "simulator",
    "hardware",
    "new_data",
)


class H200AdmissionError(ReusableManifestError):
    """Raised when a future H200 runtime cannot be bound to its manifest."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise H200AdmissionError(message)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    try:
        return require_mapping(value, field=field)
    except ReusableManifestError as exc:
        raise H200AdmissionError(str(exc)) from exc


def _string(value: Any, *, field: str) -> str:
    try:
        return require_nonempty_string(value, field=field)
    except ReusableManifestError as exc:
        raise H200AdmissionError(str(exc)) from exc


def _strict_integer(value: Any, *, field: str) -> int:
    _require(type(value) is int, f"{field} must be a JSON integer")
    return value


def _strict_number(value: Any, *, field: str) -> float:
    _require(
        type(value) in (int, float),
        f"{field} must be an explicit finite JSON number",
    )
    result = float(value)
    _require(
        math.isfinite(result),
        f"{field} must be an explicit finite JSON number",
    )
    return result


def _manifest_file(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    _require(not candidate.is_symlink(), f"manifest may not be a symlink: {candidate}")
    try:
        candidate = candidate.resolve(strict=True)
    except OSError as exc:
        raise H200AdmissionError(f"manifest path cannot be resolved: {candidate}") from exc
    _require(candidate.is_file(), f"manifest must be a regular file: {candidate}")
    return candidate


def _resolve_path(value: Any, *, base_dir: Path, field: str) -> Path:
    raw = _string(value, field=field)
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    _require(not candidate.is_symlink(), f"{field} may not be a symlink: {candidate}")
    return candidate.resolve()


def _verify(artifact: Any, *, base_dir: Path, field: str) -> dict[str, Any]:
    try:
        return verify_artifact_identity(artifact, base_dir=base_dir, field=field)
    except ReusableManifestError as exc:
        raise H200AdmissionError(str(exc)) from exc


def _require_same_identity(
    expected: Mapping[str, Any], actual: Mapping[str, Any], *, field: str
) -> None:
    expected_kind = str(expected.get("kind", "file"))
    actual_kind = str(actual.get("kind", "file"))
    _require(actual_kind == expected_kind, f"{field}.kind differs from manifest")
    _require(
        str(actual.get("resolved_path")) == str(expected.get("resolved_path")),
        f"{field}.path differs from manifest",
    )
    _require(
        str(actual.get("sha256")) == str(expected.get("sha256")),
        f"{field}.sha256 differs from manifest",
    )


def _artifact_map(
    value: Any, *, labels: list[str], field: str
) -> dict[str, Mapping[str, Any]]:
    mapping = _mapping(value, field=field)
    normalized = {str(key): item for key, item in mapping.items()}
    _require(
        set(normalized) == set(labels),
        f"{field} must contain exactly the manifest labels",
    )
    result: dict[str, Mapping[str, Any]] = {}
    for label in labels:
        result[label] = _mapping(normalized[label], field=f"{field}.{label}")
    return result


def _fresh_output(path: Path) -> None:
    if not path.exists():
        return
    _require(path.is_dir() and not path.is_symlink(), "H200 output root is not a directory")
    _require(not any(path.iterdir()), "H200 output root must be fresh and empty")


def validate_h200_inference_admission(
    manifest: Mapping[str, Any],
    admission: Mapping[str, Any],
    *,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Cross-bind a materialized inference runtime without model execution.

    ``admission`` is the resolved, stage-specific runtime document.  Its
    ``runtime_late_bound`` values must exactly equal the concrete values in
    the manifest.  Checkpoint, statistics, cohort, and noise-bank entries are
    artifact objects so both path and content identity can be verified.
    """

    try:
        validate_evaluation_manifest(manifest)
    except ReusableManifestError as exc:
        raise H200AdmissionError(str(exc)) from exc
    _require(
        manifest.get("status") == GENERIC_EVAL_STATUS,
        "H200 admission requires a concrete READY_NOT_EXECUTED manifest",
    )
    _require(
        manifest["runtime"]["backend"] == INFERENCE_BACKEND_V1,
        f"manifest runtime backend must be {INFERENCE_BACKEND_V1}",
    )
    manifest_file = _manifest_file(manifest_path)
    base_dir = manifest_file.parent
    try:
        _require(
            load_json(manifest_file) == dict(manifest),
            "manifest mapping differs from the concrete manifest file",
        )
        manifest_verified = verify_evaluation_artifacts(
            manifest, manifest_path=manifest_file
        )
    except ReusableManifestError as exc:
        raise H200AdmissionError(str(exc)) from exc

    _require(
        admission.get("schema") == H200_INFERENCE_ADMISSION_SCHEMA,
        "H200 inference admission schema mismatch",
    )
    _require(
        admission.get("backend") == INFERENCE_BACKEND_V1,
        f"H200 admission backend must be {INFERENCE_BACKEND_V1}",
    )
    _require(
        admission.get("runtime_identity") == manifest["runtime"]["identity"],
        "admitted runtime identity differs from manifest",
    )
    runtime = _mapping(
        admission.get("runtime_late_bound"), field="admission.runtime_late_bound"
    )
    try:
        resolved_runtime = validate_late_bound_runtime(
            runtime,
            field="admission.runtime_late_bound",
            allow_sentinel=False,
        )
        manifest_runtime = validate_late_bound_runtime(
            manifest["runtime"].get("late_bound"),
            field="manifest.runtime.late_bound",
            allow_sentinel=False,
        )
    except ReusableManifestError as exc:
        raise H200AdmissionError(str(exc)) from exc
    _require(
        resolved_runtime == manifest_runtime,
        "admitted runtime late-bound values differ from the manifest",
    )
    execution_head = _string(
        resolved_runtime["repository_execution_head"],
        field="admission.runtime_late_bound.repository_execution_head",
    )
    _require(
        execution_head.lower() == str(manifest["experiment"]["execution_head"]).lower(),
        "admitted repository execution HEAD differs from the manifest",
    )
    _require(
        resolved_runtime["offline_mode"] is True,
        "H200 inference admission requires offline_mode=true",
    )
    admission_identity = _string(
        resolved_runtime["runtime_admission_identity"],
        field="admission.runtime_late_bound.runtime_admission_identity",
    )
    _require(
        admission_identity
        == str(manifest_runtime["runtime_admission_identity"]),
        "runtime admission identity differs from the manifest",
    )

    output_root = _resolve_path(
        resolved_runtime["output_root"], base_dir=base_dir, field="admission.output_root"
    )
    manifest_output_root = _resolve_path(
        manifest["output"]["root"], base_dir=base_dir, field="manifest.output.root"
    )
    _require(output_root == manifest_output_root, "admitted output root differs from manifest")
    _require(
        admission.get("output_fresh") is True,
        "H200 admission must explicitly assert a fresh output root",
    )
    _fresh_output(output_root)

    checkpoints = [str(item["label"]) for item in manifest["checkpoints"]]
    admitted_checkpoints = _artifact_map(
        resolved_runtime["checkpoint_paths"],
        labels=checkpoints,
        field="admission.runtime_late_bound.checkpoint_paths",
    )
    admitted_statistics = _artifact_map(
        resolved_runtime["statistics_path"],
        labels=checkpoints,
        field="admission.runtime_late_bound.statistics_path",
    )
    verified_by_label = {
        str(item["label"]): item for item in manifest_verified["checkpoints"]
    }
    for label in checkpoints:
        actual_checkpoint = _verify(
            admitted_checkpoints[label],
            base_dir=base_dir,
            field=f"admission.checkpoint_paths.{label}",
        )
        _require_same_identity(
            verified_by_label[label]["identity"],
            actual_checkpoint,
            field=f"admission.checkpoint_paths.{label}",
        )
        actual_statistics = _verify(
            admitted_statistics[label],
            base_dir=base_dir,
            field=f"admission.statistics_path.{label}",
        )
        _require_same_identity(
            verified_by_label[label]["statistics"],
            actual_statistics,
            field=f"admission.statistics_path.{label}",
        )

    actual_cohort = _verify(
        resolved_runtime["dataset_path"],
        base_dir=base_dir,
        field="admission.runtime_late_bound.dataset_path",
    )
    _require_same_identity(
        manifest_verified["cohort"], actual_cohort, field="admission.dataset_path"
    )
    cohort = _mapping(admission.get("cohort"), field="admission.cohort")
    _require(
        cohort.get("name") == manifest["cohort"]["name"],
        "admitted cohort name differs from manifest",
    )
    _require(
        cohort.get("split") == manifest["cohort"]["split"],
        "admitted cohort split differs from manifest",
    )
    _require(
        list(cohort.get("window_ids", []))
        == [str(item["id"]) for item in manifest["cohort"]["windows"]],
        "admitted cohort window order differs from manifest",
    )
    _require(
        list(cohort.get("target_ids", []))
        == [str(item["target_id"]) for item in manifest["cohort"]["windows"]],
        "admitted cohort target order differs from manifest",
    )

    sampler = _mapping(admission.get("sampler"), field="admission.sampler")
    profile_name = _string(
        sampler.get("profile"), field="admission.sampler.profile"
    )
    _require(
        profile_name == manifest["sampler"]["profile"],
        "admitted sampler profile differs from manifest",
    )
    _require(
        admission.get("timestamp_profile") == manifest["timestamp_profile"],
        "admitted timestamp profile differs from manifest",
    )
    admitted_timesteps = sampler.get("timesteps")
    _require(
        isinstance(admitted_timesteps, list),
        "admission.sampler.timesteps must be an explicit list",
    )
    for index, value in enumerate(admitted_timesteps):
        _strict_integer(value, field=f"admission.sampler.timesteps[{index}]")
    _require(
        admitted_timesteps == manifest["sampler"]["timesteps"],
        "admitted sampler timesteps differs from manifest",
    )
    admitted_dt = _strict_number(sampler.get("dt"), field="admission.sampler.dt")
    _require(
        admitted_dt == manifest["sampler"]["dt"],
        "admitted sampler dt differs from manifest",
    )
    admitted_calls = _strict_integer(
        sampler.get("expected_calls_per_window"),
        field="admission.sampler.expected_calls_per_window",
    )
    _require(
        admitted_calls == manifest["sampler"]["expected_calls_per_window"],
        "admitted sampler call count differs from manifest",
    )
    try:
        sampler_profile(profile_name)
    except ProfileError as exc:
        raise H200AdmissionError(str(exc)) from exc
    _require(
        sampler.get("same_noise_across_checkpoints") is True,
        "H200 admission must preserve same-noise pairing",
    )
    expected_noise = {
        "kind": "file",
        "path": manifest["sampler"]["noise_bank"],
        "sha256": manifest["sampler"]["noise_bank_sha256"],
    }
    actual_noise = _verify(
        sampler.get("noise_bank"), base_dir=base_dir, field="admission.sampler.noise_bank"
    )
    expected_noise_verified = _verify(
        expected_noise, base_dir=base_dir, field="manifest.sampler.noise_bank"
    )
    _require_same_identity(
        expected_noise_verified, actual_noise, field="admission.sampler.noise_bank"
    )

    manifest_locked = _mapping(
        manifest.get("locked_boundaries"), field="manifest.locked_boundaries"
    )
    admitted_locked = _mapping(
        admission.get("locked_boundaries"), field="admission.locked_boundaries"
    )
    for key in _LOCKED_FIELDS:
        _require(manifest_locked.get(key) == "LOCKED", f"{key} is not locked in manifest")
        _require(admitted_locked.get(key) == "LOCKED", f"{key} is not locked in admission")
    try:
        reject_locked_data(admitted_locked)
    except ReusableManifestError as exc:
        raise H200AdmissionError(str(exc)) from exc

    return {
        "schema": H200_INFERENCE_ADMISSION_SCHEMA,
        "status": H200_INFERENCE_ADMISSION_STATUS,
        "backend_boundary": INFERENCE_BACKEND_V1,
        "execution_head": execution_head,
        "runtime_late_bound_fields": list(RUNTIME_LATE_BOUND_FIELDS),
        "runtime_late_bound": dict(resolved_runtime),
        "cohort": {
            "name": manifest["cohort"]["name"],
            "split": manifest["cohort"]["split"],
            "window_ids": [str(item["id"]) for item in manifest["cohort"]["windows"]],
            "identity": dict(manifest_verified["cohort"]),
        },
        "checkpoint_identities": [
            {
                "label": label,
                "identity": dict(verified_by_label[label]["identity"]),
                "statistics": dict(verified_by_label[label]["statistics"]),
            }
            for label in checkpoints
        ],
        "sampler_profile": profile_name,
        "noise_bank": dict(actual_noise),
        "output_root": str(output_root),
        "output_fresh": True,
        "locked_boundaries": {key: "LOCKED" for key in _LOCKED_FIELDS},
        "model_imported": False,
        "model_construction": False,
        "model_forward": False,
        "training": False,
        "gpu_h200": False,
        "pbs_submission": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "wbc_executed": False,
        "simulator_executed": False,
        "hardware_executed": False,
        "canary_execution": "PI_REVIEW_REQUIRED",
    }


__all__ = [
    "H200_INFERENCE_ADMISSION_SCHEMA",
    "H200_INFERENCE_ADMISSION_STATUS",
    "H200AdmissionError",
    "validate_h200_inference_admission",
]
