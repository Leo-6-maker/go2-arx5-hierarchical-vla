"""Manifest-driven, runtime-independent checkpoint cohort evaluation.

The generic evaluator owns the stable artifact boundary:

``manifest -> raw prediction handoff -> persist raw -> accepted decode -> metrics``

It never constructs a model and never imports Torch, GR00T, CUDA, PBS, WBC,
the simulator, or hardware control code.  A stage-specific runtime may produce
the raw-prediction JSONL handoff; this module then performs the same deterministic
CPU-side decode and metric path for every future experiment.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..experiments.reusable import (
    INFERENCE_BACKEND_V1,
    RUNTIME_LATE_BOUND_FIELDS,
    TEMPLATE_SENTINEL,
    ReusableManifestError,
    load_json,
    reject_locked_data,
    require_mapping,
    require_nonempty_string,
    require_sha256,
    sha256_file,
    validate_late_bound_runtime,
    verify_artifact_identity,
    write_json,
)
from ..experiments.profiles import (
    ProfileError,
    SamplerProfile,
    TimestampProfile,
    sampler_profile,
    timestamp_profile,
)
from .structured_action_eval import (
    EXPECTED_MODEL_INPUT_KEYS,
    PHYSICAL_ACTION_SHAPE,
    RAW_ACTION_SHAPE,
    STATE_V1_SHAPE,
    DecodedStructuredPrediction,
    decode_raw_prediction,
    validate_model_input_contract,
)
from .trajectory_metrics import (
    aggregate_trajectory_metrics,
    array_sha256,
    comparison_metrics,
    numeric_summary,
)

GENERIC_EVALUATION_MANIFEST_SCHEMA = (
    "GO2ARX5_REUSABLE_CHECKPOINT_EVALUATION_MANIFEST_V1"
)
GENERIC_PREDICTION_RECORD_SCHEMA = "GO2ARX5_REUSABLE_RAW_PREDICTION_RECORD_V1"
GENERIC_EVALUATION_ROW_SCHEMA = "GO2ARX5_REUSABLE_EVALUATION_ROW_V1"
GENERIC_EVALUATION_OUTPUT_SCHEMA = "GO2ARX5_REUSABLE_EVALUATION_OUTPUT_V1"
GENERIC_EVALUATION_PLAN_SCHEMA = "GO2ARX5_REUSABLE_EVALUATION_PLAN_V1"
GENERIC_EVAL_STATUS = "READY_NOT_EXECUTED"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class ReusableEvaluationError(ReusableManifestError):
    """Raised when a generic evaluation manifest or handoff is inadmissible."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReusableEvaluationError(message)


def _finite_sequence(value: Any, *, field: str, shape: tuple[int, ...]) -> list[float]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ReusableEvaluationError(f"{field} is not numeric") from exc
    _require(
        tuple(array.shape) == shape,
        f"{field} must have shape {shape}, got {array.shape}",
    )
    _require(bool(np.isfinite(array).all()), f"{field} contains non-finite values")
    return [float(item) for item in array.reshape(-1)]


def _require_digest(value: Any, *, field: str) -> str:
    digest = require_sha256(value, field=field)
    _require(
        bool(_SHA256_RE.fullmatch(digest)),
        f"{field} is not lowercase hexadecimal SHA256",
    )
    return digest


def _path_from(value: Any, *, base_dir: Path, field: str) -> Path:
    raw = require_nonempty_string(value, field=field)
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    if candidate.is_symlink():
        raise ReusableEvaluationError(f"{field} may not be a symlink: {candidate}")
    return candidate.resolve()


def _load_manifest_document(path: str | Path) -> tuple[dict[str, Any], Path]:
    manifest_path = Path(path).expanduser()
    if manifest_path.is_symlink():
        raise ReusableEvaluationError(f"manifest may not be a symlink: {manifest_path}")
    manifest_path = manifest_path.resolve(strict=True)
    value = load_json(manifest_path)
    return value, manifest_path


def _validate_identity_artifact(value: Any, *, field: str) -> None:
    artifact = require_mapping(value, field=field)
    require_nonempty_string(artifact.get("path"), field=f"{field}.path")
    _require_digest(artifact.get("sha256"), field=f"{field}.sha256")
    kind = str(artifact.get("kind", "file"))
    _require(kind in {"file", "tree"}, f"{field}.kind must be file or tree")


def _validate_checkpoint(checkpoint: Any, *, index: int) -> None:
    field = f"checkpoints[{index}]"
    item = require_mapping(checkpoint, field=field)
    label = require_nonempty_string(item.get("label"), field=f"{field}.label")
    _require(label not in {"", "UNKNOWN"}, f"{field}.label is not usable")
    _validate_identity_artifact(item.get("identity"), field=f"{field}.identity")
    _validate_identity_artifact(
        item.get("runtime_mapping"), field=f"{field}.runtime_mapping"
    )
    _validate_identity_artifact(item.get("statistics"), field=f"{field}.statistics")


def _is_template_placeholder(value: Any) -> bool:
    return isinstance(value, str) and (
        value == TEMPLATE_SENTINEL or value.startswith("__REQUIRED_")
    )


def _resolve_timestamp_profile(
    value: Any, *, field: str, template: bool
) -> TimestampProfile | None:
    name = require_nonempty_string(value, field=field)
    if template and _is_template_placeholder(name):
        return None
    try:
        return timestamp_profile(name)
    except ProfileError as exc:
        raise ReusableEvaluationError(str(exc)) from exc


def _resolve_sampler_profile(
    value: Any, *, field: str, template: bool
) -> SamplerProfile | None:
    name = require_nonempty_string(value, field=field)
    if template and _is_template_placeholder(name):
        return None
    try:
        return sampler_profile(name)
    except ProfileError as exc:
        raise ReusableEvaluationError(str(exc)) from exc


def _validate_window(
    window: Any,
    *,
    index: int,
    timestamp_policy: TimestampProfile | None,
    template: bool,
) -> None:
    field = f"cohort.windows[{index}]"
    item = require_mapping(window, field=field)
    require_nonempty_string(item.get("id"), field=f"{field}.id")
    require_nonempty_string(item.get("target_id"), field=f"{field}.target_id")
    if template and any(
        _is_template_placeholder(item.get(name))
        for name in (
            "anchor_world_m",
            "target_world_m",
            "source_timestamps",
            "observation_timestamp",
        )
    ):
        return
    _finite_sequence(
        item.get("anchor_world_m"), field=f"{field}.anchor_world_m", shape=(3,)
    )
    _finite_sequence(
        item.get("target_world_m"), field=f"{field}.target_world_m", shape=(3,)
    )
    _finite_sequence(
        item.get("source_timestamps"),
        field=f"{field}.source_timestamps",
        shape=(40,),
    )
    timestamps = np.asarray(item["source_timestamps"], dtype=np.float64)
    _require(
        bool(np.all(np.diff(timestamps) > 0.0)),
        f"{field}.source_timestamps must be strictly increasing",
    )
    if timestamp_policy is not None and timestamp_policy.cadence_hz is not None:
        spacing = 1.0 / timestamp_policy.cadence_hz
        _require(
            bool(np.allclose(np.diff(timestamps), spacing, rtol=0.0, atol=1.0e-9)),
            f"{field}.source_timestamps do not match {timestamp_policy.name}",
        )
        if timestamp_policy.duration_s is not None:
            _require(
                bool(
                    np.isclose(
                        timestamps[-1] - timestamps[0],
                        timestamp_policy.duration_s,
                        rtol=0.0,
                        atol=1.0e-9,
                    )
                ),
                f"{field}.source_timestamps duration does not match "
                f"{timestamp_policy.name}",
            )
    timestamp = float(item.get("observation_timestamp", float("nan")))
    _require(math.isfinite(timestamp), f"{field}.observation_timestamp must be finite")


def validate_evaluation_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate a future-facing manifest without touching model/runtime code."""

    _require(
        manifest.get("schema") == GENERIC_EVALUATION_MANIFEST_SCHEMA,
        "generic evaluation manifest schema mismatch",
    )
    status = manifest.get("status")
    _require(
        status in {"TEMPLATE_NOT_EXECUTABLE", GENERIC_EVAL_STATUS},
        "generic evaluation manifest status is invalid",
    )
    template = status == "TEMPLATE_NOT_EXECUTABLE"
    experiment = require_mapping(manifest.get("experiment"), field="experiment")
    require_nonempty_string(experiment.get("id"), field="experiment.id")
    require_nonempty_string(experiment.get("stage"), field="experiment.stage")
    execution_head = require_nonempty_string(
        experiment.get("execution_head"), field="experiment.execution_head"
    )
    _require(
        bool(re.fullmatch(r"[0-9a-fA-F]{40}", execution_head)),
        "execution_head must be a Git SHA",
    )
    if not template:
        _require(
            execution_head.lower() != "0" * 40,
            "execution_head must identify a concrete execution revision",
        )
    _require_digest(
        experiment.get("protocol_sha256"), field="experiment.protocol_sha256"
    )

    timestamp_policy = _resolve_timestamp_profile(
        manifest.get("timestamp_profile"),
        field="timestamp_profile",
        template=template,
    )

    cohort = require_mapping(manifest.get("cohort"), field="cohort")
    require_nonempty_string(cohort.get("name"), field="cohort.name")
    require_nonempty_string(cohort.get("split"), field="cohort.split")
    _require(
        cohort.get("window_order") == "explicit_manifest_order",
        "cohort.window_order must state explicit_manifest_order",
    )
    windows = cohort.get("windows")
    _require(
        isinstance(windows, list) and bool(windows), "cohort.windows must be non-empty"
    )
    for index, window in enumerate(windows):
        _validate_window(
            window,
            index=index,
            timestamp_policy=timestamp_policy,
            template=template,
        )
    window_ids = [str(item["id"]) for item in windows]
    _require(
        len(set(window_ids)) == len(window_ids), "cohort window IDs must be unique"
    )

    checkpoints = manifest.get("checkpoints")
    _require(
        isinstance(checkpoints, list) and bool(checkpoints),
        "checkpoints must be non-empty",
    )
    for index, checkpoint in enumerate(checkpoints):
        _validate_checkpoint(checkpoint, index=index)
    checkpoint_labels = [str(item["label"]) for item in checkpoints]
    _require(
        len(set(checkpoint_labels)) == len(checkpoint_labels),
        "checkpoint labels must be unique and ordered by the manifest",
    )

    contract = require_mapping(manifest.get("contract"), field="contract")
    _require(
        list(contract.get("raw_action_shape", [])) == list(RAW_ACTION_SHAPE),
        "raw action ABI drift",
    )
    _require(
        list(contract.get("decoded_action_shape", [])) == list(PHYSICAL_ACTION_SHAPE),
        "decoded physical action ABI drift",
    )
    _require(
        list(contract.get("state_v1_shape", [])) == list(STATE_V1_SHAPE),
        "StateV1 ABI drift",
    )
    _require(
        tuple(contract.get("model_input_keys", [])) == EXPECTED_MODEL_INPUT_KEYS,
        "model input ABI must be body_rgb,wrist_rgb,state",
    )
    _require(
        contract.get("target_xyz_in_model_input") is False,
        "target XYZ input boundary drift",
    )
    _require(
        contract.get("raw_before_decode") is True,
        "raw-before-decode ordering must be explicit",
    )
    _require(
        contract.get("decoder")
        == "go2arx5_vla.stage40_structured_decode.decode_raw_structured_action",
        "decoder authority drift",
    )
    _require(
        contract.get("inverse_normalization") == "manifest_bound_stats",
        "stats binding is not explicit",
    )
    _require(contract.get("geometry") == "K16", "geometry contract must be K16")
    _require(contract.get("timing") == "T16", "timing contract must be T16")

    sampler = require_mapping(manifest.get("sampler"), field="sampler")
    sampler_policy = _resolve_sampler_profile(
        sampler.get("profile"), field="sampler.profile", template=template
    )
    if sampler_policy is not None:
        timesteps = sampler.get("timesteps")
        _require(
            isinstance(timesteps, list)
            and [int(item) for item in timesteps]
            == list(sampler_policy.timesteps),
            f"sampler timesteps drift from {sampler_policy.name}",
        )
        _require(
            math.isclose(
                float(sampler.get("dt", float("nan"))),
                sampler_policy.dt,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"sampler dt drift from {sampler_policy.name}",
        )
        _require(
            int(sampler.get("expected_calls_per_window", -1))
            == sampler_policy.calls_per_window,
            f"sampler call count drift from {sampler_policy.name}",
        )
    _require(
        template or sampler_policy is not None,
        "sampler.profile must resolve to a versioned profile",
    )
    if not (template and sampler_policy is None):
        _require(
            sampler.get("same_noise_across_checkpoints") is True,
            "same-noise pairing must be explicit",
        )
        require_nonempty_string(
            sampler.get("noise_bank"), field="sampler.noise_bank"
        )
        _require_digest(
            sampler.get("noise_bank_sha256"), field="sampler.noise_bank_sha256"
        )

    runtime = require_mapping(manifest.get("runtime"), field="runtime")
    require_nonempty_string(runtime.get("adapter"), field="runtime.adapter")
    backend = require_nonempty_string(runtime.get("backend"), field="runtime.backend")
    if not (template and _is_template_placeholder(backend)):
        _require(
            backend == INFERENCE_BACKEND_V1,
            f"runtime.backend must be {INFERENCE_BACKEND_V1}",
        )
    _require(
        runtime.get("offline") is True,
        "generic evaluation runtime must be offline/local",
    )
    _require(
        runtime.get("no_model_construction") is True,
        "generic evaluator must not construct a model",
    )
    _require(
        runtime.get("no_training") is True, "generic evaluator must be evaluation-only"
    )
    require_nonempty_string(runtime.get("identity"), field="runtime.identity")
    try:
        validate_late_bound_runtime(
            runtime.get("late_bound"),
            field="runtime.late_bound",
            allow_sentinel=template,
        )
    except ReusableManifestError as exc:
        raise ReusableEvaluationError(str(exc)) from exc

    output = require_mapping(manifest.get("output"), field="output")
    require_nonempty_string(output.get("root"), field="output.root")
    require_nonempty_string(
        output.get("raw_predictions_manifest"), field="output.raw_predictions_manifest"
    )
    _require(output.get("fresh") is True, "evaluation output must require a fresh root")

    locked = require_mapping(
        manifest.get("locked_boundaries"), field="locked_boundaries"
    )
    for key in (
        "Validation-E",
        "Holdout-F",
        "WBC",
        "simulator",
        "hardware",
        "new_data",
    ):
        _require(locked.get(key) == "LOCKED", f"{key} must remain LOCKED")

    analysis_value = manifest.get("analysis", {})
    if analysis_value is None:
        analysis_value = {}
    analysis = require_mapping(analysis_value, field="analysis")
    pairs = analysis.get("paired_comparisons", [])
    if pairs is None:
        pairs = []
    _require(
        isinstance(pairs, list), "analysis.paired_comparisons must be a list"
    )
    for index, pair in enumerate(pairs):
        item = require_mapping(pair, field=f"analysis.paired_comparisons[{index}]")
        require_nonempty_string(
            item.get("name"), field=f"analysis.paired_comparisons[{index}].name"
        )
        left = require_nonempty_string(
            item.get("left"), field=f"analysis.paired_comparisons[{index}].left"
        )
        right = require_nonempty_string(
            item.get("right"), field=f"analysis.paired_comparisons[{index}].right"
        )
        _require(
            left in checkpoint_labels and right in checkpoint_labels,
            "analysis pair references an unknown checkpoint",
        )

    reject_locked_data(manifest)


def load_evaluation_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a generic evaluation manifest."""

    value, _ = _load_manifest_document(path)
    validate_evaluation_manifest(value)
    return value


def verify_evaluation_artifacts(
    manifest: Mapping[str, Any], *, manifest_path: str | Path
) -> dict[str, Any]:
    """Verify every checkpoint and statistics identity named by a manifest."""

    base_dir = Path(manifest_path).expanduser().resolve(strict=True).parent
    verified: dict[str, Any] = {"checkpoints": []}
    for index, checkpoint in enumerate(manifest["checkpoints"]):
        item = dict(checkpoint)
        item["identity"] = verify_artifact_identity(
            checkpoint["identity"],
            base_dir=base_dir,
            field=f"checkpoints[{index}].identity",
        )
        item["runtime_mapping"] = verify_artifact_identity(
            checkpoint["runtime_mapping"],
            base_dir=base_dir,
            field=f"checkpoints[{index}].runtime_mapping",
        )
        item["statistics"] = verify_artifact_identity(
            checkpoint["statistics"],
            base_dir=base_dir,
            field=f"checkpoints[{index}].statistics",
        )
        verified["checkpoints"].append(item)
    noise_bank = manifest["sampler"].get("noise_bank")
    noise_path = _path_from(noise_bank, base_dir=base_dir, field="sampler.noise_bank")
    noise_digest = _require_digest(
        manifest["sampler"].get("noise_bank_sha256"), field="sampler.noise_bank_sha256"
    )
    _require(
        noise_path.is_file() and not noise_path.is_symlink(),
        f"noise bank is not a regular file: {noise_path}",
    )
    actual_noise_digest = sha256_file(noise_path)
    _require(actual_noise_digest == noise_digest, "sampler noise bank SHA256 mismatch")
    verified["noise_bank"] = {
        "path": str(noise_path),
        "sha256": noise_digest,
        "verified": True,
    }
    return verified


def build_evaluation_plan(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Build a zero-inference admission plan from explicit manifest fields."""

    validate_evaluation_manifest(manifest)
    _require(
        manifest.get("status") == GENERIC_EVAL_STATUS,
        "evaluation manifest template is not executable",
    )
    checkpoint_count = len(manifest["checkpoints"])
    window_count = len(manifest["cohort"]["windows"])
    calls_per_window = int(manifest["sampler"]["expected_calls_per_window"])
    return {
        "schema": GENERIC_EVALUATION_PLAN_SCHEMA,
        "experiment_id": manifest["experiment"]["id"],
        "execution_head": manifest["experiment"]["execution_head"],
        "checkpoint_order": [item["label"] for item in manifest["checkpoints"]],
        "checkpoint_identities": [
            {
                "label": item["label"],
                "identity": dict(item["identity"]),
                "runtime_mapping": dict(item["runtime_mapping"]),
                "statistics": dict(item["statistics"]),
            }
            for item in manifest["checkpoints"]
        ],
        "window_order": [item["id"] for item in manifest["cohort"]["windows"]],
        "window_count": window_count,
        "checkpoint_count": checkpoint_count,
        "raw_prediction_count": checkpoint_count * window_count,
        "sampler_calls": checkpoint_count * window_count * calls_per_window,
        "timestamp_profile": manifest["timestamp_profile"],
        "sampler_profile": manifest["sampler"]["profile"],
        "sampler_timesteps": list(manifest["sampler"]["timesteps"]),
        "same_noise_across_checkpoints": True,
        "backend_boundary": INFERENCE_BACKEND_V1,
        "late_bound_runtime_fields": list(RUNTIME_LATE_BOUND_FIELDS),
        "backend_handoff": "external_raw_prediction_handoff_required",
        "model_construction": False,
        "model_forward": "external_stage_adapter_only",
        "training": False,
        "backward": False,
        "optimizer": False,
        "resume": False,
        "checkpoint_write": False,
        "Validation-E": "LOCKED",
        "Holdout-F": "LOCKED",
        "WBC": "LOCKED",
        "simulator": "LOCKED",
        "hardware": "LOCKED",
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _durable_json(path: str | Path, value: Any) -> None:
    """Write JSON and force it to stable storage before returning."""

    write_json(path, value)
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _durable_text(path: str | Path, value: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _persist_array(array: np.ndarray, destination: Path) -> dict[str, Any]:
    """Persist one decoded array before its row is appended to evidence."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "array_sha256": array_sha256(array),
        "shape": list(array.shape),
    }


def _append_jsonl(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(_jsonable(value), sort_keys=True, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _scalar_metric_values(
    value: Any, *, prefix: str = ""
) -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_scalar_metric_values(child, prefix=child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            result[prefix] = float(value)
    return result


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    metric_values = [
        _scalar_metric_values(row["metrics"]) for row in rows if "metrics" in row
    ]
    metric_names = sorted({name for values in metric_values for name in values})
    fieldnames = [
        "checkpoint_label",
        "checkpoint_sha256",
        "window_id",
        "target_id",
        *metric_names,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, values in zip(rows, metric_values):
            output = {
                "checkpoint_label": row["checkpoint"]["label"],
                "checkpoint_sha256": row["checkpoint"]["sha256"],
                "window_id": row["window_id"],
                "target_id": row["target_id"],
            }
            output.update({name: values.get(name, "") for name in metric_names})
            writer.writerow(output)
        handle.flush()
        os.fsync(handle.fileno())


def _write_artifact_inventory(output: Path) -> tuple[Path, str]:
    """Inventory all completed evidence except the inventory and run manifest."""

    inventory_path = output / "artifact_inventory_sha256.txt"
    lines: list[str] = []
    for candidate in sorted(output.rglob("*")):
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or candidate == inventory_path
            or candidate == output / "run_manifest.json"
        ):
            continue
        relative = candidate.relative_to(output).as_posix()
        lines.append(f"{sha256_file(candidate)}  {relative}\n")
    _durable_text(inventory_path, "".join(lines))
    return inventory_path, sha256_file(inventory_path)


def _load_records(path: str | Path) -> tuple[list[dict[str, Any]], Path]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ReusableEvaluationError(
            f"raw prediction manifest may not be a symlink: {candidate}"
        )
    candidate = candidate.resolve(strict=True)
    if not candidate.is_file():
        raise ReusableEvaluationError(
            f"raw prediction manifest is not a file: {candidate}"
        )
    if candidate.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            candidate.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReusableEvaluationError(
                    f"invalid JSONL at line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ReusableEvaluationError(
                    f"prediction record at line {line_number} is not an object"
                )
            records.append(value)
        return records, candidate
    try:
        document = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReusableEvaluationError(
            f"cannot read raw prediction JSON: {candidate}"
        ) from exc
    records_value = (
        document
        if isinstance(document, list)
        else document.get("records")
        if isinstance(document, dict)
        else None
    )
    if not isinstance(records_value, list):
        raise ReusableEvaluationError("raw prediction JSON must contain records: [...]")
    if any(not isinstance(item, dict) for item in records_value):
        raise ReusableEvaluationError("every raw prediction record must be an object")
    return [dict(item) for item in records_value], candidate


def load_prediction_records(path: str | Path) -> list[dict[str, Any]]:
    """Public loader for JSON or JSONL raw-prediction handoffs."""

    records, _ = _load_records(path)
    return records


def _safe_segment(value: str) -> str:
    result = _SAFE_SEGMENT_RE.sub("_", str(value)).strip("._")
    return result or "item"


def _array_from_record(
    record: Mapping[str, Any],
    key: str,
    *,
    shape: tuple[int, ...],
    base_dir: Path,
) -> np.ndarray:
    value = record.get(key)
    if isinstance(value, Mapping) and "path" in value:
        artifact = verify_artifact_identity(
            value, base_dir=base_dir, field=f"record.{key}"
        )
        path = Path(artifact["resolved_path"])
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ReusableEvaluationError(f"cannot load record.{key}: {path}") from exc
    else:
        array = np.asarray(value)
    if (
        tuple(array.shape) != shape
        or not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
    ):
        raise ReusableEvaluationError(f"record.{key} must be finite numeric {shape}")
    return array


def _raw_from_record(
    record: Mapping[str, Any], *, base_dir: Path
) -> tuple[np.ndarray, Path, str]:
    raw_path = _path_from(
        record.get("raw_path"), base_dir=base_dir, field="record.raw_path"
    )
    expected_file_digest = _require_digest(
        record.get("raw_sha256"), field="record.raw_sha256"
    )
    actual_file_digest = sha256_file(raw_path)
    _require(
        actual_file_digest == expected_file_digest,
        "raw prediction file SHA256 mismatch",
    )
    try:
        raw = np.load(raw_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ReusableEvaluationError(
            f"cannot load raw prediction array: {raw_path}"
        ) from exc
    if (
        tuple(raw.shape) != RAW_ACTION_SHAPE
        or not np.issubdtype(raw.dtype, np.number)
        or not np.isfinite(raw).all()
    ):
        raise ReusableEvaluationError(
            f"raw prediction must be finite numeric {RAW_ACTION_SHAPE}"
        )
    expected_array_digest = _require_digest(
        record.get("raw_array_sha256"), field="record.raw_array_sha256"
    )
    actual_array_digest = array_sha256(raw)
    _require(
        actual_array_digest == expected_array_digest,
        "raw prediction array SHA256 mismatch",
    )
    return raw, raw_path, actual_file_digest


def _checkpoint_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item["label"]): item for item in manifest["checkpoints"]}


def _window_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(item["id"]): item for item in manifest["cohort"]["windows"]}


def _validate_record_contract(
    record: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any],
    window: Mapping[str, Any],
    base_dir: Path,
) -> None:
    _require(
        record.get("schema") == GENERIC_PREDICTION_RECORD_SCHEMA,
        "raw prediction record schema mismatch",
    )
    _require(
        str(record.get("checkpoint_label")) == str(checkpoint["label"]),
        "checkpoint order/label drift",
    )
    _require(
        str(record.get("checkpoint_sha256")) == str(checkpoint["identity"]["sha256"]),
        "checkpoint identity drift",
    )
    _require(
        str(record.get("window_id")) == str(window["id"]), "window order/identity drift"
    )
    validate_model_input_contract(
        record.get("model_input_keys", ()),
        target_xyz_in_model_input=bool(record.get("target_xyz_in_model_input", True)),
    )
    _require(
        record.get("target_xyz_in_model_input") is False,
        "target XYZ must not enter model input",
    )
    _require_digest(record.get("raw_sha256"), field="record.raw_sha256")
    _require_digest(record.get("raw_array_sha256"), field="record.raw_array_sha256")
    _require_digest(record.get("noise_sha256"), field="record.noise_sha256")
    observations = require_mapping(
        record.get("observations"), field="record.observations"
    )
    _require(
        set(observations) == set(EXPECTED_MODEL_INPUT_KEYS[:2]),
        "both RGB modalities must be explicit",
    )
    for key in EXPECTED_MODEL_INPUT_KEYS[:2]:
        _validate_identity_artifact(
            observations.get(key), field=f"record.observations.{key}"
        )
        _require(
            list(observations[key].get("shape", [])) == [96, 96, 3],
            f"record.observations.{key}.shape must be [96,96,3]",
        )
        _require(
            observations[key].get("dtype") == "uint8",
            f"record.observations.{key}.dtype must be uint8",
        )
    _array_from_record(record, "state_v1", shape=STATE_V1_SHAPE, base_dir=base_dir)
    _array_from_record(
        record, "expert_physical_action", shape=PHYSICAL_ACTION_SHAPE, base_dir=base_dir
    )


def _same_noise_guard(
    records: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
) -> None:
    if manifest["sampler"].get("same_noise_across_checkpoints") is not True:
        return
    by_window: dict[str, set[str]] = {}
    for record in records:
        by_window.setdefault(str(record["window_id"]), set()).add(
            str(record["noise_sha256"])
        )
    _require(
        all(len(values) == 1 for values in by_window.values()),
        "same-noise pairing drift across checkpoint evaluations",
    )


def _persist_raw(
    source: Path,
    destination: Path,
    *,
    raw_manifest_handle: Any,
    record: Mapping[str, Any],
    raw: np.ndarray,
    source_sha256: str,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("wb") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    destination_sha256 = sha256_file(destination)
    _require(
        destination_sha256 == source_sha256,
        "persisted raw artifact changed during copy",
    )
    raw_record = {
        "schema": GENERIC_PREDICTION_RECORD_SCHEMA,
        "checkpoint_label": record["checkpoint_label"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "window_id": record["window_id"],
        "noise_sha256": record["noise_sha256"],
        "source_path": str(source),
        "path": str(destination),
        "sha256": destination_sha256,
        "array_sha256": array_sha256(raw),
        "shape": list(raw.shape),
        "raw_persisted_before_decode": True,
    }
    raw_manifest_handle.write(
        json.dumps(_jsonable(raw_record), sort_keys=True, allow_nan=False) + "\n"
    )
    raw_manifest_handle.flush()
    os.fsync(raw_manifest_handle.fileno())
    return raw_record


def _aggregate_checkpoint_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ReusableEvaluationError("cannot aggregate an empty checkpoint")
    scalar_fields = (
        "position_rmse_m",
        "position_max_abs_m",
        "endpoint_error_m",
        "prediction_path_length_m",
        "expert_path_length_m",
        "eef_rot6d_rmse",
        "eef_rot6d_max_abs",
        "gripper_rmse",
        "gripper_max_abs",
        "full_action_rmse_mixed_units",
        "roughness_ratio",
    )
    metrics = {
        field: numeric_summary(
            float(row["metrics"][field])
            for row in rows
            if row["metrics"].get(field) is not None
        )
        for field in scalar_fields
        if any(row["metrics"].get(field) is not None for row in rows)
    }
    return {
        "schema": "GO2ARX5_REUSABLE_CHECKPOINT_AGGREGATE_V1",
        "checkpoint": dict(rows[0]["checkpoint"]),
        "window_count": len(rows),
        "window_ids": [str(row["window_id"]) for row in rows],
        "raw_prediction_array_sha256": [
            row["raw_prediction"]["array_sha256"] for row in rows
        ],
        "decoded_action_array_sha256": [
            row["decoded_action"]["array_sha256"] for row in rows
        ],
        "target_metrics": aggregate_trajectory_metrics(
            row["metrics"]["target"] for row in rows
        ),
        "comparison_metrics": metrics,
        "raw_predictions_preserved_separately": True,
        "threshold_used": None,
    }


def run_evaluation(
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
    records_base_dir: str | Path | None = None,
    verify_artifacts: bool = False,
    decoder: Callable[..., DecodedStructuredPrediction] = decode_raw_prediction,
) -> dict[str, Any]:
    """Execute the generic raw-to-metric path with no model construction.

    ``decoder`` is injectable only for CPU/synthetic unit fixtures.  The CLI
    always uses the accepted Stage40 decoder above; no module path supplied by
    a manifest is imported dynamically.
    """

    validate_evaluation_manifest(manifest)
    _require(
        manifest.get("status") == GENERIC_EVAL_STATUS,
        "evaluation manifest is not execution-ready",
    )
    output = Path(output_dir).expanduser()
    if output.exists():
        _require(
            output.is_dir() and not output.is_symlink(),
            "evaluation output root must be a regular directory",
        )
        _require(
            not any(output.iterdir()), "evaluation output root must be fresh and empty"
        )
    else:
        output.mkdir(parents=True, exist_ok=False)
    if manifest_path is not None:
        manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
        expected_root = _path_from(
            manifest["output"]["root"],
            base_dir=manifest_file.parent,
            field="output.root",
        )
        _require(
            output.resolve() == expected_root,
            "output root differs from manifest-bound root",
        )
    record_base = Path(records_base_dir or Path.cwd()).expanduser().resolve()
    if verify_artifacts and manifest_path is not None:
        verified = verify_evaluation_artifacts(manifest, manifest_path=manifest_path)
    else:
        verified = {"skipped": True}

    checkpoints = list(manifest["checkpoints"])
    windows = list(manifest["cohort"]["windows"])
    expected_keys = [
        (str(cp["label"]), str(window["id"]))
        for cp in checkpoints
        for window in windows
    ]
    observed_keys = [
        (str(item.get("checkpoint_label")), str(item.get("window_id")))
        for item in records
    ]
    _require(
        observed_keys == expected_keys,
        "raw prediction records are not in checkpoint/window manifest order",
    )
    _same_noise_guard(records, manifest=manifest)
    checkpoint_by_label = _checkpoint_map(manifest)
    window_by_id = _window_map(manifest)
    raw_manifest_path = output / str(manifest["output"]["raw_predictions_manifest"])
    _require(
        not Path(manifest["output"]["raw_predictions_manifest"]).is_absolute()
        and ".." not in Path(manifest["output"]["raw_predictions_manifest"]).parts,
        "output.raw_predictions_manifest must remain inside the output root",
    )
    raw_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    per_window_path = output / "per_window.jsonl"
    rows: list[dict[str, Any]] = []
    try:
        with (
            raw_manifest_path.open("w", encoding="utf-8") as raw_handle,
            per_window_path.open("w", encoding="utf-8") as per_window_handle,
        ):
            for record in records:
                checkpoint = checkpoint_by_label[str(record["checkpoint_label"])]
                window = window_by_id[str(record["window_id"])]
                _validate_record_contract(
                    record,
                    checkpoint=checkpoint,
                    window=window,
                    base_dir=record_base,
                )
                raw, source, source_sha256 = _raw_from_record(
                    record, base_dir=record_base
                )
                for key, artifact in record["observations"].items():
                    verify_artifact_identity(
                        artifact,
                        base_dir=record_base,
                        field=f"record.observations.{key}",
                    )
                relative_raw = (
                    Path("raw")
                    / _safe_segment(str(record["checkpoint_label"]))
                    / (_safe_segment(str(record["window_id"])) + source.suffix)
                )
                raw_destination = output / relative_raw
                persisted = _persist_raw(
                    source,
                    raw_destination,
                    raw_manifest_handle=raw_handle,
                    record=record,
                    raw=raw,
                    source_sha256=source_sha256,
                )
                state = _array_from_record(
                    record, "state_v1", shape=STATE_V1_SHAPE, base_dir=record_base
                )
                expert = _array_from_record(
                    record,
                    "expert_physical_action",
                    shape=PHYSICAL_ACTION_SHAPE,
                    base_dir=record_base,
                )
                anchor = np.asarray(window["anchor_world_m"], dtype=np.float64)
                target = np.asarray(window["target_world_m"], dtype=np.float64)
                decoded = decoder(
                    raw,
                    state,
                    stats_path=_path_from(
                        checkpoint["statistics"]["path"],
                        base_dir=(
                            Path(manifest_path).resolve().parent
                            if manifest_path
                            else Path.cwd()
                        ),
                        field=f"checkpoint[{checkpoint['label']}].statistics.path",
                    ),
                    source_timestamps=np.asarray(
                        window["source_timestamps"], dtype=np.float64
                    ),
                    observation_timestamp=float(window["observation_timestamp"]),
                    chunk_id=str(window["id"]),
                )
                physical = np.asarray(decoded.physical_action)
                _require(
                    tuple(physical.shape) == PHYSICAL_ACTION_SHAPE,
                    "decoded physical action ABI drift",
                )
                _require(
                    bool(np.isfinite(physical).all()),
                    "decoded physical action contains non-finite values",
                )
                metrics = comparison_metrics(
                    physical, expert, anchor, target, prefix=str(window["id"])
                )
                relative_decoded = (
                    Path("decoded")
                    / _safe_segment(str(checkpoint["label"]))
                    / f"{_safe_segment(str(window['id']))}.npy"
                )
                decoded_persisted = _persist_array(
                    physical, output / relative_decoded
                )
                row = {
                    "schema": GENERIC_EVALUATION_ROW_SCHEMA,
                    "checkpoint": {
                        "label": str(checkpoint["label"]),
                        "sha256": str(checkpoint["identity"]["sha256"]),
                        "runtime_mapping": dict(checkpoint["runtime_mapping"]),
                        "statistics": dict(checkpoint["statistics"]),
                    },
                    "window_id": str(window["id"]),
                    "target_id": str(window["target_id"]),
                    "noise_sha256": str(record["noise_sha256"]),
                    "model_input_keys": list(EXPECTED_MODEL_INPUT_KEYS),
                    "target_xyz_in_model_input": False,
                    "raw_prediction": persisted,
                    "decoded_action": {
                        **decoded_persisted,
                        "decoder_details": _jsonable(dict(decoded.decoder_details)),
                    },
                    "metrics": _jsonable(metrics),
                    "runtime_identity": manifest["runtime"]["identity"],
                    "runtime_backend": manifest["runtime"]["backend"],
                    "timestamp_profile": manifest["timestamp_profile"],
                    "observation_identities": {
                        key: dict(value) for key, value in record["observations"].items()
                    },
                    "raw_persisted_before_decode": True,
                    "training": False,
                }
                _append_jsonl(per_window_handle, row)
                rows.append(row)
    except Exception as exc:
        failure = {
            "schema": "GO2ARX5_REUSABLE_EVALUATION_FAILURE_V1",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "rows_completed": len(rows),
            "raw_manifest": str(raw_manifest_path),
            "per_window": str(per_window_path),
            "decoded_root": str(output / "decoded"),
        }
        _durable_json(
            output / "failure_seal.json",
            {
                "schema": "GO2ARX5_REUSABLE_EVALUATION_FAILURE_SEAL_V1",
                "status": "FAILED",
                "rows_completed": len(rows),
                "raw_manifest": str(raw_manifest_path),
                "per_window": str(per_window_path),
            },
        )
        _durable_json(output / "failure.json", failure)
        raise

    _write_rows_csv(output / "per_window.csv", rows)
    _durable_json(
        output / "failure_seal.json",
        {
            "schema": "GO2ARX5_REUSABLE_EVALUATION_FAILURE_SEAL_V1",
            "status": "PASS_NO_FAILURE",
            "rows_completed": len(rows),
            "raw_manifest": str(raw_manifest_path),
            "per_window": str(per_window_path),
        },
    )
    aggregates = {
        str(label): _aggregate_checkpoint_rows(
            [row for row in rows if row["checkpoint"]["label"] == label]
        )
        for label in [str(item["label"]) for item in checkpoints]
    }
    _durable_json(output / "checkpoint_aggregates.json", aggregates)
    inventory_path, inventory_sha256 = _write_artifact_inventory(output)
    input_manifest_sha256 = None
    if manifest_path is not None:
        input_manifest_sha256 = sha256_file(Path(manifest_path).resolve(strict=True))
    run_manifest = {
        "schema": GENERIC_EVALUATION_OUTPUT_SCHEMA,
        "manifest_sha256": input_manifest_sha256,
        "experiment": dict(manifest["experiment"]),
        "runtime_identity": manifest["runtime"]["identity"],
        "runtime_backend": manifest["runtime"]["backend"],
        "runtime_late_bound": dict(manifest["runtime"]["late_bound"]),
        "timestamp_profile": manifest["timestamp_profile"],
        "sampler_profile": manifest["sampler"]["profile"],
        "checkpoint_order": [str(item["label"]) for item in checkpoints],
        "checkpoint_identities": [
            {
                "label": str(item["label"]),
                "identity": dict(item["identity"]),
                "runtime_mapping": dict(item["runtime_mapping"]),
                "statistics": dict(item["statistics"]),
            }
            for item in checkpoints
        ],
        "paired_comparisons": [
            dict(pair)
            for pair in manifest.get("analysis", {}).get(
                "paired_comparisons", []
            )
        ],
        "window_order": [str(item["id"]) for item in windows],
        "raw_prediction_count": len(rows),
        "sampler_calls": len(rows)
        * int(manifest["sampler"]["expected_calls_per_window"]),
        "raw_prediction_shape": list(RAW_ACTION_SHAPE),
        "decoded_action_shape": list(PHYSICAL_ACTION_SHAPE),
        "raw_manifest": raw_manifest_path.relative_to(output).as_posix(),
        "per_window": "per_window.jsonl",
        "per_window_csv": "per_window.csv",
        "decoded_root": "decoded",
        "checkpoint_aggregates": "checkpoint_aggregates.json",
        "failure_seal": "failure_seal.json",
        "artifact_inventory": {
            "path": inventory_path.relative_to(output).as_posix(),
            "sha256": inventory_sha256,
        },
        "backend_boundary": INFERENCE_BACKEND_V1,
        "backend_handoff": "external_raw_prediction_handoff_consumed",
        "same_noise_across_checkpoints": True,
        "model_construction": False,
        "model_forward": "external_stage_adapter_only",
        "training": False,
        "backward": False,
        "optimizer": False,
        "resume": False,
        "checkpoint_write": False,
        "Validation-E": "LOCKED",
        "Holdout-F": "LOCKED",
        "WBC": "LOCKED",
        "simulator": "LOCKED",
        "hardware": "LOCKED",
        "verified_artifacts": verified,
    }
    _durable_json(output / "run_manifest.json", run_manifest)
    return run_manifest


__all__ = [
    "GENERIC_EVALUATION_MANIFEST_SCHEMA",
    "GENERIC_EVALUATION_OUTPUT_SCHEMA",
    "GENERIC_EVALUATION_PLAN_SCHEMA",
    "GENERIC_EVALUATION_ROW_SCHEMA",
    "GENERIC_EVAL_STATUS",
    "GENERIC_PREDICTION_RECORD_SCHEMA",
    "ReusableEvaluationError",
    "build_evaluation_plan",
    "load_evaluation_manifest",
    "load_prediction_records",
    "run_evaluation",
    "validate_evaluation_manifest",
    "verify_evaluation_artifacts",
]
