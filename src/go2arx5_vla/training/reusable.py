"""Generic adaptation-training manifest and admission boundary.

The module owns explicit recipe, identity, fresh/resume, and checkpoint-save
semantics.  It is intentionally a planner/validator: a stage-specific wrapper
must supply the actual model backend after this contract is accepted.  No
Torch, GR00T, CUDA, PBS, WBC, simulator, or hardware code is imported here.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..experiments.reusable import (
    RUNTIME_LATE_BOUND_FIELDS,
    TEMPLATE_SENTINEL,
    TRAINING_BACKEND_V1,
    ReusableManifestError,
    load_json,
    reject_locked_data,
    require_mapping,
    require_nonempty_string,
    require_sha256,
    validate_late_bound_runtime,
    verify_artifact_identity,
    write_json,
)

GENERIC_TRAINING_MANIFEST_SCHEMA = "GO2ARX5_REUSABLE_ADAPTATION_TRAINING_MANIFEST_V1"
GENERIC_TRAINING_PLAN_SCHEMA = "GO2ARX5_REUSABLE_TRAINING_PLAN_V1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TRAINING_RECIPE_FIELDS = (
    "max_steps",
    "warmup_steps",
    "scheduler",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "seed",
    "batch_size",
    "gradient_accumulation_steps",
    "bf16",
    "fp16",
    "tune_projector",
    "tune_diffusion",
    "tune_visual",
    "tune_llm",
    "tune_vlln",
    "resume",
    "fresh_start",
    "save_only_model",
    "save_steps",
)


class ReusableTrainingError(ReusableManifestError):
    """Raised when an adaptation-training manifest is incomplete or unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReusableTrainingError(message)


def _digest(value: Any, *, field: str) -> str:
    result = require_sha256(value, field=field)
    _require(
        bool(_SHA256_RE.fullmatch(result)),
        f"{field} must be lowercase hexadecimal SHA256",
    )
    return result


def _artifact(value: Any, *, field: str) -> None:
    item = require_mapping(value, field=field)
    require_nonempty_string(item.get("path"), field=f"{field}.path")
    _digest(item.get("sha256"), field=f"{field}.sha256")
    _require(
        item.get("kind", "file") in {"file", "tree"},
        f"{field}.kind must be file or tree",
    )


def _bool(value: Any, *, field: str) -> bool:
    _require(type(value) is bool, f"{field} must be an explicit boolean")
    return bool(value)


def _integer(value: Any, *, field: str) -> int:
    _require(type(value) is int, f"{field} must be a JSON integer")
    return value


def validate_training_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate all generic training identity and recipe fields."""

    _require(
        manifest.get("schema") == GENERIC_TRAINING_MANIFEST_SCHEMA,
        "training manifest schema mismatch",
    )
    status = manifest.get("status")
    _require(
        status in {"TEMPLATE_NOT_EXECUTABLE", "READY_NOT_EXECUTED"},
        "training manifest status is invalid",
    )
    template = status == "TEMPLATE_NOT_EXECUTABLE"
    experiment = require_mapping(manifest.get("experiment"), field="experiment")
    require_nonempty_string(experiment.get("id"), field="experiment.id")
    require_nonempty_string(experiment.get("stage"), field="experiment.stage")
    head = require_nonempty_string(
        experiment.get("execution_head"), field="experiment.execution_head"
    )
    _require(
        bool(re.fullmatch(r"[0-9a-fA-F]{40}", head)),
        "experiment.execution_head must be a Git SHA",
    )
    if not template:
        _require(
            head.lower() != "0" * 40,
            "experiment.execution_head must identify a concrete execution revision",
        )
    _digest(experiment.get("protocol_sha256"), field="experiment.protocol_sha256")

    dataset = require_mapping(manifest.get("dataset"), field="dataset")
    require_nonempty_string(dataset.get("name"), field="dataset.name")
    require_nonempty_string(dataset.get("split"), field="dataset.split")
    _artifact(dataset.get("manifest"), field="dataset.manifest")
    _artifact(dataset.get("statistics"), field="dataset.statistics")
    provenance = require_mapping(dataset.get("provenance"), field="dataset.provenance")
    require_nonempty_string(
        provenance.get("identity"), field="dataset.provenance.identity"
    )
    _digest(provenance.get("sha256"), field="dataset.provenance.sha256")

    base = require_mapping(manifest.get("base_checkpoint"), field="base_checkpoint")
    require_nonempty_string(base.get("label"), field="base_checkpoint.label")
    _artifact(base.get("identity"), field="base_checkpoint.identity")

    recipe = require_mapping(manifest.get("recipe"), field="recipe")
    missing = [field for field in TRAINING_RECIPE_FIELDS if field not in recipe]
    _require(
        not missing, f"recipe fields must be explicit; missing: {', '.join(missing)}"
    )
    if template:
        _require(
            all(recipe[field] == TEMPLATE_SENTINEL for field in TRAINING_RECIPE_FIELDS),
            "non-executable training templates must use __REQUIRED__ recipe sentinels",
        )
    else:
        _validate_training_recipe(recipe)

    runtime = require_mapping(manifest.get("runtime"), field="runtime")
    require_nonempty_string(runtime.get("identity"), field="runtime.identity")
    backend = require_nonempty_string(runtime.get("backend"), field="runtime.backend")
    if not (template and backend == TEMPLATE_SENTINEL):
        _require(
            backend == TRAINING_BACKEND_V1,
            f"runtime.backend must be {TRAINING_BACKEND_V1}",
        )
    _bool(runtime.get("offline"), field="runtime.offline")
    _require(
        runtime["offline"] is True, "generic training runtime must be offline/local"
    )
    _bool(runtime.get("preflight_required"), field="runtime.preflight_required")
    _require(
        runtime["preflight_required"] is True, "runtime preflight must be required"
    )
    try:
        validate_late_bound_runtime(
            runtime.get("late_bound"),
            field="runtime.late_bound",
            allow_sentinel=template,
        )
    except ReusableManifestError as exc:
        raise ReusableTrainingError(str(exc)) from exc

    output = require_mapping(manifest.get("output"), field="output")
    require_nonempty_string(output.get("root"), field="output.root")
    _bool(output.get("fresh"), field="output.fresh")
    _require(output["fresh"] is True, "generic training output must be fresh-bound")
    _require(
        output.get("failure_seal") is True, "output failure sealing must be enabled"
    )

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
    reject_locked_data(manifest)


def _validate_training_recipe(recipe: Mapping[str, Any]) -> None:
    """Validate a concrete recipe after template sentinels are resolved."""

    max_steps = _integer(recipe["max_steps"], field="recipe.max_steps")
    warmup_steps = _integer(recipe["warmup_steps"], field="recipe.warmup_steps")
    _require(max_steps > 0, "recipe.max_steps must be positive")
    _require(0 <= warmup_steps <= max_steps, "recipe.warmup_steps is outside max_steps")
    require_nonempty_string(recipe["scheduler"], field="recipe.scheduler")
    require_nonempty_string(recipe["optimizer"], field="recipe.optimizer")
    for field in ("learning_rate", "weight_decay"):
        value = float(recipe[field])
        _require(
            math.isfinite(value) and value >= 0.0,
            f"recipe.{field} must be finite and non-negative",
        )
    _require(
        float(recipe["learning_rate"]) > 0.0, "recipe.learning_rate must be positive"
    )
    _integer(recipe["seed"], field="recipe.seed")
    for field in ("batch_size", "gradient_accumulation_steps"):
        value = _integer(recipe[field], field=f"recipe.{field}")
        _require(value > 0, f"recipe.{field} must be positive")
    _bool(recipe["bf16"], field="recipe.bf16")
    _bool(recipe["fp16"], field="recipe.fp16")
    _require(
        not (recipe["bf16"] and recipe["fp16"]), "bf16 and fp16 cannot both be enabled"
    )
    for field in (
        "tune_projector",
        "tune_diffusion",
        "tune_visual",
        "tune_llm",
        "tune_vlln",
        "resume",
        "fresh_start",
        "save_only_model",
    ):
        _bool(recipe[field], field=f"recipe.{field}")
    _require(
        recipe["resume"] is not recipe["fresh_start"],
        "resume and fresh_start must select exactly one mode",
    )
    save_steps = recipe["save_steps"]
    _require(
        isinstance(save_steps, list) and bool(save_steps),
        "recipe.save_steps must be an explicit list",
    )
    normalized_steps = [
        _integer(step, field=f"recipe.save_steps[{index}]")
        for index, step in enumerate(save_steps)
    ]
    _require(
        normalized_steps == sorted(set(normalized_steps)),
        "recipe.save_steps must be sorted and unique",
    )
    _require(
        all(0 < step <= max_steps for step in normalized_steps),
        "recipe.save_steps contains an invalid step",
    )
    _require(
        max_steps in normalized_steps,
        "recipe.save_steps must include max_steps for full-state sealing",
    )
    _require(
        recipe["save_only_model"] is False,
        "full trainer-state checkpoints require save_only_model=false",
    )


def _manifest_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ReusableTrainingError(
            f"training manifest may not be a symlink: {candidate}"
        )
    return candidate.resolve(strict=True)


def load_training_manifest(path: str | Path) -> dict[str, Any]:
    candidate = _manifest_path(path)
    value = load_json(candidate)
    validate_training_manifest(value)
    return value


def verify_training_artifacts(
    manifest: Mapping[str, Any], *, manifest_path: str | Path
) -> dict[str, Any]:
    base_dir = _manifest_path(manifest_path).parent
    return {
        "base_checkpoint": verify_artifact_identity(
            manifest["base_checkpoint"]["identity"],
            base_dir=base_dir,
            field="base_checkpoint.identity",
        ),
        "dataset_manifest": verify_artifact_identity(
            manifest["dataset"]["manifest"],
            base_dir=base_dir,
            field="dataset.manifest",
        ),
        "dataset_statistics": verify_artifact_identity(
            manifest["dataset"]["statistics"],
            base_dir=base_dir,
            field="dataset.statistics",
        ),
    }


def build_training_plan(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a backend handoff plan without constructing or training a model."""

    validate_training_manifest(manifest)
    _require(
        manifest.get("status") == "READY_NOT_EXECUTED",
        "training manifest template is not executable",
    )
    recipe = dict(manifest["recipe"])
    return {
        "schema": GENERIC_TRAINING_PLAN_SCHEMA,
        "experiment": dict(manifest["experiment"]),
        "base_checkpoint": dict(manifest["base_checkpoint"]),
        "dataset": dict(manifest["dataset"]),
        "recipe": recipe,
        "runtime": dict(manifest["runtime"]),
        "output": dict(manifest["output"]),
        "backend_boundary": TRAINING_BACKEND_V1,
        "late_bound_runtime_fields": list(RUNTIME_LATE_BOUND_FIELDS),
        "backend_handoff": "explicit_stage_training_backend_required",
        "backend_handoff_required": True,
        "model_construction": False,
        "model_forward": False,
        "training_started": False,
        "optimizer_steps": 0,
        "resume": bool(recipe["resume"]),
        "fresh_start": bool(recipe["fresh_start"]),
        "checkpoint_write": False,
        "failure_seal_required": True,
        "Validation-E": "LOCKED",
        "Holdout-F": "LOCKED",
        "WBC": "LOCKED",
        "simulator": "LOCKED",
        "hardware": "LOCKED",
    }


def validate_training_execution_admission(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Reject a training handoff unless all input artifacts were verified."""

    _require(
        plan.get("schema") == GENERIC_TRAINING_PLAN_SCHEMA,
        "training execution plan schema mismatch",
    )
    _require(
        plan.get("backend_handoff_required") is True,
        "training execution backend handoff is not required",
    )
    verified = plan.get("verified_artifacts")
    _require(
        isinstance(verified, Mapping),
        "training execution requires verified_artifacts",
    )
    _require(
        verified.get("skipped") is not True,
        "training execution rejects skipped artifact verification",
    )
    for key in ("base_checkpoint", "dataset_manifest", "dataset_statistics"):
        item = verified.get(key)
        _require(
            isinstance(item, Mapping) and item.get("verified") is True,
            f"training execution artifact is not verified: {key}",
        )
    return dict(verified)


def write_training_plan(
    manifest: Mapping[str, Any],
    *,
    output: str | Path,
    manifest_path: str | Path | None = None,
    verify_artifacts: bool = False,
) -> dict[str, Any]:
    plan = build_training_plan(manifest)
    if verify_artifacts:
        if manifest_path is None:
            raise ReusableTrainingError(
                "manifest_path is required when verifying artifacts"
            )
        plan["verified_artifacts"] = verify_training_artifacts(
            manifest, manifest_path=manifest_path
        )
    else:
        plan["verified_artifacts"] = {"skipped": True}
    write_json(output, plan)
    return plan


__all__ = [
    "GENERIC_TRAINING_MANIFEST_SCHEMA",
    "GENERIC_TRAINING_PLAN_SCHEMA",
    "TRAINING_RECIPE_FIELDS",
    "ReusableTrainingError",
    "build_training_plan",
    "load_training_manifest",
    "validate_training_execution_admission",
    "validate_training_manifest",
    "verify_training_artifacts",
    "write_training_plan",
]
