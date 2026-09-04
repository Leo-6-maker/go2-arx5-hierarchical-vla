#!/usr/bin/env python3
"""Audit the Stage23 O1@200 optimization semantics before H2 execution.

This is a Corelab-only, standard-library audit.  It reads the immutable
Stage23 Phase-A metadata relay and pinned source objects, but never imports
torch/transformers/safetensors, constructs a model, reads tensor payloads,
uses a GPU, runs a forward pass, trains, submits PBS, or reads validation or
holdout data.

The audit answers whether O1@200 can be resumed exactly and derives the
logged cosine LR schedule for a fresh 200-step and hypothetical 2000-step
run.  The LR indexing is deliberately calibrated against the sealed
``trainer_state_200.json`` rather than silently assuming that a displayed
global step is the scheduler's internal step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping


STAGE27_SCHEMA = "STAGE27_H2_OPTIMIZATION_SEMANTICS_AUDIT_V1"
STAGE27_BASE_SHA = "735afb7f17f0184ca9eff3e372dfd59c8e008e95"

STAGE23_FROZEN_GIT_SHA = "deb920a070141e6a32a88cd0da6719828647e1e6"
EXPECTED_GROOT_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
EXPECTED_STAGE20_LAUNCHER_SHA256 = (
    "f42a1990847cf8f5326c66db37043ffed6e43458dd907fdbd08da5dd905b00a2"
)

EXPECTED_O1_STEPS = 200
EXPECTED_H2_STEPS = 2000
EXPECTED_LEARNING_RATE = 1e-4
EXPECTED_WARMUP_RATIO = 0.05
EXPECTED_WARMUP_STEPS = 0
EXPECTED_LR_SCHEDULER = "cosine"
EXPECTED_OPTIMIZER = "adamw_torch"
EXPECTED_TRAINING_SEED = 2222
EXPECTED_BATCH_SIZE = 1
EXPECTED_GRADIENT_ACCUMULATION_STEPS = 1

EXPECTED_O1_INDEX_SHA256 = "ff5a9ac57951658b775ef7e958a3469900f767cec60cbf487b3c8d0a3d787bc7"
EXPECTED_O1_SEAL_SHA256 = "7aa07a84d62a5052bb9522ba0a04f8d1b711dac0f6b448db3e696135a663f806"
EXPECTED_O1_MANIFEST_SHA256 = "4dfb69bbd0bb36562c17046735001454c8dc5cfe5e9e0195ae3ca032259a3148"
EXPECTED_O1_TRAINER_STATE_SHA256 = "2aa506bf605bce28809739970b6d3bc86059dba5490101c7da4fdba94512e4d9"
EXPECTED_O1_TRAIN_LOG_SHA256 = "35cc1f07fff30d413d52f2c11b8b72db7429a8004ae9244a11a798fbfd8f9c07"
EXPECTED_COMMAND_CONFIG_SHA256 = "8757a0353501dcc88fe062ab6ca58ac2c0ba3643b55c9d12f4e55fb93d923eb4"
EXPECTED_PROVENANCE_SHA256 = "b9182fbb2026590253ca387ff21b0bb2e85c337c12d4ecca8106596b27c3e34c"
EXPECTED_SCOPE_SHA256 = "caa93acf09620c89b2bf66c5481cf258c7e1cd323b91046ad970a96a8f5bf00b"

DEFAULT_RELAY_ROOT = Path(
    "/var/cache/users/liuyu-nfs/go2-arx5-stage23-o1-phase-a-20260903/"
    "extracted/phase_a_seal_20260903"
)
DEFAULT_GROOT_REPO = Path("/mnt/LocalHDD1/HomeNFS/liuyu-nfs/Projects/isaac-gr00t")
DEFAULT_PROJECT_REPO = Path(
    "/mnt/LocalHDD1/HomeNFS/liuyu-nfs/Projects/go2-arx5-hierarchical-vla"
)
DEFAULT_REPORT_PATH = DEFAULT_PROJECT_REPO / "reports/STAGE27_H2_OPTIMIZATION_SEMANTICS_AUDIT.json"

STATE_ARTIFACT_RE = re.compile(
    r"(?:^|/)(?:optimizer(?:\.[^/]*)?|scheduler(?:\.[^/]*)?|"
    r"rng_state(?:[_-][^/]*)?\.[^/]+|scaler(?:\.[^/]*)?)$",
    re.IGNORECASE,
)

EXPECTED_CANDIDATE_200_SEAL_PATHS = (
    "config.json",
    "embodiment_id.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "processor_config.json",
    "statistics.json",
    "trainer_state.json",
    "training_args.bin",
    "wandb_config.json",
)

EXPECTED_TRAINER_STATE_TOP_LEVEL_KEYS = (
    "best_global_step",
    "best_metric",
    "best_model_checkpoint",
    "epoch",
    "eval_steps",
    "global_step",
    "is_hyper_param_search",
    "is_local_process_zero",
    "is_world_process_zero",
    "log_history",
    "logging_steps",
    "max_steps",
    "num_input_tokens_seen",
    "num_train_epochs",
    "save_steps",
    "stateful_callbacks",
    "total_flos",
    "train_batch_size",
    "trial_name",
    "trial_params",
)


class AuditError(RuntimeError):
    """Raised when a Stage27 identity or source assertion fails closed."""


def _sha256_file(path: Path) -> str:
    """Hash a regular, non-symlink file without following an unsafe path."""

    try:
        file_stat = path.lstat()
    except FileNotFoundError as exc:
        raise AuditError(f"missing required file: {path}") from exc
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise AuditError(f"required path is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    _sha256_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"JSON evidence must be an object: {path}")
    return value


def _require_sha(path: Path, expected: str, label: str) -> str:
    observed = _sha256_file(path)
    if observed != expected:
        raise AuditError(f"{label} SHA256 mismatch: {observed} != {expected}")
    return observed


def _git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AuditError(f"cannot resolve git HEAD for {repo}") from exc


def _git_show_bytes(repo: Path, revision: str, relative_path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{revision}:{relative_path}"],
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AuditError(
            f"cannot read pinned source object {revision}:{relative_path}"
        ) from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_substrings(text: str, anchors: Iterable[str], label: str) -> dict[str, bool]:
    checks = {anchor: anchor in text for anchor in anchors}
    missing = [anchor for anchor, present in checks.items() if not present]
    if missing:
        raise AuditError(f"{label} missing source anchors: {missing}")
    return checks


def _parse_tsv_seal(path: Path) -> list[dict[str, Any]]:
    _sha256_file(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 5:
            raise AuditError(f"malformed seal row {path}:{line_number}")
        candidate, relative_path, kind, byte_text, sha256 = fields
        try:
            byte_count = int(byte_text)
        except ValueError as exc:
            raise AuditError(f"invalid byte count in seal row {path}:{line_number}") from exc
        if byte_count < 0 or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise AuditError(f"invalid seal values in row {path}:{line_number}")
        rows.append(
            {
                "candidate": candidate,
                "relative_path": relative_path,
                "kind": kind,
                "bytes": byte_count,
                "sha256": sha256,
            }
        )
    return rows


def _candidate_seal_rows(rows: Iterable[Mapping[str, Any]], candidate: str) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows if row.get("candidate") == candidate]
    if not selected:
        raise AuditError(f"seal has no rows for {candidate}")
    paths = [str(row["relative_path"]) for row in selected]
    if len(paths) != len(set(paths)):
        raise AuditError(f"seal has duplicate paths for {candidate}")
    if tuple(sorted(paths)) != tuple(sorted(EXPECTED_CANDIDATE_200_SEAL_PATHS)):
        raise AuditError(
            f"{candidate} seal path set changed: {sorted(paths)} != "
            f"{sorted(EXPECTED_CANDIDATE_200_SEAL_PATHS)}"
        )
    if any(row.get("kind") != "regular file" for row in selected):
        raise AuditError(f"{candidate} seal contains a non-regular entry")
    return sorted(selected, key=lambda row: str(row["relative_path"]))


def _state_artifact_paths(paths: Iterable[str]) -> list[str]:
    return sorted(path for path in paths if STATE_ARTIFACT_RE.search(path))


def _recursive_state_like_keys(value: Any, prefix: str = "") -> list[str]:
    """Find serialized optimizer/scheduler/RNG/scaler keys, excluding LR logs."""

    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            location = f"{prefix}.{key_text}" if prefix else key_text
            lowered = key_text.lower()
            if (
                lowered in {"optimizer", "optimizer_state", "scheduler", "scheduler_state", "rng_state", "scaler"}
                or "optimizer.pt" in lowered
                or "scheduler.pt" in lowered
                or "rng_state" in lowered
                or lowered in {"amp_scaler", "grad_scaler"}
            ):
                found.append(location)
            found.extend(_recursive_state_like_keys(child, location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_recursive_state_like_keys(child, f"{prefix}[{index}]"))
    return found


def resolve_warmup_steps(
    total_steps: int,
    warmup_ratio: float = EXPECTED_WARMUP_RATIO,
    warmup_steps: int = EXPECTED_WARMUP_STEPS,
) -> int:
    """Mirror TrainingArguments' explicit-step-over-ratio precedence."""

    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if warmup_steps > 0:
        return int(warmup_steps)
    if warmup_ratio < 0:
        raise ValueError("warmup_ratio must be non-negative")
    return int(math.ceil(total_steps * warmup_ratio))


def cosine_schedule_factor(
    scheduler_step: int,
    total_steps: int,
    warmup_steps: int,
    num_cycles: float = 0.5,
) -> float:
    """Mirror HF get_cosine_schedule_with_warmup's lambda factor."""

    if total_steps < 1 or warmup_steps < 0:
        raise ValueError("invalid schedule dimensions")
    if scheduler_step < warmup_steps:
        return float(scheduler_step) / float(max(1, warmup_steps))
    progress = float(scheduler_step - warmup_steps) / float(
        max(1, total_steps - warmup_steps)
    )
    return max(0.0, float(0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))))


def logged_cosine_learning_rate(
    global_step: int,
    total_steps: int,
    *,
    learning_rate: float = EXPECTED_LEARNING_RATE,
    warmup_ratio: float = EXPECTED_WARMUP_RATIO,
    warmup_steps: int = EXPECTED_WARMUP_STEPS,
) -> float:
    """Return the LR displayed at a pinned Stage23 Trainer log step.

    The sealed log shows step 1 at factor 0, step 2 at factor 0.1, and step
    11 at factor 1.0 for a ten-step warmup.  Thus the displayed global step
    ``s`` maps to the scheduler lambda at ``s - 1`` for this pinned path.
    """

    if global_step < 1:
        raise ValueError("global_step must be positive")
    resolved_warmup = resolve_warmup_steps(total_steps, warmup_ratio, warmup_steps)
    scheduler_step = global_step - 1
    return float(
        learning_rate
        * cosine_schedule_factor(scheduler_step, total_steps, resolved_warmup)
    )


def schedule_rows(
    total_steps: int,
    selected_steps: Iterable[int],
    *,
    warmup_ratio: float = EXPECTED_WARMUP_RATIO,
    warmup_steps: int = EXPECTED_WARMUP_STEPS,
) -> list[dict[str, Any]]:
    warmup = resolve_warmup_steps(total_steps, warmup_ratio, warmup_steps)
    rows = []
    for global_step in selected_steps:
        rows.append(
            {
                "global_step": int(global_step),
                "scheduler_step": int(global_step) - 1,
                "warmup_steps": warmup,
                "learning_rate": logged_cosine_learning_rate(
                    global_step,
                    total_steps,
                    warmup_ratio=warmup_ratio,
                    warmup_steps=warmup_steps,
                ),
            }
        )
    return rows


def compare_logged_schedule(
    log_history: Iterable[Mapping[str, Any]], total_steps: int
) -> dict[str, Any]:
    rows = []
    max_abs_error = 0.0
    matched = 0
    for item in log_history:
        if "step" not in item or "learning_rate" not in item:
            continue
        step = int(item["step"])
        observed = float(item["learning_rate"])
        expected = logged_cosine_learning_rate(step, total_steps)
        error = abs(observed - expected)
        max_abs_error = max(max_abs_error, error)
        if math.isclose(observed, expected, rel_tol=1e-15, abs_tol=1e-15):
            matched += 1
        rows.append(
            {
                "global_step": step,
                "observed_learning_rate": observed,
                "expected_learning_rate": expected,
                "absolute_error": error,
            }
        )
    return {
        "observed_rows": len(rows),
        "matched_rows": matched,
        "all_rows_match": bool(rows) and matched == len(rows),
        "max_absolute_error": max_abs_error,
        "first_rows": rows[:5],
        "last_rows": rows[-5:],
    }


def _source_audit(groot_repo: Path, project_repo: Path) -> dict[str, Any]:
    groot_head = _git_head(groot_repo)
    if groot_head != EXPECTED_GROOT_SHA:
        raise AuditError(f"pinned GR00T HEAD mismatch: {groot_head} != {EXPECTED_GROOT_SHA}")

    groot_sources = {
        "gr00t/experiment/experiment.py": (
            "check_resume_compatibility(config.training)",
            "max_steps=config.training.max_steps",
            "lr_scheduler_type=config.training.lr_scheduler_type",
            "warmup_ratio=config.training.warmup_ratio",
            "save_only_model=config.training.save_only_model",
            "trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)",
            "trainer.save_model()",
            "set_seed(config.data.seed)",
        ),
        "gr00t/configs/training/training_config.py": (
            "lr_scheduler_type: str = \"cosine\"",
            "warmup_ratio: float = 0.05",
            "warmup_steps: int = 0",
            "save_only_model: bool = False",
            "if training.save_only_model and training.resume_from_checkpoint:",
            "the checkpoint lacks optimizer/scheduler/RNG state",
        ),
        "gr00t/experiment/trainer.py": (
            "self.args.ignore_data_skip = True",
            "TrainerState.load_from_json",
            "self.train_dataset.reset_seed(new_seed)",
            "Please note that this will make the experiment non-reproducible.",
            "super().train(resume_from_checkpoint=latest_checkpoint, **kwargs)",
        ),
    }
    source_report: dict[str, Any] = {
        "groot_head": groot_head,
        "expected_groot_head": EXPECTED_GROOT_SHA,
        "pinned_revision": EXPECTED_GROOT_SHA,
        "files": {},
    }
    for relative_path, anchors in groot_sources.items():
        payload = _git_show_bytes(groot_repo, EXPECTED_GROOT_SHA, relative_path)
        text = payload.decode("utf-8")
        source_report["files"][relative_path] = {
            "sha256": _sha256_bytes(payload),
            "anchors": _require_substrings(text, anchors, relative_path),
        }

    launcher_payload = _git_show_bytes(
        project_repo, STAGE27_BASE_SHA, "scripts/stage20s_train_gr00t.py"
    )
    launcher_sha = _sha256_bytes(launcher_payload)
    if launcher_sha != EXPECTED_STAGE20_LAUNCHER_SHA256:
        raise AuditError(
            f"Stage23 launcher SHA mismatch: {launcher_sha} != "
            f"{EXPECTED_STAGE20_LAUNCHER_SHA256}"
        )
    launcher_text = launcher_payload.decode("utf-8")
    launcher_anchors = (
        "config.training.optim = \"adamw_torch\"",
        "config.training.learning_rate = float(args.learning_rate)",
        "config.training.max_steps = int(args.max_steps)",
        "config.training.warmup_ratio = 0.05",
        "config.training.bf16 = True",
        "config.training.fp16 = False",
        "config.training.save_only_model = True",
        "config.training.resume_from_checkpoint = False",
        "config.data.seed = int(args.seed)",
    )
    source_report["stage20_launcher"] = {
        "revision": STAGE27_BASE_SHA,
        "sha256": launcher_sha,
        "expected_sha256": EXPECTED_STAGE20_LAUNCHER_SHA256,
        "anchors": _require_substrings(launcher_text, launcher_anchors, "stage20 launcher"),
    }
    return source_report


def checkpoint_state_summary(
    manifest: Mapping[str, Any], trainer_state: Mapping[str, Any]
) -> dict[str, Any]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise AuditError("run manifest artifacts must be an object")
    all_paths = sorted(str(path) for path in artifacts)
    checkpoint_prefix = f"checkpoint-{EXPECTED_O1_STEPS}/"
    checkpoint_paths = sorted(
        path[len(checkpoint_prefix) :]
        for path in all_paths
        if path.startswith(checkpoint_prefix)
    )
    state_files = _state_artifact_paths(checkpoint_paths)
    top_level = tuple(sorted(str(key) for key in trainer_state))
    embedded_state_keys = _recursive_state_like_keys(trainer_state)
    if tuple(sorted(EXPECTED_TRAINER_STATE_TOP_LEVEL_KEYS)) != top_level:
        raise AuditError(
            "trainer_state top-level schema changed: "
            f"{list(top_level)} != {list(sorted(EXPECTED_TRAINER_STATE_TOP_LEVEL_KEYS))}"
        )
    return {
        "checkpoint_artifact_paths": checkpoint_paths,
        "optimizer_state_files": [path for path in state_files if "optimizer" in path.lower()],
        "scheduler_state_files": [path for path in state_files if "scheduler" in path.lower()],
        "rng_state_files": [path for path in state_files if "rng_state" in path.lower()],
        "scaler_state_files": [path for path in state_files if "scaler" in path.lower()],
        "all_state_artifact_files": state_files,
        "trainer_state_top_level_keys": list(top_level),
        "trainer_state_embedded_state_like_keys": sorted(set(embedded_state_keys)),
        "global_step": trainer_state.get("global_step"),
        "max_steps": trainer_state.get("max_steps"),
        "train_batch_size": trainer_state.get("train_batch_size"),
        "stateful_callbacks": trainer_state.get("stateful_callbacks"),
        "scaler_applicability": "NOT_APPLICABLE_BF16_TRUE_FP16_FALSE",
        "exact_resume_possible": False,
        "exact_resume_reason": (
            "save_only_model=True omitted optimizer/scheduler/RNG state; "
            "trainer_state.json preserves metadata/global_step only; the "
            "pinned entrypoint rejects resume_from_checkpoint=True under this recipe"
        ),
    }


def optimization_schedule_summary(trainer_state: Mapping[str, Any]) -> dict[str, Any]:
    history = trainer_state.get("log_history")
    if not isinstance(history, list):
        raise AuditError("trainer_state.log_history must be a list")
    observed = compare_logged_schedule(history, EXPECTED_O1_STEPS)
    if not observed["all_rows_match"] or observed["observed_rows"] != EXPECTED_O1_STEPS:
        raise AuditError(f"sealed O1 LR history does not match pinned schedule: {observed}")

    old_rows = schedule_rows(
        EXPECTED_O1_STEPS, (1, 2, 10, 11, 12, 100, 200)
    )
    new_rows = schedule_rows(
        EXPECTED_H2_STEPS, (1, 2, 10, 11, 100, 101, 200, 1000, 1900, 2000)
    )
    fixed_warmup_rows = schedule_rows(
        EXPECTED_H2_STEPS,
        (1, 2, 10, 11, 12, 100, 200, 1000, 1900, 2000),
        warmup_steps=10,
    )
    comparison_rows = []
    first_difference: int | None = None
    same_count = 0
    for step in range(1, EXPECTED_O1_STEPS + 1):
        old_lr = logged_cosine_learning_rate(step, EXPECTED_O1_STEPS)
        new_lr = logged_cosine_learning_rate(step, EXPECTED_H2_STEPS)
        equal = math.isclose(old_lr, new_lr, rel_tol=1e-15, abs_tol=1e-15)
        if equal:
            same_count += 1
        elif first_difference is None:
            first_difference = step
        if step in {1, 2, 10, 11, 12, 100, 200}:
            comparison_rows.append(
                {
                    "global_step": step,
                    "o1_200_learning_rate": old_lr,
                    "fresh_2000_learning_rate": new_lr,
                    "absolute_difference": abs(old_lr - new_lr),
                }
            )

    return {
        "learning_rate": EXPECTED_LEARNING_RATE,
        "scheduler": EXPECTED_LR_SCHEDULER,
        "warmup_ratio": EXPECTED_WARMUP_RATIO,
        "warmup_steps_configured": EXPECTED_WARMUP_STEPS,
        "o1_200": {
            "total_steps": EXPECTED_O1_STEPS,
            "warmup_steps": resolve_warmup_steps(EXPECTED_O1_STEPS),
            "rows": old_rows,
            "sealed_log_reconciliation": observed,
        },
        "fresh_2000_hypothetical": {
            "total_steps": EXPECTED_H2_STEPS,
            "warmup_steps": resolve_warmup_steps(EXPECTED_H2_STEPS),
            "rows": new_rows,
        },
        "fresh_2000_fixed_warmup10_hypothetical": {
            "total_steps": EXPECTED_H2_STEPS,
            "warmup_steps": 10,
            "rows": fixed_warmup_rows,
            "cosine_denominator_steps": EXPECTED_H2_STEPS - 10,
        },
        "first_200_schedule_comparison": {
            "same_logged_steps": same_count,
            "different_logged_steps": EXPECTED_O1_STEPS - same_count,
            "first_difference_global_step": first_difference,
            "rows": comparison_rows,
            "fresh_base_2000_is_not_o1_prefix_equivalent": first_difference is not None,
        },
    }


def protocol_plan() -> dict[str, Any]:
    """Return the non-executing Stage27 contract and PI decision aid."""

    return {
        "schema": STAGE27_SCHEMA,
        "stage": "Stage27 H2 optimization-semantics audit",
        "stage27_base_sha": STAGE27_BASE_SHA,
        "purpose": (
            "determine exact-resume availability and expose LR-schedule "
            "confounding before any longer training"
        ),
        "frozen_scientific_inputs": {
            "dataset": "TINY1_H2STATS",
            "training_seed": EXPECTED_TRAINING_SEED,
            "action_horizon": 40,
            "action_abi": "10D physical / 132D universal dense relative EEF",
            "batch_size": EXPECTED_BATCH_SIZE,
            "gradient_accumulation_steps": EXPECTED_GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": EXPECTED_LEARNING_RATE,
            "optimizer": EXPECTED_OPTIMIZER,
            "scheduler": EXPECTED_LR_SCHEDULER,
            "warmup_ratio": EXPECTED_WARMUP_RATIO,
        },
        "audit_scope": {
            "checkpoint_metadata": True,
            "pinned_training_source": True,
            "tensor_payload_reads": False,
            "model_construction": False,
            "forward": False,
            "training": False,
            "optimizer_step": False,
            "gpu": False,
            "pbs": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "c2": False,
            "wbc": False,
            "hardware": False,
        },
        "h2_protocol_decision_aid": {
            "recommended_stock_path_candidate": "H2_FRESH_BASE_2000_FIXED_WARMUP10",
            "recipe": {
                "initialization": "fresh exact BASE",
                "max_steps": EXPECTED_H2_STEPS,
                "warmup_steps": 10,
                "warmup_ratio": "not used when explicit warmup_steps=10 is set",
                "scheduler": EXPECTED_LR_SCHEDULER,
                "learning_rate": EXPECTED_LEARNING_RATE,
                "optimizer": EXPECTED_OPTIMIZER,
            },
            "why": (
                "avoids an unavailable exact-resume claim and keeps the original "
                "ten optimizer-update warmup steps; it is one coherent stock-path "
                "run rather than a hidden optimizer/scheduler reset at step 200"
            ),
            "unavoidable_coupling": (
                "cosine denominator still changes from 190 to 1990 scheduler steps, "
                "so this is not a pure extra-exposure intervention"
            ),
            "claim_boundary": (
                "tests a fresh 2000-step adaptation recipe against O1@200; it cannot "
                "attribute any difference to extra steps alone"
            ),
        },
        "alternatives_not_selected": [
            {
                "protocol": "exact O1@200 resume",
                "status": "UNAVAILABLE",
                "reason": "optimizer/scheduler/RNG state was not serialized",
            },
            {
                "protocol": "O1 weights plus 1800 fresh updates",
                "status": "SECONDARY",
                "reason": (
                    "preserves the O1 starting weights but resets optimizer, scheduler, "
                    "RNG, and custom sampler state at the boundary"
                ),
            },
            {
                "protocol": "custom schedule preserving O1 first-200 LR values",
                "status": "NOT_STOCK_RUNNER",
                "reason": "requires new scheduler semantics and a separate PI gate",
            },
        ],
        "execution_started": False,
        "next_gate": "GPT_PI_REVIEW_AFTER_STAGE27_H2_SEMANTICS_AUDIT",
    }


def run_audit(
    relay_root: Path = DEFAULT_RELAY_ROOT,
    groot_repo: Path = DEFAULT_GROOT_REPO,
    project_repo: Path = DEFAULT_PROJECT_REPO,
) -> dict[str, Any]:
    """Run the complete metadata/source audit and return a JSON report."""

    relay_root = relay_root.resolve()
    manifest_path = relay_root / "configs/candidate_200.run_manifest.json"
    trainer_state_path = relay_root / "candidates/trainer_state_200.json"
    train_log_path = relay_root / "candidates/train_200.log"
    command_config_path = relay_root / "configs/commands_and_config.txt"
    seal_path = relay_root / "checkpoint_seal/checkpoint_file_seal.tsv"
    provenance_path = relay_root / "provenance.json"
    scope_path = relay_root / "phase_a_scope.json"

    evidence_hashes = {
        "run_manifest": _require_sha(manifest_path, EXPECTED_O1_MANIFEST_SHA256, "run manifest"),
        "trainer_state": _require_sha(
            trainer_state_path, EXPECTED_O1_TRAINER_STATE_SHA256, "trainer state"
        ),
        "train_log": _require_sha(train_log_path, EXPECTED_O1_TRAIN_LOG_SHA256, "train log"),
        "commands_and_config": _require_sha(
            command_config_path, EXPECTED_COMMAND_CONFIG_SHA256, "commands/config"
        ),
        "checkpoint_seal": _require_sha(seal_path, EXPECTED_O1_SEAL_SHA256, "checkpoint seal"),
        "provenance": _require_sha(provenance_path, EXPECTED_PROVENANCE_SHA256, "provenance"),
        "phase_a_scope": _require_sha(scope_path, EXPECTED_SCOPE_SHA256, "phase-A scope"),
    }
    manifest = _load_json(manifest_path)
    trainer_state = _load_json(trainer_state_path)
    provenance = _load_json(provenance_path)
    scope = _load_json(scope_path)
    seal_rows = _candidate_seal_rows(_parse_tsv_seal(seal_path), "candidate_200")

    required_manifest_values = {
        "schema": "STAGE20S_GR00T_TRAINING_RUN_V1",
        "status": "PASS",
        "max_steps": EXPECTED_O1_STEPS,
        "save_steps": EXPECTED_O1_STEPS,
        "load_bf16": True,
        "tune_projector": True,
        "tune_diffusion_model": True,
        "tune_visual": False,
        "tune_llm": False,
        "tune_vlln": False,
        "base_model_index_sha256": "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59",
    }
    for key, expected in required_manifest_values.items():
        if manifest.get(key) != expected:
            raise AuditError(f"run manifest {key} mismatch: {manifest.get(key)!r} != {expected!r}")
    if provenance.get("groot_sha") != EXPECTED_GROOT_SHA:
        raise AuditError("Stage23 provenance does not bind the expected GR00T revision")
    if provenance.get("frozen_git_sha") != STAGE23_FROZEN_GIT_SHA:
        raise AuditError("Stage23 provenance does not bind the expected project revision")
    if provenance.get("launcher_sha256") != EXPECTED_STAGE20_LAUNCHER_SHA256:
        raise AuditError("Stage23 provenance does not bind the expected launcher")
    if not provenance.get("phase_a_no_new_model_forward"):
        raise AuditError("Stage23 Phase-A no-forward boundary is not true")
    for key in (
        "new_model_forward",
        "new_training_after_200",
        "pbs_submitted",
        "validation_e_read",
        "holdout_f_read",
        "wbc",
        "hardware",
    ):
        if scope.get(key) is not False:
            raise AuditError(f"Stage23 Phase-A scope {key} is not false")

    state_summary = checkpoint_state_summary(manifest, trainer_state)
    sealed_state_paths = _state_artifact_paths(
        [str(row["relative_path"]) for row in seal_rows]
    )
    if sealed_state_paths:
        raise AuditError(
            "checkpoint seal unexpectedly contains optimization-state files: "
            f"{sealed_state_paths}"
        )
    if state_summary["all_state_artifact_files"]:
        raise AuditError(
            "optimization-state artifacts unexpectedly present: "
            f"{state_summary['all_state_artifact_files']}"
        )
    if state_summary["trainer_state_embedded_state_like_keys"]:
        raise AuditError(
            "trainer_state unexpectedly embeds optimization state: "
            f"{state_summary['trainer_state_embedded_state_like_keys']}"
        )
    if state_summary["global_step"] != EXPECTED_O1_STEPS or state_summary["max_steps"] != EXPECTED_O1_STEPS:
        raise AuditError("trainer_state does not bind global_step=max_steps=200")

    source_report = _source_audit(groot_repo.resolve(), project_repo.resolve())
    schedule_report = optimization_schedule_summary(trainer_state)

    return {
        "schema": STAGE27_SCHEMA,
        "status": "PASS",
        "stage": "Stage27 H2 optimization-semantics audit",
        "stage27_base_sha": STAGE27_BASE_SHA,
        "execution_boundary": protocol_plan()["audit_scope"],
        "checkpoint_evidence": {
            "relay_root": str(relay_root),
            "full_checkpoint_shards_present_in_relay": False,
            "tensor_payloads_read": False,
            "o1_index_sha256": EXPECTED_O1_INDEX_SHA256,
            "o1_seal_sha256": EXPECTED_O1_SEAL_SHA256,
            "sealed_optimization_state_files": sealed_state_paths,
            "candidate_200_seal_rows": seal_rows,
            "evidence_hashes": evidence_hashes,
            "run_manifest_recipe_fields": {
                key: manifest.get(key) for key in required_manifest_values
            },
        },
        "pinned_source": source_report,
        "checkpoint_state": state_summary,
        "optimization_schedule": schedule_report,
        "answers": {
            "exact_resume": False,
            "exact_resume_answer": (
                "NO: O1@200 is a model-only checkpoint under the exact Stage23 wrapper. "
                "Weights plus trainer_state/global_step do not reproduce optimizer, "
                "scheduler, RNG, or sampler continuation state."
            ),
            "official_entrypoint_resume": (
                "The pinned experiment.run calls check_resume_compatibility before model "
                "construction and rejects save_only_model=True with resume_from_checkpoint=True."
            ),
            "max_steps_schedule_effect": (
                "With warmup_ratio=0.05 and warmup_steps=0, max_steps controls both "
                "ceil(total_steps*0.05) warmup length and the cosine denominator."
            ),
            "fresh_base_2000_first_200": (
                "DIFFERENT: the first logged LR divergence is global step 2; only "
                "the zero-valued step 1 is equal."
            ),
            "least_confounded_protocol": protocol_plan()["h2_protocol_decision_aid"],
        },
        "claim_boundary": (
            "Stage27 establishes optimization/checkpoint semantics only. It does not "
            "establish H2 efficacy, local-flow quality, task success, generalization, "
            "representation validity, WBC validity, or hardware readiness."
        ),
        "scientific_contract_changed": False,
        "downstream_authorization_still_valid": True,
        "downstream_authorization_status": "HOPPER_H2_TRAINING_HOLD_NO_TOKEN_GRANTED",
        "next_gate": "GPT_PI_REVIEW_AFTER_STAGE27_H2_SEMANTICS_AUDIT",
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relay-root", type=Path, default=DEFAULT_RELAY_ROOT)
    parser.add_argument("--groot-repo", type=Path, default=DEFAULT_GROOT_REPO)
    parser.add_argument("--project-repo", type=Path, default=DEFAULT_PROJECT_REPO)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if args.plan_only:
        print(json.dumps(protocol_plan(), indent=2, sort_keys=True))
        return 0
    report = run_audit(args.relay_root, args.groot_repo, args.project_repo)
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
