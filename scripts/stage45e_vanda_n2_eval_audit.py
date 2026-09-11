#!/usr/bin/env python3
"""CPU-only audit and admission-plan builder for the Stage45E N2 evaluator.

This module is intentionally independent of the Corelab training runner.  It
audits the first N2 science commit, the later N2 TRAIN24 evaluator, and an
explicitly supplied N2 statistics artifact without importing Torch, GR00T,
CUDA, PBS, a simulator, or a model.  The optional artifact check is read-only.

The emitted JSON is a static preparation record.  It is not an execution
authorization and contains zero model-side counters.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "manifests/STAGE45E_VANDA_N2_EVAL_PREPARATION_V1.json"
EVALUATOR_PATH = REPO / "scripts/stage45e_n2_train24_eval.py"
BASE_EXECUTION_HEAD = "a843294780c1392369bfc987e468857c63fc13a5"
N2_FIRST_SCIENCE_COMMIT = "5080d532afebaf30b1b0d636d0f9afc365ada5a6"
EVALUATOR_SOURCE_HEAD = "341e1dca76af5bf3d88ac920da02feb2a25e327b"
PREPARATION_BRANCH = "stage45e/vanda-n2-eval-preparation"
RETURN_TOKEN = "STAGE45E_N2_VANDA_EVAL_SURFACE_READY_FOR_PI_GATE"

EXPECTED_SCIENCE_FILES = {
    "docs/stages/STAGE45E_N2_CORELAB.md",
    "manifests/STAGE45E_N2_BALANCED_XYZ_NORMALIZATION_V1.json",
    "scripts/stage45e_n2_corelab.py",
    "scripts/stage45e_n2_statistics.py",
    "src/go2arx5_vla/stage38_structured_training.py",
    "tests/test_stage45e_n2.py",
}
RECIPE_DRIFT_TOKENS = (
    "optimizer",
    "scheduler",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "gradient_accumulation",
    "bf16",
    "fp16",
    "tf32",
    "precision",
    "resume",
    "save_steps",
    "max_steps",
    "num_train",
    "trainable",
    "loss",
)
FORBIDDEN_TOP_LEVEL_IMPORTS = {
    "torch",
    "gr00t",
    "transformers",
    "isaacgym",
    "omni",
}
FORBIDDEN_CALL_ATTRIBUTES = {
    "get_action",
    "backward",
    "save_pretrained",
    "save_model",
    "save_checkpoint",
    "step",
}
LOCKED_ZERO_KEYS = (
    "gpu",
    "pbs_submissions",
    "model_construction",
    "model_forward",
    "training",
    "backward_steps",
    "optimizer_steps",
    "scheduler_steps",
    "checkpoint_writes",
    "validation_e_read",
    "holdout_f_read",
    "wbc_steps",
    "simulator_steps",
    "hardware",
)


class N2EvalAuditError(RuntimeError):
    """Raised when the static N2 evaluation surface is not admissible."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise N2EvalAuditError(message)


def _read_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"JSON must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise N2EvalAuditError(f"invalid JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    _require(not path.exists() and not path.is_symlink(), f"refusing to overwrite audit output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(str(path.parent), flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise N2EvalAuditError(f"git {' '.join(args)} failed: {exc.output.strip()}") from exc


def _git_object_exists(commit: str) -> None:
    _git("cat-file", "-e", f"{commit}^{{commit}}")


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", ancestor, descendant],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise N2EvalAuditError(f"git merge-base failed: {result.stderr.strip()}")
    return result.returncode == 0


def _top_level_imports(tree: ast.Module) -> list[str]:
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    return imports


def _function_text(source: str, tree: ast.Module, name: str) -> str:
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    _require(len(matches) == 1, f"expected one top-level function {name}")
    node = matches[0]
    lines = source.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1 : node.end_lineno])


def _call_names(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
        elif isinstance(node.func, ast.Name):
            names.append(node.func.id)
    return names


def _validate_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest = _read_json(path)
    _require(manifest.get("schema") == "VANDA_STAGE45E_N2_TRAIN24_EVAL_PREPARATION_V1", "evaluation manifest schema drift")
    _require(manifest.get("status") == "CPU_STATIC_N2_EVAL_SURFACE_READY", "evaluation manifest is not static-ready")
    authority = manifest.get("authority")
    _require(
        authority == {
            "pi_gate": "PI_GATE_STATE_0121",
            "issue_number": 11,
            "issue_comment": 5633708741,
            "issue_50_pointer_comment": 5633717912,
        },
        "evaluation authority drift",
    )
    repository = manifest.get("repository")
    _require(isinstance(repository, Mapping), "evaluation repository binding is missing")
    _require(repository.get("preparation_branch") == PREPARATION_BRANCH, "preparation branch drift")
    _require(repository.get("evaluator_branch") == "stage45e/n2-balanced-xyz-normalization", "evaluator branch drift")
    _require(repository.get("base_execution_head") == BASE_EXECUTION_HEAD, "base execution HEAD drift")
    _require(repository.get("n2_first_science_commit") == N2_FIRST_SCIENCE_COMMIT, "N2 science commit drift")
    _require(repository.get("evaluator_source_head") == EVALUATOR_SOURCE_HEAD, "evaluator source HEAD drift")

    cohort = manifest.get("cohort")
    _require(isinstance(cohort, Mapping), "cohort binding is missing")
    _require(cohort.get("name") == "FULL24_EXISTING_TRAIN", "cohort name drift")
    _require(cohort.get("split") == "TRAIN", "cohort split drift")
    _require(cohort.get("window_count") == 24, "TRAIN24 window count drift")
    _require(cohort.get("window_order") == "target_major_then_snapshot_major", "window order drift")
    _require(cohort.get("target_xyz_in_model_input") is False, "target exclusion drift")
    _require(cohort.get("validation_e_read") is False and cohort.get("holdout_f_read") is False, "locked split drift")

    _require(
        manifest.get("checkpoint_order") == ["FULL24@1000", "FULL24@2000", "N2@1000", "N2@2000"],
        "checkpoint order drift",
    )
    statistics = manifest.get("statistics_contract")
    _require(isinstance(statistics, Mapping), "statistics contract is missing")
    expected_statistics = {
        "source_schema": "CORELAB_STAGE45B_FULL24_RUNTIME_PARQUET_STATS_V2",
        "n2_schema": "CORELAB_STAGE45E_N2_STRUCTURED_STATS_V1",
        "normalization": "MINMAX_TO_NEG1_POS1",
        "timing_mode": "T16",
        "action_horizon": 40,
        "universal_action_dim": 132,
        "active_scalar_count": 190,
        "xyz_universal_dimensions": [0, 1, 2],
        "n2_common_half_range_m": 0.107349471,
        "non_xyz_statistics_must_match_generated_train_statistics": True,
        "source_runtime_v2_outer_sha256": "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337",
        "source_runtime_v2_payload_sha256": "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75",
    }
    _require(dict(statistics) == expected_statistics, "statistics contract drift")

    sampler = manifest.get("sampler_contract")
    _require(isinstance(sampler, Mapping), "sampler contract is missing")
    _require(sampler.get("noise_seed") == 4242, "evaluation noise seed drift")
    _require(sampler.get("timestep_ids") == [0, 250, 500, 750], "evaluation timestep drift")
    _require(sampler.get("steps") == 4 and sampler.get("dt") == 0.25, "evaluation Euler contract drift")
    _require(sampler.get("same_noise_across_checkpoints") is True, "same-noise contract drift")

    admission = manifest.get("admission_contract")
    _require(isinstance(admission, Mapping), "admission contract is missing")
    _require(admission.get("runtime_artifact_paths_explicit") is True, "runtime paths are not explicit")
    _require(admission.get("current_checkout_head_must_be_observed") is True, "observed HEAD binding is missing")
    _require(admission.get("expected_head_must_be_bound_at_execution") is True, "expected HEAD binding is missing")
    _require(admission.get("fresh_output_root_required") is True, "fresh output requirement is missing")
    _require(admission.get("legacy_raw_evidence_verified_before_model_import") is True, "legacy raw admission ordering drift")
    _require(admission.get("n2_statistics_verified_before_model_import") is True, "N2 stats admission ordering drift")
    _require(admission.get("pre_model_zero_boundary_required") is True, "pre-model zero boundary drift")
    _require(admission.get("model_import_after_admission_only") is True, "late-import contract drift")
    _require(
        admission.get("admission_order")
        == [
            "checkout_and_manifest",
            "training_result_and_checkpoint_identity",
            "dataset_and_statistics_identity",
            "sealed_legacy_raw_identity",
            "runtime_observation",
            "pre_model_zero_boundary",
            "write_admission_seal",
            "late_model_import",
        ],
        "admission order drift",
    )

    raw_first = manifest.get("raw_first_contract")
    _require(isinstance(raw_first, Mapping), "raw-first contract is missing")
    for key in (
        "raw_array_written_before_decode",
        "raw_file_sha256_recorded_before_decode",
        "raw_array_atomic_replace",
        "raw_file_fsync",
        "raw_parent_directory_fsync",
        "raw_manifest_append_fsync",
        "derived_metrics_may_reference_only_sealed_raw",
    ):
        _require(raw_first.get(key) is True, f"raw-first contract drift: {key}")
    _require(raw_first.get("legacy_baselines_reinference") is False, "legacy baseline reinference is enabled")

    boundaries = manifest.get("locked_boundaries")
    _require(isinstance(boundaries, Mapping), "locked boundaries are missing")
    for key in LOCKED_ZERO_KEYS:
        expected = False if key in {"training", "validation_e_read", "holdout_f_read", "hardware"} else 0
        _require(boundaries.get(key) == expected, f"locked boundary drift: {key}")
    _require(manifest.get("return_token") == RETURN_TOKEN, "return token drift")
    return manifest


def _validate_science_commit() -> dict[str, Any]:
    _git_object_exists(BASE_EXECUTION_HEAD)
    _git_object_exists(N2_FIRST_SCIENCE_COMMIT)
    _git_object_exists(EVALUATOR_SOURCE_HEAD)
    current_head = _git("rev-parse", "HEAD")
    _require(_is_ancestor(BASE_EXECUTION_HEAD, N2_FIRST_SCIENCE_COMMIT), "N2 science commit is not based on accepted execution HEAD")
    _require(_is_ancestor(N2_FIRST_SCIENCE_COMMIT, EVALUATOR_SOURCE_HEAD), "evaluator source is not based on first N2 science commit")
    _require(_is_ancestor(EVALUATOR_SOURCE_HEAD, current_head), "current preparation branch lost evaluator source history")

    changed = set(_git("diff", "--name-only", "--no-renames", BASE_EXECUTION_HEAD, N2_FIRST_SCIENCE_COMMIT).splitlines())
    _require(changed == EXPECTED_SCIENCE_FILES, f"first N2 science commit changed unexpected files: {sorted(changed)}")
    source_diff = _git(
        "diff",
        "--unified=0",
        BASE_EXECUTION_HEAD,
        N2_FIRST_SCIENCE_COMMIT,
        "--",
        "src/go2arx5_vla/stage38_structured_training.py",
    )
    recipe_hits = [
        line
        for line in source_diff.splitlines()
        if line.startswith(("+", "-"))
        and not line.startswith(("+++", "---"))
        and any(token in line.lower() for token in RECIPE_DRIFT_TOKENS)
    ]
    _require(not recipe_hits, f"N2 science diff changes frozen training recipe: {recipe_hits[:5]}")
    n2_corelab_source = (REPO / "scripts/stage45e_n2_corelab.py").read_text(encoding="utf-8")
    for marker in ('"seed": 2222', '"max_steps": TRAINING_STEPS', '"save_steps": [1000, 2000]'):
        _require(marker in n2_corelab_source, f"N2 training recipe marker missing: {marker}")
    return {
        "base_execution_head": BASE_EXECUTION_HEAD,
        "n2_first_science_commit": N2_FIRST_SCIENCE_COMMIT,
        "evaluator_source_head": EVALUATOR_SOURCE_HEAD,
        "current_preparation_head": current_head,
        "first_science_changed_files": sorted(changed),
        "stage38_training_recipe_diff_hits": recipe_hits,
        "first_science_commit_only_xyz_stats_override": True,
    }


def _validate_evaluator_source() -> dict[str, Any]:
    source = EVALUATOR_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(EVALUATOR_PATH))
    imports = _top_level_imports(tree)
    forbidden_imports = sorted(
        name for name in imports if name.split(".")[0] in FORBIDDEN_TOP_LEVEL_IMPORTS
    )
    _require(not forbidden_imports, f"evaluator has top-level model imports: {forbidden_imports}")
    call_hits = sorted(set(_call_names(tree)).intersection(FORBIDDEN_CALL_ATTRIBUTES | {"qsub"}))
    _require(not call_hits, f"evaluator has forbidden runtime calls: {call_hits}")

    run_text = _function_text(source, tree, "run")
    run_markers = [
        "_verify_evaluator_checkout(",
        "_verify_external(",
        "_verify_sealed_baseline(",
        "_gpu_identity(",
        "_prepare_environment(",
        "stage45e_n2_train24_eval_admission.json",
        "_read_legacy_rows(",
        "_run_n2_inference(",
    ]
    run_positions = [run_text.find(marker) for marker in run_markers]
    _require(all(position >= 0 for position in run_positions), "evaluator admission/raw markers are incomplete")
    _require(run_positions == sorted(run_positions), "evaluator admission order is not fail-closed")

    inference_tree = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_n2_inference"
    )
    inference_text = _function_text(source, tree, "_run_n2_inference")
    raw_markers = [
        "_write_empty_durable(",
        "_write_npy_durable(",
        "raw_file_sha = sha256_file(",
        "_decode_with_stats(",
        "_normalized_metrics(",
        "_evaluation_row(",
    ]
    raw_positions = [inference_text.find(marker) for marker in raw_markers]
    _require(raw_positions[0] >= 0, "raw manifest is not created through durable writer")
    _require(all(position >= 0 for position in raw_positions[1:]), "raw-first evidence markers are incomplete")
    _require(raw_positions[1:] == sorted(raw_positions[1:]), "raw prediction is not durable before derived decode")
    _require("import torch" in inference_text and "from stage38_structured_offline_evaluation" in inference_text, "model runtime is not late-bound")
    _require("n2.verify_n2_document" in source and "n2_stats" in inference_text, "N2 statistics identity is not bound")
    _require("legacy_baselines_reinference" not in source or "False" in source, "legacy baseline policy marker is missing")
    return {
        "path": str(EVALUATOR_PATH.relative_to(REPO)),
        "top_level_imports": imports,
        "top_level_forbidden_imports": forbidden_imports,
        "forbidden_runtime_call_hits": call_hits,
        "admission_order": run_markers,
        "raw_first_order": raw_markers,
        "late_model_import": True,
        "n2_stats_identity_binding": True,
        "durable_raw_writer": True,
        "status": "PASS",
    }


def _validate_n2_artifact(source_path: Path, artifact_path: Path) -> dict[str, Any]:
    scripts = REPO / "scripts"
    src = REPO / "src"
    for path in (scripts, src):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import stage45e_n2_statistics as n2  # noqa: E402

    source = _read_json(source_path)
    artifact = _read_json(artifact_path)
    source_identity = n2.validate_source_document(source, source_path=source_path)
    verified = n2.verify_n2_document(artifact, source=source)
    intervention = artifact.get("intervention")
    _require(isinstance(intervention, Mapping), "N2 artifact intervention is missing")
    _require(intervention.get("common_half_range_m") == n2.HALF_RANGE_M, "N2 half-range drift")
    _require(intervention.get("xyz_universal_dimensions") == [0, 1, 2], "N2 XYZ dimension drift")
    _require(verified.get("all_non_xyz_statistics_exact") is True, "N2 non-XYZ statistics are not exact")
    boundary = artifact.get("execution_boundary")
    _require(isinstance(boundary, Mapping), "N2 artifact execution boundary is missing")
    for key in ("gpu", "gr00t_forwards", "model_constructed", "training_steps", "backward_steps", "optimizer_steps", "scheduler_steps", "checkpoint_writes", "pbs_submissions", "simulator_steps", "validation_e_read", "holdout_f_read"):
        if key in boundary:
            expected = False if key in {"gpu", "model_constructed", "validation_e_read", "holdout_f_read"} else 0
            _require(boundary.get(key) == expected, f"N2 artifact execution boundary is nonzero: {key}")
    return {
        "source_path": str(source_path),
        "source_sha256": _sha256_file(source_path),
        "source_identity": source_identity,
        "artifact_path": str(artifact_path),
        "artifact_sha256": _sha256_file(artifact_path),
        "artifact_status": artifact.get("status"),
        "verification": verified,
        "execution_boundary_zero": True,
    }


def build_static_audit(
    *,
    manifest_path: Path = MANIFEST_PATH,
    source_stats: Path | None = None,
    n2_stats: Path | None = None,
) -> dict[str, Any]:
    _require((source_stats is None) == (n2_stats is None), "source and N2 stats must be supplied together")
    manifest = _validate_manifest(manifest_path)
    science = _validate_science_commit()
    evaluator = _validate_evaluator_source()
    artifact = None
    if source_stats is not None and n2_stats is not None:
        artifact = _validate_n2_artifact(Path(source_stats).resolve(), Path(n2_stats).resolve())
    boundaries = dict(manifest["locked_boundaries"])
    return {
        "schema": "VANDA_STAGE45E_N2_EVAL_STATIC_AUDIT_V1",
        "status": "PASS_STATIC_NOT_EXECUTED",
        "return_token": RETURN_TOKEN,
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest": {
            "schema": manifest["schema"],
            "pi_gate": manifest["authority"]["pi_gate"],
            "issue_comment": manifest["authority"]["issue_comment"],
            "checkpoint_order": manifest["checkpoint_order"],
            "window_count": manifest["cohort"]["window_count"],
            "active_scalar_count": manifest["statistics_contract"]["active_scalar_count"],
            "n2_half_range_m": manifest["statistics_contract"]["n2_common_half_range_m"],
        },
        "science_commit_audit": science,
        "evaluator_source_audit": evaluator,
        "n2_artifact_audit": artifact,
        "execution_boundary": boundaries,
        "gpu": False,
        "pbs_submissions": 0,
        "model_construction": 0,
        "model_forward": 0,
        "training": False,
        "validation_e_read": False,
        "holdout_f_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--source-stats", type=Path)
    parser.add_argument("--n2-stats", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = build_static_audit(
            manifest_path=args.manifest,
            source_stats=args.source_stats,
            n2_stats=args.n2_stats,
        )
        if args.output is not None:
            _write_json(args.output.resolve(), result)
    except Exception as exc:  # pragma: no cover - fail-closed CLI boundary
        print(f"STAGE45E_N2_VANDA_EVAL_AUDIT_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
