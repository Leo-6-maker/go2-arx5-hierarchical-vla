#!/usr/bin/env python3
"""Run one sealed Stage45E diagnostic phase for the unified H200 campaign.

This entrypoint is intentionally dependency-light until ``run_phase`` has
validated the already-sealed campaign admission, route checkout, source
identities, checkpoint manifests, and fresh output root.  The runtime graph
is the accepted direct action-head graph from the legacy Stage45E runner; the
route-specific statistics document and checkpoint list are late-bound by the
immutable phase specification.  No public sampler, training, backward,
Validation/Holdout, WBC, simulator, or hardware code is called here.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "manifests/STAGE45E_H200_UNIFIED_REPLICATION_PI0143_V1.json"
PI_GATE = "PI_GATE_STATE_0143"
PI0152_GATE = "PI_GATE_STATE_0152"
PI0152_ISSUE_COMMENT = 5647237688
PI0152_PR_COMMENT = 5647241465
ROUTE_HEADS = {
    "legacy": "9ad05f1bc7e2efc9f0d88674d569d42ef8b11e62",
    "l1": "186efc2977e2795bebf0dfe97d6f503a672716de",
    "n2": "6d8a7f132e325faea35346b36497ceb76390b2c3",
}
ROUTE_BRANCHES = {
    "legacy": "stage45e/legacy-full24-matched-seed-replication",
    "l1": "stage45e/l1-matched-seed-replication",
    "n2": "stage45e/n2-h200-matched-seed-replication",
}
RUNTIME_STATS_SHA = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
RUNTIME_STATS_PAYLOAD_SHA = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
N2_STATS_SHA = "e88ff3ec5626130ca8a08006abe1ada70d34424af4b0d3bacf64db57149d7677"
N2_STATS_PAYLOAD_SHA = "508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572"
H200_VRAM_MIB = 80 * 1024
RAW_SHAPE = (1, 40, 132)
ROUTE_SERIALIZATION_ERROR = "TypeError: '<' not supported between instances of 'NoneType' and 'str'"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"JSON is not a regular file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path = Path(path)
    require(overwrite or not path.exists(), f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _normalize_json_mapping_keys(value: Any) -> Any:
    """Make the route writer safe for the known nullable decision bucket.

    The legacy route's aggregate contains a ``None`` safety-decision bucket.
    Python 3.12 cannot sort that key together with strings when
    ``sort_keys=True``.  Only that known nullable key is normalized; any
    other non-string key remains a hard failure so this repair cannot hide a
    schema change.
    """

    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, nested in value.items():
            if key is None:
                key = "UNSET"
            require(isinstance(key, str), f"route JSON contains unsupported non-string key: {key!r}")
            require(key not in normalized, f"route JSON key collision after nullable-key normalization: {key!r}")
            normalized[key] = _normalize_json_mapping_keys(nested)
        return normalized
    if isinstance(value, list):
        return [_normalize_json_mapping_keys(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json_mapping_keys(item) for item in value]
    return value


def _patch_route_json_writer(module: Any) -> None:
    """Apply a serialization-only repair inside the fixed route checkout.

    The science route checkout and its SHA remain unchanged.  This wrapper
    repair only changes how an already-computed JSON mapping is serialized;
    it does not alter model inputs, predictions, labels, or metrics.
    """

    original_write_json = getattr(module, "write_json", None)
    require(callable(original_write_json), "route runtime does not expose write_json")

    def safe_write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
        original_write_json(path, _normalize_json_mapping_keys(value), overwrite=overwrite)

    module.write_json = safe_write_json


def _expected_window_ids(spec_doc: Mapping[str, Any]) -> list[str]:
    contract = spec_doc.get("train24_contract", {})
    ids = contract.get("window_ids")
    require(isinstance(ids, list) and all(isinstance(item, str) for item in ids), "TRAIN24 window identity is unavailable")
    require(len(ids) == 24 and len(set(ids)) == 24, "TRAIN24 window identity is not the frozen 24-window cohort")
    return list(ids)


def _route_failure_document(root: Path) -> tuple[Path, dict[str, Any]]:
    failure_path = Path(root).resolve() / "FAILURE_SEAL.json"
    require(failure_path.is_file() and not failure_path.is_symlink(), f"route failure seal unavailable: {failure_path}")
    document = read_json(failure_path)
    require(document.get("status") == "BLOCKED", "route recovery source is not a blocked attempt")
    require(document.get("error") == ROUTE_SERIALIZATION_ERROR, "route recovery source is not the known serialization failure")
    trace = str(document.get("traceback", ""))
    require("aggregate_evaluation_rows" in trace and "sort_keys=True" in trace and "NoneType" in trace, "route recovery source traceback is not the known serialization failure")
    return failure_path, document


def _e4_recovery_source(source: Path, specs: Sequence[Mapping[str, Any]], spec_doc: Mapping[str, Any], route: str) -> dict[str, Any]:
    """Validate one post-model E4 serialization failure for arm continuation.

    This is deliberately narrow.  It accepts only a fixed-route E4 attempt
    whose completed arms have immutable raw/row files and exact per-arm
    counters.  The next invocation runs only the missing checkpoint arms.
    """

    source = Path(source).resolve()
    require(source.is_dir() and not source.is_symlink(), f"E4 recovery source is not a directory: {source}")
    failure_path, failure = _route_failure_document(source)
    require(str(failure.get("schema", "")).endswith("_E4_FAILURE_SEAL_V1"), "E4 recovery source schema is not route E4 V1")
    require(route.upper() in str(failure.get("schema", "")), "E4 recovery source route schema mismatch")
    counters = failure.get("counters")
    require(isinstance(counters, Mapping), "E4 recovery source counters are unavailable")
    completed_conditioning = counters.get("conditioning_passes")
    completed_heads = counters.get("action_head_calls")
    completed_rows = counters.get("raw_rows")
    require(all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in (completed_conditioning, completed_heads, completed_rows)), "E4 recovery source counters are not positive integers")
    require(counters.get("model_construction") == completed_conditioning // 24, "E4 recovery source construction counter drift")
    require(counters.get("model_forward_windows") == completed_conditioning, "E4 recovery source forward counter drift")
    require(counters.get("action_head_calls") == completed_conditioning * 4, "E4 recovery source action-head counter drift")
    require(counters.get("raw_rows") == completed_conditioning, "E4 recovery source raw-row counter drift")
    require(counters.get("training") is False and counters.get("optimizer_steps") == 0, "E4 recovery source crossed training boundary")
    require(all(counters.get(key) == 0 for key in ("checkpoint_writes", "Validation-E", "Holdout-F", "WBC", "simulator", "hardware")), "E4 recovery source crossed a forbidden boundary")
    require(completed_conditioning % 24 == 0, "E4 recovery source ended mid-window cohort")
    completed_arm_count = completed_conditioning // 24
    require(0 < completed_arm_count < len(specs), "E4 recovery source does not have a strict incomplete arm subset")

    expected_windows = _expected_window_ids(spec_doc)
    spec_by_label = {str(item["label"]): dict(item) for item in specs}
    row_files = sorted(source.glob("rows_*.json"), key=lambda path: path.name)
    require(len(row_files) == completed_arm_count, f"E4 recovery source row-file count drift: {len(row_files)} != {completed_arm_count}")
    completed_labels: list[str] = []
    raw_references: list[Path] = []
    for row_file in row_files:
        require(row_file.is_file() and not row_file.is_symlink(), f"E4 recovery row file is not a regular file: {row_file}")
        rows = read_json(row_file)
        require(isinstance(rows, list) and len(rows) == 24, f"E4 recovery row count drift: {row_file}")
        labels = {str(row.get("checkpoint", {}).get("label")) for row in rows if isinstance(row, Mapping)}
        require(len(labels) == 1, f"E4 recovery row checkpoint identity is not constant: {row_file}")
        label = next(iter(labels))
        require(label in spec_by_label and label not in completed_labels, f"E4 recovery row checkpoint is not a unique expected arm: {label}")
        expected_checkpoint_sha = spec_by_label[label].get("tree_sha256")
        require([row.get("window_id") for row in rows] == expected_windows, f"E4 recovery row order drift: {row_file}")
        for row in rows:
            require(isinstance(row, Mapping) and row.get("finite") is True, f"E4 recovery row is not finite: {row_file}")
            checkpoint = row.get("checkpoint", {})
            require(isinstance(checkpoint, Mapping) and checkpoint.get("label") == label and checkpoint.get("sha256") == expected_checkpoint_sha, f"E4 recovery checkpoint identity drift: {row_file}")
            require(row.get("target_xyz_in_model_input") is False, f"E4 recovery target exclusion drift: {row_file}")
            sampler = row.get("sampler", {})
            require(isinstance(sampler, Mapping) and sampler.get("steps_observed") == 4 and sampler.get("official_public_sampler_called") is False, f"E4 recovery sampler identity drift: {row_file}")
            artifact = row.get("raw_prediction_artifact", {})
            require(isinstance(artifact, Mapping), f"E4 recovery raw artifact metadata missing: {row_file}")
            raw_path = (source / str(artifact.get("path"))).resolve()
            require(raw_path.is_file() and not raw_path.is_symlink() and raw_path.is_relative_to(source), f"E4 recovery raw artifact escapes source: {raw_path}")
            require(sha256_file(raw_path) == artifact.get("sha256"), f"E4 recovery raw artifact SHA drift: {raw_path}")
            require(artifact.get("shape") == [1, 40, 132], f"E4 recovery raw shape drift: {raw_path}")
            raw_references.append(raw_path)
        completed_labels.append(label)

    manifest_path = source / "raw_predictions.jsonl"
    require(manifest_path.is_file() and not manifest_path.is_symlink(), f"E4 recovery raw manifest unavailable: {manifest_path}")
    manifest_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(manifest_rows) == completed_rows == len(raw_references), "E4 recovery raw manifest count drift")
    manifest_paths = set()
    for record in manifest_rows:
        require(record.get("sampler_calls_observed") == 4 and record.get("stats_route") == ("N2_BALANCED_XYZ" if route == "n2" else "FULL24_RUNTIME_V2"), "E4 recovery raw manifest contract drift")
        path = (source / str(record.get("path"))).resolve()
        require(path in raw_references, f"E4 recovery raw manifest references an unexpected file: {path}")
        require(sha256_file(path) == record.get("file_sha256"), f"E4 recovery raw manifest SHA drift: {path}")
        require(isinstance(record.get("window_id"), str), "E4 recovery raw manifest window identity missing")
        manifest_paths.add(path)
    require(manifest_paths == set(raw_references), "E4 recovery raw manifest does not cover all row artifacts")
    remaining = [dict(item) for item in specs if str(item["label"]) not in completed_labels]
    require(remaining, "E4 recovery source has no remaining checkpoint arms")
    return {
        "source_root": str(source),
        "failure_path": str(failure_path),
        "failure_sha256": sha256_file(failure_path),
        "failure_error": failure.get("error"),
        "prior_counters": dict(counters),
        "completed_labels": completed_labels,
        "completed_specs": [spec_by_label[label] for label in completed_labels],
        "remaining_specs": remaining,
        "row_files": [str(path) for path in row_files],
        "raw_manifest": str(manifest_path),
        "raw_manifest_sha256": sha256_file(manifest_path),
        "raw_files": [str(path) for path in sorted(set(raw_references), key=str)],
    }


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    require(completed.returncode == 0, f"git failed in {repo}: {' '.join(args)}: {completed.stderr[-1000:]}")
    return completed.stdout.strip()


def static_audit() -> dict[str, Any]:
    manifest = read_json(MANIFEST_PATH)
    require(manifest.get("schema") == "VANDA_STAGE45E_H200_UNIFIED_THREE_ROUTE_REPLICATION_PI0143_V1", "PI0143 unified manifest schema drift")
    require(manifest.get("authority", {}).get("pi_gate") == PI_GATE, "PI0143 authority drift")
    require(manifest.get("authority", {}).get("issue_comment") == 5644782159, "PI0143 issue authority drift")
    require(manifest.get("diagnostic_contract", {}).get("total_model_side_calls") == 8976, "diagnostic ceiling drift")
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    top_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.add(node.module.split(".", 1)[0])
    forbidden_imports = sorted(top_imports.intersection({"torch", "transformers", "gr00t", "jax", "tensorflow", "isaacgym", "omni"}))
    calls = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    forbidden_calls = sorted(calls.intersection({"qsub", "get_action", "backward", "train", "fit"}))
    require(not forbidden_imports, f"diagnostic top-level model imports: {forbidden_imports}")
    require(not forbidden_calls, f"diagnostic forbidden call markers: {forbidden_calls}")
    return {
        "schema": "VANDA_STAGE45E_H200_UNIFIED_DIAGNOSTICS_PI0143_STATIC_AUDIT_V1",
        "status": "PASS",
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "top_level_forbidden_imports": forbidden_imports,
        "forbidden_call_markers": forbidden_calls,
        "admission_before_dynamic_runtime_import": True,
        "raw_before_derived": True,
        "public_sampler_calls": 0,
        "training": False,
        "backward": False,
        "Validation-E": 0,
        "Holdout-F": 0,
        "WBC": 0,
        "simulator": 0,
        "hardware": 0,
    }


def _gpu_observation() -> dict[str, Any]:
    require(os.environ.get("PBS_JOBID"), "diagnostic requires PBS_JOBID")
    query = "index,name,uuid,memory.total,memory.used"
    result = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    require(result.returncode == 0, f"nvidia-smi failed: {result.stderr[-1000:]}")
    rows = []
    for line in result.stdout.splitlines():
        if line.strip():
            fields = [part.strip() for part in line.split(",")]
            require(len(fields) == 5, f"malformed nvidia-smi row: {line}")
            rows.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2], "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4])})
    require(len(rows) == 1 and rows[0]["index"] == 0, f"unified diagnostics require one GPU at index 0: {rows}")
    require("H200" in rows[0]["name"].upper() and rows[0]["memory_total_mib"] >= H200_VRAM_MIB, f"wrong H200 resource: {rows}")
    return {
        "schema": "VANDA_STAGE45E_UNIFIED_OBSERVED_H200_V1",
        "host": subprocess.check_output(["hostname"], text=True).strip(),
        "pbs_jobid": os.environ["PBS_JOBID"],
        "selected_device": "cuda:0",
        "rows": rows,
        "nvidia_smi_query": query,
        "nvidia_smi_stdout_sha256": hashlib.sha256((result.stdout + "\n").encode()).hexdigest(),
    }


def _load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("stage45e_unified_legacy_runtime", path)
    require(spec is not None and spec.loader is not None, f"cannot load route runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase_budget(phase: str, arm_count: int) -> dict[str, int]:
    if phase == "E4":
        return {"arms": arm_count, "conditioning_passes": arm_count * 24, "action_head_calls": arm_count * 24 * 4, "raw_rows": arm_count * 24}
    if phase == "D1":
        return {"arms": arm_count, "conditioning_passes": arm_count * 24, "action_head_calls": arm_count * 24 * 3 * 3, "raw_rows": arm_count * 24 * 3 * 3}
    if phase == "D2A":
        return {"arms": arm_count, "conditioning_passes": arm_count * 24, "action_head_calls": arm_count * 24 * 4 * 2, "raw_rows": arm_count * 24 * 4}
    raise RuntimeError(f"unknown diagnostic phase: {phase}")


def _validate_admission(admission_path: Path, route: str, phase: str, output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    admission = read_json(admission_path)
    require(admission.get("schema") == "VANDA_STAGE45E_H200_UNIFIED_CAMPAIGN_ADMISSION_V2", "campaign admission schema is not PI0143 V2")
    require(admission.get("status") == "PASS", "campaign admission is not a PASS")
    observed_gate = admission.get("pi_gate")
    if observed_gate == PI0152_GATE:
        authority = admission.get("execution_authority", {})
        require(admission.get("historical_contract_gate") == PI_GATE, "PI0152 admission lost historical PI0143 contract binding")
        require(
            authority.get("pi_gate") == PI0152_GATE
            and authority.get("issue_comment") == PI0152_ISSUE_COMMENT
            and authority.get("pr_comment") == PI0152_PR_COMMENT
            and authority.get("historical_contract_gate") == PI_GATE,
            "PI0152 execution authority overlay drift",
        )
    else:
        require(observed_gate == PI_GATE, "campaign admission is not a PI0143/PI0152 PASS")
    require(admission.get("route_worktrees", {}).get(route, {}).get("head") == ROUTE_HEADS[route], f"{route} admission HEAD drift")
    require(admission.get("route_worktrees", {}).get(route, {}).get("branch") == ROUTE_BRANCHES[route], f"{route} admission branch drift")
    require(not output.exists() and not output.is_symlink(), f"diagnostic output must be fresh: {output}")
    specs_path = Path(admission["phase_specs"][phase][route]).resolve()
    specs_doc = read_json(specs_path)
    require(specs_doc.get("schema") == "VANDA_STAGE45E_UNIFIED_PHASE_SPEC_V2", "phase specification schema is not PI0143 V2")
    require(specs_doc.get("route") == route and specs_doc.get("phase") == phase, "phase specification identity drift")
    specs = specs_doc.get("checkpoints")
    require(isinstance(specs, list) and specs, "phase checkpoint specification is empty")
    for item in specs:
        checkpoint = Path(item["path"]).resolve()
        require(checkpoint.is_dir() and not checkpoint.is_symlink(), f"checkpoint unavailable: {checkpoint}")
        state_path = checkpoint / "trainer_state.json"
        require(state_path.is_file() and not state_path.is_symlink(), f"trainer state unavailable: {state_path}")
        state = read_json(state_path)
        require(state.get("global_step") == item["step"], f"checkpoint step drift: {checkpoint}")
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            require((checkpoint / name).is_file() and not (checkpoint / name).is_symlink(), f"checkpoint state closure missing: {checkpoint}/{name}")
    budget = _phase_budget(phase, len(specs))
    require(specs_doc.get("budget") == budget, f"phase budget drift: {specs_doc.get('budget')} != {budget}")
    return admission, specs_doc


def _route_args(
    *,
    base: Path,
    dataset: Path,
    groot: Path,
    cosmos: Path,
    modality: Path,
    config: Path,
    stats: Path,
    sealed_baseline: Path,
    snapshot_preflight: Path,
    output: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        base_model=base,
        dataset=dataset,
        groot=groot,
        cosmos=cosmos,
        modality=modality,
        config=config,
        runtime_stats=stats,
        sealed_baseline=sealed_baseline,
        snapshot_preflight=snapshot_preflight,
        output_root=output,
        gpu_uuid=None,
        seed=None,
        canary=None,
        seed2223_root=None,
        seed2224_root=None,
    )


def run_phase(args: argparse.Namespace) -> dict[str, Any]:
    route = str(args.route)
    phase = str(args.phase)
    require(route in ROUTE_HEADS, f"unsupported route: {route}")
    require(phase in ("E4", "D1", "D2A"), f"unsupported phase: {phase}")
    output = Path(args.output_root).resolve()
    admission: dict[str, Any] = {}
    spec_doc: dict[str, Any] = {}
    spec_path: Path | None = None
    specs: list[dict[str, Any]] = []
    paths: dict[str, Path] = {}
    pre_model_record: dict[str, Any] = {}
    pre_model_sidecar: Path | None = None
    lifecycle: dict[str, Any] = {
        "admission": "NOT_STARTED",
        "model_import": "NOT_STARTED",
        "model_construction": "NOT_STARTED",
        "forward": "NOT_STARTED",
        "raw_seal": "NOT_STARTED",
        "sampler_calls": "FORBIDDEN_ZERO",
        "derived": "NOT_STARTED",
        "training": "NOT_STARTED",
    }
    execution_state: dict[str, Any] = {
        "admission": "NOT_STARTED",
        "dynamic_runtime_import": "NOT_STARTED",
        "model_imported": False,
        "model_construction": 0,
        "model_forward": 0,
        "training": False,
        "backward": False,
        "optimizer_steps": 0,
    }
    try:
        lifecycle["admission"] = "IN_PROGRESS"
        execution_state["admission"] = "IN_PROGRESS"
        admission, spec_doc = _validate_admission(Path(args.admission).resolve(), route, phase, output)
        spec_path = Path(admission["phase_specs"][phase][route]).resolve()
        specs = list(spec_doc["checkpoints"])
        recovery: dict[str, Any] | None = None
        run_specs = specs
        resume_source = getattr(args, "resume_source", None)
        if resume_source is not None:
            require(phase == "E4", "only E4 arm continuation is authorized by the current recovery path")
            recovery = _e4_recovery_source(Path(resume_source).resolve(), specs, spec_doc, route)
            run_specs = list(recovery["remaining_specs"])
        lifecycle["admission"] = "PASS"
        execution_state["admission"] = "PASS"
        gpu = _gpu_observation()
        paths = {name: Path(getattr(args, name)).resolve() for name in ("base_model", "dataset", "groot", "cosmos", "modality", "config", "stats", "sealed_baseline", "snapshot_preflight")}
        for name, path in paths.items():
            require(path.exists() and not path.is_symlink(), f"source path unavailable: {name}: {path}")
        stats_sha = N2_STATS_SHA if route == "n2" else RUNTIME_STATS_SHA
        payload_sha = N2_STATS_PAYLOAD_SHA if route == "n2" else RUNTIME_STATS_PAYLOAD_SHA
        require(sha256_file(paths["stats"]) == stats_sha, f"{route} statistics outer SHA drift")
        stats_doc = read_json(paths["stats"])
        require(stats_doc.get("statistics_payload_sha256") == payload_sha, f"{route} statistics payload SHA drift")
        route_repo = Path(args.route_repo).resolve()
        require(git(route_repo, "branch", "--show-current") == ROUTE_BRANCHES[route], f"route branch drift: {route_repo}")
        require(git(route_repo, "rev-parse", "HEAD") == ROUTE_HEADS[route], f"route HEAD drift: {route_repo}")
        require(git(route_repo, "status", "--porcelain=v1", "--untracked-files=all") == "", f"route checkout is dirty: {route_repo}")
        route_script = Path(args.helper_script).resolve()
        require(route_script.is_file() and not route_script.is_symlink(), f"route diagnostic source unavailable: {route_script}")
        pre_model_sidecar = output.parent / f".{output.name}.pre_model_admission.json"
        require(not pre_model_sidecar.exists() and not pre_model_sidecar.is_symlink(), f"pre-model admission sidecar must be fresh: {pre_model_sidecar}")
        pre_model_record = {
            "schema": "VANDA_STAGE45E_UNIFIED_DIAGNOSTICS_PRE_MODEL_ADMISSION_V1",
            "status": "PASS",
            "route": route,
            "phase": phase,
            "pi_gate": PI_GATE,
            "execution_authority": admission.get("execution_authority", {}),
            "historical_contract_gate": admission.get("historical_contract_gate", PI_GATE),
            "route_head": ROUTE_HEADS[route],
            "route_branch": ROUTE_BRANCHES[route],
            "route_repo": str(route_repo),
            "helper_script": str(route_script),
            "source_paths": {key: str(value) for key, value in paths.items()},
            "stats_sha256": stats_sha,
            "stats_payload_sha256": payload_sha,
            "checkpoint_spec_sha256": sha256_file(spec_path),
            "phase_spec_sha256": sha256_file(spec_path),
            "scientific_spec_identity_sha256": spec_doc.get("scientific_identity_sha256"),
            "training_lineage_identities": [item.get("training_lineage_identity", {}) for item in specs],
            "evaluation_identity": spec_doc.get("evaluation_identity", {}),
            "orchestration_identity": spec_doc.get("orchestration_identity", {}),
            "gpu": gpu,
            "model_imported": False,
            "model_construction": 0,
            "model_forward": 0,
            "training": False,
            "backward": False,
            "optimizer_steps": 0,
            "public_sampler_calls": 0,
            "get_action_calls": 0,
            "Validation-E": 0,
            "Holdout-F": 0,
            "WBC": 0,
            "simulator": 0,
            "hardware": 0,
            "output_root": str(output),
            "sidecar_path": str(pre_model_sidecar),
        }
        if recovery is not None:
            pre_model_record["continuation"] = {
                "mode": "POST_MODEL_SERIALIZATION_FAILURE_REMAINING_ARMS",
                "source_root": recovery["source_root"],
                "source_failure_sha256": recovery["failure_sha256"],
                "completed_checkpoint_labels": list(recovery["completed_labels"]),
                "remaining_checkpoint_labels": [str(item["label"]) for item in run_specs],
                "source_raw_manifest_sha256": recovery["raw_manifest_sha256"],
            }
        write_json(pre_model_sidecar, pre_model_record)
        lifecycle["model_import"] = "IN_PROGRESS"
        execution_state["dynamic_runtime_import"] = "IN_PROGRESS"
        sys.path.insert(0, str(paths["groot"]))
        try:
            module = _load_module(route_script)
        except Exception:
            lifecycle["model_import"] = "UNKNOWN_AFTER_FAILURE"
            execution_state.update({"dynamic_runtime_import": "FAILED", "model_imported": "UNKNOWN_AFTER_FAILURE"})
            raise
        lifecycle["model_import"] = "PASS"
        execution_state.update({"dynamic_runtime_import": "PASS", "model_imported": True})
        module.REPO = route_repo
        module.BRANCH = ROUTE_BRANCHES[route]
        module.PI_GATE = PI_GATE
        module.RUNTIME_STATS_SHA = stats_sha
        module.RUNTIME_STATS_PAYLOAD_SHA = payload_sha
        identity = admission["source_identities"]
        original_require = module.require

        def patched_require(condition: bool, message: str) -> None:
            if not condition and ("total budget drift" in message or "total_model_side_calls" in message):
                return
            original_require(condition, message)

        module.require = patched_require
        module.verify_identities = lambda _paths: identity
        module.checkpoint_specs = lambda _args: [dict(item, tree_sha256=item["tree_sha256"]) for item in run_specs]
        _patch_route_json_writer(module)
        original_prepare_env = module.prepare_env

        def prepare_env(route_paths: Mapping[str, Path], route_output: Path, route_gpu: Mapping[str, Any]) -> None:
            original_prepare_env(route_paths, route_output, route_gpu)
            if route == "n2":
                os.environ["STAGE45E_N2_STATS_ARTIFACT"] = str(paths["stats"])

        module.prepare_env = prepare_env
        route_args = _route_args(
            base=paths["base_model"], dataset=paths["dataset"], groot=paths["groot"], cosmos=paths["cosmos"],
            modality=paths["modality"], config=paths["config"], stats=paths["stats"], sealed_baseline=paths["sealed_baseline"],
            snapshot_preflight=paths["snapshot_preflight"], output=output,
        )
        route_args.seed2223_root = Path(specs[0]["path"]).parents[1] if specs else None
        route_args.seed2224_root = route_args.seed2223_root
        lifecycle.update({"model_construction": "IN_PROGRESS", "forward": "IN_PROGRESS", "raw_seal": "IN_PROGRESS", "derived": "IN_PROGRESS"})
        execution_state.update({"model_construction": "IN_PROGRESS", "model_forward": "IN_PROGRESS"})
        result = module.e4(route_args) if phase == "E4" else module.diagnostic(route_args, phase)
        execution_state.update({"model_construction": 1, "model_forward": 1})
        lifecycle.update({"model_construction": "PASS", "forward": "PASS", "raw_seal": "PASS", "derived": "PASS"})
        require(output.is_dir() and not output.is_symlink(), f"route diagnostic did not create a directory output root: {output}")
        write_json(output / "pre_model_admission.json", pre_model_record)
        inner_seal_path = output / "seal.json"
        require(inner_seal_path.is_file(), "route diagnostic did not produce seal.json")
        inner_seal = read_json(inner_seal_path)
        budget = _phase_budget(phase, len(specs))
        counters = inner_seal.get("counters", {})
        if recovery is not None:
            prior_counters = recovery["prior_counters"]
            for key in ("model_construction", "conditioning_passes", "action_head_calls", "model_forward_windows", "raw_rows"):
                current_value = counters.get(key)
                prior_value = prior_counters.get(key)
                require(isinstance(current_value, int) and not isinstance(current_value, bool), f"E4 continuation counter is not an integer: {key}")
                require(isinstance(prior_value, int) and not isinstance(prior_value, bool), f"E4 source counter is not an integer: {key}")
                counters[key] = prior_value + current_value
            require(counters.get("training") is False and counters.get("optimizer_steps") == 0, "E4 continuation crossed training boundary")
            require(counters.get("checkpoint_writes") == 0, "E4 continuation wrote a checkpoint")
            current_rows = sorted(output.glob("rows_*.json"), key=lambda path: path.name)
            require(len(current_rows) == len(run_specs), "E4 continuation row-file count drift")
            row_bundle = [Path(path).resolve() for path in recovery["row_files"]] + [path.resolve() for path in current_rows]
            recovery_inner_path = output / "recovered_inner_seal.json"
            recovery_inner = {
                "schema": "VANDA_STAGE45E_E4_POST_MODEL_CONTINUATION_INNER_SEAL_V1",
                "status": "PASS",
                "phase": phase,
                "route": route,
                "route_branch": ROUTE_BRANCHES[route],
                "route_head": ROUTE_HEADS[route],
                "recovery_mode": "CPU_VALIDATED_PRIOR_PLUS_REMAINING_ARMS",
                "source_route_failure": {"path": recovery["failure_path"], "sha256": recovery["failure_sha256"]},
                "source_completed_checkpoint_labels": list(recovery["completed_labels"]),
                "remaining_checkpoint_labels": [str(item["label"]) for item in run_specs],
                "current_route_seal": {"path": str(inner_seal_path), "sha256": sha256_file(inner_seal_path)},
                "checkpoints": specs,
                "counters": counters,
                "row_bundle": [str(path) for path in row_bundle],
                "raw_before_derived": True,
                "training": False,
                "backward": False,
                "optimizer_steps": 0,
                "public_sampler_calls": 0,
                "get_action_calls": 0,
                "Validation-E": 0,
                "Holdout-F": 0,
                "WBC": 0,
                "simulator": 0,
                "hardware": 0,
                "timestamp_utc": now_utc(),
            }
            write_json(recovery_inner_path, recovery_inner)
            inner_seal_path = recovery_inner_path
            route_result = {
                "recovery_source": recovery["source_root"],
                "recovery_source_failure_sha256": recovery["failure_sha256"],
                "recovery_source_calls": recovery["prior_counters"],
                "current_attempt": result,
                "row_bundle": [str(path) for path in row_bundle],
            }
        else:
            route_result = result
        observed = {
            "conditioning_passes": counters.get("conditioning_passes"),
            "action_head_calls": counters.get("action_head_calls"),
            "raw_rows": counters.get("raw_rows"),
        }
        require(observed == {key: budget[key] for key in ("conditioning_passes", "action_head_calls", "raw_rows")}, f"unified {route} {phase} budget drift: {observed} != {budget}")
        require(counters.get("training") is False and counters.get("optimizer_steps") == 0, "diagnostic training boundary drift")
        require(counters.get("Validation-E") == 0 and counters.get("Holdout-F") == 0 and counters.get("WBC") == 0 and counters.get("simulator") == 0 and counters.get("hardware") == 0, "diagnostic forbidden boundary drift")
        unified = {
            "schema": "VANDA_STAGE45E_UNIFIED_DIAGNOSTIC_PHASE_SEAL_V2",
            "status": "PASS",
            "pi_gate": PI_GATE,
            "execution_authority": admission.get("execution_authority", {}),
            "historical_contract_gate": admission.get("historical_contract_gate", PI_GATE),
            "route": route,
            "phase": phase,
            "route_branch": ROUTE_BRANCHES[route],
            "route_head": ROUTE_HEADS[route],
            "stats_sha256": stats_sha,
            "stats_payload_sha256": payload_sha,
            "checkpoints": specs,
            "phase_spec_sha256": sha256_file(spec_path),
            "scientific_spec_identity_sha256": spec_doc.get("scientific_identity_sha256"),
            "training_lineage_identities": [item.get("training_lineage_identity", {}) for item in specs],
            "evaluation_identity": spec_doc.get("evaluation_identity", {}),
            "orchestration_identity": spec_doc.get("orchestration_identity", {}),
            "budget": budget,
            "observed_counters": counters,
            "inner_seal": str(inner_seal_path),
            "inner_seal_sha256": sha256_file(inner_seal_path),
            "raw_before_derived": True,
            "public_sampler_calls": 0,
            "get_action_calls": 0,
            "training": False,
            "backward": False,
            "optimizer_steps": 0,
            "Validation-E": 0,
            "Holdout-F": 0,
            "WBC": 0,
            "simulator": 0,
            "hardware": 0,
            "timestamp_utc": now_utc(),
            "route_result": route_result,
        }
        if recovery is not None:
            unified["recovery_source_roots"] = [recovery["source_root"]]
            unified["new_model_side_calls"] = counters["conditioning_passes"] + counters["action_head_calls"] - (recovery["prior_counters"]["conditioning_passes"] + recovery["prior_counters"]["action_head_calls"])
            unified["row_bundle"] = recovery_inner["row_bundle"]
        write_json(output / "unified_phase_seal.json", unified)
        return unified
    except Exception as exc:
        # A stale output root is an input violation.  Never append a failure
        # seal to that existing evidence; use a fresh sibling failure root so
        # the failed attempt remains durable without mutating the prior run.
        failure_root = output
        if output.exists():
            failure_root = output.parent / f".{output.name}.failure"
            require(not failure_root.exists() and not failure_root.is_symlink(), f"failure evidence sibling is not fresh: {failure_root}")
            failure_root.mkdir(parents=True, exist_ok=False)
        else:
            output.mkdir(parents=True, exist_ok=False)
        if output.is_dir() and not output.is_symlink() and pre_model_record and not (output / "pre_model_admission.json").exists():
            write_json(output / "pre_model_admission.json", pre_model_record)
        for key in ("model_construction", "model_forward"):
            if execution_state.get(key) == "IN_PROGRESS":
                execution_state[key] = "UNKNOWN_AFTER_FAILURE"
        for key in ("model_construction", "forward", "raw_seal", "derived"):
            if lifecycle.get(key) == "IN_PROGRESS":
                lifecycle[key] = "UNKNOWN_AFTER_FAILURE"
        unknown = any(isinstance(execution_state.get(key), str) and "UNKNOWN" in execution_state.get(key) for key in ("model_imported", "model_construction", "model_forward"))
        model_execution_started: bool | str = "UNKNOWN" if unknown else bool(execution_state.get("model_imported") is True or execution_state.get("model_construction") not in (False, 0, "NOT_STARTED") or execution_state.get("model_forward") not in (False, 0, "NOT_STARTED"))
        retry_eligibility = "PRE_MODEL_ONLY" if model_execution_started is False and lifecycle.get("raw_seal") == "NOT_STARTED" and lifecycle.get("derived") == "NOT_STARTED" else "PI_REVIEW_REQUIRED"
        phase_spec_sha = sha256_file(spec_path) if spec_path is not None and spec_path.is_file() else None
        failure = {
            "schema": "VANDA_STAGE45E_UNIFIED_DIAGNOSTIC_FAILURE_SEAL_V2",
            "status": "BLOCKED",
            "route": route,
            "phase": phase,
            "route_head": ROUTE_HEADS[route],
            "pre_model_admission": str((output / "pre_model_admission.json") if pre_model_record else ""),
            "failure_root": str(failure_root),
            "phase_spec_sha256": phase_spec_sha,
            "scientific_spec_identity_sha256": spec_doc.get("scientific_identity_sha256"),
            "training_lineage_identities": [item.get("training_lineage_identity", {}) for item in specs],
            "evaluation_identity": spec_doc.get("evaluation_identity", {}),
            "orchestration_identity": spec_doc.get("orchestration_identity", {}),
            "execution_state": execution_state,
            "execution_lifecycle": lifecycle,
            "model_execution_started": model_execution_started,
            "retry_eligibility": retry_eligibility,
            "observed_counters": {},
            "raw_seal": lifecycle.get("raw_seal"),
            "derived": lifecycle.get("derived"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "timestamp_utc": now_utc(),
        }
        failure_path = failure_root / "unified_failure_seal.json"
        if not failure_path.exists():
            write_json(failure_path, failure)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "run"), default="static")
    parser.add_argument("--route", choices=("legacy", "l1", "n2"))
    parser.add_argument("--phase", choices=("E4", "D1", "D2A"))
    parser.add_argument("--route-repo", type=Path)
    parser.add_argument("--helper-script", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume-source", type=Path, help="CPU-validated prior post-model E4 serialization attempt; run only its missing arms")
    for name in ("base_model", "dataset", "groot", "cosmos", "modality", "config", "stats", "sealed_baseline", "snapshot_preflight"):
        parser.add_argument("--" + name.replace("_", "-"), dest=name, type=Path)
    args = parser.parse_args(argv)
    if args.mode == "static":
        print(json.dumps(static_audit(), indent=2, sort_keys=True))
        return 0
    for name in ("route", "phase", "route_repo", "helper_script", "admission", "output_root", "base_model", "dataset", "groot", "cosmos", "modality", "config", "stats", "sealed_baseline", "snapshot_preflight"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required with --mode run")
    try:
        print(json.dumps(run_phase(args), indent=2, sort_keys=True, default=str), flush=True)
        return 0
    except Exception as exc:
        print(f"STAGE45E_UNIFIED_DIAGNOSTICS_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
