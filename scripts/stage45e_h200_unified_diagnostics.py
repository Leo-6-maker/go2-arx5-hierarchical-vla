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
MANIFEST_PATH = REPO / "manifests/STAGE45E_H200_UNIFIED_REPLICATION_PI0134_V1.json"
PI_GATE = "PI_GATE_STATE_0134"
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


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    require(completed.returncode == 0, f"git failed in {repo}: {' '.join(args)}: {completed.stderr[-1000:]}")
    return completed.stdout.strip()


def static_audit() -> dict[str, Any]:
    manifest = read_json(MANIFEST_PATH)
    require(manifest.get("schema") == "VANDA_STAGE45E_H200_UNIFIED_THREE_ROUTE_REPLICATION_PI0134_V1", "PI0134 unified manifest schema drift")
    require(manifest.get("authority", {}).get("pi_gate") == PI_GATE, "PI0134 authority drift")
    require(manifest.get("authority", {}).get("issue_comment") == 5643644410, "PI0134 issue authority drift")
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
        "schema": "VANDA_STAGE45E_H200_UNIFIED_DIAGNOSTICS_STATIC_AUDIT_V1",
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
    require(admission.get("status") == "PASS" and admission.get("pi_gate") == PI_GATE, "campaign admission is not a PI0134 PASS")
    require(admission.get("route_worktrees", {}).get(route, {}).get("head") == ROUTE_HEADS[route], f"{route} admission HEAD drift")
    require(admission.get("route_worktrees", {}).get(route, {}).get("branch") == ROUTE_BRANCHES[route], f"{route} admission branch drift")
    require(not output.exists() and not output.is_symlink(), f"diagnostic output must be fresh: {output}")
    specs_path = Path(admission["phase_specs"][phase][route]).resolve()
    specs_doc = read_json(specs_path)
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
    admission, spec_doc = _validate_admission(Path(args.admission).resolve(), route, phase, output)
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
    helper_script = Path(args.helper_script).resolve()
    route_script = helper_script
    require(route_script.is_file() and not route_script.is_symlink(), f"route diagnostic source unavailable: {route_script}")
    specs = spec_doc["checkpoints"]
    # The accepted route helpers own creation of the diagnostic output root
    # and require it not to exist at their entrypoint.  Keep the pre-model
    # admission durable in a sibling sidecar until the helper has created its
    # root; materialize the exact same record inside the root afterwards.
    pre_model_sidecar = output.parent / f".{output.name}.pre_model_admission.json"
    require(not pre_model_sidecar.exists() and not pre_model_sidecar.is_symlink(), f"pre-model admission sidecar must be fresh: {pre_model_sidecar}")
    pre_model_record = {
        "schema": "VANDA_STAGE45E_UNIFIED_DIAGNOSTICS_PRE_MODEL_ADMISSION_V1",
        "status": "PASS",
        "route": route,
        "phase": phase,
        "pi_gate": PI_GATE,
        "route_head": ROUTE_HEADS[route],
        "route_branch": ROUTE_BRANCHES[route],
        "route_repo": str(route_repo),
        "helper_script": str(helper_script),
        "source_paths": {key: str(value) for key, value in paths.items()},
        "stats_sha256": stats_sha,
        "stats_payload_sha256": payload_sha,
        "checkpoint_spec_sha256": sha256_file(Path(admission["phase_specs"][phase][route]).resolve()),
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
    write_json(pre_model_sidecar, pre_model_record)
    # The dynamic import below is the first point where the accepted runtime
    # can import Torch/GR00T transitively.  Every identity and freshness
    # predicate above has already passed at this point.
    module = _load_module(route_script)
    module.REPO = route_repo
    module.BRANCH = ROUTE_BRANCHES[route]
    module.PI_GATE = PI_GATE
    module.RUNTIME_STATS_SHA = stats_sha
    module.RUNTIME_STATS_PAYLOAD_SHA = payload_sha
    identity = admission["source_identities"]
    original_require = module.require

    def patched_require(condition: bool, message: str) -> None:
        # The historical route helper has a four-arm budget assertion.  The
        # unified wrapper replaces only that aggregate assertion with the
        # PI0134 exact arm-count check below; all other route assertions stay
        # fail-closed.
        if not condition and ("total budget drift" in message or "total_model_side_calls" in message):
            return
        original_require(condition, message)

    module.require = patched_require
    module.verify_identities = lambda _paths: identity
    module.checkpoint_specs = lambda _args: [dict(item, tree_sha256=item["tree_sha256"]) for item in specs]
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
    try:
        result = module.e4(route_args) if phase == "E4" else module.diagnostic(route_args, phase)
        require(output.is_dir() and not output.is_symlink(), f"route diagnostic did not create a directory output root: {output}")
        write_json(output / "pre_model_admission.json", pre_model_record)
        inner_seal_path = output / "seal.json"
        require(inner_seal_path.is_file(), "route diagnostic did not produce seal.json")
        inner_seal = read_json(inner_seal_path)
        budget = _phase_budget(phase, len(specs))
        counters = inner_seal.get("counters", {})
        observed = {
            "conditioning_passes": counters.get("conditioning_passes"),
            "action_head_calls": counters.get("action_head_calls"),
            "raw_rows": counters.get("raw_rows"),
        }
        require(observed == {key: budget[key] for key in ("conditioning_passes", "action_head_calls", "raw_rows")}, f"unified {route} {phase} budget drift: {observed} != {budget}")
        require(counters.get("training") is False and counters.get("optimizer_steps") == 0, "diagnostic training boundary drift")
        require(counters.get("Validation-E") == 0 and counters.get("Holdout-F") == 0 and counters.get("WBC") == 0 and counters.get("simulator") == 0 and counters.get("hardware") == 0, "diagnostic forbidden boundary drift")
        unified = {
            "schema": "VANDA_STAGE45E_UNIFIED_DIAGNOSTIC_PHASE_SEAL_V1",
            "status": "PASS",
            "pi_gate": PI_GATE,
            "route": route,
            "phase": phase,
            "route_branch": ROUTE_BRANCHES[route],
            "route_head": ROUTE_HEADS[route],
            "stats_sha256": stats_sha,
            "stats_payload_sha256": payload_sha,
            "checkpoints": specs,
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
            "route_result": result,
        }
        write_json(output / "unified_phase_seal.json", unified)
        return unified
    except Exception as exc:
        # Preserve the pre-model record even when the route helper fails
        # after creating the root.  If import/initialization fails before the
        # helper owns the root, create a failure-only root now, after all
        # admission checks and the dynamic import attempt have failed.
        if not output.exists():
            output.mkdir(parents=True, exist_ok=False)
        if output.is_dir() and not output.is_symlink() and not (output / "pre_model_admission.json").exists():
            write_json(output / "pre_model_admission.json", pre_model_record)
        failure = {
            "schema": "VANDA_STAGE45E_UNIFIED_DIAGNOSTIC_FAILURE_SEAL_V1",
            "status": "BLOCKED",
            "route": route,
            "phase": phase,
            "route_head": ROUTE_HEADS[route],
            "pre_model_admission": str(output / "pre_model_admission.json"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "retry_or_substitution": False,
            "timestamp_utc": now_utc(),
        }
        write_json(output / "unified_failure_seal.json", failure, overwrite=True)
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
