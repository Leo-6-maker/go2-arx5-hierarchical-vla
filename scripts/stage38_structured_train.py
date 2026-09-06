#!/usr/bin/env python3
"""Stage38-D structured GR00T training launcher.

The launcher has two explicit modes:

* ``--preflight`` is CPU/read-only with respect to all historical inputs and
  performs the identity, gate, configuration, and no-active-job checks.
* ``--train`` is permitted only inside the one PBS allocation after the same
  checks have been revalidated.  It installs the project-owned structured
  dataset factory and then calls the pinned official ``experiment.run`` path.

The training mode never resumes, never submits a replacement job, and never
opens validation/holdout/WBC/hardware paths.  The output root is new and all
source inputs remain read-only.  The launcher deliberately does not alter the
shared GR00T checkout.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

PI_GATE_ID = "PI_GATE_STATE_0026"
PR_NUMBER = 24
EXPECTED_BRANCH = "stage38/structured-groot-offline"
EXPECTED_PARENT_HEAD = "bbf7453c7060d64fc6f659281ceb777246d6d370"
EXPECTED_PROJECT = "CFP03-CF-004"
EXPECTED_QUEUE = "auto"
EXPECTED_TIMING_MODE = "T16"
EXPECTED_STEPS = 2000
EXPECTED_SAVE_STEPS = 1000
EXPECTED_SEED = 2222
EXPECTED_HORIZON = 40
EXPECTED_UNIVERSAL_DIM = 132
EXPECTED_STAGE38BC_HEAD = "b013738eeb62fbe1fa3cec9d022a81797f4d67cb"
NEXT_GATE = "GPT_PI_REVIEW_AFTER_STAGE38_STRUCTURED_GR00T_OFFLINE"

BASE_ROOT = Path(
    "/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/GR00T-N1.7-3B"
)
BASE_TREE_SHA256 = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
TINY1_ROOT = Path(
    "/scratch/e1192548/stage23_o1_20260903/input_exact/extracted/"
    "tiny1_h2stats_lerobot_v2"
)
TINY1_TREE_SHA256 = "541bf4a1b9ea459637525ef11fceea5c7f192c0902f6013f0331a2a668038157"
GROOT_ROOT = Path(
    "/scratch/e1192548/stage22_h200_runtime_deb920_20260902/src/isaac-gr00t"
)
GROOT_HEAD = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
COSMOS_ROOT = Path(
    "/scratch/e1192548/stage22_h200_cosmos_v2_20260902/hf_home/hub/"
    "models--nvidia--Cosmos-Reason2-2B"
)
COSMOS_TREE_SHA256 = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
H2_ROOT = Path(
    "/scratch/e1192548/stage28_h2_20260904/"
    "stage28_h2_600695.hopper-m-02/h2_training"
)
H2_CONFIG_PATH = H2_ROOT / "experiment_cfg/config.yaml"
H2_CONFIG_SHA256 = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
H2_1000_TREE_SHA256 = "29f3e8f46cf41441696bfc4a5e7601378648c26fa675a9685c003080da0c4114"
H2_2000_TREE_SHA256 = "3106f9b5993b390438b62594d044781d11b393452a49f80b5c8e9e910606d0ba"
H2_SEAL_PATH = H2_ROOT.parent / "stage28_h2_checkpoint_2000_seal.json"
H2_SEAL_SHA256 = "1349896f7b31a930764dbd9d849c5368e48372d0dc6273449f42f19f82a69a82"
STAGE38A_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage38_structured_groot_offline_20260906/stage38a_cpu_c0bb88c_v1"
)
STAGE38A_RESULT_SHA256 = "e08f20ae2ebe9875cf236168655d4075318962ab0ed1fa7b8c25f0e7eab73d49"
STAGE38BC_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage38_structured_groot_offline_20260906/stage38bc_cpu_b013738_v3"
)
STAGE38BC_RESULT_SHA256 = "a7adc641250f52c3b6f95f45fe34f7e8f34943dfe317793b6529d632b678da12"
STAGE38BC_RETURN_SHA256 = "238c85f800b871eb78168b4a16bc13d001a977bdf5dd6598873429c4042adb30"
STAGE38BC_SEAL_SHA256 = "dc45e633daedaa32feec2066e7ae7daaa5446496364a528852e1d6925d9891c0"
TINY1_SNAPSHOT = Path(
    "/scratch/e1192548/stage26_h1_c1_20260904/snapshots/"
    "stage22_expert_red_target_train_a/expert-0000.json"
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    Path(path).write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.check_output(
        ["git", "-C", str(cwd), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def safe_file(path: Path) -> dict[str, Any]:
    path = Path(path)
    record: dict[str, Any] = {
        "path": str(path),
        "exists": False,
        "size_bytes": None,
        "sha256": None,
    }
    if path.is_file() and not path.is_symlink():
        record["exists"] = True
        record["size_bytes"] = int(path.stat().st_size)
        record["sha256"] = sha256_file(path)
    return record


def regular_entries(root: Path) -> list[dict[str, Any]]:
    root = Path(root)
    root_stat = root.lstat()
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise RuntimeError(f"tree root is not a non-symlink directory: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise RuntimeError(f"unsafe directory entry: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise RuntimeError(f"unsafe file entry: {child}")
            entries.append(
                {
                    "path": child.relative_to(root).as_posix(),
                    "bytes": int(child.stat().st_size),
                    "sha256": sha256_file(child),
                }
            )
    return sorted(entries, key=lambda item: str(item["path"]))


def tree_sha256(root: Path, prefix: str) -> tuple[str, list[dict[str, Any]]]:
    entries = regular_entries(root)
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            f"{prefix}/{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode()
        )
    return digest.hexdigest(), entries


def qstat_snapshot() -> dict[str, Any]:
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"
    result = subprocess.run(["qstat", "-u", user], capture_output=True, text=True, check=False)
    lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
    relevant = [
        line
        for line in lines
        if any(token in line.lower() for token in ("stage38", "structured_groot", "wbc"))
    ]
    return {
        "command": f"qstat -u {user}",
        "returncode": int(result.returncode),
        "relevant_lines": relevant,
        "no_active_stage38_or_wbc_job": result.returncode == 0 and not relevant,
        "stdout_sha256": hashlib.sha256((result.stdout or "").encode()).hexdigest(),
        "stderr": (result.stderr or "")[-2000:],
    }


def check_branch(expected_head: str) -> dict[str, Any]:
    local = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    remote = git("ls-remote", "origin", f"refs/heads/{EXPECTED_BRANCH}").split()[0]
    parent = git("rev-parse", "stage37/geometry-capacity-qualification")
    parent_remote = git(
        "ls-remote", "origin", "refs/heads/stage37/geometry-capacity-qualification"
    ).split()[0]
    dirty = git("status", "--porcelain=v1")
    ancestor = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", EXPECTED_PARENT_HEAD, local],
        check=False,
    ).returncode == 0
    return {
        "branch": branch,
        "expected_branch": EXPECTED_BRANCH,
        "expected_head": expected_head,
        "local_head": local,
        "remote_head": remote,
        "parent_local": parent,
        "parent_remote": parent_remote,
        "expected_parent_head": EXPECTED_PARENT_HEAD,
        "worktree_status": dirty,
        "worktree_clean": dirty == "",
        "descends_from_stage37": ancestor,
        "status": "PASS"
        if branch == EXPECTED_BRANCH
        and local == expected_head
        and remote == expected_head
        and parent == EXPECTED_PARENT_HEAD
        and parent_remote == EXPECTED_PARENT_HEAD
        and dirty == ""
        and ancestor
        else "BLOCKED",
    }


def check_stage38bc() -> dict[str, Any]:
    result_path = STAGE38BC_ROOT / "stage38bc_result.json"
    return_path = STAGE38BC_ROOT / "stage38bc_return.json"
    seal_path = STAGE38BC_ROOT / "stage38bc_artifact_seal.json"
    result = read_json(result_path)
    returned = read_json(return_path)
    seal = read_json(seal_path)
    checks = {
        "result_file": safe_file(result_path),
        "return_file": safe_file(return_path),
        "seal_file": safe_file(seal_path),
        "result_sha_match": sha256_file(result_path) == STAGE38BC_RESULT_SHA256,
        "return_sha_match": sha256_file(return_path) == STAGE38BC_RETURN_SHA256,
        "seal_sha_match": sha256_file(seal_path) == STAGE38BC_SEAL_SHA256,
        "result_pass": result.get("result") == "STAGE38BC_CPU_CODEC_PASS_READY_FOR_TRAINING",
        "returned_pass": returned.get("result") == "STAGE38BC_CPU_CODEC_PASS_READY_FOR_TRAINING",
        "historical_inputs_untouched": returned.get("historical_inputs_mutated") is False,
        "scientific_contract_unchanged": returned.get("scientific_contract_changed") is False,
        "model_not_constructed": returned.get("model_constructed") is False,
        "training_not_started": returned.get("training") is False,
        "seal_pass": seal.get("scientific_contract_changed") is False
        and seal.get("training") is False,
        "expected_head": result.get("expected_head") == EXPECTED_STAGE38BC_HEAD,
        "api_head": result.get("api_head") == EXPECTED_STAGE38BC_HEAD,
    }
    checks["status"] = "PASS" if all(
        value for key, value in checks.items() if key not in {"result_file", "return_file", "seal_file"}
    ) else "BLOCKED"
    return checks


def historical_identity_audit() -> dict[str, Any]:
    """Recompute immutable identity bindings using the accepted B/C helper."""

    import stage38_structured_action_codec as stage38bc

    audit = stage38bc.immutable_historical_audit()
    stage38a = stage38bc.stage38a_audit()
    dataset = stage38bc.dataset_contract_audit()
    source = stage38bc.source_semantics_audit()
    return {
        "historical": audit,
        "stage38a": stage38a,
        "dataset": dataset,
        "source": source,
        "status": "PASS"
        if audit.get("status") == "PASS"
        and stage38a.get("status") == "PASS"
        and dataset.get("status") == "PASS"
        and source.get("status") == "PASS"
        else "BLOCKED",
    }


def training_contract_audit() -> dict[str, Any]:
    config_sha = sha256_file(H2_CONFIG_PATH) if H2_CONFIG_PATH.is_file() else None
    payload = {
        "h2_config": safe_file(H2_CONFIG_PATH),
        "h2_config_sha_match": config_sha == H2_CONFIG_SHA256,
        "expected": {
            "steps": EXPECTED_STEPS,
            "save_steps": EXPECTED_SAVE_STEPS,
            "seed": EXPECTED_SEED,
            "horizon": EXPECTED_HORIZON,
            "universal_dim": EXPECTED_UNIVERSAL_DIM,
            "optimizer": "adamw_torch",
            "scheduler": "cosine",
            "learning_rate": 1e-4,
            "warmup_steps": 10,
            "warmup_ratio": 0.0,
            "weight_decay": 1e-5,
            "max_grad_norm": 1.0,
            "global_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "bf16": True,
            "fp16": False,
            "tf32": True,
            "save_only_model": False,
            "resume_from_checkpoint": False,
            "tune_projector": True,
            "tune_diffusion_model": True,
            "tune_visual": False,
            "tune_llm": False,
            "tune_vlln": False,
            "use_relative_action": True,
            "use_percentiles": True,
            "num_gpus": 1,
            "dataloader_num_workers": 0,
            "eval_strategy": "no",
            "validation_e": False,
            "holdout_f": False,
        },
        "status": "PASS" if config_sha == H2_CONFIG_SHA256 else "BLOCKED",
    }
    return payload


def _module_static_audit() -> dict[str, Any]:
    module_path = REPO / "src/go2arx5_vla/stage38_structured_training.py"
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    text = source.lower()
    forbidden_imports = {"isaacgym", "omni", "wandb"}
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    # These tokens are not execution calls here; they are a cheap guard that
    # the project-side data adapter has no accidental science-side path.
    forbidden_tokens = ("qsub", "get_action", "optimizer.step", "scheduler.step", "generate_stats(")
    token_hits = [token for token in forbidden_tokens if token in text]
    return {
        "path": str(module_path),
        "sha256": sha256_file(module_path),
        "forbidden_imports": sorted(imports & forbidden_imports),
        "forbidden_training_side_calls": token_hits,
        "status": "PASS" if not imports & forbidden_imports and not token_hits else "BLOCKED",
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        existing = sorted(p.name for p in output_root.iterdir())
        allowed = {"stage38d_preflight.json"}
        if set(existing) - allowed:
            raise RuntimeError(
                "Stage38-D output root is not fresh; only its preflight report may exist: "
                f"{output_root} -> {existing}"
            )
    else:
        output_root.mkdir(parents=True, exist_ok=False)

    branch = check_branch(args.expected_project_sha)
    stage38bc = check_stage38bc()
    identity = historical_identity_audit()
    contract = training_contract_audit()
    static = _module_static_audit()
    qstat = qstat_snapshot()
    usage = shutil.disk_usage(output_root)
    scratch = {
        "path": str(output_root),
        "free_bytes": int(usage.free),
        "total_bytes": int(usage.total),
        "minimum_free_bytes": int(args.minimum_free_gib * 1024**3),
        "sufficient_free_space": usage.free >= args.minimum_free_gib * 1024**3,
    }
    paths = {
        "base": {"path": str(BASE_ROOT), "tree_sha256": identity["historical"].get("base", {}).get("observed_tree_sha256")},
        "tiny1": {"path": str(TINY1_ROOT), "tree_sha256": identity["historical"].get("tiny1", {}).get("tree_sha256")},
        "groot": {"path": str(GROOT_ROOT), "head": GROOT_HEAD},
        "cosmos": {"path": str(COSMOS_ROOT), "tree_sha256": identity["historical"].get("cosmos", {}).get("tree_sha256")},
        "h2": {"path": str(H2_ROOT), "checkpoint_1000_tree_sha256": H2_1000_TREE_SHA256, "checkpoint_2000_tree_sha256": H2_2000_TREE_SHA256, "seal_sha256": H2_SEAL_SHA256},
        "stage38bc": {"root": str(STAGE38BC_ROOT), "head": EXPECTED_STAGE38BC_HEAD, "seal_sha256": STAGE38BC_SEAL_SHA256},
    }
    all_pass = all(
        item.get("status") == "PASS"
        for item in (branch, stage38bc, identity, contract, static)
    ) and qstat.get("no_active_stage38_or_wbc_job") is True and scratch["sufficient_free_space"]
    payload = {
        "schema": "STAGE38D_STRUCTURED_GR00T_PREFLIGHT_V1",
        "pi_gate": PI_GATE_ID,
        "pr": PR_NUMBER,
        "branch_name": EXPECTED_BRANCH,
        "expected_project": EXPECTED_PROJECT,
        "expected_queue": EXPECTED_QUEUE,
        "expected_project_head": args.expected_project_sha,
        "stage38bc_execution_head": EXPECTED_STAGE38BC_HEAD,
        "timing_model_capacity_star": EXPECTED_TIMING_MODE,
        "structured_codec": "STRUCTURED_ACTION_V1_QUALIFIED",
        "model_construction_authorized_after_this_report": bool(all_pass),
        "paths": paths,
        "branch": branch,
        "stage38bc": stage38bc,
        "identity": identity,
        "training_contract": contract,
        "module_static": static,
        "qstat": qstat,
        "scratch": scratch,
        "validation_e_read": False,
        "holdout_f_read": False,
        "wbc": False,
        "hardware_control": False,
        "source_dataset_mutation_allowed": False,
        "historical_artifact_mutation_allowed": False,
        "pbs_submitted": False,
        "gpu_used": False,
        "model_constructed": False,
        "forward": False,
        "training": False,
        "scientific_contract_changed": False,
        "status": "PASS" if all_pass else "BLOCKED",
        "next_gate": "STAGE38_D_TRAINING_READY" if all_pass else "GPT_PI_REVIEW_AFTER_STAGE38_STRUCTURED_GR00T_OFFLINE",
        "timestamp_utc": now_utc(),
    }
    write_json(output_root / "stage38d_preflight.json", payload)
    return payload


def _load_training_config(output_root: Path, args: argparse.Namespace) -> Any:
    # This import is deliberately after all identity checks and GPU/PBS checks.
    from gr00t.configs.base_config import get_default_config

    config = get_default_config().load(H2_CONFIG_PATH)
    config.load_config_path = None
    config.training.output_dir = str(output_root)
    config.training.experiment_name = None
    config.training.start_from_checkpoint = str(args.base_model_path)
    config.training.resume_from_checkpoint = False
    config.training.save_only_model = False
    config.training.transformers_local_files_only = True
    config.training.transformers_cache_dir = None
    config.training.use_wandb = False
    config.data.seed = EXPECTED_SEED
    config.data.datasets[0].dataset_paths = [str(args.dataset_dir)]
    config.data.datasets[0].mix_ratio = 1.0
    return config


def _assert_loaded_config(config: Any, args: argparse.Namespace) -> None:
    checks = {
        "base": config.training.start_from_checkpoint == str(args.base_model_path),
        "dataset": len(config.data.datasets) == 1
        and config.data.datasets[0].dataset_paths == [str(args.dataset_dir)],
        "steps": config.training.max_steps == EXPECTED_STEPS,
        "save_steps": config.training.save_steps == EXPECTED_SAVE_STEPS,
        "seed": config.data.seed == EXPECTED_SEED,
        "optimizer": config.training.optim == "adamw_torch",
        "scheduler": config.training.lr_scheduler_type == "cosine",
        "learning_rate": float(config.training.learning_rate) == 1e-4,
        "warmup_steps": config.training.warmup_steps == 10,
        "warmup_ratio": float(config.training.warmup_ratio) == 0.0,
        "weight_decay": float(config.training.weight_decay) == 1e-5,
        "max_grad_norm": float(config.training.max_grad_norm) == 1.0,
        "batch": config.training.global_batch_size == 1
        and config.training.gradient_accumulation_steps == 1,
        "precision": config.training.bf16 is True
        and config.training.fp16 is False
        and config.training.tf32 is True,
        "fresh": config.training.save_only_model is False
        and config.training.resume_from_checkpoint is False,
        "model_shape": config.model.action_horizon == EXPECTED_HORIZON
        and config.model.max_action_dim == EXPECTED_UNIVERSAL_DIM
        and config.model.max_state_dim == EXPECTED_UNIVERSAL_DIM,
        "model_flags": config.model.tune_projector is True
        and config.model.tune_diffusion_model is True
        and config.model.tune_visual is False
        and config.model.tune_llm is False
        and config.model.tune_vlln is False,
        "action_semantics": config.model.use_relative_action is True
        and config.model.use_percentiles is True,
        "no_eval": config.training.eval_strategy == "no",
        "workers": config.training.num_gpus == 1 and config.training.dataloader_num_workers == 0,
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise RuntimeError(f"Stage38-D loaded H2 contract mismatch: {failed}")


def _gpu_pbs_audit() -> dict[str, Any]:
    job_id = os.environ.get("PBS_JOBID")
    if not job_id:
        raise RuntimeError("--train is permitted only inside a PBS allocation")
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    names = [line.strip() for line in (query.stdout or "").splitlines() if line.strip()]
    if query.returncode != 0 or len(names) != 1 or "H200" not in names[0]:
        raise RuntimeError(f"Stage38-D requires exactly one H200; observed {names!r}")
    return {
        "pbs_jobid": job_id,
        "pbs_queue": os.environ.get("PBS_QUEUE"),
        "gpu_names": names,
        "one_h200": True,
    }


def _checkpoint_record(path: Path, expected_step: int) -> dict[str, Any]:
    state_path = path / "trainer_state.json"
    state = read_json(state_path)
    required = ["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"]
    missing = [name for name in required if not (path / name).is_file()]
    tree, entries = tree_sha256(path, f"stage38d/{path.name}")
    return {
        "path": str(path),
        "exists": path.is_dir(),
        "global_step": state.get("global_step"),
        "expected_step": expected_step,
        "required_state_files_present": not missing,
        "missing_state_files": missing,
        "tree_sha256": tree,
        "file_count": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "status": "PASS"
        if state.get("global_step") == expected_step and not missing
        else "BLOCKED",
    }


def _artifact_listing(root: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            entries.append(
                {
                    "path": str(path.relative_to(root)),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
    return entries


def train(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve()
    preflight_path = output_root / "stage38d_preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("Stage38-D training requires a prior passing preflight")
    preflight_report = read_json(preflight_path)
    if preflight_report.get("status") != "PASS":
        raise RuntimeError("Stage38-D preflight is not PASS")
    if preflight_report.get("expected_project_head") != args.expected_project_sha:
        raise RuntimeError("Stage38-D preflight HEAD binding changed")

    branch = check_branch(args.expected_project_sha)
    stage38bc = check_stage38bc()
    identity_before = historical_identity_audit()
    contract = training_contract_audit()
    gpu = _gpu_pbs_audit()
    if branch.get("status") != "PASS" or stage38bc.get("status") != "PASS" or identity_before.get("status") != "PASS" or contract.get("status") != "PASS":
        raise RuntimeError("Stage38-D revalidation failed before model construction")

    runtime_root = output_root / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=False)
    events_path = output_root / "stage38d_events.jsonl"
    events_path.write_text(
        json.dumps({"event": "MODEL_CONSTRUCTION_ALLOWED", "timestamp_utc": now_utc(), "pbs": gpu}) + "\n",
        encoding="utf-8",
    )
    write_json(
        output_root / "stage38d_revalidation.json",
        {
            "schema": "STAGE38D_REVALIDATION_V1",
            "branch": branch,
            "stage38bc": stage38bc,
            "identity_before": identity_before,
            "training_contract": contract,
            "gpu_pbs": gpu,
            "model_constructed": False,
            "forward": False,
            "training": False,
            "pbs_submitted": True,
            "scientific_contract_changed": False,
            "timestamp_utc": now_utc(),
        },
    )

    # Keep every transient/cache location on scratch.  /tmp is full on this
    # server and is not part of the experiment's provenance surface.
    cache_root = output_root / "cache"
    tmp_root = output_root / "tmp"
    cache_root.mkdir(parents=True, exist_ok=False)
    tmp_root.mkdir(parents=True, exist_ok=False)
    os.environ["TMPDIR"] = str(tmp_root)
    os.environ["TMP"] = str(tmp_root)
    os.environ["TEMP"] = str(tmp_root)
    os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "huggingface" / "hub")
    os.environ["TORCH_HOME"] = str(cache_root / "torch")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["GROOT_COMMIT_HASH"] = GROOT_HEAD
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(args.groot_source), os.environ.get("PYTHONPATH", "")]
    )
    sys.path.insert(0, str(args.groot_source))

    # All imports below can construct the model or access the GPU, and are
    # therefore intentionally after the final immutable preflight.
    from go2arx5_vla.stage38_structured_training import install_structured_dataset_factory

    install_structured_dataset_factory()
    from gr00t.experiment.experiment import run

    config = _load_training_config(output_root, args)
    _assert_loaded_config(config, args)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "TRAINING_START", "timestamp_utc": now_utc()}) + "\n")
    run(config)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "TRAINING_RETURNED", "timestamp_utc": now_utc()}) + "\n")

    checkpoint_1000 = _checkpoint_record(output_root / "checkpoint-1000", 1000)
    checkpoint_2000 = _checkpoint_record(output_root / "checkpoint-2000", 2000)
    structured_stats = safe_file(output_root / "experiment_cfg/structured_action_statistics.json")
    identity_after = historical_identity_audit()
    tiny1_unchanged = identity_after.get("historical", {}).get("tiny1", {}).get("status") == "PASS"
    result = {
        "schema": "STAGE38D_STRUCTURED_GR00T_TRAINING_RESULT_V1",
        "pi_gate": PI_GATE_ID,
        "pr": PR_NUMBER,
        "branch": EXPECTED_BRANCH,
        "execution_head": args.expected_project_sha,
        "pbs_jobid": gpu["pbs_jobid"],
        "result": "STAGE38D_STRUCTURED_TRAINING_COMPLETE"
        if checkpoint_1000["status"] == "PASS" and checkpoint_2000["status"] == "PASS" and tiny1_unchanged
        else "STAGE38D_STRUCTURED_TRAINING_INCOMPLETE",
        "timing_model_capacity_star": EXPECTED_TIMING_MODE,
        "structured_codec": "STRUCTURED_ACTION_V1_QUALIFIED",
        "structured_checkpoint_1000": checkpoint_1000,
        "structured_checkpoint_2000": checkpoint_2000,
        "structured_action_statistics": structured_stats,
        "checkpoint_selection": {
            "primary": "STRUCTURED@1000",
            "secondary": "STRUCTURED@2000",
            "post_hoc_selection": False,
        },
        "model_constructed": True,
        "forward": True,
        "training": True,
        "pbs_submitted": True,
        "gpu_used": True,
        "validation_e_read": False,
        "holdout_f_read": False,
        "wbc": False,
        "hardware_control": False,
        "historical_inputs_mutated": not tiny1_unchanged,
        "scientific_contract_changed": False,
        "next_gate": NEXT_GATE,
        "identity_after": identity_after,
        "timestamp_utc": now_utc(),
    }
    write_json(output_root / "stage38d_result.json", result)
    return 0 if result["result"] == "STAGE38D_STRUCTURED_TRAINING_COMPLETE" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--train", action="store_true")
    parser.add_argument("--expected-project-sha", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, default=BASE_ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=TINY1_ROOT)
    parser.add_argument("--groot-source", type=Path, default=GROOT_ROOT)
    parser.add_argument("--minimum-free-gib", type=float, default=100.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.expected_project_sha == EXPECTED_STAGE38BC_HEAD:
            raise RuntimeError(
                "Stage38-D execution head must be a committed descendant of the accepted B/C head; "
                "the B/C evidence head itself is not a training wrapper head"
            )
        if args.preflight:
            report = preflight(args)
            print(json.dumps({"status": report["status"], "output_root": args.output_root}, sort_keys=True))
            return 0 if report["status"] == "PASS" else 2
        return train(args)
    except Exception as exc:
        output_root = Path(args.output_root).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        write_json(
            output_root / "stage38d_failure.json",
            {
                "schema": "STAGE38D_FAILURE_V1",
                "pi_gate": PI_GATE_ID,
                "expected_project_head": args.expected_project_sha,
                "mode": "preflight" if args.preflight else "train",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "pbs_jobid": os.environ.get("PBS_JOBID"),
                "model_construction_status": "UNKNOWN_AFTER_GATE" if args.train else "NOT_ATTEMPTED",
                "forward": False if not args.train else "UNKNOWN",
                "training": False if not args.train else "UNKNOWN",
                "pbs_submitted": bool(os.environ.get("PBS_JOBID")),
                "scientific_contract_changed": False,
                "timestamp_utc": now_utc(),
            },
        )
        print(f"STAGE38D_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
