#!/usr/bin/env python3
"""Stage45E matched Legacy FULL24 seed replication runner.

The static path is dependency-light and fail-closed.  Runtime modes are
explicitly invoked on the authorized RTX 4080S and use only the accepted
legacy FULL24 route (no N2 statistics override).  Training and diagnostics
write raw-first, hash-bound evidence; no Validation-E/Holdout-F/WBC/sim or
hardware path exists in this runner.
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
SCRIPTS = REPO / "scripts"
SRC = REPO / "src"
for item in (SCRIPTS, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

MANIFEST_PATH = REPO / "manifests/STAGE45E_LEGACY_MATCHED_SEED_REPLICATION_V1.json"
STAGE = "Stage45E"
BRANCH = "stage45e/legacy-full24-matched-seed-replication"
PI_GATE = "PI_GATE_STATE_0131"
PI_COMMENT = 5642616832
BASE_HEAD = "d762e731d2c11ceea3c001c37ac27890fe6f4e70"
BASE_TREE = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
FULL24_TREE = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
RUNTIME_STATS_SHA = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
RUNTIME_STATS_PAYLOAD_SHA = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
MODALITY_SHA = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
CONFIG_SHA = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
GPU_UUID = "GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad"
HORIZON = 40
UNIVERSAL_DIM = 132
RAW_SHAPE = (1, HORIZON, UNIVERSAL_DIM)
T16 = (0, 250, 500, 750)
DT = 0.25
WINDOW_IDS = tuple(
    f"stage22_expert_{target}:expert-{index:04d}"
    for target in ("red_target_train_a", "red_target_train_b", "red_target_train_c", "red_target_train_d")
    for index in range(6)
)
TARGETS = ("red_target_train_a", "red_target_train_b", "red_target_train_c", "red_target_train_d")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def read_json(path: Path) -> Any:
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"JSON is not a regular file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path = Path(path)
    require(overwrite or not path.exists(), f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True, stderr=subprocess.STDOUT).strip()


def tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    mode = root.lstat().st_mode if root.exists() else 0
    require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"unsafe tree root: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in dirnames:
            child = directory_path / dirname
            require(stat.S_ISDIR(child.lstat().st_mode) and not child.is_symlink(), f"unsafe tree directory: {child}")
        for filename in filenames:
            child = directory_path / filename
            require(stat.S_ISREG(child.lstat().st_mode) and not child.is_symlink(), f"unsafe tree file: {child}")
            entries.append({"path": f"{prefix}/{child.relative_to(root).as_posix()}", "bytes": int(child.stat().st_size), "sha256": sha256_file(child)})
    return sorted(entries, key=lambda item: str(item["path"]))


def tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode())
    return digest.hexdigest()


def manifest() -> dict[str, Any]:
    value = read_json(MANIFEST_PATH)
    require(value.get("schema") == "CORELAB_STAGE45E_LEGACY_MATCHED_SEED_REPLICATION_V1", "manifest schema drift")
    require(value.get("authority", {}).get("pi_gate") == PI_GATE and value["authority"].get("issue_comment") == PI_COMMENT, "PI0131 authority drift")
    require(value.get("repository", {}).get("branch") == BRANCH and value["repository"].get("base_head") == BASE_HEAD, "branch/base identity drift")
    return value


def static_audit() -> dict[str, Any]:
    value = manifest()
    require(value["training"]["seeds"] == [2223, 2224] and value["training"]["max_steps"] == 2000, "training budget drift")
    require(value["evaluation"]["conditioning_passes"] == 96 and value["evaluation"]["action_head_calls"] == 384, "E4 budget drift")
    require(value["diagnostics"]["c3_d1"]["total_model_side_calls"] == 960, "D1 budget drift")
    require(value["diagnostics"]["c4_d2a"]["total_model_side_calls"] == 864, "D2A budget drift")
    require(value["legacy_route"]["n2_statistics_forbidden"] is True, "N2 route must be forbidden")
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    top_imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.append(node.module.split(".", 1)[0])
    forbidden = sorted(set(top_imports) & {"torch", "gr00t", "transformers", "isaacgym", "omni", "pandas"})
    require("STAGE45E_N2_STATS_ARTIFACT" in source, "negative N2 override guard missing")
    return {
        "schema": "CORELAB_STAGE45E_LEGACY_MATCHED_SEED_REPLICATION_STATIC_AUDIT_V1",
        "status": "PASS" if not forbidden else "BLOCKED",
        "branch": git("branch", "--show-current"),
        "repository_head": git("rev-parse", "HEAD"),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "top_level_forbidden_imports": forbidden,
        "legacy_stats_route": "FULL24_RUNTIME_V2",
        "n2_override_present_at_static": False,
        "training_seeds": [2223, 2224],
        "training_steps": 2000,
        "e4_calls": 480,
        "d1_calls": 960,
        "d2a_calls": 864,
        "execution_boundary": {"gpu": False, "model_construction": 0, "gr00t_forwards": 0, "training_steps": 0, "optimizer_steps": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0, "PBS": 0},
    }


def paths(args: argparse.Namespace) -> dict[str, Path]:
    names = ("base_model", "dataset", "groot", "cosmos", "modality", "config", "runtime_stats", "sealed_baseline", "snapshot_preflight")
    result = {name: Path(getattr(args, name)).resolve() for name in names if getattr(args, name, None) is not None}
    return result


def gpu_identity(expected: str = GPU_UUID) -> dict[str, Any]:
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    require(query.returncode == 0, f"nvidia-smi failed: {query.stderr[-1000:]}")
    rows = []
    for line in query.stdout.splitlines():
        if line.strip():
            fields = [part.strip() for part in line.split(",")]
            require(len(fields) == 5, f"unexpected nvidia-smi row: {line}")
            rows.append({"index": fields[0], "name": fields[1], "uuid": fields[2], "memory_total_mib": int(fields[3]), "memory_used_mib_at_start": int(fields[4])})
    selected = [row for row in rows if row["uuid"] == expected]
    require(len(selected) == 1, f"requested GPU UUID not unique: {expected}; observed={rows}")
    gpu = selected[0]
    require("RTX 4080 SUPER" in gpu["name"] and gpu["memory_total_mib"] >= 30000, f"wrong Corelab GPU: {gpu}")
    gpu["host"] = subprocess.check_output(["hostname"], text=True).strip()
    gpu["query_sha256"] = sha256_bytes(query.stdout.encode())
    return gpu


def verify_identities(p: Mapping[str, Path]) -> dict[str, Any]:
    import stage45e_n2_statistics as n2
    base = tree_digest(tree_entries(p["base_model"], "payload/GR00T-N1.7-3B"))
    require(base == BASE_TREE, f"BASE tree mismatch: {base}")
    full24 = n2.validate_dataset_identity(p["dataset"])
    require(full24["tree_sha256"] == FULL24_TREE, f"FULL24 tree mismatch: {full24}")
    runtime = read_json(p["runtime_stats"])
    runtime_check = n2.validate_source_document(runtime, source_path=p["runtime_stats"])
    require(runtime_check["outer_sha256"] == RUNTIME_STATS_SHA and runtime_check["payload_sha256"] == RUNTIME_STATS_PAYLOAD_SHA, "runtime-V2 stats identity mismatch")
    require(sha256_file(p["modality"]) == MODALITY_SHA, "modality identity mismatch")
    require(sha256_file(p["config"]) == CONFIG_SHA, "config identity mismatch")
    require(git_external(p["groot"], "rev-parse", "HEAD") == GROOT_COMMIT, "GR00T source commit mismatch")
    refs = p["cosmos"] / "refs/main"
    require(refs.is_file() and refs.read_text(encoding="utf-8").strip() == COSMOS_REVISION, "Cosmos revision mismatch")
    # The tree hash is expensive but is the accepted immutable Cosmos identity.
    cosmos_tree = cosmos_digest(p["cosmos"])
    require(cosmos_tree == COSMOS_TREE, f"Cosmos tree mismatch: {cosmos_tree}")
    return {"base_tree_sha256": base, "full24": full24, "runtime_stats": runtime_check, "runtime_stats_sha256": sha256_file(p["runtime_stats"]), "runtime_stats_payload_sha256": RUNTIME_STATS_PAYLOAD_SHA, "modality_sha256": sha256_file(p["modality"]), "config_sha256": sha256_file(p["config"]), "groot_commit": GROOT_COMMIT, "cosmos_revision": COSMOS_REVISION, "cosmos_tree_sha256": cosmos_tree}


def git_external(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT).strip()


def cosmos_digest(root: Path) -> str:
    entries: list[dict[str, Any]] = []
    root = Path(root).resolve()
    require(root.is_dir() and not root.is_symlink(), f"Cosmos root unavailable: {root}")
    def visit(directory: Path, relative: Path) -> None:
        for item in sorted(directory.iterdir(), key=lambda value: value.name):
            rel = (relative / item.name).as_posix()
            mode = item.lstat().st_mode
            if stat.S_ISDIR(mode):
                entries.append({"path": rel, "kind": "directory", "size_bytes": 0, "sha256": None, "target": None, "resolved_path": None})
                visit(item, relative / item.name)
            elif stat.S_ISLNK(mode):
                resolved = item.resolve(strict=True)
                require(resolved.is_file() and resolved.is_relative_to(root), f"Cosmos symlink escapes root: {item}")
                entries.append({"path": rel, "kind": "symlink", "size_bytes": int(resolved.stat().st_size), "sha256": sha256_file(resolved), "target": os.readlink(item), "resolved_path": resolved.relative_to(root).as_posix()})
            elif stat.S_ISREG(mode):
                entries.append({"path": rel, "kind": "regular_file", "size_bytes": int(item.stat().st_size), "sha256": sha256_file(item), "target": None, "resolved_path": None})
            else:
                raise RuntimeError(f"unsupported Cosmos entry: {item}")
    visit(root, Path("."))
    lines = []
    for entry in sorted(entries, key=lambda value: str(value["path"])):
        lines.append("\t".join((str(entry["kind"]), str(entry["path"]), str(entry.get("size_bytes", 0)), str(entry.get("sha256") or "-"), str(entry.get("target") or "-"), str(entry.get("resolved_path") or "-"))))
    return sha256_bytes(("\n".join(lines) + "\n").encode())


def prepare_env(p: Mapping[str, Path], output: Path, gpu: Mapping[str, Any]) -> None:
    cache = output / "cache"
    tmp = output / "tmp"
    cache.mkdir(parents=True, exist_ok=False)
    tmp.mkdir(parents=True, exist_ok=False)
    hub = cache / "huggingface/hub"
    hub.mkdir(parents=True, exist_ok=False)
    (cache / "xdg").mkdir(parents=True, exist_ok=False)
    (hub / "models--nvidia--Cosmos-Reason2-2B").symlink_to(p["cosmos"], target_is_directory=True)
    # Explicitly remove any inherited N2 route.  A stale override is a hard
    # failure, never a convenience fallback.
    os.environ.pop("STAGE45E_N2_STATS_ARTIFACT", None)
    require("STAGE45E_N2_STATS_ARTIFACT" not in os.environ, "N2 statistics override leaked into legacy runtime")
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(gpu["index"]), "TMPDIR": str(tmp), "TMP": str(tmp), "TEMP": str(tmp), "XDG_CACHE_HOME": str(cache / "xdg"), "HF_HOME": str(cache / "huggingface"), "TRANSFORMERS_CACHE": str(hub), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "STAGE45B_LABEL_ANCHOR_RECONCILIATION": "1"})
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def checkpoint_record(path: Path, step: int) -> dict[str, Any]:
    require(path.is_dir() and not path.is_symlink(), f"checkpoint unavailable: {path}")
    state = read_json(path / "trainer_state.json")
    required = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
    missing = [item for item in required if not (path / item).is_file() or (path / item).is_symlink()]
    entries = tree_entries(path, f"stage45e_legacy/{path.name}")
    return {"path": str(path), "global_step": state.get("global_step"), "expected_step": step, "required_state_files_present": not missing, "missing_state_files": missing, "tree_sha256": tree_digest(entries), "file_count": len(entries), "total_bytes": sum(item["bytes"] for item in entries), "status": "PASS" if state.get("global_step") == step and not missing else "BLOCKED"}


def compare_stats(left: Any, right: Any, atol: float = 1e-12) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(compare_stats(left[key], right[key], atol) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(compare_stats(a, b, atol) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        return abs(float(left) - float(right)) <= atol
    return left == right


def canary(args: argparse.Namespace) -> dict[str, Any]:
    p = paths(args)
    identity = verify_identities(p)
    gpu = gpu_identity(args.gpu_uuid or GPU_UUID)
    output = Path(args.output_root).resolve()
    require(not output.exists(), f"canary output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    prepare_env(p, output, gpu)
    # Reuse only the accepted pipeline construction and canary mechanics; the
    # N2 override is absent, so the factory computes the legacy route.
    import stage45e_n2_corelab as core
    core_paths = {"base_model": p["base_model"], "dataset": p["dataset"], "groot": p["groot"], "cosmos": p["cosmos"], "config": p["config"]}
    pipeline, config, model, dataset, collator = core._load_pipeline(core_paths, output)
    import torch
    core._BOUNDARY.clear()
    core._BOUNDARY.update({"gpu": True, "model_constructed": 1, "gr00t_forwards": 0, "training_steps": 0, "backward_steps": 0, "optimizer_steps": 0, "checkpoint_writes": 0})
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    inputs = core._sample_batch(dataset, collator)
    if isinstance(inputs, Mapping):
        inputs = dict(inputs)
        inputs.pop("action", None)
        inputs.pop("action_mask", None)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        action = model.get_action(inputs)["action_pred"]
    require(tuple(action.shape) == RAW_SHAPE and bool(torch.isfinite(action).all().item()), "legacy inference canary action contract failed")
    model.train()
    batch = core._sample_batch(dataset, collator)
    before = core._parameter_sample(model)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(batch)
        loss = outputs["loss"] if isinstance(outputs, Mapping) else outputs.loss
        require(torch.is_tensor(loss) and loss.ndim == 0 and bool(torch.isfinite(loss).item()), "legacy canary loss is not finite")
        loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    require(all(bool(torch.isfinite(g).all().item()) for g in gradients), "legacy canary gradient nonfinite")
    delta = core._parameter_delta(before, model)
    require(delta["weight_update"] == 0, "legacy canary changed weights without optimizer step")
    result = {"schema": "CORELAB_STAGE45E_LEGACY_RESOURCE_CANARY_V1", "status": "PASS", "pi_gate": PI_GATE, "branch": BRANCH, "identity": identity, "gpu": gpu, "inference_forward_calls": 1, "training_batch_forward_calls": 1, "backward_steps": 1, "optimizer_steps": 0, "scheduler_steps": 0, "checkpoint_writes": 0, "weight_update": 0, "parameter_sample_max_abs_delta": delta["parameter_sample_max_abs_delta"], "loss": float(loss.detach().float().cpu().item()), "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), "execution_boundary": {"gpu": True, "model_constructed": 1, "gr00t_forwards": 2, "training_steps": 0, "backward_steps": 1, "optimizer_steps": 0, "checkpoint_writes": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0, "PBS": 0}}
    write_json(output / "legacy_resource_canary.json", result)
    return result


def train(args: argparse.Namespace) -> dict[str, Any]:
    require(args.seed in (2223, 2224), "only PI0131 seeds 2223/2224 are allowed")
    p = paths(args)
    identity = verify_identities(p)
    gpu = gpu_identity(args.gpu_uuid or GPU_UUID)
    output = Path(args.output_root).resolve()
    require(not output.exists(), f"training output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    require(shutil.disk_usage(output).free >= 64 * 1024**3, "insufficient free storage for full-state checkpoints")
    canary_doc = read_json(Path(args.canary)) if args.canary else None
    if canary_doc is not None:
        require(canary_doc.get("status") == "PASS" and canary_doc.get("weight_update") == 0 and canary_doc.get("optimizer_steps") == 0, "canary is not a zero-update PASS")
    write_json(output / "admission.json", {"schema": "CORELAB_STAGE45E_LEGACY_TRAINING_ADMISSION_V1", "status": "PASS", "pi_gate": PI_GATE, "branch": BRANCH, "repository_head": git("rev-parse", "HEAD"), "seed": args.seed, "starting_checkpoint": "exact BASE", "identity": identity, "gpu": gpu, "n2_statistics_override": False, "training_recipe": manifest()["training"], "execution_boundary": {"gpu": False, "model_construction": 0, "gr00t_forwards": 0, "training_steps": 0, "optimizer_steps": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0, "PBS": 0}})
    prepare_env(p, output, gpu)
    try:
        sys.path.insert(0, str(p["groot"]))
        from stage45b_full24_fresh_base_train import _bind_cosmos_processor, _load_config
        _bind_cosmos_processor(p["cosmos"] / "snapshots" / COSMOS_REVISION)
        from go2arx5_vla.stage38_structured_training import install_structured_dataset_factory
        install_structured_dataset_factory()
        from gr00t.experiment.experiment import run as gr00t_run
        config = _load_config(p["config"], output_root=output, base_model=p["base_model"], dataset=p["dataset"])
        config.data.seed = int(args.seed)
        require(int(config.data.seed) == args.seed, "seed override failed")
        binding = {"schema": "CORELAB_STAGE45E_LEGACY_TRAINING_BINDING_V1", "pi_gate": PI_GATE, "branch": BRANCH, "seed": args.seed, "starting_checkpoint": "BASE", "base_tree_sha256": BASE_TREE, "full24_tree_sha256": FULL24_TREE, "runtime_stats_sha256": RUNTIME_STATS_SHA, "runtime_stats_payload_sha256": RUNTIME_STATS_PAYLOAD_SHA, "n2_statistics_override": False, "recipe": manifest()["training"], "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}
        write_json(output / "training_binding.json", binding)
        gr00t_run(config)
        cp1000 = checkpoint_record(output / "checkpoint-1000", 1000)
        cp2000 = checkpoint_record(output / "checkpoint-2000", 2000)
        sidecar_path = output / "experiment_cfg/structured_action_statistics.json"
        sidecar = read_json(sidecar_path)
        require(sidecar.get("statistics_override") is None, "legacy sidecar unexpectedly carries a statistics override")
        require(canonical_sha(sidecar.get("statistics")) == RUNTIME_STATS_PAYLOAD_SHA, "legacy computed statistics do not match runtime-V2 payload")
        result = {"schema": "CORELAB_STAGE45E_LEGACY_FULL24_MATCHED_SEED_TRAINING_RESULT_V1", "status": "PASS" if cp1000["status"] == "PASS" and cp2000["status"] == "PASS" else "BLOCKED", "return_token": "STAGE45E_LEGACY_MATCHED_SEED_REPLICATION_COMPLETE_FOR_PI_REVIEW", "pi_gate": PI_GATE, "branch": BRANCH, "repository_head": git("rev-parse", "HEAD"), "seed": args.seed, "starting_checkpoint": "BASE", "base_tree_sha256": BASE_TREE, "full24_tree_sha256": FULL24_TREE, "runtime_stats_sha256": RUNTIME_STATS_SHA, "runtime_stats_payload_sha256": RUNTIME_STATS_PAYLOAD_SHA, "checkpoint_1000": cp1000, "checkpoint_2000": cp2000, "training_steps": 2000, "optimizer_steps": 2000, "backward_steps": 2000, "model_constructed": True, "n2_statistics_override": False, "statistics_sidecar_sha256": sha256_file(sidecar_path), "validation_e_read": False, "holdout_f_read": False, "wbc_steps": 0, "simulator_steps": 0, "hardware": False, "pbs_submissions": 0, "gpu": gpu, "timestamp_utc": now_utc()}
        write_json(output / "training_result.json", result)
        return result
    except Exception as exc:
        write_json(output / "FAILURE_SEAL.json", {"schema": "CORELAB_STAGE45E_LEGACY_TRAINING_FAILURE_SEAL_V1", "status": "BLOCKED", "seed": args.seed, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "retry_or_substitution": False, "n2_statistics_override": False, "timestamp_utc": now_utc()}, overwrite=True)
        raise


def load_context(p: Mapping[str, Path]) -> tuple[Any, Any, list[dict[str, Any]], Mapping[str, Any], list[dict[str, Any]], list[Any]]:
    import torch
    import stage45e_n2_train24_eval as ev
    from stage45b_full24_fresh_base_train import _bind_cosmos_processor
    _bind_cosmos_processor(p["cosmos"] / "snapshots" / COSMOS_REVISION)
    preflight = read_json(p["snapshot_preflight"])
    records = ev._snapshot_lookup(preflight["snapshots"])
    sealed = ev._verify_sealed_baseline(p["sealed_baseline"], p["snapshot_preflight"])
    processor, embodiment, modality_configs = ev._load_runtime_processor({"groot": p["groot"], "base_model": p["base_model"], "dataset": p["dataset"], "modality": p["modality"], "cosmos": p["cosmos"]})
    windows = ev._make_windows(processor, embodiment, modality_configs, p["dataset"], records)
    require(len(records) == 24 and len(windows) == 24, "TRAIN24 materialization drift")
    noise_bank = ev._make_noise_bank(torch, sealed["noise_manifest"])
    return torch, ev, records, sealed["noise_manifest"], windows, noise_bank


def checkpoint_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    roots = {2223: Path(args.seed2223_root).resolve(), 2224: Path(args.seed2224_root).resolve()}
    specs = []
    for seed in (2223, 2224):
        root = roots[seed]
        result = read_json(root / "training_result.json")
        require(result.get("status") == "PASS" and result.get("seed") == seed and result.get("starting_checkpoint") == "BASE", f"legacy seed {seed} training result is not PASS")
        require(result.get("runtime_stats_sha256") == RUNTIME_STATS_SHA and result.get("n2_statistics_override") is False, f"legacy seed {seed} stats route drift")
        for step in (1000, 2000):
            cp = root / f"checkpoint-{step}"
            rec = checkpoint_record(cp, step)
            require(rec["status"] == "PASS", f"legacy seed {seed} checkpoint {step} incomplete")
            specs.append({"label": f"LEGACY(seed{seed})@{step}", "seed": seed, "step": step, "path": cp, "tree_sha256": rec["tree_sha256"]})
    return specs


def _canonical_mask() -> Any:
    from go2arx5_vla.structured_action import structured_action_mask
    return structured_action_mask("T16")


def _array_sha(value: Any) -> str:
    import numpy as np
    return sha256_bytes(np.ascontiguousarray(value).tobytes())


def _flow_metrics(left: Any, right: Any, *, name: str) -> dict[str, Any]:
    import numpy as np
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    mask = np.asarray(_canonical_mask(), dtype=bool)
    require(lhs.shape == RAW_SHAPE and rhs.shape == RAW_SHAPE and np.isfinite(lhs).all() and np.isfinite(rhs).all(), f"{name} shape/finite drift")
    delta = lhs[0][mask] - rhs[0][mask]
    denominator = float(np.linalg.norm(lhs[0][mask]) * np.linalg.norm(rhs[0][mask]))
    return {"namespace": name, "active_scalar_count": int(mask.sum()), "l2": float(np.linalg.norm(delta)), "rmse": float(np.sqrt(np.mean(delta * delta))), "mae": float(np.mean(np.abs(delta))), "max_abs": float(np.max(np.abs(delta))), "cosine": float(np.dot(lhs[0][mask], rhs[0][mask]) / denominator) if denominator else 0.0}


def _snapshot_adapter(record: Mapping[str, Any]) -> Any:
    import numpy as np
    from types import SimpleNamespace
    state = np.asarray(record["state_v1"], dtype=np.float32)
    timestamps = np.asarray(record["source_timestamps"], dtype=np.float64)
    require(state.shape == (10,) and timestamps.shape == (HORIZON,), "sealed snapshot shape drift")
    return SimpleNamespace(example_id=str(record["window_id"]), state_v1=SimpleNamespace(as_array=lambda: state.copy()), future_timestamps=timestamps, source_frame="NATIVE_WBC_RELATIVE_REFERENCE", tcp_semantic_id="x5_link6_gripper_center", observation_timestamp=float(record["observation_timestamp"]), physical_action=np.asarray(record["expert_physical_action"], dtype=np.float64))


def diagnostic_call(torch: Any, stage38: Any, model: Any, conditioning: tuple[Any, Any, Any], state_np: Any, bucket: int) -> Any:
    import numpy as np
    processed_backbone, action_input, state_features = conditioning
    device = processed_backbone.backbone_features.device
    state = torch.from_numpy(np.ascontiguousarray(state_np, dtype=np.float32)).to(device=device, dtype=next(model.parameters()).dtype)
    with torch.inference_mode():
        velocity = stage38._offline_step(model.action_head, processed_backbone, action_input, state_features, state, int(bucket))
    result = velocity.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    require(result.shape == RAW_SHAPE and np.isfinite(result).all(), "diagnostic velocity contract failed")
    return result


def diagnostic(args: argparse.Namespace, phase: str) -> dict[str, Any]:
    import numpy as np
    output = Path(args.output_root).resolve()
    require(not output.exists(), f"{phase} output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    p = paths(args)
    identity = verify_identities(p)
    gpu = gpu_identity(args.gpu_uuid or GPU_UUID)
    prepare_env(p, output, gpu)
    torch, ev, records, noise_manifest, windows, noise_bank = load_context(p)
    import stage38_structured_offline_evaluation as stage38
    from stage45c_h200_eval_runtime_adapter import _sample_one
    from stage38_structured_offline_evaluation import _load_structured_model
    stats_doc = read_json(p["runtime_stats"])
    snapshots = [_snapshot_adapter(record) for record in records]
    specs = checkpoint_specs(args)
    counters = {"gpu": True, "model_construction": 0, "conditioning_passes": 0, "action_head_calls": 0, "raw_rows": 0, "training": False, "optimizer_steps": 0, "checkpoint_writes": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0, "PBS": 0}
    raw_manifest = output / "raw_probe_rows.jsonl"
    raw_manifest.write_text("", encoding="utf-8")
    d1 = phase == "D1"
    seed_grid = (2223, 2224, 2225) if d1 else (4242,)
    t_grid = ((0.06776241796562847, 67), (0.36966943557751086, 369), (0.7837719744658148, 783)) if d1 else ((0.0, 0), (0.25, 250), (0.5, 500), (0.75, 750))
    noise_by_seed: dict[int, list[Any]] = {}
    for seed in seed_grid:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        noise_by_seed[seed] = [torch.randn(RAW_SHAPE, generator=generator, device="cpu", dtype=torch.float32).contiguous() for _ in records]
    rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    try:
        for spec in specs:
            model = _load_structured_model(spec["path"], "cuda:0")
            counters["model_construction"] += 1
            model.eval()
            hook_counter = {"count": 0}
            hook = model.action_head.model.register_forward_hook(lambda *_a, **_k: hook_counter.__setitem__("count", hook_counter["count"] + 1))
            checkpoint_rows = 0
            try:
                for index, (window, record, snapshot) in enumerate(zip(windows, records, snapshots)):
                    _, _, x1 = stage38._structured_label_and_normalizer(np.asarray(snapshot.physical_action, dtype=np.float64)[None, ...], snapshot, stats_doc)
                    x1 = np.asarray(x1, dtype=np.float32)[None, ...]
                    with torch.inference_mode():
                        _, action_input, processed, state_features = stage38._prepare_model_conditioning(model, window["model_inputs"])
                    counters["conditioning_passes"] += 1
                    conditioning = (processed, action_input, state_features)
                    if d1:
                        for noise_seed in seed_grid:
                            x0 = noise_by_seed[noise_seed][index].numpy().copy()
                            for t, bucket in t_grid:
                                x_t = ((1.0 - t) * x0 + t * x1).astype(np.float32, copy=False)
                                v_target = (x1 - x0).astype(np.float32, copy=False)
                                v_hat = diagnostic_call(torch, stage38, model, conditioning, x_t, bucket)
                                counters["action_head_calls"] += 1
                                x1_hat = (x_t + (1.0 - t) * v_hat).astype(np.float32, copy=False)
                                arrays = {"x0": x0, "x1": x1, "x_t": x_t, "v_target": v_target, "v_hat": v_hat, "x1_hat": x1_hat, "mask": np.asarray(_canonical_mask(), dtype=np.float32)}
                                probe = output / "probes" / spec["label"].replace("@", "_at_") / f"w{index:02d}" / f"seed{noise_seed}" / f"t{bucket:03d}"
                                raw_path = probe / "raw_arrays.npz"
                                probe.mkdir(parents=True, exist_ok=True)
                                np.savez(raw_path, **arrays)
                                raw = {"schema": "CORELAB_STAGE45E_LEGACY_D1_RAW_V1", "status": "RAW_SEALED_BEFORE_DERIVED", "identity": {"checkpoint": spec["label"], "checkpoint_tree_sha256": spec["tree_sha256"], "stats_route": "FULL24_RUNTIME_V2", "stats_sha256": RUNTIME_STATS_SHA, "stats_payload_sha256": RUNTIME_STATS_PAYLOAD_SHA, "window_id": record["window_id"], "target_id": record["target_id"], "window_index": index, "noise_seed": noise_seed, "t": t, "t_bucket": bucket, "target_xyz_in_model_input": False}, "raw_archive": {"path": str(raw_path.relative_to(output)), "sha256": sha256_file(raw_path), "bytes": raw_path.stat().st_size}, "arrays": {key: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype), "sha256": _array_sha(value)} for key, value in arrays.items()}}
                                write_json(probe / "raw_manifest.json", raw)
                                derived = {"schema": "CORELAB_STAGE45E_LEGACY_D1_DERIVED_V1", "status": "DERIVED_AFTER_RAW_SEAL", "raw_before_derived": True, "normalized_local_flow": _flow_metrics(v_hat, v_target, name="NORMALIZED_LOCAL_FLOW"), "implied_endpoint": {"formula": "x1_hat=x_t+(1-t)*v_hat", "array_sha256": _array_sha(x1_hat)}}
                                write_json(probe / "derived_metrics.json", derived)
                                row = {"schema": "CORELAB_STAGE45E_LEGACY_D1_ROW_V1", "identity": raw["identity"], "raw_manifest_sha256": sha256_file(probe / "raw_manifest.json"), "raw_sha256": sha256_file(raw_path), "derived_sha256": sha256_file(probe / "derived_metrics.json"), "normalized_local_flow": derived["normalized_local_flow"], "finite": True}
                                with raw_manifest.open("a", encoding="utf-8") as handle:
                                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                                rows.append(row); checkpoint_rows += 1; counters["raw_rows"] += 1
                    else:
                        x0 = noise_by_seed[4242][index].numpy().copy(); x_fr = x0.copy()
                        for stage_index, (t, bucket) in enumerate(t_grid):
                            x_tf = ((1.0 - t) * x0 + t * x1).astype(np.float32, copy=False)
                            v_target = (x1 - x0).astype(np.float32, copy=False)
                            v_tf = diagnostic_call(torch, stage38, model, conditioning, x_tf, bucket); counters["action_head_calls"] += 1
                            v_fr = diagnostic_call(torch, stage38, model, conditioning, x_fr, bucket); counters["action_head_calls"] += 1
                            if bucket == 0:
                                require(np.array_equal(v_tf, v_fr), "D2A t=0 teacher/free mismatch")
                            x_tf_next = (x_tf + DT * v_tf).astype(np.float32, copy=False); x_fr_next = (x_fr + DT * v_fr).astype(np.float32, copy=False)
                            arrays = {"x0": x0, "x1": x1, "x_tf": x_tf, "x_fr": x_fr, "v_target": v_target, "v_tf": v_tf, "v_fr": v_fr, "x_tf_next": x_tf_next, "x_fr_next": x_fr_next, "mask": np.asarray(_canonical_mask(), dtype=np.float32)}
                            probe = output / "probes" / spec["label"].replace("@", "_at_") / f"w{index:02d}" / "seed4242" / f"t{bucket:03d}"; probe.mkdir(parents=True, exist_ok=True)
                            raw_path = probe / "raw_arrays.npz"; np.savez(raw_path, **arrays)
                            raw = {"schema": "CORELAB_STAGE45E_LEGACY_D2A_RAW_V1", "status": "RAW_SEALED_BEFORE_DERIVED", "identity": {"checkpoint": spec["label"], "checkpoint_tree_sha256": spec["tree_sha256"], "stats_route": "FULL24_RUNTIME_V2", "stats_sha256": RUNTIME_STATS_SHA, "stats_payload_sha256": RUNTIME_STATS_PAYLOAD_SHA, "window_id": record["window_id"], "target_id": record["target_id"], "window_index": index, "noise_seed": 4242, "t": t, "t_bucket": bucket, "stage_index": stage_index, "target_xyz_in_model_input": False}, "raw_archive": {"path": str(raw_path.relative_to(output)), "sha256": sha256_file(raw_path), "bytes": raw_path.stat().st_size}, "arrays": {key: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype), "sha256": _array_sha(value)} for key, value in arrays.items()}}
                            write_json(probe / "raw_manifest.json", raw)
                            derived = {"schema": "CORELAB_STAGE45E_LEGACY_D2A_DERIVED_V1", "status": "DERIVED_AFTER_RAW_SEAL", "raw_before_derived": True, "state_path_divergence": _flow_metrics(x_fr, x_tf, name="D2A_STATE_PATH_DIVERGENCE"), "field_sensitivity": _flow_metrics(v_fr, v_tf, name="D2A_FIELD_SENSITIVITY"), "teacher_bridge_local_error": _flow_metrics(v_tf, v_target, name="NORMALIZED_LOCAL_FLOW"), "off_path_diagnostic": _flow_metrics(v_fr, v_target, name="DESCRIPTIVE_OFF_PATH_DIAGNOSTIC_NOT_SUPERVISED"), "t_zero": {"checked": bucket == 0, "exact_equal": bool(np.array_equal(v_tf, v_fr)) if bucket == 0 else None}, "path_update": {"dt": DT, "free_running_reset": False, "x_tf_next_formula": "x_tf+dt*v_tf", "x_fr_next_formula": "x_fr+dt*v_fr"}}
                            write_json(probe / "derived_metrics.json", derived)
                            row = {"schema": "CORELAB_STAGE45E_LEGACY_D2A_ROW_V1", "identity": raw["identity"], "raw_manifest_sha256": sha256_file(probe / "raw_manifest.json"), "raw_sha256": sha256_file(raw_path), "derived_sha256": sha256_file(probe / "derived_metrics.json"), **{key: derived[key] for key in ("state_path_divergence", "field_sensitivity", "teacher_bridge_local_error", "off_path_diagnostic", "t_zero")}, "finite": True}
                            with raw_manifest.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                            rows.append(row); checkpoint_rows += 1; counters["raw_rows"] += 1; x_fr = x_fr_next
            finally:
                hook.remove()
            expected = 216 if d1 else 192
            require(hook_counter["count"] == expected, f"{phase} per-checkpoint action-head budget drift: {hook_counter['count']} != {expected}")
            checkpoints.append({"label": spec["label"], "seed": spec["seed"], "step": spec["step"], "tree_sha256": spec["tree_sha256"], "rows": checkpoint_rows, "conditioning_passes": 24, "action_head_calls": hook_counter["count"]})
            del model
            torch.cuda.empty_cache()
        expected_total = {"D1": (96, 864, 864), "D2A": (96, 768, 384)}[phase]
        require((counters["conditioning_passes"], counters["action_head_calls"], counters["raw_rows"]) == expected_total, f"{phase} total budget drift: {counters}")
        seal = {"schema": f"CORELAB_STAGE45E_LEGACY_{phase}_SEAL_V1", "status": "PASS", "token": f"LEGACY_{phase}_PASS", "pi_gate": PI_GATE, "branch": BRANCH, "repository_head": git("rev-parse", "HEAD"), "identity": identity, "stats_route": "FULL24_RUNTIME_V2", "checkpoints": checkpoints, "counters": counters, "raw_probe_rows_jsonl_sha256": sha256_file(raw_manifest), "gpu": gpu, "training": False, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0, "PBS": 0, "target_xyz_in_model_input": False, "timestamp_utc": now_utc()}
        write_json(output / "seal.json", seal)
        return seal
    except Exception as exc:
        write_json(output / "FAILURE_SEAL.json", {"schema": f"CORELAB_STAGE45E_LEGACY_{phase}_FAILURE_SEAL_V1", "status": "BLOCKED", "phase": phase, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "counters": counters, "retry_or_substitution": False, "timestamp_utc": now_utc()}, overwrite=True)
        raise


def e4(args: argparse.Namespace) -> dict[str, Any]:
    """Run the fixed four-step TRAIN24 E4 evaluation for four new arms."""
    import numpy as np
    output = Path(args.output_root).resolve()
    require(not output.exists(), f"E4 output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    p = paths(args)
    identity = verify_identities(p)
    gpu = gpu_identity(args.gpu_uuid or GPU_UUID)
    prepare_env(p, output, gpu)
    torch, ev, records, noise_manifest, windows, noise_bank = load_context(p)
    import stage38_structured_offline_evaluation as stage38
    from stage38_structured_offline_evaluation import _load_structured_model
    from stage45c_h200_eval_runtime_adapter import _sample_one
    from go2arx5_vla.evaluation.checkpoint_eval import aggregate_evaluation_rows
    stats_doc = read_json(p["runtime_stats"])
    specs = checkpoint_specs(args)
    raw_manifest = output / "raw_predictions.jsonl"
    raw_manifest.write_text("", encoding="utf-8")
    counters = {"gpu": True, "model_construction": 0, "conditioning_passes": 0, "action_head_calls": 0, "model_forward_windows": 0, "raw_rows": 0, "training": False, "optimizer_steps": 0, "checkpoint_writes": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0, "PBS": 0}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    try:
        for spec in specs:
            label = spec["label"]
            model = _load_structured_model(spec["path"], "cuda:0")
            counters["model_construction"] += 1
            model.eval()
            hook_counter = {"count": 0}
            hook = model.action_head.model.register_forward_hook(lambda *_a, **_k: hook_counter.__setitem__("count", hook_counter["count"] + 1))
            rows: list[dict[str, Any]] = []
            try:
                for index, (window, record) in enumerate(zip(windows, records)):
                    with torch.inference_mode():
                        raw, calls = _sample_one(model, window["model_inputs"], noise_bank[index], torch)
                    counters["conditioning_passes"] += 1
                    counters["model_forward_windows"] += 1
                    counters["action_head_calls"] += calls
                    raw_np = raw.detach().cpu().numpy().astype("float32", copy=False)
                    require(raw_np.shape == RAW_SHAPE and np.isfinite(raw_np).all(), f"E4 raw prediction contract failed: {label}/{index}")
                    raw_path = output / "raw" / label.replace("@", "_at_") / f"{index:02d}.npy"
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(raw_path, raw_np, allow_pickle=False)
                    raw_sha = sha256_file(raw_path)
                    physical, details = ev._decode_with_stats(raw_np, record["state_v1"], stats_doc, stats_sha256=RUNTIME_STATS_SHA, payload_sha256=RUNTIME_STATS_PAYLOAD_SHA, chunk_id=record["window_id"], timestamps=record["source_timestamps"], observation_timestamp=float(record["observation_timestamp"]))
                    normalized = ev._normalized_metrics(raw_np, record["expert_physical_action"], record["state_v1"], record["source_timestamps"], stats_doc)
                    checkpoint = {"label": label, "tree_sha256": spec["tree_sha256"]}
                    row = ev._evaluation_row(raw_np, physical, details, normalized, record, checkpoint, raw_path.relative_to(output), raw_sha, GROOT_COMMIT)
                    row.update({"schema": "CORELAB_STAGE45E_LEGACY_E4_ROW_V1", "seed": spec["seed"], "training_step": spec["step"], "stats_route": "FULL24_RUNTIME_V2", "sampler": {"steps_observed": calls, "timestep_ids": list(T16), "dt": DT, "official_public_sampler_called": False}, "target_xyz_in_model_input": False, "finite": True})
                    rows.append(row)
                    with raw_manifest.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"checkpoint": checkpoint, "window_id": record["window_id"], "path": str(raw_path.relative_to(output)), "file_sha256": raw_sha, "array_sha256": sha256_bytes(raw_np.tobytes()), "stats_route": "FULL24_RUNTIME_V2", "sampler_calls_observed": calls}, sort_keys=True) + "\n")
                    counters["raw_rows"] += 1
            finally:
                hook.remove()
            require(hook_counter["count"] == 96, f"E4 per-arm action-head budget drift: {label}={hook_counter['count']}")
            require(len(rows) == 24, f"E4 row count drift: {label}")
            all_rows[label] = rows
            write_json(output / f"aggregate_{label.replace('@', '_at_')}.json", aggregate_evaluation_rows(rows))
            del model
            torch.cuda.empty_cache()
        require((counters["conditioning_passes"], counters["action_head_calls"], counters["raw_rows"]) == (96, 384, 96), f"E4 total budget drift: {counters}")
        seal = {"schema": "CORELAB_STAGE45E_LEGACY_E4_SEAL_V1", "status": "PASS", "token": "LEGACY_E4_PASS", "pi_gate": PI_GATE, "branch": BRANCH, "repository_head": git("rev-parse", "HEAD"), "identity": identity, "stats_route": "FULL24_RUNTIME_V2", "checkpoints": [{"label": spec["label"], "seed": spec["seed"], "step": spec["step"], "tree_sha256": spec["tree_sha256"], "rows": len(all_rows[spec["label"]]), "conditioning_passes": 24, "action_head_calls": 96} for spec in specs], "counters": counters, "raw_predictions_manifest_sha256": sha256_file(raw_manifest), "aggregate_sha256": {label: sha256_file(output / f"aggregate_{label.replace('@', '_at_')}.json") for label in all_rows}, "gpu": gpu, "training": False, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0, "PBS": 0, "target_xyz_in_model_input": False, "timestamp_utc": now_utc()}
        write_json(output / "seal.json", seal)
        return seal
    except Exception as exc:
        write_json(output / "FAILURE_SEAL.json", {"schema": "CORELAB_STAGE45E_LEGACY_E4_FAILURE_SEAL_V1", "status": "BLOCKED", "phase": "E4", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "counters": counters, "retry_or_substitution": False, "timestamp_utc": now_utc()}, overwrite=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "canary", "train", "e4", "d1", "d2a"), default="static")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--canary", type=Path)
    parser.add_argument("--seed2223-root", type=Path)
    parser.add_argument("--seed2224-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    for name in ("base_model", "dataset", "groot", "cosmos", "modality", "config", "runtime_stats", "sealed_baseline", "snapshot_preflight"):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path)
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    if args.mode == "static":
        print(json.dumps(static_audit(), indent=2, sort_keys=True))
        return 0
    try:
        result = {"canary": canary, "train": train, "e4": e4, "d1": lambda a: diagnostic(a, "D1"), "d2a": lambda a: diagnostic(a, "D2A")}[args.mode](args)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    except Exception as exc:
        output = Path(args.output_root).resolve() if args.output_root else None
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "FAILURE_SEAL.json", {"schema": "CORELAB_STAGE45E_LEGACY_FAILURE_SEAL_V1", "status": "BLOCKED", "mode": args.mode, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "retry_or_substitution": False, "timestamp_utc": now_utc()}, overwrite=True)
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
