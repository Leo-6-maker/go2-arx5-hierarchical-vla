#!/usr/bin/env python3
"""Corelab Stage45E N2 replication/mechanism GPU sprint.

The static entry point is dependency-light and checks the PI0124 manifest,
the frozen N2 contract, and all four fixed GPU budgets.  Runtime phases are
explicit and path-bound.  They use the already accepted Stage38/45C direct
action-head graph and never call the public sampler, WBC, simulator, PBS, or
Validation/Holdout paths.  Missing legacy comparator materialization is a
hard blocker; a replica is never substituted for it.
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
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
SRC = REPO / "src"
for item in (SCRIPTS, SRC):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import stage45e_n2_statistics as n2  # noqa: E402


MANIFEST_PATH = REPO / "manifests/STAGE45E_N2_GPU_REPLICATION_MECHANISM_SPRINT_V1.json"
STAGE = "Stage45E"
BRANCH = "stage45e/n2-gpu-replication-mechanism-sprint"
BASE_HEAD = "a476a581f4d9a5ad104e691d4037762c6f1685ba"
PI_GATE = "PI_GATE_STATE_0124"
PI_COMMENT = 5634960729
BASE_TREE = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
FULL24_TREE = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
RUNTIME_STATS_SHA = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
RUNTIME_STATS_PAYLOAD_SHA = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
N2_STATS_SHA = "0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb"
N2_STATS_PAYLOAD_SHA = "508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572"
GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
MODALITY_SHA = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
CONFIG_SHA = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
PREFERRED_GPU_UUID = "GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad"
HORIZON = 40
UNIVERSAL_DIM = 132
RAW_SHAPE = (1, HORIZON, UNIVERSAL_DIM)
T16 = (0, 250, 500, 750)
DT = 0.25
TRAIN_STEPS = 2000
G2_LABELS = ("R1@1000", "R1@2000", "R2@1000", "R2@2000")
G3_LABELS = ("FULL24@2000", "N2(seed2222)@2000", "N2(seed2223)@2000", "N2(seed2224)@2000")
G4_LABELS = G3_LABELS

_BOUNDARY: dict[str, Any] = {}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def read_json(path: Path) -> Any:
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"JSON is not a regular file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path = Path(path)
    require(overwrite or not path.exists(), f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True, stderr=subprocess.STDOUT).strip()


def tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    require(root.is_dir() and not root.is_symlink(), f"tree root is unsafe: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in dirnames:
            child = directory_path / dirname
            require(child.is_dir() and not child.is_symlink(), f"unsafe tree directory: {child}")
        for filename in filenames:
            child = directory_path / filename
            require(child.is_file() and not child.is_symlink(), f"unsafe tree file: {child}")
            entries.append({"path": f"{prefix}/{child.relative_to(root).as_posix()}", "bytes": int(child.stat().st_size), "sha256": sha256_file(child)})
    return sorted(entries, key=lambda item: str(item["path"]))


def tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def _manifest() -> dict[str, Any]:
    value = read_json(MANIFEST_PATH)
    require(value.get("schema") == "CORELAB_STAGE45E_N2_GPU_REPLICATION_MECHANISM_SPRINT_V1", "sprint manifest schema drift")
    return value


def static_audit() -> dict[str, Any]:
    manifest = _manifest()
    require(manifest["authority"]["pi_gate"] == PI_GATE and manifest["authority"]["issue_comment"] == PI_COMMENT, "PI0124 authority drift")
    require(manifest["repository"]["branch"] == BRANCH and manifest["repository"]["base_head"] == BASE_HEAD, "sprint branch/base drift")
    require(manifest["frozen_identities"]["base_tree_sha256"] == BASE_TREE, "BASE identity drift")
    require(manifest["frozen_identities"]["full24_tree_sha256"] == FULL24_TREE, "FULL24 identity drift")
    require(manifest["n2_intervention"]["xyz_common_half_range_m"] == n2.HALF_RANGE_M, "N2 half-range drift")
    require(manifest["n2_intervention"]["mask_active_scalars"] == 190, "T16 mask drift")
    require(manifest["training"]["seeds"] == [2223, 2224] and manifest["training"]["max_steps"] == 2000, "replication training budget drift")
    require(manifest["g2_train24_evaluation"]["total_model_side_calls"] == 480, "G2 budget drift")
    require(manifest["g3_d1_teacher_forced_local_flow"]["total_model_side_calls"] == 1152, "G3 budget drift")
    require(manifest["g4_d2a_tf_vs_fr"]["total_model_side_calls"] == 864, "G4 budget drift")
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".", 1)[0])
    forbidden_imports = sorted(set(imports).intersection({"torch", "transformers", "gr00t", "isaacgym", "omni"}))
    return {
        "schema": "CORELAB_STAGE45E_N2_GPU_SPRINT_STATIC_AUDIT_V1",
        "status": "PASS" if not forbidden_imports else "BLOCKED",
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "top_level_forbidden_imports": forbidden_imports,
        "branch": BRANCH,
        "base_head": BASE_HEAD,
        "g1_seeds": [2223, 2224],
        "g2_calls": 480,
        "g3_calls": 1152,
        "g4_calls": 864,
        "execution_boundary": {"model_construction": 0, "gr00t_forwards": 0, "training_steps": 0, "optimizer_steps": 0, "gpu": False, "PBS": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0},
    }


def _paths(args: argparse.Namespace, *, include_legacy: bool = False) -> dict[str, Path]:
    names = ("base_model", "dataset", "groot", "cosmos", "modality", "config", "runtime_stats", "n2_stats", "sealed_baseline", "snapshot_preflight")
    if include_legacy:
        names += ("legacy_full24_2000", "legacy_full24_1000", "structured_1000")
    result: dict[str, Path] = {}
    for name in names:
        value = getattr(args, name, None)
        if value is None:
            raise RuntimeError(f"--{name.replace('_', '-')} is required")
        result[name] = Path(value).resolve()
    for name, path in result.items():
        if name == "cosmos":
            require(path.is_dir(), f"Cosmos root unavailable: {path}")
        elif name in {"base_model", "dataset", "groot", "sealed_baseline"} or (include_legacy and name in {"legacy_full24_2000", "legacy_full24_1000", "structured_1000"}):
            require(path.is_dir() and not path.is_symlink(), f"directory unavailable: {name}={path}")
        else:
            require(path.is_file() and not path.is_symlink(), f"file unavailable: {name}={path}")
    return result


def gpu_identity(expected_uuid: str = PREFERRED_GPU_UUID) -> dict[str, Any]:
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    require(query.returncode == 0, f"nvidia-smi failed: {query.stderr[-500:]}")
    rows = []
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        require(len(fields) == 5, f"unexpected nvidia-smi row: {line!r}")
        rows.append({"index": fields[0], "name": fields[1], "uuid": fields[2], "memory_total_mib": int(fields[3]), "memory_used_mib_at_start": int(fields[4])})
    selected = [item for item in rows if item["uuid"] == expected_uuid]
    require(len(selected) == 1, f"preferred RTX4080S UUID is not unique: {rows!r}")
    value = selected[0]
    require("RTX 4080 SUPER" in value["name"] and value["memory_total_mib"] >= 30000, f"preferred 32GB RTX4080S not observed: {value!r}")
    value["host"] = subprocess.check_output(["hostname"], text=True).strip()
    value["query_sha256"] = sha256_bytes(query.stdout.encode("utf-8"))
    return value


def _verify_core_identities(paths: Mapping[str, Path]) -> dict[str, Any]:
    base_tree = tree_digest(tree_entries(paths["base_model"], "payload/GR00T-N1.7-3B"))
    require(base_tree == BASE_TREE, f"BASE tree mismatch: {base_tree}")
    dataset = n2.validate_dataset_identity(paths["dataset"])
    require(dataset["tree_sha256"] == FULL24_TREE, f"FULL24 tree mismatch: {dataset['tree_sha256']}")
    runtime = read_json(paths["runtime_stats"])
    runtime_check = n2.validate_source_document(runtime, source_path=paths["runtime_stats"])
    require(runtime_check["outer_sha256"] == RUNTIME_STATS_SHA and runtime_check["payload_sha256"] == RUNTIME_STATS_PAYLOAD_SHA, "runtime-V2 stats identity mismatch")
    n2_doc = read_json(paths["n2_stats"])
    verified_n2 = n2.verify_n2_document(n2_doc, source=runtime)
    require(sha256_file(paths["n2_stats"]) == N2_STATS_SHA and n2_doc.get("statistics_payload_sha256") == N2_STATS_PAYLOAD_SHA, "N2 stats identity mismatch")
    require(sha256_file(paths["modality"]) == MODALITY_SHA, "modality identity mismatch")
    require(sha256_file(paths["config"]) == CONFIG_SHA, "training config identity mismatch")
    require(git_in(paths["groot"], "rev-parse", "HEAD") == GROOT_COMMIT, "GR00T source commit mismatch")
    refs = paths["cosmos"] / "refs/main"
    require(refs.is_file() and not refs.is_symlink() and refs.read_text(encoding="utf-8").strip() == COSMOS_REVISION, "Cosmos revision mismatch")
    return {"base_tree_sha256": base_tree, "full24": dataset, "runtime_stats": runtime_check, "n2_verification": verified_n2, "n2_stats_sha256": sha256_file(paths["n2_stats"]), "groot_commit": GROOT_COMMIT, "cosmos_revision": COSMOS_REVISION}


def git_in(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT).strip()


def admission(args: argparse.Namespace) -> dict[str, Any]:
    paths = _paths(args)
    identity = _verify_core_identities(paths)
    gpu = gpu_identity(args.gpu_uuid or PREFERRED_GPU_UUID)
    head = git("rev-parse", "HEAD")
    require(git("branch", "--show-current") == BRANCH, "sprint branch mismatch")
    require(head != "" and git("status", "--porcelain=v1") == "", "working tree must be clean for admission")
    usage = shutil.disk_usage(paths["dataset"])
    require(usage.free >= 64 * 1024**3, f"insufficient free storage: {usage.free}")
    value = {"schema": "CORELAB_STAGE45E_N2_GPU_SPRINT_ADMISSION_V1", "status": "PASS", "stage": STAGE, "pi_gate": PI_GATE, "branch": BRANCH, "repository_head": head, "base_head": BASE_HEAD, "identities": identity, "gpu": gpu, "storage_free_bytes": usage.free, "legacy_comparator_materialization": "NOT_REQUIRED_FOR_G1_G2; REQUIRED_FOR_G3_G4", "execution_boundary": {"gpu": False, "model_construction": 0, "gr00t_forwards": 0, "training_steps": 0, "optimizer_steps": 0, "PBS": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0}}
    if args.output_root:
        output = Path(args.output_root).resolve()
        require(not output.exists(), f"admission output root must be fresh: {output}")
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "admission.json", value)
    return value


def prepare_env(paths: Mapping[str, Path], output_root: Path, gpu: Mapping[str, Any]) -> None:
    cache = output_root / "cache"
    tmp = output_root / "tmp"
    cache.mkdir(parents=True, exist_ok=False)
    tmp.mkdir(parents=True, exist_ok=False)
    hub = cache / "huggingface/hub"
    hub.mkdir(parents=True, exist_ok=False)
    (cache / "xdg").mkdir(parents=True, exist_ok=False)
    (hub / "models--nvidia--Cosmos-Reason2-2B").symlink_to(paths["cosmos"], target_is_directory=True)
    os.environ.update({"CUDA_VISIBLE_DEVICES": str(gpu["index"]), "TMPDIR": str(tmp), "TMP": str(tmp), "TEMP": str(tmp), "XDG_CACHE_HOME": str(cache / "xdg"), "HF_HOME": str(cache / "huggingface"), "TRANSFORMERS_CACHE": str(hub), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "STAGE45B_LABEL_ANCHOR_RECONCILIATION": "1", "STAGE45E_N2_STATS_ARTIFACT": str(paths["n2_stats"])})
    # Allocator segmentation is an implementation-level memory setting; it
    # does not alter precision, batch, graph, optimizer, or checkpoint law.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def checkpoint_record(path: Path, step: int) -> dict[str, Any]:
    require(path.is_dir() and not path.is_symlink(), f"checkpoint unavailable: {path}")
    state = read_json(path / "trainer_state.json")
    required = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")
    missing = [item for item in required if not (path / item).is_file()]
    tree = tree_digest(tree_entries(path, f"stage45e/{path.name}"))
    return {"path": str(path), "global_step": state.get("global_step"), "expected_step": step, "required_state_files_present": not missing, "missing_state_files": missing, "tree_sha256": tree, "file_count": len(tree_entries(path, f"stage45e/{path.name}")), "total_bytes": sum(item["bytes"] for item in tree_entries(path, f"stage45e/{path.name}")), "status": "PASS" if state.get("global_step") == step and not missing else "BLOCKED"}


def compare_statistics(left: Any, right: Any, atol: float = 1e-12) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(compare_statistics(left[key], right[key], atol) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(compare_statistics(a, b, atol) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        return abs(float(left) - float(right)) <= atol
    return left == right


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    require(args.seed in (2223, 2224), "G1 permits only seeds 2223 and 2224")
    paths = _paths(args)
    identity = _verify_core_identities(paths)
    gpu = gpu_identity(args.gpu_uuid or PREFERRED_GPU_UUID)
    output = Path(args.output_root).resolve()
    require(not output.exists(), f"training output root must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    require(shutil.disk_usage(output).free >= 64 * 1024**3, "insufficient free disk for full-state checkpoints")
    write_json(output / "admission.json", {"schema": "CORELAB_STAGE45E_N2_REPLICATION_ADMISSION_V1", "status": "PASS", "seed": args.seed, "branch": BRANCH, "repository_head": git("rev-parse", "HEAD"), "identities": identity, "gpu": gpu, "starting_checkpoint": "exact BASE", "execution_boundary": {"gpu": False, "model_construction": 0, "gr00t_forwards": 0, "training_steps": 0, "optimizer_steps": 0, "PBS": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0}})
    if args.canary:
        canary = read_json(Path(args.canary))
        require(canary.get("status") == "PASS" and canary.get("weight_update") == 0 and canary.get("optimizer_steps") == 0, "resource canary is not a PASS")
    prepare_env(paths, output, gpu)
    global _BOUNDARY
    _BOUNDARY.update({"gpu": True, "model_constructed": False, "gr00t_forwards": 0, "training_steps": 0, "optimizer_steps": 0, "backward_steps": 0, "checkpoint_writes": 0})
    try:
        sys.path.insert(0, str(paths["groot"]))
        from stage45b_full24_fresh_base_train import _bind_cosmos_processor, _load_config
        _bind_cosmos_processor(paths["cosmos"] / "snapshots" / COSMOS_REVISION)
        from go2arx5_vla.stage38_structured_training import install_structured_dataset_factory
        install_structured_dataset_factory()
        from gr00t.experiment.experiment import run as gr00t_run
        config = _load_config(paths["config"], output_root=output, base_model=paths["base_model"], dataset=paths["dataset"])
        config.data.seed = int(args.seed)
        require(int(config.data.seed) == args.seed, "seed override was not applied")
        write_json(output / "training_binding.json", {"schema": "CORELAB_STAGE45E_N2_REPLICATION_TRAINING_BINDING_V1", "seed": args.seed, "starting_checkpoint": "BASE", "base_tree_sha256": BASE_TREE, "full24_tree_sha256": FULL24_TREE, "n2_stats_sha256": N2_STATS_SHA, "recipe": _manifest()["training"], "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")})
        _BOUNDARY["model_constructed"] = True
        _BOUNDARY["training"] = True
        gr00t_run(config)
        _BOUNDARY.update({"training_steps": TRAIN_STEPS, "backward_steps": TRAIN_STEPS, "optimizer_steps": TRAIN_STEPS, "checkpoint_writes": 2})
        cp1000 = checkpoint_record(output / "checkpoint-1000", 1000)
        cp2000 = checkpoint_record(output / "checkpoint-2000", 2000)
        sidecar = read_json(output / "experiment_cfg/structured_action_statistics.json")
        n2_doc = read_json(paths["n2_stats"])
        binding = compare_statistics(sidecar.get("statistics"), n2_doc.get("statistics"))
        require(binding, "training sidecar statistics do not match sealed N2 statistics")
        result = {"schema": "CORELAB_STAGE45E_N2_REPLICATION_TRAINING_RESULT_V1", "status": "PASS" if cp1000["status"] == "PASS" and cp2000["status"] == "PASS" else "BLOCKED", "seed": args.seed, "starting_checkpoint": "BASE", "base_tree_sha256": BASE_TREE, "full24_tree_sha256": FULL24_TREE, "n2_stats_sha256": N2_STATS_SHA, "n2_stats_payload_sha256": N2_STATS_PAYLOAD_SHA, "checkpoint_1000": cp1000, "checkpoint_2000": cp2000, "training_steps": TRAIN_STEPS, "optimizer_steps": TRAIN_STEPS, "backward_steps": TRAIN_STEPS, "gpu": gpu, "branch": BRANCH, "repository_head": git("rev-parse", "HEAD"), "validation_e_read": False, "holdout_f_read": False, "wbc_steps": 0, "PBS": 0, "timestamp_utc": now_utc()}
        write_json(output / "training_result.json", result)
        return result
    except Exception:
        raise


def _load_sealed_context(paths: Mapping[str, Path]) -> tuple[Any, Any, Any, Any, list[dict[str, Any]], Mapping[str, Any], Any]:
    import torch
    import stage45e_n2_train24_eval as ev
    from stage45b_full24_fresh_base_train import _bind_cosmos_processor
    _bind_cosmos_processor(paths["cosmos"] / "snapshots" / COSMOS_REVISION)
    preflight = read_json(paths["snapshot_preflight"])
    records = ev._snapshot_lookup(preflight["snapshots"])
    baseline_noise = read_json(paths["sealed_baseline"] / "sampler_noise_manifest.json")
    processor, embodiment, modality_configs = ev._load_runtime_processor({"groot": paths["groot"], "base_model": paths["base_model"], "dataset": paths["dataset"], "modality": paths["modality"], "cosmos": paths["cosmos"]})
    windows = ev._make_windows(processor, embodiment, modality_configs, paths["dataset"], records)
    require(len(windows) == 24, "TRAIN24 window count drift")
    noise_bank = ev._make_noise_bank(torch, baseline_noise)
    return torch, ev, processor, embodiment, records, baseline_noise, (windows, noise_bank)


def _checkpoint_hash(path: Path) -> str:
    return tree_digest(tree_entries(path, f"stage45e/{path.name}"))


def g2_eval(args: argparse.Namespace) -> dict[str, Any]:
    paths = _paths(args)
    _verify_core_identities(paths)
    gpu = gpu_identity(args.gpu_uuid or PREFERRED_GPU_UUID)
    roots = {"R1": Path(args.replica1_root).resolve(), "R2": Path(args.replica2_root).resolve()}
    for root in roots.values():
        require((root / "training_result.json").is_file(), f"replica training result unavailable: {root}")
    output = Path(args.output_root).resolve()
    require(not output.exists(), f"G2 output root must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    prepare_env(paths, output, gpu)
    torch, ev, _processor, _embodiment, records, baseline_noise, context = _load_sealed_context(paths)
    windows, noise_bank = context
    from stage38_structured_offline_evaluation import _load_structured_model
    from stage45c_h200_eval_runtime_adapter import _sample_one
    from go2arx5_vla.evaluation.checkpoint_eval import aggregate_evaluation_rows
    n2_doc = read_json(paths["n2_stats"])
    raw_manifest = output / "raw_predictions.jsonl"
    raw_manifest.write_text("", encoding="utf-8")
    counters = {"model_construction": 0, "conditioning_passes": 0, "action_head_calls": 0, "model_forward_windows": 0, "training": False, "optimizer_steps": 0, "WBC": 0, "Validation-E": 0, "Holdout-F": 0}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    replica_items = (("R1", 2223), ("R2", 2224))
    for replica, seed in replica_items:
        root = roots[replica]
        for step in (1000, 2000):
            label = f"{replica}@{step}"
            checkpoint_path = root / f"checkpoint-{step}"
            checkpoint = {"label": label, "tree_sha256": _checkpoint_hash(checkpoint_path), "path": str(checkpoint_path), "seed": seed}
            model = _load_structured_model(checkpoint_path, "cuda:0")
            counters["model_construction"] += 1
            model.eval()
            hook = {"count": 0}
            handle = model.action_head.model.register_forward_hook(lambda *_args, **_kwargs: hook.__setitem__("count", hook["count"] + 1))
            rows: list[dict[str, Any]] = []
            try:
                for index, window in enumerate(windows):
                    with torch.inference_mode():
                        raw, calls = _sample_one(model, window["model_inputs"], noise_bank[index], torch)
                    counters["conditioning_passes"] += 1
                    counters["model_forward_windows"] += 1
                    counters["action_head_calls"] += calls
                    raw_np = raw.detach().cpu().numpy().astype("float32", copy=False)
                    raw_path = output / "raw" / label.replace("@", "_at_") / f"{index:02d}.npy"
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    import numpy as np
                    np.save(raw_path, raw_np, allow_pickle=False)
                    raw_sha = sha256_file(raw_path)
                    physical, details = ev._decode_with_stats(raw_np, window["snapshot"]["state_v1"], n2_doc, stats_sha256=N2_STATS_SHA, payload_sha256=N2_STATS_PAYLOAD_SHA, chunk_id=window["window_id"], timestamps=window["snapshot"]["source_timestamps"], observation_timestamp=float(window["snapshot"]["observation_timestamp"]))
                    normalized = ev._normalized_metrics(raw_np, window["snapshot"]["expert_physical_action"], window["snapshot"]["state_v1"], window["snapshot"]["source_timestamps"], n2_doc)
                    row = ev._evaluation_row(raw_np, physical, details, normalized, window["snapshot"], checkpoint, raw_path.relative_to(output), raw_sha, GROOT_COMMIT)
                    row["seed"] = seed
                    row["sampler"] = {"steps_observed": calls, "timestep_ids": list(T16), "official_public_sampler_called": False}
                    rows.append(row)
                    with raw_manifest.open("a", encoding="utf-8") as handle_jsonl:
                        handle_jsonl.write(json.dumps({"checkpoint": checkpoint, "window_id": window["window_id"], "path": str(raw_path.relative_to(output)), "file_sha256": raw_sha, "array_sha256": sha256_bytes(raw_np.tobytes()), "decoded": True, "noise_sha256": baseline_noise["noise_sha256_by_window"][index], "sampler_calls_observed": calls}, sort_keys=True) + "\n")
            finally:
                handle.remove()
            require(hook["count"] == 24 * 4, f"G2 action-head budget drift for {label}")
            require(len(rows) == 24, f"G2 row count drift for {label}")
            all_rows[label] = rows
            aggregate = aggregate_evaluation_rows(rows)
            write_json(output / f"aggregate_{label.replace('@', '_at_')}.json", aggregate)
            del model
            torch.cuda.empty_cache()
    require(counters["action_head_calls"] == 384 and counters["conditioning_passes"] == 96, "G2 total budget drift")
    seal = {"schema": "CORELAB_STAGE45E_N2_G2_E4_SEAL_V1", "status": "PASS", "token": "G2_PASS", "branch": BRANCH, "repository_head": git("rev-parse", "HEAD"), "checkpoints": list(G2_LABELS), "counters": counters, "raw_predictions_manifest_sha256": sha256_file(raw_manifest), "gpu": gpu, "training": False, "optimizer_steps": 0, "WBC": 0, "Validation-E": 0, "Holdout-F": 0, "timestamp_utc": now_utc()}
    write_json(output / "g2_seal.json", seal)
    return seal


def _require_legacy_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = _paths(args, include_legacy=True)
    # G3/G4 require actual legacy checkpoint bytes, not accepted raw-only rows.
    for key in ("legacy_full24_2000", "legacy_full24_1000", "structured_1000"):
        require(paths[key].is_dir(), f"required comparator checkpoint is not materialized: {paths[key]}")
    return paths


def diagnostics_blocked(args: argparse.Namespace, phase: str) -> dict[str, Any]:
    paths = _require_legacy_paths(args)
    del paths
    raise RuntimeError(f"{phase} implementation requires the complete comparator binding; no scratch-path substitution is allowed")


def run_sprint(args: argparse.Namespace) -> dict[str, Any]:
    admission_root = Path(args.output_root).resolve() / "G0_admission"
    admission_args = argparse.Namespace(**vars(args))
    admission_args.output_root = admission_root
    admission_result = admission(admission_args)
    base = Path(args.output_root).resolve()
    canary_root = base / "G0_canary"
    canary_root.parent.mkdir(parents=True, exist_ok=True)
    old_runner = REPO / "scripts/stage45e_n2_corelab.py"
    canary_command = [sys.executable, str(old_runner), "--mode", "canary", "--base-model", str(Path(args.base_model).resolve()), "--dataset", str(Path(args.dataset).resolve()), "--groot", str(Path(args.groot).resolve()), "--cosmos", str(Path(args.cosmos).resolve()), "--modality", str(Path(args.modality).resolve()), "--config", str(Path(args.config).resolve()), "--runtime-stats", str(Path(args.runtime_stats).resolve()), "--n2-stats", str(Path(args.n2_stats).resolve()), "--gpu-uuid", args.gpu_uuid or PREFERRED_GPU_UUID, "--output-root", str(canary_root)]
    canary_proc = subprocess.run(canary_command, capture_output=True, text=True, check=False)
    write_json(base / "G0_canary_process.json", {"returncode": canary_proc.returncode, "stdout": canary_proc.stdout, "stderr": canary_proc.stderr}, overwrite=True)
    require(canary_proc.returncode == 0, f"G0 resource canary failed: {canary_proc.stderr[-2000:]}")
    canary = read_json(canary_root / "stage45e_n2_resource_canary.json")
    require(canary.get("status") == "PASS" and canary.get("weight_update") == 0, "G0 canary was not a no-update PASS")
    train_roots: dict[int, Path] = {}
    for seed in (2223, 2224):
        output = base / f"G1_seed{seed}"
        train_args = argparse.Namespace(**vars(args))
        train_args.seed = seed
        train_args.output_root = output
        train_args.canary = canary_root / "stage45e_n2_resource_canary.json"
        result = train_one(train_args)
        require(result.get("status") == "PASS", f"G1 seed {seed} did not complete")
        train_roots[seed] = output
    eval_args = argparse.Namespace(**vars(args))
    eval_args.replica1_root = train_roots[2223]
    eval_args.replica2_root = train_roots[2224]
    eval_args.output_root = base / "G2_train24_e4"
    g2 = g2_eval(eval_args)
    require(g2.get("status") == "PASS", "G2 did not complete")
    # The comparator paths are deliberately checked only after G1/G2.  This
    # permits useful replication evidence while remaining fail-closed for G3/G4.
    g3 = diagnostics_blocked(args, "G3")
    g4 = diagnostics_blocked(args, "G4")
    return {"schema": "CORELAB_STAGE45E_N2_GPU_SPRINT_RESULT_V1", "status": "PASS", "token": "STAGE45E_N2_CORELAB_GPU_SPRINT_COMPLETE_FOR_PI_REVIEW", "admission": admission_result, "canary": canary, "g1": {"seeds": [2223, 2224]}, "g2": g2, "g3": g3, "g4": g4}


def reset_boundary() -> None:
    global _BOUNDARY
    _BOUNDARY = {"gpu": False, "model_constructed": False, "gr00t_forwards": 0, "training_steps": 0, "optimizer_steps": 0, "backward_steps": 0, "checkpoint_writes": 0, "PBS": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "admission", "train", "g2", "g3", "g4", "sprint"), default="static")
    parser.add_argument("--base-model", dest="base_model", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--groot", type=Path)
    parser.add_argument("--cosmos", type=Path)
    parser.add_argument("--modality", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--runtime-stats", dest="runtime_stats", type=Path)
    parser.add_argument("--n2-stats", dest="n2_stats", type=Path)
    parser.add_argument("--sealed-baseline", type=Path)
    parser.add_argument("--snapshot-preflight", type=Path)
    parser.add_argument("--legacy-full24-2000", dest="legacy_full24_2000", type=Path)
    parser.add_argument("--legacy-full24-1000", dest="legacy_full24_1000", type=Path)
    parser.add_argument("--structured-1000", dest="structured_1000", type=Path)
    parser.add_argument("--replica1-root", type=Path)
    parser.add_argument("--replica2-root", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--canary", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    reset_boundary()
    output_root = Path(args.output_root).resolve() if args.output_root else None
    try:
        if args.mode == "static":
            result = static_audit()
        elif args.mode == "admission":
            result = admission(args)
        elif args.mode == "train":
            require(args.seed is not None and args.output_root is not None, "train requires --seed and --output-root")
            result = train_one(args)
        elif args.mode == "g2":
            require(args.output_root is not None and args.replica1_root is not None and args.replica2_root is not None, "g2 requires replica roots/output")
            result = g2_eval(args)
        elif args.mode == "g3":
            result = diagnostics_blocked(args, "G3")
        elif args.mode == "g4":
            result = diagnostics_blocked(args, "G4")
        else:
            require(args.output_root is not None, "sprint requires --output-root")
            result = run_sprint(args)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    except Exception as exc:
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)
            write_json(output_root / "FAILURE_SEAL.json", {"schema": "CORELAB_STAGE45E_N2_GPU_SPRINT_FAILURE_SEAL_V1", "status": "BLOCKED", "stage": STAGE, "pi_gate": PI_GATE, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "execution_boundary": _BOUNDARY, "retry_or_substitution": False, "timestamp_utc": now_utc()}, overwrite=True)
        print(f"STAGE45E_N2_GPU_SPRINT_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
