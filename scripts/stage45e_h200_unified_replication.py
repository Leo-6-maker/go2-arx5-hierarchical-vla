#!/usr/bin/env python3
"""PI0133 packed H200 orchestration and CPU closure for Stage45E.

The static and high-water paths are CPU-only.  The explicit ``campaign``
path is intended to run inside one PBS H200 allocation and invokes the three
already-reviewed science branches as separate subprocesses.  It never merges
those branches, never resumes a checkpoint, and never schedules a retry after
model execution has started.  Each subprocess receives absolute paths and an
empty ``PYTHONPATH`` so that the committed entrypoints are tested from a
foreign working directory.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "manifests/STAGE45E_H200_UNIFIED_REPLICATION_PI0133_V1.json"
PYTHON = Path("/scratch/e1192548/stage22_h200_runtime_deb920_20260902/runtime/venv/bin/python")
STORAGE_ROOT = Path("/hpctmp/e1192548/stage45e_h200_unified_replication_20260912")
WORKTREE_ROOT = Path("/scratch/e1192548/stage45e_h200_unified_worktrees")
DEFAULTS = {
    "base_model": Path("/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/GR00T-N1.7-3B"),
    "dataset": Path("/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/lerobot_train"),
    "groot": Path("/scratch/e1192548/stage45c_pi0094_native_runtime_20260909/groot-source-v4"),
    "cosmos": Path("/scratch/e1192548/stage22_h200_cosmos_v2_20260902/hf_home/hub/models--nvidia--Cosmos-Reason2-2B"),
    "modality": Path("/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/modality/go2_arx5_modality.py"),
    "config": Path("/scratch/e1192548/stage28_h2_20260904/stage28_h2_600695.hopper-m-02/h2_training/experiment_cfg/config.yaml"),
    "runtime_stats": Path("/scratch/e1192548/stage45b_pi0083_runtime_parquet_v2/STAGE45B_FULL24_RUNTIME_PARQUET_STATS_V2.json"),
    "n2_stats": Path("/hpctmp/e1192548/stage45e_h200_unified_replication_20260912/provenance/STAGE45E_N2_BALANCED_XYZ_STATS_PI0133.json"),
    "sealed_baseline": Path("/scratch/e1192548/stage45c_pi0100_h200_train24_eval_74e77d83d85e_20260910"),
    "snapshot_preflight": Path("/scratch/e1192548/stage45c_pi0100_h200_train24_eval_74e77d83d85e_20260910/stage45c_runtime_preflight.json"),
    "legacy_history": Path("/scratch/e1192548/stage45b_pi0087_b2_h200_20260909T052311Z"),
    "l1_history": Path("/scratch/e1192548/stage45e_l1_h200_20260911_f77f9e02_612342.hopper-m-02"),
}
BASE_TREE = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
FULL24_TREE = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
RUNTIME_STATS_SHA = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
RUNTIME_STATS_PAYLOAD_SHA = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
N2_STATS_SHA = "e88ff3ec5626130ca8a08006abe1ada70d34424af4b0d3bacf64db57149d7677"
N2_STATS_PAYLOAD_SHA = "508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572"
GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
GROOT_MANIFEST_SHA = "96b883e56b54f763bec60d3a336632ad531f4fc52d099f803e5f9efba60eab11"
GROOT_MANIFEST_PAYLOAD_SHA = "a9f10b837b6086137fa519cc148275a25bf6a5f98d58df8e3c50e0a43e5b8250"
GROOT_PAYLOAD_TREE = "c0d57a6a74ab24c48abdd1c13420e9d1c893dc0d7b5ca370fa9b6f7af32f5f1b"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
MODALITY_SHA = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
CONFIG_SHA = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
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
TRAINING_QUEUE = (
    ("legacy", 2223),
    ("legacy", 2224),
    ("l1", 2223),
    ("l1", 2224),
    ("n2", 2222),
    ("n2", 2223),
    ("n2", 2224),
)
ALL_SEEDS = (2222, 2223, 2224)
ROUTES = ("legacy", "l1", "n2")
ESTIMATED_TRAINING_ROOT_BYTES = 51_318_519_711
MIN_FREE_BYTES = 64 * 1024**3


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    require(completed.returncode == 0, f"git failed in {repo}: {' '.join(args)}: {completed.stderr[-1000:]}")
    return completed.stdout.strip()


def regular_tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    mode = root.lstat().st_mode
    require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"unsafe tree root: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in dirnames:
            child = directory_path / dirname
            child_mode = child.lstat().st_mode
            require(stat.S_ISDIR(child_mode) and not stat.S_ISLNK(child_mode), f"unsafe tree directory: {child}")
        for filename in filenames:
            child = directory_path / filename
            child_mode = child.lstat().st_mode
            require(stat.S_ISREG(child_mode) and not stat.S_ISLNK(child_mode), f"unsafe tree file: {child}")
            entries.append({"path": f"{prefix}/{child.relative_to(root).as_posix()}", "bytes": child.stat().st_size, "sha256": sha256_file(child)})
    return sorted(entries, key=lambda item: str(item["path"]))


def tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode())
    return digest.hexdigest()


def cosmos_digest(root: Path) -> str:
    root = Path(root).resolve()
    require(root.is_dir() and not root.is_symlink(), f"Cosmos root unavailable: {root}")
    entries: list[dict[str, Any]] = []

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
                entries.append({"path": rel, "kind": "symlink", "size_bytes": resolved.stat().st_size, "sha256": sha256_file(resolved), "target": os.readlink(item), "resolved_path": resolved.relative_to(root).as_posix()})
            elif stat.S_ISREG(mode):
                entries.append({"path": rel, "kind": "regular_file", "size_bytes": item.stat().st_size, "sha256": sha256_file(item), "target": None, "resolved_path": None})
            else:
                raise RuntimeError(f"unsupported Cosmos entry: {item}")

    visit(root, Path("."))
    lines = ["\t".join((str(item["kind"]), str(item["path"]), str(item["size_bytes"]), str(item.get("sha256") or "-"), str(item.get("target") or "-"), str(item.get("resolved_path") or "-"))) for item in sorted(entries, key=lambda value: str(value["path"]))]
    return sha256_bytes(("\n".join(lines) + "\n").encode())


def load_manifest() -> dict[str, Any]:
    value = read_json(MANIFEST_PATH)
    require(value.get("schema") == "VANDA_STAGE45E_H200_UNIFIED_THREE_ROUTE_REPLICATION_PI0133_V1", "PI0133 unified manifest schema drift")
    return value


def static_audit() -> dict[str, Any]:
    manifest = load_manifest()
    require(manifest["authority"]["pi_gate"] == "PI_GATE_STATE_0133" and manifest["authority"]["issue_comment"] == 5642852068, "PI0133 authority drift")
    require(manifest["orchestration_repository"]["branch"] == "stage45e/h200-unified-replication-campaign", "orchestration branch drift")
    require(manifest["diagnostic_contract"]["total_model_side_calls"] == 8976, "diagnostic total drift")
    require(manifest["diagnostic_contract"]["e4"]["total_model_side_calls"] == 1680, "E4 ceiling drift")
    require(manifest["diagnostic_contract"]["d1"]["total_model_side_calls"] == 3840, "D1 ceiling drift")
    require(manifest["diagnostic_contract"]["d2a"]["total_model_side_calls"] == 3456, "D2A ceiling drift")
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    top_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.add(node.module.split(".", 1)[0])
    forbidden = sorted(top_imports.intersection({"torch", "transformers", "gr00t", "jax", "tensorflow", "isaacgym", "omni"}))
    require(not forbidden, f"orchestration top-level model imports: {forbidden}")
    call_names = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    require(not call_names.intersection({"qsub", "merge", "checkout"}), "orchestration contains forbidden scheduler/merge mutation")
    return {
        "schema": "VANDA_STAGE45E_H200_UNIFIED_REPLICATION_STATIC_AUDIT_V1",
        "status": "PASS",
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "top_level_forbidden_imports": forbidden,
        "branch_separation": True,
        "fresh_base_training_lineages": 7,
        "e4_model_side_calls": 1680,
        "d1_model_side_calls": 3840,
        "d2a_model_side_calls": 3456,
        "total_model_side_calls": 8976,
        "gpu": 0,
        "pbs_submissions": 0,
        "model_construction": 0,
        "model_forward": 0,
        "training": 0,
        "backward": 0,
        "optimizer_steps": 0,
        "Validation-E": 0,
        "Holdout-F": 0,
        "WBC": 0,
        "simulator": 0,
        "hardware": 0,
    }


def _gpu_observation() -> dict[str, Any]:
    require(os.environ.get("PBS_JOBID"), "PI0133 campaign requires PBS_JOBID")
    query = "index,name,uuid,memory.total,memory.used"
    result = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    require(result.returncode == 0, f"nvidia-smi failed: {result.stderr[-1000:]}")
    rows = []
    for line in result.stdout.splitlines():
        if line.strip():
            fields = [part.strip() for part in line.split(",")]
            require(len(fields) == 5, f"malformed nvidia-smi row: {line}")
            rows.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2], "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4])})
    require(len(rows) == 1 and rows[0]["index"] == 0, f"PI0133 requires one H200 at index 0: {rows}")
    require("H200" in rows[0]["name"].upper() and rows[0]["memory_total_mib"] >= 80 * 1024, f"H200 identity/VRAM gate failed: {rows}")
    return {"schema": "VANDA_STAGE45E_UNIFIED_H200_ADMISSION_GPU_V1", "host": subprocess.check_output(["hostname"], text=True).strip(), "pbs_jobid": os.environ["PBS_JOBID"], "selected_device": "cuda:0", "rows": rows, "nvidia_smi_query": query, "nvidia_smi_stdout_sha256": sha256_bytes((result.stdout + "\n").encode())}


def _source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {name: Path(getattr(args, name)).resolve() for name in DEFAULTS if name not in ("legacy_history", "l1_history")}


def verify_sources(args: argparse.Namespace, n2_repo: Path) -> dict[str, Any]:
    paths = _source_paths(args)
    for name, path in paths.items():
        require(path.exists() and not path.is_symlink(), f"source path unavailable: {name}: {path}")
    base_entries = regular_tree_entries(paths["base_model"], "payload/GR00T-N1.7-3B")
    base_tree = tree_digest(base_entries)
    require(base_tree == BASE_TREE, f"BASE tree mismatch: {base_tree}")
    sys.path.insert(0, str(n2_repo / "scripts"))
    sys.path.insert(0, str(n2_repo / "src"))
    import stage45e_n2_statistics as n2  # CPU-only module

    dataset = n2.validate_dataset_identity(paths["dataset"])
    require(dataset["tree_sha256"] == FULL24_TREE, "FULL24 tree mismatch")
    runtime_doc = read_json(paths["runtime_stats"])
    runtime = n2.validate_source_document(runtime_doc, source_path=paths["runtime_stats"])
    require(runtime["outer_sha256"] == RUNTIME_STATS_SHA and runtime["payload_sha256"] == RUNTIME_STATS_PAYLOAD_SHA, "runtime-V2 stats drift")
    n2_doc = read_json(paths["n2_stats"])
    n2_verified = n2.verify_n2_document(n2_doc, source=runtime_doc)
    n2_file_sha = sha256_file(paths["n2_stats"])
    require(n2_file_sha == N2_STATS_SHA and n2_doc.get("statistics_payload_sha256") == N2_STATS_PAYLOAD_SHA, "N2 stats artifact drift")
    require(sha256_file(paths["modality"]) == MODALITY_SHA, "modality identity drift")
    require(sha256_file(paths["config"]) == CONFIG_SHA, "training config identity drift")
    groot_manifest = Path(args.groot_manifest).resolve()
    require(groot_manifest.is_file() and not groot_manifest.is_symlink(), "GR00T snapshot manifest unavailable")
    sys.path.insert(0, str(REPO / "scripts"))
    from stage45b_groot_source_snapshot import verify_extracted

    snapshot = verify_extracted(
        paths["groot"],
        groot_manifest,
        expected_manifest_file_sha256=GROOT_MANIFEST_SHA,
        expected_manifest_payload_sha256=GROOT_MANIFEST_PAYLOAD_SHA,
        expected_payload_tree_sha256=GROOT_PAYLOAD_TREE,
    )
    snapshot_manifest = read_json(groot_manifest)
    require(snapshot_manifest.get("origin_commit") == GROOT_COMMIT, "GR00T source commit drift")
    snapshot["origin_commit"] = GROOT_COMMIT
    require((paths["cosmos"] / "refs/main").read_text(encoding="utf-8").strip() == COSMOS_REVISION, "Cosmos revision drift")
    cosmos_tree = cosmos_digest(paths["cosmos"])
    require(cosmos_tree == COSMOS_TREE, f"Cosmos tree drift: {cosmos_tree}")
    return {
        "base_tree_sha256": base_tree,
        "full24": dataset,
        "runtime_stats": runtime,
        "runtime_stats_sha256": sha256_file(paths["runtime_stats"]),
        "runtime_stats_payload_sha256": RUNTIME_STATS_PAYLOAD_SHA,
        "n2_stats_sha256": n2_file_sha,
        "n2_stats_payload_sha256": N2_STATS_PAYLOAD_SHA,
        "n2_verification": n2_verified,
        "modality_sha256": MODALITY_SHA,
        "config_sha256": CONFIG_SHA,
        "groot_commit": GROOT_COMMIT,
        "groot_snapshot": snapshot,
        "groot_manifest_sha256": GROOT_MANIFEST_SHA,
        "cosmos_revision": COSMOS_REVISION,
        "cosmos_tree_sha256": cosmos_tree,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def observe_worktree(repo: Path, route: str) -> dict[str, Any]:
    repo = Path(repo).resolve()
    require(repo.is_dir() and not repo.is_symlink(), f"route worktree unavailable: {repo}")
    branch = git(repo, "branch", "--show-current")
    head = git(repo, "rev-parse", "HEAD")
    status = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    require(branch == ROUTE_BRANCHES[route], f"{route} branch drift: {branch}")
    require(head == ROUTE_HEADS[route], f"{route} HEAD drift: {head}")
    require(status == "", f"{route} worktree is dirty, including untracked files: {status}")
    return {"repo": str(repo), "branch": branch, "head": head, "clean": True}


def high_water(storage: Path, *, output: Path | None = None) -> dict[str, Any]:
    storage = Path(storage).resolve()
    require(storage.is_dir() and not storage.is_symlink(), f"storage root unavailable: {storage}")
    usage = shutil.disk_usage(storage)
    estimate = ESTIMATED_TRAINING_ROOT_BYTES * len(TRAINING_QUEUE)
    required = int(estimate * 1.15) + MIN_FREE_BYTES
    value = {
        "schema": "VANDA_STAGE45E_H200_UNIFIED_HIGH_WATER_V1",
        "status": "PASS" if usage.free >= required else "BLOCKED",
        "storage_root": str(storage),
        "free_bytes": usage.free,
        "estimated_training_bytes": estimate,
        "required_free_bytes_with_15_percent_margin_and_guard": required,
        "planned_training_roots": len(TRAINING_QUEUE),
        "estimated_root_bytes": ESTIMATED_TRAINING_ROOT_BYTES,
        "accepted_evidence_deleted": False,
        "free_margin_sufficient": usage.free >= required,
    }
    require(value["free_margin_sufficient"], f"storage high-water gate failed: {value}")
    if output is not None:
        write_json(Path(output), value)
    return value


def _historical_checkpoint(root: Path, step: int, key: str) -> dict[str, Any]:
    checkpoint = root / f"checkpoint-{step}"
    require(checkpoint.is_dir() and not checkpoint.is_symlink(), f"historical checkpoint unavailable: {checkpoint}")
    result = read_json(root / key)
    record = result.get(f"checkpoint_{step}", {})
    require(record.get("status", "PASS") in ("PASS", "COMPLETE", None), f"historical checkpoint status drift: {root}/{step}")
    require(record.get("tree_sha256"), f"historical checkpoint tree hash missing: {root}/{step}")
    return {"path": str(checkpoint), "tree_sha256": record["tree_sha256"], "global_step": step}


def _historical_training_args(root: Path) -> dict[str, Any]:
    """Read sealed TrainingArguments without constructing a model."""

    path = Path(root) / "training_args.bin"
    require(path.is_file() and not path.is_symlink(), f"historical training args unavailable: {path}")
    try:
        import torch

        try:
            args = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # compatibility with the older sealed runtime
            args = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"historical training args unreadable: {path}: {exc}") from exc
    scheduler = getattr(args, "lr_scheduler_type", "")
    fields = {
        "seed": getattr(args, "seed", None),
        "max_steps": getattr(args, "max_steps", None),
        "per_device_train_batch_size": getattr(args, "per_device_train_batch_size", None),
        "gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps", None),
        "learning_rate": getattr(args, "learning_rate", None),
        "weight_decay": getattr(args, "weight_decay", None),
        "lr_scheduler_type": str(getattr(scheduler, "value", scheduler)),
        "warmup_steps": getattr(args, "warmup_steps", None),
        "max_grad_norm": getattr(args, "max_grad_norm", None),
        "bf16": getattr(args, "bf16", None),
        "fp16": getattr(args, "fp16", None),
        "tf32": getattr(args, "tf32", None),
        "dataloader_num_workers": getattr(args, "dataloader_num_workers", None),
    }
    expected = {
        "seed": 2222,
        "max_steps": 2000,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 10,
        "max_grad_norm": 1.0,
        "bf16": True,
        "fp16": False,
        "tf32": True,
        "dataloader_num_workers": 0,
    }
    for key, value in expected.items():
        require(fields[key] == value, f"historical Legacy recipe drift in {key}: {fields[key]!r}")
    return {"path": str(path), "sha256": sha256_file(path), "fields": fields}


def verify_historical(root: Path, route: str) -> dict[str, Any]:
    root = Path(root).resolve()
    require(root.is_dir() and not root.is_symlink(), f"historical {route} root unavailable: {root}")
    if route == "legacy":
        result_path = root / "stage45b_training_result.json"
        result = read_json(result_path)
        training_args = _historical_training_args(root)
        require(training_args["fields"]["seed"] == 2222 and result.get("training_steps") == 2000, "historical Legacy seed2222 identity drift")
        gpu = result.get("gpu", {})
        gpu_name = str(gpu.get("gpu_name", gpu.get("name", "")))
        require("H200" in gpu_name.upper(), "historical Legacy seed2222 is not proven H200")
        preflight_path = root / "stage45b_runtime_preflight.json"
        preflight = read_json(preflight_path)
        base_tree = result.get("base_tree_sha256") or preflight.get("base_tree_sha256")
        full24_tree = result.get("full24_tree_sha256") or preflight.get("full24_tree_sha256")
        require(base_tree == BASE_TREE and full24_tree == FULL24_TREE, "historical Legacy source identity drift")
        stats_sha = result.get("runtime_stats_sha256") or result.get("structured_stats_sha256") or preflight.get("runtime_stats_sha256")
        stats_payload_sha = result.get("runtime_stats_payload_sha256") or result.get("structured_stats_payload_sha256") or preflight.get("runtime_stats_payload_sha256") or preflight.get("statistics_payload_sha256")
        require(stats_sha == RUNTIME_STATS_SHA and stats_payload_sha == RUNTIME_STATS_PAYLOAD_SHA and result.get("optimizer_steps") == 2000, "historical Legacy recipe/stats drift")
        return {"route": route, "seed": 2222, "root": str(root), "result_path": str(result_path), "result_sha256": sha256_file(result_path), "result": result, "training_args": training_args, "source_preflight": {"path": str(preflight_path), "sha256": sha256_file(preflight_path), "base_tree_sha256": base_tree, "full24_tree_sha256": full24_tree, "runtime_stats_sha256": stats_sha, "runtime_stats_payload_sha256": stats_payload_sha}, "checkpoints": {str(step): _historical_checkpoint(root, step, "stage45b_training_result.json") for step in (1000, 2000)}}
    result_path = root / "stage45e_l1_training_result.json"
    result = read_json(result_path)
    training_args = _historical_training_args(root)
    require(training_args["fields"]["seed"] == 2222 and result.get("training_steps") == 2000, "historical L1 seed2222 identity drift")
    gpu = result.get("gpu", {})
    gpu_name = str(gpu.get("gpu_name", gpu.get("name", "")))
    require("H200" in gpu_name.upper(), "historical L1 seed2222 is not proven H200")
    require(result.get("normalization_unchanged") is True and result.get("mask_unchanged") is True, "historical L1 contract drift")
    require(result.get("optimizer_steps") == 2000, "historical L1 recipe drift")
    return {"route": route, "seed": 2222, "root": str(root), "result_path": str(result_path), "result_sha256": sha256_file(result_path), "result": result, "training_args": training_args, "checkpoints": {str(step): _historical_checkpoint(root, step, "stage45e_l1_training_result.json") for step in (1000, 2000)}}


def _result_file(route: str, root: Path) -> Path:
    return root / ("stage45e_l1_training_result.json" if route == "l1" else "training_result.json")


def _training_pass(route: str, result: Mapping[str, Any], seed: int) -> bool:
    expected_status = "STAGE45E_L1_TRAINING_COMPLETE_FOR_EVAL" if route == "l1" else "PASS"
    observed_seed = result.get("training_seed") if route == "l1" else result.get("seed")
    observed_head = result.get("execution_head") if route == "l1" else result.get("repository_head")
    observed_branch = result.get("branch")
    input_audit = result.get("input_audit", {})
    base_tree = result.get("base_tree_sha256") or input_audit.get("base_tree_sha256")
    full24_tree = result.get("full24_tree_sha256") or input_audit.get("full24_tree_sha256")
    if route == "l1":
        stats_ok = result.get("structured_stats", {}).get("statistics_payload_sha256") == RUNTIME_STATS_PAYLOAD_SHA or result.get("structured_stats", {}).get("stats_payload_sha256") == RUNTIME_STATS_PAYLOAD_SHA or result.get("structured_stats", {}).get("payload_sha256") == RUNTIME_STATS_PAYLOAD_SHA
    elif route == "n2":
        stats_ok = result.get("n2_stats_sha256") == N2_STATS_SHA and result.get("n2_stats_payload_sha256") == N2_STATS_PAYLOAD_SHA
    else:
        stats_ok = result.get("runtime_stats_sha256") == RUNTIME_STATS_SHA and result.get("n2_statistics_override") is False
    return (
        result.get("status") == expected_status
        and observed_seed == seed
        and result.get("training_steps") == 2000
        and result.get("optimizer_steps") == 2000
        and observed_head == ROUTE_HEADS[route]
        and observed_branch == ROUTE_BRANCHES[route]
        and base_tree == BASE_TREE
        and full24_tree == FULL24_TREE
        and stats_ok
    )


def find_completed_training(storage: Path, route: str, seed: int) -> list[dict[str, Any]]:
    roots: list[dict[str, Any]] = []
    training = Path(storage) / "training"
    if not training.is_dir():
        return roots
    prefix = f"{route}_seed{seed}_"
    for root in sorted(training.iterdir(), key=lambda item: item.name):
        if not root.is_dir() or root.is_symlink() or not root.name.startswith(prefix):
            continue
        result_path = _result_file(route, root)
        if result_path.is_file() and not result_path.is_symlink():
            result = read_json(result_path)
            if _training_pass(route, result, seed):
                roots.append({"root": str(root), "result_path": str(result_path), "result_sha256": sha256_file(result_path), "result": result})
    return roots


def _incomplete_training_roots(storage: Path, route: str, seed: int) -> list[Path]:
    training = Path(storage) / "training"
    if not training.is_dir():
        return []
    prefix = f"{route}_seed{seed}_"
    return [root for root in sorted(training.iterdir(), key=lambda item: item.name) if root.is_dir() and not root.is_symlink() and root.name.startswith(prefix) and not _result_file(route, root).is_file()]


def _root_name(route: str, seed: int, job_id: str) -> str:
    safe_job = re.sub(r"[^A-Za-z0-9_.-]+", "_", job_id)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{route}_seed{seed}_{stamp}_{safe_job}"


def _common_route_args(args: argparse.Namespace) -> list[str]:
    names = ("base_model", "dataset", "groot", "groot_manifest", "cosmos", "modality", "config", "runtime_stats", "sealed_baseline", "snapshot_preflight")
    flags = {"base_model": "--base-model", "dataset": "--dataset", "groot": "--groot", "groot_manifest": "--groot-manifest", "cosmos": "--cosmos", "modality": "--modality", "config": "--config", "runtime_stats": "--runtime-stats", "sealed_baseline": "--sealed-baseline", "snapshot_preflight": "--snapshot-preflight"}
    result: list[str] = []
    for name in names:
        result.extend((flags[name], str(Path(getattr(args, name)).resolve())))
    return result


def _run_command(campaign_root: Path, name: str, command: Sequence[str], *, extra_env: Mapping[str, str] | None = None) -> None:
    logs = campaign_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    record_path = logs / f"{name}.command.json"
    stdout_path = logs / f"{name}.stdout.log"
    stderr_path = logs / f"{name}.stderr.log"
    write_json(record_path, {"schema": "VANDA_STAGE45E_UNIFIED_SUBPROCESS_RECORD_V1", "name": name, "command": [str(item) for item in command], "cwd": "/tmp", "pythonpath": "", "started_utc": now_utc()})
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": "", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "0"})
    if extra_env:
        environment.update(extra_env)
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.run([str(item) for item in command], cwd="/tmp", env=environment, stdout=stdout, stderr=stderr, check=False)
    finished = time.time()
    write_json(record_path, {"schema": "VANDA_STAGE45E_UNIFIED_SUBPROCESS_RECORD_V1", "name": name, "command": [str(item) for item in command], "cwd": "/tmp", "pythonpath": "", "started_utc": now_utc(), "finished_epoch": finished, "duration_s": finished - started, "returncode": process.returncode, "stdout": str(stdout_path), "stderr": str(stderr_path)}, overwrite=True)
    require(process.returncode == 0, f"subprocess failed ({name}); see {stderr_path}")


def _run_canary(args: argparse.Namespace, campaign_root: Path, gpu: Mapping[str, Any]) -> Path:
    route_repo = Path(args.legacy_repo).resolve()
    script = route_repo / "scripts/stage45e_legacy_matched_seed.py"
    output = campaign_root / "resource_canary"
    command = [str(PYTHON), str(script), "--mode", "canary", *_common_route_args(args), "--gpu-uuid", str(gpu["rows"][0]["uuid"]), "--output-root", str(output)]
    _run_command(campaign_root, "resource_canary", command)
    result_path = output / "legacy_resource_canary.json"
    result = read_json(result_path)
    require(result.get("status") == "PASS" and result.get("weight_update") == 0 and result.get("optimizer_steps") == 0, "resource canary did not pass as a zero-update H200 canary")
    return result_path


def _training_command(args: argparse.Namespace, route: str, seed: int, output: Path, canary: Path) -> list[str]:
    repo = Path(getattr(args, f"{route}_repo")).resolve()
    if route == "legacy":
        return [str(PYTHON), str(repo / "scripts/stage45e_legacy_matched_seed.py"), "--mode", "train", *_common_route_args(args), "--seed", str(seed), "--canary", str(canary), "--gpu-uuid", str(args.gpu_uuid), "--output-root", str(output)]
    if route == "n2":
        return [str(PYTHON), str(repo / "scripts/stage45e_n2_gpu_sprint.py"), "--mode", "train", *_common_route_args(args), "--n2-stats", str(Path(args.n2_stats).resolve()), "--seed", str(seed), "--canary", str(canary), "--gpu-uuid", str(args.gpu_uuid), "--output-root", str(output)]
    return [str(PYTHON), str(repo / "scripts/stage45e_l1_h200_train.py"), "--execute", "--seed", str(seed), "--expected-head", ROUTE_HEADS[route], "--output-root", str(output), "--python", str(PYTHON), "--base-model", str(Path(args.base_model).resolve()), "--dataset", str(Path(args.dataset).resolve()), "--groot-source", str(Path(args.groot).resolve()), "--groot-manifest", str(Path(args.groot_manifest).resolve()), "--cosmos-root", str(Path(args.cosmos).resolve()), "--modality", str(Path(args.modality).resolve()), "--config", str(Path(args.config).resolve()), "--stats", str(Path(args.runtime_stats).resolve()), "--release", str(Path(args.release).resolve()), "--video-qualification", str(Path(args.video_qualification).resolve()), "--ffmpeg-root", str(Path(args.ffmpeg_root).resolve())]


def _training_result(route: str, root: Path, seed: int) -> dict[str, Any]:
    path = _result_file(route, root)
    result = read_json(path)
    require(_training_pass(route, result, seed), f"{route} seed{seed} did not produce a complete exact 2000-step lineage")
    return {"route": route, "seed": seed, "root": str(root), "result_path": str(path), "result_sha256": sha256_file(path), "result": result}


def _bind_lineages(args: argparse.Namespace, storage: Path, campaign_root: Path) -> tuple[dict[str, dict[int, dict[str, Any]]], list[dict[str, Any]]]:
    lineages: dict[str, dict[int, dict[str, Any]]] = {route: {} for route in ROUTES}
    plan: list[dict[str, Any]] = []
    for route in ROUTES:
        for seed in ALL_SEEDS:
            if route == "legacy" and seed == 2222:
                lineages[route][seed] = verify_historical(Path(args.legacy_history), route)
                continue
            if route == "l1" and seed == 2222:
                lineages[route][seed] = verify_historical(Path(args.l1_history), route)
                continue
            completed = find_completed_training(storage, route, seed)
            require(len(completed) <= 1, f"multiple complete {route} seed{seed} lineages exist; refusing performance-based selection")
            if completed:
                lineages[route][seed] = _training_result(route, Path(completed[0]["root"]), seed)
                plan.append({"route": route, "seed": seed, "action": "REUSE_COMPLETE_READ_ONLY", "root": completed[0]["root"], "result_sha256": completed[0]["result_sha256"]})
                continue
            incomplete = _incomplete_training_roots(storage, route, seed)
            require(not incomplete, f"incomplete prior {route} seed{seed} lineage exists; preserve and audit before any retry: {incomplete}")
            root = storage / "training" / _root_name(route, seed, str(os.environ["PBS_JOBID"]))
            require(not root.exists(), f"new training root unexpectedly exists: {root}")
            lineages[route][seed] = {"route": route, "seed": seed, "root": str(root), "pending": True}
            plan.append({"route": route, "seed": seed, "action": "FRESH_BASE_2000", "root": str(root)})
    write_json(campaign_root / "lineage_plan.json", {"schema": "VANDA_STAGE45E_UNIFIED_LINEAGE_PLAN_V1", "status": "BOUND_BEFORE_TRAINING", "queue": plan, "historical_reuse": {route: lineages[route].get(2222, {}).get("root") for route in ROUTES}, "timestamp_utc": now_utc()})
    return lineages, plan


def _checkpoint_spec(route: str, seed: int, step: int, lineage: Mapping[str, Any]) -> dict[str, Any]:
    result = lineage.get("result", {})
    record = result.get(f"checkpoint_{step}", {})
    tree = record.get("tree_sha256") or lineage.get("checkpoints", {}).get(str(step), {}).get("tree_sha256")
    require(tree, f"checkpoint tree hash missing for {route} seed{seed} step{step}")
    path = Path(lineage["root"]) / f"checkpoint-{step}"
    return {"label": f"{route.upper()}(seed{seed})@{step}", "route": route, "seed": seed, "step": step, "path": str(path), "tree_sha256": tree}


def build_phase_specs(lineages: Mapping[str, Mapping[int, Mapping[str, Any]]], campaign_root: Path) -> dict[str, dict[str, Path]]:
    # Legacy seed2222 already has an accepted D1/D2A artifact and is reused
    # read-only in CPU closure.  It is intentionally absent from current
    # model-call specs; L1 seed2222 is also reused only for its own arm.
    all_specs = {
        "legacy": [_checkpoint_spec("legacy", seed, step, lineages["legacy"][seed]) for seed in (2223, 2224) for step in (1000, 2000)],
        "l1": [_checkpoint_spec("l1", seed, step, lineages["l1"][seed]) for seed in ALL_SEEDS for step in (1000, 2000)],
        "n2": [_checkpoint_spec("n2", seed, step, lineages["n2"][seed]) for seed in ALL_SEEDS for step in (1000, 2000)],
    }
    e4_specs = {route: [item for item in all_specs[route] if not (route == "l1" and item["seed"] == 2222)] for route in ROUTES}
    # E4 therefore has 4 LEGACY + 4 L1 + 6 N2 = 14 new checkpoint arms;
    # D1/D2A have 4 LEGACY + 6 L1 + 6 N2 = 16 active arms.
    phase_specs: dict[str, dict[str, Path]] = {"E4": {}, "D1": {}, "D2A": {}}
    for phase, selected in (("E4", e4_specs), ("D1", all_specs), ("D2A", all_specs)):
        for route in ROUTES:
            path = campaign_root / "specs" / f"{phase}_{route}.json"
            budget = {
                "arms": len(selected[route]),
                "conditioning_passes": len(selected[route]) * 24,
                "action_head_calls": len(selected[route]) * 24 * (4 if phase == "E4" else 9 if phase == "D1" else 8),
                "raw_rows": len(selected[route]) * 24 * (1 if phase == "E4" else 9 if phase == "D1" else 4),
            }
            write_json(path, {"schema": "VANDA_STAGE45E_UNIFIED_PHASE_SPEC_V1", "status": "SEALED_INPUT_SPEC", "route": route, "phase": phase, "checkpoints": selected[route], "budget": budget, "timestamp_utc": now_utc()})
            phase_specs[phase][route] = path
    return phase_specs


def _expected_phase_totals(phase: str) -> dict[str, dict[str, int]]:
    if phase == "E4":
        return {"legacy": {"arms": 4, "conditioning_passes": 96, "action_head_calls": 384, "raw_rows": 96}, "l1": {"arms": 4, "conditioning_passes": 96, "action_head_calls": 384, "raw_rows": 96}, "n2": {"arms": 6, "conditioning_passes": 144, "action_head_calls": 576, "raw_rows": 144}}
    if phase == "D1":
        return {"legacy": {"arms": 4, "conditioning_passes": 96, "action_head_calls": 864, "raw_rows": 864}, "l1": {"arms": 6, "conditioning_passes": 144, "action_head_calls": 1296, "raw_rows": 1296}, "n2": {"arms": 6, "conditioning_passes": 144, "action_head_calls": 1296, "raw_rows": 1296}}
    return {"legacy": {"arms": 4, "conditioning_passes": 96, "action_head_calls": 768, "raw_rows": 384}, "l1": {"arms": 6, "conditioning_passes": 144, "action_head_calls": 1152, "raw_rows": 576}, "n2": {"arms": 6, "conditioning_passes": 144, "action_head_calls": 1152, "raw_rows": 576}}


def run_diagnostics(args: argparse.Namespace, campaign_root: Path, admission_path: Path, phase_specs: Mapping[str, Mapping[str, Path]]) -> dict[str, dict[str, Path]]:
    diagnostic_script = REPO / "scripts/stage45e_h200_unified_diagnostics.py"
    helper_script = Path(args.legacy_repo).resolve() / "scripts/stage45e_legacy_matched_seed.py"
    phase_roots: dict[str, dict[str, Path]] = {"E4": {}, "D1": {}, "D2A": {}}
    for phase in ("E4", "D1", "D2A"):
        for route in ROUTES:
            output = Path(args.storage_root).resolve() / "diagnostics" / f"{campaign_root.name}_{phase}_{route}"
            stats = Path(args.n2_stats).resolve() if route == "n2" else Path(args.runtime_stats).resolve()
            command = [str(PYTHON), str(diagnostic_script), "--mode", "run", "--route", route, "--phase", phase, "--route-repo", str(Path(getattr(args, f"{route}_repo")).resolve()), "--helper-script", str(helper_script), "--admission", str(admission_path), "--output-root", str(output), "--base-model", str(Path(args.base_model).resolve()), "--dataset", str(Path(args.dataset).resolve()), "--groot", str(Path(args.groot).resolve()), "--cosmos", str(Path(args.cosmos).resolve()), "--modality", str(Path(args.modality).resolve()), "--config", str(Path(args.config).resolve()), "--stats", str(stats), "--sealed-baseline", str(Path(args.sealed_baseline).resolve()), "--snapshot-preflight", str(Path(args.snapshot_preflight).resolve())]
            _run_command(campaign_root, f"diagnostic_{phase}_{route}", command)
            seal_path = output / "unified_phase_seal.json"
            require(seal_path.is_file(), f"missing unified diagnostic seal: {seal_path}")
            seal = read_json(seal_path)
            expected = _expected_phase_totals(phase)[route]
            observed = seal.get("budget", {})
            require(observed == expected, f"{phase}/{route} budget mismatch: {observed} != {expected}")
            phase_roots[phase][route] = output
    return phase_roots


def _collect_files(root: Path, *, excluded: set[str]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name in excluded:
            continue
        files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return files


def cpu_closure(args: argparse.Namespace, campaign_root: Path, phase_roots: Mapping[str, Mapping[str, Path]], lineages: Mapping[str, Mapping[int, Mapping[str, Any]]], source_identities: Mapping[str, Any], highwater_record: Mapping[str, Any]) -> dict[str, Any]:
    expected = {phase: _expected_phase_totals(phase) for phase in ("E4", "D1", "D2A")}
    phase_seals: dict[str, dict[str, Any]] = {}
    for phase in ("E4", "D1", "D2A"):
        phase_seals[phase] = {}
        for route in ROUTES:
            seal_path = Path(phase_roots[phase][route]) / "unified_phase_seal.json"
            seal = read_json(seal_path)
            require(seal.get("status") == "PASS" and seal.get("budget") == expected[phase][route], f"CPU closure phase seal mismatch: {seal_path}")
            inner = read_json(Path(seal["inner_seal"]))
            require(inner.get("status") == "PASS", f"inner phase seal is not PASS: {seal_path}")
            phase_seals[phase][route] = {"path": str(seal_path), "sha256": sha256_file(seal_path), "route": route, "phase": phase, "budget": seal["budget"], "inner_seal_sha256": seal["inner_seal_sha256"]}
    totals = {}
    for phase in ("E4", "D1", "D2A"):
        totals[phase] = {key: sum(expected[phase][route][key] for route in ROUTES) for key in ("conditioning_passes", "action_head_calls", "raw_rows")}
        totals[phase]["total_model_side_calls"] = totals[phase]["conditioning_passes"] + totals[phase]["action_head_calls"]
    require(totals["E4"]["total_model_side_calls"] == 1680 and totals["D1"]["total_model_side_calls"] == 3840 and totals["D2A"]["total_model_side_calls"] == 3456, f"CPU total budget drift: {totals}")
    summary = {
        "schema": "VANDA_STAGE45E_H200_UNIFIED_CPU_EVIDENCE_SUMMARY_V1",
        "status": "PASS_READONLY_FROM_SEALED_GPU_EVIDENCE",
        "pi_gate": "PI_GATE_STATE_0133",
        "campaign_root": str(campaign_root),
        "source_identities": source_identities,
        "high_water": highwater_record,
        "phase_totals": totals,
        "phase_seals": phase_seals,
        "lineage_provenance": {route: {str(seed): {key: value for key, value in item.items() if key in ("route", "seed", "root", "result_path", "result_sha256", "pending")} for seed, item in lineages[route].items()} for route in ROUTES},
        "historical_reuse": {"legacy_seed2222": str(Path(args.legacy_history).resolve()), "l1_seed2222": str(Path(args.l1_history).resolve())},
        "descriptive_only": True,
        "no_performance_based_winner": True,
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
    write_json(campaign_root / "cpu_summary.json", summary)
    inventory = _collect_files(campaign_root, excluded={"CPU_EVIDENCE_INVENTORY.json", "SHA256SUMS", "final_result.json"})
    write_json(campaign_root / "CPU_EVIDENCE_INVENTORY.json", {"schema": "VANDA_STAGE45E_H200_CPU_EVIDENCE_INVENTORY_V1", "status": "PASS", "entries": inventory, "entry_count": len(inventory)})
    inventory_sha = sha256_file(campaign_root / "CPU_EVIDENCE_INVENTORY.json")
    sums_path = campaign_root / "SHA256SUMS"
    with sums_path.open("x", encoding="utf-8") as handle:
        for entry in inventory:
            handle.write(f"{entry['sha256']}  {entry['path']}\n")
        handle.write(f"{inventory_sha}  CPU_EVIDENCE_INVENTORY.json\n")
        handle.flush()
        os.fsync(handle.fileno())
    final = {
        "schema": "VANDA_STAGE45E_H200_UNIFIED_THREE_ROUTE_REPLICATION_RESULT_V1",
        "status": "PASS",
        "token": "STAGE45E_H200_UNIFIED_THREE_ROUTE_REPLICATION_COMPLETE_FOR_PI_REVIEW",
        "pi_gate": "PI_GATE_STATE_0133",
        "campaign_root": str(campaign_root),
        "cpu_summary": str(campaign_root / "cpu_summary.json"),
        "cpu_summary_sha256": sha256_file(campaign_root / "cpu_summary.json"),
        "cpu_inventory": str(campaign_root / "CPU_EVIDENCE_INVENTORY.json"),
        "cpu_inventory_sha256": inventory_sha,
        "sha256sums": str(sums_path),
        "phase_totals": totals,
        "total_model_side_calls": 8976,
        "fresh_training_lineages": 7,
        "historical_legacy_seed2222_reused": True,
        "historical_l1_seed2222_reused": True,
        "training": False,
        "Validation-E": 0,
        "Holdout-F": 0,
        "WBC": 0,
        "simulator": 0,
        "hardware": 0,
        "merge": False,
        "timestamp_utc": now_utc(),
    }
    write_json(campaign_root / "final_result.json", final)
    return final


def campaign(args: argparse.Namespace) -> dict[str, Any]:
    require(os.environ.get("PBS_JOBID"), "campaign must run inside a PBS allocation")
    storage = Path(args.storage_root).resolve()
    storage.mkdir(parents=True, exist_ok=True)
    for name in ("training", "diagnostics", "failures", "provenance", "campaigns"):
        (storage / name).mkdir(parents=True, exist_ok=True)
    job_id = os.environ["PBS_JOBID"]
    campaign_root = storage / "campaigns" / f"pi0133_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', job_id)}"
    campaign_root.mkdir(parents=True, exist_ok=False)
    gpu = _gpu_observation()
    highwater_record = high_water(storage, output=campaign_root / "high_water.json")
    orchestration_branch = git(REPO, "branch", "--show-current")
    orchestration_head = git(REPO, "rev-parse", "HEAD")
    require(orchestration_branch == "stage45e/h200-unified-replication-campaign", f"orchestration branch drift: {orchestration_branch}")
    require(git(REPO, "status", "--porcelain=v1", "--untracked-files=all") == "", "orchestration checkout is dirty")
    source_identities = verify_sources(args, Path(args.n2_repo).resolve())
    route_worktrees = {route: observe_worktree(Path(getattr(args, f"{route}_repo")), route) for route in ROUTES}
    historical = {"legacy": verify_historical(Path(args.legacy_history), "legacy"), "l1": verify_historical(Path(args.l1_history), "l1")}
    phase_spec_paths = {phase: {route: str(campaign_root / "specs" / f"{phase}_{route}.json") for route in ROUTES} for phase in ("E4", "D1", "D2A")}
    admission = {
        "schema": "VANDA_STAGE45E_H200_UNIFIED_CAMPAIGN_ADMISSION_V1",
        "status": "PASS",
        "pi_gate": "PI_GATE_STATE_0133",
        "issue_comment": 5642852068,
        "orchestration_head": orchestration_head,
        "orchestration_branch": orchestration_branch,
        "pbs_jobid": job_id,
        "gpu": gpu,
        "source_identities": source_identities,
        "route_worktrees": route_worktrees,
        "historical_lineages": historical,
        "phase_specs": phase_spec_paths,
        "storage_root": str(storage),
        "high_water": highwater_record,
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
        "timestamp_utc": now_utc(),
    }
    write_json(campaign_root / "campaign_admission.json", admission)
    lineages, plan = _bind_lineages(args, storage, campaign_root)
    canary = _run_canary(args, campaign_root, gpu)
    args.canary = canary
    args.gpu_uuid = gpu["rows"][0]["uuid"]
    append_jsonl(campaign_root / "campaign_events.jsonl", {"event": "H200_RESOURCE_CANARY_PASS", "path": str(canary), "timestamp_utc": now_utc()})
    for item in plan:
        if item["action"] != "FRESH_BASE_2000":
            append_jsonl(campaign_root / "campaign_events.jsonl", {"event": "LINEAGE_REUSED_READ_ONLY", **item, "timestamp_utc": now_utc()})
            continue
        route, seed = item["route"], int(item["seed"])
        root = Path(item["root"])
        command = _training_command(args, route, seed, root, canary)
        _run_command(campaign_root, f"train_{route}_seed{seed}", command, extra_env={"STAGE45E_L1_AUTHORITY_TOKEN": "PI_GATE_STATE_0133"})
        lineages[route][seed] = _training_result(route, root, seed)
        append_jsonl(campaign_root / "campaign_events.jsonl", {"event": "TRAINING_LINEAGE_PASS", "route": route, "seed": seed, "root": str(root), "result_sha256": lineages[route][seed]["result_sha256"], "timestamp_utc": now_utc()})
    phase_specs = build_phase_specs(lineages, campaign_root)
    append_jsonl(campaign_root / "campaign_events.jsonl", {"event": "PHASE_SPECS_SEALED", "timestamp_utc": now_utc()})
    phase_roots = run_diagnostics(args, campaign_root, campaign_root / "campaign_admission.json", phase_specs)
    append_jsonl(campaign_root / "campaign_events.jsonl", {"event": "E4_D1_D2A_SEALED", "timestamp_utc": now_utc()})
    final = cpu_closure(args, campaign_root, phase_roots, lineages, source_identities, highwater_record)
    write_json(storage / "provenance" / f"{campaign_root.name}_final_result.json", final)
    return {"status": "PASS", "token": final["token"], "campaign_root": str(campaign_root), "final_result": str(campaign_root / "final_result.json"), "final_result_sha256": sha256_file(campaign_root / "final_result.json"), "orchestration_head": orchestration_head, "route_heads": ROUTE_HEADS, "phase_totals": final["phase_totals"]}


def audit_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Verify all late-bound inputs without observing or constructing a GPU model."""

    storage = Path(args.storage_root).resolve()
    source_identities = verify_sources(args, Path(args.n2_repo).resolve())
    route_worktrees = {route: observe_worktree(Path(getattr(args, f"{route}_repo")), route) for route in ROUTES}
    historical = {"legacy": verify_historical(Path(args.legacy_history), "legacy"), "l1": verify_historical(Path(args.l1_history), "l1")}
    supplemental = {
        "groot_manifest": Path(args.groot_manifest).resolve(),
        "release": Path(args.release).resolve(),
        "video_qualification": Path(args.video_qualification).resolve(),
        "ffmpeg_root": Path(args.ffmpeg_root).resolve(),
    }
    for name, path in supplemental.items():
        require(path.exists() and not path.is_symlink(), f"supplemental path unavailable: {name}: {path}")
    highwater_record = high_water(storage)
    return {
        "schema": "VANDA_STAGE45E_H200_UNIFIED_INPUT_AUDIT_V1",
        "status": "PASS_CPU_ONLY",
        "source_identities": source_identities,
        "route_worktrees": route_worktrees,
        "historical_lineages": historical,
        "supplemental_paths": {key: str(value) for key, value in supplemental.items()},
        "high_water": highwater_record,
        "gpu": 0,
        "model_imported": False,
        "model_construction": 0,
        "model_forward": 0,
        "training": 0,
        "backward": 0,
        "optimizer_steps": 0,
        "Validation-E": 0,
        "Holdout-F": 0,
        "WBC": 0,
        "simulator": 0,
        "hardware": 0,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("static", "high-water", "audit-inputs", "campaign"), default="static")
    p.add_argument("--storage-root", type=Path, default=STORAGE_ROOT)
    p.add_argument("--n2-repo", type=Path, default=WORKTREE_ROOT / "n2")
    p.add_argument("--legacy-repo", type=Path, default=WORKTREE_ROOT / "legacy")
    p.add_argument("--l1-repo", type=Path, default=WORKTREE_ROOT / "l1")
    for name, default in DEFAULTS.items():
        p.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, default=default)
    p.add_argument("--groot-manifest", type=Path, default=Path("/scratch/e1192548/stage45b_pi0086_groot_snapshot__zhu4zz2/STAGE45B_GROOT_SOURCE_SNAPSHOT_V2.manifest.json"))
    p.add_argument("--release", type=Path, default=Path("/scratch/e1192548/stage45b_pi0087_admission_IWmsW5/STAGE45B_B2_TRAINING_RELEASE_V2.json"))
    p.add_argument("--video-qualification", type=Path, default=Path("/scratch/e1192548/stage45b_pi0087_final_video_qual_DXDgKu/VANDA_STAGE45B_PI0087_FINAL_ADMISSION_QUALIFIED.json"))
    p.add_argument("--ffmpeg-root", type=Path, default=Path("/scratch/e1192548/STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1__9243a7880b06ac3984ddf4a1a667b685441ec0f34d6d529fa0f96c3c894e10cf"))
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.mode == "static":
            result = static_audit()
        elif args.mode == "high-water":
            result = high_water(Path(args.storage_root))
        elif args.mode == "audit-inputs":
            result = audit_inputs(args)
        else:
            result = campaign(args)
        print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)
        return 0
    except Exception as exc:
        print(f"STAGE45E_H200_UNIFIED_REPLICATION_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
