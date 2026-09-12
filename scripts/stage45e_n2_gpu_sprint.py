#!/usr/bin/env python3
"""Vanda Stage45E N2 H200 matched-seed replication runner.

The static entry point is dependency-light and checks the PI0133 manifest,
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
BRANCH = "stage45e/n2-h200-matched-seed-replication"
BASE_HEAD = "d762e731d2c11ceea3c001c37ac27890fe6f4e70"
PI_GATE = "PI_GATE_STATE_0133"
PI_COMMENT = 5642852068
BASE_TREE = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
FULL24_TREE = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
RUNTIME_STATS_SHA = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
RUNTIME_STATS_PAYLOAD_SHA = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
# PI0133 materialized the accepted N2 statistics document at the unified
# Hopper storage path.  Its outer-document identity is distinct from the
# superseded PI0122 staging file, while the immutable statistics payload is
# unchanged and remains bound below.
N2_STATS_SHA = "e88ff3ec5626130ca8a08006abe1ada70d34424af4b0d3bacf64db57149d7677"
N2_STATS_PAYLOAD_SHA = "508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572"
GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
GROOT_MANIFEST_SHA = "96b883e56b54f763bec60d3a336632ad531f4fc52d099f803e5f9efba60eab11"
GROOT_MANIFEST_PAYLOAD_SHA = "a9f10b837b6086137fa519cc148275a25bf6a5f98d58df8e3c50e0a43e5b8250"
GROOT_PAYLOAD_TREE = "c0d57a6a74ab24c48abdd1c13420e9d1c893dc0d7b5ca370fa9b6f7af32f5f1b"
DEFAULT_GROOT_MANIFEST = Path("/scratch/e1192548/stage45b_pi0086_groot_snapshot__zhu4zz2/STAGE45B_GROOT_SOURCE_SNAPSHOT_V2.manifest.json")
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
MODALITY_SHA = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
CONFIG_SHA = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
PREFERRED_GPU_UUID = None
MIN_H200_VRAM_MIB = 80 * 1024
HORIZON = 40
UNIVERSAL_DIM = 132
RAW_SHAPE = (1, HORIZON, UNIVERSAL_DIM)
T16 = (0, 250, 500, 750)
DT = 0.25
TRAIN_STEPS = 2000
G2_LABELS = ("R1@1000", "R1@2000", "R2@1000", "R2@2000")
G3_LABELS = ("FULL24@2000", "N2(seed2222)@2000", "N2(seed2223)@2000", "N2(seed2224)@2000")
G4_LABELS = G3_LABELS
FULL24_2000_TREE = "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5"
N2_SEEDS = (2222, 2223, 2224)
G3_T_VALUES = (0.06776241796562847, 0.36966943557751086, 0.7837719744658148)
G3_T_BUCKETS = (67, 369, 783)
G4_T_VALUES = (0.0, 0.25, 0.5, 0.75)
G4_T_BUCKETS = (0, 250, 500, 750)
G5_D1_LABELS = ("N2(seed2222)@1000", "N2(seed2223)@1000", "N2(seed2224)@1000")
G5_D1_T_VALUES = G3_T_VALUES
G5_D1_T_BUCKETS = G3_T_BUCKETS
G5_D2A_LABELS = G5_D1_LABELS
G5_D2A_T_VALUES = G4_T_VALUES
G5_D2A_T_BUCKETS = G4_T_BUCKETS

_BOUNDARY: dict[str, Any] = {}
# Runtime processor construction registers the embodiment globally inside the
# GR00T package.  The PI0130 continuation is intentionally a single process,
# so rebuilding that processor for G4/G5 raises a duplicate-registration
# assertion even though the scientific inputs are unchanged.  Cache the
# sealed, read-only context by exact path identity and reuse it across phases.
# This is a lifecycle repair only: models are still constructed and released
# independently for each checkpoint/phase, and no inference is performed by
# the cache itself.
_SEALED_CONTEXT_CACHE: dict[tuple[str, ...], tuple[Any, Any, Any, Any, list[dict[str, Any]], Mapping[str, Any], Any]] = {}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sync_failure_boundary(counters: Mapping[str, Any], *, phase: str | None = None) -> None:
    """Mirror live phase counters into the outer failure seal.

    A continuation can fail after model construction or after several action
    calls.  Keeping this boundary synchronized prevents the top-level seal
    from falsely reporting an all-zero execution after inner work started.
    """

    global _BOUNDARY
    _BOUNDARY = {
        "phase": phase if phase is not None else counters.get("_phase"),
        "gpu": bool(counters.get("gpu", False)),
        "model_constructed": int(counters.get("model_construction", 0)),
        "gr00t_forwards": int(counters.get("model_forward", 0)),
        "training_steps": int(counters.get("training_steps", 0)),
        "optimizer_steps": int(counters.get("optimizer_steps", 0)),
        "backward_steps": int(counters.get("backward_steps", 0)),
        "checkpoint_writes": int(counters.get("checkpoint_writes", 0)),
        "PBS": int(counters.get("PBS", 0)),
        "Validation-E": int(counters.get("Validation-E", 0)),
        "Holdout-F": int(counters.get("Holdout-F", 0)),
        "WBC": int(counters.get("WBC", 0)),
        "simulator": int(counters.get("simulator", 0)),
        "hardware": int(counters.get("hardware", 0)),
    }


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


def _json_ready(value: Any) -> Any:
    """Normalize mapping keys before deterministic JSON sorting.

    The accepted metric aggregator may use ``None`` as a sentinel key (for
    example, an unset safety decision).  JSON itself stringifies such keys,
    but ``sort_keys=True`` cannot order ``None`` alongside strings.  Convert
    keys recursively at the evidence boundary only; numeric/list values and
    metric semantics remain untouched.
    """

    if isinstance(value, Mapping):
        return {("null" if key is None else str(key)): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path = Path(path)
    require(overwrite or not path.exists(), f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
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
    require(value.get("schema") == "CORELAB_STAGE45E_N2_H200_MATCHED_SEED_REPLICATION_PI0133_V1", "H200 manifest schema drift")
    return value


def static_audit() -> dict[str, Any]:
    manifest = _manifest()
    require(manifest["authority"]["pi_gate"] == PI_GATE and manifest["authority"]["issue_comment"] == PI_COMMENT, "PI0133 authority drift")
    require(manifest["repository"]["branch"] == BRANCH and manifest["repository"]["base_head"] == BASE_HEAD, "sprint branch/base drift")
    require(manifest["frozen_identities"]["base_tree_sha256"] == BASE_TREE, "BASE identity drift")
    require(manifest["frozen_identities"]["full24_tree_sha256"] == FULL24_TREE, "FULL24 identity drift")
    require(manifest["n2_intervention"]["xyz_common_half_range_m"] == n2.HALF_RANGE_M, "N2 half-range drift")
    require(manifest["n2_intervention"]["mask_active_scalars"] == 190, "T16 mask drift")
    require(manifest["continuation"]["order"] == ["H200_ADMISSION", "FRESH_BASE_SEEDS", "N2_E4", "N2_D1", "N2_D2A", "CPU_CLOSURE"], "H200 continuation order drift")
    require(set(manifest["continuation"]["forbidden_phases"]) >= {"Validation-E", "Holdout-F", "WBC", "simulator", "hardware", "D2B"}, "H200 continuation forbidden-phase drift")
    require(manifest["training"]["seeds"] == [2222, 2223, 2224] and manifest["training"]["max_steps"] == 2000, "H200 replication training budget drift")
    require(manifest["evaluation"]["arms"] == 6, "N2 E4 arm count drift")
    require(manifest["evaluation"]["conditioning_passes"] == 144, "N2 E4 conditioning budget drift")
    require(manifest["evaluation"]["action_head_calls"] == 576, "N2 E4 action-head budget drift")
    require(manifest["evaluation"]["total_model_side_calls"] == 720, "N2 E4 budget drift")
    require(manifest["diagnostics"]["d1"]["conditioning_passes"] == 144, "N2 D1 conditioning budget drift")
    require(manifest["diagnostics"]["d1"]["action_head_calls"] == 1296, "N2 D1 action-head budget drift")
    require(manifest["diagnostics"]["d1"]["total_model_side_calls"] == 1440, "N2 D1 budget drift")
    require(manifest["diagnostics"]["d2a"]["conditioning_passes"] == 144, "N2 D2A conditioning budget drift")
    require(manifest["diagnostics"]["d2a"]["action_head_calls"] == 1152, "N2 D2A action-head budget drift")
    require(manifest["diagnostics"]["d2a"]["total_model_side_calls"] == 1296, "N2 D2A budget drift")
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
        "schema": "CORELAB_STAGE45E_N2_H200_MATCHED_SEED_STATIC_AUDIT_V1",
        "status": "PASS" if not forbidden_imports else "BLOCKED",
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "top_level_forbidden_imports": forbidden_imports,
        "branch": BRANCH,
        "base_head": BASE_HEAD,
        "training_seeds": list(manifest["training"]["seeds"]),
        "e4_calls": 720,
        "d1_calls": 1440,
        "d2a_calls": 1296,
        "continuation_order": manifest["continuation"]["order"],
        "forbidden_continuation_phases": manifest["continuation"]["forbidden_phases"],
        "execution_boundary": {"model_construction": 0, "gr00t_forwards": 0, "training_steps": 0, "optimizer_steps": 0, "gpu": False, "PBS": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0},
    }


def _paths(args: argparse.Namespace, *, include_legacy: bool = False) -> dict[str, Path]:
    names = ("base_model", "dataset", "groot", "groot_manifest", "cosmos", "modality", "config", "runtime_stats", "n2_stats", "sealed_baseline", "snapshot_preflight")
    if include_legacy:
        names += ("legacy_full24_2000", "legacy_full24_1000", "structured_1000")
    result: dict[str, Path] = {}
    for name in names:
        value = getattr(args, name, None)
        if value is None:
            if name == "groot_manifest":
                result[name] = DEFAULT_GROOT_MANIFEST.resolve()
                continue
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


def gpu_identity(expected_uuid: str | None = PREFERRED_GPU_UUID) -> dict[str, Any]:
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    require(query.returncode == 0, f"nvidia-smi failed: {query.stderr[-500:]}")
    rows = []
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        require(len(fields) == 5, f"unexpected nvidia-smi row: {line!r}")
        rows.append({"index": fields[0], "name": fields[1], "uuid": fields[2], "memory_total_mib": int(fields[3]), "memory_used_mib_at_start": int(fields[4])})
    require(len(rows) == 1 and rows[0]["index"] == "0", f"PI0133 requires exactly one visible H200 at index 0: {rows!r}")
    selected = [item for item in rows if expected_uuid is None or item["uuid"] == expected_uuid]
    require(len(selected) == 1, f"requested H200 UUID is not unique: {expected_uuid}; observed={rows!r}")
    value = selected[0]
    require("H200" in value["name"].upper() and value["memory_total_mib"] >= MIN_H200_VRAM_MIB, f"H200 identity/capacity not observed: {value!r}")
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
    from stage45b_groot_source_snapshot import verify_extracted
    groot_snapshot = verify_extracted(
        paths["groot"], paths["groot_manifest"],
        expected_manifest_file_sha256=GROOT_MANIFEST_SHA,
        expected_manifest_payload_sha256=GROOT_MANIFEST_PAYLOAD_SHA,
        expected_payload_tree_sha256=GROOT_PAYLOAD_TREE,
    )
    groot_manifest = read_json(paths["groot_manifest"])
    require(groot_manifest.get("origin_commit") == GROOT_COMMIT, "GR00T source commit mismatch")
    groot_snapshot["origin_commit"] = GROOT_COMMIT
    refs = paths["cosmos"] / "refs/main"
    require(refs.is_file() and not refs.is_symlink() and refs.read_text(encoding="utf-8").strip() == COSMOS_REVISION, "Cosmos revision mismatch")
    return {"base_tree_sha256": base_tree, "full24": dataset, "runtime_stats": runtime_check, "n2_verification": verified_n2, "n2_stats_sha256": sha256_file(paths["n2_stats"]), "groot_commit": GROOT_COMMIT, "groot_snapshot": groot_snapshot, "groot_manifest_sha256": GROOT_MANIFEST_SHA, "cosmos_revision": COSMOS_REVISION}


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
    missing = [item for item in required if not (path / item).is_file() or (path / item).is_symlink()]
    entries = tree_entries(path, f"stage45e/{path.name}")
    tree = tree_digest(entries)
    return {"path": str(path), "global_step": state.get("global_step"), "expected_step": step, "required_state_files_present": not missing, "missing_state_files": missing, "tree_sha256": tree, "file_count": len(entries), "total_bytes": sum(item["bytes"] for item in entries), "status": "PASS" if state.get("global_step") == step and not missing else "BLOCKED"}


def compare_statistics(left: Any, right: Any, atol: float = 1e-12) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(compare_statistics(left[key], right[key], atol) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(compare_statistics(a, b, atol) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        return abs(float(left) - float(right)) <= atol
    return left == right


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    require(args.seed in N2_SEEDS, "PI0133 permits only N2 seeds 2222/2223/2224")
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
    cache_key = tuple(str(Path(paths[name]).resolve()) for name in ("groot", "base_model", "dataset", "modality", "cosmos", "snapshot_preflight", "sealed_baseline"))
    cached = _SEALED_CONTEXT_CACHE.get(cache_key)
    if cached is not None:
        return cached
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
    context = (torch, ev, processor, embodiment, records, baseline_noise, (windows, noise_bank))
    _SEALED_CONTEXT_CACHE[cache_key] = context
    return context


def _checkpoint_hash(path: Path) -> str:
    return tree_digest(tree_entries(path, f"stage45e/{path.name}"))


def _legacy_checkpoint_hash(path: Path) -> str:
    """Hash a transported legacy checkpoint with its accepted export prefix.

    The historical FULL24 transport manifest defines its tree identity over
    ``checkpoint-{step}/...``.  N2 training artifacts intentionally retain the
    ``stage45e/checkpoint-{step}/...`` prefix used by ``checkpoint_record``;
    keep the two provenance namespaces explicit rather than weakening either
    identity check.
    """

    return tree_digest(tree_entries(path, path.name))


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


def _training_result_path(root: Path) -> Path:
    """Accept the two historical result-file spellings, never a bare dir."""

    candidates = (root / "training_result.json", root / "stage45e_n2_training_result.json")
    present = [path for path in candidates if path.is_file() and not path.is_symlink()]
    require(len(present) == 1, f"N2 training result identity is ambiguous or absent: {root}")
    return present[0]


def _verify_n2_checkpoint_root(root: Path, expected_seed: int, n2_stats_path: Path) -> dict[str, Any]:
    require(root.is_dir() and not root.is_symlink(), f"N2 checkpoint root is unavailable: {root}")
    result_path = _training_result_path(root)
    result = read_json(result_path)
    require(result.get("seed") == expected_seed, f"N2 seed identity mismatch at {root}")
    status = result.get("status")
    require(status in {"PASS", "STAGE45E_N2_CORELAB_2000_TRAINING_COMPLETE_FOR_PI_REVIEW"}, f"N2 training result is not complete: {result_path}")
    require(result.get("starting_checkpoint") == "BASE", f"N2 root did not start from BASE: {root}")
    require(result.get("base_tree_sha256") == BASE_TREE and result.get("full24_tree_sha256") == FULL24_TREE, f"N2 training identity drift: {root}")
    require(result.get("training_steps", result.get("max_steps")) == TRAIN_STEPS, f"N2 step budget drift: {root}")
    require(result.get("n2_stats_sha256") == N2_STATS_SHA, f"N2 statistics file identity drift: {root}")
    require(result.get("n2_stats_payload_sha256") == N2_STATS_PAYLOAD_SHA, f"N2 statistics payload identity drift: {root}")
    n2_stats_path = Path(n2_stats_path).resolve()
    require(n2_stats_path.is_file() and not n2_stats_path.is_symlink(), f"N2 statistics artifact is unavailable: {n2_stats_path}")
    require(sha256_file(n2_stats_path) == N2_STATS_SHA, "accepted N2 statistics file SHA drift")
    n2_doc = read_json(n2_stats_path)
    require(n2_doc.get("statistics_payload_sha256") == N2_STATS_PAYLOAD_SHA, "accepted N2 statistics payload field drift")
    n2_statistics = n2_doc.get("statistics")
    require(isinstance(n2_statistics, Mapping), "accepted N2 statistics payload is missing")
    sidecar_path = root / "experiment_cfg/structured_action_statistics.json"
    require(sidecar_path.is_file() and not sidecar_path.is_symlink(), f"N2 statistics sidecar is absent: {sidecar_path}")
    sidecar = read_json(sidecar_path)
    sidecar_statistics = sidecar.get("statistics")
    require(isinstance(sidecar_statistics, Mapping), f"N2 statistics sidecar payload is missing: {sidecar_path}")
    require(compare_statistics(sidecar_statistics, n2_statistics), f"N2 sidecar statistics drift: {sidecar_path}")
    checkpoints: dict[str, Any] = {}
    for step in (1000, 2000):
        checkpoint = root / f"checkpoint-{step}"
        record = checkpoint_record(checkpoint, step)
        require(record["status"] == "PASS", f"N2 checkpoint state is incomplete: {checkpoint}")
        result_record = result.get(f"checkpoint_{step}")
        require(isinstance(result_record, Mapping) and result_record.get("tree_sha256") == record["tree_sha256"], f"N2 checkpoint tree identity is not bound in training result: {checkpoint}")
        checkpoints[str(step)] = record
    return {"root": str(root), "result_path": str(result_path), "result_sha256": sha256_file(result_path), "result": result, "n2_stats_sha256": N2_STATS_SHA, "n2_stats_payload_sha256": N2_STATS_PAYLOAD_SHA, "sidecar_sha256": sha256_file(sidecar_path), "sidecar_statistics_binding": {"match": True, "numeric_atol": 1e-12}, "checkpoints": checkpoints, "sidecar_statistics": sidecar_statistics}


def _require_legacy_paths(args: argparse.Namespace, *, n2_roots: Mapping[int, Path] | None = None) -> dict[str, Any]:
    """Bind actual comparator/replica bytes; no scratch-path substitution."""

    require(args.legacy_full24_2000 is not None, "G3/G4 require --legacy-full24-2000; raw-only comparator evidence is not a model")
    legacy = Path(args.legacy_full24_2000).resolve()
    require(legacy.is_dir() and not legacy.is_symlink(), f"required FULL24@2000 comparator checkpoint is not materialized: {legacy}")
    legacy_tree = _legacy_checkpoint_hash(legacy)
    require(legacy_tree == FULL24_2000_TREE, f"FULL24@2000 comparator tree mismatch: {legacy_tree}")
    roots = dict(n2_roots or {})
    for seed in N2_SEEDS:
        value = roots.get(seed)
        require(value is not None, f"G3/G4 require an actual N2 seed-{seed} checkpoint root")
    require(getattr(args, "n2_stats", None) is not None, "N2 statistics path is required for root provenance")
    verified = {seed: _verify_n2_checkpoint_root(Path(roots[seed]).resolve(), seed, Path(args.n2_stats)) for seed in N2_SEEDS}
    return {"legacy_full24_2000": legacy, "legacy_full24_2000_tree_sha256": legacy_tree, "n2": verified}


def _diagnostic_n2_roots(args: argparse.Namespace) -> dict[int, Path]:
    """Resolve late-bound N2 roots without embedding a node-local path."""

    values: dict[int, Path] = {}
    explicit = {
        2222: getattr(args, "n2_2222_root", None),
        2223: getattr(args, "n2_2223_root", None),
        2224: getattr(args, "n2_2224_root", None),
    }
    # The sprint orchestrator passes the two newly trained roots through the
    # existing replica flags; direct G3/G4 callers may use explicit flags.
    explicit[2223] = explicit[2223] or getattr(args, "replica1_root", None)
    explicit[2224] = explicit[2224] or getattr(args, "replica2_root", None)
    for seed, value in explicit.items():
        if value is not None:
            values[seed] = Path(value).resolve()
    return values


def _write_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        import numpy as np

        np.savez(handle, **{key: np.ascontiguousarray(value) for key, value in arrays.items()})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _array_sha(value: Any) -> str:
    import numpy as np

    return sha256_bytes(np.ascontiguousarray(value).tobytes())


def _canonical_mask() -> Any:
    from go2arx5_vla.structured_action import structured_action_mask

    return structured_action_mask("T16")


def _flow_metrics(left: Any, right: Any, *, name: str) -> dict[str, Any]:
    import numpy as np

    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    require(lhs.shape == RAW_SHAPE and rhs.shape == RAW_SHAPE, f"{name} shape drift")
    mask = np.asarray(_canonical_mask(), dtype=bool)
    delta = lhs[0][mask] - rhs[0][mask]
    require(np.isfinite(delta).all(), f"{name} is nonfinite")
    denominator = float(np.linalg.norm(lhs[0][mask]) * np.linalg.norm(rhs[0][mask]))
    return {
        "namespace": name,
        "active_scalar_count": int(mask.sum()),
        "l2": float(np.linalg.norm(delta)),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "mae": float(np.mean(np.abs(delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "cosine": float(np.dot(lhs[0][mask], rhs[0][mask]) / denominator) if denominator else 0.0,
    }


def _diagnostic_call(torch: Any, stage38: Any, model: Any, conditioning: tuple[Any, Any, Any], state_np: Any, bucket: int, counters: dict[str, Any]) -> Any:
    import numpy as np

    processed_backbone, action_input, state_features = conditioning
    device = processed_backbone.backbone_features.device
    state = torch.from_numpy(np.ascontiguousarray(state_np, dtype=np.float32)).to(device=device, dtype=next(model.parameters()).dtype)
    with torch.inference_mode():
        velocity = stage38._offline_step(model.action_head, processed_backbone, action_input, state_features, state, int(bucket))
    counters["action_head_calls"] += 1
    counters["model_forward"] += 1
    _sync_failure_boundary(counters)
    output = velocity.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    require(output.shape == RAW_SHAPE and np.isfinite(output).all(), "diagnostic velocity shape/finite drift")
    return output


def _sealed_snapshot_adapter(record: Mapping[str, Any]) -> Any:
    """Materialize the label-only snapshot view from sealed preflight fields.

    The accepted Stage45C preflight intentionally records the canonical
    parquet identity, whose historical absolute path is not portable to
    Corelab.  G3/G4 only need the already-sealed state/timestamp/label fields
    for the exact StructuredAction encode; the local TRAIN24 loader supplies
    the image/state observation.  No historical scratch path is dereferenced.
    """

    import numpy as np
    from types import SimpleNamespace

    state_values = np.asarray(record["state_v1"], dtype=np.float32)
    timestamps = np.asarray(record["source_timestamps"], dtype=np.float64)
    require(state_values.shape == (10,) and np.isfinite(state_values).all(), "sealed snapshot StateV1 shape/finite drift")
    require(timestamps.shape == (HORIZON,) and np.isfinite(timestamps).all(), "sealed snapshot timestamp shape/finite drift")
    state = SimpleNamespace(as_array=lambda: state_values.copy())
    return SimpleNamespace(
        example_id=str(record["window_id"]),
        state_v1=state,
        future_timestamps=timestamps,
        source_frame="NATIVE_WBC_RELATIVE_REFERENCE",
        tcp_semantic_id="x5_link6_gripper_center",
        observation_timestamp=float(record["observation_timestamp"]),
        physical_action=np.asarray(record["expert_physical_action"], dtype=np.float64),
    )


def _diagnostic_checkpoint_specs(binding: Mapping[str, Any], phase: str) -> list[dict[str, Any]]:
    """Return late-bound checkpoint specs for the frozen diagnostic phase.

    G3/G4 compare the accepted legacy FULL24@2000 against all three N2
    @2000 replicas.  PI0125 G5 is deliberately limited to the three N2
    @1000 checkpoints; it does not rerun any @2000 model call.
    """

    if phase in {"G5_D1", "G5_D2A"}:
        return [
            {
                "label": f"N2(seed{seed})@1000",
                "path": Path(binding["n2"][seed]["root"]) / "checkpoint-1000",
                "tree": binding["n2"][seed]["checkpoints"]["1000"]["tree_sha256"],
                "stats": "n2",
                "seed": seed,
            }
            for seed in N2_SEEDS
        ]
    if phase in {"N2_D1", "N2_D2A"}:
        specs: list[dict[str, Any]] = []
        for seed in N2_SEEDS:
            info = binding["n2"][seed]
            for step in (1000, 2000):
                specs.append({
                    "label": f"N2(seed{seed})@{step}",
                    "path": Path(info["root"]) / f"checkpoint-{step}",
                    "tree": info["checkpoints"][str(step)]["tree_sha256"],
                    "stats": "n2",
                    "seed": seed,
                    "step": step,
                })
        return specs
    specs = [
        {"label": "FULL24@2000", "path": binding["legacy_full24_2000"], "tree": binding["legacy_full24_2000_tree_sha256"], "stats": "runtime"},
    ]
    for seed in N2_SEEDS:
        info = binding["n2"][seed]
        specs.append({"label": f"N2(seed{seed})@2000", "path": Path(info["root"]) / "checkpoint-2000", "tree": info["checkpoints"]["2000"]["tree_sha256"], "stats": "n2", "seed": seed})
    return specs


def _require_n2_paths(args: argparse.Namespace, *, n2_roots: Mapping[int, Path]) -> dict[str, Any]:
    """Verify all three already-completed N2 roots for the G5 @1000 probe."""

    for seed in N2_SEEDS:
        require(seed in n2_roots, f"G5 requires an actual N2 seed-{seed} checkpoint root")
    require(getattr(args, "n2_stats", None) is not None, "N2 statistics path is required for root provenance")
    verified = {seed: _verify_n2_checkpoint_root(Path(n2_roots[seed]).resolve(), seed, Path(args.n2_stats)) for seed in N2_SEEDS}
    return {"n2": verified}


def _diagnostic_runtime(args: argparse.Namespace, phase: str) -> dict[str, Any]:
    """Run the frozen D1 or D2A graph after all late-bound identities pass."""

    import numpy as np

    output = Path(args.output_root).resolve()
    require(not output.exists(), f"{phase} output root must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    counters: dict[str, Any] = {
        "_phase": phase,
        "gpu": False,
        "model_constructed": 0,
        "gr00t_forwards": 0,
        "model_construction": 0,
        "conditioning_passes": 0,
        "action_head_calls": 0,
        "model_forward": 0,
        "training": False,
        "training_steps": 0,
        "backward_steps": 0,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "checkpoint_writes": 0,
        "public_sampler_calls": 0,
        "get_action_calls": 0,
        "free_running_sampler_calls": 0,
        "WBC": 0,
        "Validation-E": 0,
        "Holdout-F": 0,
        "raw_stage_count": 0,
    }
    _sync_failure_boundary(counters, phase=phase)
    try:
        is_d1 = phase.endswith("_D1")
        is_g5 = phase.startswith("G5_")
        is_n2_route = phase.startswith("N2_")
        n2_roots = _diagnostic_n2_roots(args)
        binding = _require_n2_paths(args, n2_roots=n2_roots) if (is_g5 or is_n2_route) else _require_legacy_paths(args, n2_roots=n2_roots)
        paths = _paths(args)
        identity = _verify_core_identities(paths)
        gpu = gpu_identity(args.gpu_uuid or PREFERRED_GPU_UUID)
        prepare_env(paths, output, gpu)
        counters["gpu"] = True
        _sync_failure_boundary(counters)
        torch, ev, _processor, _embodiment, records, baseline_noise, context = _load_sealed_context(paths)
        windows, _baseline_noise_bank = context
        require(len(records) == 24 and len(windows) == 24, "diagnostic TRAIN24 materialization drift")
        import stage38_structured_offline_evaluation as stage38
        snapshots = [_sealed_snapshot_adapter(record) for record in records]
        runtime_doc = read_json(paths["runtime_stats"])
        n2_doc = read_json(paths["n2_stats"])
        stats = {
            "runtime": (runtime_doc, RUNTIME_STATS_SHA, RUNTIME_STATS_PAYLOAD_SHA, "STAGE45C_RUNTIME_V2_STATS_BRIDGE"),
            "n2": (n2_doc, N2_STATS_SHA, N2_STATS_PAYLOAD_SHA, "N2_BALANCED_XYZ_STATS"),
        }
        specs = _diagnostic_checkpoint_specs(binding, phase)
        noise_bank: dict[int, list[Any]] = {}
        for seed in (2223, 2224, 2225) if is_d1 else (4242,):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            noise_bank[seed] = [torch.randn(RAW_SHAPE, generator=generator, device="cpu", dtype=torch.float32).contiguous() for _ in range(24)]
        comparator_binding = {"n2_seeds": list(N2_SEEDS)}
        if "legacy_full24_2000_tree_sha256" in binding:
            comparator_binding["legacy_full24_2000_tree_sha256"] = binding["legacy_full24_2000_tree_sha256"]
        write_json(output / "admission.json", {"schema": f"CORELAB_STAGE45E_{phase}_ADMISSION_V1", "status": "PASS", "phase": phase, "branch": BRANCH, "repository_head": git("rev-parse", "HEAD"), "identities": identity, "comparator_binding": comparator_binding, "gpu": gpu, "execution_boundary": {"gpu": True, "training": False, "optimizer_steps": 0, "WBC": 0, "Validation-E": 0, "Holdout-F": 0}})
        raw_manifest = output / "raw_probe_rows.jsonl"
        raw_manifest.write_text("", encoding="utf-8")
        all_rows: list[dict[str, Any]] = []
        checkpoint_results: list[dict[str, Any]] = []
        t_grid = list(zip(G3_T_VALUES, G3_T_BUCKETS)) if is_d1 else list(zip(G4_T_VALUES, G4_T_BUCKETS))
        for spec in specs:
            label = str(spec["label"])
            stats_doc, stats_sha, stats_payload_sha, stats_route = stats[spec["stats"]]
            model = stage38._load_structured_model(Path(spec["path"]), "cuda:0")
            counters["model_construction"] += 1
            counters["model_constructed"] = counters["model_construction"]
            _sync_failure_boundary(counters)
            model.eval()
            hook_counter = {"count": 0}
            hook = model.action_head.model.register_forward_hook(lambda *_args, **_kwargs: hook_counter.__setitem__("count", hook_counter["count"] + 1))
            row_count = 0
            conditioning_count = 0
            try:
                for index, (window, record, snapshot) in enumerate(zip(windows, records, snapshots)):
                    x1 = np.asarray(stage38._structured_label_and_normalizer(np.asarray(snapshot.physical_action, dtype=np.float64)[None, ...], snapshot, stats_doc)[2], dtype=np.float32)[None, ...]
                    if is_d1:
                        seeds = (2223, 2224, 2225)
                        # D1 conditioning is a function of the checkpoint and
                        # TRAIN24 observation only.  Build it once per window
                        # and reuse the exact tensors for all frozen noise/t
                        # probes; this is the accepted 96-pass accounting.
                        with torch.inference_mode():
                            _, action_input, processed, state_features = stage38._prepare_model_conditioning(model, window["model_inputs"])
                        conditioning_count += 1
                        counters["conditioning_passes"] += 1
                        conditioning = (processed, action_input, state_features)
                        for noise_seed in seeds:
                            x0 = noise_bank[noise_seed][index].numpy().copy()
                            for t, bucket in t_grid:
                                x_t = ((1.0 - t) * x0 + t * x1).astype(np.float32, copy=False)
                                v_target = (x1 - x0).astype(np.float32, copy=False)
                                v_hat = _diagnostic_call(torch, stage38, model, conditioning, x_t, bucket, counters)
                                x1_hat = (x_t + (1.0 - t) * v_hat).astype(np.float32, copy=False)
                                arrays = {"x0": x0, "x1": x1, "x_t": x_t, "v_target": v_target, "v_hat": v_hat, "x1_hat": x1_hat, "mask": np.asarray(_canonical_mask(), dtype=np.float32)}
                                probe_root = output / "probes" / label.replace("@", "_at_") / f"w{index:02d}" / f"seed{noise_seed}" / f"t{bucket:03d}"
                                raw_path = probe_root / "raw_arrays.npz"
                                _write_npz(raw_path, arrays)
                                raw_record = {"schema": f"CORELAB_STAGE45E_N2_{phase}_RAW_STAGE_V1", "status": "RAW_SEALED_BEFORE_DERIVED", "identity": {"checkpoint": label, "checkpoint_tree_sha256": spec["tree"], "stats_sha256": stats_sha, "stats_payload_sha256": stats_payload_sha, "stats_route": stats_route, "window_id": window["window_id"], "target_id": record["target_id"], "window_index": index, "noise_seed": noise_seed, "t": t, "t_bucket": bucket, "target_xyz_in_model_input": False}, "raw_archive": {"path": str(raw_path.relative_to(output)), "sha256": sha256_file(raw_path), "bytes": raw_path.stat().st_size}, "arrays": {key: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype), "sha256": _array_sha(value)} for key, value in arrays.items()}}
                                write_json(probe_root / "raw_manifest.json", raw_record)
                                physical, decoder = ev._decode_with_stats(x1_hat, record["state_v1"], stats_doc, stats_sha256=stats_sha, payload_sha256=stats_payload_sha, chunk_id=f"{label}:{window['window_id']}:seed{noise_seed}:t{bucket}", timestamps=record["source_timestamps"], observation_timestamp=float(record["observation_timestamp"]))
                                from go2arx5_vla.evaluation.trajectory_metrics import comparison_metrics
                                physical_metrics = comparison_metrics(np.asarray(physical, dtype=np.float64), np.asarray(record["expert_physical_action"], dtype=np.float64), np.asarray(record["anchor_world_m"], dtype=np.float64), np.asarray(record["target_world_m"], dtype=np.float64), prefix=f"{label}:{window['window_id']}:seed{noise_seed}:t{bucket}")
                                derived = {"schema": f"CORELAB_STAGE45E_N2_{phase}_DERIVED_STAGE_V1", "status": "DERIVED_AFTER_RAW_SEAL", "raw_before_derived": True, "normalized_local_flow": _flow_metrics(v_hat, v_target, name="NORMALIZED_LOCAL_FLOW"), "implied_endpoint": {"formula": "x1_hat=x_t+(1-t)*v_hat", "array_sha256": _array_sha(x1_hat)}, "physical": {"source": "x1_hat_only", "decoder": decoder, "trajectory_metrics": physical_metrics}}
                                write_json(probe_root / "derived_metrics.json", derived)
                                row = {"schema": f"CORELAB_STAGE45E_N2_{phase}_ROW_V1", "identity": raw_record["identity"], "raw_manifest_sha256": sha256_file(probe_root / "raw_manifest.json"), "raw_sha256": sha256_file(raw_path), "derived_sha256": sha256_file(probe_root / "derived_metrics.json"), "normalized_local_flow": derived["normalized_local_flow"], "physical": derived["physical"], "finite": True}
                                with raw_manifest.open("a", encoding="utf-8") as handle:
                                    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                                all_rows.append(row)
                                counters["raw_stage_count"] += 1
                                row_count += 1
                    else:
                        with torch.inference_mode():
                            _, action_input, processed, state_features = stage38._prepare_model_conditioning(model, window["model_inputs"])
                        conditioning_count += 1
                        counters["conditioning_passes"] += 1
                        conditioning = (processed, action_input, state_features)
                        x0 = noise_bank[4242][index].numpy().copy()
                        x_fr = x0.copy()
                        for stage_index, (t, bucket) in enumerate(t_grid):
                            x_tf = ((1.0 - t) * x0 + t * x1).astype(np.float32, copy=False)
                            v_target = (x1 - x0).astype(np.float32, copy=False)
                            v_tf = _diagnostic_call(torch, stage38, model, conditioning, x_tf, bucket, counters)
                            v_fr = _diagnostic_call(torch, stage38, model, conditioning, x_fr, bucket, counters)
                            if bucket == 0:
                                require(np.array_equal(v_tf, v_fr), "G4 t=0 teacher/free outputs are not byte-identical")
                            x_tf_next = (x_tf + DT * v_tf).astype(np.float32, copy=False)
                            x_fr_next = (x_fr + DT * v_fr).astype(np.float32, copy=False)
                            arrays = {"x0": x0, "x1": x1, "x_tf": x_tf, "x_fr": x_fr, "v_target": v_target, "v_tf": v_tf, "v_fr": v_fr, "x_tf_next": x_tf_next, "x_fr_next": x_fr_next, "mask": np.asarray(_canonical_mask(), dtype=np.float32)}
                            probe_root = output / "probes" / label.replace("@", "_at_") / f"w{index:02d}" / "seed4242" / f"t{bucket:03d}"
                            raw_path = probe_root / "raw_arrays.npz"
                            _write_npz(raw_path, arrays)
                            raw_record = {"schema": f"CORELAB_STAGE45E_N2_{phase}_RAW_STAGE_V1", "status": "RAW_SEALED_BEFORE_DERIVED", "identity": {"checkpoint": label, "checkpoint_tree_sha256": spec["tree"], "stats_sha256": stats_sha, "stats_payload_sha256": stats_payload_sha, "stats_route": stats_route, "window_id": window["window_id"], "target_id": record["target_id"], "window_index": index, "noise_seed": 4242, "t": t, "t_bucket": bucket, "stage_index": stage_index, "target_xyz_in_model_input": False}, "raw_archive": {"path": str(raw_path.relative_to(output)), "sha256": sha256_file(raw_path), "bytes": raw_path.stat().st_size}, "arrays": {key: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype), "sha256": _array_sha(value)} for key, value in arrays.items()}}
                            write_json(probe_root / "raw_manifest.json", raw_record)
                            derived = {"schema": f"CORELAB_STAGE45E_N2_{phase}_DERIVED_STAGE_V1", "status": "DERIVED_AFTER_RAW_SEAL", "raw_before_derived": True, "state_path_divergence": _flow_metrics(x_fr, x_tf, name="D2A_STATE_PATH_DIVERGENCE"), "field_sensitivity": _flow_metrics(v_fr, v_tf, name="D2A_FIELD_SENSITIVITY"), "teacher_bridge_local_error": _flow_metrics(v_tf, v_target, name="NORMALIZED_LOCAL_FLOW"), "off_path_diagnostic": _flow_metrics(v_fr, v_target, name="DESCRIPTIVE_OFF_PATH_DIAGNOSTIC_NOT_SUPERVISED"), "t_zero": {"checked": bucket == 0, "exact_equal": bool(np.array_equal(v_tf, v_fr)) if bucket == 0 else None}, "path_update": {"dt": DT, "free_running_reset": False, "x_tf_next_formula": "x_tf+dt*v_tf", "x_fr_next_formula": "x_fr+dt*v_fr"}}
                            write_json(probe_root / "derived_metrics.json", derived)
                            row = {"schema": f"CORELAB_STAGE45E_N2_{phase}_ROW_V1", "identity": raw_record["identity"], "raw_manifest_sha256": sha256_file(probe_root / "raw_manifest.json"), "raw_sha256": sha256_file(raw_path), "derived_sha256": sha256_file(probe_root / "derived_metrics.json"), "state_path_divergence": derived["state_path_divergence"], "field_sensitivity": derived["field_sensitivity"], "teacher_bridge_local_error": derived["teacher_bridge_local_error"], "off_path_diagnostic": derived["off_path_diagnostic"], "t_zero": derived["t_zero"], "finite": True}
                            with raw_manifest.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                            all_rows.append(row)
                            counters["raw_stage_count"] += 1
                            row_count += 1
                            x_fr = x_fr_next
            finally:
                hook.remove()
            expected_hook = 24 * 3 * 3 if is_d1 else 24 * 4 * 2
            require(hook_counter["count"] == expected_hook, f"{phase} action-head budget drift for {label}: {hook_counter['count']} != {expected_hook}")
            checkpoint_results.append({"checkpoint": label, "tree_sha256": spec["tree"], "rows": row_count, "conditioning_passes": conditioning_count, "action_head_calls": hook_counter["count"]})
            del model
            torch.cuda.empty_cache()
        expected = {
            "G3_D1": {"model_construction": 4, "conditioning_passes": 96, "action_head_calls": 864, "model_forward": 864, "raw_stage_count": 864},
            "G4_D2A": {"model_construction": 4, "conditioning_passes": 96, "action_head_calls": 768, "model_forward": 768, "raw_stage_count": 384},
            "G5_D1": {"model_construction": 3, "conditioning_passes": 72, "action_head_calls": 648, "model_forward": 648, "raw_stage_count": 648},
            "G5_D2A": {"model_construction": 3, "conditioning_passes": 72, "action_head_calls": 576, "model_forward": 576, "raw_stage_count": 288},
            "N2_D1": {"model_construction": 6, "conditioning_passes": 144, "action_head_calls": 1296, "model_forward": 1296, "raw_stage_count": 1296},
            "N2_D2A": {"model_construction": 6, "conditioning_passes": 144, "action_head_calls": 1152, "model_forward": 1152, "raw_stage_count": 576},
        }[phase]
        for key, value in expected.items():
            require(counters[key] == value, f"{phase} total budget drift {key}: {counters[key]} != {value}")
        seal = {"schema": f"CORELAB_STAGE45E_N2_{phase}_SEAL_V1", "status": "PASS", "token": f"{phase}_PASS", "branch": BRANCH, "repository_head": git("rev-parse", "HEAD"), "checkpoints": checkpoint_results, "counters": counters, "execution_boundary": dict(_BOUNDARY), "raw_probe_rows_jsonl_sha256": sha256_file(raw_manifest), "gpu": gpu, "training": False, "optimizer_steps": 0, "WBC": 0, "Validation-E": 0, "Holdout-F": 0, "target_xyz_in_model_input": False, "timestamp_utc": now_utc()}
        write_json(output / "seal.json", seal)
        return seal
    except Exception as exc:
        _sync_failure_boundary(counters, phase=phase)
        write_json(output / "FAILURE_SEAL.json", {"schema": f"CORELAB_STAGE45E_N2_{phase}_FAILURE_SEAL_V1", "status": "BLOCKED", "phase": phase, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "counters": counters, "execution_boundary": dict(_BOUNDARY), "retry_or_substitution": False, "timestamp_utc": now_utc()}, overwrite=True)
        raise


def n2_e4_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate all six N2 H200 checkpoint arms on the frozen TRAIN24 E4 graph."""

    import numpy as np

    paths = _paths(args)
    identity = _verify_core_identities(paths)
    n2_roots = _diagnostic_n2_roots(args)
    binding = _require_n2_paths(args, n2_roots=n2_roots)
    output = Path(args.output_root).resolve()
    require(not output.exists(), f"N2 E4 output root must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    gpu = gpu_identity(args.gpu_uuid or PREFERRED_GPU_UUID)
    prepare_env(paths, output, gpu)
    torch, ev, _processor, _embodiment, records, baseline_noise, context = _load_sealed_context(paths)
    windows, noise_bank = context
    require(len(records) == 24 and len(windows) == 24, "N2 E4 TRAIN24 materialization drift")
    import stage38_structured_offline_evaluation as stage38
    from stage45c_h200_eval_runtime_adapter import _sample_one
    from go2arx5_vla.evaluation.checkpoint_eval import aggregate_evaluation_rows

    n2_doc = read_json(paths["n2_stats"])
    specs = []
    for seed in N2_SEEDS:
        info = binding["n2"][seed]
        for step in (1000, 2000):
            specs.append({
                "label": f"N2(seed{seed})@{step}",
                "seed": seed,
                "step": step,
                "path": Path(info["root"]) / f"checkpoint-{step}",
                "tree": info["checkpoints"][str(step)]["tree_sha256"],
            })
    raw_manifest = output / "raw_predictions.jsonl"
    raw_manifest.write_text("", encoding="utf-8")
    counters = {
        "gpu": True, "model_construction": 0, "conditioning_passes": 0,
        "action_head_calls": 0, "model_forward": 0, "raw_rows": 0,
        "training": False, "backward_steps": 0, "optimizer_steps": 0,
        "checkpoint_writes": 0, "public_sampler_calls": 0, "get_action_calls": 0,
        "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0,
    }
    all_rows: dict[str, list[dict[str, Any]]] = {}
    try:
        for spec in specs:
            label = spec["label"]
            model = stage38._load_structured_model(spec["path"], "cuda:0")
            counters["model_construction"] += 1
            hook_counter = {"count": 0}
            hook = model.action_head.model.register_forward_hook(
                lambda *_args, **_kwargs: hook_counter.__setitem__("count", hook_counter["count"] + 1)
            )
            rows: list[dict[str, Any]] = []
            try:
                for index, (window, record) in enumerate(zip(windows, records)):
                    with torch.inference_mode():
                        raw, calls = _sample_one(model, window["model_inputs"], noise_bank[index], torch)
                    counters["conditioning_passes"] += 1
                    counters["action_head_calls"] += calls
                    counters["model_forward"] += calls
                    raw_np = raw.detach().cpu().numpy().astype("float32", copy=False)
                    require(raw_np.shape == RAW_SHAPE and np.isfinite(raw_np).all(), f"N2 E4 raw shape/finite drift: {label}/{index}")
                    raw_path = output / "raw" / label.replace("@", "_at_") / f"{index:02d}.npy"
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(raw_path, raw_np, allow_pickle=False)
                    raw_sha = sha256_file(raw_path)
                    physical, details = ev._decode_with_stats(
                        raw_np, record["state_v1"], n2_doc,
                        stats_sha256=N2_STATS_SHA, payload_sha256=N2_STATS_PAYLOAD_SHA,
                        chunk_id=record["window_id"], timestamps=record["source_timestamps"],
                        observation_timestamp=float(record["observation_timestamp"]),
                    )
                    normalized = ev._normalized_metrics(
                        raw_np, record["expert_physical_action"], record["state_v1"],
                        record["source_timestamps"], n2_doc,
                    )
                    checkpoint = {"label": label, "tree_sha256": spec["tree"], "seed": spec["seed"], "step": spec["step"]}
                    row = ev._evaluation_row(
                        raw_np, physical, details, normalized, record, checkpoint,
                        raw_path.relative_to(output), raw_sha, GROOT_COMMIT,
                    )
                    row.update({
                        "schema": "VANDA_STAGE45E_N2_H200_E4_ROW_V1",
                        "stats_route": "N2_BALANCED_XYZ_STATS",
                        "sampler": {"steps_observed": calls, "timestep_ids": list(T16), "dt": DT, "official_public_sampler_called": False},
                        "target_xyz_in_model_input": False,
                        "finite": True,
                    })
                    rows.append(row)
                    with raw_manifest.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({
                            "checkpoint": checkpoint, "window_id": record["window_id"],
                            "path": str(raw_path.relative_to(output)), "file_sha256": raw_sha,
                            "array_sha256": sha256_bytes(raw_np.tobytes()),
                            "stats_route": "N2_BALANCED_XYZ_STATS", "sampler_calls_observed": calls,
                        }, sort_keys=True) + "\n")
                    counters["raw_rows"] += 1
            finally:
                hook.remove()
            require(hook_counter["count"] == 96, f"N2 E4 action-head budget drift: {label}={hook_counter['count']}")
            require(len(rows) == 24, f"N2 E4 row count drift: {label}")
            all_rows[label] = rows
            write_json(output / f"rows_{label.replace('@', '_at_')}.json", rows)
            write_json(output / f"aggregate_{label.replace('@', '_at_')}.json", aggregate_evaluation_rows(rows))
            del model
            torch.cuda.empty_cache()
        require((counters["conditioning_passes"], counters["action_head_calls"], counters["model_forward"], counters["raw_rows"]) == (144, 576, 576, 144), f"N2 E4 total budget drift: {counters}")
        seal = {
            "schema": "VANDA_STAGE45E_N2_H200_E4_SEAL_V1", "status": "PASS", "token": "N2_E4_PASS",
            "pi_gate": PI_GATE, "branch": BRANCH, "repository_head": git("rev-parse", "HEAD"),
            "identity": identity, "checkpoint_count": 6, "checkpoints": [
                {"label": spec["label"], "seed": spec["seed"], "step": spec["step"], "tree_sha256": spec["tree"], "rows": 24, "action_head_calls": 96}
                for spec in specs
            ], "counters": counters, "raw_predictions_manifest_sha256": sha256_file(raw_manifest),
            "gpu": gpu, "training": False, "optimizer_steps": 0, "Validation-E": 0,
            "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0, "target_xyz_in_model_input": False,
            "timestamp_utc": now_utc(),
        }
        write_json(output / "seal.json", seal)
        return seal
    except Exception as exc:
        _sync_failure_boundary(counters, phase="N2_E4")
        write_json(output / "FAILURE_SEAL.json", {
            "schema": "VANDA_STAGE45E_N2_H200_E4_FAILURE_SEAL_V1", "status": "BLOCKED",
            "pi_gate": PI_GATE, "phase": "N2_E4", "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(), "counters": counters,
            "execution_boundary": dict(_BOUNDARY), "retry_or_substitution": False, "timestamp_utc": now_utc(),
        }, overwrite=True)
        raise


def g3_eval(args: argparse.Namespace) -> dict[str, Any]:
    return _diagnostic_runtime(args, "G3_D1")


def g4_eval(args: argparse.Namespace) -> dict[str, Any]:
    return _diagnostic_runtime(args, "G4_D2A")


def g5_d1_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Run the PI0125 D1 exposure probe on the three N2 @1000 checkpoints."""

    return _diagnostic_runtime(args, "G5_D1")


def g5_d2a_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Run the PI0125 matched TF/FR exposure probe on N2 @1000."""

    return _diagnostic_runtime(args, "G5_D2A")


def run_continuation_g3_g4_g5(args: argparse.Namespace) -> dict[str, Any]:
    """Run only the PI0130-authorized G3→G4→G5 continuation.

    This entrypoint intentionally has no calls to admission, ``train_one`` or
    ``g2_eval``.  G1/G2 are historical sealed evidence and must never be
    regenerated by a continuation invocation.
    """

    require(args.output_root is not None, "continuation requires --output-root")
    base = Path(args.output_root).resolve()
    require(not base.exists(), f"continuation output root must be fresh: {base}")
    base.mkdir(parents=True, exist_ok=False)

    def phase_args(name: str) -> argparse.Namespace:
        value = argparse.Namespace(**vars(args))
        value.output_root = base / name
        return value

    g3 = g3_eval(phase_args("G3_D1_teacher_forced_local_flow"))
    require(g3.get("status") == "PASS", "G3 continuation phase did not complete")
    g4 = g4_eval(phase_args("G4_D2A_tf_vs_fr"))
    require(g4.get("status") == "PASS", "G4 continuation phase did not complete")
    g5_d1 = g5_d1_eval(phase_args("G5_D1_n2_at_1000"))
    require(g5_d1.get("status") == "PASS", "G5-D1 continuation phase did not complete")
    g5_d2a = g5_d2a_eval(phase_args("G5_D2A_n2_at_1000"))
    require(g5_d2a.get("status") == "PASS", "G5-D2A continuation phase did not complete")
    result = {
        "schema": "CORELAB_STAGE45E_N2_GPU_CONTINUATION_RESULT_V2",
        "status": "PASS",
        "token": "STAGE45E_N2_CORELAB_GPU_SPRINT_G3_G4_G5_COMPLETE_FOR_PI_REVIEW",
        "pi_gate": "PI_GATE_STATE_0130",
        "repository_head": git("rev-parse", "HEAD"),
        "continuation_order": ["G3_D1", "G4_D2A", "G5_D1", "G5_D2A"],
        "forbidden_reruns": ["G0_ADMISSION", "G1_REPLICATION_TRAINING", "G2_TRAIN24_E4"],
        "g3": g3,
        "g4": g4,
        "g5_d1": g5_d1,
        "g5_d2a": g5_d2a,
        "scientific_contract_changed": False,
        "training": False,
        "Validation-E": 0,
        "Holdout-F": 0,
        "WBC": 0,
        "simulator": 0,
        "hardware": 0,
    }
    write_json(base / "continuation_result.json", result)
    return result


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
    # Comparator paths are deliberately checked only after G1/G2.  This
    # permits useful replication evidence while remaining fail-closed if the
    # accepted FULL24@2000 bytes have not been recovered to Corelab.
    diagnostic_args = argparse.Namespace(**vars(args))
    diagnostic_args.replica1_root = train_roots[2223]
    diagnostic_args.replica2_root = train_roots[2224]
    g3_args = argparse.Namespace(**vars(diagnostic_args))
    g3_args.output_root = base / "G3_D1_teacher_forced_local_flow"
    g3 = g3_eval(g3_args)
    g4_args = argparse.Namespace(**vars(diagnostic_args))
    g4_args.output_root = base / "G4_D2A_tf_vs_fr"
    g4 = g4_eval(g4_args)
    return {"schema": "CORELAB_STAGE45E_N2_GPU_SPRINT_RESULT_V1", "status": "PASS", "token": "STAGE45E_N2_CORELAB_GPU_SPRINT_COMPLETE_FOR_PI_REVIEW", "admission": admission_result, "canary": canary, "g1": {"seeds": [2223, 2224]}, "g2": g2, "g3": g3, "g4": g4}


def reset_boundary() -> None:
    global _BOUNDARY
    _BOUNDARY = {"gpu": False, "model_constructed": False, "gr00t_forwards": 0, "training_steps": 0, "optimizer_steps": 0, "backward_steps": 0, "checkpoint_writes": 0, "PBS": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "admission", "train", "g2", "g3", "g4", "g5d1", "g5d2a", "n2-e4", "n2-d1", "n2-d2a", "continue-g3-g4-g5", "sprint"), default="static")
    parser.add_argument("--base-model", dest="base_model", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--groot", type=Path)
    parser.add_argument("--groot-manifest", dest="groot_manifest", type=Path)
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
    parser.add_argument("--n2-2222-root", dest="n2_2222_root", type=Path)
    parser.add_argument("--n2-2223-root", dest="n2_2223_root", type=Path)
    parser.add_argument("--n2-2224-root", dest="n2_2224_root", type=Path)
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
            require(args.output_root is not None, "g3 requires output root")
            result = g3_eval(args)
        elif args.mode == "g4":
            require(args.output_root is not None, "g4 requires output root")
            result = g4_eval(args)
        elif args.mode == "g5d1":
            require(args.output_root is not None, "g5d1 requires output root")
            result = g5_d1_eval(args)
        elif args.mode == "g5d2a":
            require(args.output_root is not None, "g5d2a requires output root")
            result = g5_d2a_eval(args)
        elif args.mode == "n2-e4":
            require(args.output_root is not None, "n2-e4 requires output root")
            result = n2_e4_eval(args)
        elif args.mode == "n2-d1":
            require(args.output_root is not None, "n2-d1 requires output root")
            result = _diagnostic_runtime(args, "N2_D1")
        elif args.mode == "n2-d2a":
            require(args.output_root is not None, "n2-d2a requires output root")
            result = _diagnostic_runtime(args, "N2_D2A")
        elif args.mode == "continue-g3-g4-g5":
            result = run_continuation_g3_g4_g5(args)
        else:
            raise RuntimeError("legacy --mode sprint is disabled under PI0133; use the packed unified H200 orchestration entrypoint")
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    except Exception as exc:
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)
            outer_failure = output_root / "FAILURE_SEAL.json"
            # Preserve a phase-local seal written by the inner runner.  When
            # the continuation root is the outer container, keep a separate
            # top-level record rather than overwriting the more informative
            # phase counters and traceback.
            if outer_failure.exists():
                outer_failure = output_root / "OUTER_FAILURE_SEAL.json"
            if not outer_failure.exists():
                write_json(outer_failure, {"schema": "CORELAB_STAGE45E_N2_GPU_SPRINT_FAILURE_SEAL_V2", "status": "BLOCKED", "stage": STAGE, "pi_gate": PI_GATE, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "execution_boundary": _BOUNDARY, "retry_or_substitution": False, "timestamp_utc": now_utc()})
        print(f"STAGE45E_N2_GPU_SPRINT_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
