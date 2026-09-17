#!/usr/bin/env python3
"""Thin PI0208 Stage45H cohort, schedule, and training driver.

The native Stage43 runner and the accepted Stage38 dataset factory remain the
only execution implementations.  This file only freezes the new deterministic
cohort, replaces 400 FULL24 positions with unique support rows, and launches
one fresh training process per arm.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"

STAGE43_BUNDLE_SHA256 = "127d95a2a5b9a6a40e864e029d85fd09a9e4dd55441aae58071502aa76700da6"
BASE_S25_TREE_SHA256 = "93b8cd12dd1aa34ae1be91aa0e39baeacae0e189c72ea83d48a764a93c796492"
BASE_S25_TREE_PREFIX = "pi0205_S25/checkpoint-80"
BASE_S25_GLOBAL_STEP = 80
BASE_S25_LABEL = "S25@80"
FULL24_TREE_SHA256 = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
GROOT_MANIFEST_SHA256 = "96b883e56b54f763bec60d3a336632ad531f4fc52d099f803e5f9efba60eab11"
GROOT_MANIFEST_PAYLOAD_SHA256 = "a9f10b837b6086137fa519cc148275a25bf6a5f98d58df8e3c50e0a43e5b8250"
GROOT_PAYLOAD_TREE_SHA256 = "c0d57a6a74ab24c48abdd1c13420e9d1c893dc0d7b5ca370fa9b6f7af32f5f1b"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
STRUCTURED_SOURCE_STATS_SHA256 = "0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206"
MODALITY_SHA256 = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
CONFIG_SHA256 = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
LANGUAGE = "move the end effector to the red target"
TARGET_WORLD = (0.30, 0.10, 0.58)
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_ID = "x5_link6_gripper_center"
HORIZON = 40
ACTION_DIM = 10
COLLECTION_COUNT = 128
EVAL_NEW_COUNT = 11
EVAL_COUNT = 16
UPDATE_COUNT = 2000
SUPPORT_UPDATES = 400
SUPPORT_FRACTION = 0.20
MAX_SUPPORT_ROWS = COLLECTION_COUNT * 4
SEEDS = (7101, 7102, 7103)
ARM_ORDER = tuple([f"MCTRL_{seed}" for seed in SEEDS] + [f"MSUP_{seed}" for seed in SEEDS])


class Stage45HError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage45HError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def canonical_sha(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def read_json(path: Path, *, canonical: bool = False) -> Any:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise Stage45HError(f"JSON unavailable: {path}") from exc
    require(stat.S_ISREG(mode), f"JSON is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if canonical:
        require(path.read_bytes() == canonical_bytes(value), f"JSON is not canonical: {path}")
    return value


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> str:
    require(overwrite or (not path.exists() and not path.is_symlink()), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_bytes(payload)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def regular_file(path: Path, label: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise Stage45HError(f"{label} unavailable: {path}") from exc
    require(stat.S_ISREG(mode), f"{label} is not a regular file: {path}")
    return path


def regular_dir(path: Path, label: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise Stage45HError(f"{label} unavailable: {path}") from exc
    require(stat.S_ISDIR(mode), f"{label} is not a regular directory: {path}")
    return path


def child(root: Path, relative: str, label: str) -> Path:
    require(relative and "\x00" not in relative and "\\" not in relative, f"unsafe {label}")
    base = root.resolve()
    value = (base / relative).resolve()
    try:
        value.relative_to(base)
    except ValueError as exc:
        raise Stage45HError(f"{label} escapes root") from exc
    return regular_file(value, label)


def tree_digest(root: Path, prefix: str) -> tuple[str, list[dict[str, Any]]]:
    regular_dir(root, "tree root")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            item = directory_path / dirname
            require(not item.is_symlink() and item.is_dir(), f"unsafe tree directory: {item}")
        for filename in sorted(filenames):
            item = directory_path / filename
            regular_file(item, "tree entry")
            entries.append({
                "path": f"{prefix}/{item.relative_to(root).as_posix()}",
                "bytes": int(item.stat().st_size),
                "sha256": sha256_file(item),
            })
    entries.sort(key=lambda value: str(value["path"]))
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8"))
    return digest.hexdigest(), entries


def _clean_float(value: float) -> float:
    rounded = round(float(value), 8)
    return 0.0 if abs(rounded) < 5e-10 else rounded


def _polar_deltas(count: int, radius: float, phase: float, prefix: str) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for index in range(count):
        angle = phase + (2.0 * math.pi * index / count)
        delta = [_clean_float(radius * math.cos(angle)), _clean_float(radius * math.sin(angle)), 0.0]
        values[f"{prefix}_{index:03d}"] = delta
    require(len({tuple(value) for value in values.values()}) == count, f"{prefix} delta collision")
    return values


def _coordinate_key(delta: Sequence[float]) -> tuple[float, float, float]:
    return tuple(_clean_float(float(value)) for value in delta)  # type: ignore[return-value]


def cohort_manifest() -> dict[str, Any]:
    collection_phase = math.pi / 32.0
    collection = _polar_deltas(32, 0.01, collection_phase, "MH_COL")
    for radius in (0.025, 0.04, 0.05):
        offset = len(collection)
        values = _polar_deltas(32, radius, collection_phase, "MH_COL")
        collection.update({f"MH_COL_{offset + index:03d}": delta for index, delta in enumerate(values.values())})
    require(len(collection) == COLLECTION_COUNT, "collection cohort count drift")
    eval_new = _polar_deltas(EVAL_NEW_COUNT, 0.035, math.pi / 22.0, "MH_EVAL")
    evaluation = {
        "C0_NOMINAL": [0.0, 0.0, 0.0],
        "C1_X_PLUS_5CM": [0.05, 0.0, 0.0],
        "C2_X_MINUS_5CM": [-0.05, 0.0, 0.0],
        "C3_Y_PLUS_5CM": [0.0, 0.05, 0.0],
        "C4_Y_MINUS_5CM": [0.0, -0.05, 0.0],
        **eval_new,
    }
    require(len(evaluation) == EVAL_COUNT, "evaluation cohort count drift")
    require(not set(collection).intersection(evaluation), "collection/evaluation ID overlap")
    collection_coordinates = {_coordinate_key(delta) for delta in collection.values()}
    evaluation_coordinates = {_coordinate_key(delta) for delta in evaluation.values()}
    require(not collection_coordinates.intersection(evaluation_coordinates), "collection/evaluation coordinates overlap")
    return {
        "schema": "LOCALPC_PI0208_STAGE45H_COHORT_MANIFEST_V1",
        "status": "SEALED",
        "target_world_m": list(TARGET_WORLD),
        "language": LANGUAGE,
        "source_frame": SOURCE_FRAME,
        "tcp_semantic_id": TCP_ID,
        "collection_condition_order": list(collection),
        "collection_deltas_m": collection,
        "collection_count": len(collection),
        "collection_radii_m": [0.01, 0.025, 0.04, 0.05],
        "collection_phase_rad": math.pi / 32.0,
        "collection_generation_rule": "4 deterministic XY polar rings; 32 angles per ring; radii 0.010/0.025/0.040/0.050 m; phase pi/32; one pre-S0 root-state translation only",
        "evaluation_condition_order": list(evaluation),
        "evaluation_deltas_m": evaluation,
        "evaluation_count": len(evaluation),
        "evaluation_radius_m": 0.035,
        "evaluation_phase_rad": math.pi / 22.0,
        "evaluation_generation_rule": "Stage43 C0-C4 plus 11 deterministic XY points at radius 0.035 m, phase pi/22; disjoint from collection by radius",
        "target_excluded_from_model_input": True,
        "validation_e_read": False,
        "holdout_f_read": False,
        "parent_stage43_bundle_sha256": STAGE43_BUNDLE_SHA256,
    }


def make_schedule(row_ids: Sequence[str], seed: int) -> dict[str, Any]:
    available_rows = sorted({str(row_id) for row_id in row_ids})
    require(len(available_rows) == len(row_ids), "support row IDs must be unique")
    require(len(available_rows) >= SUPPORT_UPDATES, f"need at least {SUPPORT_UPDATES} unique support rows")
    rng = random.Random(int(seed))
    full: list[int] = []
    while len(full) < UPDATE_COUNT:
        cycle = list(range(24))
        rng.shuffle(cycle)
        full.extend(cycle)
    full = full[:UPDATE_COUNT]
    positions = list(range(0, UPDATE_COUNT, 5))
    require(len(positions) == SUPPORT_UPDATES, "support position count drift")
    support = list(available_rows)
    order_rng = random.Random(int(seed) + 45_001)
    order_rng.shuffle(support)
    support = support[:SUPPORT_UPDATES]
    require(len(set(support)) == SUPPORT_UPDATES, "support rows are reused within arm")
    ctrl = [{"position": index, "source": "FULL24", "full24_slot": int(slot), "support_row_id": None} for index, slot in enumerate(full)]
    sup = []
    by_position = dict(zip(positions, support))
    for index, slot in enumerate(full):
        if index in by_position:
            sup.append({"position": index, "source": "STATE_SUPPORT_MEDIUM", "full24_slot": None, "support_row_id": by_position[index]})
        else:
            sup.append({"position": index, "source": "FULL24", "full24_slot": int(slot), "support_row_id": None})
    return {
        "seed": int(seed),
        "update_count": UPDATE_COUNT,
        "support_fraction": SUPPORT_FRACTION,
        "support_update_count": SUPPORT_UPDATES,
        "support_pool_count": len(available_rows),
        "support_pool_sha256": canonical_sha(available_rows),
        "selected_support_row_ids": support,
        "selected_support_row_ids_sha256": canonical_sha(support),
        "master_full24_sequence": full,
        "master_full24_sequence_sha256": canonical_sha(full),
        "support_positions": positions,
        "support_positions_sha256": canonical_sha(positions),
        "MCTRL": ctrl,
        "MSUP": sup,
        "pairing_rule": "MCTRL and MSUP share the exact 2000-position FULL24 master sequence; MSUP replaces positions 0,5,...,1995 with a seed-deterministic without-replacement sample from the complete valid support pool",
    }


def write_cohort(args: argparse.Namespace) -> int:
    manifest = cohort_manifest()
    write_json(args.output, manifest)
    print(json.dumps({"status": "PASS", "output": str(args.output), "sha256": sha256_file(args.output), "collection": manifest["collection_count"], "evaluation": manifest["evaluation_count"]}, sort_keys=True))
    return 0


def _row_file(root: Path, row: Mapping[str, Any], key: str, label: str) -> Path:
    payload = row.get("payload_files")
    require(isinstance(payload, Mapping), f"{label} payload map missing")
    return child(root, str(payload[key]), label)


def load_rows(path: Path, support_root: Path, *, minimum: int = SUPPORT_UPDATES) -> dict[str, dict[str, Any]]:
    regular_file(path, "support row overlay")
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        require(isinstance(row, dict), "support row is not an object")
        row_id = str(row.get("row_id", ""))
        require(row_id and row_id not in rows, f"duplicate support row: {row_id}")
        require(row.get("language") == LANGUAGE, f"support language drift: {row_id}")
        require(row.get("target_excluded_from_model_input", True) is True, f"target leakage binding drift: {row_id}")
        require(row.get("model_input_fields") == ["body_rgb", "wrist_rgb", "state_v1", "language"], f"support input ABI drift: {row_id}")
        _row_file(support_root, row, "visited_row_npz", f"visited row {row_id}")
        _row_file(support_root, row, "expert_label_npz", f"expert row {row_id}")
        rows[row_id] = row
    require(len(rows) >= minimum, f"support row count {len(rows)} < {minimum}")
    require(len(rows) <= MAX_SUPPORT_ROWS, f"support row count {len(rows)} > {MAX_SUPPORT_ROWS}")
    return rows


def write_schedules(args: argparse.Namespace) -> int:
    rows = load_rows(args.rows, args.support_root)
    arms: dict[str, Any] = {}
    for seed in SEEDS:
        schedule = make_schedule(sorted(rows), seed)
        arms[f"MCTRL_{seed}"] = schedule["MCTRL"]
        arms[f"MSUP_{seed}"] = schedule["MSUP"]
    package = {
        "schema": "LOCALPC_PI0208_STAGE45H_MATCHED_SCHEDULES_V1",
        "status": "SEALED",
        "update_count": UPDATE_COUNT,
        "support_updates": SUPPORT_UPDATES,
        "support_fraction": SUPPORT_FRACTION,
        "seeds": list(SEEDS),
        "arm_order": list(ARM_ORDER),
        "row_count": len(rows),
        "row_ids_sha256": canonical_sha(sorted(rows)),
        "arms": arms,
        "pair_metadata": {str(seed): {key: value for key, value in make_schedule(sorted(rows), seed).items() if key not in {"MCTRL", "MSUP"}} for seed in SEEDS},
        "validation_e_read": False,
        "holdout_f_read": False,
    }
    write_json(args.output, package)
    print(json.dumps({"status": "PASS", "output": str(args.output), "sha256": sha256_file(args.output), "arms": {key: len(value) for key, value in arms.items()}, "row_count": len(rows)}, sort_keys=True))
    return 0


def _artifact_record(path: Path, *, tree: bool = False, prefix: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"path": str(path.resolve())}
    if path.is_file() and not path.is_symlink():
        value.update({"bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    else:
        regular_dir(path, "artifact directory")
        if tree:
            digest, entries = tree_digest(path, prefix or path.name)
            value.update({"tree_sha256": digest, "tree_prefix": prefix or path.name, "file_count": len(entries), "bytes": sum(int(item["bytes"]) for item in entries)})
    return value


def prepare_bundle(args: argparse.Namespace) -> int:
    base_path = regular_file(args.stage43_bundle, "Stage43 bundle")
    require(sha256_file(base_path) == STAGE43_BUNDLE_SHA256, "historical Stage43 bundle SHA drift")
    base = read_json(base_path, canonical=True)
    require(args.model_label == BASE_S25_LABEL, "Stage45H collection candidate must be S25@80")
    require(args.model_tree_prefix == BASE_S25_TREE_PREFIX, "Stage45H S25 tree prefix drift")
    model = regular_dir(args.model_path, "candidate checkpoint")
    tree, entries = tree_digest(model, args.model_tree_prefix)
    require(tree == BASE_S25_TREE_SHA256, f"S25@80 candidate tree drift: {tree}")
    if args.expected_tree:
        require(tree == args.expected_tree, f"candidate tree drift: {tree}")
    trainer_state = child(model, "trainer_state.json", "S25@80 trainer state")
    require(int(read_json(trainer_state).get("global_step", -1)) == BASE_S25_GLOBAL_STEP, "S25@80 global step drift")
    stats = regular_file(args.structured_stats, "structured statistics")
    bundle = copy.deepcopy(base)
    bundle["schema"] = "LOCALPC_PI0208_STAGE45H_EXECUTION_BUNDLE_V1"
    bundle["parent_stage43_bundle_sha256"] = STAGE43_BUNDLE_SHA256
    bundle["repository"] = {"branch": args.branch, "head": args.repository_head}
    bundle["stage45h"] = {
        "status": "SEALED",
        "binding_role": "COLLECTION_SOURCE_S25",
        "candidate_label": args.model_label,
        "candidate_tree_sha256": tree,
        "candidate_tree_prefix": args.model_tree_prefix,
        "candidate_file_count": len(entries),
        "candidate_bytes": sum(int(item["bytes"]) for item in entries),
        "candidate_tree_verified": True,
        "target_world_m": list(TARGET_WORLD),
        "language": LANGUAGE,
        "target_excluded_from_model_input": True,
        "validation_e_read": False,
        "holdout_f_read": False,
    }
    artifacts = bundle["artifacts"]
    for key in ("structured_checkpoint", "base_model_path", "model_path"):
        artifacts[key] = {"path": str(model.resolve()), "tree_sha256": tree, "tree_prefix": args.model_tree_prefix, "file_count": len(entries), "bytes": sum(int(item["bytes"]) for item in entries)}
    artifacts["structured_stats"] = {"path": str(stats.resolve()), "sha256": sha256_file(stats), "bytes": int(stats.stat().st_size)}
    bundle["executable_lineage"] = {
        "runner_relative_path": "scripts/stage43_corelab_initial_condition.py",
        "source_relative_path": "scripts/stage43_corelab_evidence.py",
        "runner_sha256": sha256_file(REPO / "scripts" / "stage43_corelab_initial_condition.py"),
        "source_sha256": sha256_file(REPO / "scripts" / "stage43_corelab_evidence.py"),
    }
    bundle["forbidden_operations"] = {str(key): False for key in bundle.get("forbidden_operations", {})}
    write_json(args.output, bundle)
    print(json.dumps({"status": "PASS", "output": str(args.output), "sha256": sha256_file(args.output), "candidate_tree_sha256": tree, "candidate_files": len(entries)}, sort_keys=True))
    return 0


def verify_medium_bundle(bundle: Mapping[str, Any], *, expected_head: str | None = None) -> dict[str, Any]:
    require(bundle.get("schema") == "LOCALPC_PI0208_STAGE45H_EXECUTION_BUNDLE_V1", "Stage45H bundle schema drift")
    require(bundle.get("parent_stage43_bundle_sha256") == STAGE43_BUNDLE_SHA256, "parent Stage43 bundle binding drift")
    if expected_head is not None:
        require(bundle.get("repository", {}).get("head") == expected_head, "Stage45H repository HEAD drift")
    stage = bundle.get("stage45h", {})
    require(stage.get("candidate_tree_verified") is True, "candidate tree was not verified before execution")
    role = stage.get("binding_role")
    require(role in {"COLLECTION_SOURCE_S25", "EVALUATION_CANDIDATE"}, "Stage45H binding role drift")
    if role == "COLLECTION_SOURCE_S25":
        require(stage.get("candidate_label") == BASE_S25_LABEL, "collection candidate label is not S25@80")
        require(stage.get("candidate_tree_sha256") == BASE_S25_TREE_SHA256, "collection candidate tree is not S25@80")
        require(stage.get("candidate_tree_prefix") == BASE_S25_TREE_PREFIX, "collection candidate tree prefix drift")
    artifacts = bundle.get("artifacts", {})
    required_files = ("structured_stats", "modality", "modality_config", "trajectory", "wbc_arm_checkpoint", "wbc_dog_checkpoint")
    required_dirs = ("structured_checkpoint", "tiny1_dataset", "cosmos_snapshot", "wbc_source", "stage10_root", "base_model_path", "model_path", "dataset_dir", "cosmos_snapshot_path")
    for key in required_dirs:
        item = artifacts.get(key)
        require(isinstance(item, Mapping), f"bundle artifact missing: {key}")
        regular_dir(Path(str(item["path"])), f"bundle directory {key}")
    for key in required_files:
        item = artifacts.get(key)
        require(isinstance(item, Mapping), f"bundle artifact missing: {key}")
        path = regular_file(Path(str(item["path"])), f"bundle file {key}")
        if item.get("sha256"):
            require(sha256_file(path) == item["sha256"], f"bundle file SHA drift: {key}")
    stats = artifacts["structured_stats"]
    require(isinstance(stats.get("sha256"), str) and len(stats["sha256"]) == 64, "structured stats binding incomplete")
    forbidden = bundle.get("forbidden_operations", {})
    require(isinstance(forbidden, Mapping) and all(value is False for value in forbidden.values()), "forbidden operation enabled")
    return {"status": "PASS", "candidate_label": stage.get("candidate_label"), "candidate_tree_sha256": stage.get("candidate_tree_sha256"), "candidate_stats_sha256": stats.get("sha256")}


def _gpu_uuid() -> str:
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True)
    rows = [line.strip().split(",") for line in output.splitlines() if line.strip()]
    require(rows, "nvidia-smi returned no GPU")
    return rows[0][1].strip()


def _child_bootstrap(*, runner: Path, bundle: Path, output: Path, condition_id: str, delta_map: Mapping[str, Sequence[float]], repository_head: str, candidate_stats_sha: str, candidate_tree_sha: str, gpu_uuid: str, model_python: Path, repo: Path) -> str:
    ids = tuple(delta_map)
    argv = [
        str(runner), "--bundle", str(bundle), "--output-dir", str(output), "--condition-id", condition_id,
        "--repository-head", repository_head, "--gpu-uuid", gpu_uuid,
        "--stage10-root", str(bundle), "--base-model-path", str(bundle), "--model-path", str(bundle),
        "--dataset-dir", str(bundle), "--modality-config", str(bundle), "--cosmos-snapshot", str(bundle),
        "--structured-stats", str(bundle), "--model-python", str(model_python), "--wbc-source-root", str(bundle),
        "--sim-device", "cuda:0", "--seed", "4242", "--response-timeout-s", "300.0", "--worker-ready-timeout-s", "1200.0",
    ]
    # The runner replaces these placeholder CLI paths with the frozen bundle
    # paths before native/model construction.  Keeping the argv explicit also
    # makes the child invocation auditable in the log.
    return "\n".join([
        "import runpy, sys",
        "from pathlib import Path",
        f"sys.path.insert(0, {str(Path(runner).resolve().parent)!r})",
        f"sys.path.insert(0, {str(repo.resolve() / 'src')!r})",
        f"sys.path.insert(0, {str(repo.resolve())!r})",
        "import stage43_corelab_evidence as _e",
        "import stage45h_pi0208_medium as _h",
        f"_e.COHORT_ORDER = {ids!r}",
        f"_e.COHORT_IDS = {ids!r}",
        f"_e.COHORT_DELTAS_M = {dict(delta_map)!r}",
        f"_e.COHORT_TOTAL_COUNT = {len(ids)!r}",
        f"_e.COHORT_TOTAL_FORWARDS = {len(ids) * 4!r}",
        f"_e.COHORT_TOTAL_UPDATES = {len(ids) * 3!r}",
        f"_e.STRUCTURED_CHECKPOINT_TREE_SHA256 = {candidate_tree_sha!r}",
        f"_e.STRUCTURED_STATS_SHA256 = {candidate_stats_sha!r}",
        "import go2arx5_vla.stage40_structured_decode as _d",
        f"_d.EXPECTED_STRUCTURED_STATS_SHA256 = {candidate_stats_sha!r}",
        f"_original_verify = _h.verify_medium_bundle",
        f"_e.verify_stage43_bundle_files = lambda bundle, repo_root=None: _original_verify(bundle, expected_head={repository_head!r})",
        f"sys.argv = {argv!r}",
        f"runpy.run_path({str(runner)!r}, run_name='__main__')",
    ])


def _run_child(*, runner: Path, bundle: Path, output: Path, condition_id: str, delta_map: Mapping[str, Sequence[float]], repository_head: str, gpu_uuid: str, model_python: Path, candidate_stats_sha: str, candidate_tree_sha: str, log: Path, repo: Path) -> int:
    bootstrap = _child_bootstrap(runner=runner, bundle=bundle, output=output, condition_id=condition_id, delta_map=delta_map, repository_head=repository_head, candidate_stats_sha=candidate_stats_sha, candidate_tree_sha=candidate_tree_sha, gpu_uuid=gpu_uuid, model_python=model_python, repo=repo)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "GROOT_HF_LOCAL_FIRST": "1", "GROOT_PATCH_MISTRAL": "1"})
    native_python = Path(sys.executable)
    resolved_python = native_python.resolve()
    native_public_root = resolved_python.parents[2]
    isaacgym_python = native_public_root / "isaacgym" / "python"
    native_compat_root = Path(os.environ.get("STAGE45H_NATIVE_COMPAT_ROOT", "/home/liuyu-nfs/stage13c_staging/native_compat"))
    require(isaacgym_python.is_dir(), f"historical IsaacGym runtime missing: {isaacgym_python}")
    require(native_compat_root.is_dir(), f"historical native compatibility root missing: {native_compat_root}")
    native_lib_root = resolved_python.parents[1] / "lib"
    native_torch_lib = native_lib_root / "python3.8" / "site-packages" / "torch" / "lib"
    native_preload = Path(os.environ.get("STAGE45H_NATIVE_PRELOAD", "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"))
    require(native_lib_root.is_dir(), f"native Python library root missing: {native_lib_root}")
    require(native_torch_lib.is_dir(), f"native Torch library root missing: {native_torch_lib}")
    require(native_preload.is_file(), f"host C++ runtime missing: {native_preload}")
    runtime_root = output.parent / (output.name + ".runtime")
    for name in ("home", "tmp", "cache", "config", "data", "torch_extensions", "triton"):
        (runtime_root / name).mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(runtime_root / "home")
    env["TMPDIR"] = str(runtime_root / "tmp")
    env["TMP"] = str(runtime_root / "tmp")
    env["TEMP"] = str(runtime_root / "tmp")
    env["XDG_CACHE_HOME"] = str(runtime_root / "cache")
    env["XDG_CONFIG_HOME"] = str(runtime_root / "config")
    env["XDG_DATA_HOME"] = str(runtime_root / "data")
    env["TORCH_EXTENSIONS_DIR"] = str(runtime_root / "torch_extensions")
    env["TRITON_CACHE_DIR"] = str(runtime_root / "triton")
    env["PATH"] = os.pathsep.join([str(resolved_python.parent), env.get("PATH", "")])
    env["LD_PRELOAD"] = str(native_preload)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo / "scripts"), str(repo / "src"), str(isaacgym_python), str(native_compat_root)]
    )
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        ["/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu", str(native_lib_root), str(native_torch_lib)]
        + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
    )
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.run([str(resolved_python), "-B", "-u", "-"], input=bootstrap, text=True, cwd=str(repo), env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return int(process.returncode)


def _condition_pass(root: Path) -> bool:
    result_path = root / "condition_result.json"
    seal_path = root / "condition_seal.json"
    if not result_path.is_file() or not seal_path.is_file():
        return False
    result = read_json(result_path)
    seal = read_json(seal_path)
    return result.get("status") == "PASS" and seal.get("status") == "PASS" and int(result.get("cases", -1)) == 4


def _extract_pack(
    stage_root: Path,
    output_root: Path,
    condition_ids: Sequence[str],
    origin_class: str,
    *,
    cohort_manifest_sha256: str,
    bundle_sha256: str,
    repository_head: str,
) -> list[dict[str, Any]]:
    import stage45f_pi0181_visited_state_support as support

    support.CONDITION_ORDER = tuple(condition_ids)
    support.VISITED_ROW_COUNT = 4 * len(condition_ids)
    rows, _ = support.extract_rows(stage_root, output_root)
    for row in rows:
        condition_root = stage_root / "conditions" / str(row["condition_id"])
        historical = row.pop("source_stage43", None)
        row["source_stage43_input"] = historical
        row["source_stage45h"] = {
            "schema": "LOCALPC_PI0208_STAGE45H_CONDITION_PROVENANCE_V1",
            "repository_head": repository_head,
            "cohort_manifest_sha256": cohort_manifest_sha256,
            "bundle_sha256": bundle_sha256,
            "candidate_label": BASE_S25_LABEL,
            "candidate_tree_sha256": BASE_S25_TREE_SHA256,
            "origin_class": origin_class,
            "condition_manifest_sha256": sha256_file(condition_root / "condition_manifest.json"),
            "condition_result_sha256": sha256_file(condition_root / "condition_result.json"),
            "condition_seal_sha256": sha256_file(condition_root / "condition_seal.json"),
        }
        row["origin_class"] = origin_class
    for manifest_name in ("visited_pack_manifest.json", "expert_label_manifest.json"):
        manifest_path = output_root / manifest_name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        historical = {
            key: manifest.pop(key)
            for key in ("stage43_result_sha256", "stage43_seal_sha256", "stage43_cohort_manifest_sha256")
            if key in manifest
        }
        manifest["source_stage43_input"] = historical
        manifest["source_stage45h"] = {
            "schema": "LOCALPC_PI0208_STAGE45H_PACK_PROVENANCE_V1",
            "repository_head": repository_head,
            "cohort_manifest_sha256": cohort_manifest_sha256,
            "bundle_sha256": bundle_sha256,
            "candidate_label": BASE_S25_LABEL,
            "candidate_tree_sha256": BASE_S25_TREE_SHA256,
            "origin_class": origin_class,
            "condition_ids": list(condition_ids),
        }
        write_json(manifest_path, manifest, overwrite=True)
    rows_path = output_root / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return rows


def collect(args: argparse.Namespace) -> int:
    manifest = read_json(args.cohort_manifest, canonical=True)
    require(manifest == cohort_manifest(), "Stage45H cohort manifest does not match the current frozen generator")
    bundle = read_json(args.bundle, canonical=True)
    verify_medium_bundle(bundle, expected_head=args.repository_head)
    output = args.output.resolve()
    require(not output.exists() or not (output / "collection_result.json").exists(), f"collection root already sealed: {output}")
    output.mkdir(parents=True, exist_ok=True)
    collection_ids = tuple(manifest["collection_condition_order"])
    eval_ids = tuple(manifest["evaluation_condition_order"])
    collection_deltas = manifest["collection_deltas_m"]
    eval_deltas = manifest["evaluation_deltas_m"]
    require(len(collection_ids) == COLLECTION_COUNT, "collection cohort count drift")
    require(len(eval_ids) == EVAL_COUNT, "evaluation cohort count drift")
    gpu_uuid = _gpu_uuid()
    runner = args.runner.resolve()
    completed_collection_ids: list[str] = []
    collection_failures: list[dict[str, Any]] = []
    for condition_id in collection_ids:
        root = output / "collection_cells" / "conditions" / condition_id
        if root.exists():
            if _condition_pass(root):
                completed_collection_ids.append(condition_id)
            else:
                collection_failures.append({"condition_id": condition_id, "status": "INCOMPLETE_EXISTING"})
            continue
        root.parent.mkdir(parents=True, exist_ok=True)
        rc = _run_child(
            runner=runner, bundle=args.bundle.resolve(), output=root, condition_id=condition_id,
            delta_map=collection_deltas, repository_head=args.repository_head, gpu_uuid=gpu_uuid,
            model_python=args.model_python.resolve(), candidate_stats_sha=str(bundle["artifacts"]["structured_stats"]["sha256"]),
            candidate_tree_sha=str(bundle["stage45h"]["candidate_tree_sha256"]), log=output / "child_logs" / f"collection__{condition_id}.log", repo=args.repo.resolve(),
        )
        if _condition_pass(root):
            completed_collection_ids.append(condition_id)
        else:
            collection_failures.append({"condition_id": condition_id, "status": "FAILED", "return_code": rc, "log": str((output / "child_logs" / f"collection__{condition_id}.log").relative_to(output))})

    for condition_id in eval_ids:
        root = output / "evaluation_cells" / "conditions" / condition_id
        if root.exists():
            require(_condition_pass(root), f"existing incomplete evaluation cell: {condition_id}")
            continue
        root.parent.mkdir(parents=True, exist_ok=True)
        rc = _run_child(
            runner=runner, bundle=args.bundle.resolve(), output=root, condition_id=condition_id,
            delta_map=eval_deltas, repository_head=args.repository_head, gpu_uuid=gpu_uuid,
            model_python=args.model_python.resolve(), candidate_stats_sha=str(bundle["artifacts"]["structured_stats"]["sha256"]),
            candidate_tree_sha=str(bundle["stage45h"]["candidate_tree_sha256"]), log=output / "child_logs" / f"evaluation__{condition_id}.log", repo=args.repo.resolve(),
        )
        require(_condition_pass(root), f"evaluation cell failed: {condition_id} rc={rc}")

    require(len(completed_collection_ids) * 4 >= SUPPORT_UPDATES, "completed collection cells cannot supply 400 rows")
    collection_pack = output / "collection_pack"
    eval_pack = output / "evaluation_pack"
    collection_rows = _extract_pack(output / "collection_cells", collection_pack, completed_collection_ids, "AUTONOMOUS_VISITED", cohort_manifest_sha256=sha256_file(args.cohort_manifest), bundle_sha256=sha256_file(args.bundle), repository_head=args.repository_head)
    eval_rows = _extract_pack(output / "evaluation_cells", eval_pack, eval_ids, "MEDIUM_HELD_OUT_EVAL_ONLY", cohort_manifest_sha256=sha256_file(args.cohort_manifest), bundle_sha256=sha256_file(args.bundle), repository_head=args.repository_head)
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in collection_rows:
        identity = (str(row["body_sha256"]), str(row["wrist_sha256"]), str(row["state_v1_sha256"]), str(row["language"]))
        if identity not in seen:
            seen.add(identity)
            unique.append(row)
    require(len(unique) >= SUPPORT_UPDATES, f"unique valid support rows {len(unique)} < {SUPPORT_UPDATES}")
    medium_rows = output / "medium_rows.jsonl"
    medium_eval_rows = output / "medium_eval_rows.jsonl"
    for path, rows in ((medium_rows, unique), (medium_eval_rows, eval_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    result = {
        "schema": "LOCALPC_PI0208_STAGE45H_COLLECTION_RESULT_V1",
        "status": "PASS",
        "cohort_manifest_sha256": sha256_file(args.cohort_manifest),
        "bundle_sha256": sha256_file(args.bundle),
        "candidate_tree_sha256": bundle["stage45h"]["candidate_tree_sha256"],
        "candidate_label": bundle["stage45h"]["candidate_label"],
        "gpu_uuid": gpu_uuid,
        "collection_condition_target_count": len(collection_ids),
        "collection_condition_completed_count": len(completed_collection_ids),
        "collection_failures": collection_failures,
        "evaluation_condition_count": len(eval_ids),
        "collection_rows": len(collection_rows),
        "unique_valid_support_rows": len(unique),
        "evaluation_rows": len(eval_rows),
        "medium_rows_sha256": sha256_file(medium_rows),
        "medium_eval_rows_sha256": sha256_file(medium_eval_rows),
        "new_model_forwards": (len(completed_collection_ids) + len(eval_ids)) * 4,
        "new_wbc_ticks": (len(completed_collection_ids) + len(eval_ids)) * (49 + 49 + 49 + 98),
        "expert_relabels": len(unique),
        "validation_e_read": False,
        "holdout_f_read": False,
    }
    write_json(output / "collection_result.json", result)
    print(json.dumps({"status": "PASS", "output": str(output / 'collection_result.json'), "unique_valid_support_rows": len(unique), "evaluation_rows": len(eval_rows), "gpu_uuid": gpu_uuid}, sort_keys=True))
    return 0


def _validate_training_inputs(args: argparse.Namespace, rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    for name in ("dataset", "base_model", "groot_source", "groot_manifest", "cosmos_root", "modality", "config", "stats", "ffmpeg_root"):
        require(getattr(args, name) is not None, f"missing training input: {name}")
    from stage45f_pi0183_matched_training import tree_digest as old_tree_digest

    base_tree, base_entries = old_tree_digest(args.base_model, BASE_S25_TREE_PREFIX)
    require(base_tree == BASE_S25_TREE_SHA256, f"S25@80 base tree drift: {base_tree}")
    trainer_state = read_json(child(args.base_model, "trainer_state.json", "S25@80 trainer state"))
    require(int(trainer_state.get("global_step", -1)) == BASE_S25_GLOBAL_STEP, "S25@80 global step drift")
    dataset_tree, dataset_entries = old_tree_digest(args.dataset, "payload/lerobot_train")
    require(dataset_tree == FULL24_TREE_SHA256 and len(dataset_entries) == 85, f"FULL24 tree drift: {dataset_tree}")
    require(sha256_file(args.groot_manifest) == GROOT_MANIFEST_SHA256, "GR00T manifest SHA drift")
    require(sha256_file(args.modality) == MODALITY_SHA256, "modality SHA drift")
    require(sha256_file(args.config) == CONFIG_SHA256, "training config SHA drift")
    require(sha256_file(args.stats) == STRUCTURED_SOURCE_STATS_SHA256, "structured source stats SHA drift")
    from stage45b_groot_source_snapshot import verify_extracted

    groot = verify_extracted(args.groot_source, args.groot_manifest, expected_manifest_file_sha256=GROOT_MANIFEST_SHA256, expected_manifest_payload_sha256=GROOT_MANIFEST_PAYLOAD_SHA256, expected_payload_tree_sha256=GROOT_PAYLOAD_TREE_SHA256)
    regular_dir(args.cosmos_root, "Cosmos cache")
    refs = child(args.cosmos_root, "refs/main", "Cosmos refs/main")
    require(refs.read_text(encoding="utf-8").strip() == COSMOS_REVISION, "Cosmos revision drift")
    regular_dir(args.ffmpeg_root / "lib", "FFmpeg lib")
    regular_dir(args.ffmpeg_root / "compat", "FFmpeg compat")
    require(len(rows) >= SUPPORT_UPDATES, "support row count below schedule requirement")
    return {"base_label": BASE_S25_LABEL, "base_global_step": BASE_S25_GLOBAL_STEP, "base_tree_sha256": base_tree, "base_file_count": len(base_entries), "dataset_tree_sha256": dataset_tree, "dataset_file_count": len(dataset_entries), "groot": groot, "stats_sha256": sha256_file(args.stats), "rows": len(rows)}


def train_one(args: argparse.Namespace, schedule: Mapping[str, Any], rows: Mapping[str, Mapping[str, Any]], arm: str, seed: int) -> dict[str, Any]:
    import stage45f_pi0183_matched_training as training

    output = args.output_root / arm
    if output.exists():
        result_path = output / "training_result.json"
        if result_path.is_file():
            result = read_json(result_path)
            if result.get("status") == "PASS" and int(result.get("global_step", -1)) == UPDATE_COUNT:
                return {"arm": arm, "status": "SKIPPED_EXISTING_PASS", "result_sha256": sha256_file(result_path), "checkpoint_tree_sha256": result.get("checkpoint_tree_sha256")}
        raise Stage45HError(f"training output exists but is not a completed PASS: {output}")
    env = training.prepare_runtime(output, args.cosmos_root, args.ffmpeg_root, args.groot_source, trace_only=False)
    os.environ.update({
        "STAGE45F_FROZEN_STRUCTURED_STATS_PATH": str(args.stats),
        "STAGE45F_FROZEN_STRUCTURED_STATS_SHA256": STRUCTURED_SOURCE_STATS_SHA256,
        "STAGE45H_ARM": arm,
        "STAGE45H_SEED": str(seed),
        "STAGE45H_SUPPORT_ROOT": str(args.support_root),
    })
    training.bind_cosmos_processor(args.cosmos_root)
    arm_schedule = {arm: list(schedule["arms"][arm])}
    train_args = argparse.Namespace(**vars(args))
    train_args.arm = arm
    train_args.schedule = args.schedule
    config = training.load_config(train_args, output, UPDATE_COUNT, seed=seed)
    training.install_scheduled_factory(config, train_args, arm_schedule, rows, training.PROCESSOR_CONFIG_SHA256)
    sequence = list(schedule["arms"][arm])
    invocation = {
        "schema": "LOCALPC_PI0208_STAGE45H_TRAINING_INVOCATION_V1",
        "status": "STARTED",
        "arm": arm,
        "seed": seed,
        "process_pid": os.getpid(),
        "updates": UPDATE_COUNT,
        "schedule_sha256": sha256_file(args.schedule),
        "arm_sequence_sha256": canonical_sha(sequence),
        "base_checkpoint_label": BASE_S25_LABEL,
        "base_checkpoint_global_step": BASE_S25_GLOBAL_STEP,
        "base_checkpoint_tree_sha256": BASE_S25_TREE_SHA256,
        "support_fraction": SUPPORT_FRACTION if arm.startswith("MSUP") else 0.0,
        "support_updates": SUPPORT_UPDATES if arm.startswith("MSUP") else 0,
        "runtime_environment": env,
        "science_contract": {"optimizer": "AdamW", "learning_rate": 1.0e-4, "weight_decay": 1.0e-5, "scheduler": "cosine", "warmup_steps": 10, "batch_size": 1, "gradient_accumulation_steps": 1, "precision": "bf16", "trainable_modules": ["projector", "diffusion"], "frozen_modules": ["visual", "llm", "vlln"], "fresh_process_optimizer_scheduler_rng": True},
        "Validation-E": 0,
        "Holdout-F": 0,
        "WBC": 0,
        "simulator": 0,
        "hardware": 0,
    }
    write_json(output / "training_invocation.json", invocation)
    from gr00t.experiment.experiment import run as gr00t_run

    gr00t_run(config)
    checkpoint = output / f"checkpoint-{UPDATE_COUNT}"
    state = read_json(child(checkpoint, "trainer_state.json", "final trainer state"))
    require(int(state.get("global_step", -1)) == UPDATE_COUNT, f"{arm} global step drift")
    prefix = f"stage45h/{arm}/checkpoint-{UPDATE_COUNT}"
    checkpoint_tree, entries = training.tree_digest(checkpoint, prefix)
    result = {
        "schema": "LOCALPC_PI0208_STAGE45H_TRAINING_RESULT_V1",
        "status": "PASS",
        "arm": arm,
        "seed": seed,
        "output_root": str(output),
        "final_checkpoint": str(checkpoint),
        "global_step": UPDATE_COUNT,
        "updates": UPDATE_COUNT,
        "schedule_sha256": sha256_file(args.schedule),
        "arm_sequence_sha256": canonical_sha(sequence),
        "base_checkpoint_label": BASE_S25_LABEL,
        "base_checkpoint_global_step": BASE_S25_GLOBAL_STEP,
        "base_checkpoint_tree_sha256": BASE_S25_TREE_SHA256,
        "checkpoint_tree_sha256": checkpoint_tree,
        "checkpoint_tree_prefix": prefix,
        "checkpoint_file_count": len(entries),
        "checkpoint_bytes": sum(int(item["bytes"]) for item in entries),
        "model_forward": UPDATE_COUNT,
        "backward": UPDATE_COUNT,
        "optimizer_steps": UPDATE_COUNT,
        "support_updates": SUPPORT_UPDATES if arm.startswith("MSUP") else 0,
        "support_fraction": SUPPORT_FRACTION if arm.startswith("MSUP") else 0.0,
        "identities": {"base_checkpoint": {"label": BASE_S25_LABEL, "global_step": BASE_S25_GLOBAL_STEP, "tree_sha256": BASE_S25_TREE_SHA256}, "structured_stats_sha256": STRUCTURED_SOURCE_STATS_SHA256, "full24_tree_sha256": FULL24_TREE_SHA256},
        "Validation-E": 0,
        "Holdout-F": 0,
        "WBC": 0,
        "simulator": 0,
        "hardware": 0,
    }
    write_json(output / "training_result.json", result)
    return {"arm": arm, "status": "PASS", "checkpoint_tree_sha256": checkpoint_tree, "result_sha256": sha256_file(output / "training_result.json")}


def train_all(args: argparse.Namespace) -> int:
    schedule = read_json(args.schedule, canonical=True)
    rows = load_rows(args.rows, args.support_root)
    identities = _validate_training_inputs(args, rows)
    records = []
    for arm in ARM_ORDER:
        seed = int(arm.split("_")[-1])
        command = [
            sys.executable, str(Path(__file__).resolve()), "train-arm",
            "--arm", arm, "--schedule", str(args.schedule), "--rows", str(args.rows),
            "--support-root", str(args.support_root), "--output-root", str(args.output_root),
            "--dataset", str(args.dataset), "--base-model", str(args.base_model),
            "--groot-source", str(args.groot_source), "--groot-manifest", str(args.groot_manifest),
            "--cosmos-root", str(args.cosmos_root), "--modality", str(args.modality),
            "--config", str(args.config), "--stats", str(args.stats), "--ffmpeg-root", str(args.ffmpeg_root),
        ]
        process = subprocess.run(command, cwd=str(REPO), env=os.environ.copy(), check=False)
        require(process.returncode == 0, f"{arm} fresh training process failed: rc={process.returncode}")
        result_path = args.output_root / arm / "training_result.json"
        result = read_json(result_path)
        require(result.get("status") == "PASS" and int(result.get("global_step", -1)) == UPDATE_COUNT, f"{arm} child result is not PASS")
        records.append({"arm": arm, "status": "PASS", "checkpoint_tree_sha256": result.get("checkpoint_tree_sha256"), "result_sha256": sha256_file(result_path)})
    result = {"schema": "LOCALPC_PI0208_STAGE45H_TRAINING_RESULT_SET_V1", "status": "PASS", "arm_order": list(ARM_ORDER), "records": records, "identities": identities, "updates_per_arm": UPDATE_COUNT, "total_forward": len(ARM_ORDER) * UPDATE_COUNT, "total_backward": len(ARM_ORDER) * UPDATE_COUNT, "total_optimizer_steps": len(ARM_ORDER) * UPDATE_COUNT, "Validation-E": 0, "Holdout-F": 0}
    write_json(args.output_root / "training_result_set.json", result, overwrite=True)
    print(json.dumps({"status": "PASS", "output": str(args.output_root / 'training_result_set.json'), "arms": len(records), "updates_per_arm": UPDATE_COUNT}, sort_keys=True))
    return 0


def train_arm(args: argparse.Namespace) -> int:
    require(args.arm in ARM_ORDER, f"unknown training arm: {args.arm}")
    schedule = read_json(args.schedule, canonical=True)
    rows = load_rows(args.rows, args.support_root)
    result = train_one(args, schedule, rows, args.arm, int(args.arm.split("_")[-1]))
    print(json.dumps({"status": "PASS", "arm": args.arm, "result": result}, sort_keys=True))
    return 0


def _medium_screen_support_windows(processor: Any, embodiment: Any, rows: Mapping[str, Mapping[str, Any]], support_root: Path) -> list[dict[str, Any]]:
    import numpy as np
    import stage45g_pi0208_hopper_model_screen as screen

    windows: list[dict[str, Any]] = []
    for row in rows.values():
        row_id = str(row["row_id"])
        visited_path = _row_file(support_root, row, "visited_row_npz", f"evaluation row {row_id}")
        expert_path = _row_file(support_root, row, "expert_label_npz", f"evaluation expert {row_id}")
        with np.load(visited_path, allow_pickle=False) as archive:
            body = np.asarray(archive["body_rgb"], dtype=np.uint8)
            wrist = np.asarray(archive["wrist_rgb"], dtype=np.uint8)
            state = np.asarray(archive["state_v1"], dtype=np.float32).reshape(-1)
            anchor = np.asarray(archive["anchor_world_m"], dtype=np.float64).reshape(-1)
            target = np.asarray(archive["target_world_m"], dtype=np.float64).reshape(-1)
        with np.load(expert_path, allow_pickle=False) as archive:
            timestamps = np.asarray(archive["source_timestamps"], dtype=np.float64)
        require(body.shape == (96, 96, 3) and wrist.shape == (96, 96, 3) and state.shape == (10,), f"medium screen input shape drift: {row_id}")
        require(timestamps.shape == (HORIZON,) and np.isfinite(timestamps).all(), f"medium screen timestamp drift: {row_id}")
        require(np.array_equal(target, np.asarray(TARGET_WORLD, dtype=np.float64)), f"medium screen target drift: {row_id}")
        windows.append({
            "surface": "MEDIUM_SUPPORT_EVAL",
            "index": len(windows),
            "window_id": row_id,
            "model_inputs": screen._model_inputs(processor, embodiment, body, wrist, state, row_id),
            "snapshot": {
                "window_id": row_id,
                "target_world_m": target.tolist(),
                "state_v1": state.tolist(),
                "source_timestamps": timestamps.tolist(),
                "observation_timestamp": float(row["observation_timestamp_s"]),
                "anchor_world_m": anchor.tolist(),
                "target_excluded_from_model_input": True,
            },
            "support_origin_class": row.get("origin_class", "MEDIUM_HELD_OUT_EVAL_ONLY"),
        })
    require(len(windows) == EVAL_COUNT, f"medium screen support window count drift: {len(windows)}")
    return windows


def _medium_screen_summary(rows: Sequence[Mapping[str, Any]], surface_counts: Mapping[str, int]) -> dict[str, Any]:
    import numpy as np

    summary: dict[str, Any] = {}
    for surface, expected in surface_counts.items():
        cohort = [row for row in rows if row["surface"] == surface]
        require(len(cohort) == expected, f"screen surface count drift: {surface}={len(cohort)}")
        values = lambda getter: np.asarray([float(getter(row)) for row in cohort], dtype=np.float64)
        trajectory = values(lambda row: row["metrics"]["position_rmse_m"])
        endpoint = values(lambda row: row["metrics"]["endpoint_error_m"])
        progress = values(lambda row: row["metrics"]["target"]["absolute_progress_m"])
        final_distance = values(lambda row: row["metrics"]["target"]["final_target_distance_m"])
        path = values(lambda row: row["metrics"]["target"]["path_length_m"])
        roughness = values(lambda row: row["metrics"]["target"]["roughness_position_second_difference_m"]["mean"])
        require(all(bool(np.isfinite(value).all()) for value in (trajectory, endpoint, progress, final_distance, path, roughness)), f"screen nonfinite metrics: {surface}")
        summary[surface] = {
            "count": len(cohort),
            "mean_trajectory_error_m": float(np.mean(trajectory)),
            "mean_endpoint_error_m": float(np.mean(endpoint)),
            "mean_absolute_progress_m": float(np.mean(progress)),
            "mean_final_target_distance_m": float(np.mean(final_distance)),
            "mean_path_length_m": float(np.mean(path)),
            "mean_roughness_m": float(np.mean(roughness)),
            "safety_gate_counts": {decision: int(sum(row["safety_gate_decision"] == decision for row in cohort)) for decision in ("ACCEPT", "WARN", "REJECT")},
            "finite_count": int(sum(bool(row["finite"]) for row in cohort)),
            "abi_valid_count": int(sum(bool(row["abi_valid"]) for row in cohort)),
        }
    return summary


def _medium_screen_rule(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    seed_rows: dict[str, dict[str, Mapping[str, Any]]] = {}
    for arm in ARM_ORDER:
        _, seed = arm.split("_")
        seed_rows.setdefault(seed, {})[arm.split("_")[0]] = summaries[arm]
    per_seed: dict[str, Any] = {}
    for seed, pair in seed_rows.items():
        ctrl = pair.get("MCTRL")
        sup = pair.get("MSUP")
        require(ctrl is not None and sup is not None, f"screen pair missing for seed {seed}")
        ctrl_support = ctrl["MEDIUM_SUPPORT_EVAL"]
        sup_support = sup["MEDIUM_SUPPORT_EVAL"]
        ctrl_full = ctrl["FULL24"]
        sup_full = sup["FULL24"]
        no_reject = sum(sup[surface]["safety_gate_counts"]["REJECT"] for surface in ("FULL24", "MEDIUM_SUPPORT_EVAL")) == 0
        per_seed[seed] = {
            "support_trajectory_lower": sup_support["mean_trajectory_error_m"] < ctrl_support["mean_trajectory_error_m"],
            "support_endpoint_lower": sup_support["mean_endpoint_error_m"] < ctrl_support["mean_endpoint_error_m"],
            "full24_trajectory_within_1_10x": sup_full["mean_trajectory_error_m"] <= 1.10 * ctrl_full["mean_trajectory_error_m"],
            "full24_endpoint_within_1_10x": sup_full["mean_endpoint_error_m"] <= 1.10 * ctrl_full["mean_endpoint_error_m"],
            "no_new_nonfinite_abi_or_reject": sup_support["finite_count"] == EVAL_COUNT and sup_full["finite_count"] == 24 and sup_support["abi_valid_count"] == EVAL_COUNT and sup_full["abi_valid_count"] == 24 and no_reject,
        }
        per_seed[seed]["positive"] = all(per_seed[seed].values())
    positive = sum(bool(value["positive"]) for value in per_seed.values())
    return {
        "classification": "MEDIUM_HOPPER_SCREEN_POSITIVE" if positive >= 2 else "MEDIUM_HOPPER_SCREEN_NOT_QUALIFIED",
        "positive_seed_count": positive,
        "seed_count": len(per_seed),
        "per_seed": per_seed,
        "rule": "at least 2/3 matched seeds improve both held-out support trajectory and endpoint errors while FULL24 remains within 1.10x and no new invalid/reject occurs",
    }


def screen(args: argparse.Namespace) -> int:
    import numpy as np
    import stage45c_h200_eval_runtime_adapter as stage45c
    import stage45g_pi0208_hopper_model_screen as screen_impl
    import stage45f_pi0183_matched_training as training

    output = args.output_root.resolve()
    require(not output.exists() and not output.is_symlink(), f"screen output must be fresh: {output}")
    require(args.expected_head, "--expected-head required")
    require(sha256_file(args.stats) == screen_impl.STATS_SHA256, "FULL24 runtime stats SHA drift")
    dataset_tree, dataset_entries = tree_digest(args.dataset, "payload/lerobot_train")
    require(dataset_tree == FULL24_TREE_SHA256 and len(dataset_entries) == 85, f"FULL24 dataset drift: {dataset_tree}")
    require(sha256_file(args.groot_manifest) == GROOT_MANIFEST_SHA256, "GR00T manifest SHA drift")
    require(sha256_file(args.modality) == MODALITY_SHA256, "modality SHA drift")
    require(sha256_file(args.config) == CONFIG_SHA256, "training config SHA drift")
    from stage45b_groot_source_snapshot import verify_extracted

    groot_identity = verify_extracted(args.groot_source, args.groot_manifest, expected_manifest_file_sha256=GROOT_MANIFEST_SHA256, expected_manifest_payload_sha256=GROOT_MANIFEST_PAYLOAD_SHA256, expected_payload_tree_sha256=GROOT_PAYLOAD_TREE_SHA256)
    rows = load_rows(args.rows, args.support_root, minimum=1)
    require(len(rows) == EVAL_COUNT, f"medium evaluation row count drift: {len(rows)}")
    training_root = args.training_root.resolve()
    candidate_paths: dict[str, Path] = {}
    candidate_records: dict[str, Any] = {}
    for arm in ARM_ORDER:
        arm_root = training_root / arm
        result = read_json(arm_root / "training_result.json")
        require(result.get("status") == "PASS" and int(result.get("global_step", -1)) == UPDATE_COUNT, f"{arm} training result drift")
        checkpoint = arm_root / f"checkpoint-{UPDATE_COUNT}"
        regular_dir(checkpoint, f"{arm} checkpoint")
        require(result.get("checkpoint_tree_sha256"), f"{arm} checkpoint tree is missing")
        candidate_paths[arm] = checkpoint
        candidate_records[arm] = {"tree_sha256": result["checkpoint_tree_sha256"], "tree_prefix": result.get("checkpoint_tree_prefix"), "training_result_sha256": sha256_file(arm_root / "training_result.json")}
    processor_hashes = {sha256_file(path / "processor_config.json") for path in candidate_paths.values()}
    statistics_hashes = {sha256_file(path / "statistics.json") for path in candidate_paths.values()}
    require(len(processor_hashes) == 1 and len(statistics_hashes) == 1, "candidate-local processor/statistics drift")
    output.mkdir(parents=True, exist_ok=False)
    env = training.prepare_runtime(output, args.cosmos_root, args.ffmpeg_root, args.groot_source, trace_only=False)
    write_json(output / "pre_model_audit.json", {"schema": "LOCALPC_PI0208_STAGE45H_SCREEN_PRE_MODEL_AUDIT_V1", "status": "PASS", "expected_head": args.expected_head, "dataset_tree_sha256": dataset_tree, "groot": groot_identity, "candidate_records": candidate_records, "runtime_environment": env, "model_loads": 0, "model_forwards": 0, "WBC": 0, "simulator": 0, "training": 0, "Validation-E": 0, "Holdout-F": 0})
    from stage26_h1_c1_fixed_noise_local_flow import _build_runtime_processor

    first = candidate_paths[ARM_ORDER[0]]
    runtime_processor, embodiment, modality_configs = _build_runtime_processor(groot_source=args.groot_source.resolve(), base_checkpoint=first, dataset_dir=args.dataset.resolve(), modality_path=args.modality.resolve(), cosmos_snapshot=args.cosmos_root.resolve() / "snapshots" / COSMOS_REVISION, expected_processor_config_sha256=next(iter(processor_hashes)), expected_statistics_sha256=next(iter(statistics_hashes)))
    snapshots = stage45c._verify_train24_descriptor_closure(args.snapshot_root, args.dataset)
    full_windows = stage45c._prepare_train24_windows(runtime_processor, embodiment, modality_configs, args.dataset.resolve(), snapshots["snapshots"])
    for index, window in enumerate(full_windows):
        window["surface"] = "FULL24"
        window["index"] = index
    support_windows = _medium_screen_support_windows(runtime_processor, embodiment, rows, args.support_root.resolve())
    windows = full_windows + support_windows
    require(len(windows) == 24 + EVAL_COUNT, "medium screen window count drift")
    write_json(output / "screen_manifest.json", {"schema": "LOCALPC_PI0208_STAGE45H_MEDIUM_SCREEN_MANIFEST_V1", "status": "SEALED_BEFORE_MODEL_LOAD", "checkpoint_order": list(ARM_ORDER), "surface_order": ["FULL24", "MEDIUM_SUPPORT_EVAL"], "window_ids": {surface: [window["window_id"] for window in windows if window["surface"] == surface] for surface in ("FULL24", "MEDIUM_SUPPORT_EVAL")}, "seed": 4242, "noise_shape": [1, 40, 132], "same_noise_across_checkpoints": True, "sampler": {"timesteps": list(stage45c.TIMESTEP_IDS), "dt": stage45c.SAMPLER_DT, "carried_state": True, "official_public_sampler_called": False}, "decoder_stats_sha256": screen_impl.STATS_SHA256, "target_xyz_in_model_input": False, "evaluation_rows_sha256": sha256_file(args.rows)})
    torch, load_model = stage45c._load_torch_runtime()
    require(torch.cuda.is_available(), "CUDA unavailable")
    name = torch.cuda.get_device_name(torch.device("cuda:0")).upper()
    require("H100" in name or "H200" in name, f"unsupported Hopper screen GPU: {name}")
    noise_bank = screen_impl._noise_bank(torch, len(windows))
    noise_hashes = [sha256_bytes(noise.detach().cpu().numpy().tobytes()) for noise in noise_bank]
    write_json(output / "sampler_noise_manifest.json", {"schema": "LOCALPC_PI0208_STAGE45H_MEDIUM_SCREEN_NOISE_V1", "seed": 4242, "window_ids": [window["window_id"] for window in windows], "noise_sha256": noise_hashes, "same_noise_across_checkpoints": True})
    raw_manifest = output / "raw_predictions.jsonl"
    derived_manifest = output / "derived_rows.jsonl"
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    counters = {"model_construction": 0, "conditioning_passes": 0, "action_head_calls": 0, "model_forward": 0, "raw_prediction_count": 0, "derived_row_count": 0}
    runtime_sha = sha256_bytes(json.dumps({"head": args.expected_head, "decoder": stage45c.DECODER_SOURCE_SHA256, "metrics": stage45c.METRICS_SOURCE_SHA256, "stats": screen_impl.STATS_SHA256}, sort_keys=True).encode())
    for arm in ARM_ORDER:
        checkpoint = candidate_paths[arm]
        model = load_model(checkpoint, "cuda:0")
        model.eval()
        counters["model_construction"] += 1
        hook_count = {"count": 0}
        hook = model.action_head.model.register_forward_hook(lambda *_args, **_kwargs: hook_count.__setitem__("count", hook_count["count"] + 1))
        arm_rows: list[dict[str, Any]] = []
        try:
            for index, window in enumerate(windows):
                with torch.inference_mode():
                    raw, action_calls = stage45c._sample_one(model, window["model_inputs"], noise_bank[index], torch)
                counters["conditioning_passes"] += 1
                counters["action_head_calls"] += action_calls
                counters["model_forward"] += action_calls
                counters["raw_prediction_count"] += 1
                raw_np = raw.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
                require(raw_np.shape == (1, 40, 132) and bool(np.isfinite(raw_np).all()), f"screen raw invalid: {arm}/{window['window_id']}")
                raw_path = output / "raw" / arm / f"{window['surface'].lower()}-{index:03d}.npy"
                screen_impl.write_npy(raw_path, raw_np)
                raw_file_sha = sha256_file(raw_path)
                append_jsonl(raw_manifest, {"schema": "LOCALPC_PI0208_STAGE45H_MEDIUM_SCREEN_RAW_V1", "arm": arm, "window_id": window["window_id"], "surface": window["surface"], "path": str(raw_path.relative_to(output)), "file_sha256": raw_file_sha, "array_sha256": sha256_bytes(raw_np.tobytes()), "shape": [1, 40, 132], "dtype": "float32", "noise_sha256": noise_hashes[index], "decoded": False})
                snapshot = dict(window["snapshot"])
                snapshot["surface"] = window["surface"]
                physical, details = stage45c._decode_v2(raw_np, snapshot["state_v1"], args.stats.resolve(), snapshot["source_timestamps"], float(snapshot["observation_timestamp"]), f"{arm}:{window['window_id']}")
                physical_np = np.asarray(physical)
                require(physical_np.shape == (40, 10) and bool(np.isfinite(physical_np).all()), f"screen decoded invalid: {arm}/{window['window_id']}")
                row = stage45c._evaluation_row(raw_np, physical_np, details, {"snapshot": snapshot, "window_id": window["window_id"]}, {"label": arm, "tree_sha256": candidate_records[arm]["tree_sha256"]}, runtime_sha, str(raw_path.relative_to(output)), raw_file_sha)
                decoded_path = output / "decoded" / arm / f"{window['surface'].lower()}-{index:03d}.npy"
                screen_impl.write_npy(decoded_path, np.asarray(physical_np, dtype=np.float64))
                profile, gate = screen_impl._profile_gate(np.asarray(physical_np, dtype=np.float64), snapshot, f"{arm}:{window['window_id']}")
                row.update({"surface": window["surface"], "support_origin_class": window.get("support_origin_class"), "decoded_artifact": {"path": str(decoded_path.relative_to(output)), "file_sha256": sha256_file(decoded_path), "array_sha256": sha256_bytes(np.ascontiguousarray(np.asarray(physical_np, dtype=np.float64)).tobytes()), "shape": [40, 10]}, "profile": profile, "safety_gate": gate, "safety_gate_decision": gate["decision"], "finite": True, "abi_valid": True, "target_xyz_in_model_input": False, "trajectory_correction_applied": False})
                row_path = output / "rows" / arm / f"{window['surface'].lower()}-{index:03d}.json"
                write_json(row_path, row)
                append_jsonl(derived_manifest, {"schema": "LOCALPC_PI0208_STAGE45H_MEDIUM_SCREEN_DERIVED_V1", "arm": arm, "window_id": window["window_id"], "surface": window["surface"], "raw_path": str(raw_path.relative_to(output)), "decoded_path": str(decoded_path.relative_to(output)), "row_path": str(row_path.relative_to(output)), "decoded": True})
                counters["derived_row_count"] += 1
                arm_rows.append(row)
        finally:
            hook.remove()
            del model
            torch.cuda.empty_cache()
        require(hook_count["count"] == len(windows) * len(stage45c.TIMESTEP_IDS), f"screen action-head count drift: {arm}")
        rows_by_arm[arm] = arm_rows
        summaries[arm] = _medium_screen_summary(arm_rows, {"FULL24": 24, "MEDIUM_SUPPORT_EVAL": EVAL_COUNT})
    require(counters == {"model_construction": len(ARM_ORDER), "conditioning_passes": len(ARM_ORDER) * len(windows), "action_head_calls": len(ARM_ORDER) * len(windows) * 4, "model_forward": len(ARM_ORDER) * len(windows) * 4, "raw_prediction_count": len(ARM_ORDER) * len(windows), "derived_row_count": len(ARM_ORDER) * len(windows)}, f"screen counters drift: {counters}")
    result = {"schema": "LOCALPC_PI0208_STAGE45H_MEDIUM_HOPPER_SCREEN_V1", "status": "PASS", "execution_head": args.expected_head, "checkpoint_order": list(ARM_ORDER), "summaries": summaries, "rule": _medium_screen_rule(summaries), "counters": counters, "gpu": {"name": torch.cuda.get_device_name(torch.device("cuda:0")), "uuid": _gpu_uuid()}, "scientific_zeroes": {"training": 0, "backward": 0, "optimizer_steps": 0, "WBC": 0, "simulator": 0, "hardware": 0, "Validation-E": 0, "Holdout-F": 0}, "candidate_records": candidate_records}
    write_json(output / "screen_result.json", result)
    result["terminal_token"] = "LOCALPC_PI0208_MEDIUM_HOPPER_SCREEN — " + ("POSITIVE" if result["rule"]["classification"] == "MEDIUM_HOPPER_SCREEN_POSITIVE" else "NOT_QUALIFIED")
    write_json(output / "screen_seal.json", result)
    print(json.dumps({"status": result["status"], "terminal_token": result["terminal_token"], "rule": result["rule"], "counters": counters}, sort_keys=True))
    return 0


def _candidate_result(training_root: Path, arm: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    arm_root = training_root / arm
    result_path = arm_root / "training_result.json"
    result = read_json(result_path)
    require(result.get("status") == "PASS" and int(result.get("global_step", -1)) == UPDATE_COUNT, f"{arm} training result is not PASS")
    checkpoint = arm_root / f"checkpoint-{UPDATE_COUNT}"
    regular_dir(checkpoint, f"{arm} checkpoint")
    require(isinstance(result.get("checkpoint_tree_sha256"), str) and len(result["checkpoint_tree_sha256"]) == 64, f"{arm} checkpoint tree record missing")
    return checkpoint, result, {"training_result_sha256": sha256_file(result_path), "checkpoint_tree_sha256": result["checkpoint_tree_sha256"], "checkpoint_tree_prefix": result.get("checkpoint_tree_prefix"), "checkpoint_path": str(checkpoint.resolve())}


def _candidate_bundle(template: Mapping[str, Any], checkpoint: Path, result: Mapping[str, Any], output: Path, *, head: str, arm: str) -> Path:
    bundle = copy.deepcopy(template)
    stats = regular_file(checkpoint / "experiment_cfg" / "structured_action_statistics.json", f"{arm} structured stats")
    tree = str(result["checkpoint_tree_sha256"])
    prefix = str(result.get("checkpoint_tree_prefix") or f"stage45h/{arm}/checkpoint-{UPDATE_COUNT}")
    artifacts = bundle["artifacts"]
    for key in ("structured_checkpoint", "base_model_path", "model_path"):
        artifacts[key] = {"path": str(checkpoint.resolve()), "tree_sha256": tree, "tree_prefix": prefix, "candidate_tree_verified": True}
    artifacts["structured_stats"] = {"path": str(stats.resolve()), "sha256": sha256_file(stats), "bytes": int(stats.stat().st_size)}
    bundle["repository"] = {"branch": "stage45h/medium-dagger-lite-state-support", "head": head}
    bundle["stage45h"] = {
        "status": "SEALED",
        "binding_role": "EVALUATION_CANDIDATE",
        "candidate_label": arm,
        "candidate_tree_sha256": tree,
        "candidate_tree_prefix": prefix,
        "candidate_tree_verified": True,
        "training_result_sha256": sha256_file(checkpoint.parent / "training_result.json"),
        "target_world_m": list(TARGET_WORLD),
        "language": LANGUAGE,
        "target_excluded_from_model_input": True,
        "validation_e_read": False,
        "holdout_f_read": False,
    }
    write_json(output, bundle)
    return output


def _cell_metrics(root: Path) -> dict[str, Any]:
    import numpy as np

    result = read_json(root / "condition_result.json")
    setup = read_json(root / "condition_setup.json").get("event", {})
    initial = np.asarray(setup.get("s0", {}).get("eef_world_m"), dtype=np.float64)
    positions: list[list[float]] = []
    for filename in ("rows_before_handoff.json", "rows_after_handoff.json"):
        payload = read_json(root / filename)
        for segment in payload.get("segments", []):
            positions.extend(row["achieved_pos"] for row in segment)
    achieved = np.asarray(positions, dtype=np.float64)
    require(initial.shape == (3,) and achieved.ndim == 2 and achieved.shape[1] == 3, f"cell position ABI drift: {root}")
    path = np.vstack((initial[None, :], achieved))
    target = np.asarray(TARGET_WORLD, dtype=np.float64)
    distances = np.linalg.norm(path - target[None, :], axis=1)
    diffs = np.diff(path, axis=0)
    second = np.diff(path, n=2, axis=0)
    metrics = read_json(root / "condition_metrics.json").get("metrics", {})
    gate = read_json(root / "safety_gate.json")
    decisions = list(gate.get("decisions", []))
    return {
        "status": result.get("status"),
        "cases": int(result.get("cases", 0)),
        "reference_updates": int(result.get("reference_update_count", 0)),
        "wbc_steps": int(result.get("wbc_step_count", 0)),
        "final_target_distance_m": float(metrics.get("final_target_error_m", distances[-1])),
        "absolute_progress_m": float(metrics.get("absolute_progress_m", distances[0] - distances[-1])),
        "best_progress_m": float(metrics.get("best_progress_m", distances[0] - distances.min())),
        "path_length_m": float(np.linalg.norm(diffs, axis=1).sum()),
        "roughness_m": float(np.linalg.norm(second, axis=1).mean()) if len(second) else 0.0,
        "tracking_rms_m": float(metrics.get("tracking_rms_m", float("nan"))),
        "safety_gate_decisions": decisions,
        "safety_gate_reject_count": int(sum(item == "REJECT" for item in decisions)),
        "terminal_failure": result.get("status") != "PASS" or bool(result.get("error")),
    }


def _aggregate_medium_cells(output: Path, arm: str, conditions: Sequence[str]) -> dict[str, Any]:
    import numpy as np

    cells = []
    for condition in conditions:
        root = output / arm / condition
        metrics = _cell_metrics(root)
        metrics["condition_id"] = condition
        cells.append(metrics)
    require(len(cells) == len(conditions), f"{arm} cell count drift")
    def mean(key: str) -> float:
        values = np.asarray([float(cell[key]) for cell in cells], dtype=np.float64)
        require(bool(np.isfinite(values).all()), f"{arm} {key} nonfinite")
        return float(np.mean(values))
    return {
        "arm": arm,
        "cell_count": len(cells),
        "complete_cell_count": sum(cell["status"] == "PASS" and not cell["terminal_failure"] for cell in cells),
        "scientific_reject_count": sum(cell["safety_gate_reject_count"] for cell in cells),
        "scientific_abort_count": sum(bool(cell["terminal_failure"]) for cell in cells),
        "mean_progress_m": mean("absolute_progress_m"),
        "mean_final_distance_m": mean("final_target_distance_m"),
        "mean_path_length_m": mean("path_length_m"),
        "mean_roughness_m": mean("roughness_m"),
        "mean_tracking_rms_m": mean("tracking_rms_m"),
        "per_condition": {cell["condition_id"]: cell for cell in cells},
    }


def _medium_scorecard(aggregates: Mapping[str, Mapping[str, Any]], conditions: Sequence[str], screen_result: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    per_seed: dict[str, Any] = {}
    pooled_ctrl: list[float] = []
    pooled_sup: list[float] = []
    pooled_ctrl_distance: list[float] = []
    pooled_sup_distance: list[float] = []
    for seed in SEEDS:
        ctrl = aggregates[f"MCTRL_{seed}"]
        sup = aggregates[f"MSUP_{seed}"]
        ctrl_cells = [ctrl["per_condition"][condition] for condition in conditions]
        sup_cells = [sup["per_condition"][condition] for condition in conditions]
        ctrl_progress = float(np.mean([cell["absolute_progress_m"] for cell in ctrl_cells]))
        sup_progress = float(np.mean([cell["absolute_progress_m"] for cell in sup_cells]))
        ctrl_distance = float(np.mean([cell["final_target_distance_m"] for cell in ctrl_cells]))
        sup_distance = float(np.mean([cell["final_target_distance_m"] for cell in sup_cells]))
        wins = sum(float(sup_cell["final_target_distance_m"]) < float(ctrl_cell["final_target_distance_m"]) for ctrl_cell, sup_cell in zip(ctrl_cells, sup_cells))
        pooled_ctrl.extend(cell["absolute_progress_m"] for cell in ctrl_cells)
        pooled_sup.extend(cell["absolute_progress_m"] for cell in sup_cells)
        pooled_ctrl_distance.extend(cell["final_target_distance_m"] for cell in ctrl_cells)
        pooled_sup_distance.extend(cell["final_target_distance_m"] for cell in sup_cells)
        per_seed[str(seed)] = {
            "MCTRL_mean_progress_m": ctrl_progress,
            "MSUP_mean_progress_m": sup_progress,
            "MCTRL_mean_final_distance_m": ctrl_distance,
            "MSUP_mean_final_distance_m": sup_distance,
            "final_distance_wins": wins,
            "positive": sup_progress > ctrl_progress and sup_distance < ctrl_distance,
            "MCTRL_terminal_failures": ctrl["scientific_abort_count"],
            "MSUP_terminal_failures": sup["scientific_abort_count"],
        }
    pooled_wins = sum(float(sup) < float(ctrl) for ctrl, sup in zip(pooled_ctrl_distance, pooled_sup_distance))
    screen_positive = screen_result.get("rule", {}).get("classification") == "MEDIUM_HOPPER_SCREEN_POSITIVE"
    conditions_met = {
        "at_least_2_of_3_seeds_positive": sum(item["positive"] for item in per_seed.values()) >= 2,
        "pooled_progress_better": float(np.mean(pooled_sup)) > float(np.mean(pooled_ctrl)),
        "pooled_final_distance_lower": float(np.mean(pooled_sup_distance)) < float(np.mean(pooled_ctrl_distance)),
        "pooled_final_distance_wins_at_least_29_of_48": pooled_wins >= 29,
        "msup_reject_abort_not_more_than_control": sum(aggregates[f"MSUP_{seed}"]["scientific_reject_count"] + aggregates[f"MSUP_{seed}"]["scientific_abort_count"] for seed in SEEDS) <= sum(aggregates[f"MCTRL_{seed}"]["scientific_reject_count"] + aggregates[f"MCTRL_{seed}"]["scientific_abort_count"] for seed in SEEDS),
        "no_seed_has_more_than_one_extra_terminal_failure": all(item["MSUP_terminal_failures"] <= item["MCTRL_terminal_failures"] + 1 for item in per_seed.values()),
        "tracking_rms_within_1_20x": float(np.mean([aggregates[f"MSUP_{seed}"]["mean_tracking_rms_m"] for seed in SEEDS])) <= 1.20 * float(np.mean([aggregates[f"MCTRL_{seed}"]["mean_tracking_rms_m"] for seed in SEEDS])),
        "roughness_within_1_25x": float(np.mean([aggregates[f"MSUP_{seed}"]["mean_roughness_m"] for seed in SEEDS])) <= 1.25 * float(np.mean([aggregates[f"MCTRL_{seed}"]["mean_roughness_m"] for seed in SEEDS])),
        "no_safety_gate_reject_increase": sum(aggregates[f"MSUP_{seed}"]["scientific_reject_count"] for seed in SEEDS) <= sum(aggregates[f"MCTRL_{seed}"]["scientific_reject_count"] for seed in SEEDS),
        "hopper_medium_screen_positive": screen_positive,
        "exact_16_condition_provenance_closed": all(aggregates[arm]["complete_cell_count"] == len(conditions) for arm in ARM_ORDER),
    }
    positive = all(conditions_met.values())
    return {
        "schema": "LOCALPC_PI0208_STAGE45H_TERMINAL_SCORECARD_V1",
        "classification": "LARGE_SCALE_FINE_TUNING_AUTHORIZED" if positive else "SCIENTIFIC_NOT_SUPPORTED",
        "conditions": conditions_met,
        "per_seed": per_seed,
        "pooled": {"pair_count": len(pooled_ctrl), "MCTRL_mean_progress_m": float(np.mean(pooled_ctrl)), "MSUP_mean_progress_m": float(np.mean(pooled_sup)), "MCTRL_mean_final_distance_m": float(np.mean(pooled_ctrl_distance)), "MSUP_mean_final_distance_m": float(np.mean(pooled_sup_distance)), "MSUP_final_distance_wins": pooled_wins},
        "aggregates": dict(aggregates),
        "hopper_screen": screen_result,
        "target_world_m": list(TARGET_WORLD),
        "language": LANGUAGE,
        "target_excluded_from_model_input": True,
        "validation_e_read": False,
        "holdout_f_read": False,
    }


def evaluate(args: argparse.Namespace) -> int:
    manifest = read_json(args.cohort_manifest, canonical=True)
    conditions = tuple(manifest["evaluation_condition_order"])
    deltas = manifest["evaluation_deltas_m"]
    template = read_json(args.bundle_template, canonical=True)
    require(sha256_file(args.bundle_template) == STAGE43_BUNDLE_SHA256 or template.get("schema") == "LOCALPC_PI0208_STAGE45H_EXECUTION_BUNDLE_V1", "evaluation bundle template identity drift")
    screen_result = read_json(args.screen_result, canonical=True)
    require(screen_result.get("rule", {}).get("classification") == "MEDIUM_HOPPER_SCREEN_POSITIVE", "Hopper medium screen is not positive")
    output = args.output_root.resolve()
    require(not output.exists(), f"evaluation output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)
    training_root = args.training_root.resolve()
    candidates: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER:
        checkpoint, result, record = _candidate_result(training_root, arm)
        candidates[arm] = {"checkpoint": checkpoint, "training_result": result, **record}
    overlays: dict[str, Path] = {}
    for arm in ARM_ORDER:
        overlay = output / "overlays" / f"{arm}.json"
        _candidate_bundle(template, candidates[arm]["checkpoint"], candidates[arm]["training_result"], overlay, head=args.expected_head, arm=arm)
        overlays[arm] = overlay
    write_json(output / "binding.json", {"schema": "LOCALPC_PI0208_STAGE45H_CORELAB_BINDING_V1", "status": "SEALED_BEFORE_MODEL_EXECUTION", "execution_head": args.expected_head, "cohort_manifest_sha256": sha256_file(args.cohort_manifest), "hopper_screen_sha256": sha256_file(args.screen_result), "candidate_order": list(ARM_ORDER), "condition_order": list(conditions), "candidate_records": {arm: {key: value for key, value in record.items() if key != "checkpoint"} for arm, record in candidates.items()}, "training": {"model_forward": len(ARM_ORDER) * UPDATE_COUNT, "backward": len(ARM_ORDER) * UPDATE_COUNT, "optimizer_steps": len(ARM_ORDER) * UPDATE_COUNT}, "Validation-E": 0, "Holdout-F": 0})
    gpu_uuid = _gpu_uuid()
    for arm in ARM_ORDER:
        arm_root = output / arm
        arm_root.mkdir(parents=True, exist_ok=True)
        stats_sha = sha256_file(candidates[arm]["checkpoint"] / "experiment_cfg" / "structured_action_statistics.json")
        for condition_id in conditions:
            root = arm_root / condition_id
            if root.exists():
                require(_condition_pass(root), f"existing incomplete evaluation cell: {arm}/{condition_id}")
            else:
                rc = _run_child(runner=args.runner.resolve(), bundle=overlays[arm], output=root, condition_id=condition_id, delta_map=deltas, repository_head=args.expected_head, gpu_uuid=gpu_uuid, model_python=args.model_python.resolve(), candidate_stats_sha=stats_sha, candidate_tree_sha=str(candidates[arm]["checkpoint_tree_sha256"]), log=output / "child_logs" / f"{arm}__{condition_id}.log", repo=args.repo.resolve())
                require(_condition_pass(root), f"evaluation cell failed: {arm}/{condition_id} rc={rc}")
            cell = _cell_metrics(root)
            cell_result_path = root / "cell_result.json"
            if cell_result_path.is_file():
                previous = read_json(cell_result_path)
                require(previous.get("status") == "PASS" and previous.get("arm") == arm and previous.get("condition_id") == condition_id, f"existing cell result identity drift: {arm}/{condition_id}")
            else:
                write_json(cell_result_path, {"schema": "LOCALPC_PI0208_STAGE45H_CELL_RESULT_V1", "status": "PASS", "arm": arm, "condition_id": condition_id, "condition_result_sha256": sha256_file(root / "condition_result.json"), "condition_manifest_sha256": sha256_file(root / "condition_manifest.json"), "metrics": cell, "candidate_tree_sha256": candidates[arm]["checkpoint_tree_sha256"], "gpu_uuid": gpu_uuid, "Validation-E": 0, "Holdout-F": 0})
    aggregates = {arm: _aggregate_medium_cells(output, arm, conditions) for arm in ARM_ORDER}
    card = _medium_scorecard(aggregates, conditions, screen_result)
    card["execution_head"] = args.expected_head
    card["gpu_uuid"] = gpu_uuid
    card["candidate_records"] = {arm: {key: value for key, value in record.items() if key != "checkpoint"} for arm, record in candidates.items()}
    card["counters"] = {"model_loads": len(ARM_ORDER), "model_forward_cells": len(ARM_ORDER) * len(conditions), "gr00t_forwards": len(ARM_ORDER) * len(conditions) * 4, "reference_updates": len(ARM_ORDER) * len(conditions) * 3, "native_ticks": len(ARM_ORDER) * len(conditions) * 245, "training": len(ARM_ORDER) * UPDATE_COUNT, "backward": len(ARM_ORDER) * UPDATE_COUNT, "optimizer_steps": len(ARM_ORDER) * UPDATE_COUNT, "Validation-E": 0, "Holdout-F": 0, "WBC": len(ARM_ORDER) * len(conditions) * 245, "simulator": len(ARM_ORDER) * len(conditions) * 245, "hardware": 0}
    write_json(output / "scorecard.json", card)
    seal = {"schema": "LOCALPC_PI0208_STAGE45H_RESULT_SEAL_V1", "status": "COMPLETED", "terminal_token": "LOCALPC_GOAL_MODE_STAGE45H — " + card["classification"], "scorecard_sha256": sha256_file(output / "scorecard.json"), "binding_sha256": sha256_file(output / "binding.json"), "cohort_manifest_sha256": sha256_file(args.cohort_manifest), "hopper_screen_sha256": sha256_file(args.screen_result), "gpu_uuid": gpu_uuid, "counters": card["counters"]}
    write_json(output / "result_seal.json", seal)
    print(json.dumps({"status": "PASS", "terminal_token": seal["terminal_token"], "scorecard": str(output / "scorecard.json"), "scorecard_sha256": seal["scorecard_sha256"]}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cohort", "schedule", "prepare-bundle", "collect", "train", "train-arm", "screen", "evaluate"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cohort-manifest", type=Path)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--support-root", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--bundle-template", type=Path)
    parser.add_argument("--stage43-bundle", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-label", default="S25@80")
    parser.add_argument("--model-tree-prefix", required=False, default="stage45h/candidate/checkpoint-80")
    parser.add_argument("--expected-tree")
    parser.add_argument("--structured-stats", type=Path)
    parser.add_argument("--repository-head", required=False, default="")
    parser.add_argument("--branch", default="stage45h/medium-dagger-lite-state-support")
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--model-python", type=Path)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--screen-result", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--groot-source", type=Path)
    parser.add_argument("--groot-manifest", type=Path)
    parser.add_argument("--cosmos-root", type=Path)
    parser.add_argument("--modality", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--ffmpeg-root", type=Path)
    parser.add_argument("--arm")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "cohort":
            require(args.output is not None, "--output required")
            return write_cohort(args)
        if args.mode == "schedule":
            require(args.output is not None and args.rows is not None and args.support_root is not None, "schedule paths required")
            return write_schedules(args)
        if args.mode == "prepare-bundle":
            require(args.output is not None and args.stage43_bundle is not None and args.model_path is not None and args.structured_stats is not None and args.repository_head, "bundle paths/head required")
            return prepare_bundle(args)
        if args.mode == "collect":
            for name in ("output", "cohort_manifest", "bundle", "runner", "model_python", "repository_head"):
                require(getattr(args, name), f"--{name.replace('_', '-')} required")
            return collect(args)
        if args.mode == "screen":
            for name in ("output_root", "training_root", "rows", "support_root", "expected_head", "snapshot_root", "dataset", "groot_source", "groot_manifest", "cosmos_root", "modality", "config", "stats", "ffmpeg_root"):
                require(getattr(args, name), f"--{name.replace('_', '-')} required")
            return screen(args)
        if args.mode == "evaluate":
            for name in ("output_root", "training_root", "cohort_manifest", "bundle_template", "screen_result", "expected_head", "runner", "model_python", "repo"):
                require(getattr(args, name), f"--{name.replace('_', '-')} required")
            return evaluate(args)
        if args.mode == "train-arm":
            for name in ("arm", "schedule", "rows", "support_root", "output_root", "dataset", "base_model", "groot_source", "groot_manifest", "cosmos_root", "modality", "config", "stats", "ffmpeg_root"):
                require(getattr(args, name), f"--{name.replace('_', '-')} required")
            return train_arm(args)
        require(args.schedule if hasattr(args, "schedule") else args.rows, "training schedule/rows required")
        require(args.schedule is not None, "--schedule required")
        require(args.rows is not None and args.support_root is not None and args.output_root is not None, "training paths required")
        return train_all(args)
    except Exception as exc:
        print(json.dumps({"schema": "LOCALPC_PI0208_STAGE45H_FAILURE_V1", "status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
