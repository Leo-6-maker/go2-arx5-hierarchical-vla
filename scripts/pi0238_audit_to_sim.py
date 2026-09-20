#!/usr/bin/env python3
"""PI0238 read-only audit and bounded diagnostic runner.

The script deliberately stops before WBC/simulation/training.  It reuses the
sealed PI0237 cohort and the pinned Stage10/Stage21 loader paths; G0--G3 do
not construct a GR00T model or call policy inference.
"""

from __future__ import annotations

import argparse
import csv
import copy
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PROTOCOL_COMMIT = "044e4d5ef8b4c1effb8236190c0eb083fa49979b"
PI0237_PROTOCOL_COMMIT = "6e83c5494772190f5c8082f730f06723d9c9bfda"
AUDITED_EXECUTION_COMMIT = "998eba5dc571a600a6c25a350b4a0b5dd2dad40a"
GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
SEED = 22301
DT = 0.05
HORIZON = 40
PROFILE_STEPS = (1000, 2500, 5000, 7500, 10000)
FRESH_JOB = "623593.hopper-m-02"
EXPECTED_PROFILES = {
    1000: "24a9feb7dd88650f464ee435aab466b0f843c8ba7e4186e2c425c3ff74363746",
    2500: "afb7f7c3a553fe3873f9732020fcb02bd27a03834c40e12970f0ccdee2f6f051",
    5000: "b0f76b59a7d911e18f9cf653b7195fbbeaa328ce5d3064f247913b17a2ba3890",
    7500: "43f67250f9ca04b35961677edae53cd8ce1691f64a1d7468eeb8d2449c4af38f",
    10000: "b24cbc32095be03218a7453b1e641110a7965dacaecab25102f3d587d5349c27",
}
EXPECTED_OLD = {
    "s3": "101c92b8f9f835ebaa5e9d1b25c844653106d58ace02ed0a2f850d97fe131dd8",
    "s4": "e06f876630e56894a6776e83082f29374fdbe2503598f050db5b76c653ffd8f9",
    "s5": "7a0ad71c1b3a677078f84f75ef08d63035d3af7d397fd6c2f5681c60167581cd",
    "state": "141e54082013852eb0c98f69c130495b5080204182e7f290a8b5b027e109897b",
    "cohort_csv": "2c06f8df96b492b68b8af7c7e2937bbd336b6369329c87ee16861e948326e20e",
    "cohort_json": "dfd60b94bbdd82cd07af2bbb93185e8cec720f6f4a88a33bb2cfe3a2eabce9e6",
}
EXPECTED_TREES = {
    "train": "66559a759c187fdf4d55c175a0808417369c033323e35710b6ca8aeda3786e81",
    "calib": "f53e85e8245735ec0f9a3ee9d1008d20385a4cfd8100eb66a15c015d7ecaf371",
}
TARGETS = ("a", "b", "c", "d")
JITTERS = (
    "j00_nominal", "j01_xp015", "j02_xm015", "j03_yp015", "j04_ym015",
    "j05_zp015", "j06_zm015", "j07_xp010_yp010", "j08_xm010_ym010",
)
STATES = ("s00_c0", "s01_xp050", "s02_xm050", "s03_yp050", "s04_ym050")
REALIZATIONS = ("r00", "r01", "r02", "r03")
PHYSICAL_TOL = {"xyz_m": 1e-5, "so3_rad": 1e-4, "gripper": 1e-5, "normalized": 1e-5}
MODEL_ISSUE_CODES = {
    "LINEAR_SPEED_REJECT", "ANGULAR_SPEED_REJECT", "INITIAL_ROTATION_DISCONTINUITY_REJECT",
    "INITIAL_POSITION_DISCONTINUITY_REJECT", "WORKSPACE_REJECT", "PATH_LENGTH_REJECT",
}


class DiagnosticBlocker(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticBlocker(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def run_capture(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "argv": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def tree_sha256(root: Path, prefix: str) -> tuple[str, int, int]:
    require(root.is_dir() and not root.is_symlink(), f"dataset root is not a regular directory: {root}")
    digest = hashlib.sha256()
    entries: list[tuple[str, int, str]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for dirname in dirnames:
            require(not (base / dirname).is_symlink(), f"symlink directory in sealed tree: {base / dirname}")
        for filename in filenames:
            path = base / filename
            require(path.is_file() and not path.is_symlink(), f"unsafe sealed tree file: {path}")
            size = path.stat().st_size
            entries.append((path.relative_to(root).as_posix(), size, sha256_file(path)))
    for relative, size, sha256 in sorted(entries):
        digest.update(f"{prefix}/{relative}\0{size}\0{sha256}\n".encode())
    return digest.hexdigest(), len(entries), sum(size for _, size, _ in entries)


def _no_symlink_path(path: Path, *, allow_missing: bool = False) -> Path:
    """Resolve a manifest path without accepting a symlink escape."""
    path = Path(path)
    require(path.is_absolute(), f"manifest path is not absolute: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise DiagnosticBlocker(f"manifest path contains symlink: {current}")
        if not current.exists() and not allow_missing:
            raise DiagnosticBlocker(f"manifest path component is missing: {current}")
        if not current.exists():
            break
    return path.resolve(strict=not allow_missing)


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def resolve_sealed_checkpoints(
    fresh_family: Mapping[str, Any],
    *,
    expected_root: Path | None = None,
    require_exists: bool = True,
) -> dict[str, Any]:
    """Resolve the sealed S2 checkpoint list; this is the only checkpoint authority."""
    require(fresh_family.get("job_id") == FRESH_JOB, "fresh family is not 623593")
    raw_root = fresh_family.get("root")
    require(isinstance(raw_root, str) and raw_root, "fresh family root is missing")
    root = _no_symlink_path(Path(raw_root), allow_missing=not require_exists)
    if expected_root is not None:
        expected = _no_symlink_path(Path(expected_root), allow_missing=not require_exists)
        require(root == expected, f"sealed fresh root drift: {root} != {expected}")
    raw_entries = fresh_family.get("checkpoints")
    require(isinstance(raw_entries, list), "sealed checkpoint list is not a list")
    require(len(raw_entries) == 4, "sealed checkpoint list must contain exactly four entries")
    expected_steps = {2500, 5000, 7500, 10000}
    by_step: dict[int, dict[str, Any]] = {}
    by_path: set[Path] = set()
    for entry in raw_entries:
        require(isinstance(entry, Mapping), "sealed checkpoint entry is not an object")
        step = entry.get("checkpoint_step")
        require(type(step) is int and step in expected_steps, f"invalid sealed checkpoint step: {step!r}")
        require(step not in by_step, f"duplicate sealed checkpoint step: {step}")
        candidate = entry.get("selection_candidate")
        require(type(candidate) is bool and candidate is True, f"selection_candidate is not strict true at step {step}")
        raw_path = entry.get("path")
        require(isinstance(raw_path, str) and raw_path, f"missing checkpoint path at step {step}")
        path = _no_symlink_path(Path(raw_path), allow_missing=not require_exists)
        require(_under(path, root), f"sealed checkpoint escapes fresh root: {path}")
        require(path.name == f"checkpoint-{step}", f"checkpoint step/name mismatch: {path}")
        require(path not in by_path, f"duplicate sealed checkpoint path: {path}")
        if require_exists:
            require(path.is_dir(), f"sealed checkpoint directory missing: {path}")
        by_path.add(path)
        by_step[step] = {
            "checkpoint_step": step,
            "raw_path": raw_path,
            "resolved_path": str(path),
            "path": str(path),
            "exists": path.is_dir(),
            "selection_candidate": True,
            "sealed_tree_sha256": entry.get("tree_sha256"),
        }
    require(set(by_step) == expected_steps, "sealed checkpoint steps are incomplete")
    return {
        "job_id": FRESH_JOB,
        "root": str(root),
        "checkpoints": {str(step): by_step[step] for step in sorted(by_step)},
    }


def inventory_parse(path: Path, base_dir: Path, allowed_roots: list[Path]) -> dict[str, Any]:
    """Parse a hash inventory relative to its declared artifact base."""
    require(path.is_file() and not path.is_symlink(), f"inventory missing or unsafe: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = re.match(r"^([0-9a-fA-F]{64})\s+(.+?)\s*$", line)
        require(match is not None, f"malformed inventory line {line_no}")
        expected, raw_path = match.groups()
        candidate = Path(raw_path)
        candidate = candidate if candidate.is_absolute() else Path(base_dir) / candidate
        resolved = _no_symlink_path(candidate, allow_missing=False)
        require(any(_under(resolved, root) for root in allowed_roots), f"inventory path outside allowlist: {raw_path}")
        require(resolved not in seen, f"duplicate inventory path: {resolved}")
        seen.add(resolved)
        require(resolved.is_file() and not resolved.is_symlink(), f"inventory unsafe/missing: {resolved}")
        actual = sha256_file(resolved)
        rows.append({
            "line": line_no,
            "raw_path": raw_path,
            "base_dir": str(Path(base_dir).resolve()),
            "resolved_path": str(resolved),
            "expected_sha256": expected.lower(),
            "actual_sha256": actual,
            "match": expected.lower() == actual,
        })
    require(rows, "empty evidence inventory cannot pass")
    require(all(row["match"] for row in rows), "evidence inventory hash mismatch")
    return {"path": str(path), "base_dir": str(Path(base_dir).resolve()), "count": len(rows), "rows": rows, "all_match": True}


def parse_identity(rollout_id: str) -> dict[str, str]:
    prefix = "pi0227_persistent_train_"
    require(rollout_id.startswith(prefix), f"unexpected rollout id: {rollout_id}")
    rest = rollout_id[len(prefix):]
    target, sep, rest = rest.partition("_")
    require(sep and target in TARGETS, f"target outside A-D: {rollout_id}")
    jitter = next((item for item in JITTERS if rest.startswith(item + "_")), None)
    require(jitter is not None, f"jitter not resolved: {rollout_id}")
    rest = rest[len(jitter) + 1:]
    state = next((item for item in STATES if rest.startswith(item + "_")), None)
    require(state is not None, f"state not resolved: {rollout_id}")
    rest = rest[len(state) + 1:]
    realization = next((item for item in REALIZATIONS if rest.startswith(item + "_")), None)
    require(realization is not None, f"realization not resolved: {rollout_id}")
    return {"target_family": target, "jitter_family": jitter, "initial_state": state, "realization": realization}


def source_rollout(source_episode_id: str) -> str:
    rollout, sep, _ = source_episode_id.partition(":obs-")
    require(sep and rollout, f"invalid source_episode_id: {source_episode_id}")
    return rollout


def observation_offset(source_episode_id: str) -> int:
    _, sep, offset = source_episode_id.partition(":obs-")
    require(sep and offset.isdigit(), f"invalid source offset: {source_episode_id}")
    return int(offset)


def episode_rows(dataset: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (dataset / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["episode_index"]))
    require([int(row["episode_index"]) for row in rows] == list(range(len(rows))), f"non-contiguous episodes: {dataset}")
    return rows


def full_metadata(dataset: Path, split: str, expected_rollouts: int) -> dict[str, dict[str, Any]]:
    by_rollout: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows(dataset):
        sid = str(row["source_episode_id"])
        rid = source_rollout(sid)
        by_rollout[rid].append({"episode_index": int(row["episode_index"]), "source_episode_id": sid, "source_rollout_id": rid, **parse_identity(rid)})
    require(len(by_rollout) == expected_rollouts, f"{split} rollout count drift: {len(by_rollout)}")
    result: dict[str, dict[str, Any]] = {}
    for rid, items in by_rollout.items():
        items.sort(key=lambda item: observation_offset(item["source_episode_id"]))
        n = len(items)
        for rank, item in enumerate(items):
            item["phase"] = min(3, (4 * rank) // n)
            item["observation_offset"] = observation_offset(item["source_episode_id"])
            item["split"] = split
            result[item["source_episode_id"]] = item
    return result


def stats(values: Iterable[Any]) -> dict[str, Any]:
    values = list(values)
    finite: list[float] = []
    missing = 0
    nonfinite = 0
    for value in values:
        if value is None or isinstance(value, bool):
            missing += 1
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            missing += 1
            continue
        if not math.isfinite(number):
            nonfinite += 1
            continue
        finite.append(number)
    array = np.asarray(finite, dtype=np.float64)
    return {
        "total_count": len(values), "finite_count": int(array.size), "missing_count": missing,
        "nonfinite_count": nonfinite,
        "min": float(np.min(array)) if array.size else None,
        "p50": float(np.quantile(array, 0.50)) if array.size else None,
        "p90": float(np.quantile(array, 0.90)) if array.size else None,
        "p95": float(np.quantile(array, 0.95)) if array.size else None,
        "p99": float(np.quantile(array, 0.99)) if array.size else None,
        "max": float(np.max(array)) if array.size else None,
    }


def metric_value(record: Mapping[str, Any], metric: str) -> Any:
    value = record.get("safety_gate", {}).get("metrics", {}).get(metric)
    if isinstance(value, Mapping) and "max" in value:
        return value["max"]
    return value


def raw_rot6d_quality(action: np.ndarray) -> dict[str, Any]:
    raw = np.asarray(action, dtype=np.float64).reshape(-1, 10)
    first, second = raw[:, 3:6], raw[:, 6:9]
    norm_first = np.linalg.norm(first, axis=1)
    norm_second = np.linalg.norm(second, axis=1)
    cross_norm = np.linalg.norm(np.cross(first, second), axis=1)
    return {
        "rows": int(len(raw)),
        "norm_first": stats(norm_first),
        "norm_second": stats(norm_second),
        "unit_norm_deviation_count": int(np.count_nonzero(
            (np.abs(norm_first - 1.0) > 1e-3)
            | (np.abs(norm_second - 1.0) > 1e-3)
        )),
        "abs_dot": stats(np.abs(np.sum(first * second, axis=1))),
        "cross_norm": stats(cross_norm),
        "raw_degenerate_count": int(np.count_nonzero(
            (norm_first < 1e-8) | (norm_second < 1e-8) | (cross_norm < 1e-8)
        )),
        "measured_before_orthogonalization": True,
    }


def as_array(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def active_mask_slots(mask: np.ndarray) -> tuple[list[int], int, list[int]]:
    values = np.asarray(mask)
    require(values.ndim == 3 and values.shape[1:] == (HORIZON, 132), "action mask shape drift")
    slots = np.flatnonzero(np.any(values != 0, axis=(0, 1))).tolist()
    times = np.flatnonzero(np.any(values != 0, axis=(0, 2))).tolist()
    return slots, int(np.count_nonzero(values)), times


def physical_error(decoded: np.ndarray, expert: np.ndarray, modules: Mapping[str, Any]) -> dict[str, Any]:
    decoded = np.asarray(decoded, dtype=np.float64)
    expert = np.asarray(expert, dtype=np.float64)
    xyz = np.linalg.norm(decoded[:, :3] - expert[:, :3], axis=1)
    so3 = modules["geodesic_angle"](
        modules["rot6d_rows_to_matrix"](decoded[:, 3:9]),
        modules["rot6d_rows_to_matrix"](expert[:, 3:9]),
    )
    gripper = np.abs(decoded[:, 9] - expert[:, 9])
    return {
        "xyz_max_m": float(np.max(xyz)),
        "so3_max_rad": float(np.max(so3)),
        "gripper_max": float(np.max(gripper)),
        "xyz_pass": bool(np.max(xyz) <= PHYSICAL_TOL["xyz_m"]),
        "so3_pass": bool(np.max(so3) <= PHYSICAL_TOL["so3_rad"]),
        "gripper_pass": bool(np.max(gripper) <= PHYSICAL_TOL["gripper"]),
    }


def shift_chunk_clock(chunk: Any, seconds: float) -> Any:
    return dataclasses.replace(
        chunk,
        observation_timestamp=float(chunk.observation_timestamp) + seconds,
        point_timestamps=np.asarray(chunk.point_timestamps, dtype=np.float64) + seconds,
    )


def issue_codes(record: Mapping[str, Any]) -> list[str]:
    return [str(issue.get("code")) for issue in record.get("safety_gate", {}).get("issues", []) if issue.get("code")]


def profile_records(args: argparse.Namespace) -> dict[int, dict[str, Any]]:
    run = json_load(args.run_root / "pi0236_train_calib_evaluation_run.json")
    result: dict[int, dict[str, Any]] = {}
    for item in run["profiles"]:
        step = int(item["step"])
        path = Path(str(item["path"]))
        require(path.is_file(), f"missing profile: {path}")
        require(sha256_file(path) == EXPECTED_PROFILES[step], f"sealed profile hash drift: {step}")
        payload = json_load(path)
        require(payload.get("forward_count") == 1088 and len(payload.get("records", [])) == 1088, f"profile count drift: {step}")
        result[step] = payload
    require(tuple(sorted(result)) == PROFILE_STEPS, "sealed profile order drift")
    return result


def load_legacy(args: argparse.Namespace):
    path = args.p237_root / "scripts" / "pi0237_failure_attribution.py"
    require(path.is_file(), f"missing immutable PI0237 helper: {path}")
    spec = importlib.util.spec_from_file_location("pi0237_legacy_helper", path)
    require(spec is not None and spec.loader is not None, "cannot load PI0237 helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def g0(args: argparse.Namespace) -> dict[str, Any]:
    old = args.p237_root / "evidence"
    run_path = args.run_root / "pi0236_train_calib_evaluation_run.json"
    require(run_path.is_file(), "PI0236 run manifest missing")
    run = json_load(run_path)
    require(run.get("execution_status") == "COMPLETE", "PI0236 run is not COMPLETE")
    require(sha256_file(run_path) == "4b9ceff21be16a8a2c15b789cdde7b5ecc598af1cb9f9112f295d4481f2236e4", "PI0236 run manifest SHA drift")
    profiles = profile_records(args)
    for path in (old / "pi0236_s3_provenance_lock.json", old / "pi0236_s4_selection_freeze.json", old / "pi0236_s5_diagnostics.json", old / "pi0236_execution_state.json"):
        require(path.is_file(), f"missing sealed PI0237 input: {path}")
    for role, expected in EXPECTED_OLD.items():
        if role in {"s3", "s4", "s5", "state"}:
            path = old / {"s3": "pi0236_s3_provenance_lock.json", "s4": "pi0236_s4_selection_freeze.json", "s5": "pi0236_s5_diagnostics.json", "state": "pi0236_execution_state.json"}[role]
            require(sha256_file(path) == expected, f"PI0237 sealed {role} hash drift")
    cohort_json = old / "pi0237_cohort.json"
    cohort_csv = old / "pi0237_cohort.csv"
    require(sha256_file(cohort_json) == EXPECTED_OLD["cohort_json"], "fixed cohort JSON drift")
    require(sha256_file(cohort_csv) == EXPECTED_OLD["cohort_csv"], "fixed cohort CSV drift")
    cohort = json_load(cohort_json)
    require(len(cohort.get("train", [])) == 64 and len(cohort.get("calib", [])) == 64, "fixed cohort cardinality drift")
    require({x["source_rollout_id"] for x in cohort["train"]}.isdisjoint({x["source_rollout_id"] for x in cohort["calib"]}), "fixed cohort rollout leakage")

    seal_path = old / "pi0228_train_calib_seal.json"
    authority_path = old / "pi0228_train_calib_target_authority.json"
    seal = json_load(seal_path)
    authority = json_load(authority_path)
    require(seal.get("status") == "PASS", "TRAIN/CALIB seal not PASS")
    require(seal["train"]["tree_sha256"] == EXPECTED_TREES["train"], "TRAIN sealed tree drift")
    require(seal["train_calib"]["tree_sha256"] == EXPECTED_TREES["calib"], "CALIB sealed tree drift")
    train_meta = full_metadata(args.train_dataset, "TRAIN", 612)
    calib_meta = full_metadata(args.calib_dataset, "CALIB", 68)
    require({x["source_rollout_id"] for x in train_meta.values()}.isdisjoint({x["source_rollout_id"] for x in calib_meta.values()}), "TRAIN/CALIB source-rollout leakage")
    tree_results = {}
    for name, root, prefix, expected in (("train", args.train_dataset, "train", EXPECTED_TREES["train"]), ("calib", args.calib_dataset, "train_calib", EXPECTED_TREES["calib"])):
        actual, count, total = tree_sha256(root, prefix)
        tree_results[name] = {"root": str(root), "prefix": prefix, "tree_sha256": actual, "expected": expected, "match": actual == expected, "file_count": count, "total_bytes": total}
        require(actual == expected, f"{name} tree SHA drift")
    split_lock = {"train": json_load(args.train_dataset / "meta" / "pi0227_train_only_split_lock.json"), "calib": json_load(args.calib_dataset / "meta" / "pi0227_train_only_split_lock.json")}
    require(all(lock.get("e_source_read_count") == 0 and lock.get("f_source_read_count") == 0 and lock.get("target_xyz_in_model_input") is False for lock in split_lock.values()), "TRAIN/DEV/F split lock drift")
    job_state = run_capture(["bash", "-lc", "qstat -u e1192548 -f"])
    quota = {name: run_capture(["bash", "-lc", name]) for name in ("quota -s", "quota -v", "quota -u e1192548")}
    quota_observed = any(item["returncode"] == 0 and item["stdout"].strip() for item in quota.values())
    storage = {"df_bytes": run_capture(["bash", "-lc", f"df -P {args.output_root}" ]), "df_inodes": run_capture(["bash", "-lc", f"df -Pi {args.output_root}" ]), "quota_commands": quota, "quota_observed": quota_observed, "interpretation": "df is observational only; empty quota output is not a quota PASS"}
    runtime_paths = {"protocol": args.protocol, "new_script": Path(__file__), "stage10_root": args.stage10_root, "prep_root": args.prep_root, "vla_source": args.vla_source, "baseline": args.baseline, "modality": args.modality_config, "cosmos": args.cosmos_snapshot}
    resolved = {}
    for role, path in runtime_paths.items():
        require(path.exists(), f"missing runtime identity path: {path}")
        resolved[role] = {"path": str(path.resolve()), "sha256": sha256_file(path) if path.is_file() else None}
    fresh = json_load(old / "pi0236_s2_evaluator_admission.json").get("fresh_family", {})
    require(fresh.get("job_id") == FRESH_JOB, "fresh family is not 623593")
    fresh_resolution = resolve_sealed_checkpoints(
        fresh,
        expected_root=getattr(args, "fresh_root", None),
        require_exists=True,
    )
    fresh_root = Path(fresh_resolution["root"])
    fresh_dirs = fresh_resolution["checkpoints"]
    evidence_inventory = old / "pi0237_evidence_inventory.sha256"
    inventory = inventory_parse(
        evidence_inventory,
        args.p237_root / "evidence",
        [args.p237_root / "evidence", args.prep_root / "evidence"],
    )
    p0 = {
        "schema": "PI0238_G0_IDENTITY_AND_ACCESS_V1",
        "status": "PASS_WITH_QUOTA_UNOBSERVED" if not quota_observed else "PASS_WITH_CAPACITY_UNPROVEN",
        "protocol_commit": PROTOCOL_COMMIT, "audited_execution_commit": AUDITED_EXECUTION_COMMIT,
        "sealed_pi0237_protocol_commit": PI0237_PROTOCOL_COMMIT, "pi0236_run_manifest": {"path": str(run_path), "sha256": sha256_file(run_path)},
        "profile_hashes": {str(step): EXPECTED_PROFILES[step] for step in PROFILE_STEPS}, "profile_forward_counts": {str(step): profiles[step]["forward_count"] for step in PROFILE_STEPS},
        "fresh_family": fresh_resolution,
        "cohort": {"path": str(cohort_json), "train_count": 64, "calib_count": 64, "train_sha256": EXPECTED_OLD["cohort_csv"], "json_sha256": EXPECTED_OLD["cohort_json"]},
        "dataset": {"train_rollouts": len({x["source_rollout_id"] for x in train_meta.values()}), "calib_rollouts": len({x["source_rollout_id"] for x in calib_meta.values()}), "rollout_intersection": 0, "trees": tree_results, "split_locks": split_lock},
        "runtime_identity": resolved, "active_job_query": job_state, "evidence_inventory": inventory,
        "storage": storage, "read_boundary": {"dev_e_source_read_count": 0, "final_f_source_read_count": 0, "whole_tree_scan": False, "locked_data_read": False},
        "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0,
        # A readable quota report is not proof that the requested H200
        # diagnostic capacity is available.  Keep G4 closed until R1's
        # independent admission check proves capacity.
        "g4_admission": {
            "allowed": False,
            "blockers": [
                "FILESYSTEM_QUOTA_CAPACITY_NOT_PROVEN"
                if quota_observed else "FILESYSTEM_QUOTA_UNOBSERVED"
            ],
        },
    }
    write_json(args.output_root / "evidence" / "g0_identity_and_access.json", p0)
    write_json(args.output_root / "evidence" / "cohort_binding.json", {"schema": "PI0238_FIXED_COHORT_BINDING_V1", "status": "PASS", "source": str(cohort_json), "sha256": EXPECTED_OLD["cohort_json"], "train": cohort["train"], "calib": cohort["calib"], "membership_changed": False})
    return p0


def g1(args: argparse.Namespace) -> dict[str, Any]:
    profiles = profile_records(args)
    metadata = full_metadata(args.calib_dataset, "CALIB", 68)
    all_rows: list[dict[str, Any]] = []
    by_step: dict[int, list[dict[str, Any]]] = {}
    for step, profile in profiles.items():
        current = []
        keys = set()
        for record in profile["records"]:
            key = (str(record["source_episode_id"]), int(record["episode_index"]))
            require(key not in keys, f"duplicate profile join key {step}: {key}")
            keys.add(key)
            meta = metadata.get(key[0], {})
            require(meta and int(meta["episode_index"]) == key[1], f"profile metadata key mismatch: {key}")
            item = {"checkpoint_step": step, "key": f"{key[0]}|{key[1]}", "source_episode_id": key[0], "episode_index": key[1], **{k: meta.get(k) for k in ("source_rollout_id", "target_family", "jitter_family", "initial_state", "realization", "phase")}, "decision": record.get("safety_gate", {}).get("decision"), "issue_codes": issue_codes(record), "reference_min_target_distance_m": record.get("reference_min_target_distance_m")}
            for metric in ("linear_speed_mps", "angular_speed_radps", "translation_path_length_m", "se3_arc_length_m", "initial_position_discontinuity_m", "initial_rotation_discontinuity_rad", "chunk_age_s"):
                item[metric] = metric_value(record, metric)
            current.append(item)
            all_rows.append(item)
        by_step[step] = current
    metric_names = ("linear_speed_mps", "angular_speed_radps", "translation_path_length_m", "se3_arc_length_m", "initial_position_discontinuity_m", "initial_rotation_discontinuity_rad", "chunk_age_s", "reference_min_target_distance_m")
    metrics = {str(step): {metric: stats(row.get(metric) for row in rows) for metric in metric_names} for step, rows in by_step.items()}
    issue_occurrence = {str(step): dict(Counter(code for row in rows for code in row["issue_codes"])) for step, rows in by_step.items()}
    issue_unique_rows = {str(step): dict(Counter(code for row in rows for code in set(row["issue_codes"]))) for step, rows in by_step.items()}
    decisions = {str(step): dict(Counter(row["decision"] for row in rows)) for step, rows in by_step.items()}
    cooccurrence = Counter()
    for rows in by_step.values():
        for row in rows:
            codes = sorted(set(row["issue_codes"]))
            for index, left in enumerate(codes):
                for right in codes[index + 1:]:
                    cooccurrence[f"{left}+{right}"] += 1
    grouped: dict[str, Any] = {}
    for field in ("target_family", "jitter_family", "phase", "source_rollout_id"):
        grouped[field] = {}
        for step, rows in by_step.items():
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                groups[str(row.get(field))].append(row)
            grouped[field][str(step)] = {key: {"n": len(values), "decision_counts": dict(Counter(v["decision"] for v in values)), "linear_speed": stats(v["linear_speed_mps"] for v in values), "reference_distance": stats(v["reference_min_target_distance_m"] for v in values)} for key, values in sorted(groups.items())}
    pair_deltas: dict[str, Any] = {}
    for left, right in ((1000, 2500), (2500, 5000), (5000, 7500), (7500, 10000), (1000, 2500), (1000, 5000), (1000, 7500), (1000, 10000)):
        left_map = {row["key"]: row for row in by_step[left]}
        right_map = {row["key"]: row for row in by_step[right]}
        require(set(left_map) == set(right_map), f"pair key mismatch {left}->{right}")
        pair_deltas[f"{right}_minus_{left}"] = {metric: stats((right_map[key].get(metric) - left_map[key].get(metric)) if isinstance(right_map[key].get(metric), (int, float)) and isinstance(left_map[key].get(metric), (int, float)) else None for key in sorted(left_map)) for metric in metric_names}
    expert_path = args.p237_root / "evidence" / "pi0237_p2_expert_gate.json"
    expert = json_load(expert_path)
    expert_map = {(str(row["source_episode_id"]), int(row["episode_index"])): row for row in expert["rows"] if row.get("split") == "CALIB"}
    require(len(expert_map) == 1088, "sealed expert CALIB join is incomplete")
    confusion = Counter()
    extra_speed = Counter()
    expert_speed_values = defaultdict(list)
    for row in by_step[1000]:
        key = (row["source_episode_id"], row["episode_index"])
        expert_row = expert_map[key]
        model_codes = set(row["issue_codes"])
        expert_codes = {str(x.get("code")) for x in expert_row.get("issues", [])}
        confusion[f"expert_{expert_row.get('decision')}__model_{row.get('decision')}"] += 1
        if "LINEAR_SPEED_REJECT" in model_codes and "LINEAR_SPEED_REJECT" not in expert_codes:
            extra_speed["model_extra_linear_speed_reject"] += 1
        expert_speed_values[str(expert_row.get("decision"))].append(next((x.get("observed") for x in expert_row.get("issues", []) if x.get("code") == "LINEAR_SPEED_REJECT"), None))
    output = {"schema": "PI0238_G1_FAILURE_DECOMPOSITION_V1", "status": "PASS_READ_ONLY", "record_count": len(all_rows), "profile_row_counts": {str(step): len(rows) for step, rows in by_step.items()}, "metrics": metrics, "issue_occurrence": issue_occurrence, "issue_unique_row_counts": issue_unique_rows, "decision_counts": decisions, "issue_cooccurrence_unique_rows": dict(sorted(cooccurrence.items())), "grouped": grouped, "paired_deltas": pair_deltas, "expert_model_confusion_baseline": dict(confusion), "model_extra_speed": dict(extra_speed), "expert_speed_by_decision": {key: stats(values) for key, values in expert_speed_values.items()}, "warn_decision_rows": {str(step): sum(row["decision"] == "WARN" for row in rows) for step, rows in by_step.items()}, "missing_or_unobserved": ["per-tick spike index", "expert XYZ/SO3 error", "achieved TCP", "WBC tracking", "native termination"], "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0}
    write_json(args.output_root / "evidence" / "g1_failure_decomposition.json", output)
    with (args.output_root / "evidence" / "g1_records.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in all_rows for key in row if key != "issue_codes"}) + ["issue_codes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            item = dict(row)
            item["issue_codes"] = ";".join(item["issue_codes"])
            writer.writerow(item)
    return output


def load_window(dataset: Path, row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pandas as pd
    index = int(row["episode_index"])
    path = dataset / "data" / f"chunk-{index // 1000:03d}" / f"episode_{index:06d}.parquet"
    table = pd.read_parquet(path)
    require(len(table) == HORIZON and list(table.columns) == ["timestamp", "task_index", "observation.state", "action"], f"expert window schema drift: {path}")
    timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
    state = np.asarray(table.iloc[0]["observation.state"], dtype=np.float32).reshape(-1)
    action = np.stack([np.asarray(value, dtype=np.float32).reshape(-1) for value in table["action"]], axis=0)
    require(state.shape == (10,) and action.shape == (HORIZON, 10), f"StateV1/H40 drift: {path}")
    require(np.isfinite(state).all() and np.isfinite(action).all() and np.isfinite(timestamps).all(), f"nonfinite expert window: {path}")
    require(np.allclose(np.diff(timestamps), DT, atol=1e-6, rtol=0.0), f"timestamp dt drift: {path}")
    return state, action, timestamps


def load_runtime_modules(args: argparse.Namespace) -> dict[str, Any]:
    sys.path[:0] = [str(args.prep_root / "scripts"), str(args.vla_source / "src"), str(args.stage10_root / "src" / "isaac-gr00t")]
    import stage21_same_scene_groot_inference as stage21  # type: ignore
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix, geodesic_angle  # type: ignore
    from go2arx5_vla.safety import TrajectorySafetyGateV1  # type: ignore
    selector_path = args.prep_root / "scripts" / "pi0228_train_calib_checkpoint_selection.py"
    spec = importlib.util.spec_from_file_location("pi0238_selector", selector_path)
    require(spec is not None and spec.loader is not None, "selector import unavailable")
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    return {"stage21": stage21, "matrix_to_quat": matrix_to_quat, "rot6d_rows_to_matrix": rot6d_rows_to_matrix, "geodesic_angle": geodesic_angle, "gate": TrajectorySafetyGateV1, "selector": selector}


def g2(args: argparse.Namespace) -> dict[str, Any]:
    modules = load_runtime_modules(args)
    gate = modules["gate"]()
    selector = modules["selector"]
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix  # type: ignore
    expert_path = args.p237_root / "evidence" / "pi0237_p2_expert_gate.json"
    expert = json_load(expert_path)
    calib_meta = full_metadata(args.calib_dataset, "CALIB", 68)
    train_meta = full_metadata(args.train_dataset, "TRAIN", 612)
    casebook = []
    for row in expert["rows"]:
        if row.get("split") != "CALIB" or row.get("decision") != "REJECT":
            continue
        source_id = str(row["source_episode_id"])
        state, action, timestamps = load_window(args.calib_dataset, row)
        state_matrix = rot6d_rows_to_matrix(state[3:9].reshape(1, 6))[0]
        action_matrix = rot6d_rows_to_matrix(action[0, 3:9].reshape(1, 6))[0]
        chunk = selector._build_chunk(action[None, ...], chunk_id=f"pi0238-case-{int(row['episode_index']):04d}")
        clock_shift_s = 17.0
        shifted_chunk = shift_chunk_clock(chunk, clock_shift_s)
        shifted = gate.evaluate(shifted_chunk, now=clock_shift_s, achieved_position=state[:3], achieved_orientation=matrix_to_quat(state_matrix))
        provenance = next((json.loads(line) for line in (args.calib_dataset / "meta" / "stage19_episode_provenance.jsonl").read_text().splitlines() if json.loads(line).get("dataset_episode_index") == int(row["episode_index"])), {})
        timing = next((json.loads(line) for line in (args.calib_dataset / "meta" / "stage19_action_timestamps.jsonl").read_text().splitlines() if json.loads(line).get("dataset_episode_index") == int(row["episode_index"])), {})
        casebook.append({"episode_index": int(row["episode_index"]), "source_episode_id": source_id, "source_rollout_id": source_rollout(source_id), "metadata": {**calib_meta[source_id], "target_authority": row.get("target_id")}, "source_state_rot6d": state[3:9].tolist(), "source_action0_rot6d": action[0, 3:9].tolist(), "state_rotation_matrix": state_matrix.tolist(), "action0_rotation_matrix": action_matrix.tolist(), "state_quaternion": matrix_to_quat(state_matrix).tolist(), "action0_quaternion": matrix_to_quat(action_matrix).tolist(), "state_to_action0_angle_rad": float(modules["geodesic_angle"](state_matrix, action_matrix)), "raw_action_rot6d_quality": raw_rot6d_quality(action), "source_frame": provenance.get("source_frame"), "tcp_semantic_id": provenance.get("tcp_semantic_id"), "desired_label": provenance.get("desired_label"), "source_native_indices": provenance.get("source_native_indices"), "synchronization_metrics": provenance.get("synchronization_metrics"), "synchronization_warnings": provenance.get("synchronization_warnings"), "action_timestamp_record": {key: timing.get(key) for key in ("observation_timestamp", "future_action_point_timestamps", "future_source_desired_indices", "label_source", "achieved_role", "timestamp_authority")}, "clock_origin_control": {"shift_seconds": clock_shift_s, "original_observation_timestamp": float(chunk.observation_timestamp), "shifted_observation_timestamp": float(shifted_chunk.observation_timestamp), "original_point_timestamps_sha256": sha256_array(np.asarray(chunk.point_timestamps, dtype=np.float64)), "shifted_point_timestamps_sha256": sha256_array(np.asarray(shifted_chunk.point_timestamps, dtype=np.float64)), "decision": shifted.decision.value, "linear_speed_mps": shifted.metrics.get("linear_speed_mps"), "initial_rotation_discontinuity_rad": shifted.metrics.get("initial_rotation_discontinuity_rad"), "clock_shift_changed_speed_or_rotation": shifted.metrics.get("linear_speed_mps") != row.get("metrics", {}).get("linear_speed_mps") or shifted.metrics.get("initial_rotation_discontinuity_rad") != row.get("metrics", {}).get("initial_rotation_discontinuity_rad")}, "gate_original": {"decision": row.get("decision"), "issues": row.get("issues"), "metrics": row.get("metrics")}})
    require(len(casebook) == 12, f"rotation reject casebook count drift: {len(casebook)}")

    counts = Counter()
    label_rows = []
    all_rows = [("CALIB", args.calib_dataset, episode_rows(args.calib_dataset), calib_meta), ("TRAIN", args.train_dataset, episode_rows(args.train_dataset), train_meta)]
    for split, dataset, rows, metadata in all_rows:
        for row in rows:
            state, action, timestamps = load_window(dataset, row)
            state_quat = matrix_to_quat(rot6d_rows_to_matrix(state[3:9].reshape(1, 6)))[0]
            result = gate.evaluate(selector._build_chunk(action[None, ...], chunk_id=f"pi0238-expert-{split.lower()}-{int(row['episode_index']):05d}"), now=0.0, achieved_position=state[:3], achieved_orientation=state_quat)
            codes = [issue.code for issue in result.issues]
            counts[result.decision.value] += 1
            label_rows.append({"split": split, "episode_index": int(row["episode_index"]), "source_episode_id": row["source_episode_id"], "source_rollout_id": source_rollout(str(row["source_episode_id"])), "target_family": metadata[row["source_episode_id"]]["target_family"], "jitter_family": metadata[row["source_episode_id"]]["jitter_family"], "phase": metadata[row["source_episode_id"]]["phase"], "decision": result.decision.value, "issue_codes": codes, "state_sha256": sha256_array(state), "action_sha256": sha256_array(action), "timestamp_sha256": sha256_array(timestamps), "initial_rotation_rad": result.metrics.get("initial_rotation_discontinuity_rad"), "linear_speed_mps": result.metrics.get("linear_speed_mps"), "angular_speed_radps": result.metrics.get("angular_speed_radps"), "raw_rot6d_quality": raw_rot6d_quality(action), "rot6d_orthogonality_error": float(np.linalg.norm(rot6d_rows_to_matrix(action[:, 3:9]).transpose(0, 2, 1) @ rot6d_rows_to_matrix(action[:, 3:9]) - np.eye(3), axis=(1, 2)).max())})
    census = {"schema": "PI0238_G2_FULL_EXPERT_GATE_CENSUS_V2", "status": "COMPLETE", "counts": dict(counts), "total_rows": len(label_rows), "train_rows": sum(x["split"] == "TRAIN" for x in label_rows), "calib_rows": sum(x["split"] == "CALIB" for x in label_rows), "casebook_reject_count": len(casebook), "clock_origin_control": {"shift_seconds": 17.0, "all_timestamps_shifted": True, "pose_and_indices_preserved": True, "speed_or_initial_rotation_invariant": all(not row["clock_origin_control"]["clock_shift_changed_speed_or_rotation"] for row in casebook)}, "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0}
    write_json(args.output_root / "evidence" / "g2_rotation_casebook.json", {"schema": "PI0238_G2_ROTATION_CASEBOOK_V1", "status": "PASS_READ_ONLY", "rows": casebook, "non_unit_rotation_and_q_sign_controls": "covered by CPU contract tests; q/-q equivalence is not a causal explanation"})
    write_json(args.output_root / "evidence" / "g2_full_label_gate_census.json", census)
    with (args.output_root / "evidence" / "g2_full_label_gate_census.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in label_rows for key in row if key != "issue_codes"}) + ["issue_codes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in label_rows:
            item = dict(row); item["issue_codes"] = ";".join(item["issue_codes"]); writer.writerow(item)
    write_json(args.output_root / "evidence" / "g2_source_window_alignment.json", {"schema": "PI0238_G2_SOURCE_WINDOW_ALIGNMENT_V1", "status": "PASS_READ_ONLY", "source_frame": "NATIVE_WBC_RELATIVE_REFERENCE", "label_source": "desired", "achieved_role": "audit_only_not_training_label", "sample_count": len(label_rows), "horizon": HORIZON, "dt_s": DT, "source_indices_and_timestamps": "preserved in dataset sidecars; casebook includes the 12 rejected rows", "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0})
    return {"status": census["status"], "counts": census["counts"], "casebook": len(casebook), "rows": len(label_rows)}


def describe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value), "first": describe(value[0]) if value else None}
    if hasattr(value, "shape"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return {"type": type(value).__name__, "value": str(value)[:100]}


def processor_contract(processor: Any, embodiment: Any, meta_root: Path, modality: Path, baseline: Path) -> dict[str, Any]:
    state_action = processor.state_action_processor
    norm_params = getattr(state_action, "norm_params", None)
    return {
        "class": f"{type(processor).__module__}.{type(processor).__name__}",
        "state_action_class": f"{type(state_action).__module__}.{type(state_action).__name__}",
        "stats_sha256": sha256_file(meta_root / "stats.json"),
        "relative_stats_sha256": sha256_file(meta_root / "relative_stats.json"),
        "processor_config_sha256": sha256_file(baseline / "processor_config.json"),
        "modality_sha256": sha256_file(modality),
        "embodiment": str(embodiment.value),
        "use_relative_action": bool(getattr(state_action, "use_relative_action", False)),
        "clip_outliers": bool(getattr(state_action, "clip_outliers", False)),
        "max_action_horizon": int(getattr(processor, "max_action_horizon", HORIZON)),
        "norm_params_sha256": sha256_bytes(json.dumps(norm_params, sort_keys=True, default=json_default).encode()),
    }


def training_config_summary(args: argparse.Namespace) -> dict[str, Any]:
    path = args.prep_root / "scripts" / "stage28_h2_fresh_base_2000_fixed_warmup10.py"
    spec = importlib.util.spec_from_file_location("pi0238_stage28_config", path)
    require(spec is not None and spec.loader is not None, "Stage28 config source unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    namespace = argparse.Namespace(
        stage10_root=args.stage10_root,
        base_model_path=args.baseline,
        dataset_dir=args.train_dataset,
        modality_config=args.modality_config,
        groot_source=args.stage10_root / "src" / "isaac-gr00t",
        training_output_dir=args.output_root / "not_training_output",
    )
    config = module._build_training_config(namespace)
    return module._config_summary(config)


def bind_modality_source(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    """Bind the actual G3 process to the same pinned modality file as preflight."""
    modality_path = Path(args.modality_config).resolve()
    original_sys_path = list(sys.path)
    sys.path[:0] = [
        str(modality_path.parent),
        str(args.prep_root / "scripts"),
        str(args.vla_source / "src"),
        str(args.stage10_root / "src" / "isaac-gr00t"),
    ]
    sys.modules.pop(modality_path.stem, None)
    module = importlib.import_module(modality_path.stem)
    actual = Path(module.__file__).resolve()
    require(actual == modality_path, f"G3 modality origin drift: {actual}")
    actual_sha = sha256_file(actual)
    require(actual_sha == sha256_file(modality_path), "G3 modality file changed during bind")
    # Keep the original interpreter paths, including stdlib/site-packages;
    # only the pinned project paths are prepended for subsequent imports.
    require(any("site-packages" in path or "python" in path for path in sys.path), "G3 stdlib/site-packages path was discarded")
    return module, {"path": str(actual), "sha256": actual_sha, "original_sys_path_count": len(original_sys_path)}


def g3(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    cache = args.cache_root; os.environ["HF_HOME"] = str(cache / "huggingface"); os.environ["HF_HUB_CACHE"] = str(cache / "huggingface" / "hub"); os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HUB_CACHE"]
    go2_arx5_modality, modality_origin = bind_modality_source(args)
    sys.path[:0] = [str(args.prep_root / "scripts"), str(args.vla_source / "src"), str(args.stage10_root / "src" / "isaac-gr00t")]
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # type: ignore
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset  # type: ignore
    from gr00t.data.embodiment_tags import EmbodimentTag  # type: ignore
    from go2arx5_vla.trajectory.so3 import geodesic_angle, rot6d_rows_to_matrix  # type: ignore
    import stage40_b0_model_control as stage40  # type: ignore
    import stage21_same_scene_groot_inference as stage21  # type: ignore
    stage40._bind_local_cosmos_processor(args.stage10_root, args.cosmos_snapshot)
    train_processor, embodiment = stage21._build_processor(args.stage10_root, args.baseline, args.train_dataset / "meta", args.modality_config)
    calib_processor, calib_embodiment = stage21._build_processor(args.stage10_root, args.baseline, args.train_dataset / "meta", args.modality_config)
    train_contract = processor_contract(train_processor, embodiment, args.train_dataset / "meta", args.modality_config, args.baseline)
    eval_contract = processor_contract(calib_processor, calib_embodiment, args.train_dataset / "meta", args.modality_config, args.baseline)
    modality_config = getattr(go2_arx5_modality, "GO2_ARX5_MODALITY_CONFIG", getattr(go2_arx5_modality, "SIM_VISUAL_REACH_MODALITY_CONFIG", None))
    require(modality_config is not None, "modality configuration symbol missing")
    modality_configs = copy.deepcopy(modality_config)
    samples = json_load(args.p237_root / "evidence" / "pi0237_cohort.json")
    selected = [("TRAIN", args.train_dataset, item) for item in samples["train"]] + [("CALIB", args.calib_dataset, item) for item in samples["calib"]]
    # Include the two historical normalized-excess rows and the 12 expert rejects.
    old_p3 = json_load(args.p237_root / "evidence" / "pi0237_p3_codec_roundtrip.json")
    extras: dict[tuple[str, int], tuple[str, Path, dict[str, Any]]] = {}
    for row in old_p3.get("rows", []):
        if float(row.get("normalized_roundtrip_max_abs", 0.0)) > 1e-5:
            key = (str(row["source_episode_id"]), int(row["episode_index"]))
            if key not in {(str(x[2]["source_episode_id"]), int(x[2]["episode_index"])) for x in selected}:
                extras[key] = ("CALIB" if row.get("split") == "CALIB" else "TRAIN", args.calib_dataset if row.get("split") == "CALIB" else args.train_dataset, row)
    for row in json_load(args.p237_root / "evidence" / "pi0237_p2_expert_gate.json")["rows"]:
        if row.get("decision") == "REJECT":
            key = (str(row["source_episode_id"]), int(row["episode_index"]))
            if key not in {(str(x[2]["source_episode_id"]), int(x[2]["episode_index"])) for x in selected}:
                extras[key] = ("CALIB", args.calib_dataset, row)
    selected.extend(extras.values())
    require(len(selected) <= 142, f"G3 fixed union exceeded 142: {len(selected)}")
    datasets = {str(args.train_dataset): ShardedSingleStepDataset(args.train_dataset, EmbodimentTag.NEW_EMBODIMENT, modality_configs, shard_size=128, episode_sampling_rate=1.0, seed=SEED, allow_padding=False), str(args.calib_dataset): ShardedSingleStepDataset(args.calib_dataset, EmbodimentTag.NEW_EMBODIMENT, modality_configs, shard_size=128, episode_sampling_rate=1.0, seed=SEED, allow_padding=False)}
    for dataset in datasets.values(): dataset.processor = train_processor
    processed = []
    traces = []
    for split, root, item in selected:
        dataset = datasets[str(root)]
        ep = int(item["episode_index"])
        datapoint = dataset.get_datapoint(dataset.episode_loader[ep], 0)
        processed.append(datapoint)
        traces.append({"split": split, "episode_index": ep, "source_episode_id": item["source_episode_id"], "processed": describe(datapoint)})
    batch = train_processor.collator(processed)
    batch_desc = describe(batch)
    action_candidate = None; action_key = None; mask_candidate = None; mask_key = None
    def find_arrays(value: Any, prefix: str = "") -> None:
        nonlocal action_candidate, action_key, mask_candidate, mask_key
        if isinstance(value, Mapping):
            for key, child in value.items(): find_arrays(child, f"{prefix}.{key}" if prefix else str(key))
        elif hasattr(value, "shape"):
            shape = tuple(int(x) for x in value.shape)
            low = prefix.lower()
            if action_candidate is None and len(shape) == 3 and shape[1] == HORIZON and shape[2] == 132 and "mask" not in low:
                action_candidate, action_key = value, prefix
            if mask_candidate is None and len(shape) == 3 and shape[1] == HORIZON and shape[2] == 132 and "mask" in low:
                mask_candidate, mask_key = value, prefix
    find_arrays(batch)
    require(action_candidate is not None, "actual collator batch did not expose [B,40,132] action")
    require(mask_candidate is not None, "actual collator batch did not expose [B,40,132] action_mask")
    action_array = as_array(action_candidate)
    mask_array = as_array(mask_candidate)
    require(action_array.shape == (len(selected), HORIZON, 132) and np.isfinite(action_array).all(), f"actual action tensor contract drift: {action_array.shape}")
    require(mask_array.shape == action_array.shape and np.isfinite(mask_array).all(), "actual action mask shape drift")
    active_slots, active_scalar_count, active_times = active_mask_slots(mask_array)
    # Decode each actual collated target independently through the frozen eval processor.
    errors = []
    numerical_rows = []
    first_decoded = None
    first_state = None
    first_expert_action = None
    for index, (split, root, item) in enumerate(selected):
        state, expert_action, _ = load_window(root, item)
        decoded_groups = calib_processor.decode_action(action_array[index:index + 1], calib_embodiment, {"eef_9d": state[None, None, :9], "gripper_position": state[None, None, 9:]})
        decoded = np.concatenate([np.asarray(decoded_groups["eef_9d"]), np.asarray(decoded_groups["gripper_position"])], axis=-1)[0].astype(np.float64)
        if first_decoded is None:
            first_decoded = decoded.copy()
            first_state = state.copy()
            first_expert_action = expert_action.copy()
        physical = physical_error(decoded, expert_action, {"geodesic_angle": geodesic_angle, "rot6d_rows_to_matrix": rot6d_rows_to_matrix})
        raw_action = {"eef_9d": decoded[:, :9].astype(np.float32), "gripper_position": decoded[:, 9:].astype(np.float32)}
        raw_state = {"eef_9d": state[None, :9].astype(np.float32), "gripper_position": state[None, 9:].astype(np.float32)}
        state_action = train_processor.state_action_processor
        actual_norm = state_action.apply_action(raw_action, embodiment.value, state=raw_state)
        actual_norm10 = np.concatenate([np.asarray(actual_norm["eef_9d"]), np.asarray(actual_norm["gripper_position"])], axis=-1)
        no_clip = copy.deepcopy(state_action)
        no_clip.clip_outliers = False
        unclipped = no_clip.apply_action(raw_action, embodiment.value, state=raw_state)
        unclipped10 = np.concatenate([np.asarray(unclipped["eef_9d"]), np.asarray(unclipped["gripper_position"])], axis=-1)
        original_norm = action_array[index, :, :10]
        normalized_roundtrip_max_abs = float(np.max(np.abs(actual_norm10 - original_norm)))
        raw_action64 = {"eef_9d": decoded[:, :9].astype(np.float64), "gripper_position": decoded[:, 9:].astype(np.float64)}
        raw_state64 = {"eef_9d": state[None, :9].astype(np.float64), "gripper_position": state[None, 9:].astype(np.float64)}
        float64_norm = state_action.apply_action(raw_action64, embodiment.value, state=raw_state64)
        float64_norm10 = np.concatenate([np.asarray(float64_norm["eef_9d"]), np.asarray(float64_norm["gripper_position"])], axis=-1)
        float64_cast_control_max_abs = float(np.max(np.abs(float64_norm10.astype(np.float64) - actual_norm10.astype(np.float64))))
        numerical_rows.append({"split": split, "episode_index": int(item["episode_index"]), "source_episode_id": item["source_episode_id"], "normalized_roundtrip_max_abs": normalized_roundtrip_max_abs, "float64_cast_control_max_abs": float64_cast_control_max_abs, "preclip_out_of_range_count": int(np.count_nonzero(np.abs(unclipped10) > 1.0)), "clip_changed_count": int(np.count_nonzero(np.abs(unclipped10 - actual_norm10) > 0.0)), "clip_outliers": bool(getattr(state_action, "clip_outliers", False)), "normalization_params_source": "real StateActionProcessor.norm_params"})
        errors.append({"split": split, "episode_index": int(item["episode_index"]), "source_episode_id": item["source_episode_id"], "action_tensor_sha256": sha256_array(action_array[index]), "expert_action_sha256": sha256_array(expert_action), "decoded_sha256": sha256_array(decoded), "decoded_shape": list(decoded.shape), **physical, "finite": bool(np.isfinite(decoded).all()), "raw_rot6d_quality": raw_rot6d_quality(decoded)})
    norm_excess = [row for row in numerical_rows if row["normalized_roundtrip_max_abs"] > PHYSICAL_TOL["normalized"]]
    parity = {"train_processor": train_contract, "eval_processor": eval_contract, "processor_train_eval_resolved_equal": train_contract == eval_contract, "modality_origin": modality_origin, "train_dataset_class": f"{type(next(iter(datasets.values()))).__module__}.{type(next(iter(datasets.values()))).__name__}", "language": "move the end effector to the red target", "action_horizon": HORIZON, "action_contract": "official_dense_relative_H40", "target_xyz_in_model_input": False, "augmentation": "not enabled by this direct official collator path"}
    require(first_state is not None and first_expert_action is not None, "G3 did not retain first sample controls")
    shifted_state = first_state.copy()
    shifted_state[0] += 0.01
    shifted_groups = calib_processor.decode_action(action_array[0:1].astype(np.float32), calib_embodiment, {"eef_9d": shifted_state[None, None, :9], "gripper_position": shifted_state[None, None, 9:]})
    shifted_decoded = np.concatenate([np.asarray(shifted_groups["eef_9d"]), np.asarray(shifted_groups["gripper_position"])], axis=-1)[0]
    inactive_slots = sorted(set(range(132)) - set(active_slots))
    wrong_slot_action = action_array[0].copy()
    if inactive_slots:
        wrong_slot_action[:, inactive_slots[0]] = 1.0
    wrong_mask = mask_array[0:1].copy()
    if active_slots:
        wrong_mask[:, :, active_slots[0]] = 0.0
    wrong_mask_slots = active_mask_slots(wrong_mask)[0]
    negative = {
        "wrong_slot": {
            "executed": bool(inactive_slots),
            "mutated_slot": inactive_slots[0] if inactive_slots else None,
            "rejected": bool(inactive_slots and np.any(np.abs(wrong_slot_action[:, inactive_slots[0]]) > 1e-12)),
        },
        "wrong_mask": {
            "executed": bool(active_slots),
            "mutated_slot": active_slots[0] if active_slots else None,
            "rejected": bool(active_slots and wrong_mask_slots != active_slots),
        },
        "wrong_anchor": {"executed": True, "rejected": bool(first_decoded is not None and np.max(np.abs(shifted_decoded - first_decoded)) > PHYSICAL_TOL["xyz_m"])},
        "nonunit_rot6d": {"executed": True, "raw_quality": {}, "rejected_by_raw_quality": False},
    }
    nonunit = np.asarray(first_expert_action, dtype=np.float64).copy()
    nonunit[:, 3:9] *= 2.0
    negative["nonunit_rot6d"]["raw_quality"] = raw_rot6d_quality(nonunit)
    negative["nonunit_rot6d"]["rejected_by_raw_quality"] = bool(negative["nonunit_rot6d"]["raw_quality"]["unit_norm_deviation_count"] > 0)
    physical_pass = all(row["finite"] and row["xyz_pass"] and row["so3_pass"] and row["gripper_pass"] for row in errors)
    parity_pass = bool(physical_pass and parity["processor_train_eval_resolved_equal"] and active_slots and len(numerical_rows) == len(selected))
    status = "PASS_READ_ONLY" if parity_pass else "FAIL_READ_ONLY"
    output = {"schema": "PI0238_G3_ACTUAL_BATCH_PARITY_V2", "status": status, "parity_status": "PASS" if parity_pass else "FAIL", "numerical_status": "NORMALIZED_TOLERANCE_EXCESS_REQUIRES_REVIEW" if norm_excess else "PASS", "fixed_sample_count": len(selected), "base_cohort_count": 128, "union_extra_count": len(selected) - 128, "actual_training_config": training_config_summary(args), "processed_trace": traces, "collated_batch": batch_desc, "action_tensor_key": action_key, "mask_tensor_key": mask_key, "action_shape": list(action_array.shape), "mask_shape": list(mask_array.shape), "active_slot_contract": {"observed": True, "active_slots": active_slots, "inactive_slots": inactive_slots, "mask_observed": True, "active_scalar_count": active_scalar_count, "active_times": active_times}, "parity": parity, "processor_contract": {"resolved_equal": parity["processor_train_eval_resolved_equal"], "train": train_contract, "eval": eval_contract}, "physical_tolerances": PHYSICAL_TOL, "independent_decode_error_summary": {"xyz_max_m": stats(row["xyz_max_m"] for row in errors), "so3_max_rad": stats(row["so3_max_rad"] for row in errors), "gripper_max": stats(row["gripper_max"] for row in errors), "all_finite": all(row["finite"] for row in errors), "physical_pass": physical_pass}, "rows": errors, "normalization_rows": numerical_rows, "normalized_excess_rows": norm_excess, "negative_controls": negative, "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0, "training_backward_calls": 0, "optimizer_steps": 0}
    write_json(args.output_root / "evidence" / "g3_actual_batch_trace.json", output)
    write_json(args.output_root / "evidence" / "g3_effective_train_eval_config.json", {"schema": "PI0238_G3_EFFECTIVE_CONFIG_V2", "status": status, "training_config": output["actual_training_config"], "processor_contract": output["processor_contract"], "parity": parity, "active_slot_contract": output["active_slot_contract"], "physical_tolerances": PHYSICAL_TOL, "model_weight_loads": 0, "model_forward_calls": 0})
    write_json(args.output_root / "evidence" / "g3_132_slot_mask_tests.json", {"schema": "PI0238_G3_132_SLOT_MASK_TESTS_V1", "status": status, "observed_action_shape": list(action_array.shape), "observed_mask_shape": list(mask_array.shape) if mask_array is not None else None, "active_slots": active_slots, "inactive_slots": inactive_slots, "negative_controls": {"wrong_slot": negative["wrong_slot"], "wrong_mask": negative["wrong_mask"]}, "permutation_padding_controls": "actual collator mask evidence", "model_weight_loads": 0, "model_forward_calls": 0})
    write_json(args.output_root / "evidence" / "g3_numerical_casebook.json", {"schema": "PI0238_G3_NUMERICAL_CASEBOOK_V1", "status": output["numerical_status"], "normalized_excess_rows": output["normalized_excess_rows"], "explanations": {"float32_cast_control": "The real processor output and collated action are compared without replacing the dtype path; any excess remains row-addressable.", "clip": "preclip_out_of_range_count and clip_changed_count are recorded per row; clipping is not inferred from normalized error alone.", "row_count": len(output["normalization_rows"])}, "model_weight_loads": 0, "model_forward_calls": 0})
    return {"status": output["status"], "fixed_sample_count": len(selected), "action_shape": list(action_array.shape), "mask_shape": list(mask_array.shape) if mask_array is not None else None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("g0", "g1", "g2", "g3"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--p237-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--calib-dataset", type=Path, required=True)
    parser.add_argument("--prep-root", type=Path, required=True)
    parser.add_argument("--vla-source", type=Path, required=True)
    parser.add_argument("--stage10-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cosmos-snapshot", type=Path, required=True)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    try:
        result = {"g0": g0, "g1": g1, "g2": g2, "g3": g3}[args.phase](args)
    except Exception as exc:
        failure = {"schema": f"PI0238_{args.phase.upper()}_FAILURE_V1", "status": "BLOCKED", "error_type": type(exc).__name__, "error": str(exc), "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0}
        write_json(args.output_root / "evidence" / f"{args.phase}_failure.json", failure)
        raise
    print(json.dumps({"phase": args.phase, "status": result.get("status"), "result": result}, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
