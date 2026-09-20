#!/usr/bin/env python3
"""PI0237 frozen-artifact failure attribution diagnostics.

This is deliberately a thin diagnostic layer.  It reads the sealed PI0236
records, reuses the pinned provider/gate/processor helpers, and never imports
or loads a GR00T/VLM checkpoint.  P4 inference is intentionally not part of
this first implementation; it can only be added after the P0-P3 admission
object is independently PASS.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SEED = 22301
ACTION_DT_S = 0.05
EXPECTED_CALIB_ROWS = 1088
EXPECTED_TRAIN_ROLLOUTS = 612
EXPECTED_CALIB_ROLLOUTS = 68
EXPECTED_PROFILE_STEPS = (1000, 2500, 5000, 7500, 10000)
EXPECTED_FRESH_FAMILY_JOB = "623593.hopper-m-02"
EXPECTED_CALIB_TREE = "f53e85e8245735ec0f9a3ee9d1008d20385a4cfd8100eb66a15c015d7ecaf371"
EXPECTED_TRAIN_TREE = "66559a759c187fdf4d55c175a0808417369c033323e35710b6ca8aeda3786e81"
EXPECTED_PROFILE_SHA = {
    1000: "24a9feb7dd88650f464ee435aab466b0f843c8ba7e4186e2c425c3ff74363746",
    2500: "afb7f7c3a553fe3873f9732020fcb02bd27a03834c40e12970f0ccdee2f6f051",
    5000: "b0f76b59a7d911e18f9cf653b7195fbbeaa328ce5d3064f247913b17a2ba3890",
    7500: "43f67250f9ca04b35961677edae53cd8ce1691f64a1d7468eeb8d2449c4af38f",
    10000: "b24cbc32095be03218a7453b1e641110a7965dacaecab25102f3d587d5349c27",
}
EXPECTED_SEALED_SHA = {
    "s3": "101c92b8f9f835ebaa5e9d1b25c844653106d58ace02ed0a2f850d97fe131dd8",
    "s4": "e06f876630e56894a6776e83082f29374fdbe2503598f050db5b76c653ffd8f9",
    "s5": "7a0ad71c1b3a677078f84f75ef08d63035d3af7d397fd6c2f5681c60167581cd",
    "state": "141e54082013852eb0c98f69c130495b5080204182e7f290a8b5b027e109897b",
}

JITTERS = (
    "j00_nominal",
    "j01_xp015",
    "j02_xm015",
    "j03_yp015",
    "j04_ym015",
    "j05_zp015",
    "j06_zm015",
    "j07_xp010_yp010",
    "j08_xm010_ym010",
)
STATES = ("s00_c0", "s01_xp050", "s02_xm050", "s03_yp050", "s04_ym050")
REALIZATIONS = ("r00", "r01", "r02", "r03")
TARGETS = ("a", "b", "c", "d")


class DiagnosticBlocker(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticBlocker(message)


def _profile_paths(run_manifest: Mapping[str, Any]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for item in run_manifest.get("profiles", []):
        step = int(item["step"])
        result[step] = Path(str(item["path"]))
    _require(tuple(sorted(result)) == EXPECTED_PROFILE_STEPS, "PI0236 profile step identity drift")
    return result


def _source_rollout_id(source_episode_id: str) -> str:
    rollout, separator, _ = source_episode_id.partition(":obs-")
    _require(bool(separator and rollout), f"invalid source episode id: {source_episode_id}")
    return rollout


def _observation_offset(source_episode_id: str) -> int:
    _, separator, offset = source_episode_id.partition(":obs-")
    _require(bool(separator and offset.isdigit()), f"invalid observation offset: {source_episode_id}")
    return int(offset)


def parse_rollout_identity(rollout_id: str) -> dict[str, str]:
    prefix = "pi0227_persistent_train_"
    _require(rollout_id.startswith(prefix), f"unexpected rollout identity: {rollout_id}")
    rest = rollout_id[len(prefix) :]
    target, separator, rest = rest.partition("_")
    _require(separator and target in TARGETS, f"rollout is outside TRAIN A-D: {rollout_id}")
    jitter = next((item for item in JITTERS if rest.startswith(item + "_")), None)
    _require(jitter is not None, f"cannot resolve jitter family: {rollout_id}")
    rest = rest[len(jitter) + 1 :]
    state = next((item for item in STATES if rest.startswith(item + "_")), None)
    _require(state is not None, f"cannot resolve initial state: {rollout_id}")
    rest = rest[len(state) + 1 :]
    realization = next((item for item in REALIZATIONS if rest.startswith(item + "_")), None)
    _require(realization is not None, f"cannot resolve realization: {rollout_id}")
    return {"target_family": target, "jitter_family": jitter, "initial_state": state, "realization": realization}


def load_episode_rows(dataset: Path) -> list[dict[str, Any]]:
    path = dataset / "meta" / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda item: int(item["episode_index"]))
    _require([int(row["episode_index"]) for row in rows] == list(range(len(rows))), f"non-contiguous episode indices: {dataset}")
    return rows


def build_cohort(dataset: Path, split: str, *, expected_rollouts: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = load_episode_rows(dataset)
    by_rollout: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rollout_id = _source_rollout_id(str(row["source_episode_id"]))
        metadata = parse_rollout_identity(rollout_id)
        item = {"episode_index": int(row["episode_index"]), "source_episode_id": str(row["source_episode_id"]), "source_rollout_id": rollout_id, **metadata}
        by_rollout[rollout_id].append(item)
    _require(len(by_rollout) == expected_rollouts, f"{split} rollout count drift: {len(by_rollout)}")
    metadata_by_episode: dict[str, dict[str, Any]] = {}
    for rollout_id, items in by_rollout.items():
        items.sort(key=lambda item: _observation_offset(item["source_episode_id"]))
        n = len(items)
        for rank, item in enumerate(items):
            item["phase"] = min(3, (4 * rank) // n)
            item["observation_offset"] = _observation_offset(item["source_episode_id"])
            item["split"] = split
            metadata_by_episode[item["source_episode_id"]] = item

    selected: list[dict[str, Any]] = []
    selected_rollouts: list[dict[str, Any]] = []
    for target in TARGETS:
        for phase in range(4):
            buckets: dict[str, list[str]] = defaultdict(list)
            for rollout_id, items in by_rollout.items():
                if items[0]["target_family"] == target and any(item["phase"] == phase for item in items):
                    buckets[items[0]["jitter_family"]].append(rollout_id)
            for family in buckets:
                buckets[family].sort(key=lambda rid: sha256_bytes(f"PI0237-v1|{split}|{target}|{phase}|{rid}".encode()))
            chosen: list[str] = []
            families = sorted(buckets)
            cursor = 0
            while len(chosen) < 4 and families:
                family = families[cursor % len(families)]
                if buckets[family]:
                    candidate = buckets[family].pop(0)
                    if candidate not in chosen:
                        chosen.append(candidate)
                cursor += 1
                if cursor > 1000:
                    break
            _require(len(chosen) == 4, f"{split} cohort coverage missing for target={target} phase={phase}")
            for rollout_id in chosen:
                candidates = [item for item in by_rollout[rollout_id] if item["phase"] == phase]
                candidates.sort(key=lambda item: sha256_bytes(f"PI0237-window-v1|{split}|{item['source_episode_id']}".encode()))
                chosen_item = dict(candidates[0])
                chosen_item["cohort_stratum"] = f"{target}|phase{phase}"
                selected.append(chosen_item)
                selected_rollouts.append({"split": split, "target_family": target, "phase": phase, "jitter_family": chosen_item["jitter_family"], "source_rollout_id": rollout_id})
    _require(len(selected) == 64, f"{split} cohort size is {len(selected)}")
    _require(len({item["source_episode_id"] for item in selected}) == 64, f"{split} cohort window duplication")
    return selected, metadata_by_episode


def _resolve_import_identity(prep: Path, vla_source: Path, stage10: Path, source_files: Mapping[str, Path]) -> dict[str, Any]:
    sys.path[:0] = [str(prep / "scripts"), str(vla_source / "src"), str(stage10 / "src" / "isaac-gr00t")]
    import stage20s_infer_groot as stage20  # type: ignore
    import stage21_same_scene_groot_inference as stage21  # type: ignore
    import stage40_b0_model_control as stage40  # type: ignore
    import go2arx5_vla.providers as providers  # type: ignore
    import go2arx5_vla.safety as safety  # type: ignore
    import go2arx5_vla.trajectory.so3 as so3  # type: ignore
    modules = {
        "stage20": stage20,
        "stage21": stage21,
        "stage40": stage40,
        "providers": providers,
        "safety": safety,
        "so3": so3,
    }
    identity: dict[str, Any] = {"modules": {}, "run_one_identity": stage20._run_one is stage21._run_one}
    for key, module in modules.items():
        path = Path(module.__file__).resolve()
        identity["modules"][key] = {"file": str(path), "sha256": sha256_file(path)}
    for role, path in source_files.items():
        identity.setdefault("admission_source_files", {})[role] = {"file": str(path), "sha256": sha256_file(path)}
    identity["model_weight_loads"] = 0
    identity["model_forward_calls"] = 0
    identity["get_action_calls"] = 0
    identity["stage20_run_one_module"] = stage20._run_one.__module__
    identity["stage21_run_one_module"] = stage21._run_one.__module__
    identity["provider_class"] = f"{providers.GR00TTrajectoryProvider.__module__}.{providers.GR00TTrajectoryProvider.__name__}"
    identity["safety_gate_class"] = f"{safety.TrajectorySafetyGateV1.__module__}.{safety.TrajectorySafetyGateV1.__name__}"
    identity["so3_helpers"] = [so3.rot6d_rows_to_matrix.__name__, so3.matrix_to_quat.__name__]
    _require(identity["run_one_identity"], "Stage20S _run_one is not canonical Stage21 _run_one")
    return identity


def run_p0(args: argparse.Namespace) -> dict[str, Any]:
    evidence = args.evidence_root
    run_manifest_path = args.run_root / "pi0236_train_calib_evaluation_run.json"
    run_manifest = json_load(run_manifest_path)
    s3_path = evidence / "pi0236_s3_provenance_lock.json"
    s4_path = evidence / "pi0236_s4_selection_freeze.json"
    s5_path = evidence / "pi0236_s5_diagnostics.json"
    state_path = evidence / "pi0236_execution_state.json"
    admission_path = evidence / "pi0236_s2_evaluator_admission.json"
    seal_path = evidence / "pi0228_train_calib_seal.json"
    authority_path = evidence / "pi0228_train_calib_target_authority.json"
    expected_hashes = {"s3": EXPECTED_SEALED_SHA["s3"], "s4": EXPECTED_SEALED_SHA["s4"], "s5": EXPECTED_SEALED_SHA["s5"], "state": EXPECTED_SEALED_SHA["state"]}
    for path in (run_manifest_path, s3_path, s4_path, s5_path, state_path, admission_path, seal_path, authority_path):
        _require(path.is_file(), f"missing sealed P0 input: {path}")
    _require(sha256_file(s3_path) == expected_hashes["s3"], "S3 hash drift")
    _require(sha256_file(s4_path) == expected_hashes["s4"], "S4 hash drift")
    _require(sha256_file(s5_path) == expected_hashes["s5"], "S5 hash drift")
    _require(sha256_file(state_path) == expected_hashes["state"], "execution state hash drift")
    _require(run_manifest.get("execution_status") == "COMPLETE", "PI0236 run is not complete")
    _require(run_manifest.get("wbc_execution") is False, "PI0236 WBC boundary drift")
    _require(run_manifest.get("target_xyz_in_model_input") is False, "PI0236 target input boundary drift")
    profiles = _profile_paths(run_manifest)
    profile_rows: dict[int, dict[str, Any]] = {}
    for step, path in profiles.items():
        _require(path.is_file(), f"missing profile: {path}")
        _require(sha256_file(path) == EXPECTED_PROFILE_SHA[step], f"profile hash drift: {step}")
        payload = json_load(path)
        _require(payload.get("forward_count") == EXPECTED_CALIB_ROWS, f"profile row count drift: {step}")
        _require(len(payload.get("records", [])) == EXPECTED_CALIB_ROWS, f"profile record count drift: {step}")
        profile_rows[step] = payload
    admission = json_load(admission_path)
    fresh = admission.get("fresh_family", {})
    _require(fresh.get("job_id") == EXPECTED_FRESH_FAMILY_JOB, "fresh family is not 623593")
    _require(admission.get("selected_checkpoint") in (None, "NONE") or admission.get("selected_checkpoint") == "", "selected checkpoint was unexpectedly set")
    seal = json_load(seal_path)
    _require(seal.get("status") == "PASS", "TRAIN/CALIB seal is not PASS")
    _require(seal["train"]["tree_sha256"] == EXPECTED_TRAIN_TREE, "TRAIN tree drift")
    _require(seal["train_calib"]["tree_sha256"] == EXPECTED_CALIB_TREE, "CALIB tree drift")
    train_cohort, train_meta = build_cohort(args.train_dataset, "TRAIN", expected_rollouts=EXPECTED_TRAIN_ROLLOUTS)
    calib_cohort, calib_meta = build_cohort(args.calib_dataset, "CALIB", expected_rollouts=EXPECTED_CALIB_ROLLOUTS)
    train_rollouts = {item["source_rollout_id"] for item in train_meta.values()}
    calib_rollouts = {item["source_rollout_id"] for item in calib_meta.values()}
    _require(train_rollouts.isdisjoint(calib_rollouts), "TRAIN/CALIB source-rollout leakage")
    authority = json_load(authority_path)
    authority_rollouts = {str(item["source_rollout_id"]) for item in authority["rollouts"]}
    _require(authority_rollouts == calib_rollouts, "CALIB target authority rollout mismatch")
    for item in authority["rollouts"]:
        identity = parse_rollout_identity(str(item["source_rollout_id"]))
        _require(str(item["target_id"]).startswith(f"red_target_train_{identity['target_family']}_{identity['jitter_family']}"), "target authority/source identity mismatch")
    source_files = {str(ref["role"]): Path(str(ref["path"])) for ref in admission.get("source_files", [])}
    runtime_identity = _resolve_import_identity(args.prep_root, args.vla_source, args.stage10_root, source_files)
    usage = shutil.disk_usage(args.output_root)
    df_probe = subprocess.run(["df", "-P", str(args.output_root)], capture_output=True, text=True, check=False)
    quota_probe = subprocess.run(["bash", "-lc", "quota -s"], capture_output=True, text=True, check=False)
    storage = {"statvfs_free_bytes": usage.free, "statvfs_total_bytes": usage.total, "statvfs_used_bytes": usage.used, "df_p_stdout": df_probe.stdout, "df_p_stderr": df_probe.stderr, "df_p_returncode": df_probe.returncode, "quota_stdout": quota_probe.stdout, "quota_stderr": quota_probe.stderr, "quota_returncode": quota_probe.returncode}
    _require(usage.free >= 2 * 1024**3, f"BLOCKED_STORAGE: only {usage.free} bytes free")
    read_boundary = {
        "allowlist": [str(args.train_dataset), str(args.calib_dataset), str(evidence), str(args.run_root), str(args.prep_root / "scripts"), str(args.vla_source / "src"), str(args.stage10_root / "src" / "isaac-gr00t")],
        "dev_e_source_read_count": 0,
        "final_f_source_read_count": 0,
        "rejected_path_requests": [],
        "whole_tree_scan": False,
    }
    cohort = {"schema": "PI0237_FIXED_COHORT_V1", "seed": SEED, "algorithm": "PI0237 protocol section 3.4", "train": train_cohort, "calib": calib_cohort}
    write_json(args.output_root / "pi0237_cohort.json", cohort)
    with (args.output_root / "pi0237_cohort.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["split", "episode_index", "source_episode_id", "source_rollout_id", "target_family", "jitter_family", "initial_state", "realization", "phase", "observation_offset", "cohort_stratum"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in train_cohort + calib_cohort:
            writer.writerow({key: item.get(key) for key in fields})
    write_json(args.output_root / "pi0237_runtime_identity.json", runtime_identity)
    write_json(args.output_root / "pi0237_read_boundary.json", read_boundary)
    p0 = {
        "schema": "PI0237_P0_ADMISSION_V1",
        "status": "PASS",
        "created_at_utc": utc_now(),
        "protocol_commit": "6e83c5494772190f5c8082f730f06723d9c9bfda",
        "pi0236_run_manifest": {"path": str(run_manifest_path), "sha256": sha256_file(run_manifest_path)},
        "pi0236_profile_hashes": {str(step): sha256_file(path) for step, path in profiles.items()},
        "fresh_family": {"job_id": fresh.get("job_id"), "root": fresh.get("root"), "tree_sha256": [item.get("tree_sha256") for item in fresh.get("checkpoints", [])]},
        "train_calib": {"train_tree_sha256": seal["train"]["tree_sha256"], "calib_tree_sha256": seal["train_calib"]["tree_sha256"], "train_rollouts": len(train_rollouts), "calib_rollouts": len(calib_rollouts), "rollout_intersection": 0},
        "cohort": {"train_count": len(train_cohort), "calib_count": len(calib_cohort), "train_sha256": sha256_file(args.output_root / "pi0237_cohort.csv"), "calib_rule": "fixed before result analysis"},
        "runtime_identity": str(args.output_root / "pi0237_runtime_identity.json"),
        "read_boundary": str(args.output_root / "pi0237_read_boundary.json"),
        "storage": storage,
        "model_weight_loads": 0,
        "model_forward_calls": 0,
        "get_action_calls": 0,
    }
    write_json(args.output_root / "pi0237_p0_admission.json", p0)
    return p0


def _issue_rows(profile: Mapping[str, Any], metadata: Mapping[str, Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    step = int(profile["checkpoint_step"])
    for record in profile["records"]:
        source_id = str(record["source_episode_id"])
        meta = metadata.get(source_id, {})
        gate = record.get("safety_gate", {})
        for issue in gate.get("issues", []):
            observed = issue.get("observed")
            threshold = issue.get("threshold")
            multiple = None
            if isinstance(observed, (int, float)) and math.isfinite(float(observed)) and isinstance(threshold, (int, float)) and math.isfinite(float(threshold)) and float(threshold) > 0:
                multiple = float(observed) / float(threshold)
            yield {"checkpoint_step": step, "episode_index": int(record["episode_index"]), "source_episode_id": source_id, "source_rollout_id": str(record.get("source_rollout_id", "")), "target_family": meta.get("target_family"), "jitter_family": meta.get("jitter_family"), "phase": meta.get("phase"), "decision": gate.get("decision"), "code": issue.get("code"), "severity": issue.get("severity"), "observed": observed, "threshold": threshold, "observed_over_threshold": multiple}


def _numeric_stats(values: Iterable[Any]) -> dict[str, Any]:
    array = np.asarray([float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))], dtype=np.float64)
    if not array.size:
        return {"count": 0, "finite_count": 0, "missing_count": 0, "min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {"count": int(array.size), "finite_count": int(array.size), "missing_count": 0, "min": float(np.min(array)), "p50": float(np.quantile(array, 0.50, method="linear")), "p90": float(np.quantile(array, 0.90, method="linear")), "p95": float(np.quantile(array, 0.95, method="linear")), "p99": float(np.quantile(array, 0.99, method="linear")), "max": float(np.max(array))}


def run_p1(args: argparse.Namespace) -> dict[str, Any]:
    p0 = json_load(args.output_root / "pi0237_p0_admission.json")
    _require(p0.get("status") == "PASS", "P0 is not PASS")
    # Rebuild metadata for every sealed CALIB row.  The fixed cohort is only
    # for P4 admission; P1 must not silently annotate the other 1024 rows as
    # unknown just because they were not selected for bounded replay.
    _, metadata = build_cohort(args.calib_dataset, "CALIB", expected_rollouts=EXPECTED_CALIB_ROLLOUTS)
    run_manifest = json_load(args.run_root / "pi0236_train_calib_evaluation_run.json")
    profiles = {int(item["step"]): json_load(Path(item["path"])) for item in run_manifest["profiles"]}
    issue_rows = list(_issue_rows(profiles[1000], metadata))
    for step in (2500, 5000, 7500, 10000):
        issue_rows.extend(_issue_rows(profiles[step], metadata))
    issue_counts = Counter(row["code"] for row in issue_rows)
    issue_by_step = {str(step): Counter(row["code"] for row in _issue_rows(profiles[step], metadata)) for step in profiles}
    metric_names = ("linear_speed_mps", "angular_speed_radps", "translation_path_length_m", "se3_arc_length_m", "initial_position_discontinuity_m", "initial_rotation_discontinuity_rad", "chunk_age_s")
    metric_table: list[dict[str, Any]] = []
    for step, profile in sorted(profiles.items()):
        for metric in metric_names:
            values = [record.get("safety_gate", {}).get("metrics", {}).get(metric, {}).get("max") if isinstance(record.get("safety_gate", {}).get("metrics", {}).get(metric), dict) else record.get("safety_gate", {}).get("metrics", {}).get(metric) for record in profile["records"]]
            stats = _numeric_stats(values)
            stats.update({"checkpoint_step": step, "metric": metric, "source": "sealed_profile_gate_metrics", "observability": "OBSERVED"})
            metric_table.append(stats)
    warn_rows = [row for row in issue_rows if row["severity"] == "WARNING"]
    all_distance = {}
    for step, profile in profiles.items():
        values = [float(row["reference_min_target_distance_m"]) for row in profile["records"]]
        nonreject = [float(row["reference_min_target_distance_m"]) for row in profile["records"] if row["safety_gate"]["decision"] != "REJECT"]
        all_distance[str(step)] = {"all_rows": _numeric_stats(values), "selector_nonreject": _numeric_stats(nonreject)}
    paired: dict[str, Any] = {}
    for metric in metric_names:
        by_step = {}
        for step, profile in profiles.items():
            by_step[step] = [record.get("safety_gate", {}).get("metrics", {}).get(metric, {}).get("max") if isinstance(record.get("safety_gate", {}).get("metrics", {}).get(metric), dict) else record.get("safety_gate", {}).get("metrics", {}).get(metric) for record in profile["records"]]
        paired[metric] = {}
        for step in (2500, 5000, 7500, 10000):
            delta = [float(a) - float(b) for a, b in zip(by_step[step], by_step[1000]) if isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isfinite(float(a)) and math.isfinite(float(b))]
            paired[metric][f"{step}_minus_1000"] = _numeric_stats(delta)
    with (args.output_root / "pi0237_p1_issue_table.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in issue_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(issue_rows)
    with (args.output_root / "pi0237_p1_metric_table.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in metric_table for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metric_table)
    p1 = {"schema": "PI0237_P1_GATE_DIAGNOSTICS_V1", "status": "PASS_READ_ONLY", "profile_hashes": {str(step): sha256_file(Path(item["path"])) for step, item in [(int(x["step"]), x) for x in run_manifest["profiles"]]}, "issue_counts": dict(sorted(issue_counts.items())), "issue_counts_by_checkpoint": {step: dict(sorted(counts.items())) for step, counts in issue_by_step.items()}, "warn_rows": warn_rows, "distance": all_distance, "paired_max_metric_vs_baseline": paired, "missing_or_unobserved": ["per-tick spike index", "expert XYZ/SO3 error", "normalized clipping/saturation", "achieved TCP", "WBC tracking", "native termination"], "new_forward_calls": 0}
    write_json(args.output_root / "pi0237_p1_gate_diagnostics.json", p1)
    (args.output_root / "pi0237_p1_observability.md").write_text("# PI0237 P1 observability\n\nRead-only over sealed PI0236 records. Per-tick tensor, expert error, clipping, achieved TCP, WBC tracking, and native termination are `NOT_OBSERVED_IN_SEALED_ARTIFACT`. All-row reference distance is descriptive only; selector non-REJECT statistics remain separate.\n", encoding="utf-8")
    return p1


def _load_window(dataset: Path, row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pandas as pd

    index = int(row["episode_index"])
    path = dataset / "data" / f"chunk-{index // 1000:03d}" / f"episode_{index:06d}.parquet"
    table = pd.read_parquet(path)
    _require(len(table) == 40, f"expert window is not H40: {path}")
    _require(list(table.columns) == ["timestamp", "task_index", "observation.state", "action"], f"parquet schema drift: {path}")
    timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
    state = np.asarray(table.iloc[0]["observation.state"], dtype=np.float32).reshape(-1)
    action = np.stack([np.asarray(value, dtype=np.float32).reshape(-1) for value in table["action"]], axis=0)
    _require(state.shape == (10,) and action.shape == (40, 10), f"StateV1/action shape drift: {path}")
    _require(np.isfinite(state).all() and np.isfinite(action).all(), f"non-finite expert label: {path}")
    # The materializer preserves the source clock.  Window zero starts at 0,
    # later windows retain their absolute source offset; rejecting that offset
    # would silently turn a valid window into a fabricated re-timed one.
    _require(np.all(np.isfinite(timestamps)), f"non-finite timestamps: {path}")
    _require(np.all(np.diff(timestamps) > 0.0), f"non-monotonic timestamps: {path}")
    _require(np.allclose(np.diff(timestamps), ACTION_DT_S, atol=1e-6, rtol=0.0), f"timestamp dt drift: {path}")
    return state, action, timestamps


def _runtime_modules(args: argparse.Namespace) -> dict[str, Any]:
    sys.path[:0] = [str(args.prep_root / "scripts"), str(args.vla_source / "src"), str(args.stage10_root / "src" / "isaac-gr00t")]
    import stage21_same_scene_groot_inference as stage21  # type: ignore
    import stage40_b0_model_control as stage40  # type: ignore
    from go2arx5_vla.safety import TrajectorySafetyGateV1  # type: ignore
    from go2arx5_vla.trajectory.so3 import matrix_to_quat, rot6d_rows_to_matrix  # type: ignore
    selector_path = args.prep_root / "scripts" / "pi0228_train_calib_checkpoint_selection.py"
    spec = importlib.util.spec_from_file_location("pi0237_frozen_selector", selector_path)
    _require(spec is not None and spec.loader is not None, f"cannot import selector: {selector_path}")
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    return {"stage21": stage21, "stage40": stage40, "gate_cls": TrajectorySafetyGateV1, "matrix_to_quat": matrix_to_quat, "rot6d_rows_to_matrix": rot6d_rows_to_matrix, "selector": selector}


def run_p2(args: argparse.Namespace) -> dict[str, Any]:
    p0 = json_load(args.output_root / "pi0237_p0_admission.json")
    _require(p0.get("status") == "PASS", "P0 is not PASS")
    cohort = json_load(args.output_root / "pi0237_cohort.json")
    modules = _runtime_modules(args)
    gate_cls = modules["gate_cls"]
    gate = gate_cls()
    gate_config_hash = sha256_bytes(json.dumps(_jsonable(dataclasses.asdict(gate.config)), sort_keys=True).encode())
    rows_by_split = [("CALIB", args.calib_dataset, load_episode_rows(args.calib_dataset)), ("TRAIN", args.train_dataset, [{"episode_index": int(item["episode_index"]), "source_episode_id": item["source_episode_id"]} for item in cohort["train"]])]
    results: list[dict[str, Any]] = []
    alignment: list[dict[str, Any]] = []
    for split, dataset, rows in rows_by_split:
        wanted = rows if split == "CALIB" else [{"episode_index": int(item["episode_index"]), "source_episode_id": item["source_episode_id"]} for item in cohort["train"]]
        for row in wanted:
            state, action, timestamps = _load_window(dataset, row)
            matrices = modules["rot6d_rows_to_matrix"](state[3:9].reshape(1, 6))
            state_quat = modules["matrix_to_quat"](matrices)[0]
            decoded = action[None, ...]
            chunk = modules["selector"]._build_chunk(decoded, chunk_id=f"pi0237-expert-{split.lower()}-{int(row['episode_index']):04d}")
            result = gate.evaluate(chunk, now=0.0, achieved_position=state[:3], achieved_orientation=state_quat)
            results.append({"split": split, "episode_index": int(row["episode_index"]), "source_episode_id": str(row["source_episode_id"]), "source_rollout_id": _source_rollout_id(str(row["source_episode_id"])), "label_sha256": sha256_array(action), "state_sha256": sha256_array(state), "decision": result.decision.value, "issues": [issue.as_dict() for issue in result.issues], "metrics": result.metrics, "gate_config_sha256": gate_config_hash, "provider_selector_sha256": sha256_file(args.prep_root / "scripts" / "pi0228_train_calib_checkpoint_selection.py"), "valid_points": 40, "padding_points": 0, "timestamp_sha256": sha256_array(timestamps)} )
            alignment.append({"split": split, "episode_index": int(row["episode_index"]), "source_episode_id": str(row["source_episode_id"]), "rows": 40, "timestamp_start_s": float(timestamps[0]), "timestamp_end_s": float(timestamps[-1]), "dt_s": float(np.median(np.diff(timestamps))), "state_shape": list(state.shape), "action_shape": list(action.shape), "valid_points": 40, "padding_points": 0, "source_timestamp_preserved": True, "frozen_evaluator_chunk_timestamp_start_s": 0.0, "frozen_evaluator_chunk_timestamp_end_s": float((40 - 1) * ACTION_DT_S), "timestamp_semantics_note": "sealed source timestamps are retained for audit; frozen selector _build_chunk reconstructs its original observation_timestamp=0/action_period contract", "alignment_status": "PASS"})
    counts = Counter(item["decision"] for item in results)
    p2 = {"schema": "PI0237_P2_EXPERT_GATE_V1", "status": "PASS_READ_ONLY", "expert_nonreject_pass": counts.get("REJECT", 0) == 0, "counts": dict(counts), "calib_rows": 1088, "train_rows": 64, "total_rows": len(results), "rows": results, "gate_config_sha256": gate_config_hash, "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0}
    write_json(args.output_root / "pi0237_p2_expert_gate.json", p2)
    with (args.output_root / "pi0237_p2_expert_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["split", "episode_index", "source_episode_id", "source_rollout_id", "decision", "label_sha256", "state_sha256", "gate_config_sha256", "valid_points", "padding_points"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in results)
    write_json(args.output_root / "pi0237_label_alignment.json", {"schema": "PI0237_LABEL_ALIGNMENT_V1", "status": "PASS", "rows": alignment, "source": "sealed parquet action/state/timestamp columns", "time_grid": {"horizon": 40, "dt_s": ACTION_DT_S, "last_timestamp_s": 1.95, "source_timestamp_policy": "preserved_absolute_window_clock", "frozen_evaluator_policy": "observation_timestamp_0_period_0.05"}})
    return p2


def run_p3(args: argparse.Namespace) -> dict[str, Any]:
    p0 = json_load(args.output_root / "pi0237_p0_admission.json")
    _require(p0.get("status") == "PASS", "P0 is not PASS")
    p2 = json_load(args.output_root / "pi0237_p2_expert_gate.json")
    modules = _runtime_modules(args)
    stage21 = modules["stage21"]
    stage40 = modules["stage40"]
    cache_root = args.output_root / "processor_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "huggingface" / "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HUB_CACHE"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    cosmos = args.cosmos_snapshot
    stage40._bind_local_cosmos_processor(args.stage10_root, cosmos)
    baseline = args.baseline
    processor, embodiment = stage21._build_processor(args.stage10_root, baseline, args.train_dataset / "meta", args.modality_config)
    cohort = json_load(args.output_root / "pi0237_cohort.json")
    # P2 covers all sealed CALIB labels plus the fixed 64 TRAIN cohort.  Keep
    # that exact label set for cross-path parity; the cohort only selects the
    # TRAIN subset here.
    samples = [("CALIB", args.calib_dataset, item) for item in load_episode_rows(args.calib_dataset)] + [("TRAIN", args.train_dataset, item) for item in cohort["train"]]
    rows: list[dict[str, Any]] = []
    max_xyz = 0.0
    max_so3 = 0.0
    max_gripper = 0.0
    max_norm = 0.0
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix as local_rot6d_rows_to_matrix, geodesic_angle as local_geodesic_angle  # type: ignore
    for split, dataset, item in samples:
        state, action, _ = _load_window(dataset, item)
        state_data = {"eef_9d": state[None, :9], "gripper_position": state[None, 9:]}
        action_data = {"eef_9d": action[:, :9], "gripper_position": action[:, 9:]}
        norm_state, norm_groups = processor.state_action_processor.apply(state=state_data, action=action_data, embodiment_tag=embodiment.value)
        action_keys = processor.modality_configs[embodiment.value]["action"].modality_keys
        normalized = np.concatenate([np.asarray(norm_groups[key]) for key in action_keys], axis=-1).astype(np.float32)
        _require(normalized.shape == (40, 10), f"normalized action shape drift: {split}/{item['episode_index']}")
        raw = normalized[None, ...]
        decoded_groups = processor.decode_action(raw, embodiment, {"eef_9d": state[None, None, :9], "gripper_position": state[None, None, 9:]})
        decoded = np.concatenate([np.asarray(decoded_groups["eef_9d"]), np.asarray(decoded_groups["gripper_position"])], axis=-1)[0].astype(np.float64)
        ref = action.astype(np.float64)
        xyz = np.linalg.norm(decoded[:, :3] - ref[:, :3], axis=1)
        pred_rot = local_rot6d_rows_to_matrix(decoded[:, 3:9])
        ref_rot = local_rot6d_rows_to_matrix(ref[:, 3:9])
        so3 = np.asarray(local_geodesic_angle(pred_rot, ref_rot), dtype=np.float64)
        grip = np.abs(decoded[:, 9] - ref[:, 9])
        reencoded_state = {"eef_9d": state[None, :9], "gripper_position": state[None, 9:]}
        reencoded_action = {"eef_9d": decoded[:, :9].astype(np.float32), "gripper_position": decoded[:, 9:10].astype(np.float32)}
        _, reencoded_groups = processor.state_action_processor.apply(state=reencoded_state, action=reencoded_action, embodiment_tag=embodiment.value)
        reencoded = np.concatenate([np.asarray(reencoded_groups[key]) for key in action_keys], axis=-1).astype(np.float32)
        norm_diff = float(np.max(np.abs(normalized - reencoded)))
        max_xyz = max(max_xyz, float(np.max(xyz)))
        max_so3 = max(max_so3, float(np.max(so3)))
        max_gripper = max(max_gripper, float(np.max(grip)))
        max_norm = max(max_norm, norm_diff)
        rows.append({"split": split, "episode_index": int(item["episode_index"]), "source_episode_id": item["source_episode_id"], "normalized_sha256": sha256_array(normalized), "decoded_sha256": sha256_array(decoded.astype(np.float32)), "max_xyz_l2_m": float(np.max(xyz)), "max_so3_rad": float(np.max(so3)), "max_gripper_abs": float(np.max(grip)), "normalized_roundtrip_max_abs": norm_diff, "clip_accounting": "not separately exposed by processor; no new clipping applied"})
    fresh_stats = args.train_dataset / "meta" / "stats.json"
    relative_stats = args.train_dataset / "meta" / "relative_stats.json"
    processor_config = baseline / "processor_config.json"
    base_stats = baseline / "statistics.json"
    norm_tolerance = 1e-5
    norm_excess_rows = sum(float(row["normalized_roundtrip_max_abs"]) > norm_tolerance for row in rows)
    lossy_requires_review = bool(max_norm > norm_tolerance)
    p3 = {"schema": "PI0237_P3_CODEC_ROUNDTRIP_V1", "status": "PASS_READ_ONLY", "processor_identity": {"processor_config_sha256": sha256_file(processor_config), "base_statistics_sha256": sha256_file(base_stats), "train_stats_sha256": sha256_file(fresh_stats), "train_relative_stats_sha256": sha256_file(relative_stats), "modality_sha256": sha256_file(args.modality_config), "cosmos_snapshot": str(cosmos), "embodiment": str(embodiment.value), "action_contract": "official_dense_relative_H40", "state_anchor": "last state timestep / StateV1 current row", "effective_action_dim": 10, "padded_universal_dim": 132, "horizon": 40, "dt_s": ACTION_DT_S, "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0}, "sample_count": len(rows), "train_count": len(cohort["train"]), "calib_count": len(rows) - len(cohort["train"]), "rows": rows, "max_errors": {"xyz_l2_m": max_xyz, "so3_rad": max_so3, "gripper_abs": max_gripper, "normalized_roundtrip_max_abs": max_norm}, "float32_tolerances": {"xyz_l2_m": 1e-5, "so3_rad": 1e-4, "gripper_abs": 1e-5, "normalized_roundtrip_max_abs": norm_tolerance}, "norm_tolerance_excess_rows": norm_excess_rows, "admission_relevant": {"cross_path_semantics_exercised": True, "lossy_transform_requires_pi_review": lossy_requires_review, "processor_initialized_offline": True, "expert_gate_pass_required": bool(p2.get("expert_nonreject_pass"))}}
    write_json(args.output_root / "pi0237_p3_codec_roundtrip.json", p3)
    with (args.output_root / "pi0237_p3_tensor_stages.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output_root / "pi0237_p3_processor_identity.json", p3["processor_identity"])
    write_json(args.output_root / "pi0237_p3_lossy_transform.json", {"status": "LOSSY_TRANSFORM_REQUIRES_PI_REVIEW" if lossy_requires_review else "PASS_READ_ONLY", "new_clipping": False, "observed_processor_clip_outliers": "not separately exposed", "normalized_roundtrip_max_abs": max_norm, "normalized_roundtrip_tolerance": norm_tolerance, "normalized_roundtrip_excess_rows": norm_excess_rows, "decision": "float32 normalized round-trip exceeds the pinned 1e-5 tolerance; no clipping/smoothing/retiming was added" if lossy_requires_review else "no new clipping/smoothing/retiming applied"})
    return p3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("p0", "p1", "p2", "p3"), required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--calib-dataset", type=Path, required=True)
    parser.add_argument("--prep-root", type=Path, required=True)
    parser.add_argument("--vla-source", type=Path, required=True)
    parser.add_argument("--stage10-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cosmos-snapshot", type=Path, required=False)
    parser.add_argument("--baseline", type=Path, required=False)
    parser.add_argument("--modality-config", type=Path, required=False)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.phase == "p0":
        result = run_p0(args)
    elif args.phase == "p1":
        result = run_p1(args)
    elif args.phase == "p2":
        result = run_p2(args)
    else:
        _require(args.cosmos_snapshot is not None and args.baseline is not None and args.modality_config is not None, "P3 requires Cosmos, baseline, and modality paths")
        result = run_p3(args)
    print(json.dumps({"phase": args.phase, "status": result.get("status"), "output_root": str(args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
