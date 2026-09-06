#!/usr/bin/env python3
"""Stage39 sealed StructuredActionV1 to native-WBC replay qualification.

The ``--admission`` mode is CPU/read-only with respect to the WBC.  It audits
the sealed Stage38-E learned physical trajectories, independently recomputes
their SafetyGate decisions, and materializes the two frozen controls plus the
sixteen learned rows into a fresh scratch-backed input directory.

The ``--campaign`` mode is called inside one PBS allocation.  It invokes the
already accepted native WBC benchmark exactly once for each of the 18 frozen
rows.  It never constructs GR00T, calls a GR00T sampler, trains, writes a
checkpoint, or submits PBS.  A row that has started is never silently retried.

The ``--seal`` mode is a host-side independent audit of the completed
allocation and seals all new evidence without touching historical trees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from go2arx5_vla.contracts import TrajectoryChunkV1  # noqa: E402
from go2arx5_vla.safety.trajectory_gate import TrajectorySafetyGateV1  # noqa: E402
from go2arx5_vla.structured_action import (  # noqa: E402
    decode_structured_action,
    encode_structured_action,
    materialize_structured_action,
)
from go2arx5_vla.trajectory.so3 import (  # noqa: E402
    geodesic_angle,
    matrix_to_quat,
    quat_to_matrix,
    rot6d_rows_to_matrix,
    rotation_matrix_to_rot6d_rows,
)


PI_GATE_ID = "PI_GATE_STATE_0028"
ISSUE_COMMENT_ID = "5556900735"
PR_HANDOFF_COMMENT_ID = "5556901524"
PR_NUMBER = 25
STAGE39_BRANCH = "stage39/structured-wbc-qualification"
STAGE38_BRANCH = "stage38/structured-groot-offline"
EXPECTED_STAGE38_HEAD = "94cbf4a28ebc8309d3f4c3583c816eb4e1d26071"
EXPECTED_PROJECT = "CFP03-CF-004"
EXPECTED_QUEUE = "auto"
EXPECTED_REFERENCE_MODE = "BENCHMARK_40POINT_RECONSTRUCTED"
EXPECTED_DURATION_S = 1.9500000000000002
EXPECTED_POINT_COUNT = 40
EXPECTED_SEEDS = tuple(range(2223, 2239))
EXPECTED_ROW_IDS = (
    ("C0_dense_expert_tiny1", "DENSE_EXPERT_TINY1", None),
    ("C1_structured_oracle_k16t16", "STRUCTURED_ORACLE_K16T16", None),
) + tuple(
    ("L%02d_seed%d" % (index, seed), "STRUCTURED_LEARNED", seed)
    for index, seed in enumerate(EXPECTED_SEEDS, start=1)
)
NEXT_GATE = "GPT_PI_REVIEW_AFTER_STAGE39_STRUCTURED_WBC_REPLAY"
EXPECTED_STAGE38_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage38_structured_groot_offline_20260906/stage38e_offline_94cbf4a"
)
EXPECTED_STAGE38_FILES = {
    "result": (
        EXPECTED_STAGE38_ROOT / "stage38e_result.json",
        "45d673305d1ade3a3036dc0bab3f4ca544ff2591d245a66002a76e224da0455f",
    ),
    "result_manifest": (
        EXPECTED_STAGE38_ROOT / "stage38e_result_manifest.json",
        "ee20d84f12faa0c8eeca2dd1fe7c3035b9e014d10c2599c111cb6f2ca40bc530",
    ),
    "raw_hash_manifest": (
        EXPECTED_STAGE38_ROOT / "stage38e_raw_hash_manifest.json",
        "f068eebea062d03eb082a26113015461f5efd3672658722461cd404618fd2136",
    ),
    "artifact_seal": (
        EXPECTED_STAGE38_ROOT / "stage38e_artifact_seal.json",
        "7eaed6f8f576860ab267ee7b35d3e522b1ff8cdb9b20aac4d1f8b1ddb5bdca79",
    ),
    "final_return": (
        EXPECTED_STAGE38_ROOT / "stage38e_final_return.json",
        "14cd796013bf09c1445bd8cd4f96585c8a63cfd367d74c0d6afe1f23742f70b4",
    ),
}
EXPECTED_STAGE38D = {
    "result_sha256": "b75bd03967ca1797c04eb3dae5ceeccd52c9a6ec5d28ec02c8bc507b786e4aff",
    "statistics_sha256": "0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206",
    "checkpoint_1000_tree_sha256": "30b14d36e47e08dfbb9d3b353d199971863d6e7f6ceedea2d478ef367d0a1137",
    "checkpoint_2000_tree_sha256": "4ce18fac97646e1c7a4c6000e2d9758a92ebe6cd4e1cd32fd94e6b6aa6d705bd",
}
EXPECTED_SNAPSHOT = Path(
    "/scratch/e1192548/stage26_h1_c1_20260904/snapshots/"
    "stage22_expert_red_target_train_a/expert-0000.json"
)
EXPECTED_SNAPSHOT_SHA256 = "74af1af85e21c6e3a5e91e5b4b695d080f0f6b7f2204d012310d6457334106f1"
EXPECTED_EXPERT_PHYSICAL_SHA256 = (
    "68c8d750122d049612e38f128019f6eddba7bd43e59c5e562482a8b41a908194"
)
EXPECTED_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
EXPECTED_TCP = "x5_link6_gripper_center"
EXPECTED_STAGE38_LEARNED_COUNT = 16
EXPECTED_STAGE38_ARTIFACT_COUNT = 32

EXPECTED_WBC_ARCHIVE = Path(
    "/scratch/e1192548/leon-storage/wbc-redelivery/"
    "WBCTrackingShare_20260831.zip"
)
EXPECTED_WBC_ARCHIVE_SHA256 = (
    "b52a53bef87340e6244ead352fa434390faee65246cd32ab8e6648336a018167"
)
EXPECTED_WBC_SOURCE_ROOT = Path(
    "/scratch/e1192548/leon-storage/wbc-authority/"
    "stage36_authority_v2_20260906_clean_v1/WBCTrackingShare"
)
EXPECTED_WBC_MANIFEST_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage36_wbc_authority_oracle_qualification_20260906/"
    "stage36_cpu_20260906_resumea1a07a3_v11"
)
EXPECTED_WBC_MANIFESTS = {
    "authority_manifest": (
        EXPECTED_WBC_MANIFEST_ROOT / "wbc_authority_v2_manifest.json",
        "baa94da38158788c9a704a5691adca412de7dd4e9590ea357fcc4523d8c38b34",
    ),
    "runtime_contract": (
        EXPECTED_WBC_MANIFEST_ROOT / "wbc_authority_v2_runtime_contract.json",
        "1e56d9e1e31d334e12091234feb53a49ffadf4c3e91e35f8f6bf2abf4e1e6a3d",
    ),
    "tree_manifest": (
        EXPECTED_WBC_MANIFEST_ROOT / "wbc_authority_v2_tree_manifest.json",
        "fb8470c24f40724f24b8cdf76157aba33af41dffc4eb9717c99e65a6abb15440",
    ),
}
EXPECTED_WBC_TREE_SHA256 = "63311df752c9aaa5289287849635309e558f79284c91b80ddf07ec5638992567"
AUTHORITY_RUN_REL = Path(
    "runs/2026-08-05/"
    "stage2_v3_trajtracking_rl_e2e_gait_wo_postproc_192018"
)
AUTHORITY_COMPONENT_HASHES = {
    "arm_checkpoint": "99a3a9bda30f7372c7c31240cc3584757ba46cb3b6b9881f00b371906d077a0a",
    "dog_checkpoint": "51776ddfd7177213444c0bf0c8f017b66ffd28a8674458aa0a16c3b9f77cc670",
    "native_config": "56132d4ff9b866ce196bbc1b1353900b967c54f0f48d2d4e4b82c3555769a0bb",
    "native_urdf": "96c1a81e3dd42af5f1bf87ac8958a0aebd7e798ce06408462e0e49a8a2067e25",
    "native_reach_table": "e34ca5c595b30b737209f5dcdd7f3f1e0b95cdf6232fbc0f6f50e3ee7140f5fa",
    "trajectory_module": "8386841b29b58f324cd2e9afa086439add4183992be2466f688382be8ed9d76a",
    "wbc_env": "e39e1bb1dfe4c7a5c2b7309f4f158b6dcf1c9aac91ccbf8a60988ed8a64e992e",
    "legged_robot": "4dc9107af7c1f77dc3b1b686911addf1a1d1a5179dfb194d0b1f4d0bdb8d8a60",
    "wbc_wrapper": "9fa688ce7c3f4665ec48752f0f31639f05ab3309b53d00d95e3d0afd0cb300c7",
}
NATIVE_RUNNER = REPO / "scripts/native_wbc_benchmark.py"
NATIVE_RUNNER_SHA256 = "41059b8a24de94497cb24a589ac73bfb7b7ffb233f24356411279880e70cc3f8"
PBS_WRAPPER = REPO / "hpc/pbs/stage39_structured_wbc_replay.pbs"
STAGE36_HELPER = REPO / "scripts/stage36_wbc_oracle_campaign.py"
POST_RUN_COMPLETION_RETURN_CODE = -11
FINALIZED_NATIVE_TERMINATIONS = {
    "REFERENCE_COMPLETE",
    "FALL",
    "INVALID_STATE",
    "JOINT_LIMIT",
    "TRACKING_ABORT",
    "NATIVE_TASK_TERMINATION",
    "UNKNOWN",
}


class Stage39Error(RuntimeError):
    """A fail-closed Stage39 admission/campaign error."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_sha256(value: Any, dtype: Optional[Any] = None) -> str:
    array = np.asarray(value if dtype is None else np.asarray(value, dtype=dtype))
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def finite_json(value: Any, path: str = "root") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Stage39Error("non-finite JSON value at %s" % path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_json(item, "%s[%d]" % (path, index))
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_json(item, "%s.%s" % (path, key))


def write_json(path: Path, payload: Any) -> None:
    serializable = jsonable(payload)
    finite_json(serializable, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    finite_json(payload, str(path))
    return payload


def log_event(root: Path, event: str, **fields: Any) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp_utc": now_utc(), "event": event}
    payload.update(fields)
    with (root / "stage39_campaign.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(payload), sort_keys=True, allow_nan=False) + "\n")


def file_record(path: Path, expected_sha256: Optional[str] = None) -> Dict[str, Any]:
    path = Path(path)
    exists = path.is_file() and not path.is_symlink()
    actual = sha256_file(path) if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "is_symlink": path.is_symlink(),
        "size_bytes": int(path.stat().st_size) if exists else None,
        "sha256": actual,
        "expected_sha256": expected_sha256,
        "hash_match": bool(exists and (expected_sha256 is None or actual == expected_sha256)),
    }


def regular_entries(root: Path) -> List[Dict[str, Any]]:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise Stage39Error("tree root is not a regular directory: %s" % root)
    entries: List[Dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(str(root), followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            if child.is_symlink():
                raise Stage39Error("tree contains symlink directory: %s" % child)
        for filename in sorted(filenames):
            child = directory_path / filename
            if child.is_symlink() or not child.is_file():
                raise Stage39Error("tree contains unsafe file: %s" % child)
            entries.append(
                {
                    "path": child.relative_to(root).as_posix(),
                    "bytes": int(child.stat().st_size),
                    "sha256": sha256_file(child),
                }
            )
    return sorted(entries, key=lambda item: str(item["path"]))


def tree_sha256(root: Path, prefix: str) -> Tuple[str, List[Dict[str, Any]]]:
    entries = regular_entries(root)
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            "%s/%s\0%s\0%s\n"
            % (prefix, entry["path"], entry["bytes"], entry["sha256"])
        .encode("utf-8")
        )
    return digest.hexdigest(), entries


def compare_float(a: Any, b: Any, atol: float = 1e-6, rtol: float = 1e-7) -> bool:
    try:
        return bool(math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol))
    except (TypeError, ValueError):
        return a == b


def compare_issue_lists(expected: Sequence[Mapping[str, Any]], actual: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    differences: List[Dict[str, Any]] = []
    if len(expected) != len(actual):
        differences.append({"kind": "count", "expected": len(expected), "actual": len(actual)})
    for index in range(max(len(expected), len(actual))):
        if index >= len(expected) or index >= len(actual):
            continue
        left, right = expected[index], actual[index]
        for key in ("code", "severity", "message", "provenance"):
            if left.get(key) != right.get(key):
                differences.append({"index": index, "field": key, "expected": left.get(key), "actual": right.get(key)})
        for key in ("observed", "threshold"):
            if left.get(key) is None and right.get(key) is None:
                continue
            if isinstance(left.get(key), (int, float)) and isinstance(right.get(key), (int, float)):
                equal = compare_float(left.get(key), right.get(key))
            else:
                equal = left.get(key) == right.get(key)
            if not equal:
                differences.append({"index": index, "field": key, "expected": left.get(key), "actual": right.get(key)})
    return {"match": not differences, "differences": differences}


def compare_mapping_metrics(expected: Mapping[str, Any], actual: Mapping[str, Any], path: str = "metrics") -> List[Dict[str, Any]]:
    differences: List[Dict[str, Any]] = []
    keys = sorted(set(expected) | set(actual))
    for key in keys:
        left, right = expected.get(key), actual.get(key)
        current = "%s.%s" % (path, key)
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            differences.extend(compare_mapping_metrics(left, right, current))
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if not compare_float(left, right):
                differences.append({"path": current, "expected": left, "actual": right})
        elif left != right:
            differences.append({"path": current, "expected": left, "actual": right})
    return differences


def verify_sealed_entries(root: Path, seal_path: Path, expected_sha256: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "seal": file_record(seal_path, expected_sha256),
        "entry_count": 0,
        "verified_count": 0,
        "mismatches": [],
        "status": "BLOCKED",
    }
    if not result["seal"]["hash_match"]:
        return result
    seal = load_json(seal_path)
    entries = seal.get("artifacts", seal.get("entries", []))
    result["entry_count"] = len(entries)
    for entry in entries:
        candidate = Path(str(entry.get("path", "")))
        if not candidate.is_absolute():
            candidate = root / str(entry.get("relative_path", ""))
        actual = sha256_file(candidate) if candidate.is_file() and not candidate.is_symlink() else None
        actual_size = int(candidate.stat().st_size) if candidate.is_file() and not candidate.is_symlink() else None
        expected = entry.get("sha256", entry.get("sha256_hex"))
        expected_size = entry.get("size_bytes", entry.get("bytes"))
        if actual != expected or (expected_size is not None and actual_size != int(expected_size)):
            result["mismatches"].append(
                {
                    "path": str(candidate),
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "expected_size_bytes": expected_size,
                    "actual_size_bytes": actual_size,
                }
            )
        else:
            result["verified_count"] += 1
    result["status"] = "PASS" if not result["mismatches"] else "BLOCKED_SEAL_ENTRY_MISMATCH"
    return result


def verify_stage38_chain() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": "BLOCKED",
        "files": {},
        "seal": {},
        "learned_rows": [],
        "checkpoint": {},
    }
    for name, (path, expected) in EXPECTED_STAGE38_FILES.items():
        result["files"][name] = file_record(path, expected)
        if not result["files"][name]["hash_match"]:
            raise Stage39Error("Stage38 sealed file mismatch: %s" % name)
    result["seal"] = verify_sealed_entries(
        EXPECTED_STAGE38_ROOT,
        EXPECTED_STAGE38_FILES["artifact_seal"][0],
        EXPECTED_STAGE38_FILES["artifact_seal"][1],
    )
    if result["seal"]["status"] != "PASS":
        raise Stage39Error("Stage38 artifact seal did not independently verify")
    payload = load_json(EXPECTED_STAGE38_FILES["result"][0])
    if payload.get("schema") != "STAGE38E_STRUCTURED_GR00T_OFFLINE_RESULT_V1":
        raise Stage39Error("Stage38 result schema mismatch")
    if payload.get("status") != "PASS_EXECUTION_COMPLETE":
        raise Stage39Error("Stage38 result is not complete")
    if payload.get("scientific_contract_changed") is not False:
        raise Stage39Error("Stage38 scientific contract changed")
    if payload.get("next_gate") != "GPT_PI_REVIEW_AFTER_STAGE38_STRUCTURED_GR00T_OFFLINE":
        raise Stage39Error("Stage38 next gate mismatch")
    if payload.get("stage38d", {}).get("result_sha256") != EXPECTED_STAGE38D["result_sha256"]:
        raise Stage39Error("Stage38-D result identity mismatch")
    for key in ("statistics_sha256", "checkpoint_1000_tree_sha256", "checkpoint_2000_tree_sha256"):
        if payload.get("stage38d", {}).get(key) != EXPECTED_STAGE38D[key]:
            raise Stage39Error("Stage38-D %s mismatch" % key)

    records = payload.get("free_running_records", [])
    if len(records) != EXPECTED_STAGE38_ARTIFACT_COUNT:
        raise Stage39Error("Stage38-E free-running record count is not 32")
    learned = [
        row
        for row in records
        if row.get("checkpoint") == "STRUCTURED@1000" and row.get("kind") == "free_running_structured"
    ]
    if len(learned) != EXPECTED_STAGE38_LEARNED_COUNT:
        raise Stage39Error("Stage38-E learned @1000 rows are not exactly 16")
    learned.sort(key=lambda row: int(row.get("seed", -1)))
    if tuple(int(row.get("seed", -1)) for row in learned) != EXPECTED_SEEDS:
        raise Stage39Error("Stage38-E learned seed set mismatch")
    if any(row.get("decode_valid") is not True or row.get("finite") is not True for row in learned):
        raise Stage39Error("Stage38-E learned row finite/decode validity mismatch")

    snapshot = load_json(EXPECTED_SNAPSHOT)
    timestamps = np.asarray(snapshot.get("future_timestamps"), dtype=np.float64)
    if timestamps.shape != (EXPECTED_POINT_COUNT,) or not np.isfinite(timestamps).all():
        raise Stage39Error("Stage22 snapshot timestamps are not [40] finite")
    if not np.all(np.diff(timestamps) > 0.0) or not compare_float(timestamps[0], 0.0, 1e-10, 0.0) or not compare_float(timestamps[-1], EXPECTED_DURATION_S, 1e-10, 0.0):
        raise Stage39Error("Stage22 snapshot timestamp contract mismatch")
    if snapshot.get("source_frame") != EXPECTED_FRAME or snapshot.get("tcp_semantic_id") != EXPECTED_TCP:
        raise Stage39Error("Stage22 snapshot frame/TCP contract mismatch")
    state = np.asarray(snapshot.get("state_v1", {}).get("values"), dtype=np.float64)
    if state.shape != (10,) or not np.isfinite(state).all():
        raise Stage39Error("Stage22 snapshot state_v1 mismatch")

    physical_hashes: Dict[str, str] = {}
    expert_physical: Optional[np.ndarray] = None
    for row in learned:
        artifact = Path(str(row.get("artifact", {}).get("path", "")))
        expected_artifact = row.get("artifact", {}).get("sha256")
        if artifact.parent != EXPECTED_STAGE38_ROOT / "free_traces" or not artifact.is_file() or artifact.is_symlink():
            raise Stage39Error("Stage38 learned artifact path is not the sealed free_traces path")
        if sha256_file(artifact) != expected_artifact:
            raise Stage39Error("Stage38 learned artifact hash mismatch: %s" % artifact)
        with np.load(str(artifact), allow_pickle=False) as arrays:
            if "decoded_physical_action" not in arrays or "expert_physical_action" not in arrays:
                raise Stage39Error("Stage38 learned artifact lacks physical arrays")
            decoded = np.asarray(arrays["decoded_physical_action"])
            expert = np.asarray(arrays["expert_physical_action"])
            if decoded.shape != (1, EXPECTED_POINT_COUNT, 10) or expert.shape != (1, EXPECTED_POINT_COUNT, 10):
                raise Stage39Error("Stage38 learned physical array shape mismatch")
            if decoded.dtype != np.float32 or expert.dtype != np.float32:
                raise Stage39Error("Stage38 learned physical arrays are not float32")
            if not np.isfinite(decoded).all() or not np.isfinite(expert).all():
                raise Stage39Error("Stage38 learned physical arrays are not finite")
            physical_hashes[str(int(row["seed"]))] = array_sha256(decoded)
            if expert_physical is None:
                expert_physical = expert.copy()
            elif not np.array_equal(expert_physical, expert):
                raise Stage39Error("Stage38 expert reference differs across learned artifacts")
        decoded_meta = row.get("decoded_structured", {})
        if decoded_meta.get("source_frame") != EXPECTED_FRAME or decoded_meta.get("tcp_semantic_id") != EXPECTED_TCP:
            raise Stage39Error("Stage38 learned frame/TCP metadata mismatch")
        if not compare_float(decoded_meta.get("source_duration_s"), EXPECTED_DURATION_S, 1e-10, 0.0):
            raise Stage39Error("Stage38 learned source duration mismatch")
        if decoded_meta.get("timing_mode") != "T16":
            raise Stage39Error("Stage38 learned timing mode mismatch")
        safety = row.get("metrics_to_expert", {}).get("safety_gate", {})
        if safety.get("decision") not in ("ACCEPT", "WARN", "REJECT"):
            raise Stage39Error("Stage38 learned SafetyGate decision is invalid")
        result["learned_rows"].append(
            {
                "seed": int(row["seed"]),
                "artifact": file_record(artifact, expected_artifact),
                "physical_bytes_sha256": physical_hashes[str(int(row["seed"]))],
                "safety_gate": safety,
                "chunk_id": decoded_meta.get("chunk_id"),
                "source_frame": decoded_meta.get("source_frame"),
                "tcp_semantic_id": decoded_meta.get("tcp_semantic_id"),
                "source_duration_s": decoded_meta.get("source_duration_s"),
            }
        )
    if expert_physical is None or array_sha256(expert_physical) != EXPECTED_EXPERT_PHYSICAL_SHA256:
        raise Stage39Error("Stage38 sealed expert physical hash mismatch")
    result["snapshot"] = {
        "path": str(EXPECTED_SNAPSHOT),
        "sha256": sha256_file(EXPECTED_SNAPSHOT),
        "expected_sha256": EXPECTED_SNAPSHOT_SHA256,
        "timestamps": timestamps.tolist(),
        "state_v1": state.tolist(),
        "source_frame": EXPECTED_FRAME,
        "tcp_semantic_id": EXPECTED_TCP,
        "duration_s": float(timestamps[-1] - timestamps[0]),
    }
    if result["snapshot"]["sha256"] != EXPECTED_SNAPSHOT_SHA256:
        raise Stage39Error("Stage22 snapshot hash mismatch")
    result["expert_physical_bytes_sha256"] = EXPECTED_EXPERT_PHYSICAL_SHA256
    result["checkpoint"] = payload.get("stage38d", {})
    result["status"] = "PASS"
    return result


def verify_wbc_authority(source_root: Path) -> Dict[str, Any]:
    source_root = Path(source_root).resolve()
    if source_root != EXPECTED_WBC_SOURCE_ROOT.resolve():
        raise Stage39Error("WBC authority source root is not the accepted extracted root")
    selected = {
        "arm_checkpoint": source_root / AUTHORITY_RUN_REL / "checkpoints_arm/ac_weights_039999.pt",
        "dog_checkpoint": source_root / AUTHORITY_RUN_REL / "checkpoints_dog/ac_weights_039999.pt",
        "native_config": source_root / AUTHORITY_RUN_REL / "scripts/config/wbc.py",
        "native_urdf": source_root / "resources/robots/go2_x5_v3/urdf/go2_x5.urdf",
        "native_reach_table": source_root / "resources/reach_tables/go2_x5_2d.pt",
        "trajectory_module": source_root / "modules/trajectory.py",
        "wbc_env": source_root / "go1_gym/envs/roboduet/wbc_env.py",
        "legged_robot": source_root / "go1_gym/envs/roboduet/legged_robot.py",
        "wbc_wrapper": source_root / "go1_gym/envs/roboduet/wbc_env_wrapper.py",
    }
    components = {
        name: file_record(path, AUTHORITY_COMPONENT_HASHES[name])
        for name, path in selected.items()
    }
    if not all(item["hash_match"] for item in components.values()):
        raise Stage39Error("WBC authority component hash mismatch")
    manifests = {
        name: file_record(path, expected)
        for name, (path, expected) in EXPECTED_WBC_MANIFESTS.items()
    }
    if not all(item["hash_match"] for item in manifests.values()):
        raise Stage39Error("WBC authority manifest hash mismatch")
    archive = file_record(EXPECTED_WBC_ARCHIVE, EXPECTED_WBC_ARCHIVE_SHA256)
    if not archive["hash_match"]:
        raise Stage39Error("WBC authority archive hash mismatch")
    tree_manifest = load_json(EXPECTED_WBC_MANIFESTS["tree_manifest"][0])
    observed_tree = tree_manifest.get("archive_tree_sha256", tree_manifest.get("tree_sha256"))
    if observed_tree != EXPECTED_WBC_TREE_SHA256:
        raise Stage39Error("WBC authority tree identity mismatch")
    return {
        "source_root": str(source_root),
        "components": components,
        "manifests": manifests,
        "archive": archive,
        "archive_tree_sha256": observed_tree,
        "status": "PASS",
    }


def physical_to_chunk(
    physical: np.ndarray,
    timestamps: np.ndarray,
    *,
    chunk_id: str,
    source: str,
    provenance: Optional[Mapping[str, Any]] = None,
) -> TrajectoryChunkV1:
    physical = np.asarray(physical, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if physical.shape != (EXPECTED_POINT_COUNT, 10):
        raise Stage39Error("physical trajectory must be [40,10]")
    if timestamps.shape != (EXPECTED_POINT_COUNT,) or not np.all(np.diff(timestamps) > 0.0):
        raise Stage39Error("trajectory timestamps are invalid")
    matrices = rot6d_rows_to_matrix(physical[:, 3:9])
    return TrajectoryChunkV1(
        chunk_id=chunk_id,
        source=source,
        observation_timestamp=float(timestamps[0]),
        point_timestamps=timestamps,
        eef_position=physical[:, :3],
        eef_orientation=matrix_to_quat(matrices),
        source_frame=EXPECTED_FRAME,
        tcp_semantic_id=EXPECTED_TCP,
        validity_start=float(timestamps[0]),
        validity_end=float(timestamps[-1]),
        gripper=physical[:, 9],
        provenance={} if provenance is None else dict(provenance),
    )


def chunk_to_native_arrays(chunk: TrajectoryChunkV1) -> Dict[str, np.ndarray]:
    return {
        "timestamp": np.asarray(chunk.point_timestamps, dtype=np.float64),
        "position": np.asarray(chunk.eef_position, dtype=np.float64),
        "quaternion": np.asarray(chunk.eef_orientation, dtype=np.float64),
        "gripper": np.asarray(chunk.gripper, dtype=np.float64).reshape(-1),
    }


def save_trajectory(path: Path, chunk: TrajectoryChunkV1) -> Dict[str, Any]:
    arrays = chunk_to_native_arrays(chunk)
    if arrays["timestamp"].shape != (EXPECTED_POINT_COUNT,) or not np.isfinite(arrays["timestamp"]).all():
        raise Stage39Error("native trajectory timestamp output invalid")
    if not np.isclose(arrays["timestamp"][0], 0.0, rtol=0.0, atol=1e-10) or not np.isclose(arrays["timestamp"][-1], EXPECTED_DURATION_S, rtol=0.0, atol=1e-10):
        raise Stage39Error("native trajectory duration would be changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "timestamp_sha256": array_sha256(arrays["timestamp"]),
        "position_sha256": array_sha256(arrays["position"]),
        "quaternion_sha256": array_sha256(arrays["quaternion"]),
        "duration_s": float(arrays["timestamp"][-1] - arrays["timestamp"][0]),
        "point_count": EXPECTED_POINT_COUNT,
        "frame": EXPECTED_FRAME,
        "tcp": EXPECTED_TCP,
    }


def reconstruct_c1_oracle(expert_physical: np.ndarray, snapshot: Mapping[str, Any]) -> Tuple[TrajectoryChunkV1, Dict[str, Any]]:
    timestamps = np.asarray(snapshot["future_timestamps"], dtype=np.float64)
    state = np.asarray(snapshot["state_v1"]["values"], dtype=np.float32).astype(np.float64)
    label = encode_structured_action(
        np.asarray(expert_physical, dtype=np.float64),
        timestamps,
        state,
        timing_mode="T16",
        chunk_id="C1_structured_oracle_k16t16",
        source_frame=EXPECTED_FRAME,
        tcp_semantic_id=EXPECTED_TCP,
        observation_timestamp=float(snapshot["observation_timestamp"]),
        provenance={
            "stage": "Stage39",
            "source": "Stage22 sealed TINY1 expert",
            "representation": "K16+T16",
        },
    )
    intent = decode_structured_action(label, intent_id="C1_structured_oracle_k16t16")
    chunk = materialize_structured_action(label, intent, source_timestamps=timestamps)
    physical = np.concatenate(
        [
            np.asarray(chunk.eef_position, dtype=np.float64),
            rotation_matrix_to_rot6d_rows(quat_to_matrix(chunk.eef_orientation)),
            np.asarray(chunk.gripper, dtype=np.float64).reshape(-1, 1),
        ],
        axis=-1,
    )
    return chunk, {
        "codec": "StructuredActionV1",
        "geometry_k": 16,
        "timing_mode": "T16",
        "active_scalar_count": 190,
        "physical_bytes_sha256": array_sha256(physical.astype(np.float32)),
        "roundtrip_finite": bool(np.isfinite(physical).all()),
        "source_timestamp_policy": "ORIGINAL_TIMESTAMPED_EXPERT_TIME_LAW",
        "source_frame": EXPECTED_FRAME,
        "tcp_semantic_id": EXPECTED_TCP,
    }


def safety_for_chunk(chunk: TrajectoryChunkV1, expert_chunk: TrajectoryChunkV1) -> Dict[str, Any]:
    return TrajectorySafetyGateV1().evaluate(
        chunk,
        now=float(expert_chunk.observation_timestamp),
        achieved_position=expert_chunk.eef_position[0],
        achieved_orientation=expert_chunk.eef_orientation[0],
    ).as_dict()


def stage38_learned_chunk(row: Mapping[str, Any], timestamps: np.ndarray) -> Tuple[TrajectoryChunkV1, np.ndarray]:
    artifact = Path(str(row["artifact"]["path"]))
    with np.load(str(artifact), allow_pickle=False) as arrays:
        physical = np.asarray(arrays["decoded_physical_action"], dtype=np.float32)
    chunk_id = str(row.get("decoded_structured", {}).get("chunk_id"))
    chunk = physical_to_chunk(
        physical[0],
        timestamps,
        chunk_id=chunk_id,
        source="SEALED_STAGE38E_STRUCTURED_AT_1000",
        provenance={
            "stage38e_artifact": str(artifact),
            "stage38e_artifact_sha256": sha256_file(artifact),
            "representation": "STRUCTURED@1000",
            "checkpoint": "STRUCTURED@1000",
        },
    )
    return chunk, physical


def severity_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = {"ACCEPT": 0, "WARN": 0, "REJECT": 0}
    for row in rows:
        decision = str(row.get("decision"))
        counts[decision] = counts.get(decision, 0) + 1
    return counts


def build_stage39_inputs(admission_root: Path, stage38: Mapping[str, Any]) -> Dict[str, Any]:
    if admission_root.exists():
        raise Stage39Error("Stage39-A output root already exists; do not overwrite evidence")
    admission_root.mkdir(parents=True)
    inputs_root = admission_root / "inputs"
    inputs_root.mkdir()
    snapshot = load_json(EXPECTED_SNAPSHOT)
    timestamps = np.asarray(snapshot["future_timestamps"], dtype=np.float64)
    expert_physical: Optional[np.ndarray] = None
    for row in stage38["learned_rows"]:
        if expert_physical is None:
            with np.load(str(Path(row["artifact"]["path"])), allow_pickle=False) as arrays:
                expert_physical = np.asarray(arrays["expert_physical_action"], dtype=np.float32).copy()
            break
    if expert_physical is None:
        raise Stage39Error("Stage38 expert physical reference is unavailable")
    expert_chunk = physical_to_chunk(
        expert_physical[0],
        timestamps,
        chunk_id="C0_dense_expert_tiny1",
        source="SEALED_STAGE22_EXPERT_TINY1",
        provenance={"stage": "Stage39", "reference_class": "DENSE_EXPERT_TINY1"},
    )
    c1_chunk, c1_meta = reconstruct_c1_oracle(expert_physical[0], snapshot)
    plans: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    def add_plan(
        row_id: str,
        reference_class: str,
        chunk: TrajectoryChunkV1,
        *,
        seed: Optional[int],
        source_artifact: Optional[Mapping[str, Any]],
        stage38_row: Optional[Mapping[str, Any]] = None,
        codec: Optional[Mapping[str, Any]] = None,
    ) -> None:
        traj_path = inputs_root / (row_id + ".npz")
        metadata_path = inputs_root / (row_id + ".json")
        trajectory_record = save_trajectory(traj_path, chunk)
        safety = safety_for_chunk(chunk, expert_chunk)
        sealed_safety = None if stage38_row is None else stage38_row.get("safety_gate")
        comparison = {"not_applicable": True}
        if sealed_safety is not None:
            comparison = {
                **compare_issue_lists(sealed_safety.get("issues", []), safety.get("issues", [])),
                "decision_match": sealed_safety.get("decision") == safety.get("decision"),
                "sealed_decision": sealed_safety.get("decision"),
                "recomputed_decision": safety.get("decision"),
                "metric_differences": compare_mapping_metrics(
                    sealed_safety.get("metrics", {}), safety.get("metrics", {})
                ),
            }
            comparison["match"] = bool(
                comparison["match"]
                and comparison["decision_match"]
                and len(comparison["metric_differences"]) == 0
            )
        metadata = {
            "schema": "STAGE39_NATIVE_WBC_REPLAY_CASE_METADATA_V1",
            "pi_gate": PI_GATE_ID,
            "stage": "Stage39-B",
            "row_id": row_id,
            "reference_class": reference_class,
            "seed": seed,
            "representation": "DENSE_H40" if reference_class == "DENSE_EXPERT_TINY1" else ("K16+T16" if reference_class == "STRUCTURED_ORACLE_K16T16" else "STRUCTURED@1000"),
            "geometry_k": None if reference_class == "DENSE_EXPERT_TINY1" else 16,
            "timing_mode": None if reference_class == "DENSE_EXPERT_TINY1" else "T16",
            "reference_mode": EXPECTED_REFERENCE_MODE,
            "seconds": EXPECTED_DURATION_S,
            "point_count": EXPECTED_POINT_COUNT,
            "timestamp_policy": "SOURCE_TIMESTAMPED_1P95S_PRESERVED; NATIVE_RUNNER_40POINT_RECONSTRUCTED",
            "source_frame": EXPECTED_FRAME,
            "tcp_semantic_id": EXPECTED_TCP,
            "trajectory": trajectory_record,
            "source_artifact": source_artifact,
            "stage38_safety_gate": sealed_safety,
            "pre_wbc_safety_gate": safety,
            "stage38_safety_recompute": comparison,
            "codec": None if codec is None else dict(codec),
            "dense_rerun": False,
            "new_gr00t_inference": False,
            "new_structured_sampler_call": False,
        }
        write_json(metadata_path, metadata)
        plan = {
            "row_id": row_id,
            "reference_class": reference_class,
            "seed": seed,
            "trajectory_path": str(traj_path),
            "trajectory_sha256": sha256_file(traj_path),
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "trajectory_record": trajectory_record,
            "source_artifact": source_artifact,
            "pre_wbc_safety_gate": safety,
            "stage38_safety_gate": sealed_safety,
            "stage38_safety_recompute": comparison,
            "geometry_k": metadata["geometry_k"],
            "timing_mode": metadata["timing_mode"],
            "reference_mode": EXPECTED_REFERENCE_MODE,
            "seconds": EXPECTED_DURATION_S,
            "frame": EXPECTED_FRAME,
            "tcp": EXPECTED_TCP,
        }
        plans.append(plan)
        if reference_class == "STRUCTURED_LEARNED":
            warnings.append(
                {
                    "row_id": row_id,
                    "seed": seed,
                    "decision": safety["decision"],
                    "issues": safety.get("issues", []),
                    "warning_details": [
                        {
                            "code": issue.get("code"),
                            "severity": issue.get("severity"),
                            "provenance": issue.get("provenance"),
                            "observed": issue.get("observed"),
                            "threshold": issue.get("threshold"),
                        }
                        for issue in safety.get("issues", [])
                        if issue.get("severity") == "WARNING"
                    ],
                    "sealed_stage38_decision": None if sealed_safety is None else sealed_safety.get("decision"),
                    "stage38_recompute_match": comparison,
                }
            )

    add_plan(
        "C0_dense_expert_tiny1",
        "DENSE_EXPERT_TINY1",
        expert_chunk,
        seed=None,
        source_artifact={
            "role": "STAGE22_TINY1_EXPERT_SNAPSHOT",
            "path": str(EXPECTED_SNAPSHOT),
            "sha256": EXPECTED_SNAPSHOT_SHA256,
            "physical_bytes_sha256": EXPECTED_EXPERT_PHYSICAL_SHA256,
        },
    )
    add_plan(
        "C1_structured_oracle_k16t16",
        "STRUCTURED_ORACLE_K16T16",
        c1_chunk,
        seed=None,
        source_artifact={
            "role": "C0_EXPERT_REENCODED_THROUGH_STRUCTURED_ACTION_V1",
            "expert_physical_sha256": EXPECTED_EXPERT_PHYSICAL_SHA256,
        },
        codec=c1_meta,
    )
    for index, stage38_row in enumerate(stage38["learned_rows"], start=1):
        seed = int(stage38_row["seed"])
        chunk, physical = stage38_learned_chunk(stage38_row, timestamps)
        add_plan(
            "L%02d_seed%d" % (index, seed),
            "STRUCTURED_LEARNED",
            chunk,
            seed=seed,
            source_artifact={
                "role": "SEALED_STAGE38E_STRUCTURED_AT_1000",
                "path": stage38_row["artifact"]["path"],
                "sha256": stage38_row["artifact"]["sha256"],
                "physical_bytes_sha256": array_sha256(physical),
                "checkpoint": "STRUCTURED@1000",
            },
        )

    if tuple(item["row_id"] for item in plans) != tuple(item[0] for item in EXPECTED_ROW_IDS):
        raise Stage39Error("Stage39 input row order or identity mismatch")
    learned_safety = [item["pre_wbc_safety_gate"] for item in plans if item["reference_class"] == "STRUCTURED_LEARNED"]
    counts = severity_counts(learned_safety)
    if counts != {"ACCEPT": 3, "WARN": 13, "REJECT": 0}:
        raise Stage39Error("Stage38-E SafetyGate counts changed: %s" % counts)
    for item in plans:
        if item["reference_class"] == "STRUCTURED_LEARNED":
            comparison = item["stage38_safety_recompute"]
            if not comparison.get("match"):
                raise Stage39Error("Stage38-E SafetyGate recomputation mismatch: %s" % item["row_id"])
            if item["pre_wbc_safety_gate"].get("decision") == "REJECT":
                raise Stage39Error("Stage39-A found a new learned SafetyGate REJECT")

    warn_payload = {
        "schema": "STAGE39_STAGE38E_WARN_STRATIFICATION_V1",
        "pi_gate": PI_GATE_ID,
        "learned_row_count": len(warnings),
        "decision_counts": counts,
        "rows": warnings,
        "warning_issue_count": sum(len(row["warning_details"]) for row in warnings),
        "unknown_warning_code_or_provenance": 0,
        "stage38_safety_decisions_preserved": True,
    }
    write_json(admission_root / "stage39_warn_stratification.json", warn_payload)
    write_json(
        admission_root / "stage39_input_manifest.json",
        {
            "schema": "STAGE39_NATIVE_WBC_REPLAY_INPUT_MANIFEST_V1",
            "pi_gate": PI_GATE_ID,
            "stage": "Stage39-A/B",
            "project": EXPECTED_PROJECT,
            "queue": EXPECTED_QUEUE,
            "reference_mode": EXPECTED_REFERENCE_MODE,
            "duration_s": EXPECTED_DURATION_S,
            "point_count": EXPECTED_POINT_COUNT,
            "rows": plans,
            "row_count": len(plans),
            "learned_seed_set": list(EXPECTED_SEEDS),
            "stage38_root": str(EXPECTED_STAGE38_ROOT),
            "stage38_result_sha256": EXPECTED_STAGE38_FILES["result"][1],
            "stage38_artifact_seal_sha256": EXPECTED_STAGE38_FILES["artifact_seal"][1],
            "expert_physical_bytes_sha256": EXPECTED_EXPERT_PHYSICAL_SHA256,
            "no_dense_rerun": True,
            "new_gr00t_inference": False,
            "new_structured_sampler_calls": 0,
        },
    )
    manifest_path = admission_root / "stage39_input_manifest.json"
    warn_path = admission_root / "stage39_warn_stratification.json"
    recomputed_decisions = [item["pre_wbc_safety_gate"]["decision"] for item in plans if item["reference_class"] == "STRUCTURED_LEARNED"]
    stage39a = {
        "schema": "STAGE39A_STRUCTURED_WBC_REPLAY_CPU_ADMISSION_V1",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "issue_comment": ISSUE_COMMENT_ID,
        "pr_handoff_comment": PR_HANDOFF_COMMENT_ID,
        "branch": STAGE39_BRANCH,
        "base_branch": STAGE38_BRANCH,
        "accepted_stage38_head": EXPECTED_STAGE38_HEAD,
        "stage38_chain": stage38,
        "wbc_authority": verify_wbc_authority(EXPECTED_WBC_SOURCE_ROOT),
        "input_manifest_path": str(manifest_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "warn_stratification_path": str(warn_path),
        "warn_stratification_sha256": sha256_file(warn_path),
        "primary_rows_exact": "16/16",
        "seeds_exact": list(EXPECTED_SEEDS),
        "physical_trajectory_hashes": "MATCH_SEALED_STAGE38E",
        "finite": "16/16",
        "frame_tcp_contract": "PASS",
        "timestamps": {
            "status": "VALID_AND_UNCHANGED",
            "point_count": EXPECTED_POINT_COUNT,
            "duration_s": EXPECTED_DURATION_S,
            "source_timestamps_preserved": True,
            "retiming": False,
            "sorting": False,
            "smoothing": False,
            "clipping": False,
        },
        "safety_gate_recomputation": {
            "sealed_decisions": {"ACCEPT": 3, "WARN": 13, "REJECT": 0},
            "recomputed_decisions": severity_counts(
                [{"decision": value} for value in recomputed_decisions]
            ),
            "decision_vector_exact": True,
            "all_warning_details_recorded": True,
            "unknown_warning_code_or_provenance": 0,
        },
        "static_guard": {
            "selected_execution_mode": "native external WBC replay only",
            "gr00t_model_construction": False,
            "gr00t_forward": False,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "scheduler_step": False,
            "resume": False,
            "checkpoint_write": False,
            "new_structured_sampler_calls": 0,
            "qsub_in_stage39_runner": False,
        },
        "scientific_contract_changed": False,
        "model_constructed": False,
        "gr00t_inference": False,
        "wbc_executed": False,
        "pbs_submitted": False,
        "historical_artifacts_modified": False,
        "status": "PASS",
        "next_gate": "STAGE39_B_ONE_BOUNDED_NATIVE_WBC_ALLOCATION",
    }
    write_json(admission_root / "stage39a_result.json", stage39a)
    return stage39a


def ensure_fresh_campaign_root(root: Path) -> None:
    if not root.exists():
        root.mkdir(parents=True)
        return
    if root.is_symlink() or not root.is_dir():
        raise Stage39Error("campaign root is not a real directory")
    allowed = {"runtime", "stage39_job.log"}
    entries = {item.name for item in root.iterdir()}
    if not entries <= allowed or "runtime" not in entries:
        raise Stage39Error("campaign root is not the fresh wrapper scaffold")
    runtime = root / "runtime"
    if runtime.is_symlink() or not runtime.is_dir():
        raise Stage39Error("campaign runtime scaffold invalid")
    for item in runtime.rglob("*"):
        if item.is_symlink() or item.is_file():
            raise Stage39Error("campaign runtime scaffold contains data: %s" % item)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    result: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.append(json.loads(line))
    finite_json(result, str(path))
    return result


def native_finalized(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], timed_out: bool) -> bool:
    if timed_out or not rows:
        return False
    if summary.get("status") not in {
        "NATIVE_BENCHMARK_40POINT_RECONSTRUCTED_PASS",
        "NATIVE_BENCHMARK_40POINT_RECONSTRUCTED_PARTIAL",
    }:
        return False
    if summary.get("error") is not None:
        return False
    if str(summary.get("termination")) not in FINALIZED_NATIVE_TERMINATIONS:
        return False
    if int(summary.get("steps", -1)) != len(rows):
        return False
    return bool(summary.get("reference_completed", False) or summary.get("termination") in FINALIZED_NATIVE_TERMINATIONS)


def load_native_helpers() -> Any:
    import stage36_wbc_oracle_campaign as helpers

    return helpers


def quat_error(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    return geodesic_angle(quat_to_matrix(q0), quat_to_matrix(q1))


def path_length(values: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(values, axis=0), axis=-1).sum()) if len(values) > 1 else 0.0


def se3_arc_length(position: np.ndarray, quaternion: np.ndarray) -> float:
    if len(position) <= 1:
        return 0.0
    translation = np.linalg.norm(np.diff(position, axis=0), axis=-1)
    rotation = quat_error(quaternion[:-1], quaternion[1:])
    return float(np.sqrt(translation * translation + (0.15 * rotation) ** 2).sum())


def native_metrics(raw_rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> Dict[str, Any]:
    if not raw_rows:
        return {"finite": False}
    desired_pos = np.asarray([row.get("desired_pos") for row in raw_rows], dtype=np.float64)
    desired_quat = np.asarray([row.get("desired_quat") for row in raw_rows], dtype=np.float64)
    achieved_pos = np.asarray([row.get("achieved_pos") for row in raw_rows], dtype=np.float64)
    achieved_quat = np.asarray([row.get("achieved_quat") for row in raw_rows], dtype=np.float64)
    base_pos = np.asarray([row.get("base_pos") for row in raw_rows], dtype=np.float64)
    base_quat = np.asarray([row.get("base_quat") for row in raw_rows], dtype=np.float64)
    arm_q = np.asarray([row.get("arm_q") for row in raw_rows], dtype=np.float64)
    pos_error = np.linalg.norm(desired_pos - achieved_pos, axis=-1)
    rot_error = quat_error(desired_quat, achieved_quat)
    reference_path = path_length(desired_pos)
    achieved_path = path_length(achieved_pos)
    ref_arc = se3_arc_length(desired_pos, desired_quat)
    achieved_arc = se3_arc_length(achieved_pos, achieved_quat)
    reference_duration = float(summary.get("metrics", {}).get("steps", len(raw_rows)))
    if "reference_duration" in raw_rows[-1]:
        reference_duration = float(raw_rows[-1].get("reference_duration"))
    finite = bool(
        np.isfinite(desired_pos).all()
        and np.isfinite(desired_quat).all()
        and np.isfinite(achieved_pos).all()
        and np.isfinite(achieved_quat).all()
        and np.isfinite(base_pos).all()
        and np.isfinite(base_quat).all()
        and np.isfinite(arm_q).all()
        and np.isfinite(pos_error).all()
        and np.isfinite(rot_error).all()
    )
    return {
        "finite": finite,
        "eef_tracking_xyz_m": {
            "mean": float(np.mean(pos_error)),
            "p95": float(np.percentile(pos_error, 95)),
            "max": float(np.max(pos_error)),
        },
        "eef_tracking_so3_rad": {
            "mean": float(np.mean(rot_error)),
            "p95": float(np.percentile(rot_error, 95)),
            "max": float(np.max(rot_error)),
        },
        "endpoint_tracking": {
            "xyz_m": float(pos_error[-1]),
            "so3_rad": float(rot_error[-1]),
        },
        "reference_path_length_m": reference_path,
        "achieved_path_length_m": achieved_path,
        "reference_se3_arc_length_m": ref_arc,
        "achieved_se3_arc_length_m": achieved_arc,
        "base_translation": {
            "displacement_m": float(np.linalg.norm(base_pos[-1] - base_pos[0])),
            "path_length_m": path_length(base_pos),
        },
        "base_yaw": {"change_rad": float(quat_error(base_quat[0], base_quat[-1]))},
        "arm_motion": {
            "q_path_length_rad": path_length(arm_q),
            "max_abs_q_rad": float(np.max(np.abs(arm_q))),
        },
        "reference_duration_s": reference_duration,
        "reference_steps": len(raw_rows),
    }


def native_row_result(plan: Mapping[str, Any], output_dir: Path, returncode: int, timed_out: bool, helpers: Any) -> Dict[str, Any]:
    summary_path = output_dir / "native_external_summary.json"
    raw_path = output_dir / "native_external_rows.jsonl"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    raw_rows = read_jsonl(raw_path)
    finite = bool(helpers.finite_value(summary) and helpers.finite_value(raw_rows))
    finalized = native_finalized(summary, raw_rows, timed_out)
    reference_complete = bool(summary.get("reference_completed", False) and summary.get("termination") == "REFERENCE_COMPLETE")
    safety = helpers.safety_profile(raw_rows)
    runner_pass = bool(
        finalized
        and finite
        and safety.get("finite") is True
        and (returncode == 0 or returncode == POST_RUN_COMPLETION_RETURN_CODE)
    )
    hard_flags = safety.get("hard_failure_flags", {})
    lifecycle_qualified = bool(
        runner_pass
        and reference_complete
        and not any(bool(value) for value in hard_flags.values())
    )
    metrics = native_metrics(raw_rows, summary) if raw_rows else {"finite": False}
    no_new_reject = str(plan.get("pre_wbc_safety_gate", {}).get("decision")) != "REJECT"
    lifecycle_qualified = bool(lifecycle_qualified and no_new_reject and metrics.get("finite") is True)
    result = {
        "row_id": plan["row_id"],
        "reference_class": plan["reference_class"],
        "seed": plan.get("seed"),
        "representation": "DENSE_H40" if plan["reference_class"] == "DENSE_EXPERT_TINY1" else ("K16+T16" if plan["reference_class"] == "STRUCTURED_ORACLE_K16T16" else "STRUCTURED@1000"),
        "reference_mode": plan["reference_mode"],
        "seconds": plan["seconds"],
        "trajectory_path": plan["trajectory_path"],
        "trajectory_sha256_planned": plan["trajectory_sha256"],
        "output_dir": str(output_dir),
        "returncode": int(returncode),
        "timed_out": bool(timed_out),
        "native_status": summary.get("status"),
        "native_termination": summary.get("termination"),
        "native_summary": summary,
        "raw_rows_path": str(raw_path),
        "raw_rows_count": len(raw_rows),
        "finite": bool(finite and safety.get("finite") is True and metrics.get("finite") is True),
        "finite_flags": {
            "summary": bool(helpers.finite_value(summary)),
            "raw_rows": bool(helpers.finite_value(raw_rows)),
            "metrics": bool(metrics.get("finite") is True),
        },
        "pre_wbc_safety_gate": plan.get("pre_wbc_safety_gate"),
        "safety_gate_transitions": safety.get("transitions", []),
        "lifecycle_flags": hard_flags,
        "joint_limit_diagnostic": bool(safety.get("diagnostic_joint_margin_violation", False)),
        "reference_complete": reference_complete,
        "scientific_row_finalized": bool(finalized),
        "runner_pass": runner_pass,
        "lifecycle_qualified": lifecycle_qualified,
        "new_safety_gate_reject": not no_new_reject,
        "metrics": metrics,
        "raw_rows_sha256": sha256_file(raw_path) if raw_path.is_file() else None,
        "summary_sha256": sha256_file(summary_path) if summary_path.is_file() else None,
    }
    write_json(output_dir / "stage39_row_result.json", result)
    return result


def campaign_command(native_runner: Path, source_root: Path, arm: Path, dog: Path, plan: Mapping[str, Any], output_dir: Path) -> List[str]:
    command = [
        sys.executable,
        str(native_runner),
        "--source-root",
        str(source_root),
        "--arm",
        str(arm),
        "--dog",
        str(dog),
        "--reference-mode",
        EXPECTED_REFERENCE_MODE,
        "--trajectory",
        str(plan["trajectory_path"]),
        "--output-dir",
        str(output_dir),
        "--seconds",
        str(EXPECTED_DURATION_S),
        "--seed",
        str(13 if plan.get("seed") is None else plan["seed"]),
        "--metadata-json",
        str(plan["metadata_path"]),
    ]
    forbidden = ("--train", "--resume", "--no-stop-at-reference-completion", "--disable-trajectory-termination", "qsub")
    text = " ".join(command)
    if any(token in text for token in forbidden):
        raise Stage39Error("selected native WBC command contains a forbidden token")
    if "--reference-mode %s" % EXPECTED_REFERENCE_MODE not in text:
        raise Stage39Error("selected native WBC command mode mismatch")
    return command


def write_campaign_result(root: Path, plans: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], error: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    c0 = next((row for row in rows if row.get("row_id") == "C0_dense_expert_tiny1"), None)
    c1 = next((row for row in rows if row.get("row_id") == "C1_structured_oracle_k16t16"), None)
    learned = [row for row in rows if row.get("reference_class") == "STRUCTURED_LEARNED"]
    learned_qualified = sum(bool(row.get("lifecycle_qualified")) for row in learned)
    complete = bool(len(rows) == len(plans) and all(bool(row.get("scientific_row_finalized")) for row in rows))
    controls_pass = bool(c0 and c1 and c0.get("lifecycle_qualified") and c1.get("lifecycle_qualified"))
    if not complete or not controls_pass:
        classification = "STRUCTURED_WBC_REPLAY_CONTROL_OR_AUTHORITY_FAILURE" if complete and not controls_pass else "STRUCTURED_WBC_REPLAY_PARTIAL_OR_BLOCKED"
    elif learned_qualified == 16:
        classification = "STRUCTURED_WBC_REPLAY_FULLY_QUALIFIED"
    elif learned_qualified >= 12:
        classification = "STRUCTURED_WBC_REPLAY_LIFECYCLE_MOSTLY_QUALIFIED"
    elif learned_qualified >= 8:
        classification = "STRUCTURED_WBC_REPLAY_MIXED"
    else:
        classification = "STRUCTURED_WBC_REPLAY_NO_CLEAR_COMPATIBILITY"
    safety_counts = severity_counts([row.get("pre_wbc_safety_gate", {}) for row in learned])
    result = {
        "schema": "STAGE39_STRUCTURED_WBC_REPLAY_RESULT_V1",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "issue_comment": ISSUE_COMMENT_ID,
        "pr_handoff_comment": PR_HANDOFF_COMMENT_ID,
        "pr": PR_NUMBER,
        "branch": STAGE39_BRANCH,
        "project": EXPECTED_PROJECT,
        "queue": EXPECTED_QUEUE,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "reference_mode": EXPECTED_REFERENCE_MODE,
        "reference_duration_s": EXPECTED_DURATION_S,
        "planned_row_count": len(plans),
        "row_count": len(rows),
        "all_planned_rows_completed": complete,
        "all_finite": bool(rows) and all(bool(row.get("finite")) for row in rows),
        "control_rows": {
            "C0_dense_expert_tiny1": c0,
            "C1_structured_oracle_k16t16": c1,
            "both_lifecycle_qualified": controls_pass,
        },
        "learned_rows": {
            "planned": 16,
            "completed": len(learned),
            "lifecycle_qualified": learned_qualified,
            "safety_gate_stratification": safety_counts,
            "new_safety_gate_reject_count": sum(bool(row.get("new_safety_gate_reject")) for row in learned),
        },
        "classification": classification,
        "rows": list(rows),
        "forbidden_execution": {
            "gr00t_model_construction": False,
            "gr00t_forward": False,
            "new_structured_sampler_calls": 0,
            "dense_sampler_calls": 0,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "scheduler_step": False,
            "resume": False,
            "checkpoint_write": False,
            "free_running_gr00t": False,
            "closed_loop_hardware": False,
            "second_pbs_allocation": False,
        },
        "execution": {
            "wbc_executed": bool(rows),
            "one_bounded_pbs_allocation_only": True,
            "scientific_execution_begun": bool(rows),
            "native_rows_started": len(rows),
        },
        "historical_artifacts_modified": False,
        "error": None if error is None else dict(error),
        "status": "PASS_SCIENTIFIC_RESULT_NEEDS_PI_REVIEW" if complete else "PARTIAL_OR_BLOCKED_NEEDS_PI_REVIEW",
        "next_gate": NEXT_GATE,
    }
    write_json(root / "stage39_result.json", result)
    raw_rows = []
    for row in rows:
        raw_path = Path(str(row.get("raw_rows_path", "")))
        summary_path = Path(str(row.get("output_dir", ""))) / "native_external_summary.json"
        raw_rows.append(
            {
                "row_id": row.get("row_id"),
                "summary_path": str(summary_path),
                "summary_sha256": sha256_file(summary_path) if summary_path.is_file() else None,
                "raw_rows_path": str(raw_path),
                "raw_rows_sha256": sha256_file(raw_path) if raw_path.is_file() else None,
                "stage39_row_result_path": str(Path(str(row.get("output_dir", ""))) / "stage39_row_result.json"),
                "stage39_row_result_sha256": sha256_file(Path(str(row.get("output_dir", ""))) / "stage39_row_result.json") if (Path(str(row.get("output_dir", ""))) / "stage39_row_result.json").is_file() else None,
            }
        )
    write_json(
        root / "stage39_raw_hash_manifest.json",
        {
            "schema": "STAGE39_RAW_HASH_MANIFEST_V1",
            "pi_gate": PI_GATE_ID,
            "row_count": len(raw_rows),
            "rows": raw_rows,
        },
    )
    write_json(
        root / "stage39_result_manifest.json",
        {
            "schema": "STAGE39_RESULT_MANIFEST_V1",
            "pi_gate": PI_GATE_ID,
            "result_path": str(root / "stage39_result.json"),
            "result_sha256": sha256_file(root / "stage39_result.json"),
            "raw_hash_manifest_path": str(root / "stage39_raw_hash_manifest.json"),
            "raw_hash_manifest_sha256": sha256_file(root / "stage39_raw_hash_manifest.json"),
            "row_count": len(rows),
            "all_planned_rows_completed": complete,
            "classification": classification,
            "scientific_contract_changed": False,
            "next_gate": NEXT_GATE,
        },
    )
    return result


def run_campaign(args: argparse.Namespace) -> int:
    root = Path(args.output_root).absolute()
    ensure_fresh_campaign_root(root)
    admission_root = Path(args.admission_root).resolve()
    admission_result_path = admission_root / "stage39a_result.json"
    manifest_path = admission_root / "stage39_input_manifest.json"
    warn_path = admission_root / "stage39_warn_stratification.json"
    state = {
        "schema": "STAGE39_EXECUTION_STATE_V1",
        "pi_gate": PI_GATE_ID,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "started_utc": now_utc(),
        "scientific_execution_begun": False,
        "wbc_rows_started": 0,
        "wbc_rows_finalized": 0,
        "second_pbs_allocation": False,
    }
    write_json(root / "stage39_execution_state.json", state)
    plans: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    helpers = load_native_helpers()
    error: Optional[Dict[str, Any]] = None
    try:
        if os.environ.get("PBS_O_QUEUE") != EXPECTED_QUEUE:
            raise Stage39Error("PBS request queue is not auto")
        admission = load_json(admission_result_path)
        if admission.get("status") != "PASS" or admission.get("pi_gate") != PI_GATE_ID:
            raise Stage39Error("Stage39-A admission is not PASS")
        if args.expected_admission_sha256 and sha256_file(admission_result_path) != args.expected_admission_sha256:
            raise Stage39Error("Stage39-A result hash changed")
        manifest = load_json(manifest_path)
        if manifest.get("row_count") != 18 or len(manifest.get("rows", [])) != 18:
            raise Stage39Error("Stage39 input manifest is not exactly 18 rows")
        if args.expected_manifest_sha256 and sha256_file(manifest_path) != args.expected_manifest_sha256:
            raise Stage39Error("Stage39 input manifest hash changed")
        if args.expected_warn_sha256 and sha256_file(warn_path) != args.expected_warn_sha256:
            raise Stage39Error("Stage39 warning stratification hash changed")
        plans = [dict(item) for item in manifest["rows"]]
        if tuple(item["row_id"] for item in plans) != tuple(item[0] for item in EXPECTED_ROW_IDS):
            raise Stage39Error("Stage39 campaign row identity/order mismatch")
        authority = verify_wbc_authority(Path(args.source_root))
        if sha256_file(NATIVE_RUNNER) != NATIVE_RUNNER_SHA256:
            raise Stage39Error("native WBC runner hash is not the accepted authority")
        if not Path(args.arm).is_file() or not Path(args.dog).is_file():
            raise Stage39Error("native checkpoint path is missing")
        if not Path(args.native_runner).is_file():
            raise Stage39Error("native runner path is missing")
        runtime_binding = {
            "schema": "STAGE39_RUNTIME_BINDING_V1",
            "identity_class": "RECONSTRUCTED_COMPATIBLE_RUNTIME",
            "hostname": platform.node(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "source_root": str(Path(args.source_root).resolve()),
            "authority": authority,
            "runtime_site": os.environ.get("STAGE39_RUNTIME_SITE"),
            "runtime_ninja": os.environ.get("STAGE39_RUNTIME_NINJA"),
            "apptainer_image": os.environ.get("STAGE39_RUNTIME_IMAGE"),
            "requested_queue": os.environ.get("PBS_O_QUEUE"),
            "execution_queue": os.environ.get("PBS_QUEUE"),
            "gpu_name": os.environ.get("STAGE39_GPU_NAME"),
            "gpu_count": os.environ.get("STAGE39_GPU_COUNT"),
            "status": "PASS",
        }
        write_json(root / "stage39_runtime_binding.json", runtime_binding)
        write_json(root / "stage39_input_snapshot.json", {
            "admission_result": file_record(admission_result_path, args.expected_admission_sha256),
            "input_manifest": file_record(manifest_path, args.expected_manifest_sha256),
            "warn_stratification": file_record(warn_path, args.expected_warn_sha256),
            "authority": authority,
        })
        for index, plan in enumerate(plans, start=1):
            row_output = root / "rows" / str(plan["row_id"])
            if row_output.exists():
                raise Stage39Error("row output already exists: %s" % row_output)
            row_output.mkdir(parents=True)
            input_copy = row_output / "reference_input.npz"
            metadata_copy = row_output / "case_metadata.json"
            shutil.copy2(plan["trajectory_path"], input_copy)
            shutil.copy2(plan["metadata_path"], metadata_copy)
            if sha256_file(input_copy) != plan["trajectory_sha256"] or sha256_file(metadata_copy) != plan["metadata_sha256"]:
                raise Stage39Error("row input copy hash mismatch: %s" % plan["row_id"])
            command = campaign_command(
                Path(args.native_runner),
                Path(args.source_root).resolve(),
                Path(args.arm).resolve(),
                Path(args.dog).resolve(),
                {**plan, "trajectory_path": str(input_copy), "metadata_path": str(metadata_copy)},
                row_output,
            )
            state["scientific_execution_begun"] = True
            state["wbc_rows_started"] = index
            write_json(root / "stage39_execution_state.json", state)
            log_event(root, "ROW_START", index=index, row_id=plan["row_id"], command=command)
            run_log = row_output / "run.log"
            timed_out = False
            with run_log.open("w", encoding="utf-8") as handle:
                try:
                    completed = subprocess.run(
                        command,
                        cwd=str(Path(args.source_root).resolve()),
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=float(args.timeout_seconds),
                    )
                    returncode = int(completed.returncode)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    returncode = 124
                    handle.write("STAGE39_TIMEOUT\n")
            row = native_row_result(plan, row_output, returncode, timed_out, helpers)
            rows.append(row)
            state["wbc_rows_finalized"] = len(rows)
            write_json(root / "stage39_partial_records.json", rows)
            write_json(root / "stage39_execution_state.json", state)
            log_event(
                root,
                "ROW_END",
                row_id=plan["row_id"],
                returncode=returncode,
                scientific_row_finalized=row["scientific_row_finalized"],
                lifecycle_qualified=row["lifecycle_qualified"],
            )
            if not row["scientific_row_finalized"]:
                raise Stage39Error("native row did not finalize; no row retry is authorized")
        result = write_campaign_result(root, plans, rows)
        state["finished_utc"] = now_utc()
        state["status"] = result["status"]
        write_json(root / "stage39_execution_state.json", state)
        return 0 if result["all_planned_rows_completed"] else 2
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        log_event(root, "ERROR", **error)
        write_campaign_result(root, plans, rows, error=error)
        state["finished_utc"] = now_utc()
        state["status"] = "PARTIAL_OR_BLOCKED"
        state["error"] = error
        write_json(root / "stage39_execution_state.json", state)
        write_json(root / "stage39_partial_records.json", rows)
        return 2


def independent_validation(root: Path, expected_head: str, api_head: str) -> Dict[str, Any]:
    result_path = root / "stage39_result.json"
    raw_manifest_path = root / "stage39_raw_hash_manifest.json"
    result_manifest_path = root / "stage39_result_manifest.json"
    result = load_json(result_path)
    raw_manifest = load_json(raw_manifest_path)
    result_manifest = load_json(result_manifest_path)
    errors: List[str] = []
    expected_ids = {item[0] for item in EXPECTED_ROW_IDS}
    actual_ids = {str(row.get("row_id")) for row in result.get("rows", [])}
    if result.get("row_count") != 18 or actual_ids != expected_ids:
        errors.append("row_identity")
    if result.get("reference_mode") != EXPECTED_REFERENCE_MODE or not compare_float(result.get("reference_duration_s"), EXPECTED_DURATION_S, 1e-10, 0.0):
        errors.append("reference_contract")
    if result.get("forbidden_execution", {}).get("gr00t_model_construction") is not False or result.get("forbidden_execution", {}).get("training") is not False:
        errors.append("forbidden_execution")
    manifest_by_id = {str(row.get("row_id")): row for row in raw_manifest.get("rows", [])}
    audits: List[Dict[str, Any]] = []
    helpers = load_native_helpers()
    for row in result.get("rows", []):
        row_id = str(row.get("row_id"))
        output_dir = Path(str(row.get("output_dir", "")))
        summary_path = output_dir / "native_external_summary.json"
        raw_path = output_dir / "native_external_rows.jsonl"
        summary = load_json(summary_path) if summary_path.is_file() else {}
        raw_rows = read_jsonl(raw_path)
        summary_hash = sha256_file(summary_path) if summary_path.is_file() else None
        raw_hash = sha256_file(raw_path) if raw_path.is_file() else None
        independent_safety = helpers.safety_profile(raw_rows)
        finalized = native_finalized(summary, raw_rows, bool(row.get("timed_out")))
        metrics = native_metrics(raw_rows, summary) if raw_rows else {"finite": False}
        row_error = []
        entry = manifest_by_id.get(row_id)
        if entry is None:
            row_error.append("raw_manifest_missing")
        else:
            if entry.get("summary_sha256") != summary_hash or entry.get("raw_rows_sha256") != raw_hash:
                row_error.append("raw_hash")
        if not finalized or not metrics.get("finite") or not helpers.finite_value(summary) or not helpers.finite_value(raw_rows):
            row_error.append("finite_or_finalized")
        if str(row.get("native_termination")) != str(summary.get("termination")):
            row_error.append("termination")
        audits.append({
            "row_id": row_id,
            "summary_sha256": summary_hash,
            "raw_rows_sha256": raw_hash,
            "raw_rows_count": len(raw_rows),
            "finalized": finalized,
            "finite": bool(metrics.get("finite") and helpers.finite_value(summary) and helpers.finite_value(raw_rows)),
            "independent_safety": independent_safety,
            "independent_metrics": metrics,
            "errors": row_error,
        })
        if row_error:
            errors.extend(["%s:%s" % (row_id, item) for item in row_error])
    result_manifest_ok = bool(
        result_manifest.get("result_sha256") == sha256_file(result_path)
        and result_manifest.get("raw_hash_manifest_sha256") == sha256_file(raw_manifest_path)
    )
    if not result_manifest_ok:
        errors.append("result_manifest")
    independent = {
        "schema": "STAGE39_INDEPENDENT_VALIDATION_V1",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "expected_head": expected_head,
        "api_head": api_head,
        "result_sha256": sha256_file(result_path),
        "raw_hash_manifest_sha256": sha256_file(raw_manifest_path),
        "result_manifest_sha256": sha256_file(result_manifest_path),
        "row_audits": audits,
        "result_manifest_consistent": result_manifest_ok,
        "errors": errors,
        "status": "PASS" if not errors else "BLOCKED_INDEPENDENT_VALIDATION",
        "historical_artifacts_modified": False,
    }
    write_json(root / "stage39_independent_validation.json", independent)
    if errors:
        raise Stage39Error("independent Stage39 validation failed: %s" % ",".join(errors[:20]))
    return independent


def seal_campaign(args: argparse.Namespace) -> int:
    root = Path(args.output_root).resolve()
    if not root.is_dir() or not str(root).startswith("/scratch/"):
        raise Stage39Error("Stage39 seal root must be scratch-backed")
    result = load_json(root / "stage39_result.json")
    if result.get("pi_gate") != PI_GATE_ID or result.get("planned_row_count") != 18:
        raise Stage39Error("Stage39 result contract mismatch before seal")
    independent = independent_validation(root, args.expected_head, args.api_head)
    if independent.get("status") != "PASS":
        raise Stage39Error("Stage39 independent validation is not PASS")
    source_records = []
    for role, path in (
        ("stage39_evaluator", REPO / "scripts/stage39_structured_wbc_replay.py"),
        ("stage39_pbs_wrapper", PBS_WRAPPER),
        ("native_wbc_runner", NATIVE_RUNNER),
        ("stage36_native_helper", STAGE36_HELPER),
    ):
        source_records.append({
            "role": role,
            "path": str(path),
            "size_bytes": int(path.stat().st_size) if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        })
    external_records = []
    pbs_log = Path(args.pbs_log) if args.pbs_log else None
    if pbs_log is not None and pbs_log.is_file():
        external_records.append({"role": "pbs_log", "path": str(pbs_log), "size_bytes": pbs_log.stat().st_size, "sha256": sha256_file(pbs_log)})
    excluded = {"stage39_artifact_seal.json", "stage39_final_return.json"}
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.name not in excluded:
            entries.append({
                "path": str(path),
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            })
    seal = {
        "schema": "STAGE39_STRUCTURED_WBC_REPLAY_ARTIFACT_SEAL_V1",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "pbs_job_id": result.get("pbs_job_id"),
        "project": EXPECTED_PROJECT,
        "branch": STAGE39_BRANCH,
        "expected_head": args.expected_head,
        "api_head": args.api_head,
        "artifact_root": str(root),
        "entries": entries,
        "source_artifacts": source_records,
        "external_artifacts": external_records,
        "wbc_authority_archive": file_record(EXPECTED_WBC_ARCHIVE, EXPECTED_WBC_ARCHIVE_SHA256),
        "stage38_result_sha256": EXPECTED_STAGE38_FILES["result"][1],
        "stage38_artifact_seal_sha256": EXPECTED_STAGE38_FILES["artifact_seal"][1],
        "historical_artifacts_modified": False,
        "scientific_contract_changed": False,
        "status": "PASS",
        "next_gate": NEXT_GATE,
    }
    seal_path = root / "stage39_artifact_seal.json"
    write_json(seal_path, seal)
    seal_sha = sha256_file(seal_path)
    final_return = {
        "ROLE": "CODEX_HOPPER_SERVER_LONG_HORIZON",
        "ACTIVE_PI_GATE": PI_GATE_ID,
        "STAGE": "Stage39 sealed StructuredActionV1 native-WBC replay qualification",
        "PR": PR_NUMBER,
        "BRANCH": STAGE39_BRANCH,
        "EXPECTED_HEAD": args.expected_head,
        "API_HEAD": args.api_head,
        "LOCAL_HEAD": args.expected_head,
        "PBS_PROJECT": EXPECTED_PROJECT,
        "PBS_JOB_ID": result.get("pbs_job_id"),
        "PBS_SUBMITTED": True,
        "GPU_USED": True,
        "RESULT": result.get("status"),
        "CLASSIFICATION": result.get("classification"),
        "C0": result.get("control_rows", {}).get("C0_dense_expert_tiny1"),
        "C1": result.get("control_rows", {}).get("C1_structured_oracle_k16t16"),
        "LEARNED": result.get("learned_rows"),
        "ARTIFACT_ROOT": str(root),
        "RESULT_PATH": str(root / "stage39_result.json"),
        "RESULT_SHA256": sha256_file(root / "stage39_result.json"),
        "INDEPENDENT_VALIDATION_PATH": str(root / "stage39_independent_validation.json"),
        "INDEPENDENT_VALIDATION_SHA256": sha256_file(root / "stage39_independent_validation.json"),
        "ARTIFACT_SEAL_PATH": str(seal_path),
        "ARTIFACT_SEAL_SHA256": seal_sha,
        "NEXT_GATE": NEXT_GATE,
        "GR00T_MODEL_CONSTRUCTED": False,
        "GR00T_FORWARD": False,
        "NEW_STRUCTURED_SAMPLER_CALLS": 0,
        "DENSE_RERUN": False,
        "TRAINING": False,
        "BACKWARD": False,
        "OPTIMIZER_STEP": False,
        "SCHEDULER_STEP": False,
        "CHECKPOINT_WRITE": False,
        "CLOSED_LOOP_HARDWARE": False,
        "SECOND_PBS_ALLOCATION": False,
        "SCIENTIFIC_CONTRACT_CHANGED": False,
        "HISTORICAL_ARTIFACTS_MODIFIED": False,
    }
    write_json(root / "stage39_final_return.json", final_return)
    verify = verify_sealed_entries(root, seal_path, seal_sha)
    if verify.get("status") != "PASS":
        raise Stage39Error("Stage39 seal self-verification failed")
    print(json.dumps({
        "status": "PASS",
        "classification": result.get("classification"),
        "result_sha256": final_return["RESULT_SHA256"],
        "seal_sha256": seal_sha,
        "final_return": str(root / "stage39_final_return.json"),
    }, sort_keys=True))
    return 0


def run_admission(args: argparse.Namespace) -> int:
    stage38 = verify_stage38_chain()
    admission_root = Path(args.output_root).absolute()
    result = build_stage39_inputs(admission_root, stage38)
    print(json.dumps({
        "status": result.get("status"),
        "admission_root": str(admission_root),
        "stage39a_result_sha256": sha256_file(admission_root / "stage39a_result.json"),
        "input_manifest_sha256": sha256_file(admission_root / "stage39_input_manifest.json"),
        "warn_stratification_sha256": sha256_file(admission_root / "stage39_warn_stratification.json"),
    }, sort_keys=True))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--admission", action="store_true")
    mode.add_argument("--campaign", action="store_true")
    mode.add_argument("--seal", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--admission-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--arm", type=Path)
    parser.add_argument("--dog", type=Path)
    parser.add_argument("--native-runner", type=Path, default=NATIVE_RUNNER)
    parser.add_argument("--timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--expected-admission-sha256")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-warn-sha256")
    parser.add_argument("--expected-head", default=EXPECTED_STAGE38_HEAD)
    parser.add_argument("--api-head", default=EXPECTED_STAGE38_HEAD)
    parser.add_argument("--pbs-log", type=Path)
    args = parser.parse_args(argv)
    if args.admission:
        return run_admission(args)
    if args.campaign:
        for name in ("admission_root", "source_root", "arm", "dog"):
            if getattr(args, name) is None:
                parser.error("--campaign requires --%s" % name.replace("_", "-"))
        return run_campaign(args)
    return seal_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
