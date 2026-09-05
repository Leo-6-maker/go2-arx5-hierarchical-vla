#!/usr/bin/env python3
"""Stage33 dual-track bottleneck map.

This runner is deliberately split into standard-library-only qualification
paths and two explicitly GPU-bound paths:

* MA: paired stock-sampler exposure transfer on the immutable H2@1000 and
  H2@2000 checkpoints;
* MB: direct action-head nuisance-noise sensitivity on the frozen nine-probe
  grid;
* MC: CPU/static DROID projector-transfer audit, never model execution;
* WA: zero-simulation audit of the node-local native WBC support boundary.

The historical Stage31/Stage32 artifacts are read-only inputs.  Every new
scientific call is persisted independently before the next call is started.
No training, backward pass, optimizer/scheduler step, checkpoint write,
sampler sweep, WBC, hardware, validation-e, or holdout-f operation belongs in
this file.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import geometric_mean, mean, median
from typing import Any, Callable, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "STAGE33_DUAL_TRACK_BOTTLENECK_MAP_V1"
PI_GATE = "PI_GATE_STATE_0015"
NEXT_GATE = "GPT_PI_REVIEW_AFTER_STAGE33_BOTTLENECK_MAP"
PROJECT = "CFP03-CF-004"
BASE_COMMIT = "440dae61758c90211fb98239abcfac8778124cba"
HISTORICAL_TRAINING_SHA = "a361e5cb8a4be33f40a21c9bbf7ad742e25c7fd0"
TRAINING_RUNNER_SHA256 = "d61c429a2b8d53c25df63d9b5b9e2464b33c36e9e3c0a610b109b10a6bd63fe4"
GROOT_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"

BASE_INDEX_SHA256 = "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59"
BASE_TREE_SHA256 = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
BASE_TREE_PREFIX = "payload/GR00T-N1.7-3B"
TINY1_TREE_SHA256 = "541bf4a1b9ea459637525ef11fceea5c7f192c0902f6013f0331a2a668038157"
TINY1_STATS_SHA256 = "fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23"
TINY1_RELATIVE_STATS_SHA256 = "8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1"
MODALITY_SHA256 = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE_SHA256 = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"

STAGE26_MANIFEST_SHA256 = "eb33c81b68bb8bd49ce0ed9b0b818fdbb29546b49a2d79a3ee62845e172afb7b"
STAGE26_MANIFEST_ROWS = 18
STAGE26_PROBE_IDS = tuple(
    f"c1_seed{seed}_tq{quantile:02d}"
    for seed in (2223, 2224, 2225)
    for quantile in (10, 50, 90)
)

H2_1000_TREE_SHA256 = "29f3e8f46cf41441696bfc4a5e7601378648c26fa675a9685c003080da0c4114"
H2_1000_INDEX_SHA256 = "ff5a9ac57951658b775ef7e958a3469900f767cec60cbf487b3c8d0a3d787bc7"
H2_2000_TREE_SHA256 = "3106f9b5993b390438b62594d044781d11b393452a49f80b5c8e9e910606d0ba"
H2_2000_SEAL_SHA256 = "1349896f7b31a930764dbd9d849c5368e48372d0dc6273449f42f19f82a69a82"
STAGE31_RESULT_SHA256 = "c1d9cb9312cada95b35515fb44f87520a2133c7948785d19248613d6c3baf982"
STAGE32_RESULT_SHA256 = "719be43ea515c670db0bad5e5fb0ac01bd296936db97426089332120ee96d6d8"
STAGE32_SEAL_SHA256 = "e7768283006ac40dfafb22a578d19bccb5e0d2e081b60a2ec85231b6394faad2"
STAGE32_SALVAGE_RESULT_SHA256 = "08962dbd607bec236b2edf3a5a59a811691cb4b0b7e7cd7b6089c4140477adda"
STAGE32_SALVAGE_TRACE_SHA256 = "49fc754c5197b214b55b753640a2820d2427a38b4af8327ce3e098c12489769b"

HORIZON = 40
UNIVERSAL_DIM = 132
ACTIVE_DIM = 10
INACTIVE_DIM = UNIVERSAL_DIM - ACTIVE_DIM
ACTION_PERIOD_S = 0.05
SAMPLER_STEPS = 4
MA_SEEDS = tuple(range(2223, 2239))
MA_NEW_SEEDS = tuple(range(2226, 2239))
MA_CHECKPOINTS = ("H2@1000", "H2@2000")
MA_MAX_NEW_SAMPLER_CALLS = 26

MB_CHECKPOINTS = ("H2@1000", "H2@2000")
MB_CONDITIONS = ("A_STOCK", "B_RESAMPLE_1", "C_RESAMPLE_2", "D_RESAMPLE_3", "E_ZERO_INACTIVE")
MB_MAX_DIRECT_CALLS = 2 * 9 * 5
MB_MATERIAL_RATIO = 0.10
MB_MATERIAL_ZERO_IMPROVEMENTS = 8

EXPECTED_DROID_ID = 24
EXPECTED_ID10 = 10
MC_SOURCE_FILES = {
    "gr00t/data/embodiment_tags.py": "eadc1a395175209cc0d57afa21541cc64be7d848300602209de81fe42713ca47",
    "gr00t/configs/data/embodiment_configs.py": "4c7626426cfd184034df772bca9f1ab5c2014c139c388f833012ca98ac6f948a",
    "gr00t/model/gr00t_n1d7/gr00t_n1d7.py": "6bd66ef768f1484ac5a7889c40358a599f1862b1d554d47d357e113f2f96501c",
    "gr00t/model/modules/embodiment_conditioned_mlp.py": "d8c60775642016262e3df742e7ed10da38007d7a6bc6aa94fcb427f2ff9e1752",
    "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py": "77fd0e1f6972cdcf0334f18f2e84724629f6746612721b6b27135ccb23f00b3f",
    "examples/DROID/README.md": "e59c44bfb5a77c117abcf9ff83958589b730222b64796e42b0edce65f23fae16",
}
MC_SOURCE_ANCHORS = {
    "gr00t/data/embodiment_tags.py": (
        "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
        "oxe_droid_relative_eef_relative_joint",
    ),
    "gr00t/configs/data/embodiment_configs.py": (
        '"oxe_droid_relative_eef_relative_joint"',
        'modality_keys=["eef_9d", "gripper_position", "joint_position"]',
        "ActionRepresentation.RELATIVE",
        "ActionRepresentation.ABSOLUTE",
    ),
    "gr00t/model/modules/embodiment_conditioned_mlp.py": (
        "class CategorySpecificLinear",
        "class CategorySpecificMLP",
        "class MultiEmbodimentActionEncoder",
    ),
    "gr00t/model/gr00t_n1d7/gr00t_n1d7.py": (
        "self.action_encoder",
        "self.action_decoder",
        "embodiment_id",
    ),
    "examples/DROID/README.md": (
        "relative end-effector (XYZ + rotation 6D)",
        "gripper_position",
        "relative joint positions (7D)",
        "17D",
    ),
}

STAGE16_ARCHIVE_SHA256 = "b52a53bef87340e6244ead352fa434390faee65246cd32ab8e6648336a018167"
STAGE16_URDF_SHA256 = "96c1a81e3dd42af5f1bf87ac8958a0aebd7e798ce06408462e0e49a8a2067e25"
STAGE16_ARM_CHECKPOINT_SHA256 = "99a3a9bda30f7372c7c31240cc3584757ba46cb3b6b9881f00b371906d077a0a"
STAGE16_DOG_CHECKPOINT_SHA256 = "51776ddfd7177213444c0bf0c8f017b66ffd28a8674458aa0a16c3b9f77cc670"
STAGE16_NATIVE_CONFIG_SHA256 = "56132d4ff9b866ce196bbc1b1353900b967c54f0f48d2d4e4b82c3555769a0bb"
STAGE16_P1M_USD_SHA256 = "92d64f697d3f5ea6f17bbeaab72142607019f0309b5068c47f37a623499b9a53"

DEFAULT_BASE = Path("/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/GR00T-N1.7-3B")
DEFAULT_TINY1 = Path("/scratch/e1192548/stage23_o1_20260903/input_exact/extracted/tiny1_h2stats_lerobot_v2")
DEFAULT_GROOT = Path("/scratch/e1192548/stage22_h200_runtime_deb920_20260902/src/isaac-gr00t")
DEFAULT_MODALITY = Path("/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/modality/go2_arx5_modality.py")
DEFAULT_COSMOS_CACHE = Path("/scratch/e1192548/stage22_h200_cosmos_v2_20260902/hf_home/hub/models--nvidia--Cosmos-Reason2-2B")
DEFAULT_COSMOS_SNAPSHOT = DEFAULT_COSMOS_CACHE / "snapshots" / COSMOS_REVISION
DEFAULT_SNAPSHOT_ROOT = Path("/scratch/e1192548/stage26_h1_c1_20260904/snapshots")
DEFAULT_H2_TRAINING_ROOT = Path("/scratch/e1192548/stage28_h2_20260904/stage28_h2_600695.hopper-m-02")
DEFAULT_STAGE26_MANIFEST = Path("/scratch/e1192548/stage26_h1_c1_20260904/runs/600532.hopper-m-02/output/stage26_h1_c1_hash_manifest.json")
DEFAULT_STAGE31_ROOT = Path("/scratch/e1192548/leon-storage/runs/stage31_on_trajectory_20260905/stage31_601759.hopper-m-02")
DEFAULT_STAGE31_RESULT = DEFAULT_STAGE31_ROOT / "stage31_on_trajectory_flow_result.json"
DEFAULT_STAGE32_ROOT = Path("/scratch/e1192548/leon-storage/runs/stage32_h2_1000_20260905/stage32_602036.hopper-m-02")
DEFAULT_STAGE32_RESULT = DEFAULT_STAGE32_ROOT / "stage32_h2_1000_result.json"
DEFAULT_STAGE32_SALVAGE_RESULT = Path("/scratch/e1192548/leon-storage/runs/stage32_h2_1000_20260905/r1_seed2223_salvage/stage32_seed2223_salvage.json")
DEFAULT_STAGE32_SALVAGE_TRACE = Path("/scratch/e1192548/stage32_h2_1000_20260905/stage32_601973.hopper-m-02/traces/seed_2223_h2_1000_stock_sampler_trace.npz")
DEFAULT_STAGE16_ROOT = Path("/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage16")
DEFAULT_OUTPUT_PARENT = Path("/scratch/e1192548/leon-storage/runs/stage33_bottleneck_map_20260905")
LIVE_JOB_LOG_NAME = "stage33_job.log"


class Stage33Error(RuntimeError):
    """Fail-closed Stage33 contract error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    return str(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise Stage33Error(f"missing file: {path}") from exc
    if stat.S_IFMT(mode) != stat.S_IFREG:
        raise Stage33Error(f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise Stage33Error(f"missing or symlinked JSON: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise Stage33Error(f"invalid JSON: {path}") from exc


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git(repo: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *arguments], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Stage33Error(f"git query failed for {repo}: {arguments}") from exc


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Stage33Error(f"cannot import helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_stage28_helper() -> Any:
    return _load_module(
        REPO_ROOT / "scripts" / "stage28_h2_fresh_base_2000_fixed_warmup10.py",
        "stage28_helper_for_stage33",
    )


def _load_stage26_helper() -> Any:
    return _load_module(
        REPO_ROOT / "scripts" / "stage26_h1_c1_fixed_noise_local_flow.py",
        "stage26_helper_for_stage33",
    )


def _load_stage29_helper() -> Any:
    return _load_module(
        REPO_ROOT / "scripts" / "stage29_c2_free_running_transfer.py",
        "stage29_helper_for_stage33",
    )


def _load_stage31_helper() -> Any:
    return _load_module(
        REPO_ROOT / "scripts" / "stage31_on_trajectory_flow_field_audit.py",
        "stage31_helper_for_stage33",
    )


def _regular_entries(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise Stage33Error(f"unsafe or missing tree root: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in sorted(dirnames):
            child = base / name
            mode = child.lstat().st_mode
            if stat.S_IFMT(mode) != stat.S_IFDIR:
                raise Stage33Error(f"tree contains non-directory entry: {child}")
        for name in sorted(filenames):
            child = base / name
            mode = child.lstat().st_mode
            if stat.S_IFMT(mode) != stat.S_IFREG:
                raise Stage33Error(f"tree contains non-regular file: {child}")
            entries.append(
                {
                    "path": child.relative_to(root).as_posix(),
                    "bytes": int(child.stat().st_size),
                    "sha256": _sha256_file(child),
                }
            )
    return sorted(entries, key=lambda row: str(row["path"]))


def _tree_digest(entries: Iterable[Mapping[str, Any]], prefix: str) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda row: str(row["path"])):
        digest.update(
            f"{prefix}/{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _ensure_external_output(output_root: Path, immutable_roots: Iterable[Path]) -> None:
    output = output_root.resolve()
    for immutable in immutable_roots:
        source = immutable.resolve()
        if output == source or output in source.parents or source in output.parents:
            raise Stage33Error(f"output overlaps immutable input: {output} vs {source}")
    output.mkdir(parents=True, exist_ok=True)


def _array_sha256(array: Any) -> str:
    value = array.detach().float().contiguous().cpu().numpy() if hasattr(array, "detach") else array
    import numpy as np

    return _sha256_bytes(np.ascontiguousarray(value).tobytes())


def _tensor_record(value: Any) -> dict[str, Any]:
    import torch

    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "device": str(value.device),
        "raw_sha256": _sha256_bytes(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()),
        "float32_sha256": _array_sha256(value),
        "finite": bool(torch.isfinite(value).all().item()),
    }


def _update_state(path: Path, fields: Mapping[str, Any]) -> dict[str, Any]:
    current = _read_json(path) if path.exists() else {}
    merged = dict(current)
    merged.update(_jsonable(fields))
    for key in ("sampler_calls_started", "sampler_calls_completed", "direct_calls_started", "direct_calls_completed"):
        old = int(current.get(key, 0))
        new = int(fields.get(key, old))
        if new < old:
            raise Stage33Error(f"execution counter regressed: {key} {old}->{new}")
        merged[key] = max(old, new)
    if current.get("scientific_execution_begun") is True:
        merged["scientific_execution_begun"] = True
    _write_json_atomic(path, merged)
    return merged


def _validate_hex(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text.lower()):
        raise Stage33Error(f"{label} is not a 64-hex SHA256: {value!r}")
    return text.lower()


def _verify_stage26_manifest(path: Path) -> dict[str, Any]:
    observed = _sha256_file(path)
    if observed != STAGE26_MANIFEST_SHA256:
        raise Stage33Error(f"Stage26 manifest SHA mismatch: {observed}")
    payload = _read_json(path)
    decoded = payload.get("decoded_action_hashes")
    raw = payload.get("raw_velocity_hashes")
    if not isinstance(decoded, list) or not isinstance(raw, list):
        raise Stage33Error("Stage26 manifest arrays are missing")
    # The manifest contains two parallel hash tables.  Each table has the
    # same 18 probe rows (nine BASE and nine O1@200); the scientific row
    # count is therefore 18 per table, not 18 after summing both tables.
    if len(decoded) != STAGE26_MANIFEST_ROWS or len(raw) != STAGE26_MANIFEST_ROWS:
        raise Stage33Error("Stage26 manifest hash-table row count is not 18")
    if any(not isinstance(row, Mapping) for row in (*decoded, *raw)):
        raise Stage33Error("Stage26 manifest contains a non-object row")
    decoded_keys = {(str(row.get("checkpoint")), str(row.get("probe_id"))) for row in decoded}
    raw_keys = {(str(row.get("checkpoint")), str(row.get("probe_id"))) for row in raw}
    expected_base = {("BASE", probe) for probe in STAGE26_PROBE_IDS}
    expected_o1 = {("O1@200", probe) for probe in STAGE26_PROBE_IDS}
    if decoded_keys != expected_base | expected_o1 or raw_keys != expected_base | expected_o1:
        raise Stage33Error("Stage26 manifest checkpoint/probe key set drifted")
    if sum(row[0] == "BASE" for row in decoded_keys) != 9 or sum(row[0] == "BASE" for row in raw_keys) != 9:
        raise Stage33Error("Stage26 manifest does not contain exactly nine BASE rows per hash table")
    for row in decoded:
        _validate_hex(row.get("decoded_physical_action_sha256"), "decoded physical action hash")
    for row in raw:
        _validate_hex(row.get("raw_predicted_velocity_sha256"), "raw velocity hash")
    return {
        "path": str(path),
        "sha256": observed,
        "total_rows": STAGE26_MANIFEST_ROWS,
        "decoded_rows": len(decoded),
        "raw_rows": len(raw),
        "base_rows": 9,
        "probe_ids": list(STAGE26_PROBE_IDS),
        "malformed_duplicate_missing_extra_cases_checked": True,
    }


def _verify_historical_records(args: argparse.Namespace) -> dict[str, Any]:
    stage31_sha = _sha256_file(args.stage31_result)
    stage32_sha = _sha256_file(args.stage32_result)
    if stage31_sha != STAGE31_RESULT_SHA256:
        raise Stage33Error(f"Stage31 result SHA mismatch: {stage31_sha}")
    if stage32_sha != STAGE32_RESULT_SHA256:
        raise Stage33Error(f"Stage32 result SHA mismatch: {stage32_sha}")
    if _sha256_file(args.stage32_result_seal) != STAGE32_SEAL_SHA256:
        raise Stage33Error("Stage32 result seal SHA mismatch")
    if _sha256_file(args.stage32_salvage_result) != STAGE32_SALVAGE_RESULT_SHA256:
        raise Stage33Error("Stage32 salvage result SHA mismatch")
    if _sha256_file(args.stage32_salvage_trace) != STAGE32_SALVAGE_TRACE_SHA256:
        raise Stage33Error("Stage32 salvage trace SHA mismatch")

    salvage = _read_json(args.stage32_salvage_result)
    salvage_source = salvage.get("source", {})
    salvage_trace_path = Path(str(salvage_source.get("partial_trace_path", "")))
    if salvage.get("status") != "PASS_R1_ZERO_MODEL_SALVAGE":
        raise Stage33Error("Stage32 salvage result is not the accepted zero-model salvage")
    if salvage.get("historical_files_modified") is not False or salvage.get("scientific_contract_changed") is not False:
        raise Stage33Error("Stage32 salvage result reports historical or contract mutation")
    if salvage_trace_path.resolve() != args.stage32_salvage_trace.resolve():
        raise Stage33Error("Stage32 salvage trace path drifted")
    if salvage_source.get("partial_trace_sha256") != STAGE32_SALVAGE_TRACE_SHA256:
        raise Stage33Error("Stage32 salvage result does not bind the frozen trace SHA")

    stage31 = _read_json(args.stage31_result)
    stage32 = _read_json(args.stage32_result)
    stage31_records = stage31.get("records")
    stage32_records = stage32.get("sampler_records")
    if stage31.get("status") != "PASS_EXECUTION_COMPLETE" or not isinstance(stage31_records, list):
        raise Stage33Error("Stage31 historical result is not complete")
    if stage32.get("status") != "PASS_EXECUTION_COMPLETE" or not isinstance(stage32_records, list):
        raise Stage33Error("Stage32 historical result is not complete")
    h2_2000 = [row for row in stage31_records if row.get("checkpoint") == "H2@2000"]
    h2_1000 = [row for row in stage32_records if row.get("checkpoint") == "H2@1000"]
    if len(h2_2000) != 3 or {int(row.get("seed")) for row in h2_2000} != {2223, 2224, 2225}:
        raise Stage33Error("Stage31 H2@2000 historical seed set drifted")
    if len(h2_1000) != 3 or {int(row.get("seed")) for row in h2_1000} != {2223, 2224, 2225}:
        raise Stage33Error("Stage32 H2@1000 historical seed set drifted")
    trace_rows: list[dict[str, Any]] = []
    for row in h2_2000:
        artifact = row.get("trace_artifact", {})
        path = Path(str(artifact.get("path", "")))
        if path.resolve().parent != args.stage31_root.resolve() / "traces":
            raise Stage33Error(f"Stage31 trace path drifted: {path}")
        if _sha256_file(path) != artifact.get("sha256"):
            raise Stage33Error(f"historical trace SHA mismatch: {path}")
        trace_rows.append(
            {
                "checkpoint": row["checkpoint"],
                "seed": int(row["seed"]),
                "path": str(path),
                "sha256": artifact["sha256"],
                "source_result": "Stage31" if row in h2_2000 else "Stage32",
            }
        )
    for row in h2_1000:
        artifact = row.get("trace_artifact", {})
        path = Path(str(artifact.get("path", "")))
        if int(row["seed"]) == 2223:
            if path.resolve() != args.stage32_salvage_trace.resolve():
                raise Stage33Error(f"Stage32 salvaged trace path drifted: {path}")
        elif path.resolve().parent != args.stage32_root.resolve() / "traces":
            raise Stage33Error(f"Stage32 continuation trace path drifted: {path}")
        if _sha256_file(path) != artifact.get("sha256"):
            raise Stage33Error(f"historical trace SHA mismatch: {path}")
        trace_rows.append(
            {
                "checkpoint": row["checkpoint"],
                "seed": int(row["seed"]),
                "path": str(path),
                "sha256": artifact["sha256"],
                "source_result": "Stage32",
            }
        )
    return {
        "stage31_result": {"path": str(args.stage31_result), "sha256": stage31_sha, "rows": len(stage31_records)},
        "stage32_result": {"path": str(args.stage32_result), "sha256": stage32_sha, "rows": len(stage32_records)},
        "stage32_result_seal_sha256": STAGE32_SEAL_SHA256,
        "stage32_salvage_result_sha256": STAGE32_SALVAGE_RESULT_SHA256,
        "stage32_salvage_trace_sha256": STAGE32_SALVAGE_TRACE_SHA256,
        "historical_sampler_records": trace_rows,
        "historical_artifacts_modified": False,
        "stage32_rerun": False,
    }


def _verify_groot_source(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise Stage33Error(f"GR00T source root unavailable: {path}")
    head = _git(path, "rev-parse", "HEAD")
    if head != GROOT_SHA:
        raise Stage33Error(f"GR00T HEAD mismatch: {head}")
    if _git(path, "status", "--porcelain=v1"):
        raise Stage33Error("GR00T source worktree is dirty")
    return {"path": str(path), "head": head, "clean": True}


def _verify_base_tiny_h2(args: argparse.Namespace) -> dict[str, Any]:
    if args.h2_training_root.resolve() != DEFAULT_H2_TRAINING_ROOT.resolve():
        raise Stage33Error("H2 training root is not the historical job-600695 path")
    stage28 = _load_stage28_helper()
    base = stage28._verify_base_checkpoint(args.base_model_path)
    if base.get("tree_sha256") != BASE_TREE_SHA256 or base.get("index", {}).get("index_sha256") != BASE_INDEX_SHA256:
        raise Stage33Error("BASE identity did not match the frozen Stage22 tree/index")
    tiny = stage28._verify_tiny_dataset(args.dataset_dir)
    if tiny.get("tree_sha256") != TINY1_TREE_SHA256:
        raise Stage33Error("TINY1 tree identity mismatch")
    if tiny.get("stats_json_sha256") != TINY1_STATS_SHA256 or tiny.get("relative_stats_json_sha256") != TINY1_RELATIVE_STATS_SHA256:
        raise Stage33Error("TINY1 statistics identity mismatch")
    modality_sha = _sha256_file(args.modality_config)
    if modality_sha != MODALITY_SHA256:
        raise Stage33Error(f"modality SHA mismatch: {modality_sha}")
    cosmos = stage28._verify_cosmos_cache(args.cosmos_cache_root)
    if cosmos.get("revision") != COSMOS_REVISION or cosmos.get("tree_sha256") != COSMOS_TREE_SHA256:
        raise Stage33Error("Cosmos identity mismatch")
    if args.cosmos_snapshot.resolve() != Path(str(cosmos["snapshot"])).resolve():
        raise Stage33Error("Cosmos snapshot is not the verified revision snapshot")
    h2 = stage28._verify_final_checkpoint_artifacts(
        args.h2_training_root,
        expected_checkpoint_tree_sha256=H2_2000_TREE_SHA256,
        expected_seal_sha256=H2_2000_SEAL_SHA256,
    )
    if h2["checkpoint-1000"]["tree_sha256"] != H2_1000_TREE_SHA256:
        raise Stage33Error("H2@1000 tree identity mismatch")
    if h2["checkpoint-1000"]["index"].get("index_sha256") != H2_1000_INDEX_SHA256:
        raise Stage33Error("H2@1000 index identity mismatch")
    if h2["checkpoint-2000"]["tree_sha256"] != H2_2000_TREE_SHA256:
        raise Stage33Error("H2@2000 tree identity mismatch")
    return {
        "base": base,
        "tiny1": tiny,
        "modality_sha256": modality_sha,
        "cosmos": cosmos,
        "h2_training": h2,
        "h2_checkpoint_1000": str(args.h2_training_root / "h2_training" / "checkpoint-1000"),
        "h2_checkpoint_2000": str(args.h2_training_root / "h2_training" / "checkpoint-2000"),
        "historical_training_source_sha": HISTORICAL_TRAINING_SHA,
        "historical_training_runner_sha256": TRAINING_RUNNER_SHA256,
    }


def protocol_plan() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "pi_gate": PI_GATE,
        "stage": "Stage33 dual-track bottleneck map",
        "base_commit": BASE_COMMIT,
        "ma": {
            "question": "paired H2@1000 to H2@2000 exposure transfer across sampler initial noise",
            "seeds": list(MA_SEEDS),
            "new_seeds": list(MA_NEW_SEEDS),
            "checkpoints": list(MA_CHECKPOINTS),
            "sampler": "official stock N1.7 four-step Euler sampler",
            "horizon": HORIZON,
            "universal_dim": UNIVERSAL_DIM,
            "active_dim": ACTIVE_DIM,
            "maximum_new_sampler_calls": MA_MAX_NEW_SAMPLER_CALLS,
            "base_sampler_calls": 0,
            "training": False,
            "paired_initial_noise": True,
            "classification": {
                "positive_if_h2_lower_count_at_least": 12,
                "and_geometric_mean_paired_ratio_less_than": 1.0,
                "otherwise": "EXPOSURE_ROBUST_MIXED",
            },
        },
        "mb": {
            "question": "inactive122 perturbation of active10 direct-flow velocity",
            "checkpoints": list(MB_CHECKPOINTS),
            "probe_ids": list(STAGE26_PROBE_IDS),
            "conditions": list(MB_CONDITIONS),
            "maximum_direct_calls": MB_MAX_DIRECT_CALLS,
            "inactive_slice": [ACTIVE_DIM, UNIVERSAL_DIM],
            "active_slice": [0, ACTIVE_DIM],
            "intervention": "deterministic CPU-fp32 inactive resamples; E zeros inactive only",
            "material_coupling": {
                "median_ratio_at_least": MB_MATERIAL_RATIO,
                "h2_zero_inactive_improvements_at_least": MB_MATERIAL_ZERO_IMPROVEMENTS,
            },
            "training": False,
            "official_sampler": False,
        },
        "mc": {
            "question": "DROID relative-EEF projector transfer semantics and shapes",
            "source_only": True,
            "model_constructed": False,
            "forward": False,
            "training": False,
            "droid_category_id": EXPECTED_DROID_ID,
            "id10_category_id": EXPECTED_ID10,
            "droid_semantic_prefix": ["eef_9d[0:9]", "gripper_position[9:10]"],
            "droid_incompatible_joint_slice": [10, 17],
        },
        "wa": {
            "question": "support-aware continuous-base/timing envelope of frozen UMI/native WBC",
            "zero_sim_audit_first": True,
            "maximum_new_rollouts": 24,
            "gr00t": False,
            "wbc_training": False,
            "hardware": False,
        },
        "execution_boundary": {
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "scheduler_step": False,
            "resume": False,
            "checkpoint_write": False,
            "sampler_sweep": False,
            "rtc": False,
            "guidance": False,
            "clipping": False,
            "smoothing": False,
            "retiming": False,
            "validation_e": False,
            "holdout_f": False,
            "wbc": False,
            "hardware": False,
        },
        "next_gate": NEXT_GATE,
    }


def assert_protocol_static(plan: Mapping[str, Any] | None = None) -> None:
    candidate = protocol_plan() if plan is None else dict(plan)
    if candidate.get("schema") != SCHEMA or candidate.get("pi_gate") != PI_GATE:
        raise Stage33Error("Stage33 schema or PI gate drifted")
    ma = candidate.get("ma", {})
    if ma.get("seeds") != list(MA_SEEDS) or ma.get("new_seeds") != list(MA_NEW_SEEDS):
        raise Stage33Error("Stage33-MA seed set drifted")
    if ma.get("checkpoints") != list(MA_CHECKPOINTS) or ma.get("maximum_new_sampler_calls") != MA_MAX_NEW_SAMPLER_CALLS:
        raise Stage33Error("Stage33-MA checkpoint or budget drifted")
    mb = candidate.get("mb", {})
    if mb.get("checkpoints") != list(MB_CHECKPOINTS) or mb.get("conditions") != list(MB_CONDITIONS):
        raise Stage33Error("Stage33-MB condition contract drifted")
    if mb.get("probe_ids") != list(STAGE26_PROBE_IDS) or mb.get("maximum_direct_calls") != MB_MAX_DIRECT_CALLS:
        raise Stage33Error("Stage33-MB probe or budget drifted")
    boundary = candidate.get("execution_boundary", {})
    if any(boundary.get(key) is not False for key in boundary):
        raise Stage33Error("Stage33 execution boundary contains an enabled forbidden operation")


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.authorization_token != PI_GATE:
        raise Stage33Error("PI gate token mismatch")
    if args.pbs_project != PROJECT:
        raise Stage33Error("PBS project mismatch")
    assert_protocol_static()
    project_head = _git(args.project_repo, "rev-parse", "HEAD")
    if project_head != args.expected_project_sha:
        raise Stage33Error(f"project HEAD mismatch: {project_head} != {args.expected_project_sha}")
    if _git(args.project_repo, "status", "--porcelain=v1"):
        raise Stage33Error("Stage33 project worktree is dirty")
    ancestor = subprocess.run(
        ["git", "-C", str(args.project_repo), "merge-base", "--is-ancestor", BASE_COMMIT, project_head],
        check=False,
    )
    if ancestor.returncode != 0:
        raise Stage33Error("Stage33 project HEAD does not descend from accepted Stage32 HEAD")
    pinned_paths = {
        "BASE": (args.base_model_path, DEFAULT_BASE),
        "TINY1": (args.dataset_dir, DEFAULT_TINY1),
        "GR00T": (args.groot_source, DEFAULT_GROOT),
        "modality": (args.modality_config, DEFAULT_MODALITY),
        "Cosmos cache": (args.cosmos_cache_root, DEFAULT_COSMOS_CACHE),
        "Cosmos snapshot": (args.cosmos_snapshot, DEFAULT_COSMOS_SNAPSHOT),
        "Stage22 snapshot root": (args.snapshot_root, DEFAULT_SNAPSHOT_ROOT),
        "H2 training root": (args.h2_training_root, DEFAULT_H2_TRAINING_ROOT),
        "Stage26 manifest": (args.stage26_manifest, DEFAULT_STAGE26_MANIFEST),
        "Stage31 root": (args.stage31_root, DEFAULT_STAGE31_ROOT),
        "Stage31 result": (args.stage31_result, DEFAULT_STAGE31_RESULT),
        "Stage32 root": (args.stage32_root, DEFAULT_STAGE32_ROOT),
        "Stage32 result": (args.stage32_result, DEFAULT_STAGE32_RESULT),
        "Stage32 result seal": (args.stage32_result_seal, DEFAULT_STAGE32_ROOT / "stage32_result_seal.json"),
        "Stage32 salvage result": (args.stage32_salvage_result, DEFAULT_STAGE32_SALVAGE_RESULT),
        "Stage32 salvage trace": (args.stage32_salvage_trace, DEFAULT_STAGE32_SALVAGE_TRACE),
        "Stage16 root": (args.stage16_root, DEFAULT_STAGE16_ROOT),
    }
    for label, (observed, expected) in pinned_paths.items():
        if observed.resolve() != expected.resolve():
            raise Stage33Error(f"{label} path drifted: {observed} != {expected}")
    _ensure_external_output(
        args.output_root,
        (
            args.base_model_path,
            args.dataset_dir,
            args.h2_training_root,
            args.stage31_root,
            args.stage32_root,
            args.stage16_root,
            args.cosmos_cache_root,
        ),
    )
    identities = _verify_base_tiny_h2(args)
    identities["stage26_manifest"] = _verify_stage26_manifest(args.stage26_manifest)
    identities["historical"] = _verify_historical_records(args)
    identities["groot"] = _verify_groot_source(args.groot_source)
    snapshot_helper = _load_stage26_helper()
    snapshot_identity, snapshot_path = snapshot_helper._verify_snapshot_identity(args.snapshot_root)
    if snapshot_path.resolve() != (args.snapshot_root / "stage22_expert_red_target_train_a" / "expert-0000.json").resolve():
        raise Stage33Error("selected Stage22 snapshot path drifted")
    identities["snapshot"] = snapshot_identity
    identities["stage26_manifest_path"] = str(args.stage26_manifest)
    identities["stage28_historical_job"] = "600695.hopper-m-02"
    identities["stage28_historical_job_rerun"] = False
    return {
        "schema": SCHEMA,
        "status": "PASS_PREFLIGHT_NO_MODEL",
        "timestamp_utc": _utc_now(),
        "pi_gate": PI_GATE,
        "project": {
            "head": project_head,
            "expected_head": args.expected_project_sha,
            "clean": True,
            "stage32_ancestor": BASE_COMMIT,
        },
        "identities": identities,
        "execution": {
            "model_constructed": False,
            "model_forward": False,
            "sampler_calls": 0,
            "direct_flow_calls": 0,
            "training": False,
            "backward": False,
            "checkpoint_write": False,
            "wbc": False,
            "hardware": False,
            "validation_e": False,
            "holdout_f": False,
        },
        "scientific_contract_changed": False,
        "historical_paths_unchanged": True,
    }


def _historical_trace_row(
    *,
    row: Mapping[str, Any],
    target: Any,
    expert: Any,
    checkpoint_path: Path,
    label: str,
    stage26: Any,
    stage29: Any,
    stage31: Any,
    processor: Any,
    embodiment: Any,
    snapshot: Any,
) -> dict[str, Any]:
    import numpy as np

    artifact = row.get("trace_artifact", {})
    trace_path = Path(str(artifact.get("path", "")))
    observed_sha = _sha256_file(trace_path)
    if observed_sha != artifact.get("sha256"):
        raise Stage33Error(f"historical trace artifact changed: {trace_path}")
    with np.load(trace_path, allow_pickle=False) as payload:
        x = np.asarray(payload["x_trajectory_float32"], dtype=np.float32)
        v = np.asarray(payload["v_trajectory_float32"], dtype=np.float32)
        timestep_ids = np.asarray(payload["timestep_ids"], dtype=np.int64)
    trace_metrics = stage31.trace_metrics(x, v, target)
    raw = np.ascontiguousarray(x[4, 0][None, ...], dtype=np.float32)
    if raw.shape != (1, HORIZON, UNIVERSAL_DIM) or not np.isfinite(raw).all():
        raise Stage33Error(f"historical final sampler state invalid: {trace_path}")
    decoded = stage26._decode_physical_action(processor, embodiment, raw, snapshot)
    physical = stage29._trajectory_metrics(
        decoded, expert, stage29_profile_module(), stage26, processor, embodiment, snapshot, checkpoint_path, label
    )
    return _ma_record(
        seed=int(row["seed"]),
        checkpoint=label,
        checkpoint_path=checkpoint_path,
        physical=physical,
        decoded=decoded,
        expert=expert,
        trace_metrics=trace_metrics,
        raw=raw,
        initial_noise=row.get("initial_noise", {}),
        conditioning_sha256=str(row.get("conditioning_sha256", "HISTORICAL_CONDITIONING_BOUND_BY_STAGE31_STAGE32")),
        artifact={"path": str(trace_path), "bytes": trace_path.stat().st_size, "sha256": observed_sha},
        new_sampler_call=False,
        provenance=str(row.get("provenance", "HISTORICAL_REUSED")),
        historical_source="Stage32" if label == "H2@1000" else "Stage31",
        timestep_ids=timestep_ids.tolist(),
    )


def stage29_profile_module() -> Any:
    # Stage29's metric helper expects Phase-B's provider profile function.  It
    # is loaded lazily so plan/preflight/MC/WA remain model-package-free.
    return _load_module(
        REPO_ROOT / "scripts" / "stage23_phase_b_attribution_audit.py",
        "stage23_profile_for_stage33",
    )


def _ma_record(
    *,
    seed: int,
    checkpoint: str,
    checkpoint_path: Path,
    physical: Mapping[str, Any],
    decoded: Any,
    expert: Any,
    trace_metrics: Mapping[str, Any],
    raw: Any,
    initial_noise: Mapping[str, Any],
    conditioning_sha256: str,
    artifact: Mapping[str, Any],
    new_sampler_call: bool,
    provenance: str,
    historical_source: str | None = None,
    timestep_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    import numpy as np

    pred = np.asarray(decoded, dtype=np.float32)
    target = np.asarray(expert, dtype=np.float32)
    delta = pred - target
    profile = physical["predicted_profile"]
    final_scope = trace_metrics.get("final_x4", {}).get("scopes", {})
    raw_array = np.asarray(raw, dtype=np.float32)
    if raw_array.shape != (1, HORIZON, UNIVERSAL_DIM) or not np.isfinite(raw_array).all():
        raise Stage33Error("MA raw universal action shape/finite contract failed")
    required_profile = (
        "path_length_m", "linear_speed_mps", "second_difference_m",
        "adjacent_direction_cosine", "back_and_forth_rate", "first_difference_dft",
    )
    if any(key not in profile for key in required_profile):
        raise Stage33Error("MA physical profile is missing a mandatory metric")
    return {
        "schema": "STAGE33_MA_SAMPLER_RECORD_V1",
        "seed": int(seed),
        "checkpoint": checkpoint,
        "checkpoint_path": str(checkpoint_path),
        "new_sampler_call": bool(new_sampler_call),
        "provenance": provenance,
        "historical_source": historical_source,
        "conditioning_sha256": conditioning_sha256,
        "initial_noise": dict(initial_noise),
        "raw_universal_action": {
            "shape": list(raw_array.shape),
            "dtype": str(raw_array.dtype),
            "sha256": _array_sha256(raw_array),
            "finite": bool(np.isfinite(raw_array).all()),
        },
        "decoded_physical_action": {
            "shape": list(pred.shape),
            "dtype": str(pred.dtype),
            "sha256": _array_sha256(pred),
            "finite": bool(np.isfinite(pred).all()),
        },
        "artifact": dict(artifact),
        "trace_timestep_ids": list(timestep_ids or []),
        "trace_metrics": dict(trace_metrics),
        "physical_metrics": dict(physical),
        "mandatory_metrics": {
            "final_active10_mse": float(physical["physical_action_mse_active_10d"]),
            "xyz_mse": float(physical["eef_position_mse"]),
            "x_mse": float(np.mean(np.square(delta[..., 0].astype(np.float64)))),
            "y_mse": float(np.mean(np.square(delta[..., 1].astype(np.float64)))),
            "z_mse": float(np.mean(np.square(delta[..., 2].astype(np.float64)))),
            "endpoint_error_m": float(physical["endpoint_position_error_m"]),
            "path_length_m": float(profile["path_length_m"]),
            "mean_speed_mps": float(profile["linear_speed_mps"]["mean"]),
            "max_speed_mps": float(profile["linear_speed_mps"]["max"]),
            "roughness_mean_m": float(profile["second_difference_m"]["mean"]),
            "adjacent_cosine_median": float(profile["adjacent_direction_cosine"]["median"]),
            "back_and_forth_rate": float(profile["back_and_forth_rate"]),
            "high_frequency_fraction": float(profile["first_difference_dft"]["high_fraction"]),
            "raw_support_occupancy_active10_abs_gt_1": float(final_scope.get("active10", {}).get("raw_fraction_abs_gt_1", np.mean(np.abs(raw_array[..., :ACTIVE_DIM]) > 1.0))),
            "raw_support_occupancy_universal_abs_gt_1": float(np.mean(np.abs(raw_array) > 1.0)),
            "safety_gate": physical["safety_gate"],
            "all_finite": bool(physical["all_finite"] and np.isfinite(raw_array).all()),
        },
        "execution": {
            "sampler": "official_stock_n1d7_four_step_euler",
            "rtc": False,
            "guidance": False,
            "clipping": False,
            "smoothing": False,
            "retiming": False,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "scheduler_step": False,
            "resume": False,
            "checkpoint_write": False,
            "wbc": False,
            "hardware": False,
        },
    }


def _load_historical_ma_records(args: argparse.Namespace, processor: Any, embodiment: Any, snapshot: Any, prepared: Mapping[str, Any], stage26: Any, stage29: Any, stage31: Any) -> list[dict[str, Any]]:
    import numpy as np

    profile_module = stage29_profile_module()
    target = prepared["expert_action_cpu"].numpy().astype(np.float32, copy=False)
    expert = stage26._decode_physical_action(processor, embodiment, target, snapshot)
    stage31_result = _read_json(args.stage31_result)
    stage32_result = _read_json(args.stage32_result)
    h2_2000 = {int(row["seed"]): row for row in stage31_result["records"] if row.get("checkpoint") == "H2@2000"}
    h2_1000 = {int(row["seed"]): row for row in stage32_result["sampler_records"] if row.get("checkpoint") == "H2@1000"}
    rows: list[dict[str, Any]] = []
    for seed in MA_SEEDS[:3]:
        rows.append(_historical_trace_row(row=h2_1000[seed], target=target, expert=expert, checkpoint_path=args.h2_checkpoint_1000, label="H2@1000", stage26=stage26, stage29=stage29, stage31=stage31, processor=processor, embodiment=embodiment, snapshot=snapshot))
        rows.append(_historical_trace_row(row=h2_2000[seed], target=target, expert=expert, checkpoint_path=args.h2_checkpoint_2000, label="H2@2000", stage26=stage26, stage29=stage29, stage31=stage31, processor=processor, embodiment=embodiment, snapshot=snapshot))
    # Keep one deterministic seed-major ordering for historical and new rows.
    return sorted(rows, key=lambda row: (int(row["seed"]), MA_CHECKPOINTS.index(str(row["checkpoint"]))))


def _write_ma_partial(output_root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    _write_json_atomic(
        output_root / "stage33_ma_partial_records.json",
        {
            "schema": "STAGE33_MA_SAMPLER_RECORD_V1",
            "status": "PARTIAL_EVIDENCE",
            "record_count": len(records),
            "records": list(records),
        },
    )


def _ma_aggregates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lookup = {(int(row["seed"]), str(row["checkpoint"])): row for row in records}
    expected = {(seed, checkpoint) for seed in MA_SEEDS for checkpoint in MA_CHECKPOINTS}
    if set(lookup) != expected or len(records) != len(expected):
        raise Stage33Error("MA record key set/count drifted")
    ratios: list[float] = []
    per_seed: list[dict[str, Any]] = []
    improved = 0
    for seed in MA_SEEDS:
        one = lookup[(seed, "H2@1000")]["mandatory_metrics"]
        two = lookup[(seed, "H2@2000")]["mandatory_metrics"]
        first = float(one["final_active10_mse"])
        second = float(two["final_active10_mse"])
        if not (math.isfinite(first) and math.isfinite(second) and first > 0 and second > 0):
            raise Stage33Error(f"MA non-positive/non-finite MSE for seed {seed}")
        ratio = second / first
        ratios.append(ratio)
        is_better = second < first
        improved += int(is_better)
        per_seed.append(
            {
                "seed": seed,
                "h2_1000_mse": first,
                "h2_2000_mse": second,
                "paired_ratio_h2_2000_over_h2_1000": ratio,
                "h2_2000_lower": is_better,
                "initial_noise_raw_sha256_h2_1000": lookup[(seed, "H2@1000")]["initial_noise"].get("raw_sha256"),
                "initial_noise_raw_sha256_h2_2000": lookup[(seed, "H2@2000")]["initial_noise"].get("raw_sha256"),
                "initial_noise_exact": lookup[(seed, "H2@1000")]["initial_noise"].get("raw_sha256") == lookup[(seed, "H2@2000")]["initial_noise"].get("raw_sha256"),
                "new_sampler_calls": int(lookup[(seed, "H2@1000")]["new_sampler_call"]) + int(lookup[(seed, "H2@2000")]["new_sampler_call"]),
            }
        )
    exact_noise = all(item["initial_noise_exact"] for item in per_seed)
    if not exact_noise:
        raise Stage33Error("MA paired initial-noise equality failed")
    geometric = float(geometric_mean(ratios))
    classification = "EXPOSURE_ROBUST_POSITIVE" if improved >= 12 and geometric < 1.0 else "EXPOSURE_ROBUST_MIXED"
    metric_names = (
        "final_active10_mse", "xyz_mse", "x_mse", "y_mse", "z_mse", "endpoint_error_m",
        "path_length_m", "mean_speed_mps", "max_speed_mps", "roughness_mean_m",
        "adjacent_cosine_median", "back_and_forth_rate", "high_frequency_fraction",
    )
    per_metric: dict[str, Any] = {}
    for name in metric_names:
        values_1000 = [float(lookup[(seed, "H2@1000")]["mandatory_metrics"][name]) for seed in MA_SEEDS]
        values_2000 = [float(lookup[(seed, "H2@2000")]["mandatory_metrics"][name]) for seed in MA_SEEDS]
        per_metric[name] = {
            "h2_1000": {"count": len(values_1000), "mean": mean(values_1000), "median": median(values_1000)},
            "h2_2000": {"count": len(values_2000), "mean": mean(values_2000), "median": median(values_2000)},
        }
    new_calls = sum(int(row["new_sampler_call"]) for row in records)
    if new_calls != MA_MAX_NEW_SAMPLER_CALLS:
        raise Stage33Error(f"MA new sampler call count is {new_calls}, not {MA_MAX_NEW_SAMPLER_CALLS}")
    return {
        "classification": classification,
        "exposure_robust_positive": classification == "EXPOSURE_ROBUST_POSITIVE",
        "total_paired_seeds": len(MA_SEEDS),
        "h2_2000_lower_count": improved,
        "geometric_mean_paired_ratio": geometric,
        "median_paired_ratio": median(ratios),
        "per_seed": per_seed,
        "per_metric": per_metric,
        "paired_initial_noise": {"all_exact": True, "pair_count": len(per_seed)},
        "new_sampler_calls": new_calls,
        "base_sampler_calls": 0,
        "maximum_new_sampler_calls": MA_MAX_NEW_SAMPLER_CALLS,
        "descriptive_only": True,
        "scientific_interpretation": "DEFERRED_TO_GPT_PI",
    }


def run_ma(args: argparse.Namespace) -> dict[str, Any]:
    preflight = _read_json(args.output_root / "stage33_preflight.json")
    if preflight.get("status") != "PASS_PREFLIGHT_NO_MODEL":
        raise Stage33Error("MA requires a passing Stage33 preflight")
    state_path = args.output_root / "stage33_ma_execution_state.json"
    partial_path = args.output_root / "stage33_ma_partial_records.json"
    if state_path.exists() or partial_path.exists() or (args.output_root / "stage33_ma_result.json").exists():
        raise Stage33Error("MA output root is not fresh; resume/replacement is forbidden")
    import numpy as np
    import torch

    if args.device != "cuda:0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Stage33Error("MA requires exactly one visible CUDA device at cuda:0")
    if "H200" not in torch.cuda.get_device_name(0):
        raise Stage33Error(f"MA requires H200, observed {torch.cuda.get_device_name(0)}")
    stage26 = _load_stage26_helper()
    stage29 = _load_stage29_helper()
    stage31 = _load_stage31_helper()
    from gr00t.data.types import MessageType, VLAStepData

    snapshot = stage26._load_snapshot(stage26._verify_snapshot_identity(args.snapshot_root)[1])
    processor, embodiment, modality = stage26._build_runtime_processor(
        groot_source=args.groot_source,
        base_checkpoint=args.base_model_path,
        dataset_dir=args.dataset_dir,
        modality_path=args.modality_config,
        cosmos_snapshot=args.cosmos_snapshot,
    )
    prepared = stage26._prepare_official_sample(
        processor=processor, embodiment=embodiment, modality_configs=modality, dataset_dir=args.dataset_dir
    )
    source_step = prepared["step_data"]
    inference_step = VLAStepData(
        images=source_step.images, states=source_step.states, actions={}, text=source_step.text, embodiment=embodiment
    )
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": inference_step}])
    inference_batch = processor.collator([processed])
    if "action" in inference_batch.get("inputs", {}) or "action_mask" in inference_batch.get("inputs", {}):
        raise Stage33Error("MA inference conditioning contains ground-truth action")
    conditioning = stage29._conditioning_manifest(inference_batch)
    historical = _load_historical_ma_records(args, processor, embodiment, snapshot, prepared, stage26, stage29, stage31)
    if len(historical) != 6:
        raise Stage33Error("MA historical reuse count is not six")
    for seed in (2223, 2224, 2225):
        left = next(row for row in historical if row["seed"] == seed and row["checkpoint"] == "H2@1000")
        right = next(row for row in historical if row["seed"] == seed and row["checkpoint"] == "H2@2000")
        if left["initial_noise"].get("raw_sha256") != right["initial_noise"].get("raw_sha256"):
            raise Stage33Error(f"historical paired noise mismatch at seed {seed}")
    target = prepared["expert_action_cpu"].numpy().astype(np.float32, copy=False)
    expert = stage26._decode_physical_action(processor, embodiment, target, snapshot)
    h2_paths = {"H2@1000": args.h2_checkpoint_1000, "H2@2000": args.h2_checkpoint_2000}
    models = {label: stage29._load_model(path, args.device, torch) for label, path in h2_paths.items()}
    _write_json_atomic(
        state_path,
        {
            "schema": "STAGE33_MA_EXECUTION_STATE_V1",
            "status": "READY_AFTER_MODEL_CONSTRUCTION_BEFORE_FIRST_SAMPLER_CALL",
            "scientific_execution_begun": False,
            "model_constructed": True,
            "sampler_calls_started": 0,
            "sampler_calls_completed": 0,
            "maximum_new_sampler_calls": MA_MAX_NEW_SAMPLER_CALLS,
            "pbs_job_id": args.pbs_job_id,
            "new_seed_set": list(MA_NEW_SEEDS),
            "updated_at_utc": _utc_now(),
        },
    )
    records = list(historical)
    expected_noise: dict[int, str] = {}
    for seed in (2223, 2224, 2225):
        expected_noise[seed] = next(row for row in historical if row["seed"] == seed and row["checkpoint"] == "H2@1000")["initial_noise"]["raw_sha256"]
    new_calls = 0
    for seed in MA_NEW_SEEDS:
        for label in MA_CHECKPOINTS:
            if new_calls >= MA_MAX_NEW_SAMPLER_CALLS:
                raise Stage33Error("MA sampler-call budget exhausted before planned records")
            _update_state(
                state_path,
                {
                    "status": "READY_FOR_FIRST_SCIENTIFIC_CALL" if new_calls == 0 else "BETWEEN_AUTHORIZED_CALLS_AFTER_SCIENCE_BEGUN",
                    "scientific_execution_begun": bool(new_calls),
                    "current_seed": seed,
                    "current_checkpoint": label,
                    "sampler_calls_started": new_calls + 1,
                    "updated_at_utc": _utc_now(),
                },
            )
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            rng = stage29._reset_rng(seed, torch, np)
            call_inputs = copy.deepcopy(inference_batch)
            if stage29._conditioning_manifest(call_inputs)["sha256"] != conditioning["sha256"]:
                raise Stage33Error("MA conditioning changed across sampler calls")
            observer = stage31.StockSamplerTrace(models[label].action_head)
            started = time.perf_counter()
            with stage29._capture_stock_initial_noise(torch) as noise_calls:
                with observer:
                    _update_state(
                        state_path,
                        {"status": "SCIENTIFIC_EXECUTION_BEGUN", "scientific_execution_begun": True, "updated_at_utc": _utc_now()},
                    )
                    with torch.inference_mode():
                        prediction = models[label].get_action(**call_inputs)
            torch.cuda.synchronize()
            noise = stage29._validate_noise_capture(noise_calls, models[label])
            noise_tensor = noise.pop("tensor_float32")
            if seed in expected_noise and noise["raw_sha256"] != expected_noise[seed]:
                raise Stage33Error(f"MA paired initial noise mismatch at seed {seed}")
            expected_noise.setdefault(seed, noise["raw_sha256"])
            trace = stage31._finalize_trace(observer, prediction["action_pred"], torch)
            raw = prediction["action_pred"].detach().float().contiguous().cpu().numpy().astype(np.float32, copy=False)
            if raw.shape != (1, HORIZON, UNIVERSAL_DIM) or not np.isfinite(raw).all():
                raise Stage33Error("MA official sampler returned invalid universal action")
            decoded = stage26._decode_physical_action(processor, embodiment, raw, snapshot)
            artifact = args.output_root / "ma_traces" / f"seed_{seed}_{'h2_1000' if label == 'H2@1000' else 'h2_2000'}_stock_sampler.npz"
            if artifact.exists():
                raise Stage33Error(f"MA artifact already exists: {artifact}")
            artifact.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                artifact,
                initial_noise_float32=noise_tensor.numpy().astype(np.float32, copy=False),
                raw_universal_action=raw,
                decoded_physical_action=decoded,
                expert_physical_action=expert,
                x_trajectory_float32=trace["x_trajectory_float32"],
                v_trajectory_float32=trace["v_trajectory_float32"],
                timestep_ids=np.asarray(trace["timestep_ids"], dtype=np.int64),
            )
            trace_metrics = stage31.trace_metrics(trace["x_trajectory_float32"], trace["v_trajectory_float32"], target)
            physical = stage29._trajectory_metrics(
                decoded, expert, stage29_profile_module(), stage26, processor, embodiment, snapshot, h2_paths[label], label
            )
            row = _ma_record(
                seed=seed,
                checkpoint=label,
                checkpoint_path=h2_paths[label],
                physical=physical,
                decoded=decoded,
                expert=expert,
                trace_metrics=trace_metrics,
                raw=raw,
                initial_noise=noise,
                conditioning_sha256=conditioning["sha256"],
                artifact={"path": str(artifact), "bytes": artifact.stat().st_size, "sha256": _sha256_file(artifact)},
                new_sampler_call=True,
                provenance="STAGE33_MA_NEW",
                timestep_ids=trace["timestep_ids"],
            )
            row["rng_before_call"] = rng
            row["latency_s"] = time.perf_counter() - started
            records.append(row)
            new_calls += 1
            _write_ma_partial(args.output_root, records)
            _update_state(
                state_path,
                {
                    "status": "SCIENTIFIC_EXECUTION_IN_PROGRESS",
                    "scientific_execution_begun": True,
                    "sampler_calls_started": new_calls,
                    "sampler_calls_completed": new_calls,
                    "last_completed_seed": seed,
                    "last_completed_checkpoint": label,
                    "updated_at_utc": _utc_now(),
                },
            )
    if sum(int(row["new_sampler_call"]) for row in records) != MA_MAX_NEW_SAMPLER_CALLS:
        raise Stage33Error("MA new-call counter does not equal 26")
    aggregates = _ma_aggregates(records)
    result = {
        "schema": "STAGE33_MA_RESULT_V1",
        "status": "PASS_EXECUTION_COMPLETE",
        "stage": "Stage33-MA paired exposure robustness",
        "pi_gate": PI_GATE,
        "pbs": {"job_id": args.pbs_job_id, "project": args.pbs_project, "gpu": torch.cuda.get_device_name(0)},
        "preflight": preflight,
        "conditioning": {"sha256": conditioning["sha256"], "ground_truth_action_excluded": True},
        "records": records,
        "aggregates": aggregates,
        "execution": {
            "scientific_execution_begun": True,
            "new_sampler_calls": MA_MAX_NEW_SAMPLER_CALLS,
            "base_sampler_calls": 0,
            "maximum_new_sampler_calls": MA_MAX_NEW_SAMPLER_CALLS,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "scheduler_step": False,
            "resume": False,
            "checkpoint_write": False,
            "wbc": False,
            "hardware": False,
            "validation_e": False,
            "holdout_f": False,
        },
        "all_finite": all(row["mandatory_metrics"]["all_finite"] for row in records),
        "scientific_contract_changed": False,
        "claim_boundary": "single locked TINY1 sample; sixteen paired H2 exposure-transfer seeds; descriptive only",
        "next_gate": NEXT_GATE,
    }
    _write_json_atomic(args.output_root / "stage33_ma_result.json", result)
    _write_json_atomic(
        args.output_root / "stage33_ma_raw_hash_manifest.json",
        {
            "schema": "STAGE33_MA_RAW_HASH_MANIFEST_V1",
            "status": "PASS",
            "records": [
                {
                    "seed": row["seed"],
                    "checkpoint": row["checkpoint"],
                    "new_sampler_call": row["new_sampler_call"],
                    "initial_noise_raw_sha256": row["initial_noise"].get("raw_sha256"),
                    "raw_universal_sha256": row["raw_universal_action"]["sha256"],
                    "decoded_physical_sha256": row["decoded_physical_action"]["sha256"],
                    "artifact_sha256": row["artifact"]["sha256"],
                }
                for row in records
            ],
            "scientific_contract_changed": False,
        },
    )
    _update_state(
        state_path,
        {"status": "PASS_EXECUTION_COMPLETE", "scientific_execution_begun": True, "sampler_calls_completed": MA_MAX_NEW_SAMPLER_CALLS, "updated_at_utc": _utc_now()},
    )
    return result


def _mb_seed(checkpoint_index: int, probe_index: int, resample_index: int) -> int:
    if checkpoint_index not in (0, 1) or probe_index not in range(9) or resample_index not in (1, 2, 3):
        raise Stage33Error("invalid MB resample index")
    return 733100 + checkpoint_index * 100000 + probe_index * 100 + resample_index


def mb_intervention_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint_index, checkpoint in enumerate(MB_CHECKPOINTS):
        for probe_index, probe_id in enumerate(STAGE26_PROBE_IDS):
            for condition in MB_CONDITIONS:
                resample_index = None
                seed = None
                if condition.startswith(("B_", "C_", "D_")):
                    resample_index = {"B_RESAMPLE_1": 1, "C_RESAMPLE_2": 2, "D_RESAMPLE_3": 3}[condition]
                    seed = _mb_seed(checkpoint_index, probe_index, resample_index)
                rows.append(
                    {
                        "checkpoint": checkpoint,
                        "checkpoint_index": checkpoint_index,
                        "probe_id": probe_id,
                        "probe_index": probe_index,
                        "condition": condition,
                        "resample_index": resample_index,
                        "cpu_generator_seed": seed,
                        "active_slice": [0, ACTIVE_DIM],
                        "inactive_slice": [ACTIVE_DIM, UNIVERSAL_DIM],
                    }
                )
    if len(rows) != MB_MAX_DIRECT_CALLS:
        raise Stage33Error("MB intervention plan count drifted")
    return rows


def _mb_condition(fixed: Mapping[str, Any], condition: str, checkpoint_index: int, probe_index: int, torch: Any) -> tuple[Any, int | None]:
    if condition not in MB_CONDITIONS:
        raise Stage33Error(f"unknown MB condition: {condition}")
    noisy = fixed["noisy_action"].detach().clone().to(device="cpu", dtype=torch.float32)
    seed: int | None = None
    if condition.startswith(("B_", "C_", "D_")):
        resample_index = {"B_RESAMPLE_1": 1, "C_RESAMPLE_2": 2, "D_RESAMPLE_3": 3}[condition]
        seed = _mb_seed(checkpoint_index, probe_index, resample_index)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        inactive = torch.randn((1, HORIZON, INACTIVE_DIM), generator=generator, device="cpu", dtype=torch.float32)
        noisy[..., ACTIVE_DIM:] = inactive
    elif condition == "E_ZERO_INACTIVE":
        noisy[..., ACTIVE_DIM:] = 0.0
    if not torch.equal(noisy[..., :ACTIVE_DIM], fixed["noisy_action"].detach().to(device="cpu", dtype=torch.float32)[..., :ACTIVE_DIM]):
        raise Stage33Error(f"MB active noisy state changed for {condition}")
    if not bool(torch.isfinite(noisy).all().item()):
        raise Stage33Error(f"MB noisy action non-finite for {condition}")
    return noisy, seed


def _mb_metrics(predicted: Any, fixed: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    pred = predicted.detach().float().cpu().numpy().astype(np.float32, copy=False)[0]
    target = fixed["target_velocity"].detach().float().cpu().numpy().astype(np.float32, copy=False)[0]
    active = pred[:, :ACTIVE_DIM]
    target_active = target[:, :ACTIVE_DIM]
    if pred.shape != (HORIZON, UNIVERSAL_DIM) or not np.isfinite(pred).all():
        raise Stage33Error("MB predicted velocity shape/finite contract failed")
    delta = active.astype(np.float64) - target_active.astype(np.float64)
    axes = {"x": 0, "y": 1, "z": 2}
    return {
        "active10_oracle_mse": float(np.mean(np.square(delta))),
        "xyz_oracle_mse": float(np.mean(np.square(delta[:, :3]))),
        "axis_oracle_mse": {name: float(np.mean(np.square(delta[:, index]))) for name, index in axes.items()},
        "active10_predicted_mean": float(np.mean(active)),
        "active10_predicted_std": float(np.std(active)),
        "finite": True,
    }


def _write_mb_partial(output_root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    _write_json_atomic(
        output_root / "stage33_mb_partial_records.json",
        {"schema": "STAGE33_MB_DIRECT_RECORD_V1", "status": "PARTIAL_EVIDENCE", "record_count": len(records), "records": list(records)},
    )


def _mb_aggregates(args: argparse.Namespace, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    expected_keys = {(checkpoint, probe, condition) for checkpoint in MB_CHECKPOINTS for probe in STAGE26_PROBE_IDS for condition in MB_CONDITIONS}
    observed_keys = {(str(row["checkpoint"]), str(row["probe_id"]), str(row["condition"])) for row in records}
    if observed_keys != expected_keys or len(records) != MB_MAX_DIRECT_CALLS:
        raise Stage33Error("MB record key set/count drifted")
    groups: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    for row in records:
        groups.setdefault((str(row["checkpoint"]), str(row["probe_id"])), {})[str(row["condition"])] = row
    probe_rows: list[dict[str, Any]] = []
    for checkpoint in MB_CHECKPOINTS:
        for probe_id in STAGE26_PROBE_IDS:
            group = groups[(checkpoint, probe_id)]
            arrays = []
            for condition in MB_CONDITIONS:
                row = group[condition]
                path = Path(str(row["artifact"]["path"]))
                if _sha256_file(path) != row["artifact"]["sha256"]:
                    raise Stage33Error(f"MB output artifact changed: {path}")
                with np.load(path, allow_pickle=False) as payload:
                    value = np.asarray(payload["predicted_velocity_float32"], dtype=np.float32)
                if value.shape != (1, HORIZON, UNIVERSAL_DIM) or not np.isfinite(value).all():
                    raise Stage33Error(f"MB output artifact shape/finite failure: {path}")
                arrays.append(value[0, :, :ACTIVE_DIM])
            stack = np.stack(arrays[:4], axis=0).astype(np.float64)
            variance = np.var(stack, axis=0, ddof=0)
            variance_xyz = variance[:, :3]
            stock = group["A_STOCK"]
            stock_mse = float(stock["metrics"]["active10_oracle_mse"])
            if stock_mse <= 0 or not math.isfinite(stock_mse):
                raise Stage33Error(f"MB stock oracle MSE is not positive at {checkpoint}/{probe_id}")
            center = np.mean(stack, axis=0)
            inactive_induced = float(np.mean(np.square(stack - center)))
            ratio = inactive_induced / stock_mse
            zero_mse = float(group["E_ZERO_INACTIVE"]["metrics"]["active10_oracle_mse"])
            probe_rows.append(
                {
                    "checkpoint": checkpoint,
                    "probe_id": probe_id,
                    "predicted_velocity_variance": {
                        "active10_mean": float(np.mean(variance)),
                        "active10_max": float(np.max(variance)),
                        "xyz_mean": float(np.mean(variance_xyz)),
                        "xyz_max": float(np.max(variance_xyz)),
                        "x_mean": float(np.mean(variance[:, 0])),
                        "y_mean": float(np.mean(variance[:, 1])),
                        "z_mean": float(np.mean(variance[:, 2])),
                    },
                    "mse_change_relative_to_stock": {
                        condition: float(group[condition]["metrics"]["active10_oracle_mse"] - stock_mse)
                        for condition in MB_CONDITIONS if condition != "A_STOCK"
                    },
                    "oracle_velocity_mse": {
                        condition: float(group[condition]["metrics"]["active10_oracle_mse"])
                        for condition in MB_CONDITIONS
                    },
                    "inactive_induced_active_mse": inactive_induced,
                    "inactive_induced_to_stock_oracle_mse": ratio,
                    "zero_inactive_improves_oracle_mse": zero_mse < stock_mse,
                    "condition_hashes": {condition: group[condition]["noisy_action_sha256"] for condition in MB_CONDITIONS},
                    "active_noisy_action_hashes_equal": len({group[condition]["active_noisy_action_sha256"] for condition in MB_CONDITIONS}) == 1,
                    "target_velocity_hashes_equal": len({group[condition]["target_velocity_sha256"] for condition in MB_CONDITIONS}) == 1,
                }
            )
    if not all(row["active_noisy_action_hashes_equal"] and row["target_velocity_hashes_equal"] for row in probe_rows):
        raise Stage33Error("MB active state or target changed across conditions")
    h2_rows = [row for row in probe_rows if row["checkpoint"] == "H2@2000"]
    ratios_by_checkpoint = {
        checkpoint: [row["inactive_induced_to_stock_oracle_mse"] for row in probe_rows if row["checkpoint"] == checkpoint]
        for checkpoint in MB_CHECKPOINTS
    }
    zero_counts = {
        checkpoint: sum(bool(row["zero_inactive_improves_oracle_mse"]) for row in probe_rows if row["checkpoint"] == checkpoint)
        for checkpoint in MB_CHECKPOINTS
    }
    material = median(ratios_by_checkpoint["H2@2000"]) >= MB_MATERIAL_RATIO and zero_counts["H2@2000"] >= MB_MATERIAL_ZERO_IMPROVEMENTS
    return {
        "probe_rows": probe_rows,
        "per_checkpoint_nuisance_sensitivity": {
            checkpoint: {
                "median_inactive_to_stock_oracle_mse": median(values),
                "mean_inactive_to_stock_oracle_mse": mean(values),
                "zero_inactive_improvement_count": zero_counts[checkpoint],
            }
            for checkpoint, values in ratios_by_checkpoint.items()
        },
        "inactive_coupling_strong": bool(material),
        "material_coupling_criterion": {
            "median_threshold": MB_MATERIAL_RATIO,
            "h2_zero_improvement_threshold": MB_MATERIAL_ZERO_IMPROVEMENTS,
            "h2_median_observed": median(ratios_by_checkpoint["H2@2000"]),
            "h2_zero_improvement_count_observed": zero_counts["H2@2000"],
        },
        "h2_1000_vs_h2_2000_nuisance_sensitivity": {
            "median_ratio_delta": median(ratios_by_checkpoint["H2@2000"]) - median(ratios_by_checkpoint["H2@1000"]),
            "h2_1000_median": median(ratios_by_checkpoint["H2@1000"]),
            "h2_2000_median": median(ratios_by_checkpoint["H2@2000"]),
        },
        "direct_calls": len(records),
        "maximum_direct_calls": MB_MAX_DIRECT_CALLS,
        "descriptive_only": True,
        "scientific_interpretation": "DEFERRED_TO_GPT_PI",
    }


def run_mb(args: argparse.Namespace) -> dict[str, Any]:
    preflight = _read_json(args.output_root / "stage33_preflight.json")
    if preflight.get("status") != "PASS_PREFLIGHT_NO_MODEL":
        raise Stage33Error("MB requires a passing Stage33 preflight")
    if not (args.output_root / "stage33_ma_result.json").is_file():
        raise Stage33Error("MB requires completed MA evidence in the same campaign")
    state_path = args.output_root / "stage33_mb_execution_state.json"
    partial_path = args.output_root / "stage33_mb_partial_records.json"
    if state_path.exists() or partial_path.exists() or (args.output_root / "stage33_mb_result.json").exists():
        raise Stage33Error("MB output root is not fresh; resume/replacement is forbidden")
    import numpy as np
    import torch

    if args.device != "cuda:0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Stage33Error("MB requires exactly one visible CUDA device at cuda:0")
    if "H200" not in torch.cuda.get_device_name(0):
        raise Stage33Error(f"MB requires H200, observed {torch.cuda.get_device_name(0)}")
    stage26 = _load_stage26_helper()
    stage29 = _load_stage29_helper()
    from scripts import stage23_phase_c_flow_field_diagnostic as stage23_flow
    from gr00t.data.types import MessageType, VLAStepData

    stage23_flow.assert_protocol_static(stage23_flow.protocol_plan())
    snapshot = stage26._load_snapshot(stage26._verify_snapshot_identity(args.snapshot_root)[1])
    processor, embodiment, modality = stage26._build_runtime_processor(
        groot_source=args.groot_source,
        base_checkpoint=args.base_model_path,
        dataset_dir=args.dataset_dir,
        modality_path=args.modality_config,
        cosmos_snapshot=args.cosmos_snapshot,
    )
    prepared = stage26._prepare_official_sample(
        processor=processor, embodiment=embodiment, modality_configs=modality, dataset_dir=args.dataset_dir
    )
    step = prepared["step_data"]
    inference_step = VLAStepData(images=step.images, states=step.states, actions={}, text=step.text, embodiment=embodiment)
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": inference_step}])
    batch = processor.collator([processed])
    if "action" in batch.get("inputs", {}) or "action_mask" in batch.get("inputs", {}):
        raise Stage33Error("MB inference conditioning contains ground-truth action")
    contract = stage26._static_probe_contract()
    probes = contract["probes"]
    if len(probes) != 9 or [str(row["probe_id"]) for row in probes] != list(STAGE26_PROBE_IDS):
        raise Stage33Error("MB fixed probe order drifted")
    fixed_probes = [stage26._canonical_probe_inputs(stage23_flow, prepared["expert_action_cpu"], probe) for probe in probes]
    intervention_plan = mb_intervention_plan()
    _write_json_atomic(args.output_root / "stage33_mb_intervention_plan.json", {"schema": "STAGE33_MB_INTERVENTION_PLAN_V1", "status": "PASS", "plan": intervention_plan, "scientific_contract_changed": False})
    h2_paths = {"H2@1000": args.h2_checkpoint_1000, "H2@2000": args.h2_checkpoint_2000}
    _write_json_atomic(
        state_path,
        {
            "schema": "STAGE33_MB_EXECUTION_STATE_V1",
            "status": "READY_BEFORE_FIRST_SCIENTIFIC_CALL",
            "scientific_execution_begun": False,
            "model_constructed": False,
            "direct_calls_started": 0,
            "direct_calls_completed": 0,
            "maximum_direct_calls": MB_MAX_DIRECT_CALLS,
            "pbs_job_id": args.pbs_job_id,
            "updated_at_utc": _utc_now(),
        },
    )
    records: list[dict[str, Any]] = []
    call_index = 0
    for checkpoint_index, checkpoint in enumerate(MB_CHECKPOINTS):
        model = stage29._load_model(h2_paths[checkpoint], args.device, torch)
        _update_state(state_path, {"model_constructed": True, "current_checkpoint": checkpoint, "status": "MODEL_CONSTRUCTED_BEFORE_DIRECT_CALLS", "updated_at_utc": _utc_now()})
        model_inputs = prepared["model_inputs"]
        backbone_inputs, action_inputs = model.prepare_input(dict(model_inputs))
        with torch.inference_mode():
            backbone_output = model.backbone(backbone_inputs)
            processed_backbone = model.action_head.process_backbone_output(backbone_output)
        model_dtype = next(iter(model.parameters())).dtype
        for probe_index, (probe, fixed) in enumerate(zip(probes, fixed_probes)):
            if int(fixed["t_discretized"].reshape(-1)[0].item()) != int(probe["t_discretized"]):
                raise Stage33Error("MB timestep bucket drifted")
            for condition in MB_CONDITIONS:
                if call_index >= MB_MAX_DIRECT_CALLS:
                    raise Stage33Error("MB direct-flow budget exhausted")
                noisy_cpu, seed = _mb_condition(fixed, condition, checkpoint_index, probe_index, torch)
                active_hash = _array_sha256(noisy_cpu[..., :ACTIVE_DIM])
                target_hash = _array_sha256(fixed["target_velocity"].detach().float().cpu())
                noisy_hash = _array_sha256(noisy_cpu)
                _update_state(
                    state_path,
                    {
                        "status": "READY_FOR_FIRST_SCIENTIFIC_CALL" if call_index == 0 else "BETWEEN_AUTHORIZED_CALLS_AFTER_SCIENCE_BEGUN",
                        "scientific_execution_begun": bool(call_index),
                        "current_checkpoint": checkpoint,
                        "current_probe_id": probe["probe_id"],
                        "current_condition": condition,
                        "direct_calls_started": call_index + 1,
                        "updated_at_utc": _utc_now(),
                    },
                )
                started = time.perf_counter()
                with torch.inference_mode():
                    predicted = stage26._teacher_forced_local_flow(
                        action_head=model.action_head,
                        backbone_output=processed_backbone,
                        action_input=action_inputs,
                        noisy_action=noisy_cpu.to(device=processed_backbone.backbone_features.device, dtype=model_dtype),
                        t_discretized=fixed["t_discretized"].to(device=processed_backbone.backbone_features.device),
                        on_before_direct_probe_forward=lambda: _update_state(state_path, {"status": "SCIENTIFIC_EXECUTION_BEGUN", "scientific_execution_begun": True, "updated_at_utc": _utc_now()}),
                    )
                elapsed = time.perf_counter() - started
                predicted_cpu = predicted.detach().float().contiguous().cpu()
                metrics = _mb_metrics(predicted_cpu, fixed)
                artifact = args.output_root / "mb_outputs" / checkpoint.lower().replace("@", "_") / str(probe["probe_id"]) / f"{condition.lower()}.npz"
                if artifact.exists():
                    raise Stage33Error(f"MB artifact already exists: {artifact}")
                artifact.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    artifact,
                    predicted_velocity_float32=predicted_cpu.numpy().astype(np.float32, copy=False),
                    noisy_action_float32=noisy_cpu.numpy().astype(np.float32, copy=False),
                    target_velocity_float32=fixed["target_velocity"].detach().float().cpu().numpy().astype(np.float32, copy=False),
                    t_discretized=fixed["t_discretized"].detach().cpu().numpy().astype(np.int64, copy=False),
                )
                row = {
                    "schema": "STAGE33_MB_DIRECT_RECORD_V1",
                    "call_index": call_index + 1,
                    "checkpoint": checkpoint,
                    "checkpoint_path": str(h2_paths[checkpoint]),
                    "probe_id": str(probe["probe_id"]),
                    "probe_index": probe_index,
                    "condition": condition,
                    "cpu_generator_seed": seed,
                    "t_value": float(probe["t_value"]),
                    "t_discretized": int(probe["t_discretized"]),
                    "active_noisy_action_sha256": active_hash,
                    "noisy_action_sha256": noisy_hash,
                    "target_velocity_sha256": target_hash,
                    "predicted_velocity_sha256": _array_sha256(predicted_cpu),
                    "predicted_velocity_shape": list(predicted_cpu.shape),
                    "predicted_velocity_dtype": str(predicted_cpu.dtype),
                    "metrics": metrics,
                    "artifact": {"path": str(artifact), "bytes": artifact.stat().st_size, "sha256": _sha256_file(artifact)},
                    "elapsed_s": elapsed,
                    "forward_scope": "direct_teacher_forced_action_head_only_after_one_shared_backbone_pass",
                    "iterative_sampler": False,
                    "training": False,
                    "backward": False,
                    "wbc": False,
                    "hardware": False,
                }
                records.append(row)
                call_index += 1
                _write_mb_partial(args.output_root, records)
                _update_state(
                    state_path,
                    {
                        "status": "SCIENTIFIC_EXECUTION_IN_PROGRESS",
                        "scientific_execution_begun": True,
                        "direct_calls_started": call_index,
                        "direct_calls_completed": call_index,
                        "last_completed_checkpoint": checkpoint,
                        "last_completed_probe_id": probe["probe_id"],
                        "last_completed_condition": condition,
                        "updated_at_utc": _utc_now(),
                    },
                )
        del model
        torch.cuda.empty_cache()
    if call_index != MB_MAX_DIRECT_CALLS:
        raise Stage33Error(f"MB direct call count is {call_index}, not {MB_MAX_DIRECT_CALLS}")
    aggregates = _mb_aggregates(args, records)
    result = {
        "schema": "STAGE33_MB_RESULT_V1",
        "status": "PASS_EXECUTION_COMPLETE",
        "stage": "Stage33-MB inactive-122 nuisance-noise sensitivity audit",
        "pi_gate": PI_GATE,
        "pbs": {"job_id": args.pbs_job_id, "project": args.pbs_project, "gpu": torch.cuda.get_device_name(0)},
        "preflight": preflight,
        "intervention_plan": {"path": str(args.output_root / "stage33_mb_intervention_plan.json"), "sha256": _sha256_file(args.output_root / "stage33_mb_intervention_plan.json")},
        "records": records,
        "aggregates": aggregates,
        "execution": {
            "scientific_execution_begun": True,
            "direct_flow_calls": MB_MAX_DIRECT_CALLS,
            "maximum_direct_flow_calls": MB_MAX_DIRECT_CALLS,
            "official_sampler_calls": 0,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "scheduler_step": False,
            "resume": False,
            "checkpoint_write": False,
            "iterative_sampler": False,
            "wbc": False,
            "hardware": False,
            "validation_e": False,
            "holdout_f": False,
        },
        "all_finite": all(bool(row["metrics"]["finite"]) for row in records),
        "scientific_contract_changed": False,
        "claim_boundary": "same nine fixed direct-flow probes; diagnostic nuisance intervention only; descriptive only",
        "next_gate": NEXT_GATE,
    }
    _write_json_atomic(args.output_root / "stage33_mb_result.json", result)
    _write_json_atomic(
        args.output_root / "stage33_mb_raw_hash_manifest.json",
        {
            "schema": "STAGE33_MB_RAW_HASH_MANIFEST_V1",
            "status": "PASS",
            "records": [
                {
                    "call_index": row["call_index"],
                    "checkpoint": row["checkpoint"],
                    "probe_id": row["probe_id"],
                    "condition": row["condition"],
                    "predicted_velocity_sha256": row["predicted_velocity_sha256"],
                    "artifact_sha256": row["artifact"]["sha256"],
                }
                for row in records
            ],
            "scientific_contract_changed": False,
        },
    )
    _update_state(state_path, {"status": "PASS_EXECUTION_COMPLETE", "scientific_execution_begun": True, "direct_calls_completed": MB_MAX_DIRECT_CALLS, "updated_at_utc": _utc_now()})
    return result


def run_mc(args: argparse.Namespace) -> dict[str, Any]:
    preflight = _read_json(args.output_root / "stage33_preflight.json")
    if preflight.get("status") != "PASS_PREFLIGHT_NO_MODEL":
        raise Stage33Error("MC requires a passing Stage33 preflight")
    output = args.output_root / "stage33_mc_mapping_manifest.json"
    if output.exists() or (args.output_root / "stage33_mc_result.json").exists():
        raise Stage33Error("MC output is not fresh")
    audit_module = _load_module(REPO_ROOT / "scripts" / "stage24_static_checkpoint_audit.py", "stage24_for_stage33_mc")
    audit = audit_module.audit_checkpoint(args.base_model_path, tree_prefix=BASE_TREE_PREFIX)
    source_rows = []
    for relative, expected in MC_SOURCE_FILES.items():
        path = args.groot_source / relative
        observed = _sha256_file(path)
        if observed != expected:
            raise Stage33Error(f"MC pinned source hash mismatch: {relative}: {observed}")
        text = path.read_text(encoding="utf-8")
        anchors = MC_SOURCE_ANCHORS.get(relative, ())
        missing = [anchor for anchor in anchors if anchor not in text]
        if missing:
            raise Stage33Error(f"MC source anchors missing in {relative}: {missing}")
        source_rows.append({"path": relative, "sha256": observed, "anchors": list(anchors)})
    names = audit["checkpoint_shape"]["category_names"].get(str(EXPECTED_DROID_ID), [])
    if "oxe_droid_relative_eef_relative_joint" not in names:
        raise Stage33Error(f"DROID category ID {EXPECTED_DROID_ID} is not bound by checkpoint embodiment_id.json")
    profiles = audit["category_weight_profiles"]
    weight_shapes = {}
    droid_non_initializer = {}
    for key, rows in profiles.items():
        droid = rows.get(str(EXPECTED_DROID_ID))
        if not isinstance(droid, Mapping):
            raise Stage33Error(f"DROID row missing for {key}")
        shape = droid.get("shape")
        if not isinstance(shape, list):
            raise Stage33Error(f"DROID shape missing for {key}")
        weight_shapes[key] = shape
        droid_non_initializer[key] = bool(droid.get("finite") is True and float(droid.get("initializer_rms_ratio", 0.0)) > 1.10)
    required_shapes = {
        "action_head.action_encoder.W1.W": [132, weight_shapes["action_head.action_encoder.W1.W"][1]],
        "action_head.action_encoder.W2.W": weight_shapes["action_head.action_encoder.W2.W"],
        "action_head.action_encoder.W3.W": weight_shapes["action_head.action_encoder.W3.W"],
        "action_head.action_decoder.layer1.W": weight_shapes["action_head.action_decoder.layer1.W"],
        "action_head.action_decoder.layer2.W": [weight_shapes["action_head.action_decoder.layer2.W"][0], 132],
    }
    shape_contract = {
        "observed_row_shapes": weight_shapes,
        "w1_input_dimension": weight_shapes["action_head.action_encoder.W1.W"][0] == 132,
        "decoder_output_dimension": weight_shapes["action_head.action_decoder.layer2.W"][1] == 132,
        "hidden_to_hidden_blocks": [
            "action_head.action_encoder.W2.W",
            "action_head.action_encoder.W3.W",
            "action_head.action_decoder.layer1.W",
        ],
        "shape_summary": required_shapes,
    }
    mapping = {
        "status": "QUALIFIED" if all(droid_non_initializer.values()) and shape_contract["w1_input_dimension"] and shape_contract["decoder_output_dimension"] else "NOT_QUALIFIED",
        "source_category_id": EXPECTED_DROID_ID,
        "target_category_id": EXPECTED_ID10,
        "target_action_abi": {"horizon": HORIZON, "active_dimensions": 10, "universal_dimensions": 132, "active_semantics": ["relative EEF XYZ", "relative EEF rot6d", "absolute gripper"]},
        "droid_action_abi": {"first_10": ["relative EEF XYZ", "relative EEF rot6d", "absolute gripper"], "joint_dimensions": [10, 17], "total_modal_action_dimension": 17},
        "copy_operations": [
            {"key": "action_head.action_encoder.W1.W", "axis": 1, "source_range": [0, 10], "target_range": [0, 10], "operation": "copy_compatible_first10_semantics"},
            {"key": "action_head.action_encoder.W2.W", "axis": "all", "operation": "copy_hidden_to_hidden_if_shape_exact"},
            {"key": "action_head.action_encoder.W3.W", "axis": "all", "operation": "copy_hidden_to_hidden_if_shape_exact"},
        ],
        "neutralization_operations": [
            {"key": "action_head.action_encoder.W1.W", "axis": 1, "target_range": [10, 132], "operation": "zero_target_inactive_input_columns"},
            {"source_joint_range": [10, 17], "operation": "never_interpret_as_go2_arx5_action"},
            {"source_padded_range": [17, 132], "operation": "never_copy_as_semantic_droid_joint_input"},
        ],
        "uncopied_decoder": "ID10 action decoder remains unchanged; MC is a mapping audit, not an applied checkpoint edit",
        "scientific_contract_changed": False,
    }
    _write_json_atomic(
        output,
        {
            "schema": "STAGE33_MC_DROID_PROJECTOR_MAPPING_V1",
            "status": "PASS",
            "checkpoint": {"path": str(args.base_model_path), "tree_sha256": BASE_TREE_SHA256, "static_audit_classification": audit["id10"]["classification"]},
            "source": {"git_head": GROOT_SHA, "files": source_rows},
            "droid_identity": {"category_id": EXPECTED_DROID_ID, "category_names": names},
            "shape_contract": shape_contract,
            "droid_rows_finite_non_initializer_like": droid_non_initializer,
            "mapping": mapping,
            "execution_boundary": {"model_constructed": False, "forward": False, "training": False, "gpu_used": False, "checkpoint_modified": False},
            "claim_boundary": "source/parameter/semantic mapping only; no DROID-initialized model was trained or evaluated",
        },
    )
    result = {
        "schema": "STAGE33_MC_RESULT_V1",
        "status": "PASS_EXECUTION_COMPLETE",
        "stage": "Stage33-MC DROID relative-EEF projector transfer audit",
        "pi_gate": PI_GATE,
        "preflight": preflight,
        "mapping_manifest": {"path": str(output), "sha256": _sha256_file(output)},
        "mapping_status": mapping["status"],
        "static_audit": {"id10_classification": audit["id10"]["classification"], "checkpoint_shape": audit["checkpoint_shape"]},
        "execution": {"model_constructed": False, "forward": False, "training": False, "gpu_used": False, "checkpoint_modified": False, "validation_e": False, "holdout_f": False, "wbc": False, "hardware": False},
        "scientific_contract_changed": False,
        "next_gate": NEXT_GATE,
    }
    _write_json_atomic(args.output_root / "stage33_mc_result.json", result)
    return result


def _stage16_report_entry(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        return {"path": str(path), "present": False}
    return {"path": str(path), "present": True, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def run_wa(args: argparse.Namespace) -> dict[str, Any]:
    preflight = _read_json(args.output_root / "stage33_preflight.json")
    if preflight.get("status") != "PASS_PREFLIGHT_NO_MODEL":
        raise Stage33Error("WA requires a passing Stage33 preflight")
    output = args.output_root / "stage33_wa_zero_sim_audit.json"
    if output.exists() or (args.output_root / "stage33_wa_result.json").exists():
        raise Stage33Error("WA output is not fresh")
    report_paths = [
        REPO_ROOT / "reports" / "STAGE16_FINAL_REPORT.md",
        REPO_ROOT / "reports" / "STAGE16_CONTINUOUS_BASE_CHALLENGE_DECISION.md",
        REPO_ROOT / "reports" / "STAGE16_PROGRESSIVE_RELOCATION_COHORT.md",
        REPO_ROOT / "reports" / "STAGE16_RESOURCE_GATE.md",
    ]
    reports = [_stage16_report_entry(path) for path in report_paths]
    if not all(row.get("present") for row in reports):
        raise Stage33Error("committed Stage16 reports are unavailable")
    report_text = "\n".join(path.read_text(encoding="utf-8") for path in report_paths)
    if "NO_VALID_NATIVE_CONTINUOUS_BASE_COHORT_FOUND" not in report_text or "27" not in report_text or "28" not in report_text:
        raise Stage33Error("Stage16 report boundary anchors are not present")
    required_assets = {
        "stage16_root": _stage16_report_entry(args.stage16_root),
        "progressive_cohort_manifest": _stage16_report_entry(args.stage16_root / "progressive_relocation" / "progressive_cohort.json"),
    }
    missing_support = [name for name, row in required_assets.items() if not row.get("present")]
    classification = "NO_VALID_SUPPORT_AWARE_CONTINUOUS_COHORT" if missing_support else "AMBIGUOUS"
    payload = {
        "schema": "STAGE33_WA_ZERO_SIM_AUDIT_V1",
        "status": "PASS_ZERO_SIM_AUDIT",
        "stage16_report_evidence": reports,
        "required_immutable_inputs": {
            "delivered_wbc_archive_sha256": STAGE16_ARCHIVE_SHA256,
            "native_urdf_sha256": STAGE16_URDF_SHA256,
            "arm_checkpoint_sha256": STAGE16_ARM_CHECKPOINT_SHA256,
            "dog_checkpoint_sha256": STAGE16_DOG_CHECKPOINT_SHA256,
            "native_config_sha256": STAGE16_NATIVE_CONFIG_SHA256,
            "p1m_usd_sha256": STAGE16_P1M_USD_SHA256,
        },
        "available_support": required_assets,
        "missing_support_keys": missing_support,
        "deterministic_feasible_reference_bounds": {
            "source": "Stage16 committed report only",
            "reference_complete_count": 27,
            "reference_total_count": 28,
            "falls": 0,
            "bounded_progressive_linear_speed_mps_approx": 0.25,
            "bounded_progressive_angular_speed_radps_approx": 0.65,
            "continuous_base_cohort_previously_found": False,
        },
        "cohort_rule": {
            "support_aware": True,
            "deterministic": True,
            "required_families": ["longitudinal", "lateral", "diagonal_curved", "matched_timing_speed", "boundary_near_valid"],
            "maximum_new_rollouts": 24,
        },
        "simulation": {"performed": False, "new_rollouts": 0, "wbc_executed": False, "hardware": False, "gr00t": False},
        "classification": classification,
        "claim_boundary": "zero-simulation support audit; no WBC failure or continuous-base impossibility claim",
        "scientific_contract_changed": False,
    }
    _write_json_atomic(output, payload)
    result = {
        "schema": "STAGE33_WA_RESULT_V1",
        "status": "PASS_EXECUTION_COMPLETE",
        "stage": "Stage33-WA UMI baseline support-aware continuous-reference audit",
        "pi_gate": PI_GATE,
        "preflight": preflight,
        "zero_sim_audit": {"path": str(output), "sha256": _sha256_file(output)},
        "classification": classification,
        "simulation_rollouts": 0,
        "execution": {"model_constructed": False, "forward": False, "training": False, "wbc": False, "hardware": False, "validation_e": False, "holdout_f": False},
        "scientific_contract_changed": False,
        "next_gate": NEXT_GATE,
    }
    _write_json_atomic(args.output_root / "stage33_wa_result.json", result)
    return result


def choose_stage34_branch(ma: Mapping[str, Any], mb: Mapping[str, Any], mc: Mapping[str, Any], wa: Mapping[str, Any]) -> dict[str, Any]:
    ma_positive = ma.get("aggregates", {}).get("exposure_robust_positive") is True
    ma_mixed = ma.get("aggregates", {}).get("classification") == "EXPOSURE_ROBUST_MIXED"
    mb_strong = mb.get("aggregates", {}).get("inactive_coupling_strong") is True
    mc_qualified = mc.get("mapping_status") == "QUALIFIED"
    if mb_strong:
        branch = "M"
        reason = "inactive coupling criterion is strong"
    elif ma_positive:
        branch = "S"
        reason = "exposure robust positive and inactive coupling is not strong"
    elif ma_mixed and mc_qualified:
        branch = "D"
        reason = "exposure mixed and qualified DROID semantic mapping is available"
    else:
        branch = None
        reason = "no deterministic Stage34 predicate is satisfied"
    return {
        "selected_branch": branch,
        "reason": reason,
        "ma_positive": ma_positive,
        "ma_mixed": ma_mixed,
        "mb_strong": mb_strong,
        "mc_qualified": mc_qualified,
        "wa_classification": wa.get("classification"),
        "next_gate_if_no_branch": NEXT_GATE,
    }


def run_aggregate(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "ma": args.output_root / "stage33_ma_result.json",
        "mb": args.output_root / "stage33_mb_result.json",
        "mc": args.output_root / "stage33_mc_result.json",
        "wa": args.output_root / "stage33_wa_result.json",
    }
    if any(not path.is_file() for path in paths.values()):
        raise Stage33Error("Stage33 aggregate requires complete MA/MB/MC/WA result files")
    results = {name: _read_json(path) for name, path in paths.items()}
    if any(result.get("status") != "PASS_EXECUTION_COMPLETE" for result in results.values()):
        raise Stage33Error("Stage33 subtrack result is not complete")
    decision = choose_stage34_branch(results["ma"], results["mb"], results["mc"], results["wa"])
    payload = {
        "schema": "STAGE33_PROGRAM_DECISION_V1",
        "status": "PASS_STAGE33_COMPLETE",
        "pi_gate": PI_GATE,
        "subtracks": {name: {"path": str(path), "sha256": _sha256_file(path)} for name, path in paths.items()},
        "decision": decision,
        "stage34_execution_started": False,
        "scientific_contract_changed": False,
        "next_gate": NEXT_GATE if decision["selected_branch"] is None else "STAGE34_BRANCH_" + str(decision["selected_branch"]),
    }
    _write_json_atomic(args.output_root / "stage33_program_decision.json", payload)
    return payload


def seal_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    result = _read_json(args.output_root / "stage33_program_decision.json")
    if result.get("status") != "PASS_STAGE33_COMPLETE":
        raise Stage33Error("cannot seal incomplete Stage33 campaign")
    seal_path = args.output_root / "stage33_artifact_seal.json"
    if seal_path.exists():
        raise Stage33Error("Stage33 artifact seal already exists")
    artifacts: list[dict[str, Any]] = []
    for path in sorted(args.output_root.rglob("*")):
        if not path.is_file() or path == seal_path or path.is_symlink():
            continue
        if path.name == LIVE_JOB_LOG_NAME:
            # The PBS wrapper keeps this live log open through the final
            # status print.  It is intentionally excluded; the wrapper
            # snapshots it to stage33_job.log.sealed immediately before this
            # mode, and that immutable snapshot is sealed below.
            continue
        artifacts.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    for path, role in (
        (Path(__file__).resolve(), "stage33_runner"),
        (args.pbs_template, "stage33_pbs_wrapper"),
    ):
        artifacts.append({"path": str(path.resolve(strict=True)), "role": role, "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    payload = {
        "schema": "STAGE33_ARTIFACT_SEAL_V1",
        "status": "PASS",
        "created_at_utc": _utc_now(),
        "pi_gate": PI_GATE,
        "pbs_job_id": args.pbs_job_id,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "historical_artifacts_modified": False,
        "scientific_contract_changed": False,
        "next_gate": result["next_gate"],
    }
    _write_json_atomic(seal_path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "preflight", "ma", "mb", "mc", "wa", "aggregate", "seal"), default="plan")
    parser.add_argument("--project-repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--expected-project-sha", default=BASE_COMMIT)
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_TINY1)
    parser.add_argument("--groot-source", type=Path, default=DEFAULT_GROOT)
    parser.add_argument("--modality-config", type=Path, default=DEFAULT_MODALITY)
    parser.add_argument("--cosmos-cache-root", type=Path, default=DEFAULT_COSMOS_CACHE)
    parser.add_argument("--cosmos-snapshot", type=Path, default=DEFAULT_COSMOS_SNAPSHOT)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--h2-training-root", type=Path, default=DEFAULT_H2_TRAINING_ROOT)
    parser.add_argument("--stage26-manifest", type=Path, default=DEFAULT_STAGE26_MANIFEST)
    parser.add_argument("--stage31-root", type=Path, default=DEFAULT_STAGE31_ROOT)
    parser.add_argument("--stage31-result", type=Path, default=DEFAULT_STAGE31_RESULT)
    parser.add_argument("--stage32-root", type=Path, default=DEFAULT_STAGE32_ROOT)
    parser.add_argument("--stage32-result", type=Path, default=DEFAULT_STAGE32_RESULT)
    parser.add_argument("--stage32-result-seal", type=Path, default=DEFAULT_STAGE32_ROOT / "stage32_result_seal.json")
    parser.add_argument("--stage32-salvage-result", type=Path, default=DEFAULT_STAGE32_SALVAGE_RESULT)
    parser.add_argument("--stage32-salvage-trace", type=Path, default=DEFAULT_STAGE32_SALVAGE_TRACE)
    parser.add_argument("--stage16-root", type=Path, default=DEFAULT_STAGE16_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_PARENT / "local_preflight")
    parser.add_argument("--pbs-template", type=Path, default=REPO_ROOT / "hpc" / "pbs" / "stage33_bottleneck_map.pbs")
    parser.add_argument("--authorization-token", default=PI_GATE)
    parser.add_argument("--pbs-job-id", default="PREFLIGHT_NO_PBS")
    parser.add_argument("--pbs-project", default=PROJECT)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "plan":
        assert_protocol_static()
        print(json.dumps(protocol_plan(), indent=2, sort_keys=True))
        return 0
    args.h2_checkpoint_1000 = args.h2_training_root / "h2_training" / "checkpoint-1000"
    args.h2_checkpoint_2000 = args.h2_training_root / "h2_training" / "checkpoint-2000"
    args.output_root.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == "preflight":
            result = run_preflight(args)
            _write_json_atomic(args.output_root / "stage33_preflight.json", result)
        elif args.mode == "ma":
            result = run_ma(args)
        elif args.mode == "mb":
            result = run_mb(args)
        elif args.mode == "mc":
            result = run_mc(args)
        elif args.mode == "wa":
            result = run_wa(args)
        elif args.mode == "aggregate":
            result = run_aggregate(args)
        else:
            result = seal_artifacts(args)
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "status": "BLOCKED",
            "error_type": type(exc).__name__,
            "error_summary": str(exc).splitlines()[0][:2000],
            "scientific_execution_begun": bool((args.output_root / "stage33_ma_execution_state.json").exists() or (args.output_root / "stage33_mb_execution_state.json").exists()),
            "scientific_contract_changed": False,
            "autonomous_retry_authorized": False,
            "next_gate": NEXT_GATE,
        }
        try:
            _write_json_atomic(args.output_root / "stage33_failure.json", failure)
        except Exception:
            pass
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
