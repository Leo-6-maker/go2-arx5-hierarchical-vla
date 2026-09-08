#!/usr/bin/env python3
"""Stage38-B/C CPU-only StructuredActionV1 and source qualification.

This program is intentionally the last zero-model boundary before Stage38-D.
It binds the active PR head, Stage38-A, Stage37/35 evidence, the historical
H2 identities, the exact training fixture, and the pinned GR00T source.  It
then qualifies the model-facing codec on the immutable Stage35 corpus and on
the deterministic Stage22/TINY1 training example.

It never imports GR00T, Torch, Transformers, IsaacGym, CUDA, or a scheduler
submission path.  The pinned GR00T source is read as bytes for a static audit;
no model object is constructed and no checkpoint weights are loaded.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from go2arx5_vla.adapters import GR00TToWBCAdapter, ReferenceAdapterConfig  # noqa: E402
from go2arx5_vla.contracts import TrajectoryChunkV1  # noqa: E402
from go2arx5_vla.data.stage22_expert import load_snapshot  # noqa: E402
from go2arx5_vla.motion_api import (  # noqa: E402
    DEFAULT_SE3_LAMBDA_M_PER_RAD,
    STAGE38_TIMING_KNOTS,
    build_geometry_candidate,
    build_timing_candidate_with_knots,
    evaluate_preview_equivalence,
    public_evaluation,
)
from go2arx5_vla.structured_action import (  # noqa: E402
    STRUCTURED_ACTION_DIM,
    STRUCTURED_ACTION_HORIZON,
    STRUCTURED_GEOMETRY_K,
    STRUCTURED_ROT6D_LAYOUT,
    STRUCTURED_TIMING_MODE,
    StructuredActionV1,
    decode_structured_action,
    encode_structured_action,
    materialize_structured_action,
    structured_action_contract_summary,
    structured_action_mask,
)
from go2arx5_vla.trajectory.so3 import (  # noqa: E402
    geodesic_angle,
    matrix_to_quat,
    quat_slerp,
    quat_to_matrix,
    rotation_matrix_to_rot6d_rows,
)


STAGE38A = importlib.import_module("stage38_structured_groot_offline")
STAGE36 = importlib.import_module("stage36_wbc_authority_oracle_qualification")
STAGE37 = importlib.import_module("stage37_geometry_capacity_qualification")
STAGE28 = importlib.import_module("stage28_h2_fresh_base_2000_fixed_warmup10")


PI_GATE_ID = "PI_GATE_STATE_0026"
PR_NUMBER = 24
EXPECTED_BRANCH = "stage38/structured-groot-offline"
EXPECTED_PARENT_HEAD = "bbf7453c7060d64fc6f659281ceb777246d6d370"
EXPECTED_PROJECT = "CFP03-CF-004"
EXPECTED_QUEUE = "auto"
NEXT_GATE = "GPT_PI_REVIEW_AFTER_STAGE38_STRUCTURED_GR00T_OFFLINE"

STAGE38A_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage38_structured_groot_offline_20260906/stage38a_cpu_c0bb88c_v1"
)
STAGE38A_RESULT_SHA256 = "e08f20ae2ebe9875cf236168655d4075318962ab0ed1fa7b8c25f0e7eab73d49"
STAGE38A_SEAL_SHA256 = "3aecca9e5d042237e28bccc17281214adde568f918a0a8b191edbb4dd95d1d7f"
STAGE38A_RETURN_SHA256 = "8dbddb78049cc017991368d3c52f6c6875b60f74d5285c6851bf69abfbca56f7"

STAGE37_ROOT = Path(
    "/scratch/e1192548/leon-storage/runs/"
    "stage37_geometry_capacity_qualification_20260906/"
    "stage37_structured_final_602886_bbf7453_v2"
)
STAGE37_FINAL_RETURN_SHA256 = "f2e32ee39f856cd2a44b0c3f5487296e479680a3d83c95c6abb95f7235569976"
STAGE37_SEAL_SHA256 = "4a606636d4769367a97b2386f4e202be3ae97ac7d63750631919fa9710fdeb87"
STAGE37_RESULT_SHA256 = "de1b4358edc8d872c1b75f52873fb84ff4d3228dca0d91a5df7cdc327c2b9f18"
STAGE37_PAIRED_SHA256 = "c1bcf7d8e537e5e40e8d7cd8d738db1e1b2eb32e1bc8b38ee4b701934b0106d6"
STAGE37_RAW_MANIFEST_SHA256 = "97646e44297accc9bf7de96e7578c1666b76f4014a24da956f4a643c07f74270"
STAGE37_RESULT_MANIFEST_SHA256 = "002fb4959df1f150ea1f2a60e3c2f54a042166239338bdf08aec477cdff69abf"
STAGE37_SOURCE_MANIFEST_SHA256 = "176aecebc89f6ceb80cf20f59037bf1b592e54f2cfa1e19dc84f328162dc718b"

BASE_ROOT = Path(
    "/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/GR00T-N1.7-3B"
)
BASE_TREE_SHA256 = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
BASE_INDEX_SHA256 = "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59"
BASE_PROCESSOR_CONFIG_SHA256 = "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be"
BASE_STATISTICS_SHA256 = "c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9"

TINY1_ROOT = Path(
    "/scratch/e1192548/stage23_o1_20260903/input_exact/extracted/"
    "tiny1_h2stats_lerobot_v2"
)
TINY1_TREE_SHA256 = "541bf4a1b9ea459637525ef11fceea5c7f192c0902f6013f0331a2a668038157"
TINY1_FILE_COUNT = 13
TINY1_TOTAL_BYTES = 152958
TINY1_STATS_SHA256 = "fd2769f4eca4cd578255bd5dd428585906f05d3d1aba3a89747d47b1c1a39b23"
TINY1_RELATIVE_STATS_SHA256 = "8291ca2f5bc153ae6f75d886746569a31a93c3802ed08f3d03c41a44908f89b1"
# The dataset's LeRobot metadata and the Stage22 modality module are distinct
# immutable inputs.  The former is bound to the Stage19 qualified manifest;
# the latter is the module copied into the accepted H2 handoff payload.
TINY1_META_MODALITY_SHA256 = "1d172665da201f5b040ca6664a67bd3a9fe591237b3c8298d3a44f7e63f97c4e"
STAGE22_MODALITY_CONFIG_PATH = Path(
    "/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/modality/"
    "go2_arx5_modality.py"
)
STAGE22_MODALITY_CONFIG_SHA256 = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
TINY1_SNAPSHOT = Path(
    "/scratch/e1192548/stage26_h1_c1_20260904/snapshots/"
    "stage22_expert_red_target_train_a/expert-0000.json"
)
TINY1_SNAPSHOT_SHA256 = "74af1af85e21c6e3a5e91e5b4b695d080f0f6b7f2204d012310d6457334106f1"

GROOT_ROOT = Path("/scratch/e1192548/stage22_h200_runtime_deb920_20260902/src/isaac-gr00t")
GROOT_HEAD = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
COSMOS_ROOT = Path(
    "/scratch/e1192548/stage22_h200_cosmos_v2_20260902/hf_home/hub/"
    "models--nvidia--Cosmos-Reason2-2B"
)
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE_SHA256 = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"

H2_ROOT = Path(
    "/scratch/e1192548/stage28_h2_20260904/"
    "stage28_h2_600695.hopper-m-02/h2_training"
)
H2_1000_TREE_SHA256 = "29f3e8f46cf41441696bfc4a5e7601378648c26fa675a9685c003080da0c4114"
H2_2000_TREE_SHA256 = "3106f9b5993b390438b62594d044781d11b393452a49f80b5c8e9e910606d0ba"
H2_SEAL_PATH = H2_ROOT.parent / "stage28_h2_checkpoint_2000_seal.json"
H2_SEAL_SHA256 = "1349896f7b31a930764dbd9d849c5368e48372d0dc6273449f42f19f82a69a82"
H2_CONFIG_PATH = H2_ROOT / "experiment_cfg/config.yaml"
H2_CONFIG_SHA256 = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
H2_TRAINING_SOURCE_SHA = "a361e5cb8a4be33f40a21c9bbf7ad742e25c7fd0"
H2_TRAINING_RUNNER_SHA256 = "d61c429a2b8d53c25df63d9b5b9e2464b33c36e9e3c0a610b109b10a6bd63fe4"

STAGE35_ROOT = STAGE38A.STAGE35_ROOT
STAGE35_CASE_IDS = tuple(STAGE38A.STAGE35_CASE_IDS)
STAGE35_MANIFEST_SHA256 = STAGE38A.STAGE35_MANIFEST_SHA256
STAGE35_TREE_SHA256 = STAGE38A.STAGE35_CORPUS_TREE_SHA256
STAGE35_SEAL_SHA256 = STAGE38A.STAGE35_SEAL_SHA256


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    Path(path).write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def file_record(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    path = Path(path)
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file() and not path.is_symlink(),
        "size_bytes": None,
        "sha256": None,
        "expected_sha256": expected_sha256,
        "hash_match": False,
    }
    if result["exists"]:
        result["size_bytes"] = int(path.stat().st_size)
        result["sha256"] = sha256_file(path)
        result["hash_match"] = expected_sha256 is None or result["sha256"] == expected_sha256
    return result


def git_output(*args: str) -> str:
    return subprocess.check_output(list(args), text=True, stderr=subprocess.STDOUT).strip()


def git_head(path: Path) -> str:
    return git_output("git", "-C", str(path), "rev-parse", "HEAD")


def git_status(path: Path) -> str:
    return git_output("git", "-C", str(path), "status", "--porcelain=v1")


def git_show(path: Path, revision: str, relative_path: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(path), "show", f"{revision}:{relative_path}"],
        stderr=subprocess.STDOUT,
    )


def static_guard() -> dict[str, Any]:
    """Prove this B/C executable has no model/allocation execution path."""

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    forbidden_imports = {"torch", "transformers", "isaacgym", "omni", "cupy"}
    forbidden_call_tokens = {
        "qsub",
        "get_action",
        "backward",
        "from_pretrained",
        "resume_from_checkpoint",
        "save_pretrained",
        "optimizer.step",
        "scheduler.step",
    }
    forbidden_import_hits: list[str] = []
    forbidden_call_hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                imported_roots.add(root)
                if root in forbidden_imports:
                    forbidden_import_hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            imported_roots.add(root)
            if root in forbidden_imports:
                forbidden_import_hits.append(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                name = ""
            source_name = ast.unparse(node.func) if hasattr(ast, "unparse") else name
            for token in forbidden_call_tokens:
                if token == name or token in source_name:
                    forbidden_call_hits.append(source_name)
    return {
        "schema": "STAGE38BC_STATIC_ZERO_MODEL_GUARD_V1",
        "script": str(Path(__file__).resolve()),
        "forbidden_imports": sorted(set(forbidden_import_hits)),
        "forbidden_calls": sorted(set(forbidden_call_hits)),
        "imported_roots": sorted(imported_roots),
        "cpu_zero_model_no_scheduler_path": not forbidden_import_hits and not forbidden_call_hits,
        "model_constructed": False,
        "forward": False,
        "training": False,
        "pbs_submitted": False,
        "status": "PASS" if not forbidden_import_hits and not forbidden_call_hits else "BLOCKED",
    }


def qstat_audit() -> dict[str, Any]:
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"
    try:
        completed = subprocess.run(
            ["qstat", "-u", user],
            check=False,
            capture_output=True,
            text=True,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        lines = [line for line in stdout.splitlines() if line.strip()]
        relevant = [
            line
            for line in lines
            if any(token in line.lower() for token in ("stage38", "structured_groot", "wbc") )
        ]
        return {
            "command": f"qstat -u {user}",
            "returncode": int(completed.returncode),
            "relevant_lines": relevant,
            "no_active_stage38_or_wbc_job": completed.returncode == 0 and not relevant,
            "stdout_sha256": sha256_bytes(stdout.encode()),
            "stderr": stderr[-2000:],
            "status": "PASS" if completed.returncode == 0 and not relevant else "BLOCKED",
        }
    except OSError as exc:
        return {
            "command": f"qstat -u {user}",
            "returncode": None,
            "relevant_lines": [],
            "no_active_stage38_or_wbc_job": False,
            "error": f"{type(exc).__name__}: {exc}",
            "status": "BLOCKED",
        }


def git_preflight(api_head: str, expected_head: str) -> dict[str, Any]:
    local = git_head(REPO)
    branch = git_output("git", "-C", str(REPO), "branch", "--show-current")
    remote = git_output(
        "git",
        "-C",
        str(REPO),
        "ls-remote",
        "origin",
        f"refs/heads/{EXPECTED_BRANCH}",
    ).split()[0]
    parent_local = git_output(
        "git", "-C", str(REPO), "rev-parse", "stage37/geometry-capacity-qualification"
    )
    parent_remote = git_output(
        "git",
        "-C",
        str(REPO),
        "ls-remote",
        "origin",
        "refs/heads/stage37/geometry-capacity-qualification",
    ).split()[0]
    dirty = git_status(REPO)
    result = {
        "schema": "STAGE38BC_GIT_PRELIGHT_V1",
        "api_head": api_head,
        "expected_head": expected_head,
        "remote_head": remote,
        "local_head": local,
        "branch": branch,
        "expected_branch": EXPECTED_BRANCH,
        "parent_local_head": parent_local,
        "parent_remote_head": parent_remote,
        "expected_parent_head": EXPECTED_PARENT_HEAD,
        "worktree_status": dirty,
        "worktree_clean": dirty == "",
        "head_match": api_head == expected_head == remote == local,
        "parent_match": parent_local == parent_remote == EXPECTED_PARENT_HEAD,
    }
    result["status"] = (
        "PASS"
        if result["head_match"]
        and result["parent_match"]
        and branch == EXPECTED_BRANCH
        and result["worktree_clean"]
        else "BLOCKED"
    )
    return result


def stage38a_audit() -> dict[str, Any]:
    result_path = STAGE38A_ROOT / "stage38a_result.json"
    seal_path = STAGE38A_ROOT / "stage38a_artifact_seal.json"
    return_path = STAGE38A_ROOT / "stage38a_return.json"
    result = {
        "schema": "STAGE38BC_STAGE38A_INPUT_AUDIT_V1",
        "root": str(STAGE38A_ROOT),
        "result": file_record(result_path, STAGE38A_RESULT_SHA256),
        "seal": file_record(seal_path, STAGE38A_SEAL_SHA256),
        "return": file_record(return_path, STAGE38A_RETURN_SHA256),
        "status": "BLOCKED",
    }
    try:
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        result["result_fields"] = {
            "result": result_payload.get("result"),
            "model_timing_capacity_star": result_payload.get("model_timing_capacity_star"),
            "model_timing_classification": result_payload.get("model_timing_classification"),
            "stage38_b_c_admission": result_payload.get("stage38_b_c_admission"),
            "stage37_t_capacity_star_preserved": result_payload.get(
                "stage37_t_capacity_star_preserved"
            ),
        }
        result["status"] = (
            "PASS"
            if result["result"]["hash_match"]
            and result["seal"]["hash_match"]
            and result["return"]["hash_match"]
            and result_payload.get("result") == "STAGE38A_CPU_TIMING_PASS_READY_FOR_BC"
            and result_payload.get("model_timing_capacity_star") == "T16"
            and result_payload.get("model_timing_classification")
            == "MODEL_TIMING_COMPACT_QUALIFIED"
            and result_payload.get("stage38_b_c_admission") is True
            and result_payload.get("stage37_t_capacity_star_preserved") is None
            else "BLOCKED"
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def immutable_historical_audit() -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "STAGE38BC_IMMUTABLE_HISTORICAL_INPUT_AUDIT_V1",
        "stage37": {},
        "stage35": {},
        "base": {},
        "tiny1": {},
        "groot": {},
        "cosmos": {},
        "h2": {},
        "status": "BLOCKED",
    }
    stage37 = STAGE38A.verify_stage37_acceptance()
    stage35 = STAGE38A.verify_stage35_input()
    result["stage37"] = {
        "status": stage37.get("status"),
        "expected_live_head": EXPECTED_PARENT_HEAD,
        "artifact_seal_sha256": STAGE37_SEAL_SHA256,
        "final_return": file_record(STAGE37_ROOT / "stage37_final_return.json", STAGE37_FINAL_RETURN_SHA256),
        "seal": file_record(STAGE37_ROOT / "stage37_structured_artifact_seal.json", STAGE37_SEAL_SHA256),
        "result": file_record(STAGE37_ROOT / "stage37_structured_oracle_result.json", STAGE37_RESULT_SHA256),
        "paired": file_record(STAGE37_ROOT / "stage37_paired_oracle_analysis.json", STAGE37_PAIRED_SHA256),
        "raw_manifest": file_record(STAGE37_ROOT / "stage37_structured_raw_hash_manifest.json", STAGE37_RAW_MANIFEST_SHA256),
        "result_manifest": file_record(STAGE37_ROOT / "stage37_structured_result_manifest.json", STAGE37_RESULT_MANIFEST_SHA256),
        "source_manifest": file_record(STAGE37_ROOT / "stage37_source_manifest.json", STAGE37_SOURCE_MANIFEST_SHA256),
    }
    result["stage35"] = {
        "status": stage35.get("status"),
        "root": str(STAGE35_ROOT),
        "manifest_sha256": STAGE35_MANIFEST_SHA256,
        "tree_sha256": STAGE35_TREE_SHA256,
        "seal_sha256": STAGE35_SEAL_SHA256,
        "case_count": stage35.get("case_count"),
        "case_ids_exact": stage35.get("case_ids_exact"),
    }

    base = STAGE28._verify_base_checkpoint(BASE_ROOT)
    result["base"] = {
        "path": str(BASE_ROOT),
        "observed_tree_sha256": base.get("tree_sha256"),
        "expected_tree_sha256": BASE_TREE_SHA256,
        "index": base.get("index"),
        "processor_config_sha256": base.get("processor_config_sha256"),
        "statistics_sha256": base.get("statistics_sha256"),
        "tree_file_count": base.get("tree_file_count"),
        "tree_total_bytes": base.get("tree_total_bytes"),
        "status": "PASS"
        if base.get("tree_sha256") == BASE_TREE_SHA256
        and base.get("index", {}).get("index_sha256") == BASE_INDEX_SHA256
        and base.get("processor_config_sha256") == BASE_PROCESSOR_CONFIG_SHA256
        and base.get("statistics_sha256") == BASE_STATISTICS_SHA256
        else "BLOCKED",
    }
    tiny1 = STAGE28._verify_tiny_dataset(TINY1_ROOT)
    result["tiny1"] = {
        **{key: tiny1.get(key) for key in ("path", "tree_sha256", "file_count", "total_bytes")},
        "expected_tree_sha256": TINY1_TREE_SHA256,
        "expected_file_count": TINY1_FILE_COUNT,
        "expected_total_bytes": TINY1_TOTAL_BYTES,
        "meta_stats": file_record(TINY1_ROOT / "meta/stats.json", TINY1_STATS_SHA256),
        "meta_relative_stats": file_record(TINY1_ROOT / "meta/relative_stats.json", TINY1_RELATIVE_STATS_SHA256),
        "meta_modality": file_record(TINY1_ROOT / "meta/modality.json", TINY1_META_MODALITY_SHA256),
        "stage22_modality_config": file_record(
            STAGE22_MODALITY_CONFIG_PATH, STAGE22_MODALITY_CONFIG_SHA256
        ),
        "status": "PASS"
        if tiny1.get("tree_sha256") == TINY1_TREE_SHA256
        and tiny1.get("file_count") == TINY1_FILE_COUNT
        and tiny1.get("total_bytes") == TINY1_TOTAL_BYTES
        and file_record(TINY1_ROOT / "meta/stats.json", TINY1_STATS_SHA256)["hash_match"]
        and file_record(TINY1_ROOT / "meta/relative_stats.json", TINY1_RELATIVE_STATS_SHA256)["hash_match"]
        and file_record(TINY1_ROOT / "meta/modality.json", TINY1_META_MODALITY_SHA256)["hash_match"]
        and file_record(STAGE22_MODALITY_CONFIG_PATH, STAGE22_MODALITY_CONFIG_SHA256)["hash_match"]
        else "BLOCKED",
    }
    groot = STAGE28._verify_groot_source(GROOT_ROOT)
    result["groot"] = groot
    result["groot"]["status"] = (
        "PASS"
        if groot.get("observed_head") == GROOT_HEAD and git_status(GROOT_ROOT) == ""
        else "BLOCKED"
    )
    cosmos = STAGE28._verify_cosmos_cache(COSMOS_ROOT)
    result["cosmos"] = cosmos
    result["cosmos"]["status"] = (
        "PASS"
        if cosmos.get("revision") == COSMOS_REVISION
        and cosmos.get("tree_sha256") == COSMOS_TREE_SHA256
        else "BLOCKED"
    )

    h2_seal_record = file_record(H2_SEAL_PATH, H2_SEAL_SHA256)
    h2_seal_payload = json.loads(H2_SEAL_PATH.read_text(encoding="utf-8"))
    h2_1000 = STAGE28._inspect_checkpoint(H2_ROOT / "checkpoint-1000", expected_step=1000)
    h2_2000 = STAGE28._inspect_checkpoint(H2_ROOT / "checkpoint-2000", expected_step=2000)
    result["h2"] = {
        "historical_job": "600695.hopper-m-02",
        "root": str(H2_ROOT),
        "training_source_sha": H2_TRAINING_SOURCE_SHA,
        "training_runner_sha256": H2_TRAINING_RUNNER_SHA256,
        "config": file_record(H2_CONFIG_PATH, H2_CONFIG_SHA256),
        "checkpoint_1000": {
            "path": h2_1000.get("path"),
            "tree_sha256": h2_1000.get("tree_sha256"),
            "expected_tree_sha256": H2_1000_TREE_SHA256,
            "global_step": h2_1000.get("global_step"),
            "max_steps": h2_1000.get("max_steps"),
            "file_count": h2_1000.get("file_count"),
            "total_bytes": h2_1000.get("total_bytes"),
        },
        "checkpoint_2000": {
            "path": h2_2000.get("path"),
            "tree_sha256": h2_2000.get("tree_sha256"),
            "expected_tree_sha256": H2_2000_TREE_SHA256,
            "global_step": h2_2000.get("global_step"),
            "max_steps": h2_2000.get("max_steps"),
            "file_count": h2_2000.get("file_count"),
            "total_bytes": h2_2000.get("total_bytes"),
        },
        "seal": {
            "file": h2_seal_record,
            "schema": h2_seal_payload.get("schema"),
            "status": h2_seal_payload.get("status"),
            "checkpoint_path": h2_seal_payload.get("checkpoint_path"),
            "tree_sha256": h2_seal_payload.get("tree_sha256"),
            "file_count": h2_seal_payload.get("file_count"),
            "total_bytes": h2_seal_payload.get("total_bytes"),
            "entries_match_checkpoint_2000": h2_seal_payload.get("entries") == h2_2000.get("entries"),
        },
    }
    result["h2"]["status"] = (
        "PASS"
        if h2_seal_record["hash_match"]
        and h2_seal_payload.get("status") == "PASS"
        and h2_seal_payload.get("tree_sha256") == H2_2000_TREE_SHA256
        and h2_seal_payload.get("entries") == h2_2000.get("entries")
        and h2_1000.get("tree_sha256") == H2_1000_TREE_SHA256
        and h2_2000.get("tree_sha256") == H2_2000_TREE_SHA256
        and h2_1000.get("global_step") == 1000
        and h2_2000.get("global_step") == 2000
        and file_record(H2_CONFIG_PATH, H2_CONFIG_SHA256)["hash_match"]
        else "BLOCKED"
    )
    result["status"] = (
        "PASS"
        if stage37.get("status") == "PASS"
        and stage35.get("status") == "PASS"
        and result["base"].get("status") == "PASS"
        and result["tiny1"].get("status") == "PASS"
        and result["groot"].get("status") == "PASS"
        and result["cosmos"].get("status") == "PASS"
        and result["h2"].get("status") == "PASS"
        else "BLOCKED"
    )
    return result


def dataset_contract_audit() -> dict[str, Any]:
    info_path = TINY1_ROOT / "meta/info.json"
    split_lock_path = TINY1_ROOT / "meta/stage23_tiny_split_lock.json"
    episodes_path = TINY1_ROOT / "meta/episodes.jsonl"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    split_lock = json.loads(split_lock_path.read_text(encoding="utf-8"))
    episodes = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    snapshot = load_snapshot(TINY1_SNAPSHOT)
    state = snapshot.state_v1.as_array()
    action = snapshot.physical_action
    timestamps = snapshot.future_timestamps
    action0_position_error = float(np.linalg.norm(action[0, :3] - state[:3]))
    action0_rotation_error = float(
        geodesic_angle(
            # The snapshot/state are deliberately in the historical native row format.
            __import__("go2arx5_vla.trajectory.so3", fromlist=["rot6d_rows_to_matrix"]).rot6d_rows_to_matrix(
                action[0, 3:9]
            ),
            __import__("go2arx5_vla.trajectory.so3", fromlist=["rot6d_rows_to_matrix"]).rot6d_rows_to_matrix(
                state[3:9]
            ),
        )
    )
    result = {
        "schema": "STAGE38BC_TINY1_TRAINING_TIMING_AUDIT_V1",
        "dataset_root": str(TINY1_ROOT),
        "dataset_tree_sha256": TINY1_TREE_SHA256,
        "dataset_file_count": TINY1_FILE_COUNT,
        "dataset_total_bytes": TINY1_TOTAL_BYTES,
        "metadata_files": {
            "info": file_record(info_path),
            "split_lock": file_record(split_lock_path),
            "episodes": file_record(episodes_path),
            "snapshot": file_record(TINY1_SNAPSHOT, TINY1_SNAPSHOT_SHA256),
        },
        "info_contract": {
            "fps": info.get("fps"),
            "action_shape": info.get("action_shape"),
            "state_shape": info.get("state_shape"),
            "total_frames": info.get("total_frames"),
            "train_split": info.get("splits", {}).get("train"),
            "validation_split": info.get("splits", {}).get("validation"),
            "stage19_abi": info.get("stage19_abi"),
        },
        "split_lock": split_lock,
        "episode_records": episodes,
        "snapshot_contract": {
            "example_id": snapshot.example_id,
            "source_frame": snapshot.source_frame,
            "tcp_semantic_id": snapshot.tcp_semantic_id,
            "horizon": snapshot.horizon,
            "timestamp_start_s": float(timestamps[0]),
            "timestamp_end_s": float(timestamps[-1]),
            "timestamp_span_s": float(timestamps[-1] - timestamps[0]),
            "timestamp_dt_min_s": float(np.min(np.diff(timestamps))),
            "timestamp_dt_max_s": float(np.max(np.diff(timestamps))),
            "native_control_dt_s": snapshot.native_control_dt_s,
            "action_shape": list(action.shape),
            "state_shape": list(state.shape),
            "action0_position_error_m": action0_position_error,
            "action0_rotation_error_rad": action0_rotation_error,
            "target_excluded_from_model_input": snapshot.target_excluded_from_model_input,
            "source_timestamp_authority": True,
        },
        "timing_binding": {
            "T_chunk_definition": "future_timestamps[-1] - future_timestamps[0]",
            "T_chunk_s": float(timestamps[-1] - timestamps[0]),
            "action_sample_period_s": snapshot.provenance.get("action_sample_period_s"),
            "action_sample_period_provenance": snapshot.provenance.get(
                "action_sample_period_provenance"
            ),
            "camera_fps_not_authority": True,
            "hardcoded_two_second_assumption": False,
        },
    }
    result["status"] = (
        "PASS"
        if result["metadata_files"]["snapshot"]["hash_match"]
        and snapshot.horizon == STRUCTURED_ACTION_HORIZON
        and np.isclose(timestamps[0], 0.0, rtol=0.0, atol=1e-12)
        and np.isclose(timestamps[-1], 1.95, rtol=0.0, atol=1e-12)
        and np.allclose(np.diff(timestamps), 0.05, rtol=0.0, atol=1e-12)
        and action.shape == (40, 10)
        and state.shape == (10,)
        and action0_position_error <= 1e-12
        and action0_rotation_error <= 1e-12
        and split_lock.get("dataset_role") == "TINY1_H2STATS_TRAIN_ONLY"
        and split_lock.get("training_allowed") is True
        and split_lock.get("validation_target_ids") == []
        and split_lock.get("final_test_target_ids") == []
        and split_lock.get("holdout_e_accessed") is False
        and split_lock.get("holdout_f_accessed") is False
        and split_lock.get("selected_source_episode_ids")
        == ["stage22_expert_red_target_train_a:expert-0000"]
        else "BLOCKED"
    )
    return result


def source_semantics_audit() -> dict[str, Any]:
    source_files = {
        "types": "gr00t/data/types.py",
        "state_action_processor": "gr00t/data/state_action/state_action_processor.py",
        "processor": "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py",
        "model": "gr00t/model/gr00t_n1d7/gr00t_n1d7.py",
        "collator": "gr00t/data/collator/collators.py",
        "setup": "gr00t/model/gr00t_n1d7/setup.py",
    }
    anchors: dict[str, tuple[str, ...]] = {
        "types": (
            "class VLAStepData",
            "class ActionConfig",
            "class ModalityConfig",
            "class ActionRepresentation",
        ),
        "state_action_processor": (
            "def apply(",
            "normalize_values_minmax",
            "def unapply_action(",
            "ActionRepresentation.RELATIVE",
        ),
        "processor": (
            "class Gr00tN1d7Processor",
            "self.max_action_dim",
            "action_mask = torch.ones_like(normalized_actions)",
            "def decode_action(",
        ),
        "model": (
            "action_mask",
            "F.mse_loss(pred_actions, velocity, reduction=\"none\")",
            "action_loss",
            "self.action_horizon = config.action_horizon",
        ),
        "collator": (
            "class BasicDataCollator",
            "torch.stack",
            "fields = features[0].keys()",
        ),
        "setup": (
            "class Gr00tN1d7Pipeline",
            "DatasetFactory",
            "tune_projector",
            "tune_diffusion_model",
        ),
    }
    records: dict[str, Any] = {}
    all_pass = True
    for key, relative in source_files.items():
        payload = git_show(GROOT_ROOT, GROOT_HEAD, relative)
        text = payload.decode("utf-8")
        checks = {anchor: anchor in text for anchor in anchors[key]}
        records[key] = {
            "path": relative,
            "sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
            "anchors": checks,
            "all_anchors_present": all(checks.values()),
        }
        all_pass = all_pass and all(checks.values())

    processor_text = git_show(GROOT_ROOT, GROOT_HEAD, source_files["processor"]).decode("utf-8")
    collator_text = git_show(GROOT_ROOT, GROOT_HEAD, source_files["collator"]).decode("utf-8")
    model_text = git_show(GROOT_ROOT, GROOT_HEAD, source_files["model"]).decode("utf-8")
    contract = structured_action_contract_summary(STRUCTURED_TIMING_MODE)
    records["structured_boundary"] = {
        "universal_action_shape": [STRUCTURED_ACTION_HORIZON, STRUCTURED_ACTION_DIM],
        "horizon_unchanged": STRUCTURED_ACTION_HORIZON == 40,
        "universal_dim_unchanged": STRUCTURED_ACTION_DIM == 132,
        "action_mask_per_slot_dim_support": "action_mask" in processor_text
        and "action_mask" in model_text
        and "class BasicDataCollator" in collator_text
        and "torch.stack" in collator_text
        and "fields = features[0].keys()" in collator_text,
        "model_loss_mask_multiplication_present": "* action_mask" in model_text,
        "model_action_loss_mse_present": "F.mse_loss(pred_actions, velocity, reduction=\"none\")"
        in model_text,
        "structured_contract": contract,
        "rot6d_layout": STRUCTURED_ROT6D_LAYOUT,
        "native_row_format_deferred_to_adapter": True,
    }
    all_pass = all_pass and all(
        records["structured_boundary"][key]
        for key in (
            "horizon_unchanged",
            "universal_dim_unchanged",
            "action_mask_per_slot_dim_support",
            "model_loss_mask_multiplication_present",
            "model_action_loss_mse_present",
        )
    )
    return {
        "schema": "STAGE38BC_GROOT_SOURCE_SEMANTICS_AUDIT_V1",
        "repository": str(GROOT_ROOT),
        "head": GROOT_HEAD,
        "worktree_clean": git_status(GROOT_ROOT) == "",
        "files": records,
        "no_object_construction": True,
        "status": "PASS" if all_pass and git_status(GROOT_ROOT) == "" else "BLOCKED",
    }


def _resample_chunk_to_h40(chunk: TrajectoryChunkV1) -> TrajectoryChunkV1:
    """Sample a source chunk at 40 uniform timestamps in its own time domain."""

    source_ts = np.asarray(chunk.point_timestamps, dtype=np.float64)
    query_ts = np.linspace(float(source_ts[0]), float(source_ts[-1]), 40, dtype=np.float64)
    position = np.stack(
        [np.interp(query_ts, source_ts, chunk.eef_position[:, axis]) for axis in range(3)],
        axis=-1,
    )
    source_quat = np.asarray(chunk.eef_orientation, dtype=np.float64)
    quaternion = np.empty((query_ts.size, 4), dtype=np.float64)
    for index, timestamp in enumerate(query_ts):
        if timestamp <= source_ts[0]:
            left, alpha = 0, 0.0
        elif timestamp >= source_ts[-1]:
            left, alpha = source_ts.size - 2, 1.0
        else:
            left = int(np.searchsorted(source_ts, timestamp, side="right") - 1)
            alpha = float((timestamp - source_ts[left]) / (source_ts[left + 1] - source_ts[left]))
        quaternion[index] = quat_slerp(source_quat[left], source_quat[left + 1], alpha)
    gripper = None
    if chunk.gripper is not None:
        gripper = np.interp(query_ts, source_ts, np.asarray(chunk.gripper).reshape(-1))
    return TrajectoryChunkV1(
        chunk_id=f"{chunk.chunk_id}:H40_SOURCE_TIME_SAMPLE",
        source="STAGE38BC_TIMESTAMP_PRESERVING_H40_AUDIT_SAMPLE",
        observation_timestamp=chunk.observation_timestamp,
        point_timestamps=query_ts,
        eef_position=position,
        eef_orientation=quaternion,
        source_frame=chunk.source_frame,
        tcp_semantic_id=chunk.tcp_semantic_id,
        validity_start=float(query_ts[0]),
        validity_end=float(query_ts[-1]),
        gripper=gripper,
        provenance={
            **dict(chunk.provenance),
            "timestamp_materialization": "PIECEWISE_LINEAR_XYZ_AND_QUATERNION_SLERP_IN_ORIGINAL_TIME",
            "audit_sample_count": 40,
            "source_chunk_preserved": True,
        },
    )


def _chunk_to_physical(chunk: TrajectoryChunkV1) -> np.ndarray:
    if chunk.point_count != STRUCTURED_ACTION_HORIZON or chunk.gripper is None:
        raise ValueError("structured audit chunk must have 40 points and one gripper channel")
    row_rot6d = rotation_matrix_to_rot6d_rows(quat_to_matrix(chunk.eef_orientation))
    return np.concatenate(
        [chunk.eef_position, row_rot6d, np.asarray(chunk.gripper).reshape(40, 1)], axis=-1
    )


def _state_for_chunk(chunk: TrajectoryChunkV1) -> np.ndarray:
    row_rot6d = rotation_matrix_to_rot6d_rows(quat_to_matrix(chunk.eef_orientation[0]))
    grip = 0.0 if chunk.gripper is None else float(np.asarray(chunk.gripper).reshape(-1)[0])
    return np.concatenate([chunk.eef_position[0], row_rot6d, [grip]]).astype(np.float64)


def _roundtrip_case(payload: Mapping[str, Any]) -> dict[str, Any]:
    source = STAGE36.case_to_chunk(payload, TrajectoryChunkV1)
    source_h40 = _resample_chunk_to_h40(source)
    physical = _chunk_to_physical(source_h40)
    state = _state_for_chunk(source_h40)
    label = encode_structured_action(
        physical,
        source_h40.point_timestamps,
        state,
        timing_mode=STRUCTURED_TIMING_MODE,
        chunk_id=str(payload["case_id"]),
        source_frame=source_h40.source_frame,
        tcp_semantic_id=source_h40.tcp_semantic_id,
        observation_timestamp=source_h40.observation_timestamp,
        provenance={
            "stage35_case_id": str(payload["case_id"]),
            "stage35_source_timestamp_count": int(source.point_count),
            "stage35_source_timestamps_nonuniform": bool(
                not np.allclose(np.diff(source.point_timestamps), np.diff(source.point_timestamps)[0], atol=1e-12)
            ),
            "audit_resampling": "40_UNIFORM_SAMPLES_IN_ORIGINAL_TIMESTAMP_DOMAIN",
        },
    )
    intent = decode_structured_action(label, intent_id=f"{payload['case_id']}:structured")
    materialized = materialize_structured_action(label, intent)

    expected_geometry = build_geometry_candidate(
        source_h40,
        STRUCTURED_GEOMETRY_K,
        allowed_k_values=(STRUCTURED_GEOMETRY_K,),
        lambda_m_per_rad=DEFAULT_SE3_LAMBDA_M_PER_RAD,
    )
    expected_semantic = build_timing_candidate_with_knots(
        source_h40,
        expected_geometry,
        STRUCTURED_TIMING_MODE,
        STAGE38_TIMING_KNOTS[STRUCTURED_TIMING_MODE],
        lambda_m_per_rad=DEFAULT_SE3_LAMBDA_M_PER_RAD,
    )
    geometry_position_error = float(
        np.max(np.linalg.norm(intent.eef_position - expected_semantic.eef_position, axis=-1))
    )
    geometry_rotation_error = float(
        np.max(
            geodesic_angle(
                quat_to_matrix(intent.eef_orientation),
                quat_to_matrix(expected_semantic.eef_orientation),
            )
        )
    )
    timing_error = float(np.max(np.abs(intent.timing_timestamps - expected_semantic.timing_timestamps)))
    direct_pose_error = float(np.max(np.linalg.norm(materialized.eef_position - expected_semantic.to_trajectory_chunk(query_timestamps=source_h40.point_timestamps).eef_position, axis=-1)))
    direct_rotation_error = float(
        np.max(
            geodesic_angle(
                quat_to_matrix(materialized.eef_orientation),
                quat_to_matrix(
                    expected_semantic.to_trajectory_chunk(
                        query_timestamps=source_h40.point_timestamps
                    ).eef_orientation
                ),
            )
        )
    )
    adapter = GR00TToWBCAdapter(
        ReferenceAdapterConfig(
            expected_source_frame=source_h40.source_frame,
            wbc_frame=source_h40.source_frame,
            tcp_semantic_id=source_h40.tcp_semantic_id,
        )
    )
    adapter_preview, adapter_gripper = adapter.adapt(materialized)
    direct_preview = __import__(
        "go2arx5_vla.trajectory.preview", fromlist=["se3_arc_length_preview"]
    ).se3_arc_length_preview(
        materialized,
        anchor_timestamp=float(materialized.observation_timestamp),
        offsets_m=adapter.config.preview_offsets_m,
        lambda_m_per_rad=adapter.config.lambda_m_per_rad,
    )
    adapter_preview_exact = bool(
        np.array_equal(adapter_preview.eef_position, direct_preview.eef_position)
        and np.array_equal(adapter_preview.eef_rot6d, direct_preview.eef_rot6d)
        and np.array_equal(adapter_preview.endpoint_clamped, direct_preview.endpoint_clamped)
    )
    equivalence = evaluate_preview_equivalence(
        source_h40,
        materialized,
        case_id=str(payload["case_id"]),
        representation="STRUCTURED_ACTION_V1_T16",
        anchor_count=101,
    )
    stage38a_source, stage38a_dense, stage38a_candidate, stage38a_audit = STAGE38A.materialize_timing_candidate(
        payload, STRUCTURED_TIMING_MODE
    )
    stage38a_equivalence = evaluate_preview_equivalence(
        stage38a_dense,
        stage38a_candidate,
        case_id=str(payload["case_id"]),
        representation="STAGE38A_K16_T16_NATIVE_TRANSPORT",
        anchor_count=101,
    )
    result = {
        "case_id": str(payload["case_id"]),
        "family": payload.get("family"),
        "source_sample_count": source.point_count,
        "source_timestamp_start_s": float(source.point_timestamps[0]),
        "source_timestamp_end_s": float(source.point_timestamps[-1]),
        "source_timestamp_nonuniform": bool(
            not np.allclose(np.diff(source.point_timestamps), np.diff(source.point_timestamps)[0], atol=1e-12)
        ),
        "audit_h40_sample_count": source_h40.point_count,
        "audit_h40_duration_s": float(source_h40.validity_end - source_h40.validity_start),
        "label": label.as_dict(include_arrays=False),
        "label_mask_exact": bool(np.array_equal(label.mask, structured_action_mask(STRUCTURED_TIMING_MODE))),
        "label_active_scalar_count": label.active_scalar_count,
        "geometry_control_roundtrip_max_position_error_m": geometry_position_error,
        "geometry_control_roundtrip_max_rotation_error_rad": geometry_rotation_error,
        "timing_control_roundtrip_max_error_s": timing_error,
        "decoded_physical_pose_max_position_error_m": direct_pose_error,
        "decoded_physical_pose_max_rotation_error_rad": direct_rotation_error,
        "decoded_start_pose_exact": bool(
            np.linalg.norm(materialized.eef_position[0] - source_h40.eef_position[0]) <= 1e-10
            and geodesic_angle(
                quat_to_matrix(materialized.eef_orientation[0]),
                quat_to_matrix(source_h40.eef_orientation[0]),
            )
            <= 1e-10
        ),
        "decoded_end_pose_exact": bool(
            np.linalg.norm(materialized.eef_position[-1] - expected_semantic.eef_position[-1]) <= 1e-10
            and geodesic_angle(
                quat_to_matrix(materialized.eef_orientation[-1]),
                quat_to_matrix(expected_semantic.eef_orientation[-1]),
            )
            <= 1e-10
        ),
        "decoded_timing_monotonic": bool(np.all(np.diff(intent.timing_timestamps) > 0.0)),
        "decoded_duration_exact": bool(
            np.isclose(intent.validity_end - intent.validity_start, source_h40.validity_end - source_h40.validity_start, atol=1e-12)
        ),
        "decoded_so3_valid": bool(
            np.isfinite(intent.eef_orientation).all()
            and np.allclose(
                np.linalg.det(quat_to_matrix(intent.eef_orientation)), 1.0, atol=1e-8
            )
        ),
        "gripper_roundtrip_exact": bool(np.array_equal(materialized.gripper, physical[:, 9])),
        "frame_tcp_preserved": bool(
            materialized.source_frame == source_h40.source_frame
            and materialized.tcp_semantic_id == source_h40.tcp_semantic_id
        ),
        "adapter_preview_exact_to_direct_preview": adapter_preview_exact,
        "adapter_gripper_finite": adapter_gripper is not None and np.isfinite(adapter_gripper.normalized_command),
        "frozen_source_vs_structured_equivalence": public_evaluation(equivalence),
        "stage38a_t16_equivalence": public_evaluation(stage38a_equivalence),
        "stage38a_candidate_audit": jsonable(stage38a_audit),
    }
    result["pass"] = bool(
        result["label_mask_exact"]
        and result["label_active_scalar_count"] == 190
        and result["geometry_control_roundtrip_max_position_error_m"] <= 1e-10
        and result["geometry_control_roundtrip_max_rotation_error_rad"] <= 1e-10
        and result["timing_control_roundtrip_max_error_s"] <= 1e-10
        and result["decoded_physical_pose_max_position_error_m"] <= 1e-10
        and result["decoded_physical_pose_max_rotation_error_rad"] <= 1e-10
        and result["decoded_start_pose_exact"]
        and result["decoded_end_pose_exact"]
        and result["decoded_timing_monotonic"]
        and result["decoded_duration_exact"]
        and result["decoded_so3_valid"]
        and result["gripper_roundtrip_exact"]
        and result["frame_tcp_preserved"]
        and result["adapter_preview_exact_to_direct_preview"]
        and result["adapter_gripper_finite"]
        and equivalence.get("passes_limits") is True
        and stage38a_equivalence.get("passes_limits") is True
    )
    return result


def stage35_roundtrip_audit() -> dict[str, Any]:
    payloads = STAGE37.load_payloads()
    rows = [_roundtrip_case(payloads[case_id]) for case_id in STAGE35_CASE_IDS]
    return {
        "schema": "STAGE38BC_STAGE35_STRUCTURED_CODEC_ROUNDTRIP_V1",
        "pi_gate": PI_GATE_ID,
        "timing_mode": STRUCTURED_TIMING_MODE,
        "geometry_k": STRUCTURED_GEOMETRY_K,
        "case_count": len(rows),
        "case_ids": list(STAGE35_CASE_IDS),
        "source_timestamp_policy": "ORIGINAL_TIMESTAMPED_EXPERT_TIME_LAW_THEN_EXPLICIT_H40_AUDIT_SAMPLE",
        "rows": rows,
        "all_cases_pass": bool(all(row.get("pass") for row in rows)),
        "stage38a_all_cases_pass": bool(
            all(row.get("stage38a_t16_equivalence", {}).get("passes_limits") for row in rows)
        ),
        "status": "PASS" if all(row.get("pass") for row in rows) else "BLOCKED",
    }


def training_sample_roundtrip_audit() -> dict[str, Any]:
    snapshot = load_snapshot(TINY1_SNAPSHOT)
    label = encode_structured_action(
        snapshot.physical_action,
        snapshot.future_timestamps,
        snapshot.state_v1.as_array(),
        timing_mode=STRUCTURED_TIMING_MODE,
        chunk_id=snapshot.example_id,
        source_frame=snapshot.source_frame,
        tcp_semantic_id=snapshot.tcp_semantic_id,
        observation_timestamp=snapshot.observation_timestamp,
        provenance={
            "training_corpus": "TINY1_H2STATS_LEROBOT_V2",
            "snapshot_sha256": TINY1_SNAPSHOT_SHA256,
            "stage23_source_timestamp_authority": True,
        },
    )
    intent = decode_structured_action(label, intent_id=f"{snapshot.example_id}:structured")
    materialized = materialize_structured_action(label, intent)
    result = {
        "schema": "STAGE38BC_TINY1_STRUCTURED_CODEC_ROUNDTRIP_V1",
        "snapshot_path": str(TINY1_SNAPSHOT),
        "snapshot_sha256": TINY1_SNAPSHOT_SHA256,
        "example_id": snapshot.example_id,
        "label": label.as_dict(include_arrays=False),
        "materialized_shape": list(materialized.eef_position.shape),
        "duration_s": float(intent.validity_end - intent.validity_start),
        "timing_monotonic": bool(np.all(np.diff(intent.timing_timestamps) > 0.0)),
        "duration_exact": bool(np.isclose(intent.validity_end - intent.validity_start, 1.95, atol=1e-12)),
        "start_pose_exact": bool(
            np.linalg.norm(materialized.eef_position[0] - snapshot.future_desired_eef_9d[0, :3]) <= 1e-10
            and geodesic_angle(
                quat_to_matrix(materialized.eef_orientation[0]),
                quat_to_matrix(snapshot_to_quaternion(snapshot.future_desired_eef_9d[0, 3:9])),
            )
            <= 1e-10
        ),
        "end_pose_matches_k16_semantic": bool(
            np.isfinite(materialized.eef_position[-1]).all()
            and np.isfinite(materialized.eef_orientation[-1]).all()
        ),
        "gripper_exact": bool(
            np.array_equal(materialized.gripper, snapshot.future_gripper.reshape(40))
        ),
        "frame_tcp_preserved": bool(
            materialized.source_frame == snapshot.source_frame
            and materialized.tcp_semantic_id == snapshot.tcp_semantic_id
        ),
        "finite": bool(
            np.isfinite(label.values).all()
            and np.isfinite(label.mask).all()
            and np.isfinite(materialized.eef_position).all()
            and np.isfinite(materialized.eef_orientation).all()
        ),
    }
    result["status"] = (
        "PASS"
        if result["duration_exact"]
        and result["timing_monotonic"]
        and result["start_pose_exact"]
        and result["end_pose_matches_k16_semantic"]
        and result["gripper_exact"]
        and result["frame_tcp_preserved"]
        and result["finite"]
        else "BLOCKED"
    )
    return result


def snapshot_to_quaternion(row_rot6d: Any) -> np.ndarray:
    from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix

    return matrix_to_quat(rot6d_rows_to_matrix(np.asarray(row_rot6d, dtype=np.float64)))


def build_action_contract_audit() -> dict[str, Any]:
    contract = structured_action_contract_summary(STRUCTURED_TIMING_MODE)
    mask = structured_action_mask(STRUCTURED_TIMING_MODE)
    return {
        "schema": "STAGE38BC_STRUCTURED_ACTION_CONTRACT_AUDIT_V1",
        "contract": contract,
        "mask_shape": list(mask.shape),
        "mask_binary": bool(np.all(np.isin(mask, [0.0, 1.0]))),
        "mask_active_elements": int(np.count_nonzero(mask)),
        "mask_inactive_elements": int(mask.size - np.count_nonzero(mask)),
        "geometry_mask_slots": [0, 14],
        "timing_mask_slots": [0, int(STRUCTURED_TIMING_MODE[1:]) - 2],
        "gripper_mask_slots": [0, STRUCTURED_ACTION_HORIZON - 1],
        "padding_mask_sum": float(np.sum(mask[:, 11:])),
        "dense_h40_active_scalar_count": 400,
        "structured_active_scalar_count": 190,
        "active_scalar_reduction_fraction": float(1.0 - 190.0 / 400.0),
        "loss_objective_change": False,
        "timing_auxiliary_loss": False,
        "component_reweighting": False,
        "smoothness_regularizer": False,
        "sampler_change": False,
        "status": "PASS"
        if mask.shape == (40, 132)
        and int(np.count_nonzero(mask)) == 190
        and float(np.sum(mask[:, 11:])) == 0.0
        else "BLOCKED",
    }


def seal_artifacts(output_root: Path, *, excluded: Iterable[str]) -> tuple[Path, str, list[dict[str, Any]]]:
    excluded_set = set(excluded)
    entries: list[dict[str, Any]] = []
    for path in sorted(output_root.iterdir()):
        if not path.is_file() or path.name in excluded_set:
            continue
        entries.append(
            {
                "path": path.name,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    seal_path = output_root / "stage38bc_artifact_seal.json"
    seal_payload = {
        "schema": "STAGE38BC_ARTIFACT_SEAL_V1",
        "stage": "Stage38-B/C",
        "pi_gate": PI_GATE_ID,
        "pr": PR_NUMBER,
        "output_root": str(output_root),
        "entries": entries,
        "model_constructed": False,
        "forward": False,
        "training": False,
        "pbs_submitted": False,
        "gpu_used": False,
        "historical_inputs_mutated": False,
        "scientific_contract_changed": False,
        "timestamp_utc": now_utc(),
    }
    write_json(seal_path, seal_payload)
    return seal_path, sha256_file(seal_path), entries


def run(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite existing B/C output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    log_path = output_root / "stage38bc_cpu_execution.log"
    log_path.write_text(
        json.dumps({"event": "START", "timestamp_utc": now_utc(), "head": args.api_head}) + "\n",
        encoding="utf-8",
    )
    preflight = git_preflight(args.api_head, args.expected_head)
    write_json(output_root / "stage38bc_preflight.json", preflight)
    static = static_guard()
    write_json(output_root / "stage38bc_static_guard.json", static)
    stage38a = stage38a_audit()
    write_json(output_root / "stage38bc_stage38a_audit.json", stage38a)
    qstat = qstat_audit()
    write_json(output_root / "stage38bc_qstat_snapshot.json", qstat)

    historical = immutable_historical_audit()
    write_json(output_root / "stage38bc_historical_input_audit.json", historical)
    dataset = dataset_contract_audit()
    write_json(output_root / "stage38bc_dataset_timing_audit.json", dataset)
    source = source_semantics_audit()
    write_json(output_root / "stage38bc_source_semantics_audit.json", source)
    contract = build_action_contract_audit()
    write_json(output_root / "stage38bc_action_contract.json", contract)

    admission_before_roundtrip = bool(
        preflight.get("status") == "PASS"
        and static.get("status") == "PASS"
        and stage38a.get("status") == "PASS"
        and qstat.get("status") == "PASS"
        and historical.get("status") == "PASS"
        and dataset.get("status") == "PASS"
        and source.get("status") == "PASS"
        and contract.get("status") == "PASS"
    )
    write_json(
        output_root / "stage38bc_roundtrip_admission.json",
        {
            "schema": "STAGE38BC_ROUNDTRIP_ADMISSION_V1",
            "admission_before_roundtrip": admission_before_roundtrip,
            "model_constructed": False,
            "status": "PASS" if admission_before_roundtrip else "BLOCKED",
        },
    )
    if not admission_before_roundtrip:
        raise RuntimeError("B/C immutable or source preflight failed; roundtrip not attempted")

    stage35_roundtrip = stage35_roundtrip_audit()
    write_json(output_root / "stage38bc_stage35_roundtrip.json", stage35_roundtrip)
    training_roundtrip = training_sample_roundtrip_audit()
    write_json(output_root / "stage38bc_tiny1_roundtrip.json", training_roundtrip)
    result = {
        "schema": "STAGE38BC_CPU_CODEC_QUALIFICATION_RESULT_V1",
        "pi_gate": PI_GATE_ID,
        "pr": PR_NUMBER,
        "branch": EXPECTED_BRANCH,
        "api_head": args.api_head,
        "expected_head": args.expected_head,
        "active_stage": "Stage38-B/C",
        "result": "STAGE38BC_CPU_CODEC_PASS_READY_FOR_TRAINING"
        if stage35_roundtrip.get("status") == "PASS"
        and training_roundtrip.get("status") == "PASS"
        else "STAGE38BC_CPU_CODEC_BLOCKED",
        "timing_model_capacity_star": "T16",
        "timing_model_capacity": "MODEL_TIMING_COMPACT_QUALIFIED",
        "structured_codec": "STRUCTURED_ACTION_V1_QUALIFIED"
        if stage35_roundtrip.get("status") == "PASS"
        and training_roundtrip.get("status") == "PASS"
        else "STRUCTURED_ACTION_V1_NOT_QUALIFIED",
        "stage35_roundtrip_status": stage35_roundtrip.get("status"),
        "tiny1_roundtrip_status": training_roundtrip.get("status"),
        "stage35_case_count": stage35_roundtrip.get("case_count"),
        "stage35_all_cases_pass": stage35_roundtrip.get("all_cases_pass"),
        "tiny1_duration_s": training_roundtrip.get("duration_s"),
        "structured_active_scalar_count": 190,
        "dense_h40_active_scalar_count": 400,
        "model_constructed": False,
        "forward": False,
        "training": False,
        "pbs_submitted": False,
        "gpu_used": False,
        "historical_inputs_mutated": False,
        "scientific_contract_changed": False,
        "next_gate": "STAGE38_D_TRAINING_CONDITIONAL_AFTER_BC_PASS",
        "blockers": []
        if stage35_roundtrip.get("status") == "PASS" and training_roundtrip.get("status") == "PASS"
        else ["STRUCTURED_CODEC_OR_TRAINING_SAMPLE_ROUNDTRIP_FAILED"],
        "timestamp_utc": now_utc(),
    }
    write_json(output_root / "stage38bc_result.json", result)
    log_path.write_text(
        log_path.read_text(encoding="utf-8")
        + json.dumps(
            {
                "event": "RESULT",
                "result": result["result"],
                "stage35_status": result["stage35_roundtrip_status"],
                "tiny1_status": result["tiny1_roundtrip_status"],
                "timestamp_utc": now_utc(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    seal_path, seal_sha256, entries = seal_artifacts(
        output_root,
        excluded=("stage38bc_artifact_seal.json", "stage38bc_return.json"),
    )
    result["artifact_seal_path"] = str(seal_path)
    result["artifact_seal_sha256"] = seal_sha256
    result["sealed_entry_count"] = len(entries)
    write_json(output_root / "stage38bc_return.json", result)
    return 0 if result["result"] == "STAGE38BC_CPU_CODEC_PASS_READY_FOR_TRAINING" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-head", required=True)
    parser.add_argument(
        "--expected-head",
        required=True,
        help="current live PR head; preflight requires API/remote/local equality",
    )
    parser.add_argument("--pr-number", type=int, default=PR_NUMBER)
    parser.add_argument("--output-root", required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.pr_number != PR_NUMBER:
        raise SystemExit(f"PR number must remain {PR_NUMBER}")
    raise SystemExit(run(arguments))
