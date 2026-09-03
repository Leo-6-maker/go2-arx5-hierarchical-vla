#!/usr/bin/env python3
"""Prepare or execute the Stage25 BASE -> O1@200 parameter-delta audit.

The audit path is deliberately a direct safetensors comparison.  It never
constructs a GR00T model, imports CUDA, runs a forward pass, performs
inference, trains, or submits a scheduler job.  The preparation path uses
only the immutable Stage23 Phase-A evidence relay and is runnable with the
Python standard library.  The comparison path imports ``torch`` and
``safetensors`` lazily and opens both trees with ``device="cpu"``.

The preparation command is the Stage25 Corelab deliverable.  The compare
command is fail-closed until exact BASE and O1 checkpoint bytes plus the
sealed O1 per-file manifest are supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import stat
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable, Mapping


STAGE24_BASE_SHA = "c5ded6e119df920402451497aacdc69274ca8fc1"
STAGE23_FROZEN_GIT_SHA = "deb920a070141e6a32a88cd0da6719828647e1e6"
EXPECTED_GROOT_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"

EXPECTED_ARCHIVE_SHA256 = "720aca9a1a1c7146919620df4000a76cc228a71729b057a2a0d155787ea50dcd"
EXPECTED_PAYLOAD_MANIFEST_SHA256 = "dfeb6b44410b0042e53bac261e49acf7dbaa23ee2a61839ab65d4171b6c42b6a"
EXPECTED_BASE_INDEX_SHA256 = "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59"
EXPECTED_BASE_TREE_SHA256 = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
EXPECTED_BASE_TREE_PREFIX = "payload/GR00T-N1.7-3B"
EXPECTED_O1_INDEX_SHA256 = "ff5a9ac57951658b775ef7e958a3469900f767cec60cbf487b3c8d0a3d787bc7"
EXPECTED_O1_SEAL_SHA256 = "7aa07a84d62a5052bb9522ba0a04f8d1b711dac0f6b448db3e696135a663f806"
EXPECTED_O1_STEPS = 200
EXPECTED_ACTION_HORIZON = 40
EXPECTED_CATEGORY_IDS = (4, 5, 6, 7, 10, 11, 24, 25, 26, 27)
EXPECTED_ACTIVE_ACTION_DIM = 10
EXPECTED_UNIVERSAL_ACTION_DIM = 132
EXPECTED_INACTIVE_ACTION_RANGE = [10, 132]
EPSILON = 1e-12
HASH_BLOCK_SIZE = 8 * 1024 * 1024


# These are the serialized names emitted by the pinned N1.7 modules.  There
# are 4 state-encoder keys, 6 action-encoder keys, and 4 action-decoder keys.
CATEGORY_PROJECTOR_KEYS = (
    "action_head.state_encoder.layer1.W",
    "action_head.state_encoder.layer1.b",
    "action_head.state_encoder.layer2.W",
    "action_head.state_encoder.layer2.b",
    "action_head.action_encoder.W1.W",
    "action_head.action_encoder.W1.b",
    "action_head.action_encoder.W2.W",
    "action_head.action_encoder.W2.b",
    "action_head.action_encoder.W3.W",
    "action_head.action_encoder.W3.b",
    "action_head.action_decoder.layer1.W",
    "action_head.action_decoder.layer1.b",
    "action_head.action_decoder.layer2.W",
    "action_head.action_decoder.layer2.b",
)
ACTION_ENCODER_W1_KEY = "action_head.action_encoder.W1.W"

MODULE_GROUP_RULES = (
    {
        "group": "CATEGORY_PROJECTORS",
        "patterns": [
            r"^action_head\.(state_encoder|action_encoder|action_decoder)\.",
            r"^action_head\.position_embedding\.",
        ],
        "expected_trainable": True,
        "basis": "tune_projector=true in the pinned Stage23 recipe",
    },
    {
        "group": "DIFFUSION_ACTION_HEAD_SHARED_TRAINABLE",
        "patterns": [r"^action_head\.model\."],
        "expected_trainable": True,
        "basis": "tune_diffusion_model=true in the pinned Stage23 recipe",
    },
    {
        "group": "VISUAL_LLM_VLLN_EXPECTED_FROZEN",
        "patterns": [
            r"^backbone\.",
            r"^action_head\.vlln\.",
            r"^action_head\.vl_self_attention\.",
        ],
        "expected_trainable": False,
        "basis": "tune_visual=false, tune_llm=false, tune_vlln=false",
    },
    {
        "group": "OTHER_EXPECTED_FROZEN",
        "patterns": [r"."],
        "expected_trainable": False,
        "basis": "conservative fallback; any observed delta is unexpected until source-explained",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_object(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def safe_relative_path(value: str) -> str:
    if not value or "\x00" in value:
        raise ValueError(f"invalid relative path: {value!r}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path escapes root: {value!r}")
    return path.as_posix()


def assert_regular(path: Path) -> None:
    mode = os.lstat(path).st_mode
    if stat.S_IFMT(mode) != stat.S_IFREG:
        raise ValueError(f"expected a regular file: {path}")


def walk_regular_files(root: Path) -> list[dict[str, Any]]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"checkpoint root is not a directory: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            if child.is_symlink():
                raise ValueError(f"symlink directory is not allowed: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            assert_regular(child)
            relative = child.relative_to(root).as_posix()
            entries.append(
                {
                    "path": relative,
                    "bytes": child.stat().st_size,
                    "sha256": sha256_file(child),
                }
            )
    return sorted(entries, key=lambda item: item["path"])


def stage24_tree_sha256(entries: Iterable[Mapping[str, Any]], prefix: str) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        path = f"{prefix}/{entry['path']}"
        digest.update(
            f"{path}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _assert_output_path_is_external(output: Path, source: Path) -> None:
    try:
        output.resolve().relative_to(source.resolve())
    except ValueError:
        return
    raise ValueError("output must remain outside the immutable evidence root")


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_phase_a_evaluation(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = text.find('"schema": "STAGE23_O1_TINY1_H2STATS_EVALUATION_V1"')
    if marker < 0:
        raise ValueError(f"evaluation schema marker not found: {path}")
    start = text.rfind("{", 0, marker)
    if start < 0:
        raise ValueError(f"evaluation JSON start not found: {path}")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError(f"evaluation payload is not an object: {path}")
    return value


def _read_unheaded_tsv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not row or all(not field for field in row):
                continue
            if len(row) != 5:
                raise ValueError(f"expected five TSV fields at {path}:{line_number}")
            directory, filename, kind, size, digest = row
            if kind != "regular file":
                raise ValueError(f"unsupported seal row at {path}:{line_number}")
            relative = safe_relative_path(f"{directory}/{filename}")
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"invalid SHA256 at {path}:{line_number}")
            rows.append(
                {
                    "sealed_path": relative,
                    "relative_path": relative,
                    "bytes": int(size),
                    "sha256": digest,
                }
            )
    return rows


def _o1_seal_rows(seal_path: Path) -> list[dict[str, Any]]:
    rows = _read_unheaded_tsv(seal_path)
    selected = [row for row in rows if row["sealed_path"].startswith("candidate_200/")]
    if not selected:
        raise ValueError("O1@200 rows are absent from checkpoint seal")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in selected:
        relative = row["sealed_path"][len("candidate_200/") :]
        relative = safe_relative_path(relative)
        if relative in seen:
            raise ValueError(f"duplicate O1@200 sealed file: {relative}")
        seen.add(relative)
        result.append({**row, "relative_path": relative})
    return sorted(result, key=lambda item: item["relative_path"])


def _o1_seal_sha256_status(seal_path: Path) -> dict[str, Any]:
    """Return the frozen whole-seal digest comparison without reading tensors."""

    assert_regular(seal_path)
    observed = sha256_file(seal_path)
    return {
        "expected_o1_seal_sha256": EXPECTED_O1_SEAL_SHA256,
        "observed_o1_seal_sha256": observed,
        "verified": observed == EXPECTED_O1_SEAL_SHA256,
    }


def _require_o1_seal_sha256(seal_path: Path) -> dict[str, Any]:
    result = _o1_seal_sha256_status(seal_path)
    if not result["verified"]:
        raise ValueError(
            "O1 checkpoint seal SHA256 mismatch: "
            f"expected {result['expected_o1_seal_sha256']}, "
            f"observed {result['observed_o1_seal_sha256']}"
        )
    return result


def _verify_payload_manifest(phase_root: Path, seal_root: Path) -> dict[str, Any]:
    manifest_path = seal_root / "payload_file_manifest.tsv"
    if not manifest_path.is_file():
        raise ValueError(f"payload manifest missing: {manifest_path}")
    observed_manifest_sha = sha256_file(manifest_path)
    rows: list[dict[str, str]] = []
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"relative_path", "type", "bytes", "sha256"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("payload manifest schema is not understood")
        rows.extend(dict(row) for row in reader)
    verified = 0
    errors: list[str] = []
    seal_resolved = seal_root.resolve(strict=True)
    for row in rows:
        relative = safe_relative_path(row["relative_path"])
        path = (seal_root / relative).resolve()
        try:
            path.relative_to(seal_resolved)
        except ValueError:
            errors.append(f"escaped payload path: {relative}")
            continue
        if not path.is_file():
            errors.append(f"missing payload file: {relative}")
            continue
        if path.stat().st_size != int(row["bytes"]):
            errors.append(f"payload size mismatch: {relative}")
            continue
        if sha256_file(path) != row["sha256"]:
            errors.append(f"payload SHA256 mismatch: {relative}")
            continue
        verified += 1
    return {
        "path": str(manifest_path),
        "sha256": observed_manifest_sha,
        "expected_sha256": EXPECTED_PAYLOAD_MANIFEST_SHA256,
        "sha256_match": observed_manifest_sha == EXPECTED_PAYLOAD_MANIFEST_SHA256,
        "row_count": len(rows),
        "files_verified": verified,
        "errors": errors,
        "pass": observed_manifest_sha == EXPECTED_PAYLOAD_MANIFEST_SHA256
        and not errors
        and verified == len(rows),
    }


def _checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint is not a directory: {checkpoint}")
    index_path = checkpoint / "model.safetensors.index.json"
    assert_regular(index_path)
    index = load_json_object(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"model index has no weight_map: {index_path}")
    normalized_map: dict[str, str] = {}
    for key, value in weight_map.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("model index contains a non-string key or shard")
        normalized_map[key] = safe_relative_path(value)
    return {
        "path": str(checkpoint),
        "index_path": str(index_path),
        "index_sha256": sha256_file(index_path),
        "index": index,
        "weight_map": normalized_map,
        "keys": sorted(normalized_map),
    }


def _verify_o1_seal_against_checkpoint(checkpoint: Path, seal_path: Path) -> dict[str, Any]:
    seal_sha_status = _o1_seal_sha256_status(seal_path)
    rows = _o1_seal_rows(seal_path)
    actual_entries = walk_regular_files(checkpoint)
    actual_by_path = {entry["path"]: entry for entry in actual_entries}
    expected_by_path = {row["relative_path"]: row for row in rows}
    errors: list[str] = []
    if set(actual_by_path) != set(expected_by_path):
        errors.append("O1 checkpoint file set differs from candidate_200 seal")
    for relative, expected in expected_by_path.items():
        actual = actual_by_path.get(relative)
        if actual is None:
            continue
        if actual["bytes"] != expected["bytes"]:
            errors.append(f"O1 size mismatch: {relative}")
        if actual["sha256"] != expected["sha256"]:
            errors.append(f"O1 SHA256 mismatch: {relative}")
    return {
        "seal_path": str(seal_path.resolve()),
        "seal_sha256": seal_sha_status["observed_o1_seal_sha256"],
        "expected_o1_seal_sha256": seal_sha_status["expected_o1_seal_sha256"],
        "observed_o1_seal_sha256": seal_sha_status["observed_o1_seal_sha256"],
        "verified": seal_sha_status["verified"],
        "expected_file_count": len(rows),
        "actual_file_count": len(actual_entries),
        "entries": rows,
        "errors": errors,
        "pass": not errors and seal_sha_status["verified"],
    }


def _prepare_protocol(phase_root: Path, output: Path) -> dict[str, Any]:
    phase_root = phase_root.resolve(strict=True)
    _assert_output_path_is_external(output, phase_root)
    archive = phase_root / "phase_a_evidence_bundle_20260903.tar.gz"
    seal_root = phase_root / "extracted" / "phase_a_seal_20260903"
    relay_path = phase_root / "relay_verification_v1.json"
    provenance_path = seal_root / "provenance.json"
    run_manifest_path = seal_root / "configs" / "candidate_200.run_manifest.json"
    trainer_state_path = seal_root / "candidates" / "trainer_state_200.json"
    evaluation_log_path = seal_root / "candidates" / "eval_200.log"
    seal_path = seal_root / "checkpoint_seal" / "checkpoint_file_seal.tsv"

    for path in (archive, seal_root, relay_path, provenance_path, run_manifest_path, trainer_state_path, evaluation_log_path, seal_path):
        if not path.exists():
            raise ValueError(f"required Stage23 evidence is missing: {path}")

    archive_sha = sha256_file(archive)
    if archive_sha != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(f"Stage23 archive SHA256 mismatch: {archive_sha}")
    payload = _verify_payload_manifest(phase_root, seal_root)
    if not payload["pass"]:
        raise ValueError("Stage23 payload manifest verification failed")

    relay = load_json_object(relay_path)
    provenance = load_json_object(provenance_path)
    run_manifest = load_json_object(run_manifest_path)
    trainer_state = load_json_object(trainer_state_path)
    evaluation = _parse_phase_a_evaluation(evaluation_log_path)
    seal_rows = _o1_seal_rows(seal_path)
    seal_sha_status = _require_o1_seal_sha256(seal_path)

    if run_manifest.get("schema") != "STAGE20S_GR00T_TRAINING_RUN_V1":
        raise ValueError("O1 run manifest schema is not the pinned Stage20S schema")
    if run_manifest.get("status") != "PASS" or run_manifest.get("max_steps") != EXPECTED_O1_STEPS:
        raise ValueError("O1 run manifest does not bind a successful 200-step run")
    if run_manifest.get("groot_repo_sha") != EXPECTED_GROOT_SHA:
        raise ValueError("O1 run manifest GR00T SHA differs from the pinned source")
    if run_manifest.get("base_model_index_sha256") != EXPECTED_BASE_INDEX_SHA256:
        raise ValueError("O1 run manifest base index identity differs from Stage24")
    if trainer_state.get("global_step") != EXPECTED_O1_STEPS:
        raise ValueError("trainer state does not bind global_step=200")
    if evaluation.get("status") not in {"GATE_FAIL", "PASS", "GATE_PASS"}:
        raise ValueError("O1 evaluation status is not a recognized sealed result")
    if evaluation.get("model_path") and not str(evaluation["model_path"]).endswith("checkpoint-200"):
        raise ValueError("O1 evaluation model path does not end at checkpoint-200")

    expected_index_rows = [row for row in seal_rows if row["relative_path"] == "model.safetensors.index.json"]
    if len(expected_index_rows) != 1 or expected_index_rows[0]["sha256"] != EXPECTED_O1_INDEX_SHA256:
        raise ValueError("O1 per-file seal does not bind the expected index SHA256")
    shard_rows = [
        row for row in seal_rows if row["relative_path"].endswith(".safetensors")
    ]
    if len(shard_rows) != 2:
        raise ValueError("expected exactly two O1 safetensor shard seal entries")

    # The Phase-A relay intentionally contains metadata and seals, not the
    # full O1 checkpoint bytes.  Check the explicit expected location only;
    # do not search arbitrary roots or substitute another checkpoint.
    expected_historical_path = str(evaluation.get("model_path") or "")
    candidate_paths = [
        seal_root / "candidates" / "checkpoint_200",
        seal_root / "candidates" / "checkpoint_200_metadata",
    ]
    full_bytes_present = any(
        (candidate / "model-00001-of-00002.safetensors").is_file()
        and (candidate / "model-00002-of-00002.safetensors").is_file()
        for candidate in candidate_paths
    )

    scope = load_json_object(seal_root / "phase_a_scope.json")
    contains_credentials = [
        value
        for value in (
            relay.get("contains_credentials"),
            scope.get("contains_credentials"),
        )
        if isinstance(value, bool)
    ]
    if contains_credentials and any(contains_credentials):
        raise ValueError("Stage23 evidence explicitly reports credentials")

    protocol = {
        "schema": "STAGE25_O1_PARAMETER_DELTA_PROTOCOL_V1",
        "status": "FROZEN_FOR_CPU_STATIC_EXECUTION",
        "stage24_base_sha": STAGE24_BASE_SHA,
        "stage23_frozen_git_sha": STAGE23_FROZEN_GIT_SHA,
        "groot_sha": EXPECTED_GROOT_SHA,
        "evidence": {
            "phase_a_root": str(phase_root),
            "archive_sha256": archive_sha,
            "payload_manifest_sha256": payload["sha256"],
            "payload_manifest_rows": payload["row_count"],
            "payload_files_verified": payload["files_verified"],
            "relay_sha256": sha256_file(relay_path),
            "relay_contains_credentials": relay.get("contains_credentials") is False,
        },
        "base_identity": {
            "index_sha256": EXPECTED_BASE_INDEX_SHA256,
            "tree_sha256": EXPECTED_BASE_TREE_SHA256,
            "tree_prefix": EXPECTED_BASE_TREE_PREFIX,
            "identity_must_be_reverified": True,
        },
        "o1_identity": {
            "historical_path": expected_historical_path,
            "run_manifest_path": str(run_manifest_path),
            "run_manifest_sha256": sha256_file(run_manifest_path),
            "trainer_state_path": str(trainer_state_path),
            "trainer_state_sha256": sha256_file(trainer_state_path),
            "global_step": trainer_state.get("global_step"),
            "index_sha256": EXPECTED_O1_INDEX_SHA256,
            "checkpoint_seal_path": str(seal_path),
            "checkpoint_seal_sha256": seal_sha_status["observed_o1_seal_sha256"],
            "expected_o1_seal_sha256": seal_sha_status["expected_o1_seal_sha256"],
            "observed_o1_seal_sha256": seal_sha_status["observed_o1_seal_sha256"],
            "seal_sha256_verified": seal_sha_status["verified"],
            "per_file_seal_available": True,
            "per_file_seal_entries": seal_rows,
            "full_bytes_present_in_phase_a_relay": full_bytes_present,
            "full_bytes_required_for_delta_execution": True,
        },
        "category_projector_key_contract": {
            "count": len(CATEGORY_PROJECTOR_KEYS),
            "keys": list(CATEGORY_PROJECTOR_KEYS),
            "category_ids": list(EXPECTED_CATEGORY_IDS),
            "category_id_labels": {
                "4-7": "deterministic unused/control comparison rows",
                "10": "NEW_EMBODIMENT",
                "11": "UNITREE_G1_SONIC",
                "24-27": "pretrained reference rows",
            },
        },
        "action_encoder_w1_partition": {
            "key": ACTION_ENCODER_W1_KEY,
            "active_columns": [0, EXPECTED_ACTIVE_ACTION_DIM],
            "inactive_columns": list(EXPECTED_INACTIVE_ACTION_RANGE),
            "inactive_dimension_count": EXPECTED_UNIVERSAL_ACTION_DIM - EXPECTED_ACTIVE_ACTION_DIM,
            "requires_per_column_normalized_metrics": True,
        },
        "module_classification": {
            "recipe": {
                "tune_projector": True,
                "tune_diffusion_model": True,
                "tune_visual": False,
                "tune_llm": False,
                "tune_vlln": False,
            },
            "rules": list(MODULE_GROUP_RULES),
            "every_serialized_key_must_match_exactly_one_rule": True,
            "unexpected_frozen_group_delta_is_stop": True,
        },
        "required_key_metrics": [
            "exact key / shard / shape / dtype",
            "BASE tensor SHA256",
            "O1 tensor SHA256",
            "absolute delta L2 / RMS / max-abs",
            "relative delta L2",
            "cosine(BASE, O1)",
            "changed-element fraction at stored dtype",
        ],
        "required_group_metrics": [
            "key count",
            "parameter count",
            "changed key count",
            "aggregate delta L2 / RMS",
            "aggregate BASE norm",
            "relative delta",
            "max per-key relative delta",
        ],
        "execution_boundary": {
            "model_constructed": False,
            "cuda_used": False,
            "forward": False,
            "inference": False,
            "training": False,
            "pbs_submitted": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "wbc": False,
            "hardware": False,
        },
        "comparison_status": "PREPARED_NOT_EXECUTED",
        "next_gate": "HOPPER_CPU_STATIC_DELTA_EXECUTION",
    }
    result = {
        "schema": "STAGE25_O1_PARAMETER_DELTA_PREPARATION_V1",
        "status": "PASS_PREPARATION_O1_BYTES_NOT_IN_PHASE_A_RELAY",
        "protocol": protocol,
        "historical_o1_run": {
            "evaluation_status": evaluation.get("status"),
            "model_path": expected_historical_path,
            "base_model_path": evaluation.get("base_model_path"),
            "base_model_index_sha256": evaluation.get("base_model_index_sha256"),
            "output_dir": run_manifest.get("output_dir"),
            "max_steps": run_manifest.get("max_steps"),
            "checkpoint_index_sha256": EXPECTED_O1_INDEX_SHA256,
            "run_manifest_sha256": sha256_file(run_manifest_path),
            "trainer_state_sha256": sha256_file(trainer_state_path),
            "trainer_global_step": trainer_state.get("global_step"),
            "checkpoint_seal_sha256": seal_sha_status["observed_o1_seal_sha256"],
            "expected_o1_seal_sha256": seal_sha_status["expected_o1_seal_sha256"],
            "observed_o1_seal_sha256": seal_sha_status["observed_o1_seal_sha256"],
            "seal_sha256_verified": seal_sha_status["verified"],
            "per_file_seal_available": True,
            "full_checkpoint_bytes_in_current_phase_a_relay": full_bytes_present,
        },
        "source_evidence": {
            "provenance": provenance,
            "payload_manifest": payload,
            "relay_sha256": sha256_file(relay_path),
            "contains_credentials": False,
            "token_content_read": False,
        },
        "delta_execution": {
            "status": "NOT_EXECUTED",
            "reason": "Phase-A relay intentionally contains O1 metadata/seals but not the two full safetensor shards",
            "base_checkpoint_bytes_required": True,
            "o1_checkpoint_bytes_required": True,
            "results": None,
        },
        "execution_boundary": protocol["execution_boundary"],
    }
    _write_json_once(output, result)
    return result


def _load_tensor_backend() -> tuple[Any, Any]:
    try:
        import torch
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - exercised on dependency-light hosts
        raise RuntimeError(
            "direct delta execution requires CPU torch and safetensors; no model work was attempted"
        ) from exc
    return torch, safe_open


def _tensor_selection(shape: tuple[int, ...], start: int, end: int) -> tuple[Any, ...]:
    if not shape:
        return tuple()
    return (slice(start, end),) + tuple(slice(None) for _ in shape[1:])


def _iter_tensor_chunks(handle: Any, key: str, torch: Any, chunk_elements: int = 4 * 1024 * 1024):
    tensor_slice = handle.get_slice(key)
    shape = tuple(int(value) for value in tensor_slice.get_shape())
    if not shape:
        tensor = handle.get_tensor(key)
        yield tensor.reshape(1)
        return
    row_elements = math.prod(shape[1:]) if len(shape) > 1 else 1
    rows_per_chunk = max(1, chunk_elements // max(1, row_elements))
    for start in range(0, shape[0], rows_per_chunk):
        end = min(shape[0], start + rows_per_chunk)
        value = tensor_slice[_tensor_selection(shape, start, end)]
        if not torch.is_tensor(value):
            raise ValueError(f"safetensor slice did not return a torch tensor: {key}")
        if value.device.type != "cpu":
            raise ValueError(f"non-CPU tensor returned for {key}: {value.device}")
        yield value


def _canonical_tensor_bytes(tensor: Any, torch: Any) -> bytes:
    value = tensor.detach().to(device="cpu").contiguous()
    if value.numel() == 0:
        return b""
    # Viewing as uint8 preserves the stored dtype, including bfloat16, while
    # avoiding an implicit float conversion in the tensor hash.
    return value.view(torch.uint8).numpy().tobytes(order="C")


def _assert_tensor_finite(tensor: Any, torch: Any, key: str) -> None:
    if tensor.is_floating_point() or tensor.is_complex():
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"non-finite tensor values in {key}")


def _compare_tensor_values(base: Any, other: Any, torch: Any, *, label: str) -> dict[str, Any]:
    if tuple(base.shape) != tuple(other.shape):
        raise ValueError(f"shape mismatch for {label}: {tuple(base.shape)} != {tuple(other.shape)}")
    if base.dtype != other.dtype:
        raise ValueError(f"dtype mismatch for {label}: {base.dtype} != {other.dtype}")
    _assert_tensor_finite(base, torch, label)
    _assert_tensor_finite(other, torch, label)
    base_float = base.detach().to(dtype=torch.float64, device="cpu")
    other_float = other.detach().to(dtype=torch.float64, device="cpu")
    delta = other_float - base_float
    base_sq = float(torch.sum(base_float * base_float).item())
    other_sq = float(torch.sum(other_float * other_float).item())
    delta_sq = float(torch.sum(delta * delta).item())
    dot = float(torch.sum(base_float * other_float).item())
    count = int(base.numel())
    changed = int(torch.count_nonzero(base != other).item())
    delta_max = float(torch.max(torch.abs(delta)).item()) if count else 0.0
    base_l2 = math.sqrt(max(0.0, base_sq))
    other_l2 = math.sqrt(max(0.0, other_sq))
    delta_l2 = math.sqrt(max(0.0, delta_sq))
    return {
        "shape": list(base.shape),
        "dtype": str(base.dtype),
        "parameter_count": count,
        "finite": True,
        "base_l2": base_l2,
        "other_l2": other_l2,
        "delta_l2": delta_l2,
        "delta_rms": math.sqrt(max(0.0, delta_sq) / count) if count else 0.0,
        "delta_max_abs": delta_max,
        "relative_delta_l2": delta_l2 / max(base_l2, EPSILON),
        "cosine_base_other": dot / (base_l2 * other_l2) if base_l2 > 0 and other_l2 > 0 else 0.0,
        "changed_element_count": changed,
        "changed_element_fraction": changed / count if count else 0.0,
        "base_tensor_sha256": hashlib.sha256(_canonical_tensor_bytes(base, torch)).hexdigest(),
        "other_tensor_sha256": hashlib.sha256(_canonical_tensor_bytes(other, torch)).hexdigest(),
    }


def _compare_streamed_key(base_handle: Any, other_handle: Any, key: str, torch: Any) -> dict[str, Any]:
    base_slice = base_handle.get_slice(key)
    other_slice = other_handle.get_slice(key)
    base_shape = tuple(int(value) for value in base_slice.get_shape())
    other_shape = tuple(int(value) for value in other_slice.get_shape())
    if base_shape != other_shape:
        raise ValueError(f"shape mismatch for {key}: {base_shape} != {other_shape}")

    digest_base = hashlib.sha256()
    digest_other = hashlib.sha256()
    total: dict[str, float] = {
        "base_sq": 0.0,
        "other_sq": 0.0,
        "delta_sq": 0.0,
        "dot": 0.0,
        "delta_max_abs": 0.0,
        "changed": 0.0,
        "count": 0.0,
    }
    base_dtype: str | None = None
    other_dtype: str | None = None
    base_chunks = _iter_tensor_chunks(base_handle, key, torch)
    other_chunks = _iter_tensor_chunks(other_handle, key, torch)
    for base_chunk, other_chunk in zip(base_chunks, other_chunks):
        if tuple(base_chunk.shape) != tuple(other_chunk.shape):
            raise ValueError(f"chunk shape mismatch for {key}")
        if base_chunk.dtype != other_chunk.dtype:
            raise ValueError(f"chunk dtype mismatch for {key}")
        base_dtype = str(base_chunk.dtype)
        other_dtype = str(other_chunk.dtype)
        _assert_tensor_finite(base_chunk, torch, key)
        _assert_tensor_finite(other_chunk, torch, key)
        base_float = base_chunk.detach().to(dtype=torch.float64, device="cpu")
        other_float = other_chunk.detach().to(dtype=torch.float64, device="cpu")
        delta = other_float - base_float
        total["base_sq"] += float(torch.sum(base_float * base_float).item())
        total["other_sq"] += float(torch.sum(other_float * other_float).item())
        total["delta_sq"] += float(torch.sum(delta * delta).item())
        total["dot"] += float(torch.sum(base_float * other_float).item())
        total["delta_max_abs"] = max(total["delta_max_abs"], float(torch.max(torch.abs(delta)).item()))
        total["changed"] += int(torch.count_nonzero(base_chunk != other_chunk).item())
        total["count"] += int(base_chunk.numel())
        digest_base.update(_canonical_tensor_bytes(base_chunk, torch))
        digest_other.update(_canonical_tensor_bytes(other_chunk, torch))
    try:
        next(base_chunks)
        raise ValueError(f"base/other chunk counts differ for {key}")
    except StopIteration:
        pass
    try:
        next(other_chunks)
        raise ValueError(f"base/other chunk counts differ for {key}")
    except StopIteration:
        pass

    count = int(total["count"])
    base_l2 = math.sqrt(max(0.0, total["base_sq"]))
    other_l2 = math.sqrt(max(0.0, total["other_sq"]))
    delta_l2 = math.sqrt(max(0.0, total["delta_sq"]))
    return {
        "shape": list(base_shape),
        "dtype": base_dtype,
        "parameter_count": count,
        "finite": True,
        "base_l2": base_l2,
        "other_l2": other_l2,
        "delta_l2": delta_l2,
        "delta_rms": math.sqrt(max(0.0, total["delta_sq"]) / count) if count else 0.0,
        "delta_max_abs": total["delta_max_abs"],
        "relative_delta_l2": delta_l2 / max(base_l2, EPSILON),
        "cosine_base_other": total["dot"] / (base_l2 * other_l2) if base_l2 > 0 and other_l2 > 0 else 0.0,
        "changed_element_count": int(total["changed"]),
        "changed_element_fraction": total["changed"] / count if count else 0.0,
        "base_tensor_sha256": digest_base.hexdigest(),
        "other_tensor_sha256": digest_other.hexdigest(),
    }


def _row_tensor(handle: Any, key: str, row: int, torch: Any) -> Any:
    tensor_slice = handle.get_slice(key)
    shape = tuple(int(value) for value in tensor_slice.get_shape())
    if not shape or row < 0 or row >= shape[0]:
        raise ValueError(f"category row {row} unavailable for {key} with shape {shape}")
    value = tensor_slice[(row,) + tuple(slice(None) for _ in shape[1:])]
    if not torch.is_tensor(value) or value.device.type != "cpu":
        raise ValueError(f"category slice is not CPU tensor: {key}[{row}]")
    return value


def _numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": percentile(0.5),
        "mean": sum(ordered) / len(ordered),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def _partition_metrics(base: Any, other: Any, torch: Any, label: str) -> dict[str, Any]:
    comparison = _compare_tensor_values(base, other, torch, label=label)
    base_rows = base.detach().to(dtype=torch.float64, device="cpu").reshape(base.shape[0], -1)
    other_rows = other.detach().to(dtype=torch.float64, device="cpu").reshape(other.shape[0], -1)
    delta_rows = other_rows - base_rows
    base_rms = torch.sqrt(torch.mean(base_rows * base_rows, dim=1)).tolist()
    other_rms = torch.sqrt(torch.mean(other_rows * other_rows, dim=1)).tolist()
    delta_rms = torch.sqrt(torch.mean(delta_rows * delta_rows, dim=1)).tolist()
    comparison["per_column_rms"] = {
        "base": _numeric_summary(base_rms),
        "other": _numeric_summary(other_rms),
        "delta": _numeric_summary(delta_rms),
    }
    return comparison


def _module_group(key: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    # The final rule is an explicit fallback, not a competing catch-all
    # regex.  Treating it as an ordinary regex would make every key match two
    # rules and would defeat the exact-one-group invariant.
    for rule in MODULE_GROUP_RULES[:-1]:
        if any(re.search(pattern, key) for pattern in rule["patterns"]):
            matches.append(rule)
    if len(matches) > 1:
        raise ValueError(f"module key matched {len(matches)} rules: {key}")
    rule = matches[0] if matches else MODULE_GROUP_RULES[-1]
    return {
        "group": rule["group"],
        "expected_trainable": rule["expected_trainable"],
        "basis": rule["basis"],
    }


def _aggregate_groups(key_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for key, metrics in key_metrics.items():
        classification = _module_group(key)
        group_name = classification["group"]
        group = groups.setdefault(
            group_name,
            {
                "expected_trainable": classification["expected_trainable"],
                "basis": classification["basis"],
                "key_count": 0,
                "parameter_count": 0,
                "changed_key_count": 0,
                "changed_element_count": 0,
                "base_sq": 0.0,
                "other_sq": 0.0,
                "delta_sq": 0.0,
                "max_per_key_relative_delta": 0.0,
                "keys": [],
            },
        )
        group["key_count"] += 1
        group["parameter_count"] += int(metrics["parameter_count"])
        group["changed_element_count"] += int(metrics["changed_element_count"])
        group["changed_key_count"] += int(metrics["changed_element_count"] > 0)
        group["base_sq"] += float(metrics["base_l2"]) ** 2
        group["other_sq"] += float(metrics["other_l2"]) ** 2
        group["delta_sq"] += float(metrics["delta_l2"]) ** 2
        group["max_per_key_relative_delta"] = max(
            group["max_per_key_relative_delta"], float(metrics["relative_delta_l2"])
        )
        group["keys"].append(key)

    result: dict[str, Any] = {}
    for name, group in sorted(groups.items()):
        base_l2 = math.sqrt(max(0.0, group.pop("base_sq")))
        other_l2 = math.sqrt(max(0.0, group.pop("other_sq")))
        delta_l2 = math.sqrt(max(0.0, group.pop("delta_sq")))
        count = int(group["parameter_count"])
        group["aggregate_base_norm_l2"] = base_l2
        group["aggregate_other_norm_l2"] = other_l2
        group["aggregate_delta_l2"] = delta_l2
        group["aggregate_delta_rms"] = math.sqrt(
            max(0.0, delta_l2 * delta_l2) / count
        ) if count else 0.0
        group["aggregate_relative_delta_l2"] = delta_l2 / max(base_l2, EPSILON)
        group["unexpected_frozen_changes"] = bool(
            not group["expected_trainable"] and group["changed_element_count"] > 0
        )
        group["keys"] = sorted(group["keys"])
        result[name] = group
    return result


def _audit_checkpoint_pair(
    base_checkpoint: Path,
    o1_checkpoint: Path,
    o1_seal: Path,
    output: Path,
    *,
    base_tree_prefix: str,
) -> dict[str, Any]:
    _assert_output_path_is_external(output, base_checkpoint)
    try:
        output.resolve().relative_to(o1_checkpoint.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("output must remain outside the O1 checkpoint tree")
    o1_seal_path = o1_seal.resolve(strict=True)
    seal_sha_status = _o1_seal_sha256_status(o1_seal_path)
    if not seal_sha_status["verified"]:
        rejection = {
            "schema": "STAGE25_O1_PARAMETER_DELTA_AUDIT_V1",
            "status": "REJECT_O1_SEAL_SHA256_MISMATCH",
            "o1_seal_sha256": seal_sha_status,
            "execution_boundary": {
                "model_constructed": False,
                "cuda_used": False,
                "forward": False,
                "inference": False,
                "training": False,
                "pbs_submitted": False,
                "tensor_read": False,
            },
        }
        _write_json_once(output, rejection)
        raise ValueError(
            "O1 checkpoint seal SHA256 mismatch: "
            f"expected {seal_sha_status['expected_o1_seal_sha256']}, "
            f"observed {seal_sha_status['observed_o1_seal_sha256']}"
        )
    base = _checkpoint_metadata(base_checkpoint)
    o1 = _checkpoint_metadata(o1_checkpoint)
    if base["index_sha256"] != EXPECTED_BASE_INDEX_SHA256:
        raise ValueError("BASE index SHA256 does not match the frozen Stage24 identity")
    base_entries = walk_regular_files(base_checkpoint)
    base_tree = stage24_tree_sha256(base_entries, base_tree_prefix)
    if base_tree != EXPECTED_BASE_TREE_SHA256:
        raise ValueError(f"BASE tree SHA256 mismatch: {base_tree}")
    if o1["index_sha256"] != EXPECTED_O1_INDEX_SHA256:
        raise ValueError("O1 index SHA256 does not match the immutable Stage23 seal")
    o1_seal_result = _verify_o1_seal_against_checkpoint(o1_checkpoint, o1_seal_path)
    if not o1_seal_result["pass"]:
        raise ValueError("O1 checkpoint does not match the immutable per-file seal")

    base_keys = set(base["keys"])
    o1_keys = set(o1["keys"])
    if base_keys != o1_keys:
        missing = sorted(base_keys - o1_keys)
        extra = sorted(o1_keys - base_keys)
        raise ValueError(f"serialized key set mismatch; missing={missing[:8]} extra={extra[:8]}")
    missing_category = sorted(set(CATEGORY_PROJECTOR_KEYS) - base_keys)
    if missing_category:
        raise ValueError(f"required category-specific keys missing: {missing_category}")

    base_shards = {
        shard: base_checkpoint / safe_relative_path(shard)
        for shard in sorted(set(base["weight_map"].values()))
    }
    o1_shards = {
        shard: o1_checkpoint / safe_relative_path(shard)
        for shard in sorted(set(o1["weight_map"].values()))
    }
    for path in [*base_shards.values(), *o1_shards.values()]:
        assert_regular(path)

    # Import the tensor backend only after all cheap identity/path gates pass.
    # A missing or mismatched input therefore fails closed even on a
    # dependency-light host without touching model code.
    torch, safe_open = _load_tensor_backend()
    key_metrics: dict[str, dict[str, Any]] = {}
    category_metrics: dict[str, dict[str, Any]] = {}
    with ExitStack() as stack:
        base_handles = {
            shard: stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
            for shard, path in base_shards.items()
        }
        o1_handles = {
            shard: stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
            for shard, path in o1_shards.items()
        }
        for key in sorted(base_keys):
            base_handle = base_handles[base["weight_map"][key]]
            o1_handle = o1_handles[o1["weight_map"][key]]
            metrics = _compare_streamed_key(base_handle, o1_handle, key, torch)
            classification = _module_group(key)
            metrics.update(
                {
                    "key": key,
                    "base_shard": base["weight_map"][key],
                    "o1_shard": o1["weight_map"][key],
                    "module_group": classification["group"],
                }
            )
            key_metrics[key] = metrics

        for key in CATEGORY_PROJECTOR_KEYS:
            base_handle = base_handles[base["weight_map"][key]]
            o1_handle = o1_handles[o1["weight_map"][key]]
            row_result: dict[str, Any] = {
                "key": key,
                "base_shard": base["weight_map"][key],
                "o1_shard": o1["weight_map"][key],
                "rows": {},
            }
            for category_id in EXPECTED_CATEGORY_IDS:
                base_row = _row_tensor(base_handle, key, category_id, torch)
                o1_row = _row_tensor(o1_handle, key, category_id, torch)
                row_result["rows"][str(category_id)] = _compare_tensor_values(
                    base_row, o1_row, torch, label=f"{key}[{category_id}]"
                )
            if key == ACTION_ENCODER_W1_KEY:
                row_result["active_inactive_columns"] = {}
                for category_id in EXPECTED_CATEGORY_IDS:
                    base_row = _row_tensor(base_handle, key, category_id, torch)
                    o1_row = _row_tensor(o1_handle, key, category_id, torch)
                    if base_row.shape[0] != EXPECTED_UNIVERSAL_ACTION_DIM:
                        raise ValueError(
                            f"{key} category row has unexpected action dimension: {base_row.shape}"
                        )
                    row_result["active_inactive_columns"][str(category_id)] = {
                        "active_0_10": _partition_metrics(
                            base_row[0:10], o1_row[0:10], torch, f"{key}[{category_id}].active"
                        ),
                        "inactive_10_132": _partition_metrics(
                            base_row[10:132], o1_row[10:132], torch, f"{key}[{category_id}].inactive"
                        ),
                    }
            category_metrics[key] = row_result

    groups = _aggregate_groups(key_metrics)
    unexpected = [name for name, group in groups.items() if group["unexpected_frozen_changes"]]
    result = {
        "schema": "STAGE25_O1_PARAMETER_DELTA_AUDIT_V1",
        "status": "STOP_UNEXPECTED_FROZEN_DELTA" if unexpected else "PASS",
        "identity": {
            "stage24_base_sha": STAGE24_BASE_SHA,
            "base_index_sha256": base["index_sha256"],
            "base_tree_sha256": base_tree,
            "base_tree_prefix": base_tree_prefix,
            "o1_index_sha256": o1["index_sha256"],
            "o1_seal_sha256": seal_sha_status,
            "o1_seal": o1_seal_result,
            "groot_sha": EXPECTED_GROOT_SHA,
        },
        "module_classification": {
            "recipe": {
                "tune_projector": True,
                "tune_diffusion_model": True,
                "tune_visual": False,
                "tune_llm": False,
                "tune_vlln": False,
            },
            "rules": list(MODULE_GROUP_RULES),
            "unexpected_frozen_groups": unexpected,
        },
        "serialized_model_key_count": len(key_metrics),
        "serialized_model_key_metrics": key_metrics,
        "category_projector_key_metrics": category_metrics,
        "module_group_aggregates": groups,
        "execution_boundary": {
            "model_constructed": False,
            "cuda_used": False,
            "forward": False,
            "inference": False,
            "training": False,
            "pbs_submitted": False,
            "checkpoint_modified": False,
        },
    }
    _write_json_once(output, result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="freeze protocol from Stage23 evidence")
    prepare.add_argument("--phase-a-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)

    audit = subparsers.add_parser("audit", help="compare exact BASE and O1 checkpoint trees")
    audit.add_argument("--base-checkpoint", type=Path, required=True)
    audit.add_argument("--o1-checkpoint", type=Path, required=True)
    audit.add_argument("--o1-seal", type=Path, required=True)
    audit.add_argument("--base-tree-prefix", default=EXPECTED_BASE_TREE_PREFIX)
    audit.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def audit_exit_code(result: Mapping[str, Any]) -> int:
    """Map an audit result to a fail-closed process exit status."""

    status = result.get("status")
    if status == "PASS":
        return 0
    if status == "STOP_UNEXPECTED_FROZEN_DELTA":
        return 1
    return 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "prepare":
            result = _prepare_protocol(args.phase_a_root, args.output)
        else:
            result = _audit_checkpoint_pair(
                args.base_checkpoint,
                args.o1_checkpoint,
                args.o1_seal,
                args.output,
                base_tree_prefix=args.base_tree_prefix,
            )
    except Exception as exc:
        raise SystemExit(f"STAGE25_FAIL_CLOSED: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "prepare":
        return 0
    return audit_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
