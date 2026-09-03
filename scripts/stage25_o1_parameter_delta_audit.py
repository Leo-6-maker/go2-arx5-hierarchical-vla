#!/usr/bin/env python3
"""Prepare or execute the Stage25 BASE -> O1@200 parameter-delta audit.

The audit path is deliberately a direct safetensors comparison.  It never
constructs a GR00T model, imports CUDA, runs a forward pass, performs
inference, trains, or submits a scheduler job.  The preparation path uses
only the immutable Stage23 Phase-A evidence relay and is runnable with the
Python standard library.  The comparison path imports ``torch`` and
``safetensors`` lazily and opens both trees with ``device="cpu"``.  Its O1
root scope is two-layered: the immutable Stage23 ``candidate_200`` core seal
plus an exact five-file Stage25 supplemental metadata allow-list.

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
from tempfile import TemporaryDirectory
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
SUPPLEMENTAL_METADATA_ALLOWLIST = (
    "experiment_cfg/conf.yaml",
    "experiment_cfg/config.yaml",
    "experiment_cfg/dataset_statistics.json",
    "experiment_cfg/final_model_config.json",
    "experiment_cfg/final_processor_config.json",
)
SUPPLEMENTAL_METADATA_CLASSIFICATION = "PRESENT_DAY_UNSEALED_METADATA"
SUPPLEMENTAL_METADATA_MANIFEST_ALGORITHM = (
    'sha256(relative_path + "\\0" + bytes + "\\0" + file_sha256 + "\\n") '
    "over sorted relative_path"
)
SUPPLEMENTAL_METADATA_PRIOR_BINDING = {
    "experiment_cfg/conf.yaml": {
        "status": "EXACT_HISTORICAL_HASH_FOUND",
        "bytes": 5487,
        "sha256": "d183317ff91f530000956aa9ba67ba1e33bad3f1d50e2f95b8c6626f4df8da76",
        "evidence": [
            "Stage23 payload_file_manifest.tsv:candidates/checkpoint_200_metadata/experiment_cfg/conf.yaml",
            "Stage23 configs/candidate_200.run_manifest.json:artifacts.checkpoint-200/experiment_cfg/conf.yaml",
        ],
    },
    "experiment_cfg/config.yaml": {
        "status": "EXACT_HISTORICAL_HASH_FOUND",
        "bytes": 5456,
        "sha256": "6d2d799b3a37ecf45dcf65a3479ec82246669a65d673fd53ce92373d6690ba69",
        "evidence": [
            "Stage23 payload_file_manifest.tsv:candidates/checkpoint_200_metadata/experiment_cfg/config.yaml",
            "Stage23 configs/candidate_200.run_manifest.json:artifacts.checkpoint-200/experiment_cfg/config.yaml",
        ],
    },
    "experiment_cfg/dataset_statistics.json": {
        "status": "EXACT_HISTORICAL_HASH_FOUND",
        "bytes": 58966,
        "sha256": "28ef7a15fc7c540dd38733eddbf55ebdc3fa2f62c3bafa8b75bfff8bbee54450",
        "evidence": [
            "Stage23 payload_file_manifest.tsv:candidates/checkpoint_200_metadata/experiment_cfg/dataset_statistics.json",
            "Stage23 configs/candidate_200.run_manifest.json:artifacts.checkpoint-200/experiment_cfg/dataset_statistics.json",
        ],
    },
    "experiment_cfg/final_model_config.json": {
        "status": "EXACT_HISTORICAL_HASH_FOUND",
        "bytes": 1513,
        "sha256": "4b88c3dd5fc45e871687526df79c4e93e26c803d428d8de54ea3a3e39c6d5246",
        "evidence": [
            "Stage23 payload_file_manifest.tsv:candidates/checkpoint_200_metadata/experiment_cfg/final_model_config.json",
            "Stage23 configs/candidate_200.run_manifest.json:artifacts.checkpoint-200/experiment_cfg/final_model_config.json",
        ],
    },
    "experiment_cfg/final_processor_config.json": {
        "status": "EXACT_HISTORICAL_HASH_FOUND",
        "bytes": 2548004,
        "sha256": "84b890c59dee08ffe2fce36b4c9f95fe689d993e0d3f0cb80c8a4e8da3fee823",
        "evidence": [
            "Stage23 payload_file_manifest.tsv:candidates/checkpoint_200_metadata/experiment_cfg/final_processor_config.json",
            "Stage23 configs/candidate_200.run_manifest.json:artifacts.checkpoint-200/experiment_cfg/final_processor_config.json",
        ],
    },
}
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
    root_path = Path(root)
    if root_path.is_symlink():
        raise ValueError(f"symlink checkpoint root is not allowed: {root_path}")
    root = root_path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"checkpoint root is not a directory: {root}")

    def raise_walk_error(error: OSError) -> None:
        raise ValueError(f"checkpoint walk failed: {error}") from error

    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(
        root, followlinks=False, onerror=raise_walk_error
    ):
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
                    "regular_file": True,
                    "symlink": False,
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


def _normalize_scope_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    raw_path = entry.get("path")
    if not isinstance(raw_path, str):
        raise ValueError("checkpoint scope entry path is not a string")
    relative = safe_relative_path(raw_path)
    if entry.get("symlink") is True:
        raise ValueError(f"symlink file is not allowed: {relative}")
    if entry.get("regular_file") is False:
        raise ValueError(f"checkpoint scope entry is not a regular file: {relative}")
    size = entry.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"invalid byte count for checkpoint scope entry: {relative}")
    digest = entry.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"invalid SHA256 for checkpoint scope entry: {relative}")
    normalized = dict(entry)
    normalized.update(
        {
            "path": relative,
            "bytes": size,
            "sha256": digest,
            "regular_file": True,
            "symlink": False,
        }
    )
    return normalized


def _normalize_sealed_scope_entry(row: Mapping[str, Any]) -> dict[str, Any]:
    relative = row.get("relative_path")
    if not isinstance(relative, str):
        raise ValueError("sealed scope row has no relative_path")
    normalized = _normalize_scope_entry(
        {
            "path": relative,
            "bytes": row.get("bytes"),
            "sha256": row.get("sha256"),
            "regular_file": True,
            "symlink": False,
        }
    )
    normalized["sealed_path"] = row.get("sealed_path", relative)
    return normalized


def stage25_supplemental_manifest_sha256(
    entries: Iterable[Mapping[str, Any]],
) -> str:
    """Hash exactly the five present-day supplemental metadata records."""

    normalized = [_normalize_scope_entry(entry) for entry in entries]
    paths = [entry["path"] for entry in normalized]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate supplemental metadata path")
    if set(paths) != set(SUPPLEMENTAL_METADATA_ALLOWLIST):
        raise ValueError("supplemental metadata manifest is not the exact five-file allow-list")

    digest = hashlib.sha256()
    for entry in sorted(normalized, key=lambda item: item["path"]):
        digest.update(
            f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _verify_supplemental_historical_binding(
    entries: Iterable[Mapping[str, Any]],
    historical_binding: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Require present supplemental metadata to reproduce the historical binding."""

    binding = (
        SUPPLEMENTAL_METADATA_PRIOR_BINDING
        if historical_binding is None
        else historical_binding
    )
    normalized = [_normalize_scope_entry(entry) for entry in entries]
    present_by_path = {entry["path"]: entry for entry in normalized}
    allowed_paths = set(SUPPLEMENTAL_METADATA_ALLOWLIST)
    mismatches: list[dict[str, Any]] = []
    per_file: list[dict[str, Any]] = []

    for path in SUPPLEMENTAL_METADATA_ALLOWLIST:
        present = present_by_path.get(path)
        historical = binding.get(path)
        historical_bytes = (
            historical.get("bytes")
            if isinstance(historical, Mapping)
            else None
        )
        historical_sha256 = (
            historical.get("sha256")
            if isinstance(historical, Mapping)
            else None
        )
        historical_status = (
            historical.get("status")
            if isinstance(historical, Mapping)
            else None
        )
        evidence_sources = (
            historical.get("evidence")
            if isinstance(historical, Mapping)
            else None
        )
        if not isinstance(evidence_sources, list) or not all(
            isinstance(source, str) and source for source in evidence_sources
        ):
            evidence_sources = []

        reasons: list[str] = []
        if not isinstance(historical, Mapping):
            reasons.append("historical binding is missing")
        if historical_status != "EXACT_HISTORICAL_HASH_FOUND":
            reasons.append(
                "historical status is not EXACT_HISTORICAL_HASH_FOUND"
            )
        if (
            isinstance(historical_bytes, bool)
            or not isinstance(historical_bytes, int)
            or historical_bytes < 0
        ):
            reasons.append("historical byte count is invalid or missing")
        if not isinstance(historical_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", historical_sha256
        ):
            reasons.append("historical SHA256 is invalid or missing")
        if not evidence_sources:
            reasons.append("historical evidence sources are missing")
        if present is None:
            reasons.append("present supplemental file is missing")
        else:
            if present["bytes"] != historical_bytes:
                reasons.append("present bytes differ from historical bytes")
            if present["sha256"] != historical_sha256:
                reasons.append("present SHA256 differs from historical SHA256")

        record = {
            "path": path,
            "historical_status": historical_status,
            "historical_bytes": historical_bytes,
            "present_bytes": present["bytes"] if present is not None else None,
            "historical_sha256": historical_sha256,
            "present_sha256": present["sha256"] if present is not None else None,
            "exact_match": not reasons,
            "historical_evidence_sources": evidence_sources,
        }
        per_file.append(record)
        if reasons:
            mismatches.append({"path": path, "reasons": reasons})

    unexpected_history_paths = sorted(set(binding) - allowed_paths)
    if unexpected_history_paths:
        mismatches.append(
            {
                "path": "<historical-binding>",
                "reasons": [
                    "historical binding contains unexpected path(s): "
                    + ", ".join(unexpected_history_paths)
                ],
            }
        )

    passed = not mismatches
    return {
        "SUPPLEMENTAL_METADATA_HISTORY_BINDING_STATUS": (
            "EXACT_MATCH_TO_STAGE23_HISTORICAL_EVIDENCE"
            if passed
            else "MISMATCH_TO_STAGE23_HISTORICAL_EVIDENCE"
        ),
        "SUPPLEMENTAL_METADATA_HISTORY_BINDING_PASS": passed,
        "SUPPLEMENTAL_METADATA_HISTORY_MISMATCHES": mismatches,
        "per_file": per_file,
    }


def _partition_o1_checkpoint_scope(
    actual_entries: Iterable[Mapping[str, Any]],
    sealed_rows: Iterable[Mapping[str, Any]],
    *,
    historical_binding: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Partition a checkpoint into immutable core, five supplementals, and extras."""

    errors: list[str] = []
    invalid_entries: list[str] = []
    actual: list[dict[str, Any]] = []
    for entry in actual_entries:
        try:
            actual.append(_normalize_scope_entry(entry))
        except ValueError as exc:
            raw_path = entry.get("path") if isinstance(entry, Mapping) else None
            label = raw_path if isinstance(raw_path, str) else "<unknown>"
            invalid_entries.append(label)
            errors.append(f"invalid checkpoint entry {label}: {exc}")

    sealed: list[dict[str, Any]] = []
    for row in sealed_rows:
        try:
            sealed.append(_normalize_sealed_scope_entry(row))
        except ValueError as exc:
            errors.append(f"invalid immutable seal row: {exc}")
    if not sealed:
        errors.append("immutable Stage23 seal has no sealed core rows")

    def duplicate_paths(entries: Iterable[Mapping[str, Any]]) -> list[str]:
        counts: dict[str, int] = {}
        for entry in entries:
            path = str(entry["path"])
            counts[path] = counts.get(path, 0) + 1
        return sorted(path for path, count in counts.items() if count > 1)

    duplicate_actual = duplicate_paths(actual)
    duplicate_sealed = duplicate_paths(sealed)
    if duplicate_actual:
        errors.append(f"duplicate checkpoint paths: {duplicate_actual}")
    if duplicate_sealed:
        errors.append(f"duplicate sealed core paths: {duplicate_sealed}")

    actual_by_path = {entry["path"]: entry for entry in actual}
    sealed_by_path = {entry["path"]: entry for entry in sealed}
    actual_paths = set(actual_by_path)
    sealed_paths = set(sealed_by_path)
    supplemental_paths = set(SUPPLEMENTAL_METADATA_ALLOWLIST)
    core_supplemental_overlap = sorted(sealed_paths & supplemental_paths)
    if core_supplemental_overlap:
        errors.append(
            f"supplemental path cannot be in immutable sealed core: {core_supplemental_overlap}"
        )
    permitted_paths = sealed_paths | supplemental_paths
    missing_core = sorted(sealed_paths - actual_paths)
    missing_supplemental = sorted(supplemental_paths - actual_paths)
    forbidden_extra = sorted(actual_paths - permitted_paths)
    if missing_core:
        errors.append(f"missing sealed core file(s): {missing_core}")
    if missing_supplemental:
        errors.append(f"missing supplemental metadata file(s): {missing_supplemental}")
    if forbidden_extra:
        errors.append(f"forbidden unsealed extra file(s): {forbidden_extra}")

    core_mismatches: list[str] = []
    for relative, expected in sorted(sealed_by_path.items()):
        actual_entry = actual_by_path.get(relative)
        if actual_entry is None:
            continue
        if actual_entry["bytes"] != expected["bytes"]:
            core_mismatches.append(f"sealed core size mismatch: {relative}")
        if actual_entry["sha256"] != expected["sha256"]:
            core_mismatches.append(f"sealed core SHA256 mismatch: {relative}")
    errors.extend(core_mismatches)

    supplemental_entries = [
        actual_by_path[path]
        for path in SUPPLEMENTAL_METADATA_ALLOWLIST
        if path in actual_by_path
    ]
    supplemental_manifest_sha256: str | None = None
    if set(entry["path"] for entry in supplemental_entries) == supplemental_paths:
        supplemental_manifest_sha256 = stage25_supplemental_manifest_sha256(
            supplemental_entries
        )
    historical_result = _verify_supplemental_historical_binding(
        supplemental_entries,
        historical_binding,
    )
    errors.extend(
        "supplemental historical binding mismatch: "
        f"{item['path']}: {', '.join(item['reasons'])}"
        for item in historical_result["SUPPLEMENTAL_METADATA_HISTORY_MISMATCHES"]
    )

    return {
        "schema": "STAGE25_TWO_LAYER_PROVENANCE_SCOPE_V1",
        "sealed_core": {
            "classification": "STAGE23_CORE_SEAL: IMMUTABLE_HISTORICAL",
            "scope_rule": "exactly the candidate_200 rows from the immutable Stage23 checkpoint_file_seal.tsv",
            "expected_file_count": len(sealed),
            "observed_file_count": len(sealed_paths & actual_paths),
            "missing_paths": missing_core,
            "mismatches": core_mismatches,
            "entries": sealed,
        },
        "allowed_supplemental_metadata": {
            "classification": SUPPLEMENTAL_METADATA_CLASSIFICATION,
            "expected_file_count": len(SUPPLEMENTAL_METADATA_ALLOWLIST),
            "expected_paths": list(SUPPLEMENTAL_METADATA_ALLOWLIST),
            "observed_file_count": len(supplemental_entries),
            "observed_entries": supplemental_entries,
            "missing_paths": missing_supplemental,
            "manifest_field": "STAGE25_SUPPLEMENTAL_METADATA_MANIFEST_SHA256",
            "manifest_algorithm": SUPPLEMENTAL_METADATA_MANIFEST_ALGORITHM,
            "STAGE25_SUPPLEMENTAL_METADATA_MANIFEST_SHA256": supplemental_manifest_sha256,
            "used_for_tensor_identity": False,
            **historical_result,
        },
        "forbidden_unsealed_extra": {
            "paths": forbidden_extra,
            "invalid_entries": sorted(set(invalid_entries)),
            "fail_closed": True,
        },
        "STAGE25_SUPPLEMENTAL_METADATA_MANIFEST_SHA256": supplemental_manifest_sha256,
        **historical_result,
        "errors": sorted(errors),
        "pass": not errors,
    }


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
    if not seal_sha_status["verified"]:
        return {
            "seal_path": str(seal_path.resolve()),
            "seal_sha256": seal_sha_status["observed_o1_seal_sha256"],
            "expected_o1_seal_sha256": seal_sha_status["expected_o1_seal_sha256"],
            "observed_o1_seal_sha256": seal_sha_status["observed_o1_seal_sha256"],
            "verified": False,
            "expected_file_count": None,
            "actual_file_count": None,
            "entries": [],
            "scope": None,
            "errors": ["O1 checkpoint seal SHA256 mismatch"],
            "pass": False,
        }
    rows = _o1_seal_rows(seal_path)
    actual_entries = walk_regular_files(checkpoint)
    scope = _partition_o1_checkpoint_scope(actual_entries, rows)
    return {
        "seal_path": str(seal_path.resolve()),
        "seal_sha256": seal_sha_status["observed_o1_seal_sha256"],
        "expected_o1_seal_sha256": seal_sha_status["expected_o1_seal_sha256"],
        "observed_o1_seal_sha256": seal_sha_status["observed_o1_seal_sha256"],
        "verified": seal_sha_status["verified"],
        "expected_file_count": len(rows),
        "expected_total_file_count": len(rows) + len(SUPPLEMENTAL_METADATA_ALLOWLIST),
        "actual_file_count": len(actual_entries),
        "entries": rows,
        "scope": scope,
        "errors": scope["errors"],
        "pass": scope["pass"] and seal_sha_status["verified"],
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
    seal_sha_status = _require_o1_seal_sha256(seal_path)
    evaluation = _parse_phase_a_evaluation(evaluation_log_path)
    seal_rows = _o1_seal_rows(seal_path)

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
            "stage23_core_seal_immutable": True,
            "supplemental_metadata_allowlist": list(SUPPLEMENTAL_METADATA_ALLOWLIST),
            "supplemental_metadata_file_count_expected": len(SUPPLEMENTAL_METADATA_ALLOWLIST),
            "supplemental_metadata_used_for_tensor_identity": False,
        },
        "provenance_scope": {
            "stage23_core_seal": {
                "classification": "IMMUTABLE_HISTORICAL",
                "path": str(seal_path),
                "sha256_expected": EXPECTED_O1_SEAL_SHA256,
                "sha256_observed": seal_sha_status["observed_o1_seal_sha256"],
                "immutable": True,
                "scope_rule": "SEALED_CORE is exactly the candidate_200 rows from this seal; the Stage23 seal is not rewritten",
            },
            "supplemental_metadata": {
                "classification": SUPPLEMENTAL_METADATA_CLASSIFICATION,
                "allowlist": list(SUPPLEMENTAL_METADATA_ALLOWLIST),
                "expected_file_count": len(SUPPLEMENTAL_METADATA_ALLOWLIST),
                "used_for_tensor_identity": False,
                "tensor_identity_rule": "Stage23 immutable core seal plus exact O1 index SHA256; supplemental metadata cannot establish safetensor identity",
                "prior_binding_status": SUPPLEMENTAL_METADATA_PRIOR_BINDING,
                "current_checkpoint_manifest_status": "OBSERVED_ON_HOPPER_PENDING_REEXECUTION",
                "SUPPLEMENTAL_METADATA_HISTORY_BINDING_STATUS": "PENDING_PRESENT_ROOT_COMPARISON",
                "SUPPLEMENTAL_METADATA_HISTORY_BINDING_PASS": None,
                "SUPPLEMENTAL_METADATA_HISTORY_MISMATCHES": [],
                "SUPPLEMENTAL_METADATA_HISTORY_BINDING_IMPLEMENTED": True,
                "SUPPLEMENTAL_METADATA_PROMOTED_TO_STAGE23_CORE_SEAL": False,
                "historical_equivalence_required_for_stage25_provenance_gate": True,
                "promoted_to_stage23_core_seal": False,
                "manifest_field": "STAGE25_SUPPLEMENTAL_METADATA_MANIFEST_SHA256",
                "manifest_algorithm": SUPPLEMENTAL_METADATA_MANIFEST_ALGORITHM,
                "STAGE25_SUPPLEMENTAL_METADATA_MANIFEST_SHA256": None,
            },
            "forbidden_unsealed_extra_rule": "Any regular file, symlink, directory escape, or path outside SEALED_CORE union the exact five supplemental paths fails closed before tensor read",
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
        "next_gate": "GPT_PI_REVIEW_AFTER_STAGE25_SUPPLEMENTAL_HISTORY_BINDING_PATCH",
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
        raise ValueError("O1 checkpoint does not match the two-layer provenance scope")

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


def _self_check_entry(path: str, payload: bytes) -> dict[str, Any]:
    return {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "regular_file": True,
        "symlink": False,
    }


def _run_self_check() -> dict[str, Any]:
    """Run dependency-free scope and fail-close checks for Corelab review."""

    checks: list[str] = []

    def require(condition: bool, label: str) -> None:
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    core_entries = [
        _self_check_entry("config.json", b"core-config"),
        _self_check_entry("model.safetensors", b"core-tensor"),
    ]
    sealed_rows = [
        {
            "sealed_path": f"candidate_200/{entry['path']}",
            "relative_path": entry["path"],
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
        }
        for entry in core_entries
    ]
    supplemental_entries = [
        _self_check_entry(path, f"supplemental:{path}".encode("utf-8"))
        for path in SUPPLEMENTAL_METADATA_ALLOWLIST
    ]
    historical_binding = {
        str(entry["path"]): {
            "status": "EXACT_HISTORICAL_HASH_FOUND",
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
            "evidence": ["dependency-free self-check fixture"],
        }
        for entry in supplemental_entries
    }
    complete_entries = core_entries + supplemental_entries
    passed = _partition_o1_checkpoint_scope(
        complete_entries,
        sealed_rows,
        historical_binding=historical_binding,
    )
    require(passed["pass"], "exact allow-list PASS fixture")
    require(
        passed["allowed_supplemental_metadata"]["expected_file_count"] == 5
        and passed["allowed_supplemental_metadata"][
            "STAGE25_SUPPLEMENTAL_METADATA_MANIFEST_SHA256"
        ],
        "supplemental manifest PASS",
    )

    history_sha_changed = [
        {**entry, "sha256": "0" * 64}
        if entry["path"] == "experiment_cfg/config.yaml"
        else entry
        for entry in complete_entries
    ]
    history_sha_result = _partition_o1_checkpoint_scope(
        history_sha_changed,
        sealed_rows,
        historical_binding=historical_binding,
    )
    require(
        not history_sha_result["pass"],
        "supplemental historical SHA mismatch FAIL",
    )

    history_bytes_changed = [
        {**entry, "bytes": entry["bytes"] + 1}
        if entry["path"] == "experiment_cfg/dataset_statistics.json"
        else entry
        for entry in complete_entries
    ]
    history_bytes_result = _partition_o1_checkpoint_scope(
        history_bytes_changed,
        sealed_rows,
        historical_binding=historical_binding,
    )
    require(
        not history_bytes_result["pass"],
        "supplemental historical byte mismatch FAIL",
    )

    missing_history_binding = dict(historical_binding)
    del missing_history_binding["experiment_cfg/config.yaml"]
    missing_history_result = _partition_o1_checkpoint_scope(
        complete_entries,
        sealed_rows,
        historical_binding=missing_history_binding,
    )
    require(
        not missing_history_result["pass"],
        "supplemental historical binding missing FAIL",
    )

    missing_core = _partition_o1_checkpoint_scope(
        [entry for entry in complete_entries if entry["path"] != "config.json"],
        sealed_rows,
        historical_binding=historical_binding,
    )
    require(not missing_core["pass"], "sealed core missing FAIL")

    modified_core = [
        {**entry, "bytes": entry["bytes"] + 1}
        if entry["path"] == "config.json"
        else entry
        for entry in complete_entries
    ]
    modified = _partition_o1_checkpoint_scope(
        modified_core,
        sealed_rows,
        historical_binding=historical_binding,
    )
    require(not modified["pass"], "sealed core mutation FAIL")

    extra = _partition_o1_checkpoint_scope(
        complete_entries + [_self_check_entry("unexpected.txt", b"extra")],
        sealed_rows,
        historical_binding=historical_binding,
    )
    require(not extra["pass"], "arbitrary extra root file FAIL")

    sixth = _partition_o1_checkpoint_scope(
        complete_entries + [_self_check_entry("experiment_cfg/sixth.txt", b"extra")],
        sealed_rows,
        historical_binding=historical_binding,
    )
    require(not sixth["pass"], "sixth experiment_cfg file FAIL")

    symlink_entry = [
        {**entry, "symlink": True}
        if entry["path"] == SUPPLEMENTAL_METADATA_ALLOWLIST[0]
        else entry
        for entry in complete_entries
    ]
    symlink_scope = _partition_o1_checkpoint_scope(
        symlink_entry,
        sealed_rows,
        historical_binding=historical_binding,
    )
    require(not symlink_scope["pass"], "supplemental symlink FAIL")

    traversal = _partition_o1_checkpoint_scope(
        complete_entries + [_self_check_entry("experiment_cfg/../escape", b"escape")],
        sealed_rows,
        historical_binding=historical_binding,
    )
    require(not traversal["pass"], "supplemental path traversal FAIL")

    with TemporaryDirectory(prefix="stage25-scope-self-check-") as temporary:
        temporary_root = Path(temporary)
        checkpoint = temporary_root / "checkpoint"
        checkpoint.mkdir()
        experiment_cfg = checkpoint / "experiment_cfg"
        experiment_cfg.mkdir()
        outside = temporary_root / "outside.txt"
        outside.write_bytes(b"outside")
        os.symlink(outside, experiment_cfg / SUPPLEMENTAL_METADATA_ALLOWLIST[0].split("/", 1)[1])
        try:
            walk_regular_files(checkpoint)
        except ValueError:
            checks.append("filesystem symlink FAIL")
        else:
            raise AssertionError("filesystem symlink was not rejected")

        seal = temporary_root / "checkpoint_file_seal.tsv"
        seal.write_text(
            "candidate_200\tmodel.safetensors\tregular file\t1\t"
            + hashlib.sha256(b"x").hexdigest()
            + "\n",
            encoding="utf-8",
        )
        require(not _o1_seal_sha256_status(seal)["verified"], "Stage23 seal SHA FAIL")
        try:
            _require_o1_seal_sha256(seal)
        except ValueError:
            checks.append("Stage23 seal pre-parse gate FAIL")
        else:
            raise AssertionError("altered Stage23 seal was accepted")

    require(audit_exit_code({"status": "PASS"}) == 0, "PASS exit code")
    require(
        audit_exit_code({"status": "STOP_UNEXPECTED_FROZEN_DELTA"}) != 0,
        "frozen delta exit code",
    )
    require(audit_exit_code({"status": "UNKNOWN"}) != 0, "unknown status exit code")
    return {
        "schema": "STAGE25_PROVENANCE_SCOPE_SELF_CHECK_V1",
        "status": "PASS",
        "checks": checks,
    }


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
    subparsers.add_parser("self-check", help="run dependency-free Corelab self-checks")
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
        elif args.command == "audit":
            result = _audit_checkpoint_pair(
                args.base_checkpoint,
                args.o1_checkpoint,
                args.o1_seal,
                args.output,
                base_tree_prefix=args.base_tree_prefix,
            )
        else:
            result = _run_self_check()
    except Exception as exc:
        raise SystemExit(f"STAGE25_FAIL_CLOSED: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command in {"prepare", "self-check"}:
        return 0
    return audit_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
