#!/usr/bin/env python3
"""Normalize sealed Stage45E TRAIN24 row bundles for neutral PI comparison.

The adapter reads only explicitly supplied, already-derived per-window row JSON
artifacts and verifies their hashes, schema, checkpoint labels/tree identities,
window order, TRAIN24 cardinality, and explicit upstream evidence lineage. It
never reads model checkpoints, raw tensors, datasets, Validation-E, Holdout-F,
or GPU state.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "manifests/STAGE45E_NEUTRAL_CANDIDATE_SELECTION_PREP_V1.json"
FORBIDDEN_TOP_LEVEL_IMPORTS = {
    "torch",
    "transformers",
    "gr00t",
    "jax",
    "tensorflow",
    "isaacgym",
    "omni",
    "subprocess",
    "numpy",
}


class CandidateEvidenceAdapterError(RuntimeError):
    """Raised when a sealed row bundle violates the neutral adapter contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateEvidenceAdapterError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"JSON input is unavailable or unsafe: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), f"output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_string(value: Any, field: str) -> str:
    require(isinstance(value, str) and len(value) == 64, f"{field} must be 64-char SHA256")
    require(all(character in "0123456789abcdef" for character in value.lower()), f"{field} is not hexadecimal")
    return value.lower()


def manifest() -> dict[str, Any]:
    value = read_json(MANIFEST_PATH)
    require(value.get("schema") == "STAGE45E_NEUTRAL_CANDIDATE_SELECTION_PREP_V1", "manifest schema drift")
    require(value.get("status") == "PREP_ONLY_NO_EXECUTION", "manifest is not prep-only")
    provenance = value.get("required_candidate_provenance")
    require(isinstance(provenance, Mapping), "manifest candidate provenance contract missing")
    require(provenance.get("row_checkpoint_sha_must_equal_checkpoint_tree_sha256") is True, "row/checkpoint provenance binding relaxed")
    return value


def static_guard() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    forbidden = sorted(imports.intersection(FORBIDDEN_TOP_LEVEL_IMPORTS))
    return {
        "schema": "STAGE45E_TRAIN24_ROW_ADAPTER_STATIC_GUARD_V1",
        "status": "PASS" if not forbidden else "BLOCKED",
        "forbidden_top_level_imports": forbidden,
        "model_construction": 0,
        "model_forward": 0,
        "training": 0,
        "gpu": 0,
        "pbs": 0,
        "validation_e_read": False,
        "holdout_f_read": False,
        "wbc": 0,
        "simulator": 0,
        "hardware": 0,
    }


def plan() -> dict[str, Any]:
    cfg = manifest()
    guard = static_guard()
    require(guard["status"] == "PASS", "static guard failed")
    return {
        "schema": "STAGE45E_TRAIN24_ROW_ADAPTER_PLAN_V1",
        "status": "PASS_STATIC_NOT_EXECUTED",
        "authority": cfg["authority"],
        "adapter_input_schema": cfg["adapter_input_schema"],
        "accepted_row_schema": cfg["accepted_row_schema"],
        "candidate_order": cfg["candidate_order"],
        "source_checkpoint_label": cfg["source_checkpoint_label"],
        "required_candidate_provenance": cfg["required_candidate_provenance"],
        "row_metric_mapping": cfg["row_metric_mapping"],
        "normalized_output_schema": cfg["normalized_input_schema"],
        "static_guard": guard,
    }


def _finite(value: Any, field: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{field} must be numeric")
    number = float(value)
    require(math.isfinite(number), f"{field} must be finite")
    return number


def _metric_path(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        require(isinstance(current, Mapping) and part in current, f"row metric path missing: {dotted}")
        current = current[part]
    return current


def _extract_metrics(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    mapping = cfg["row_metric_mapping"]
    optional = set(cfg["optional_numeric_metrics"])
    result: dict[str, Any] = {}
    for metric in cfg["numeric_metrics"]:
        value = _metric_path(row, mapping[metric])
        if value is None:
            require(metric in optional, f"required metric is null: {metric}")
            result[metric] = None
        else:
            result[metric] = _finite(value, metric)
    sign = _metric_path(row, mapping["progress_sign"])
    require(isinstance(sign, str) and sign in {"POSITIVE", "ZERO", "NEGATIVE"}, "progress_sign drift")
    result["progress_sign"] = sign
    return result


def _checkpoint_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    checkpoint = row.get("checkpoint")
    require(isinstance(checkpoint, Mapping), "row checkpoint identity missing")
    label = checkpoint.get("label")
    require(isinstance(label, str) and label, "row checkpoint label missing")
    sha256 = _sha256_string(checkpoint.get("sha256"), "row checkpoint.sha256")
    return label, sha256


def _validate_provenance(canonical_label: str, provenance: Any, cfg: Mapping[str, Any]) -> dict[str, Any]:
    require(isinstance(provenance, Mapping), f"{canonical_label}: provenance missing")
    contract = cfg["required_candidate_provenance"]
    result = dict(provenance)
    for field in contract["sha256_fields"]:
        result[field] = _sha256_string(provenance.get(field), f"{canonical_label}: provenance.{field}")
    for field in contract["identity_fields"]:
        value = provenance.get(field)
        require(isinstance(value, str) and value.strip(), f"{canonical_label}: provenance.{field} missing")
        result[field] = value.strip()
    require(
        result["lineage_kind"] in set(contract["allowed_lineage_kinds"]),
        f"{canonical_label}: provenance.lineage_kind invalid",
    )
    return result


def _normalize_candidate(
    canonical_label: str,
    spec: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_path_value = spec.get("rows_json")
    expected_sha = spec.get("rows_sha256")
    require(isinstance(rows_path_value, str) and rows_path_value, f"{canonical_label}: rows_json missing")
    expected_sha = _sha256_string(expected_sha, f"{canonical_label}: rows_sha256")
    provenance = _validate_provenance(canonical_label, spec.get("provenance"), cfg)
    checkpoint_tree_sha256 = provenance["checkpoint_tree_sha256"]

    rows_path = Path(rows_path_value).expanduser().resolve()
    observed_sha = sha256_file(rows_path)
    require(observed_sha == expected_sha, f"{canonical_label}: rows JSON SHA mismatch")
    payload = read_json(rows_path)
    require(isinstance(payload, list) and len(payload) == cfg["window_count"], f"{canonical_label}: row count drift")
    expected_checkpoint_label = cfg["source_checkpoint_label"][canonical_label]
    normalized_rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        require(isinstance(row, Mapping), f"{canonical_label}: row {index} is not an object")
        require(row.get("schema") == cfg["accepted_row_schema"], f"{canonical_label}: row schema drift")
        row_checkpoint_label, row_checkpoint_sha = _checkpoint_identity(row)
        require(row_checkpoint_label == expected_checkpoint_label, f"{canonical_label}: checkpoint label drift")
        require(
            row_checkpoint_sha == checkpoint_tree_sha256,
            f"{canonical_label}: row checkpoint SHA does not match provenance checkpoint tree",
        )
        window_id = row.get("window_id")
        target_id = row.get("target_id")
        require(isinstance(window_id, str) and window_id, f"{canonical_label}: window_id missing")
        require(isinstance(target_id, str) and target_id, f"{canonical_label}: target_id missing")
        require(row.get("target_xyz_in_model_input") is False, f"{canonical_label}:{window_id}: target XYZ input drift")
        require(row.get("finite") is True, f"{canonical_label}:{window_id}: row is not finite")
        normalized_rows.append(
            {
                "window_id": window_id,
                "target_id": target_id,
                "metrics": _extract_metrics(row, cfg),
            }
        )
    require(len({row["window_id"] for row in normalized_rows}) == cfg["window_count"], f"{canonical_label}: duplicate window ids")
    normalized_provenance = dict(provenance)
    normalized_provenance.update(
        {
            "canonical_candidate_label": canonical_label,
            "rows_json_sha256": observed_sha,
            "source_checkpoint_label": expected_checkpoint_label,
            "accepted_row_schema": cfg["accepted_row_schema"],
            "row_checkpoint_tree_binding": "EXACT",
        }
    )
    return normalized_rows, normalized_provenance


def adapt(index: Mapping[str, Any]) -> dict[str, Any]:
    cfg = manifest()
    require(isinstance(index, Mapping), "adapter index must be an object")
    require(index.get("schema") == cfg["adapter_input_schema"], "adapter index schema drift")
    require(index.get("split") == "TRAIN24", "adapter accepts TRAIN24 only")
    require(index.get("validation_e_read") is False, "adapter index indicates Validation-E read")
    require(index.get("holdout_f_read") is False, "adapter index indicates Holdout-F read")
    candidates = index.get("candidates")
    require(isinstance(candidates, Mapping), "adapter candidates missing")
    require(list(candidates.keys()) == cfg["candidate_order"], "adapter candidate order/set drift")

    normalized_candidates: dict[str, Any] = {}
    reference_windows: list[str] | None = None
    reference_targets: dict[str, str] | None = None
    for label in cfg["candidate_order"]:
        rows, provenance = _normalize_candidate(label, candidates[label], cfg)
        window_ids = [row["window_id"] for row in rows]
        targets = {row["window_id"]: row["target_id"] for row in rows}
        if reference_windows is None:
            reference_windows = window_ids
            reference_targets = targets
        else:
            require(window_ids == reference_windows, f"{label}: TRAIN24 window order mismatch")
            require(targets == reference_targets, f"{label}: target identity mismatch")
        normalized_candidates[label] = {"provenance": provenance, "rows": rows}

    require(reference_windows is not None and len(reference_windows) == cfg["window_count"], "reference TRAIN24 windows missing")
    return {
        "schema": cfg["normalized_input_schema"],
        "split": "TRAIN24",
        "window_ids": reference_windows,
        "validation_e_read": False,
        "holdout_f_read": False,
        "candidates": normalized_candidates,
        "adapter": {
            "schema": "STAGE45E_TRAIN24_ROW_ADAPTER_RESULT_V1",
            "status": "PASS_NORMALIZED_FROM_SEALED_ROWS",
            "automatic_winner": False,
            "model_construction": 0,
            "model_forward": 0,
            "training": 0,
            "gpu": 0,
            "pbs": 0,
        },
    }


def adapt_from_file(index_path: Path, output_path: Path) -> dict[str, Any]:
    index = read_json(index_path)
    normalized = adapt(index)
    normalized["adapter"]["index_sha256"] = sha256_file(index_path)
    write_json(output_path, normalized)
    return normalized


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    modes = value.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--adapt", action="store_true")
    value.add_argument("--index", type=Path)
    value.add_argument("--output", type=Path)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.plan_only:
        print(json.dumps(plan(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    require(args.index is not None and args.output is not None, "--adapt requires --index and --output")
    normalized = adapt_from_file(args.index, args.output)
    print(json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CandidateEvidenceAdapterError as exc:
        print(f"STAGE45E_CANDIDATE_EVIDENCE_ADAPTER_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
