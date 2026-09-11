#!/usr/bin/env python3
"""Manifest-only exact-duplicate audit for hypothetical expanded TRAIN data.

Different window IDs do not guarantee new information. This tool checks caller-
supplied immutable source/label fingerprints for exact duplicate payloads across
accepted base windows and proposed new windows. It does not attempt perceptual or
near-duplicate similarity and does not open dataset payloads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


class DuplicateAuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DuplicateAuditError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(value: Any, field: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value), f"{field} must be lowercase sha256")
    return value


def _records(value: Any, name: str, *, proposed: bool) -> list[dict[str, Any]]:
    require(isinstance(value, list), f"{name} must be list")
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, item in enumerate(value):
        require(isinstance(item, Mapping), f"{name}[{index}] must be object")
        window_id = item.get("window_id")
        require(isinstance(window_id, str) and window_id, f"{name}[{index}].window_id missing")
        require(window_id not in ids, f"duplicate window id within {name}: {window_id}")
        ids.add(window_id)
        source_sha = _sha(item.get("source_state_observation_sha256"), f"{name}[{index}].source_state_observation_sha256")
        label_sha = _sha(item.get("desired_future_eef_label_sha256"), f"{name}[{index}].desired_future_eef_label_sha256")
        combined_sha = _sha(item.get("window_payload_fingerprint_sha256"), f"{name}[{index}].window_payload_fingerprint_sha256")
        if proposed:
            require(item.get("validation_or_holdout_origin") is False, f"{name}[{index}] locked split origin")
            require(item.get("label_source") in {"INDEPENDENT_EXPERT", "SIMULATOR_EXPERT", "ORACLE", "HUMAN_EXPERT"}, f"{name}[{index}] label provenance invalid")
        result.append({
            "window_id": window_id,
            "source_state_observation_sha256": source_sha,
            "desired_future_eef_label_sha256": label_sha,
            "window_payload_fingerprint_sha256": combined_sha,
        })
    return result


def analyze(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "request must be object")
    require(value.get("schema") == "STAGE45F_DATASET_DUPLICATE_AUDIT_REQUEST_V1", "schema drift")
    require(value.get("source_split") == "TRAIN", "audit is TRAIN-only")
    require(value.get("validation_e_read") is False and value.get("holdout_f_read") is False, "locked split read indicated")
    base = _records(value.get("base_windows"), "base_windows", proposed=False)
    added = _records(value.get("added_windows"), "added_windows", proposed=True)
    require(base and added, "base and added windows must both be non-empty")

    def index(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for item in records:
            result.setdefault(str(item[field]), []).append(str(item["window_id"]))
        return result

    combined = [*base, *added]
    exact_payload_groups = [sorted(ids) for ids in index(combined, "window_payload_fingerprint_sha256").values() if len(ids) > 1]
    exact_source_groups = [sorted(ids) for ids in index(combined, "source_state_observation_sha256").values() if len(ids) > 1]
    exact_label_groups = [sorted(ids) for ids in index(combined, "desired_future_eef_label_sha256").values() if len(ids) > 1]
    base_payloads = {item["window_payload_fingerprint_sha256"] for item in base}
    added_exact_base_duplicates = sorted(
        item["window_id"] for item in added if item["window_payload_fingerprint_sha256"] in base_payloads
    )
    added_internal_duplicates = [
        sorted(ids) for ids in index(added, "window_payload_fingerprint_sha256").values() if len(ids) > 1
    ]
    return {
        "schema": "STAGE45F_DATASET_EXACT_DUPLICATE_AUDIT_V1",
        "status": "PASS_MANIFEST_ANALYSIS_ONLY",
        "decision_state": "PI_DECISION_REQUIRED",
        "base_window_count": len(base),
        "added_window_count": len(added),
        "added_exact_duplicates_of_base_window_ids": added_exact_base_duplicates,
        "added_internal_exact_duplicate_groups": added_internal_duplicates,
        "all_exact_payload_duplicate_groups": exact_payload_groups,
        "all_exact_source_state_observation_duplicate_groups": exact_source_groups,
        "all_exact_label_duplicate_groups": exact_label_groups,
        "exact_payload_duplicate_present": bool(exact_payload_groups),
        "exact_source_reuse_present": bool(exact_source_groups),
        "exact_label_reuse_present": bool(exact_label_groups),
        "near_duplicate_or_perceptual_similarity_evaluated": False,
        "duplicate_windows_automatically_removed": False,
        "data_collection_authorized": False,
        "training_authorized": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "claim_boundary": "Only exact caller-supplied fingerprints are compared. Shared labels or observations may be scientifically legitimate; this tool reports them rather than defining an automatic exclusion threshold. Near-duplicates are not assessed.",
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(analyze(read_json(args.input)), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DuplicateAuditError as exc:
        print(f"STAGE45F_DATASET_DUPLICATE_AUDIT_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
