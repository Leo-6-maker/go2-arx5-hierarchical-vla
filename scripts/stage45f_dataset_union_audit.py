#!/usr/bin/env python3
"""CPU-only lineage audit for a hypothetical expanded TRAIN dataset union.

The tool validates a manifest-level union of a frozen base TRAIN set and newly
added TRAIN windows.  It never opens dataset payload bytes or executes training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


class DatasetUnionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetUnionError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _window_list(value: Any, name: str, *, new: bool) -> list[dict[str, Any]]:
    require(isinstance(value, list), f"{name} must be list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        require(isinstance(item, Mapping), f"{name}[{index}] must be object")
        window_id = item.get("window_id")
        require(isinstance(window_id, str) and window_id, f"{name}[{index}].window_id missing")
        require(window_id not in seen, f"duplicate {name} window_id: {window_id}")
        seen.add(window_id)
        source_episode_id = item.get("source_episode_id")
        target_id = item.get("target_id")
        require(isinstance(source_episode_id, str) and source_episode_id, f"{name}[{index}].source_episode_id missing")
        require(isinstance(target_id, str) and target_id, f"{name}[{index}].target_id missing")
        record = {
            "window_id": window_id,
            "source_episode_id": source_episode_id,
            "target_id": target_id,
        }
        if new:
            label_source = item.get("label_source")
            require(label_source in {"INDEPENDENT_EXPERT", "SIMULATOR_EXPERT", "ORACLE", "HUMAN_EXPERT"}, f"{name}[{index}].label_source invalid")
            require(item.get("validation_or_holdout_origin") is False, f"{name}[{index}] comes from locked split")
            label_artifact_sha256 = item.get("label_artifact_sha256")
            require(isinstance(label_artifact_sha256, str) and len(label_artifact_sha256) == 64, f"{name}[{index}].label_artifact_sha256 invalid")
            record.update({
                "label_source": label_source,
                "validation_or_holdout_origin": False,
                "label_artifact_sha256": label_artifact_sha256,
            })
        result.append(record)
    return result


def analyze(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "input must be object")
    require(value.get("schema") == "STAGE45F_DATASET_UNION_PROPOSAL_V1", "schema drift")
    require(value.get("source_split") == "TRAIN", "union must be TRAIN-only")
    require(value.get("validation_e_read") is False and value.get("holdout_f_read") is False, "locked split read indicated")
    base_tree = value.get("base_dataset_tree_sha256")
    require(isinstance(base_tree, str) and len(base_tree) == 64, "base dataset tree invalid")
    base = _window_list(value.get("base_windows"), "base_windows", new=False)
    added = _window_list(value.get("added_windows"), "added_windows", new=True)
    require(base, "base_windows must be non-empty")
    require(added, "added_windows must be non-empty")
    base_ids = {item["window_id"] for item in base}
    added_ids = {item["window_id"] for item in added}
    overlap = sorted(base_ids.intersection(added_ids))
    require(not overlap, f"new windows overlap base identities: {overlap[:10]}")
    union = [*base, *added]
    union_ids = [item["window_id"] for item in union]
    require(len(union_ids) == len(set(union_ids)), "union window IDs are not unique")
    target_counts: dict[str, int] = {}
    for item in union:
        target_counts[item["target_id"]] = target_counts.get(item["target_id"], 0) + 1
    return {
        "schema": "STAGE45F_DATASET_UNION_AUDIT_V1",
        "status": "PASS_MANIFEST_ONLY",
        "decision_state": "PI_DECISION_REQUIRED",
        "base_dataset_tree_sha256": base_tree,
        "base_window_count": len(base),
        "added_window_count": len(added),
        "union_window_count": len(union),
        "base_window_ids_preserved_exactly": True,
        "new_window_ids_disjoint_from_base": True,
        "target_window_counts": dict(sorted(target_counts.items())),
        "added_label_source_counts": {
            source: sum(item["label_source"] == source for item in added)
            for source in sorted({item["label_source"] for item in added})
        },
        "union_manifest_sha256": canonical_sha(union),
        "dataset_payload_tree_not_yet_verified": True,
        "training_authorized": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "required_future_closure": [
            "materialized expanded dataset tree hash",
            "payload file inventory",
            "split lock proving Validation-E/Holdout-F exclusion",
            "stats-route decision after current-stats support audit",
            "sampling-mix contract",
        ],
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
    except DatasetUnionError as exc:
        print(f"STAGE45F_DATASET_UNION_AUDIT_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
