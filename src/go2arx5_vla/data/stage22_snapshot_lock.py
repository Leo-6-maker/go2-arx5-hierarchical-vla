"""Frozen Stage22 validation snapshot identity and seed contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SNAPSHOT_LOCK_SCHEMA = "STAGE22_VALIDATION_SNAPSHOT_LOCK_V1"
EXPECTED_VALIDATION_TARGET_IDS = ("red_target_holdout_e",)
EXPECTED_VALIDATION_SNAPSHOT_IDS = tuple(
    f"stage22_expert_red_target_holdout_e:expert-{index:04d}" for index in range(6)
)
EXPECTED_INFERENCE_SEEDS = (2223, 2224, 2225)


def snapshot_ids_sha256(snapshot_ids: list[str] | tuple[str, ...]) -> str:
    """Hash the ordered snapshot-ID list used by the frozen lock."""

    normalized = [str(value) for value in snapshot_ids]
    return hashlib.sha256(("\n".join(normalized) + "\n").encode("utf-8")).hexdigest()


def validate_snapshot_lock(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the exact Stage22 validation snapshot lock."""

    if payload.get("schema") != SNAPSHOT_LOCK_SCHEMA:
        raise ValueError("unexpected Stage22 validation snapshot lock schema")
    if payload.get("status") != "LOCKED_BEFORE_H2_INFERENCE":
        raise ValueError("validation snapshot lock is not frozen before H2 inference")
    if payload.get("split") != "validation":
        raise ValueError("validation snapshot lock must describe split=validation")
    if payload.get("target_ids") != list(EXPECTED_VALIDATION_TARGET_IDS):
        raise ValueError("validation snapshot target lock differs from red_target_holdout_e")
    snapshot_ids = payload.get("snapshot_ids")
    expected_ids = list(EXPECTED_VALIDATION_SNAPSHOT_IDS)
    if snapshot_ids != expected_ids:
        raise ValueError("validation snapshot IDs differ from the frozen six-snapshot set")
    if payload.get("snapshot_count") != len(expected_ids):
        raise ValueError("validation snapshot count does not equal the frozen set")
    if payload.get("expected_overlap_count") != len(expected_ids) - 1:
        raise ValueError("validation snapshot overlap count does not match the frozen sequence")
    if payload.get("snapshot_ids_sha256") != snapshot_ids_sha256(expected_ids):
        raise ValueError("validation snapshot ID-list hash does not match its contents")
    return dict(payload)


def load_snapshot_lock(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"snapshot lock is not a JSON object: {path}")
    return validate_snapshot_lock(value)


def snapshot_lock_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
