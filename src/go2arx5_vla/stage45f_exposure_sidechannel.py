"""Preparation-only side-channel recorder for future exact sample exposure.

The recorder is intentionally independent from model tensors and collator
payloads. A future separately gated dataset patch may call ``record_fetch``
after it has resolved episode/step identity, while returning the original
processed datapoint unchanged. This module itself is inert unless constructed
and called explicitly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "STAGE45F_TRAIN_SAMPLE_FETCH_EVENT_V1"


class ExposureSideChannelError(RuntimeError):
    pass


class ExposureSideChannel:
    """Durably append exact dataset fetch identities without touching batches."""

    def __init__(self, path: Path, *, split: str = "TRAIN") -> None:
        path = Path(path)
        if split != "TRAIN":
            raise ExposureSideChannelError("exposure side channel is TRAIN-only")
        if path.exists() or path.is_symlink():
            raise ExposureSideChannelError(f"exposure trace must be fresh: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=False)
        self.path = path
        self.split = split
        self._ordinal = 0

    def record_fetch(
        self,
        *,
        dataset_identity: str,
        episode_index: int,
        source_episode_index: int,
        step_index: int,
        window_id: str,
        rank: int = 0,
    ) -> dict[str, Any]:
        if not dataset_identity or not window_id:
            raise ExposureSideChannelError("dataset_identity/window_id must be non-empty")
        for name, value in (
            ("episode_index", episode_index),
            ("source_episode_index", source_episode_index),
            ("step_index", step_index),
            ("rank", rank),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ExposureSideChannelError(f"{name} must be a non-negative integer")
        self._ordinal += 1
        event = {
            "schema": SCHEMA,
            "fetch_ordinal": self._ordinal,
            "dataset_identity": dataset_identity,
            "episode_index": episode_index,
            "source_episode_index": source_episode_index,
            "step_index": step_index,
            "window_id": window_id,
            "rank": rank,
            "split": self.split,
            "validation_e": False,
            "holdout_f": False,
            "optimizer_step": None,
            "optimizer_step_binding": "NOT_CAPTURED_BY_DATASET_SIDECHANNEL",
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return event


__all__ = ["ExposureSideChannel", "ExposureSideChannelError", "SCHEMA"]
