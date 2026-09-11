"""Preparation-only side-channel recorder for future sample-fetch exposure.

This module is deliberately separate from model tensors and collator payloads.
A future separately gated dataset patch may call ``record_fetch`` after it has
resolved episode/step identity while returning the original processed datapoint
unchanged.

Two design constraints are explicit:
1. fetch events are *not* optimizer-step events;
2. logging must not add a per-sample ``fsync`` that materially perturbs training.

Each recorder is bound to one rank and one fresh trace file. Events are buffered
and become durable only after ``seal()``. Therefore an unsealed trace must never
be used for exact exposure claims.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "STAGE45F_TRAIN_SAMPLE_FETCH_EVENT_V2"
SEAL_SCHEMA = "STAGE45F_TRAIN_SAMPLE_FETCH_TRACE_SEAL_V1"


class ExposureSideChannelError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ExposureSideChannel:
    """Buffer exact dataset fetch identities without touching model batches."""

    def __init__(
        self,
        path: Path,
        *,
        rank: int,
        split: str = "TRAIN",
        buffer_event_limit: int = 256,
    ) -> None:
        path = Path(path)
        if split != "TRAIN":
            raise ExposureSideChannelError("exposure side channel is TRAIN-only")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
            raise ExposureSideChannelError("rank must be a non-negative integer")
        if not isinstance(buffer_event_limit, int) or isinstance(buffer_event_limit, bool) or buffer_event_limit <= 0:
            raise ExposureSideChannelError("buffer_event_limit must be a positive integer")
        if path.exists() or path.is_symlink():
            raise ExposureSideChannelError(f"exposure trace must be fresh: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=False)
        self.path = path
        self.rank = rank
        self.split = split
        self.buffer_event_limit = buffer_event_limit
        self._ordinal = 0
        self._buffer: list[dict[str, Any]] = []
        self._flushed_event_count = 0
        self._sealed = False

    @property
    def sealed(self) -> bool:
        return self._sealed

    def _flush_buffer(self, *, sync: bool) -> None:
        if not self._buffer:
            if sync:
                with self.path.open("ab") as handle:
                    handle.flush()
                    os.fsync(handle.fileno())
            return
        with self.path.open("a", encoding="utf-8") as handle:
            for event in self._buffer:
                handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            if sync:
                os.fsync(handle.fileno())
        self._flushed_event_count += len(self._buffer)
        self._buffer.clear()

    def record_fetch(
        self,
        *,
        dataset_identity: str,
        episode_index: int,
        source_episode_index: int,
        step_index: int,
        window_id: str,
    ) -> dict[str, Any]:
        if self._sealed:
            raise ExposureSideChannelError("cannot append to a sealed exposure trace")
        if not dataset_identity or not window_id:
            raise ExposureSideChannelError("dataset_identity/window_id must be non-empty")
        for name, value in (
            ("episode_index", episode_index),
            ("source_episode_index", source_episode_index),
            ("step_index", step_index),
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
            "rank": self.rank,
            "split": self.split,
            "validation_e": False,
            "holdout_f": False,
            "optimizer_step": None,
            "optimizer_step_binding": "NOT_CAPTURED_BY_DATASET_SIDECHANNEL",
        }
        self._buffer.append(event)
        if len(self._buffer) >= self.buffer_event_limit:
            # Write buffered text but do not fsync on each sample/fetch.
            self._flush_buffer(sync=False)
        return dict(event)

    def seal(self, seal_path: Path) -> dict[str, Any]:
        if self._sealed:
            raise ExposureSideChannelError("exposure trace is already sealed")
        seal_path = Path(seal_path)
        if seal_path.exists() or seal_path.is_symlink():
            raise ExposureSideChannelError(f"seal path must be fresh: {seal_path}")
        self._flush_buffer(sync=True)
        if self._flushed_event_count != self._ordinal:
            raise ExposureSideChannelError("flushed exposure event count mismatch")
        trace_sha = _sha256_file(self.path)
        seal = {
            "schema": SEAL_SCHEMA,
            "status": "SEALED_FETCH_TRACE_ONLY",
            "trace_path": str(self.path.resolve()),
            "trace_sha256": trace_sha,
            "rank": self.rank,
            "event_count": self._ordinal,
            "split": self.split,
            "optimizer_step_binding": "NOT_CAPTURED_BY_DATASET_SIDECHANNEL",
            "exact_fetch_count_claim_allowed": True,
            "optimizer_step_exposure_claim_allowed": False,
            "epoch_claim_allowed": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "claim_boundary": "This seal proves a completed rank-local dataset-fetch trace only. It does not bind fetches to optimizer steps and does not imply effective epochs.",
        }
        seal_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = seal_path.with_name(seal_path.name + ".tmp")
        temporary.write_text(json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, seal_path)
        self._sealed = True
        return seal


def record_fetch_without_payload_mutation(
    recorder: ExposureSideChannel,
    payload: Any,
    **identity: Any,
) -> Any:
    """Record a fetch and return the exact same payload object by identity."""

    recorder.record_fetch(**identity)
    return payload


__all__ = [
    "ExposureSideChannel",
    "ExposureSideChannelError",
    "SCHEMA",
    "SEAL_SCHEMA",
    "record_fetch_without_payload_mutation",
]
