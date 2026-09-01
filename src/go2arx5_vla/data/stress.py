"""Controlled synchronization stress cases for Stage19 qualification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable

import numpy as np

from .quality import EpisodeQualityGateConfig, EpisodeQualityGateV1
from .readiness import CalibrationBundleV1, CameraStreamV1, RawEpisodeV1, SynchronizerV1


def _raw_fingerprint(raw: RawEpisodeV1) -> str:
    payload = json.dumps(
        raw.as_dict(include_arrays=True, include_camera_paths=True),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _clone_raw(
    raw: RawEpisodeV1,
    *,
    native_timestamps: Any | None = None,
    camera_streams: dict[str, Any] | None = None,
) -> RawEpisodeV1:
    """Make a test-only source view without invoking corrective code."""

    clone = object.__new__(RawEpisodeV1)
    for field in fields(RawEpisodeV1):
        object.__setattr__(clone, field.name, getattr(raw, field.name))
    if native_timestamps is not None:
        object.__setattr__(clone, "native_timestamps", np.array(native_timestamps, dtype=np.float64, copy=True))
    if camera_streams is not None:
        object.__setattr__(clone, "camera_streams", MappingProxyType(dict(camera_streams)))
    return clone


def _camera_view(stream: CameraStreamV1, timestamps: Any, *, allow_invalid: bool = False) -> Any:
    values = np.array(timestamps, dtype=np.float64, copy=True)
    monotonic = values.size <= 1 or bool(np.all(np.diff(values) > 0.0))
    if not monotonic and not allow_invalid:
        raise ValueError("test camera view is not monotonic")
    if monotonic:
        frame_paths = tuple(stream.frame_paths)
        if len(frame_paths) != values.size:
            # The only shortened source case in this stress suite is a single
            # dropped frame at the same index as the timestamp deletion.
            if len(frame_paths) == values.size + 1:
                drop_index = min(10, len(frame_paths) - 1)
                frame_paths = tuple(
                    path for index, path in enumerate(frame_paths) if index != drop_index
                )
            else:
                frame_paths = frame_paths[: values.size]
        return CameraStreamV1(
            name=stream.name,
            frame_paths=frame_paths,
            timestamps=values,
            declared_fps=stream.declared_fps,
            frame=stream.frame,
            provenance=stream.provenance,
            intrinsics=stream.intrinsics,
            extrinsics=stream.extrinsics,
        )
    # CameraStreamV1 intentionally rejects malformed streams in normal code.
    # The duck-typed view exists only so the quality gate can prove that it
    # rejects preserved bad source data rather than silently repairing it.
    return SimpleNamespace(
        name=stream.name,
        frame_paths=stream.frame_paths,
        timestamps=values,
        declared_fps=stream.declared_fps,
        frame=stream.frame,
        provenance=stream.provenance,
        intrinsics=stream.intrinsics,
        extrinsics=stream.extrinsics,
        frame_count=int(values.size),
    )


def _camera_case(
    raw: RawEpisodeV1,
    transform: Callable[[np.ndarray, str], np.ndarray],
    *,
    allow_invalid: bool = False,
) -> RawEpisodeV1:
    streams = {}
    for name, stream in raw.camera_streams.items():
        streams[name] = _camera_view(
            stream,
            transform(np.asarray(stream.timestamps, dtype=np.float64), name),
            allow_invalid=allow_invalid,
        )
    return _clone_raw(raw, camera_streams=streams)


def _drop_one(values: np.ndarray) -> np.ndarray:
    if values.size < 2:
        return values.copy()
    return np.delete(values, min(10, values.size - 1))


def _duplicate_one(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    index = min(10, values.size - 1)
    return np.insert(values, index, values[index])


def _make_non_monotonic(values: np.ndarray) -> np.ndarray:
    if values.size < 2:
        return values.copy()
    index = min(10, values.size - 1)
    result = values.copy()
    result[index] = result[index - 1] - 0.001
    return result


def run_synchronization_stress(raw: RawEpisodeV1) -> list[dict[str, Any]]:
    """Run jitter/drop/clock-corruption cases and record gate outcomes."""

    original_fingerprint = _raw_fingerprint(raw)
    base_calibration = CalibrationBundleV1(
        bundle_id=raw.calibration_bundle_id,
        version="D3_V0",
        status="SIMULATION_ONLY",
    )
    gate = EpisodeQualityGateV1(
        EpisodeQualityGateConfig(
            max_sync_skew_warn_s=0.010,
            max_sync_skew_reject_s=0.050,
            simulation_fixture_allowed=True,
        )
    )
    cases: list[tuple[str, RawEpisodeV1]] = []
    cases.append(
        (
            "camera_jitter",
            _camera_case(
                raw,
                lambda timestamps, name: timestamps
                + (0.002 * np.sin(np.arange(timestamps.size) * 1.7) if name == "body" else 0.001 * np.cos(np.arange(timestamps.size))),
            ),
        )
    )
    native = np.asarray(raw.native_timestamps, dtype=np.float64)
    state_jitter = native + 0.001 * np.sin(np.arange(native.size) * 0.7)
    cases.append(("state_jitter", _clone_raw(raw, native_timestamps=state_jitter)))
    cases.append(
        (
            "dropped_wrist_frame",
            _camera_case(
                raw,
                lambda timestamps, name: _drop_one(timestamps) if name == "wrist" else timestamps,
            ),
        )
    )
    cases.append(
        (
            "duplicated_wrist_timestamp",
            _camera_case(
                raw,
                lambda timestamps, name: np.array(
                    _duplicate_one(timestamps) if name == "wrist" else timestamps,
                    dtype=np.float64,
                ),
                allow_invalid=True,
            ),
        )
    )
    cases.append(
        (
            "non_monotonic_wrist_timestamp",
            _camera_case(
                raw,
                lambda timestamps, name: np.array(
                    _make_non_monotonic(timestamps) if name == "wrist" else timestamps,
                    dtype=np.float64,
                ),
                allow_invalid=True,
            ),
        )
    )
    cases.append(
        (
            "bounded_clock_offset",
            _camera_case(
                raw,
                lambda timestamps, name: timestamps + (0.020 if name == "wrist" else 0.0),
            ),
        )
    )
    cases.append(
        (
            "slow_clock_drift",
            _camera_case(
                raw,
                lambda timestamps, name: timestamps
                + (np.linspace(0.0, 0.020, timestamps.size) if name == "wrist" else 0.0),
            ),
        )
    )
    cases.append(
        (
            "body_wrist_async_reject",
            _camera_case(
                raw,
                lambda timestamps, name: timestamps + (0.060 if name == "body" else 0.120),
            ),
        )
    )

    results: list[dict[str, Any]] = []
    for name, candidate in cases:
        synchronization = None
        sync_error = None
        try:
            synchronization = SynchronizerV1().synchronize(candidate)
        except Exception as exc:  # malformed source should be explicitly visible
            sync_error = {"type": type(exc).__name__, "message": str(exc).splitlines()[0]}
        quality = gate.evaluate(
            candidate,
            synchronization=synchronization,
            calibration=base_calibration,
        )
        results.append(
            {
                "case": name,
                "quality_decision": quality.decision.value,
                "issue_codes": [issue.code for issue in quality.issues],
                "synchronizer_error": sync_error,
                "synchronization": None if synchronization is None else synchronization.as_dict(),
                "source_timestamps_preserved": True,
                "corrective_action": quality.provenance.get("corrective_action"),
                "expected_behavior": (
                    "REJECT malformed/unsafe source, WARN bounded perturbation, "
                    "and never alter source arrays"
                ),
            }
        )
    final_fingerprint = _raw_fingerprint(raw)
    return [
        {
            "schema": "Stage19SynchronizationStressV1",
            "case_results": results,
            "source_unchanged": original_fingerprint == final_fingerprint,
            "source_fingerprint_before": original_fingerprint,
            "source_fingerprint_after": final_fingerprint,
            "corrective_action": "NONE",
            "production_action_timing": "TBD_BY_ADAPTATION_DATASET",
        }
    ]
