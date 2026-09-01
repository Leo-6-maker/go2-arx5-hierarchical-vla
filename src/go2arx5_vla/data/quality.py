"""Non-corrective quality gate for raw robot episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from ..trajectory.so3 import geodesic_angle, rot6d_rows_to_matrix
from .readiness import CalibrationBundleV1, RawEpisodeV1, SynchronizationResultV1, _jsonable


class EpisodeGateDecision(str, Enum):
    ACCEPT = "ACCEPT"
    WARN = "WARN"
    REJECT = "REJECT"


class EpisodeIssueSeverity(str, Enum):
    WARNING = "WARNING"
    REJECT = "REJECT"


@dataclass(frozen=True)
class EpisodeQualityIssue:
    code: str
    severity: EpisodeIssueSeverity
    message: str
    provenance: str
    observed: Any = None
    threshold: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "provenance": self.provenance,
            "observed": _jsonable(self.observed),
            "threshold": _jsonable(self.threshold),
        }


@dataclass(frozen=True)
class EpisodeQualityGateConfig:
    required_camera_streams: tuple[str, ...] = ("body", "wrist")
    max_sync_skew_warn_s: float = 0.010
    max_sync_skew_reject_s: float = 0.050
    declared_rate_tolerance: float = 0.01
    target_linear_speed_warn_mps: Optional[float] = None
    target_linear_speed_reject_mps: Optional[float] = None
    target_angular_speed_warn_radps: Optional[float] = None
    target_angular_speed_reject_radps: Optional[float] = None
    gripper_range: Optional[tuple[float, float]] = None
    check_camera_files: bool = False
    simulation_fixture_allowed: bool = True
    threshold_provenance: Mapping[str, str] = field(
        default_factory=lambda: {
            "max_sync_skew_warn_s": "TEST_ONLY",
            "max_sync_skew_reject_s": "TEST_ONLY",
            "declared_rate_tolerance": "BENCHMARK_DERIVED",
            "target_linear_speed_warn_mps": "TEST_ONLY",
            "target_linear_speed_reject_mps": "TEST_ONLY",
            "target_angular_speed_warn_radps": "TEST_ONLY",
            "target_angular_speed_reject_radps": "TEST_ONLY",
            "gripper_range": "CONTROLLER_CONTRACT",
        }
    )

    def __post_init__(self) -> None:
        if not self.required_camera_streams:
            raise ValueError("at least one required camera stream is needed")
        if not (0.0 <= self.max_sync_skew_warn_s <= self.max_sync_skew_reject_s):
            raise ValueError("sync skew thresholds must satisfy 0 <= warn <= reject")
        if not np.isfinite(self.declared_rate_tolerance) or self.declared_rate_tolerance < 0.0:
            raise ValueError("declared_rate_tolerance must be non-negative")
        for name in (
            "target_linear_speed_warn_mps",
            "target_linear_speed_reject_mps",
            "target_angular_speed_warn_radps",
            "target_angular_speed_reject_radps",
        ):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be non-negative and finite")
        if (
            self.target_linear_speed_warn_mps is not None
            and self.target_linear_speed_reject_mps is not None
            and self.target_linear_speed_warn_mps > self.target_linear_speed_reject_mps
        ):
            raise ValueError("linear speed warn threshold cannot exceed reject threshold")
        if (
            self.target_angular_speed_warn_radps is not None
            and self.target_angular_speed_reject_radps is not None
            and self.target_angular_speed_warn_radps > self.target_angular_speed_reject_radps
        ):
            raise ValueError("angular speed warn threshold cannot exceed reject threshold")
        if self.gripper_range is not None:
            low, high = map(float, self.gripper_range)
            if not np.isfinite(low) or not np.isfinite(high) or low > high:
                raise ValueError("gripper_range must be a finite ordered pair")
        allowed = {"CONTROLLER_CONTRACT", "GENERATOR_SUPPORT", "BENCHMARK_DERIVED", "TEST_ONLY", "REAL_ROBOT_TBD"}
        for key, value in self.threshold_provenance.items():
            if str(value) not in allowed:
                raise ValueError(f"unknown threshold provenance for {key}: {value}")


@dataclass(frozen=True)
class EpisodeQualityResult:
    decision: EpisodeGateDecision
    issues: tuple[EpisodeQualityIssue, ...]
    metrics: Mapping[str, Any]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.decision is not EpisodeGateDecision.REJECT

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "EpisodeQualityResultV1",
            "decision": self.decision.value,
            "accepted": self.accepted,
            "issues": [issue.as_dict() for issue in self.issues],
            "metrics": _jsonable(self.metrics),
            "provenance": _jsonable(self.provenance),
        }


def _safe_stats(values: Any) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        return {"count": int(array.size), "mean": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


class EpisodeQualityGateV1:
    """Classify an episode without silently repairing it."""

    def __init__(self, config: EpisodeQualityGateConfig | None = None) -> None:
        self.config = EpisodeQualityGateConfig() if config is None else config

    def evaluate(
        self,
        raw: RawEpisodeV1,
        *,
        synchronization: Optional[SynchronizationResultV1] = None,
        calibration: Optional[CalibrationBundleV1] = None,
    ) -> EpisodeQualityResult:
        issues: list[EpisodeQualityIssue] = []
        metrics: dict[str, Any] = {"episode_id": getattr(raw, "episode_id", None)}

        def add(code: str, severity: EpisodeIssueSeverity, message: str, provenance: str, observed: Any = None, threshold: Any = None) -> None:
            issues.append(EpisodeQualityIssue(code, severity, message, provenance, observed, threshold))

        def reject(code: str, message: str, provenance: str, observed: Any = None, threshold: Any = None) -> None:
            add(code, EpisodeIssueSeverity.REJECT, message, provenance, observed, threshold)

        def warn(code: str, message: str, provenance: str, observed: Any = None, threshold: Any = None) -> None:
            add(code, EpisodeIssueSeverity.WARNING, message, provenance, observed, threshold)

        calibration_id = str(getattr(raw, "calibration_bundle_id", ""))
        if not calibration_id:
            reject("MISSING_CALIBRATION_BUNDLE_ID", "episode has no calibration bundle ID", "CONTROLLER_CONTRACT")
        if calibration is not None:
            if calibration.bundle_id != calibration_id:
                reject(
                    "CALIBRATION_BUNDLE_ID_MISMATCH",
                    "raw episode and supplied calibration bundle IDs differ",
                    "CONTROLLER_CONTRACT",
                    {"episode": calibration_id, "bundle": calibration.bundle_id},
                )
            if calibration.status == "SIMULATION_ONLY" and self.config.simulation_fixture_allowed:
                warn(
                    "SIMULATION_CALIBRATION_NOT_REAL",
                    "calibration is a simulation/proxy bundle and cannot qualify physical collection",
                    "REAL_ROBOT_TBD",
                    calibration.status,
                )
            elif calibration.status == "SIMULATION_ONLY":
                reject(
                    "SIMULATION_CALIBRATION_NOT_ALLOWED",
                    "simulation calibration cannot pass a real-robot collection gate",
                    "REAL_ROBOT_TBD",
                    calibration.status,
                )
            elif not calibration.real_robot_calibration:
                warn(
                    "REAL_CALIBRATION_NOT_CONFIRMED",
                    "calibration bundle does not assert real-robot calibration",
                    "REAL_ROBOT_TBD",
                )

        native = np.asarray(getattr(raw, "native_timestamps", []), dtype=np.float64)
        if native.ndim != 1 or native.size == 0 or not np.isfinite(native).all():
            reject("INVALID_NATIVE_TIMESTAMPS", "native timestamps are empty or non-finite", "CONTROLLER_CONTRACT")
        elif native.size > 1 and np.any(np.diff(native) <= 0.0):
            reject("NON_MONOTONIC_NATIVE_TIMESTAMPS", "native timestamps are not strictly increasing", "CONTROLLER_CONTRACT")

        for name in ("state_eef_9d", "desired_eef_9d", "achieved_eef_9d", "state_gripper", "desired_gripper"):
            value = np.asarray(getattr(raw, name, []), dtype=np.float64)
            if value.size == 0 or not np.isfinite(value).all():
                reject("NON_FINITE_STATE_OR_LABEL", f"{name} is empty or non-finite", "CONTROLLER_CONTRACT", name)
        achieved_gripper = getattr(raw, "achieved_gripper", None)
        if achieved_gripper is None:
            warn("ACHIEVED_GRIPPER_MISSING", "measured/achieved gripper stream is absent", "CONTROLLER_CONTRACT")
        elif not np.isfinite(np.asarray(achieved_gripper, dtype=np.float64)).all():
            reject("NON_FINITE_ACHIEVED_GRIPPER", "achieved gripper stream is non-finite", "CONTROLLER_CONTRACT")
        for name in ("state_eef_9d", "desired_eef_9d", "achieved_eef_9d"):
            value = np.asarray(getattr(raw, name, []), dtype=np.float64)
            if value.ndim == 2 and value.shape[-1] == 9 and np.isfinite(value).all():
                try:
                    rot6d_rows_to_matrix(value[:, 3:9])
                except (ValueError, TypeError):
                    reject("INVALID_EEF_SO3", f"{name} contains invalid SO(3) rot6d", "CONTROLLER_CONTRACT", name)

        for name in self.config.required_camera_streams:
            stream = getattr(raw, "camera_streams", {}).get(name)
            if stream is None or stream.frame_count == 0:
                reject("REQUIRED_STREAM_MISSING", f"required camera stream {name!r} is missing", "CONTROLLER_CONTRACT", name)
                continue
            timestamps = np.asarray(stream.timestamps, dtype=np.float64)
            if not np.isfinite(timestamps).all() or (timestamps.size > 1 and np.any(np.diff(timestamps) <= 0.0)):
                reject("NON_MONOTONIC_CAMERA_TIMESTAMPS", f"camera stream {name!r} timestamps are invalid", "CONTROLLER_CONTRACT", name)
            if self.config.check_camera_files:
                missing = [path for path in stream.frame_paths if not Path(path).is_file()]
                if missing:
                    reject("MISSING_CAMERA_FRAME", f"camera stream {name!r} has missing image files", "CONTROLLER_CONTRACT", {"count": len(missing)})

        if not getattr(raw, "lifecycle_events", ()):
            warn("MISSING_LIFECYCLE_EVENTS", "episode has no lifecycle event records", "CONTROLLER_CONTRACT")
        if getattr(raw, "safety_events", ()):
            warn(
                "SAFETY_EVENTS_PRESENT",
                "episode contains safety/lifecycle events that require review",
                "CONTROLLER_CONTRACT",
                len(raw.safety_events),
            )

        if synchronization is None:
            warn("SYNCHRONIZATION_NOT_RUN", "episode has not been evaluated by SynchronizerV1", "CONTROLLER_CONTRACT")
        else:
            metrics["synchronization"] = dict(synchronization.metrics)
            max_skew = float(synchronization.metrics.get("max_abs_skew_s", 0.0))
            if not np.isfinite(max_skew) or max_skew > self.config.max_sync_skew_reject_s:
                reject("SYNC_SKEW_REJECT", "source streams exceed synchronization skew bound", self.config.threshold_provenance["max_sync_skew_reject_s"], max_skew, self.config.max_sync_skew_reject_s)
            elif max_skew > self.config.max_sync_skew_warn_s:
                warn("SYNC_SKEW_WARNING", "source streams exceed synchronization warning bound", self.config.threshold_provenance["max_sync_skew_warn_s"], max_skew, self.config.max_sync_skew_warn_s)
            for code in synchronization.warnings:
                if code == "DECLARED_EFFECTIVE_RATE_MISMATCH":
                    warn(
                        "D3_CAMERA_FPS_METADATA_MISMATCH" if str(getattr(raw, "source", "")).startswith("D2") or "D3" in str(getattr(raw, "metadata", {})) else "DECLARED_EFFECTIVE_RATE_MISMATCH",
                        "declared camera rate differs from effective source timestamp rate; timestamps are retained as truth",
                        self.config.threshold_provenance["declared_rate_tolerance"],
                        synchronization.metrics.get("declared_rate_mismatch"),
                        self.config.declared_rate_tolerance,
                    )
            if any(value > 0 for value in synchronization.non_monotonic_counts.values()):
                reject("SYNC_NON_MONOTONIC_SOURCE", "synchronizer found non-monotonic source timestamps", "CONTROLLER_CONTRACT", dict(synchronization.non_monotonic_counts))
            if any(value > 0 for value in synchronization.dropped_frames.values()):
                warn("DROPPED_OR_GAPPED_FRAMES", "synchronizer detected source timestamp gaps", "BENCHMARK_DERIVED", dict(synchronization.dropped_frames))

        timestamps = native if native.ndim == 1 and native.size > 0 else np.asarray([], dtype=np.float64)
        desired = np.asarray(getattr(raw, "desired_eef_9d", []), dtype=np.float64)
        if timestamps.size > 1 and desired.shape == (timestamps.size, 9) and np.isfinite(desired).all():
            position_steps = np.linalg.norm(np.diff(desired[:, :3], axis=0), axis=-1)
            rotations = rot6d_rows_to_matrix(desired[:, 3:9])
            orientation_steps = geodesic_angle(rotations[:-1], rotations[1:])
            dt = np.diff(timestamps)
            linear_speed = position_steps / dt
            angular_speed = orientation_steps / dt
            metrics["desired_linear_speed_mps"] = _safe_stats(linear_speed)
            metrics["desired_angular_speed_radps"] = _safe_stats(angular_speed)
            for value, warn_limit, reject_limit, label in (
                (linear_speed, self.config.target_linear_speed_warn_mps, self.config.target_linear_speed_reject_mps, "linear"),
                (angular_speed, self.config.target_angular_speed_warn_radps, self.config.target_angular_speed_reject_radps, "angular"),
            ):
                maximum = float(np.max(value)) if value.size else 0.0
                if reject_limit is not None and maximum > reject_limit:
                    reject(f"TARGET_{label.upper()}_SPEED_REJECT", f"desired EEF {label} speed exceeds configured bound", "TEST_ONLY", maximum, reject_limit)
                elif warn_limit is not None and maximum > warn_limit:
                    warn(f"TARGET_{label.upper()}_SPEED_WARNING", f"desired EEF {label} speed exceeds configured warning bound", "TEST_ONLY", maximum, warn_limit)
            metrics["target_position_step_m"] = _safe_stats(position_steps)
            metrics["target_rotation_step_rad"] = _safe_stats(orientation_steps)

        gripper = np.asarray(getattr(raw, "desired_gripper", []), dtype=np.float64)
        if gripper.size and np.isfinite(gripper).all() and self.config.gripper_range is not None:
            low, high = self.config.gripper_range
            if np.any(gripper < low) or np.any(gripper > high):
                reject("GRIPPER_OUT_OF_RANGE", "desired gripper is outside the declared normalized range", self.config.threshold_provenance["gripper_range"], [float(np.min(gripper)), float(np.max(gripper))], [low, high])
        elif gripper.size == 0 or not np.isfinite(gripper).all():
            reject("INVALID_GRIPPER", "desired gripper stream is missing or non-finite", "CONTROLLER_CONTRACT")

        decision = EpisodeGateDecision.ACCEPT
        if any(issue.severity is EpisodeIssueSeverity.REJECT for issue in issues):
            decision = EpisodeGateDecision.REJECT
        elif issues:
            decision = EpisodeGateDecision.WARN
        return EpisodeQualityResult(
            decision=decision,
            issues=tuple(issues),
            metrics=metrics,
            provenance={
                "gate": "EpisodeQualityGateV1",
                "corrective_action": "NONE",
                "source_timestamps_preserved": True,
                "production_action_timing": "TBD_BY_ADAPTATION_DATASET",
            },
        )
