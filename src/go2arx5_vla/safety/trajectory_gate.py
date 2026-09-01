"""Canonical, non-corrective validation for external EEF trajectories.

The safety gate never clips, smooths, re-times, or otherwise changes a
trajectory.  It returns a decision and an auditable list of observations so a
caller can stop or explicitly handle a rejected/warned chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

import numpy as np

from ..contracts.trajectory import TrajectoryChunkV1
from ..trajectory.so3 import geodesic_angle, quat_to_matrix, rot6d_to_matrix


class GateDecision(str, Enum):
    ACCEPT = "ACCEPT"
    WARN = "WARN"
    REJECT = "REJECT"


class GateSeverity(str, Enum):
    WARNING = "WARNING"
    REJECT = "REJECT"


class ThresholdProvenance(str, Enum):
    CONTROLLER_CONTRACT = "CONTROLLER_CONTRACT"
    GENERATOR_SUPPORT = "GENERATOR_SUPPORT"
    BENCHMARK_DERIVED = "BENCHMARK_DERIVED"
    TEST_ONLY = "TEST_ONLY"
    REAL_ROBOT_TBD = "REAL_ROBOT_TBD"


@dataclass(frozen=True)
class GateIssue:
    code: str
    severity: GateSeverity
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
            "observed": self.observed,
            "threshold": self.threshold,
        }


def _default_threshold_provenance() -> dict[str, str]:
    return {
        "initial_position_warn_m": ThresholdProvenance.BENCHMARK_DERIVED.value,
        "initial_position_reject_m": ThresholdProvenance.BENCHMARK_DERIVED.value,
        "initial_rotation_warn_rad": ThresholdProvenance.BENCHMARK_DERIVED.value,
        "initial_rotation_reject_rad": ThresholdProvenance.BENCHMARK_DERIVED.value,
        "linear_speed_warn_mps": ThresholdProvenance.TEST_ONLY.value,
        "linear_speed_reject_mps": ThresholdProvenance.TEST_ONLY.value,
        "angular_speed_warn_radps": ThresholdProvenance.TEST_ONLY.value,
        "angular_speed_reject_radps": ThresholdProvenance.TEST_ONLY.value,
        "workspace_warn_min_m": ThresholdProvenance.GENERATOR_SUPPORT.value,
        "workspace_warn_max_m": ThresholdProvenance.GENERATOR_SUPPORT.value,
        "workspace_reject_min_m": ThresholdProvenance.TEST_ONLY.value,
        "workspace_reject_max_m": ThresholdProvenance.TEST_ONLY.value,
        "max_path_length_m": ThresholdProvenance.TEST_ONLY.value,
        "min_arc_length_m": ThresholdProvenance.CONTROLLER_CONTRACT.value,
        "aging_age_s": ThresholdProvenance.CONTROLLER_CONTRACT.value,
        "stale_age_s": ThresholdProvenance.CONTROLLER_CONTRACT.value,
        "min_points": ThresholdProvenance.CONTROLLER_CONTRACT.value,
    }


@dataclass(frozen=True)
class TrajectorySafetyGateConfig:
    """Explicit gate thresholds and provenance labels.

    The speed/workspace defaults are test/benchmark values, not real-robot
    safety limits.  Production values remain ``REAL_ROBOT_TBD`` until the
    adaptation/deployment contract supplies them.
    """

    initial_position_warn_m: float = 0.25
    initial_position_reject_m: float = 0.75
    initial_rotation_warn_rad: float = 0.50
    initial_rotation_reject_rad: float = 1.50
    linear_speed_warn_mps: float = 1.0
    linear_speed_reject_mps: float = 2.0
    angular_speed_warn_radps: float = 2.0
    angular_speed_reject_radps: float = 4.0
    workspace_warn_min_m: Optional[np.ndarray] = None
    workspace_warn_max_m: Optional[np.ndarray] = None
    workspace_reject_min_m: Optional[np.ndarray] = None
    workspace_reject_max_m: Optional[np.ndarray] = None
    max_path_length_m: float = 5.0
    min_arc_length_m: float = 1e-6
    aging_age_s: float = 0.20
    stale_age_s: float = 0.75
    min_points: int = 2
    lambda_m_per_rad: float = 0.15
    threshold_provenance: Mapping[str, str] = field(default_factory=_default_threshold_provenance)

    def __post_init__(self) -> None:
        ordered = (
            ("initial_position_warn_m", self.initial_position_warn_m, 0.0),
            ("initial_position_reject_m", self.initial_position_reject_m, self.initial_position_warn_m),
            ("initial_rotation_warn_rad", self.initial_rotation_warn_rad, 0.0),
            ("initial_rotation_reject_rad", self.initial_rotation_reject_rad, self.initial_rotation_warn_rad),
            ("linear_speed_warn_mps", self.linear_speed_warn_mps, 0.0),
            ("linear_speed_reject_mps", self.linear_speed_reject_mps, self.linear_speed_warn_mps),
            ("angular_speed_warn_radps", self.angular_speed_warn_radps, 0.0),
            ("angular_speed_reject_radps", self.angular_speed_reject_radps, self.angular_speed_warn_radps),
            ("max_path_length_m", self.max_path_length_m, 0.0),
            ("min_arc_length_m", self.min_arc_length_m, 0.0),
            ("aging_age_s", self.aging_age_s, 0.0),
            ("stale_age_s", self.stale_age_s, self.aging_age_s),
        )
        for name, value, lower in ordered:
            value = float(value)
            if not np.isfinite(value) or value < lower:
                raise ValueError(f"{name} must be finite and >= {lower}")
        if int(self.min_points) < 1:
            raise ValueError("min_points must be positive")
        if not np.isfinite(self.lambda_m_per_rad) or self.lambda_m_per_rad <= 0.0:
            raise ValueError("lambda_m_per_rad must be positive and finite")
        for name in (
            "workspace_warn_min_m",
            "workspace_warn_max_m",
            "workspace_reject_min_m",
            "workspace_reject_max_m",
        ):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=np.float64)
                if array.shape != (3,) or not np.isfinite(array).all():
                    raise ValueError(f"{name} must be finite with shape [3]")
                object.__setattr__(self, name, array)
        if self.workspace_warn_min_m is not None and self.workspace_warn_max_m is not None:
            if np.any(self.workspace_warn_min_m > self.workspace_warn_max_m):
                raise ValueError("workspace warning min must not exceed max")
        if self.workspace_reject_min_m is not None and self.workspace_reject_max_m is not None:
            if np.any(self.workspace_reject_min_m > self.workspace_reject_max_m):
                raise ValueError("workspace rejection min must not exceed max")
        provenance = dict(self.threshold_provenance)
        required = set(_default_threshold_provenance())
        missing = sorted(required.difference(provenance))
        if missing:
            raise ValueError("missing threshold provenance: " + ", ".join(missing))
        valid_provenance = {item.value for item in ThresholdProvenance}
        normalized_provenance = {}
        for key, value in provenance.items():
            normalized = value.value if isinstance(value, ThresholdProvenance) else str(value)
            if normalized not in valid_provenance:
                raise ValueError(f"unknown threshold provenance for {key}: {value}")
            normalized_provenance[key] = normalized
        object.__setattr__(self, "threshold_provenance", normalized_provenance)

    def provenance_for(self, name: str) -> str:
        return str(self.threshold_provenance[name])


@dataclass(frozen=True)
class SafetyGateResult:
    decision: GateDecision
    issues: tuple[GateIssue, ...]
    metrics: Mapping[str, Any]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.decision is not GateDecision.REJECT

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "issues": [issue.as_dict() for issue in self.issues],
            "metrics": dict(self.metrics),
            "provenance": dict(self.provenance),
        }


class TrajectorySafetyGateV1:
    """Validate a chunk and return ACCEPT/WARN/REJECT without mutation."""

    def __init__(self, config: TrajectorySafetyGateConfig | None = None) -> None:
        self.config = TrajectorySafetyGateConfig() if config is None else config

    def evaluate(
        self,
        chunk: TrajectoryChunkV1,
        *,
        now: float | None = None,
        achieved_position: Any | None = None,
        achieved_orientation: Any | None = None,
    ) -> SafetyGateResult:
        issues: list[GateIssue] = []
        metrics: dict[str, Any] = {"chunk_id": getattr(chunk, "chunk_id", None)}

        def reject(code: str, message: str, provenance: str, observed: Any = None, threshold: Any = None) -> None:
            issues.append(GateIssue(code, GateSeverity.REJECT, message, provenance, observed, threshold))

        def warn(code: str, message: str, provenance: str, observed: Any = None, threshold: Any = None) -> None:
            issues.append(GateIssue(code, GateSeverity.WARNING, message, provenance, observed, threshold))

        try:
            timestamps = np.asarray(chunk.point_timestamps, dtype=np.float64)
            positions = np.asarray(chunk.eef_position, dtype=np.float64)
            orientations = np.asarray(chunk.eef_orientation, dtype=np.float64)
        except Exception as exc:  # pragma: no cover - defensive boundary
            reject("INVALID_CHUNK", f"cannot read chunk arrays: {exc}", ThresholdProvenance.CONTROLLER_CONTRACT.value)
            return self._result(issues, metrics)

        if timestamps.ndim != 1 or positions.shape != (len(timestamps), 3) or orientations.shape != (len(timestamps), 4):
            reject("INVALID_SHAPE", "timestamps/EEF arrays do not match the chunk contract", ThresholdProvenance.CONTROLLER_CONTRACT.value)
            return self._result(issues, metrics)
        if len(timestamps) < self.config.min_points:
            reject(
                "SHORT_OR_EMPTY_CHUNK",
                f"chunk has {len(timestamps)} points; at least {self.config.min_points} are required",
                self.config.provenance_for("min_points"),
                len(timestamps),
                self.config.min_points,
            )
        if not np.isfinite(timestamps).all() or not np.isfinite(positions).all() or not np.isfinite(orientations).all():
            reject("NON_FINITE", "trajectory contains NaN or Inf", ThresholdProvenance.CONTROLLER_CONTRACT.value)
            return self._result(issues, metrics)
        if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0.0):
            reject(
                "NON_MONOTONIC_TIMESTAMPS",
                "point timestamps must be strictly increasing at the runtime boundary",
                ThresholdProvenance.CONTROLLER_CONTRACT.value,
            )
            return self._result(issues, metrics)

        try:
            norms = np.linalg.norm(orientations, axis=-1)
            matrices = quat_to_matrix(orientations)
            orthogonal = np.allclose(
                np.matmul(np.swapaxes(matrices, -1, -2), matrices),
                np.eye(3),
                atol=1e-5,
            )
            determinants = np.linalg.det(matrices)
            valid_so3 = bool(np.all(norms > 1e-10) and orthogonal and np.allclose(determinants, 1.0, atol=1e-5))
            if not valid_so3:
                reject("INVALID_SO3", "EEF orientation is not a valid SO(3) pose", ThresholdProvenance.CONTROLLER_CONTRACT.value)
            if getattr(chunk, "eef_rot6d", None) is not None:
                rot6d_to_matrix(np.asarray(chunk.eef_rot6d, dtype=np.float64))
        except Exception as exc:
            reject("INVALID_SO3", f"EEF orientation/rot6d validation failed: {exc}", ThresholdProvenance.CONTROLLER_CONTRACT.value)
            matrices = None

        if len(timestamps) > 1:
            dt = np.diff(timestamps)
            linear_speeds = np.linalg.norm(np.diff(positions, axis=0), axis=-1) / dt
            metrics["linear_speed_mps"] = self._stats(linear_speeds)
            if matrices is not None:
                angular_speeds = geodesic_angle(matrices[:-1], matrices[1:]) / dt
                metrics["angular_speed_radps"] = self._stats(angular_speeds)
            else:
                angular_speeds = np.asarray([], dtype=np.float64)
            self._speed_issue(
                issues,
                "LINEAR_SPEED",
                float(np.max(linear_speeds)),
                self.config.linear_speed_warn_mps,
                self.config.linear_speed_reject_mps,
                "linear_speed_warn_mps",
                "linear_speed_reject_mps",
                "m/s",
            )
            if angular_speeds.size:
                self._speed_issue(
                    issues,
                    "ANGULAR_SPEED",
                    float(np.max(angular_speeds)),
                    self.config.angular_speed_warn_radps,
                    self.config.angular_speed_reject_radps,
                    "angular_speed_warn_radps",
                    "angular_speed_reject_radps",
                    "rad/s",
                )
        else:
            linear_speeds = np.asarray([], dtype=np.float64)
            angular_speeds = np.asarray([], dtype=np.float64)

        path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=-1).sum()) if len(positions) > 1 else 0.0
        arc_length = path_length
        if matrices is not None and len(matrices) > 1:
            rotation_steps = geodesic_angle(matrices[:-1], matrices[1:])
            arc_length = float(np.sqrt(np.linalg.norm(np.diff(positions, axis=0), axis=-1) ** 2 + (self.config.lambda_m_per_rad * rotation_steps) ** 2).sum())
        metrics.update({"translation_path_length_m": path_length, "se3_arc_length_m": arc_length})
        if path_length > self.config.max_path_length_m:
            reject(
                "PATH_LENGTH_SANITY",
                "trajectory translation path exceeds the configured test envelope",
                self.config.provenance_for("max_path_length_m"),
                path_length,
                self.config.max_path_length_m,
            )
        if len(positions) > 1 and arc_length <= self.config.min_arc_length_m:
            warn(
                "ARC_LENGTH_DEGENERACY",
                "trajectory has negligible SE(3) arc length; preview will hold a pose",
                self.config.provenance_for("min_arc_length_m"),
                arc_length,
                self.config.min_arc_length_m,
            )

        self._workspace_issues(issues, positions)

        if achieved_position is not None:
            achieved = np.asarray(achieved_position, dtype=np.float64)
            if achieved.shape != (3,) or not np.isfinite(achieved).all():
                reject("INVALID_ACHIEVED_POSITION", "achieved EEF position is not finite [3]", ThresholdProvenance.CONTROLLER_CONTRACT.value)
            else:
                initial_jump = float(np.linalg.norm(positions[0] - achieved))
                metrics["initial_position_discontinuity_m"] = initial_jump
                self._threshold_issue(
                    issues,
                    "INITIAL_POSITION_DISCONTINUITY",
                    initial_jump,
                    self.config.initial_position_warn_m,
                    self.config.initial_position_reject_m,
                    "initial_position_warn_m",
                    "initial_position_reject_m",
                    "m",
                )
        else:
            warn(
                "INITIAL_POSITION_NOT_EVALUATED",
                "achieved EEF position was not supplied",
                ThresholdProvenance.CONTROLLER_CONTRACT.value,
            )

        if achieved_orientation is not None:
            try:
                achieved_q = np.asarray(achieved_orientation, dtype=np.float64)
                if achieved_q.shape != (4,):
                    raise ValueError("expected [4]")
                initial_rotation = float(geodesic_angle(quat_to_matrix(achieved_q), matrices[0]))
                metrics["initial_rotation_discontinuity_rad"] = initial_rotation
                self._threshold_issue(
                    issues,
                    "INITIAL_ROTATION_DISCONTINUITY",
                    initial_rotation,
                    self.config.initial_rotation_warn_rad,
                    self.config.initial_rotation_reject_rad,
                    "initial_rotation_warn_rad",
                    "initial_rotation_reject_rad",
                    "rad",
                )
            except Exception as exc:
                reject("INVALID_ACHIEVED_ORIENTATION", f"achieved EEF orientation is invalid: {exc}", ThresholdProvenance.CONTROLLER_CONTRACT.value)
        elif matrices is not None:
            warn(
                "INITIAL_ROTATION_NOT_EVALUATED",
                "achieved EEF orientation was not supplied",
                ThresholdProvenance.CONTROLLER_CONTRACT.value,
            )

        if now is not None:
            now_value = float(now)
            if not np.isfinite(now_value):
                reject("INVALID_NOW", "gate evaluation timestamp is not finite", ThresholdProvenance.CONTROLLER_CONTRACT.value)
            else:
                age = now_value - float(chunk.observation_timestamp)
                metrics["chunk_age_s"] = age
                if age < 0.0:
                    reject("OUT_OF_ORDER_TIME", "evaluation time precedes chunk observation time", ThresholdProvenance.CONTROLLER_CONTRACT.value, age, 0.0)
                elif age > self.config.stale_age_s:
                    reject("STALE_CHUNK", "chunk exceeds the stale-age threshold", self.config.provenance_for("stale_age_s"), age, self.config.stale_age_s)
                elif age > self.config.aging_age_s:
                    warn("AGING_CHUNK", "chunk exceeds the fresh-age threshold", self.config.provenance_for("aging_age_s"), age, self.config.aging_age_s)
        else:
            warn("CHUNK_AGE_NOT_EVALUATED", "gate evaluation timestamp was not supplied", ThresholdProvenance.CONTROLLER_CONTRACT.value)

        return self._result(issues, metrics)

    def _workspace_issues(self, issues: list[GateIssue], positions: np.ndarray) -> None:
        config = self.config
        if config.workspace_warn_min_m is not None:
            outside_low = np.any(positions < config.workspace_warn_min_m, axis=1)
            outside_high = np.any(positions > config.workspace_warn_max_m, axis=1) if config.workspace_warn_max_m is not None else np.zeros(len(positions), bool)
            count = int(np.count_nonzero(outside_low | outside_high))
            if count:
                issues.append(
                    GateIssue(
                        "WORKSPACE_WARNING",
                        GateSeverity.WARNING,
                        f"{count} trajectory points leave the configured warning envelope",
                        config.provenance_for("workspace_warn_min_m"),
                        count,
                        {
                            "min": config.workspace_warn_min_m.tolist(),
                            "max": None if config.workspace_warn_max_m is None else config.workspace_warn_max_m.tolist(),
                        },
                    )
                )
        reject_low = bool(config.workspace_reject_min_m is not None and np.any(positions < config.workspace_reject_min_m))
        reject_high = bool(config.workspace_reject_max_m is not None and np.any(positions > config.workspace_reject_max_m))
        if reject_low or reject_high:
            count = int(np.count_nonzero(np.any(positions < config.workspace_reject_min_m, axis=1) if config.workspace_reject_min_m is not None else np.zeros(len(positions), bool)) + np.count_nonzero(np.any(positions > config.workspace_reject_max_m, axis=1) if config.workspace_reject_max_m is not None else np.zeros(len(positions), bool)))
            warn_or_reject = GateIssue(
                "WORKSPACE_REJECTION",
                GateSeverity.REJECT,
                f"{count} trajectory points leave the configured rejection envelope",
                config.provenance_for("workspace_reject_min_m"),
                count,
                {"min": None if config.workspace_reject_min_m is None else config.workspace_reject_min_m.tolist(), "max": None if config.workspace_reject_max_m is None else config.workspace_reject_max_m.tolist()},
            )
            issues.append(warn_or_reject)

    def _speed_issue(
        self,
        issues: list[GateIssue],
        code: str,
        observed: float,
        warn_threshold: float,
        reject_threshold: float,
        warn_name: str,
        reject_name: str,
        unit: str,
    ) -> None:
        if observed > reject_threshold:
            issues.append(GateIssue(code + "_REJECT", GateSeverity.REJECT, f"{code.lower()} exceeds rejection threshold ({unit})", self.config.provenance_for(reject_name), observed, reject_threshold))
        elif observed > warn_threshold:
            issues.append(GateIssue(code + "_WARN", GateSeverity.WARNING, f"{code.lower()} exceeds warning threshold ({unit})", self.config.provenance_for(warn_name), observed, warn_threshold))

    def _threshold_issue(
        self,
        issues: list[GateIssue],
        code: str,
        observed: float,
        warn_threshold: float,
        reject_threshold: float,
        warn_name: str,
        reject_name: str,
        unit: str,
    ) -> None:
        self._speed_issue(issues, code, observed, warn_threshold, reject_threshold, warn_name, reject_name, unit)

    @staticmethod
    def _stats(values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    def _result(self, issues: list[GateIssue], metrics: Mapping[str, Any]) -> SafetyGateResult:
        decision = GateDecision.REJECT if any(i.severity is GateSeverity.REJECT for i in issues) else GateDecision.WARN if issues else GateDecision.ACCEPT
        return SafetyGateResult(
            decision=decision,
            issues=tuple(issues),
            metrics=dict(metrics),
            provenance={
                "gate": "TrajectorySafetyGateV1",
                "corrective_action": "NONE",
                "threshold_provenance": dict(self.config.threshold_provenance),
            },
        )
