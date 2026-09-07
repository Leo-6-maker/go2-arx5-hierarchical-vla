#!/usr/bin/env python3
"""CPU-only fail-closed admissions for the Vanda Stage40 A40 lane.

This module deliberately has no torch, CUDA, simulator, WBC, or scheduler
imports.  It validates the immutable Stage40 boundary before a later GPU
launcher is allowed to construct or execute a runtime.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

DEV_SEED = 4242
FORMAL_SEEDS = frozenset({2223, 2228, 2233, 2238})
STAGE40_ACTION_HORIZON = 40
STAGE40_UNIVERSAL_ACTION_DIM = 132
STAGE40_PHYSICAL_ACTION_DIM = 10
STAGE40_PERIOD_S = 0.05
STAGE40_DURATION_S = 1.95
STAGE40_HANDOFF_S = 0.9750000000000001
STAGE40_TIMING_MODE = "T16"
STAGE40_SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
STAGE40_TCP_SEMANTIC_ID = "x5_link6_gripper_center"
STAGE40_DECODER = "go2arx5_vla.stage40_structured_decode.decode_raw_structured_action"


class Stage40AdmissionError(ValueError):
    """Raised when a frozen Stage40 boundary is not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage40AdmissionError(message)


def require_development_seed(seed: Any) -> int:
    """Allow only the fixed development seed; reject formal/other seeds."""

    try:
        value = int(seed)
    except (TypeError, ValueError) as exc:
        raise Stage40AdmissionError(f"development seed is not an integer: {seed!r}") from exc
    _require(value == DEV_SEED, f"development mode requires DEV_SEED={DEV_SEED}, got {value}")
    _require(value not in FORMAL_SEEDS, f"formal seed {value} is forbidden in development mode")
    return value


def handoff_steps_for_native_dt(native_dt_s: Any) -> int:
    """Derive the only allowed prefix from the frozen time target."""

    try:
        dt = float(native_dt_s)
    except (TypeError, ValueError) as exc:
        raise Stage40AdmissionError(f"native control dt is not numeric: {native_dt_s!r}") from exc
    _require(math.isfinite(dt) and dt > 0.0, f"native control dt must be finite and positive: {dt!r}")
    steps = int(round(STAGE40_HANDOFF_S / dt))
    _require(steps >= 1, f"handoff schedule produced no native control steps: dt={dt!r}")
    return steps


def _shape(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise Stage40AdmissionError(f"shape is not an integer sequence: {value!r}") from exc


def validate_structured_action_contract(
    *,
    raw_shape: Sequence[int],
    source_timestamps: Sequence[float],
    timing_mode: str,
    physical_shape: Sequence[int],
    source_frame: str = STAGE40_SOURCE_FRAME,
    tcp_semantic_id: str = STAGE40_TCP_SEMANTIC_ID,
    decoder: str = STAGE40_DECODER,
    generic_decode_action_used: bool = False,
) -> dict[str, Any]:
    """Validate the exact D132/H40/T16/D10 live decode boundary."""

    _require(
        _shape(raw_shape) == (1, STAGE40_ACTION_HORIZON, STAGE40_UNIVERSAL_ACTION_DIM),
        "raw StructuredActionV1 output must be [1,40,132]",
    )
    _require(
        _shape(physical_shape) == (STAGE40_ACTION_HORIZON, STAGE40_PHYSICAL_ACTION_DIM),
        "physical StructuredActionV1 output must be [40,10]",
    )
    _require(timing_mode == STAGE40_TIMING_MODE, f"timing mode must be {STAGE40_TIMING_MODE}")
    _require(decoder == STAGE40_DECODER, "non-canonical StructuredActionV1 decoder selected")
    _require(not generic_decode_action_used, "generic processor.decode_action is forbidden")
    _require(source_frame == STAGE40_SOURCE_FRAME, "Stage40 source frame mismatch")
    _require(tcp_semantic_id == STAGE40_TCP_SEMANTIC_ID, "Stage40 TCP semantic identity mismatch")

    try:
        timestamps = tuple(float(item) for item in source_timestamps)
    except (TypeError, ValueError) as exc:
        raise Stage40AdmissionError("source timestamps are not numeric") from exc
    _require(
        len(timestamps) == STAGE40_ACTION_HORIZON,
        "Stage40 source timestamps must contain exactly H40=40 points",
    )
    _require(all(math.isfinite(item) for item in timestamps), "source timestamps must be finite")
    deltas = tuple(right - left for left, right in zip(timestamps, timestamps[1:]))
    _require(
        all(math.isclose(delta, STAGE40_PERIOD_S, rel_tol=0.0, abs_tol=1e-9) for delta in deltas),
        "Stage40 source timestamps must use the frozen 0.05 s period",
    )
    _require(
        math.isclose(
            timestamps[-1] - timestamps[0],
            STAGE40_DURATION_S,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "Stage40 source timestamps must span the frozen 1.95 s duration",
    )
    return {
        "decoder": STAGE40_DECODER,
        "timing_mode": STAGE40_TIMING_MODE,
        "raw_shape": list(_shape(raw_shape)),
        "physical_shape": list(_shape(physical_shape)),
        "source_timestamp_count": len(timestamps),
        "source_period_s": STAGE40_PERIOD_S,
        "source_duration_s": STAGE40_DURATION_S,
        "source_frame": STAGE40_SOURCE_FRAME,
        "tcp_semantic_id": STAGE40_TCP_SEMANTIC_ID,
        "generic_decode_action_used": False,
    }


def validate_artifact_identities(
    expected: Mapping[str, str],
    observed: Mapping[str, str],
) -> dict[str, Any]:
    """Require an exact logical-name/hash mapping; never repair drift."""

    expected_map = {str(key): str(value) for key, value in expected.items()}
    observed_map = {str(key): str(value) for key, value in observed.items()}
    _require(bool(expected_map), "artifact identity map must not be empty")
    _require(
        set(expected_map) == set(observed_map),
        "artifact identity set drift: expected and observed logical names differ",
    )
    mismatches = sorted(
        key for key in expected_map if expected_map[key] != observed_map[key]
    )
    _require(not mismatches, "artifact identity drift: " + ", ".join(mismatches))
    return {"identity_match": True, "artifact_count": len(expected_map)}


def validate_safety_gate_decision(decision: Any) -> str:
    """Allow only ACCEPT/WARN at the caller boundary; REJECT stops execution."""

    value = getattr(decision, "value", decision)
    value = str(value)
    _require(value in {"ACCEPT", "WARN"}, f"SafetyGate decision {value!r} is not admissible")
    return value


def validate_fake_lifecycle(
    events: Sequence[str],
    *,
    handoff_steps: int,
    executed_prefix_steps: int,
    safety_q0: str = "ACCEPT",
    safety_q1: str = "ACCEPT",
    reset: bool = False,
    teleport: bool = False,
    policy_history_reset: bool = False,
) -> dict[str, Any]:
    """Validate the deterministic fake V2-V5 lifecycle and its fail-closed edges."""

    expected = (
        "observation_q0",
        "q0_decode",
        "safety_q0",
        "initial_bind",
        "q0_prefix",
        "handoff_pause",
        "observation_q1",
        "q1_decode",
        "safety_q1",
        "reference_update",
        "q1_continuation",
        "reference_complete",
        "seal",
    )
    values = tuple(str(item) for item in events)
    _require(values.count("reference_update") == 1, "exactly one reference update is required")
    _require(
        not any(item in values for item in {
            "second_reference_update",
            "third_reference_update",
            "multi_handoff",
            "reset_before_handoff",
            "termination_before_handoff",
            "physics_reset",
            "physics_teleport",
            "policy_history_reset",
        }),
        "forbidden reset/termination/multi-handoff lifecycle event",
    )
    _require(not reset, "physics reset flag is forbidden")
    _require(not teleport, "physics teleport flag is forbidden")
    _require(not policy_history_reset, "policy-history reset flag is forbidden")
    _require(
        values.index("reference_update") > values.index("handoff_pause"),
        "reference update must occur after the paused handoff boundary",
    )
    cursor = -1
    for item in expected:
        try:
            cursor = values.index(item, cursor + 1)
        except ValueError as exc:
            raise Stage40AdmissionError(f"fake lifecycle missing or out of order event: {item}") from exc
    _require(
        int(executed_prefix_steps) == int(handoff_steps),
        "executed prefix does not equal the derived Stage40 handoff step count",
    )
    _require(safety_q0 in {"ACCEPT", "WARN"}, "Q0 SafetyGate REJECT must stop the lifecycle")
    _require(safety_q1 in {"ACCEPT", "WARN"}, "Q1 SafetyGate REJECT must stop the lifecycle")
    return {
        "lifecycle": "FAKE_STAGE40_ONE_HANDOFF",
        "reference_update_count": 1,
        "executed_prefix_steps": int(executed_prefix_steps),
        "handoff_steps": int(handoff_steps),
        "safety_q0": str(safety_q0),
        "safety_q1": str(safety_q1),
        "physics_reset": False,
        "physics_teleport": False,
        "policy_history_reset": False,
    }


def validate_receding_horizon_lifecycle(
    events: Sequence[str],
    *,
    handoff_count: int,
    handoff_steps: int,
    executed_prefix_steps: int | Sequence[int],
    safety_decisions: Sequence[str] | None = None,
    reset: bool = False,
    teleport: bool = False,
    policy_history_reset: bool = False,
) -> dict[str, Any]:
    """Validate the Stage40-local bounded multi-handoff control flow.

    This validator is deliberately transport- and simulator-independent.  It
    checks only the event ordering and the already-frozen per-handoff timing;
    it does not change the shared executor or controller semantics.
    """

    try:
        count = int(handoff_count)
    except (TypeError, ValueError) as exc:
        raise Stage40AdmissionError("handoff_count is not an integer") from exc
    _require(count >= 1, "handoff_count must be positive")
    try:
        steps = int(handoff_steps)
    except (TypeError, ValueError) as exc:
        raise Stage40AdmissionError("handoff_steps is not an integer") from exc
    _require(steps >= 1, "handoff_steps must be positive")

    if isinstance(executed_prefix_steps, Sequence) and not isinstance(
        executed_prefix_steps, (str, bytes)
    ):
        try:
            prefixes = tuple(int(item) for item in executed_prefix_steps)
        except (TypeError, ValueError) as exc:
            raise Stage40AdmissionError("executed_prefix_steps is not an integer sequence") from exc
    else:
        try:
            prefixes = (int(executed_prefix_steps),) * count
        except (TypeError, ValueError) as exc:
            raise Stage40AdmissionError("executed_prefix_steps is not an integer") from exc
    _require(len(prefixes) == count, "one executed prefix is required for every handoff")
    _require(
        prefixes == (steps,) * count,
        "every executed prefix must equal the derived Stage40 handoff step count",
    )

    values = tuple(str(item) for item in events)
    expected: list[str] = [
        "observation_q0",
        "q0_decode",
        "safety_q0",
        "initial_bind",
        "q0_prefix",
    ]
    for index in range(1, count + 1):
        expected.extend(
            [
                f"handoff_pause_{index}",
                f"observation_q{index}",
                f"q{index}_decode",
                f"safety_q{index}",
                f"reference_update_{index}",
                f"q{index}_continuation",
            ]
        )
    expected.extend([f"q{count}_completion", "seal"])
    _require(
        values == tuple(expected),
        "receding-horizon lifecycle has missing, extra, or out-of-order events",
    )
    _require(
        values.count("initial_bind") == 1,
        "exactly one initial reference bind is required",
    )
    _require(
        sum(item.startswith("reference_update_") for item in values) == count,
        "reference update count does not match the bounded handoff count",
    )
    _require(
        not any(
            item in values
            for item in {
                "second_reference_update",
                "third_reference_update",
                "multi_handoff",
                "reset_before_handoff",
                "termination_before_handoff",
                "physics_reset",
                "physics_teleport",
                "policy_history_reset",
            }
        ),
        "forbidden reset/termination/multi-handoff lifecycle event",
    )
    _require(not reset, "physics reset flag is forbidden")
    _require(not teleport, "physics teleport flag is forbidden")
    _require(not policy_history_reset, "policy-history reset flag is forbidden")

    decisions = ("ACCEPT",) * (count + 1) if safety_decisions is None else tuple(
        str(item) for item in safety_decisions
    )
    _require(
        len(decisions) == count + 1,
        "one SafetyGate decision is required for Q0 through QN",
    )
    _require(
        all(item in {"ACCEPT", "WARN"} for item in decisions),
        "SafetyGate REJECT must stop the receding-horizon lifecycle",
    )
    return {
        "lifecycle": "FAKE_STAGE40_RECEDING_HORIZON",
        "handoff_count": count,
        "forward_count": count + 1,
        "reference_update_count": count,
        "executed_prefix_steps": list(prefixes),
        "handoff_steps": steps,
        "safety_decisions": list(decisions),
        "physics_reset": False,
        "physics_teleport": False,
        "policy_history_reset": False,
    }


_REQUIRED_RESULT_FIELDS = (
    "schema",
    "stage",
    "pi_gate",
    "status",
    "seed",
    "handoff_target_s",
    "handoff_steps",
    "actual_switch_s",
    "switch_tick_error_s",
    "executed_prefix_steps",
    "single_reference_handoff",
    "physics_reset",
    "physics_teleport",
    "policy_history_reset",
    "q0_forward_count",
    "q1_forward_count",
    "wbc_forward_count",
    "gpu_uuid",
    "vram_baseline_mib",
    "vram_peak_mib",
    "artifact_identities",
    "runtime_identities",
    "git_head",
)


def validate_stage40_result_schema(result: Mapping[str, Any]) -> dict[str, Any]:
    """Require the fields needed for an auditable V5 engineering seal."""

    missing = sorted(set(_REQUIRED_RESULT_FIELDS).difference(result))
    _require(not missing, "Stage40 result schema is missing: " + ", ".join(missing))
    _require(str(result["stage"]) == "Stage40", "result stage must be Stage40")
    _require(str(result["status"]) == "PASS", "only a PASS result may be sealed")
    _require(require_development_seed(result["seed"]) == DEV_SEED, "result seed is not DEV_SEED")
    for name in ("handoff_target_s", "actual_switch_s", "switch_tick_error_s", "vram_baseline_mib", "vram_peak_mib"):
        try:
            value = float(result[name])
        except (TypeError, ValueError) as exc:
            raise Stage40AdmissionError(f"{name} is not numeric") from exc
        _require(math.isfinite(value), f"{name} must be finite")
    _require(
        math.isclose(float(result["handoff_target_s"]), STAGE40_HANDOFF_S, rel_tol=0.0, abs_tol=1e-12),
        "handoff_target_s changed from the frozen value",
    )
    _require(int(result["handoff_steps"]) == int(result["executed_prefix_steps"]), "handoff step mismatch")
    _require(bool(result["single_reference_handoff"]), "result does not prove exactly one reference handoff")
    _require(not bool(result["physics_reset"]), "result contains a physics reset")
    _require(not bool(result["physics_teleport"]), "result contains a physics teleport")
    _require(not bool(result["policy_history_reset"]), "result contains a policy-history reset")
    for name in ("q0_forward_count", "q1_forward_count", "wbc_forward_count"):
        value = result[name]
        _require(isinstance(value, int) and value >= 0, f"{name} must be a non-negative integer")
    _require(bool(str(result["gpu_uuid"]).strip()), "GPU UUID is required")
    _require(bool(result["artifact_identities"]), "artifact identities are required")
    _require(bool(result["runtime_identities"]), "runtime identities are required")
    _require(bool(str(result["git_head"]).strip()), "git HEAD is required")
    return {"schema_valid": True, "required_fields": list(_REQUIRED_RESULT_FIELDS)}


_REQUIRED_V6_RESULT_FIELDS = (
    "schema",
    "stage",
    "pi_gate",
    "status",
    "seed",
    "handoff_target_s",
    "handoff_steps",
    "handoff_count",
    "forward_count",
    "reference_update_count",
    "executed_prefix_steps",
    "cumulative_actual_switch_s",
    "cumulative_switch_error_s",
    "physics_reset",
    "physics_teleport",
    "policy_history_reset",
    "gpu_uuid",
    "vram_baseline_mib",
    "vram_peak_mib",
    "artifact_identities",
    "runtime_identities",
    "git_head",
)


def validate_stage40_receding_horizon_result_schema(result: Mapping[str, Any]) -> dict[str, Any]:
    """Require the bounded three-handoff Stage40 development seal."""

    missing = sorted(set(_REQUIRED_V6_RESULT_FIELDS).difference(result))
    _require(not missing, "Stage40 V6 result schema is missing: " + ", ".join(missing))
    _require(str(result["stage"]) == "Stage40", "result stage must be Stage40")
    _require(str(result["status"]) == "PASS", "only a PASS result may be sealed")
    _require(require_development_seed(result["seed"]) == DEV_SEED, "result seed is not DEV_SEED")
    try:
        handoff_target = float(result["handoff_target_s"])
    except (TypeError, ValueError) as exc:
        raise Stage40AdmissionError("handoff_target_s is not numeric") from exc
    _require(
        math.isfinite(handoff_target)
        and math.isclose(handoff_target, STAGE40_HANDOFF_S, rel_tol=0.0, abs_tol=1e-12),
        "handoff_target_s changed from the frozen value",
    )
    _require(int(result["handoff_count"]) == 3, "V6 requires exactly three handoffs")
    _require(int(result["forward_count"]) == 4, "V6 requires exactly four GR00T forwards")
    _require(
        int(result["reference_update_count"]) == 3,
        "V6 requires exactly three reference-only updates",
    )
    prefixes = result["executed_prefix_steps"]
    _require(
        isinstance(prefixes, Sequence) and not isinstance(prefixes, (str, bytes)),
        "V6 executed_prefix_steps must be a sequence",
    )
    _require(
        len(prefixes) == 3 and len({int(item) for item in prefixes}) == 1,
        "V6 must record three equal executed prefixes",
    )
    _require(int(prefixes[0]) == int(result["handoff_steps"]), "V6 handoff prefix mismatch")
    for name in ("vram_baseline_mib", "vram_peak_mib"):
        try:
            value = float(result[name])
        except (TypeError, ValueError) as exc:
            raise Stage40AdmissionError(f"{name} is not numeric") from exc
        _require(math.isfinite(value), f"{name} must be finite")
    for name in ("cumulative_actual_switch_s", "cumulative_switch_error_s"):
        values = result[name]
        _require(
            isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and len(values) == 3,
            f"{name} must contain three finite values",
        )
        try:
            numeric = tuple(float(item) for item in values)
        except (TypeError, ValueError) as exc:
            raise Stage40AdmissionError(f"{name} contains a non-numeric value") from exc
        _require(all(math.isfinite(item) for item in numeric), f"{name} must be finite")
    _require(not bool(result["physics_reset"]), "V6 result contains a physics reset")
    _require(not bool(result["physics_teleport"]), "V6 result contains a physics teleport")
    _require(not bool(result["policy_history_reset"]), "V6 result contains a policy-history reset")
    _require(bool(str(result["gpu_uuid"]).strip()), "GPU UUID is required")
    _require(bool(result["artifact_identities"]), "artifact identities are required")
    _require(bool(result["runtime_identities"]), "runtime identities are required")
    _require(bool(str(result["git_head"]).strip()), "git HEAD is required")
    return {"schema_valid": True, "required_fields": list(_REQUIRED_V6_RESULT_FIELDS)}


__all__ = [
    "DEV_SEED",
    "FORMAL_SEEDS",
    "STAGE40_ACTION_HORIZON",
    "STAGE40_DECODER",
    "STAGE40_DURATION_S",
    "STAGE40_HANDOFF_S",
    "STAGE40_PERIOD_S",
    "STAGE40_PHYSICAL_ACTION_DIM",
    "STAGE40_SOURCE_FRAME",
    "STAGE40_TCP_SEMANTIC_ID",
    "STAGE40_TIMING_MODE",
    "STAGE40_UNIVERSAL_ACTION_DIM",
    "Stage40AdmissionError",
    "handoff_steps_for_native_dt",
    "require_development_seed",
    "validate_artifact_identities",
    "validate_fake_lifecycle",
    "validate_receding_horizon_lifecycle",
    "validate_safety_gate_decision",
    "validate_stage40_receding_horizon_result_schema",
    "validate_stage40_result_schema",
    "validate_structured_action_contract",
]
