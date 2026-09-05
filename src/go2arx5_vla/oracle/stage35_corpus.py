"""Deterministic Stage35 oracle-corpus candidate generator.

This module intentionally contains only NumPy and pure geometry.  It creates
an auditable candidate corpus for the Stage35 motion-API qualification, but
it never promotes a generated trajectory to native Stage16 authority.  The
native WBC archive, reach table, and exact historical reference manifest must
still be recovered independently before any WBC campaign can use the output.
"""

from __future__ import annotations

from typing import Any

import numpy as np


CORPUS_SCHEMA = "STAGE35_ORACLE_TRAJECTORY_V1"
PROTOCOL_SCHEMA = "STAGE35_ORACLE_CORPUS_PROTOCOL_V1"
GENERATOR_VERSION = "stage35-deterministic-corpus-v1"
CORPUS_CLASSES = ("N", "T", "P", "C")
CASES_PER_CLASS = 8
CORPUS_SAMPLE_COUNT = 121
DURATION_S = 2.0
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP_SEMANTIC_ID = "x5_link6_gripper_center"
LAMBDA_M_PER_RAD = 0.15

# These are deliberately an analytic witness, not a substitute for the
# delivered WBC reach table.  C cases therefore remain non-authoritative until
# the exact native support inputs are recovered and independently bound.
ANALYTIC_SUPPORT_X_MIN_M = -0.35
ANALYTIC_SUPPORT_X_MAX_M = 0.35
ANALYTIC_SUPPORT_Y_MIN_M = -0.30
ANALYTIC_SUPPORT_Y_MAX_M = 0.30
ANALYTIC_SUPPORT_Z_MIN_M = -0.25
ANALYTIC_SUPPORT_Z_MAX_M = 0.25


def _unit_quaternion_z(angle: np.ndarray) -> np.ndarray:
    half = 0.5 * np.asarray(angle, dtype=np.float64)
    return np.stack(
        [
            np.zeros_like(half),
            np.zeros_like(half),
            np.sin(half),
            np.cos(half),
        ],
        axis=-1,
    )


def _strict_timestamps(progress: np.ndarray, index: int, family: str) -> np.ndarray:
    """Return deterministic source timestamps for a trajectory parameter."""

    u = np.asarray(progress, dtype=np.float64)
    phase = 0.13 * float(index + 1)
    if family == "N":
        warp = 0.008 * np.sin(2.0 * np.pi * u + phase)
    elif family == "T":
        # T cases are intentionally time-law variants.  The amplitude remains
        # small enough that the derivative is strictly positive everywhere.
        warp = (0.055 + 0.008 * (index % 4)) * np.sin(2.0 * np.pi * u + phase)
    elif family == "P":
        warp = 0.025 * np.sin(2.0 * np.pi * u + phase)
    elif family == "C":
        warp = (0.040 + 0.004 * (index % 3)) * np.sin(2.0 * np.pi * u + phase)
    else:
        raise ValueError(f"unknown corpus family: {family}")
    times = DURATION_S * (u + warp * u * (1.0 - u))
    times[0] = 0.0
    times[-1] = DURATION_S
    if np.any(np.diff(times) <= 0.0):
        raise AssertionError("deterministic timestamp law is not strictly increasing")
    return times


def _base_witness(progress: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a hidden smooth base witness and arm-relative EEF path."""

    u = np.asarray(progress, dtype=np.float64)
    phase = 0.17 * float(index + 1)
    base = np.stack(
        [
            (0.80 + 0.025 * index) * u,
            0.10 * np.sin(np.pi * u + phase) - 0.05 * np.sin(phase),
            np.zeros_like(u),
        ],
        axis=-1,
    )
    relative = np.stack(
        [
            0.04 + 0.015 * np.sin(2.0 * np.pi * u + phase),
            0.06 * np.sin(2.0 * np.pi * u + 0.5 * phase),
            0.04 + 0.025 * np.sin(np.pi * u + phase),
        ],
        axis=-1,
    )
    return base, relative


def build_case(index: int, family: str) -> dict[str, Any]:
    """Build one deterministic candidate trajectory as JSON-compatible data."""

    family = str(family).upper()
    if family not in CORPUS_CLASSES:
        raise ValueError(f"family must be one of {CORPUS_CLASSES}")
    if int(index) < 0 or int(index) >= CASES_PER_CLASS:
        raise ValueError(f"index must be in [0,{CASES_PER_CLASS})")
    index = int(index)
    u = np.linspace(0.0, 1.0, CORPUS_SAMPLE_COUNT, dtype=np.float64)
    phase = 0.19 * float(index + 1)

    if family == "C":
        base, relative = _base_witness(u, index)
        positions = base + relative
        hidden_support: dict[str, Any] = {
            "base_witness_hidden_from_wbc": True,
            "base_position_m": base.tolist(),
            "arm_relative_position_m": relative.tolist(),
            "support_proof_status": "ANALYTIC_WITNESS_ONLY_NOT_NATIVE_WBC_PROOF",
            "native_reach_table_required": True,
            "fixed_base_projection": "x",
            "fixed_base_support_interval_m": [
                ANALYTIC_SUPPORT_X_MIN_M,
                ANALYTIC_SUPPORT_X_MAX_M,
            ],
            "fixed_base_world_x_span_m": float(np.ptp(positions[:, 0])),
            "fixed_base_impossibility_proven_under_analytic_envelope": bool(
                np.ptp(positions[:, 0])
                > ANALYTIC_SUPPORT_X_MAX_M - ANALYTIC_SUPPORT_X_MIN_M
            ),
            "wbc_visible_fields": ["eef_position", "eef_orientation", "timestamps", "gripper"],
            "wbc_hidden_fields": ["base_position_m", "arm_relative_position_m"],
        }
    else:
        base = np.zeros((CORPUS_SAMPLE_COUNT, 3), dtype=np.float64)
        relative = None
        if family == "N":
            positions = np.stack(
                [
                    0.16 + 0.22 * u + 0.018 * np.sin(2.0 * np.pi * u + phase),
                    0.04 + 0.075 * np.sin(np.pi * u + phase),
                    0.06 + 0.025 * np.sin(2.0 * np.pi * u + 0.5 * phase),
                ],
                axis=-1,
            )
        elif family == "T":
            positions = np.stack(
                [
                    0.14 + 0.24 * u + 0.035 * np.sin(3.0 * np.pi * u + phase),
                    0.02 + (0.05 + 0.006 * index) * np.sin(2.0 * np.pi * u + phase),
                    0.05 + 0.030 * np.cos(2.0 * np.pi * u + phase),
                ],
                axis=-1,
            )
        elif family == "P":
            positions = np.stack(
                [
                    0.10 + (0.32 + 0.010 * index) * u,
                    -0.10 + (0.18 + 0.008 * index) * u + 0.025 * np.sin(np.pi * u + phase),
                    0.03 + 0.035 * np.sin(np.pi * u + phase),
                ],
                axis=-1,
            )
        else:
            raise AssertionError(f"unhandled family {family}")
        hidden_support = {
            "base_witness_hidden_from_wbc": False,
            "base_position_m": base.tolist(),
            "arm_relative_position_m": positions.tolist(),
            "support_proof_status": "NOT_APPLICABLE_UNTIL_NATIVE_AUTHORITY",
            "native_reach_table_required": True,
            "wbc_visible_fields": ["eef_position", "eef_orientation", "timestamps", "gripper"],
            "wbc_hidden_fields": [],
        }

    yaw = 0.10 * np.sin(np.pi * u + phase) + 0.035 * u
    orientation = _unit_quaternion_z(yaw)
    timestamps = _strict_timestamps(u, index, family)
    gripper = np.clip(0.20 + 0.60 * u + 0.04 * np.sin(2.0 * np.pi * u + phase), 0.0, 1.0)
    case_id = f"{family}{index + 1:02d}"
    return {
        "schema": CORPUS_SCHEMA,
        "case_id": case_id,
        "family": family,
        "family_index": index,
        "sample_count": CORPUS_SAMPLE_COUNT,
        "duration_s": DURATION_S,
        "timestamps": timestamps.tolist(),
        "eef_position_m": positions.tolist(),
        "eef_orientation_xyzw": orientation.tolist(),
        "gripper": gripper.tolist(),
        "source_frame": SOURCE_FRAME,
        "tcp_semantic_id": TCP_SEMANTIC_ID,
        "validity_start": 0.0,
        "validity_end": DURATION_S,
        "provenance": {
            "generator": GENERATOR_VERSION,
            "family": family,
            "family_index": index,
            "parameterization": "u=uniform_normalized_source_parameter",
            "timestamp_policy": "deterministic_strict_monotonic_analytic_law",
            "geometry_policy": "deterministic_piecewise_smooth_source_witness",
            "outcome_independent": True,
            "authoritative_stage16_source": False,
            "native_wbc_inputs_bound": False,
        },
        "support_witness": hidden_support,
    }


def case_ids() -> list[str]:
    """Return the canonical lexical order of all 32 candidate case IDs."""

    return [
        f"{family}{index + 1:02d}"
        for family in CORPUS_CLASSES
        for index in range(CASES_PER_CLASS)
    ]


def protocol_manifest() -> dict[str, Any]:
    """Return the committed, outcome-independent corpus protocol manifest."""

    return {
        "schema": PROTOCOL_SCHEMA,
        "version": 1,
        "generator": "go2arx5_vla.oracle.stage35_corpus",
        "generator_version": GENERATOR_VERSION,
        "case_count": len(CORPUS_CLASSES) * CASES_PER_CLASS,
        "cases_per_class": CASES_PER_CLASS,
        "class_counts": {family: CASES_PER_CLASS for family in CORPUS_CLASSES},
        "case_ids": case_ids(),
        "sample_count": CORPUS_SAMPLE_COUNT,
        "duration_s": DURATION_S,
        "source_frame": SOURCE_FRAME,
        "tcp_semantic_id": TCP_SEMANTIC_ID,
        "se3_lambda_m_per_rad": LAMBDA_M_PER_RAD,
        "geometry_parameterization": "deterministic_source_trajectory_with_uniform_u_grid",
        "geometry_interpolation_for_candidates": "LINEAR_XYZ_QUATERNION_SLERP",
        "timing_parameterization": "strict_monotonic_source_timestamps",
        "classes": {
            "N": "nominal reaching, eight deterministic cases",
            "T": "timing/speed/curvature variation, eight deterministic cases",
            "P": "progressive relocation, eight deterministic cases",
            "C": "continuous-base hidden-witness candidates, eight deterministic cases",
        },
        "continuous_base_design": {
            "world_eef": "B(t) compose r(t)",
            "base_hidden_from_wbc": True,
            "support_witness": "analytic_projection_envelope_only",
            "native_reach_table_required_before_authority": True,
            "fixed_base_outcome_independent": True,
        },
        "promotion_policy": {
            "authoritative_until_native_stage16_recovery": False,
            "sentinel_promotion_allowed": False,
            "wbc_campaign_allowed_from_this_manifest_alone": False,
        },
    }


__all__ = [
    "CASES_PER_CLASS",
    "CORPUS_CLASSES",
    "CORPUS_SAMPLE_COUNT",
    "DURATION_S",
    "SOURCE_FRAME",
    "TCP_SEMANTIC_ID",
    "build_case",
    "case_ids",
    "protocol_manifest",
]
