"""Versioned, runtime-independent execution profiles.

Profiles are the only place where a reusable manifest may name governed
sampling or timestamp semantics.  Generic validation consumes a profile by
name; changing a profile's meaning requires a new versioned name.
"""

from __future__ import annotations

from dataclasses import dataclass


class ProfileError(ValueError):
    """Raised when a manifest names an unknown execution profile."""


@dataclass(frozen=True)
class SamplerProfile:
    """A versioned sampler contract supplied by a stage-specific backend."""

    name: str
    timesteps: tuple[int, ...]
    dt: float
    calls_per_window: int
    semantics: str


@dataclass(frozen=True)
class TimestampProfile:
    """Timestamp validation policy for a source trajectory."""

    name: str
    cadence_hz: float | None
    duration_s: float | None
    semantics: str


SAMPLER_PROFILES = {
    "GROOT_N1D7_SOURCE_MAPPED_EULER4_V1": SamplerProfile(
        name="GROOT_N1D7_SOURCE_MAPPED_EULER4_V1",
        timesteps=(0, 250, 500, 750),
        dt=0.25,
        calls_per_window=4,
        semantics=(
            "Stage45C-proven four-step source-mapped Euler sampler; "
            "same noise is reused for paired checkpoints."
        ),
    ),
}


TIMESTAMP_PROFILES = {
    "GENERIC_MONOTONIC_H_V1": TimestampProfile(
        name="GENERIC_MONOTONIC_H_V1",
        cadence_hz=None,
        duration_s=None,
        semantics="finite, H-length, strictly increasing source timestamps",
    ),
    "STAGE40_T16_20HZ_V1": TimestampProfile(
        name="STAGE40_T16_20HZ_V1",
        cadence_hz=20.0,
        duration_s=1.95,
        semantics="accepted Stage40 T16 fixed-cadence decoder input",
    ),
}


def sampler_profile(name: str) -> SamplerProfile:
    """Resolve a sampler profile without silently accepting new semantics."""

    try:
        return SAMPLER_PROFILES[name]
    except KeyError as exc:
        raise ProfileError(f"unknown sampler profile: {name}") from exc


def timestamp_profile(name: str) -> TimestampProfile:
    """Resolve a timestamp profile without applying a hidden cadence default."""

    try:
        return TIMESTAMP_PROFILES[name]
    except KeyError as exc:
        raise ProfileError(f"unknown timestamp profile: {name}") from exc


__all__ = [
    "ProfileError",
    "SAMPLER_PROFILES",
    "TIMESTAMP_PROFILES",
    "SamplerProfile",
    "TimestampProfile",
    "sampler_profile",
    "timestamp_profile",
]
