"""Simulation-only task fixtures used by the Stage20S qualification."""

from .visual_reach import (
    INITIAL_LOCAL_POSITION,
    INITIAL_QUATERNION_XYZW,
    SIM_VISUAL_REACH_INSTRUCTION,
    SimVisualReachSpec,
    build_visual_reach_raw,
    default_visual_reach_specs,
    write_visual_reach_manifest,
)

__all__ = [
    "INITIAL_LOCAL_POSITION",
    "INITIAL_QUATERNION_XYZW",
    "SIM_VISUAL_REACH_INSTRUCTION",
    "SimVisualReachSpec",
    "build_visual_reach_raw",
    "default_visual_reach_specs",
    "write_visual_reach_manifest",
]
