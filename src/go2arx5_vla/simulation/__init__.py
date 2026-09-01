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
from .same_scene import (
    NativeSceneObservationV1,
    SameSceneCounterfactualV1,
    SameSceneTargetV1,
    compare_counterfactuals,
    image_sha256,
)

__all__ = [
    "INITIAL_LOCAL_POSITION",
    "INITIAL_QUATERNION_XYZW",
    "SIM_VISUAL_REACH_INSTRUCTION",
    "SimVisualReachSpec",
    "build_visual_reach_raw",
    "default_visual_reach_specs",
    "write_visual_reach_manifest",
    "SameSceneTargetV1",
    "NativeSceneObservationV1",
    "SameSceneCounterfactualV1",
    "compare_counterfactuals",
    "image_sha256",
]
