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
from .expert import (
    SameSceneExpertConfigV1,
    build_same_scene_expert_chunk,
    expert_pose_sequence,
    write_rgb_ppm,
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
    "SameSceneExpertConfigV1",
    "build_same_scene_expert_chunk",
    "expert_pose_sequence",
    "write_rgb_ppm",
]
