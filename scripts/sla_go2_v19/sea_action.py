"""Bind an official decoded return to its paused target observation and tool frame."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from sea_reference import PhysicalIntent, SourceCurve, pose, require
from policy_query import input_identity, tensor_sha

SOURCE_HZ = 30
SOURCE_ROWS = 50


@dataclass(frozen=True)
class BoundPolicyAction:
    intent: PhysicalIntent
    decoded: np.ndarray
    query_state: np.ndarray

    @property
    def decoded_sha256(self):
        return tensor_sha(self.decoded)

    @property
    def times_s(self):
        return np.arange(SOURCE_ROWS) / SOURCE_HZ


def bind_decoded_return(*, episode, tick, observation, expected_observation_sha256, return_observation_sha256,
                        absolute, decoded_sha256, policy_origin, link6_to_tcp, actual_tcp, provenance):
    """Preserve both arms and all 30 Hz rows. Compilation handles the native seam separately."""
    received = input_identity(observation)[0]
    require(received == expected_observation_sha256 == return_observation_sha256,
            "ACTION_OBSERVATION_MISMATCH")
    require(provenance in {"RAW_SENSOR", "DEPLOYABLE_SEA"}, "ORACLE_OR_UNBOUND_PROVENANCE")
    require(absolute.shape == (SOURCE_ROWS, 16) and absolute.dtype == np.float32
            and np.isfinite(absolute).all() and tensor_sha(absolute) == decoded_sha256,
            "ACTION_RETURN_IDENTITY")
    state = np.array(observation["state"], copy=True)
    require(state.shape == (16,) and state.dtype == np.float32 and np.isfinite(state).all()
            and abs(np.linalg.norm(state[11:15])-1) <= 1e-4, "INVALID_ACTION_QUERY_STATE")
    origin, tool, measured = (pose(x).copy() for x in (policy_origin, link6_to_tcp, actual_tcp))
    local = np.eye(4)
    local[:3, :3] = Rotation.from_quat(state[[12, 13, 14, 11]]).as_matrix()
    local[:3, 3] = state[8:11]
    require(np.allclose(origin @ local @ tool, measured, rtol=0, atol=2e-7), "ACTION_QUERY_TOOL_FRAME_MISMATCH")
    decoded = absolute.copy()
    decoded.setflags(write=False)
    state.setflags(write=False)
    curve = SourceCurve(decoded, origin, tool)
    intent = PhysicalIntent(episode, tick, received, provenance, measured, curve)
    intent.verify(episode, tick, measured)
    return BoundPolicyAction(intent, decoded, state)
