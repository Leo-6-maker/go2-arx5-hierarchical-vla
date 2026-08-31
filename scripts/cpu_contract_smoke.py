#!/usr/bin/env python3
"""Dependency-light contract smoke for environments without pytest.

The full test suite remains in ``tests/`` and runs in CI with pytest. This
runner uses only NumPy and the package, so the existing Python 3.8 node-local
runtime can verify the canonical math without changing system packages.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from go2arx5_vla.adapters import GR00TToWBCAdapter, ReferenceAdapterConfig
from go2arx5_vla.contracts import TrajectoryChunkV1
from go2arx5_vla.metrics import task_space_error
from go2arx5_vla.trajectory.preview import DEFAULT_WBC_PREVIEW_OFFSETS_M, se3_arc_length_preview
from go2arx5_vla.trajectory.so3 import geodesic_angle, matrix_to_quat, rot6d_to_matrix, rotation_matrix_to_rot6d


def z_rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def chunk(position, rotations=None, times=None, gripper=None):
    position = np.asarray(position, dtype=float)
    if times is None:
        times = np.linspace(0.0, 1.0, len(position))
    if rotations is None:
        rotations = np.repeat(np.eye(3)[None], len(position), axis=0)
    return TrajectoryChunkV1(
        chunk_id="cpu-smoke",
        source="cpu-smoke",
        observation_timestamp=0.0,
        point_timestamps=times,
        eef_position=position,
        eef_orientation=matrix_to_quat(np.asarray(rotations)),
        source_frame="wbc_world",
        tcp_semantic_id="arx5_tcp_v0",
        validity_start=float(times[0]),
        validity_end=float(times[-1]),
        gripper=gripper,
    )


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print("PASS", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=None)
    args = parser.parse_args()
    del args

    rotations = np.asarray([z_rotation(0.2), z_rotation(-1.1)])
    decoded = rot6d_to_matrix(rotation_matrix_to_rot6d(rotations))
    check("rot6d_round_trip", np.max(np.abs(decoded - rotations)) < 1e-12)

    line = chunk([[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]])
    preview = se3_arc_length_preview(line)
    check("arc_length_offsets", np.allclose(preview.eef_position[:, 0], DEFAULT_WBC_PREVIEW_OFFSETS_M))
    check("arc_length_endpoint", bool(preview.endpoint_clamped[-1]))

    stationary = se3_arc_length_preview(chunk(np.repeat([[0.2, -0.1, 0.4]], 4, axis=0)))
    check("stationary_hold", np.allclose(stationary.eef_position, [0.2, -0.1, 0.4]))

    rotation_only = se3_arc_length_preview(
        chunk(np.zeros((3, 3)), [z_rotation(0), z_rotation(np.pi / 2), z_rotation(np.pi)])
    )
    rot_preview = rot6d_to_matrix(rotation_only.eef_rot6d)
    check("rotation_only_finite", np.isfinite(rotation_only.xyz_rot6d).all())
    check("rotation_only_progress", geodesic_angle(np.eye(3), rot_preview)[-1] > 0.0)

    ts = np.linspace(0.0, 2.0, 40)
    xyz = np.c_[0.2 * ts, 0.03 * np.sin(ts * 5), np.zeros_like(ts)]
    r6 = rotation_matrix_to_rot6d(np.repeat(np.eye(3)[None], len(ts), axis=0))
    adapter = GR00TToWBCAdapter(
        ReferenceAdapterConfig("wbc_world", "wbc_world", "arx5_tcp_v0", stale_timeout_s=0.1)
    )
    for age_ms in (0, 20, 50, 100, 200):
        out, _ = adapter.adapt_xyz_rot6d(
            chunk_id="latency",
            observation_timestamp=0.0,
            point_timestamps=ts,
            eef_xyz=xyz,
            eef_rot6d=r6,
            controller_timestamp=age_ms / 1000.0,
        )
        check("latency_%dms" % age_ms, np.isfinite(out.xyz_rot6d).all())
    high = chunk(
        np.c_[0.25 * ts, 0.08 * np.sin(8 * np.pi * ts / ts[-1]), np.zeros_like(ts)],
        [z_rotation(1.2 * np.sin(6 * np.pi * t / ts[-1])) for t in ts],
        ts,
    )
    high_preview = se3_arc_length_preview(high)
    check("high_curvature", np.isfinite(high_preview.xyz_rot6d).all())

    eye = np.repeat(np.eye(3)[None], 3, axis=0)
    p = np.zeros((3, 3))
    q = p.copy()
    q[:, 0] = 0.1
    result = task_space_error(p, q, eye, eye, metric_name="CONTROLLER_TRACKING_ERROR")
    check("known_metric", abs(result["position_m"]["mean"] - 0.1) < 1e-12)
    print("CPU_CONTRACT_SMOKE_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
