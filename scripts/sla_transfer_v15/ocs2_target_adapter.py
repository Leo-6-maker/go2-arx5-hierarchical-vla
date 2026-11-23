"""Convert a complete execution-time window to the installed OCS2 world-reference ABI."""

import json

import numpy as np

from go2arx5_vla.trajectory.so3 import matrix_to_quat


def wire(poses, base, epoch, seq, now):
    def part(arr):
        a = np.asarray(arr)
        if a.shape != (41, 4, 4) or not np.isfinite(a).all():
            raise ValueError("world pose window shape")
        rot = a[:, :3, :3]
        if (
            not np.allclose(a[:, 3, :], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(
                rot.transpose(0, 2, 1) @ rot, np.eye(3), atol=1e-7, rtol=0
            )
            or not np.allclose(np.linalg.det(rot), 1, atol=1e-7, rtol=0)
        ):
            raise ValueError("invalid rigid transform")
        return {
            "frame": "world",
            "start_delay": 0.0,
            "lead_in": 0.0,
            "waypoints": [
                {
                    "time": i * 0.05,
                    "position": p[:3, 3].tolist(),
                    "quaternion_xyzw": q.tolist(),
                }
                for i, (p, q) in enumerate(
                    zip(a, matrix_to_quat(a[:, :3, :3]), strict=True)
                )
            ],
        }

    return json.dumps(
        {
            "_pi0294": {
                "episode": epoch,
                "sequence": seq,
                "source_mpc_s": now,
                "ttl_s": 0.4,
                "operation": "install",
            },
            "ee": part(poses),
            "base": part(base),
        },
        separators=(",", ":"),
        allow_nan=False,
    )
