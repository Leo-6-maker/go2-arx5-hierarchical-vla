"""Install external world EEF references into the private native WBC chain.

The private trajectory module is injected, never bundled. Native WBC retains
preview, spatial progress, arm plan, dog commands and safety termination.
"""

from dataclasses import replace

import numpy as np

from ..contracts.trajectory import TrajectoryChunkV1
from ..trajectory.so3 import geodesic_angle, quat_to_matrix, rot6d_to_matrix


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


class ExternalEEFReferenceBridge:
    mode = "external_task_reference"
    _progress_buffers = ("traj_s", "traj_s_prev", "traj_sim_time", "traj_d_lat",
                         "traj_sdot_meas", "traj_sdot_ref", "traj_timing_err", "traj_twist_err")

    def __init__(self, wbc, trajectory_module, *, tcp_semantic_id, world_frame="world",
                 gripper_limits=(0.0, 1.0)):
        self.wbc, self.native = wbc, trajectory_module
        self.tcp_semantic_id, self.world_frame = tcp_semantic_id, world_frame
        self.gripper_limits = np.asarray(gripper_limits, dtype=float)
        if (self.gripper_limits.shape != (2,) or not np.isfinite(self.gripper_limits).all()
                or self.gripper_limits[0] >= self.gripper_limits[1]):
            raise ValueError("finite ordered gripper limits are required")
        if not tcp_semantic_id or not world_frame:
            raise ValueError("TCP and world frame identity are required")
        if not callable(getattr(trajectory_module, "to_canonical", None)):
            raise ValueError("native trajectory module must expose to_canonical")
        if not hasattr(wbc, "traj_batch") or any(not hasattr(wbc, name) for name in self._progress_buffers):
            raise ValueError("initialize the native trajectory buffers before creating the bridge")
        self.active = {}
        self.receipts = []

    def _env_id(self, env_id):
        if isinstance(env_id, bool) or not isinstance(env_id, (int, np.integer)) or not 0 <= env_id < self.wbc.num_envs:
            raise ValueError("env_id must be a valid integer")

    def _validate(self, chunk, issue, install, transform):
        if not isinstance(chunk, TrajectoryChunkV1) or not chunk.chunk_id:
            raise ValueError("a named TrajectoryChunkV1 is required")
        # Revalidate/copy mutable ndarray inputs so gripper commands cannot change after install.
        chunk = replace(chunk, point_timestamps=chunk.point_timestamps.copy(),
                        eef_position=chunk.eef_position.copy(), eef_orientation=chunk.eef_orientation.copy(),
                        gripper=None if chunk.gripper is None else chunk.gripper.copy(),
                        eef_rot6d=None if chunk.eef_rot6d is None else chunk.eef_rot6d.copy())
        clocks = np.asarray([chunk.observation_timestamp, issue, install, chunk.validity_start, chunk.validity_end])
        if not np.isfinite(clocks).all() or not chunk.observation_timestamp <= issue <= install:
            raise ValueError("observation <= issue <= install timestamps must be finite")
        if not chunk.validity_start <= install <= chunk.validity_end:
            raise ValueError("reference is not valid at installation")
        if chunk.tcp_semantic_id != self.tcp_semantic_id:
            raise ValueError("TCP semantic identity mismatch")
        ts = chunk.point_timestamps
        if (ts.size < 2 or np.any(np.diff(ts) <= 0) or ts[0] < chunk.observation_timestamp
                or ts[0] < chunk.validity_start or ts[-1] > chunk.validity_end
                or not ts[0] <= install <= ts[-1]):
            raise ValueError("EEF grid must cover installation within observation/validity bounds")
        if not np.allclose(np.diff(ts), ts[1] - ts[0], rtol=1e-6, atol=1e-9):
            raise ValueError("native to_canonical requires a uniform EEF time grid")
        if chunk.gripper is None:
            raise ValueError("an independent gripper schedule is required")
        if np.any(chunk.gripper < self.gripper_limits[0]) or np.any(chunk.gripper > self.gripper_limits[1]):
            raise ValueError("gripper schedule exceeds declared limits")
        if chunk.eef_rot6d is not None and not np.allclose(
                rot6d_to_matrix(chunk.eef_rot6d), quat_to_matrix(chunk.eef_orientation), atol=1e-7):
            raise ValueError("column-packed canonical rot6d disagrees with quaternion")
        if chunk.source_frame != self.world_frame:
            tf = np.asarray(transform, dtype=float)
            if (tf.shape != (4, 4) or not np.isfinite(tf).all()
                    or not np.allclose(tf[3], [0, 0, 0, 1], atol=1e-8)
                    or not np.allclose(tf[:3, :3].T @ tf[:3, :3], np.eye(3), atol=1e-8)
                    or not np.isclose(np.linalg.det(tf[:3, :3]), 1, atol=1e-8)):
                raise ValueError("a rigid observation-time source_to_world transform is required")
            return chunk.transformed(tf, target_frame=self.world_frame)
        if transform is not None:
            raise ValueError("world reference must not be transformed again")
        return chunk

    def submit(self, env_id, chunk, *, issue_timestamp, install_timestamp, source_to_world=None):
        """Validate before mutation; rejection holds the previous reference at actual TCP."""
        self._env_id(env_id)
        try:
            world = self._validate(chunk, issue_timestamp, install_timestamp, source_to_world)
            gamma, time_law = self.native.to_canonical(
                world.eef_position, quat_to_matrix(world.eef_orientation),
                float(world.point_timestamps[1] - world.point_timestamps[0]))
            batch = self.wbc.traj_batch
            if len(gamma.s_grid) > batch.max_gamma_points or len(time_law.t_grid) > batch.max_tl_points:
                raise ValueError("native Gamma/TimeLaw capacity insufficient; truncation rejected")
            for field in (gamma.s_grid, gamma.p, gamma.R, gamma.tangent,
                          time_law.t_grid, time_law.s_of_t, time_law.sdot_of_t):
                if not np.isfinite(_numpy(field)).all():
                    raise ValueError("native canonicalization produced nonfinite values")
        except (ValueError, TypeError) as error:
            held = (self.hold(env_id, timestamp=install_timestamp, reason="REJECTED_HOLD")
                    if env_id in self.active and isinstance(install_timestamp, (int, float, np.number))
                    and np.isfinite(install_timestamp) else None)
            receipt = {"status": "REJECTED", "env_id": int(env_id), "ref_id": getattr(chunk, "chunk_id", None),
                       "reason": str(error), "hold_ref_id": None if held is None else held["ref_id"]}
            self.receipts.append(receipt)
            return receipt
        return self._install(env_id, chunk, world, gamma, time_law, issue_timestamp, install_timestamp)

    def _install(self, env_id, raw, world, gamma, time_law, issue, install, status="INSTALLED"):
        import torch
        batch = self.wbc.traj_batch
        ids = torch.tensor([env_id], device=batch.device, dtype=torch.long)
        # Capacity was checked before native load, whose default silently clips tails.
        batch.load(ids, [gamma], [time_law])
        elapsed = float(install - world.point_timestamps[0])
        for name in self._progress_buffers:
            getattr(self.wbc, name)[env_id] = 0
        self.wbc.traj_sim_time[env_id] = elapsed
        times = self.wbc.traj_sim_time
        s_ref = batch.s_ref(times)
        self.wbc.arm_goal_pos_world[env_id] = batch.p_at(s_ref)[env_id]
        self.wbc.arm_goal_quat_world[env_id] = batch.quat_at(s_ref)[env_id]
        sample_times = torch.as_tensor(world.point_timestamps - world.point_timestamps[0], dtype=torch.float32)
        canonical_s = time_law.s_ref(sample_times)
        canonical_pos = _numpy(gamma.p_at(canonical_s))
        canonical_rot = _numpy(gamma.R_at(canonical_s))
        receipt = {
            "status": status, "env_id": int(env_id), "ref_id": raw.chunk_id,
            "mode": self.mode, "source_frame": raw.source_frame, "installed_frame": self.world_frame,
            "source_to_world_at_observation": world.provenance.get("frame_transform"),
            "tcp_semantic_id": self.tcp_semantic_id, "rot6d_convention": "CANONICAL_COLUMNS_NATIVE_PREVIEW_ROWS",
            "observation_timestamp": raw.observation_timestamp, "issue_timestamp": float(issue),
            "install_timestamp": float(install), "install_latency_s": float(install - issue),
            "observation_to_install_s": float(install - raw.observation_timestamp),
            "replaces_ref_id": self.active[env_id].chunk_id if env_id in self.active else None,
            "reference_elapsed_at_install_s": elapsed, "validity": [raw.validity_start, raw.validity_end],
            "raw": {"timestamps": raw.point_timestamps.tolist(), "position": raw.eef_position.tolist(),
                    "quaternion_xyzw": raw.eef_orientation.tolist(), "gripper": raw.gripper.reshape(-1).tolist()},
            "world": {"position": world.eef_position.tolist(), "quaternion_xyzw": world.eef_orientation.tolist()},
            "canonical": {"s": _numpy(gamma.s_grid).tolist(), "position": _numpy(gamma.p).tolist(),
                          "rotation_matrix": _numpy(gamma.R).tolist(), "time": _numpy(time_law.t_grid).tolist(),
                          "s_ref": _numpy(time_law.s_of_t).tolist()},
            "installed": {"position": _numpy(batch.gamma_p[env_id, :len(gamma.s_grid)]).tolist(),
                          "s": _numpy(batch.gamma_s[env_id, :len(gamma.s_grid)]).tolist(),
                          "quaternion_xyzw": _numpy(batch.gamma_quat[env_id, :len(gamma.s_grid)]).tolist(),
                          "time": _numpy(batch.tl_t[env_id, :len(time_law.t_grid)]).tolist(),
                          "s_ref": _numpy(batch.tl_s[env_id, :len(time_law.t_grid)]).tolist(),
                          "sdot_ref": _numpy(batch.tl_sdot[env_id, :len(time_law.t_grid)]).tolist()},
            "canonical_deviation": {"position_max_m": float(np.linalg.norm(canonical_pos - world.eef_position, axis=-1).max()),
                                    "SO3_max_rad": float(geodesic_angle(canonical_rot, quat_to_matrix(world.eef_orientation)).max())},
        }
        self.active[env_id] = world
        for array in (world.point_timestamps, world.eef_position, world.eef_orientation, world.gripper):
            array.setflags(write=False)
        self.receipts.append(receipt)
        return receipt

    def gripper_at(self, env_id, timestamp):
        self._env_id(env_id)
        chunk = self.active[env_id]
        if not np.isfinite(timestamp) or not chunk.validity_start <= timestamp <= chunk.validity_end:
            raise ValueError("gripper schedule expired; install boundary hold before stepping")
        # The gripper uses the same absolute clock; it is never appended to WBC actions.
        return float(np.interp(timestamp, chunk.point_timestamps, chunk.gripper.reshape(-1)))

    def hold(self, env_id, *, timestamp, gripper_command=None, reason="HOLD"):
        self._env_id(env_id)
        if not np.isfinite(timestamp):
            raise ValueError("hold timestamp must be finite")
        if gripper_command is None:
            previous = self.active[env_id]
            gripper_command = float(np.interp(timestamp, previous.point_timestamps, previous.gripper.reshape(-1)))
        if not np.isfinite(gripper_command) or not self.gripper_limits[0] <= gripper_command <= self.gripper_limits[1]:
            raise ValueError("hold gripper command exceeds declared limits")
        pose = _numpy(self.wbc.end_effector_state[env_id, :7]).copy()
        duration = max(float(self.wbc.dt), 0.05)
        # ponytail: a stationary native path is held until explicit replacement/cancel/reset.
        chunk = TrajectoryChunkV1(
            chunk_id=f"{reason}:{env_id}:{timestamp:.9f}", source=self.mode,
            observation_timestamp=timestamp, point_timestamps=np.array([timestamp, timestamp + duration]),
            eef_position=np.repeat(pose[None, :3], 2, axis=0),
            eef_orientation=np.repeat(pose[None, 3:7], 2, axis=0),
            source_frame=self.world_frame, tcp_semantic_id=self.tcp_semantic_id,
            validity_start=timestamp, validity_end=np.finfo(float).max,
            gripper=np.full(2, gripper_command))
        gamma, time_law = self.native.to_canonical(chunk.eef_position, quat_to_matrix(chunk.eef_orientation), duration)
        if len(gamma.s_grid) > self.wbc.traj_batch.max_gamma_points or len(time_law.t_grid) > self.wbc.traj_batch.max_tl_points:
            raise ValueError("native capacity cannot hold two points")
        return self._install(env_id, chunk, chunk, gamma, time_law, timestamp, timestamp, reason)

    def cancel(self, env_id, *, timestamp, gripper_command=None):
        return self.hold(env_id, timestamp=timestamp, gripper_command=gripper_command, reason="CANCELLED")

    def boundary(self, timestamp):
        """Call with refreshed actual TCP before native policy inference each boundary."""
        if not np.isfinite(timestamp):
            raise ValueError("boundary timestamp must be finite")
        return [self.hold(env_id, timestamp=timestamp, reason="EXPIRED_HOLD")
                for env_id, chunk in list(self.active.items()) if timestamp > chunk.validity_end]

    def reset(self, env_ids):
        """Forget only selected episode references; root reset initializes native history/state."""
        for env_id in env_ids:
            self._env_id(env_id)
        for env_id in env_ids:
            self.active.pop(int(env_id), None)
