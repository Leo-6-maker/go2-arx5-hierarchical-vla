"""CPU-only conversion of a planning witness to persistent EEF10 chunks.

This prepares references and interval accounting. It cannot launch a worker,
authorize physics, command joints, prove WBC consumption or score task success.
"""
import hashlib
import json
import math

import numpy as np
from go2arx5_vla.task_data.codec import pose10
from go2arx5_vla.trajectory.so3 import geodesic_angle
from .base_intent import eef_document
from .persistent_approach_draft import pose
from .task_clock import actual_source_age
from .task_gripper_contract import require_working_target
from .task_path_planning import native_configuration_witness

VERSION = 'PERSISTENT_PLANNED_WORLD_EEF10_REFERENCE_PREPARATION_V1'


def rates(poses, times):
    dt = np.diff(times)
    linear = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)/dt
    angular = np.array([geodesic_angle(a[:3, :3], b[:3, :3])
        for a, b in zip(poses, poses[1:])])/dt
    return float(linear.max()), float(angular.max())


def timed_native_fk(native, joints):
    """FK of planned joints in a private copy, explicitly not actual feedback."""
    import mujoco
    result = []
    try:
        for row in joints:
            native.data.qpos[:] = native.qpos
            native.data.qpos[native.joint_addresses[:6]] = row
            mujoco.mj_forward(native.model, native.data)
            p = np.eye(4)
            p[:3, 3] = native.data.site_xpos[native.site]
            p[:3, :3] = native.data.site_xmat[native.site].reshape(3, 3)
            result.append(p)
    finally:
        native.data.qpos[:] = native.qpos
        mujoco.mj_forward(native.model, native.data)
    return np.asarray(result)


def retime_witness(native, joints, source_times):
    """One deterministic global slowdown, then recheck the actual 20Hz knots.

    The mathematical path is explicitly piecewise linear in the supplied joint
    witness. No measured observation changes, bias correction or rate search.
    """
    q, t = np.asarray(joints, dtype=float), np.asarray(source_times, dtype=float)
    if (q.ndim != 2 or q.shape[1] != 6 or not 2 <= len(q) <= 10000
            or t.shape != (len(q),) or not np.isfinite(q).all() or not np.isfinite(t).all()
            or abs(t[0]) > 1e-10 or np.any(np.diff(t) <= 0) or t[-1] > 32):
        raise ValueError('FINITE_ORDERED_BOUNDED_PLANNING_WITNESS_REQUIRED')
    if not np.allclose(q[0], native.qpos[native.joint_addresses[:6]], atol=1e-10, rtol=0):
        raise ValueError('PLANNED_PATH_MUST_START_AT_THE_SAVED_ACTUAL_CONFIGURATION')
    if np.any(q < native.arm_limits[:, 0]) or np.any(q > native.arm_limits[:, 1]):
        raise ValueError('PLANNED_NATIVE_JOINT_LIMIT_REJECT_NO_CLIP')
    source_poses = timed_native_fk(native, q)
    linear, angular = rates(source_poses, t)
    factor = math.ceil(max(1., linear/.06, angular/.3)*10)/10
    duration = math.ceil(t[-1]*factor/.05)*.05
    if duration > 32:
        raise ValueError('TIMED_TASK_PATH_EXCEEDS_DECLARED_32_SECONDS')
    times = np.arange(round(duration/.05)+1)*.05
    parameter = times/duration*t[-1]
    witness = np.stack([np.interp(parameter, t, q[:, j]) for j in range(6)], axis=1)
    check, _ = native_configuration_witness(native, witness)
    if check['status'] != 'DISCRETE_NATIVE_CONFIGURATION_CHECK_PASS':
        return dict(status=check['status'], configuration_check=check, launch_allowed=False), None
    poses = timed_native_fk(native, witness)
    speed, spin = rates(poses, times)
    if speed > .06+1e-9 or spin > .3+1e-9:
        raise ValueError('RESAMPLED_CARTESIAN_RATE_REJECT_NO_ITERATIVE_RETIMING')
    return dict(status='TIMED_STATIC_REFERENCE_PREPARED_NOT_EXECUTED', factor=factor,
        duration_s=duration, samples=len(times), linear_max_m_s=speed,
        angular_max_rad_s=spin, configuration_check=check,
        continuous_swept_collision_proved=False, physical_execution_qualified=False,
        launch_allowed=False), dict(time_s=times, poses_world=poses,
            joint_witness=witness, joint_witness_is_command=False)


class PersistentPathReference:
    """A proposed task-stage path; observation clock and references stay distinct.

    Exact installation ACK permits interval accounting, never a solver-consumption
    claim. The physical owner still applies native safety, STOP and admission.
    """
    def __init__(self, *, episode_id, poses_world, opening, initial_native_step):
        poses = np.asarray(poses_world, dtype=float)
        values = np.asarray(opening, dtype=float)
        if (not isinstance(episode_id, str) or not episode_id
                or type(initial_native_step) is not int or initial_native_step < 0
                or poses.ndim != 3 or poses.shape[1:] != (4, 4) or not 2 <= len(poses) <= 641
                or values.shape != (len(poses),)):
            raise ValueError('EXPLICIT_BOUNDED_WORLD_PATH_AND_NATIVE_CLOCK_REQUIRED')
        self.poses = np.stack([pose(p) for p in poses])
        require_working_target(values)
        if not np.isfinite(values).all():
            raise ValueError('FINITE_GRIPPER_TARGET_REQUIRED')
        speed, spin = rates(self.poses, np.arange(len(poses))*.05)
        if speed > .06+1e-9 or spin > .3+1e-9:
            raise ValueError('TASK_PATH_CARTESIAN_RATE_REJECT')
        self.opening = values.copy()
        self.episode_id, self.initial_step = episode_id, initial_native_step
        self.index = -1
        self.pending = self.active = None
        self.stop_reason = None
        self.intervals = []
        self.actual = None
        payload = self.poses.astype('<f8').tobytes()+self.opening.astype('<f8').tobytes()
        self.path_sha256 = hashlib.sha256(payload).hexdigest()

    def observe(self, *, native_step, raw_time_s, actual_tcp, native_safe):
        if type(native_safe) is not bool or not native_safe:
            self.stop_reason = 'ORIGINAL_NATIVE_SAFETY_STOP'
            raise PermissionError(self.stop_reason)
        if self.stop_reason:
            raise PermissionError(self.stop_reason)
        expected = self.initial_step+(self.index+1)*25
        if (type(native_step) is not int or native_step != expected or not np.isfinite(raw_time_s)
                or abs(raw_time_s-native_step*.002) > 1e-8 or self.index+1 > 640):
            raise ValueError('EXACT_NATIVE_20HZ_ACTUAL_OBSERVATION_REQUIRED')
        actual = pose(actual_tcp)
        if self.index == -1 and not np.allclose(actual, self.poses[0], atol=1e-8, rtol=0):
            raise ValueError('REPLAN_REQUIRED_ACTUAL_START_DIFFERS_FROM_PATH')
        if self.index >= 0:
            if self.active is None or not self.active['start_index'] <= self.index < self.active['start_index']+8:
                self.stop_reason = 'NO_INSTALLED_PREFIX_FOR_PRECEDING_PHYSICAL_INTERVAL'
                raise PermissionError(self.stop_reason)
            prefix_index = self.index-self.active['start_index']
            self.intervals.append(dict(interval_index=self.index,
                ref_id=self.active['ref_id'], prefix_index=prefix_index,
                action_EEF10=self.active['action'][prefix_index].copy(),
                start_native_step=native_step-25, end_native_step=native_step,
                installation_known=True, physical_interval_observed=True,
                selected_MPC_policy_consumption='UNKNOWN'))
        self.index += 1
        self.actual, self.raw_time_s = actual, float(raw_time_s)
        return dict(observation_index=self.index,
            path_end_time_reached=self.index >= len(self.poses)-1,
            stage_transition='REQUIRES_ACTUAL_GEOMETRY_NOT_PATH_TIMER',
            task_success=False)

    def chunk(self):
        if self.stop_reason:
            raise PermissionError(self.stop_reason)
        if self.index < 0 or self.index % 8 or self.index == 640 or self.pending is not None:
            raise ValueError('ONE_NORMAL_H40_PER_EIGHT_ACTUAL_OBSERVATIONS')
        if self.active is not None and self.active['start_index'] == self.index:
            raise ValueError('DUPLICATE_TASK_PATH_CHUNK')
        indices = np.minimum(self.index+np.arange(40), len(self.poses)-1)
        action = np.stack([pose10(self.poses[k], self.opening[k]) for k in indices])
        source = round(self.raw_time_s*1e9)/1e9
        if not 0 <= actual_source_age(self.raw_time_s, source) <= .025:
            raise ValueError('SAME_ACTUAL_PACKET_SOURCE_CLOCK_REQUIRED')
        document = eef_document(action)
        sequence = self.index//8
        document['_pi0294'] = dict(episode=self.episode_id, sequence=sequence,
            source_mpc_s=source, ttl_s=.4, operation='install')
        wire = json.dumps(document, separators=(',', ':'), allow_nan=False)
        ref_id = hashlib.sha256(wire.encode()).hexdigest()
        self.pending = dict(ref_id=ref_id, start_index=self.index,
            source=source, sequence=sequence, action=action.copy())
        return dict(version=VERSION, launch_allowed=False, wire=wire,
            ref_id=ref_id, action_EEF10=action.copy(), path_sha256=self.path_sha256,
            start_index=self.index, sequence=sequence, source_mpc_s=source,
            prefix_points=8, action_labels='PROPOSED_UNTIL_EXACT_INSTALL_ACK_AND_INTERVAL_RECORD',
            BASE_supervision=False, task_success=False)

    def acknowledge_install(self, *, ref_id, native_step):
        if self.stop_reason:
            raise PermissionError(self.stop_reason)
        if (self.pending is None or ref_id != self.pending['ref_id']
                or type(native_step) is not int
                or native_step != self.initial_step+self.index*25):
            raise ValueError('EXACT_SAME_OBSERVATION_INSTALL_ACK_REQUIRED')
        self.active, self.pending = self.pending, None

    def stop(self, reason):
        self.stop_reason = str(reason)
