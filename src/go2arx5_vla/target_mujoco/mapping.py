"""Named MuJoCo addresses; the native 18 controls never include task state."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

NATIVE_JOINTS = tuple(f"{leg}_{part}_joint" for leg in ("FR", "FL", "RR", "RL")
                      for part in ("hip", "thigh", "calf")) + tuple(f"x5_joint{i}" for i in range(1, 7))
NATIVE_ACTUATORS = tuple(name.removesuffix("_joint") for name in NATIVE_JOINTS[:12]) + NATIVE_JOINTS[12:]
FINGER_JOINTS = ("pi0288_left_finger", "pi0288_right_finger")
GRIPPER_ACTUATOR = "pi0288_gripper_motor"
BASE_JOINT = "pi0288_base_free"
OBJECT_JOINT = "pi0288_cube_free"
CAMERAS = ("pi0288_body", "pi0288_wrist")


@dataclass(frozen=True)
class JointAddress:
    name: str
    id: int
    type: int
    qposadr: int
    dofadr: int
    nq: int
    nv: int


@dataclass(frozen=True)
class NamedMap:
    nq: int
    nv: int
    nu: int
    joints: Mapping[str, JointAddress]
    actuators: Mapping[str, int]
    sensors: Mapping[str, tuple[int, int, int]]
    cameras: Mapping[str, int]
    tcp_site_id: int

    def manifest(self) -> dict[str, Any]:
        return asdict(self)

    def state(self, qpos: Any, qvel: Any) -> dict[str, np.ndarray]:
        q, v = np.asarray(qpos, dtype=float), np.asarray(qvel, dtype=float)
        if q.shape != (self.nq,) or v.shape != (self.nv,) or not np.isfinite(q).all() or not np.isfinite(v).all():
            raise ValueError("finite qpos[nq] and qvel[nv] required")
        result = {}
        for prefix, name in (("base", BASE_JOINT), ("object", OBJECT_JOINT)):
            if name not in self.joints:
                continue
            j = self.joints[name]
            pose = q[j.qposadr:j.qposadr + 7].copy()
            if not np.isclose(np.linalg.norm(pose[3:]), 1., atol=1e-6):
                raise ValueError(f"{name}: non-unit WXYZ quaternion")
            result[f"{prefix}_xyz"] = pose[:3]
            result[f"{prefix}_quaternion_xyzw"] = pose[[4, 5, 6, 3]]
            result[f"{prefix}_velocity"] = v[j.dofadr:j.dofadr + 6].copy()
        for prefix, names in (("native", NATIVE_JOINTS), ("finger", FINGER_JOINTS)):
            if not all(n in self.joints for n in names):
                continue
            result[f"{prefix}_qpos"] = np.array([q[self.joints[n].qposadr] for n in names])
            result[f"{prefix}_qvel"] = np.array([v[self.joints[n].dofadr] for n in names])
        return result

    def sensor_values(self, sensordata: Any) -> dict[str, np.ndarray]:
        values = np.asarray(sensordata, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("finite 1D sensordata required")
        if any(adr + dim > values.size for _, adr, dim in self.sensors.values()):
            raise ValueError("sensordata is shorter than named sensor map")
        return {name: values[adr:adr + dim].copy() for name, (_, adr, dim) in self.sensors.items()}

    def dispatch(self, ctrl: Any, native_torques: Any = None, gripper_force: float | None = None) -> None:
        """Write only the explicitly named controls on the owner's mjData."""
        if np.shape(ctrl) != (self.nu,):
            raise ValueError("ctrl must have shape [nu]")
        if native_torques is not None:
            values = np.asarray(native_torques, dtype=float)
            if values.shape != (18,) or not np.isfinite(values).all():
                raise ValueError("native dispatch requires finite original 18 torques")
            for name, value in zip(NATIVE_ACTUATORS, values):
                ctrl[self.actuators[name]] = value
        if gripper_force is not None:
            if not np.isfinite(gripper_force):
                raise ValueError("gripper force must be finite")
            ctrl[self.actuators[GRIPPER_ACTUATOR]] = gripper_force


def build_named_map(model: Any, *, equipped: bool = True) -> NamedMap:
    """Accept a MuJoCo MjModel or a synthetic model exposing named accessors.

    MuJoCo types are free=0, ball=1, slide=2, hinge=3. No MuJoCo import,
    model load, physics step or asset copy occurs here.
    """
    names = (BASE_JOINT,) + NATIVE_JOINTS + (FINGER_JOINTS + (OBJECT_JOINT,) if equipped else ())
    joints = {}
    for name in names:
        try:
            jid = int(model.joint(name).id)
        except (KeyError, ValueError):
            if name != BASE_JOINT or equipped:
                raise
            # Native delivery leaves its freejoint unnamed; resolve through the named base.
            jid = int(model.body_jntadr[int(model.body("base_link").id)])
        kind = int(model.jnt_type[jid])
        if kind not in (0, 1, 2, 3):
            raise ValueError(f"{name}: invalid joint type {kind}")
        nq, nv = ((7, 6), (4, 3), (1, 1), (1, 1))[kind]
        expected = 0 if name in (BASE_JOINT, OBJECT_JOINT) else 2 if name in FINGER_JOINTS else 3
        if kind != expected:
            raise ValueError(f"{name}: unexpected joint type {kind}")
        joints[name] = JointAddress(name, jid, kind, int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]), nq, nv)
    for field, width, size in (("qposadr", "nq", int(model.nq)), ("dofadr", "nv", int(model.nv))):
        addresses = [i for j in joints.values() for i in range(getattr(j, field), getattr(j, field) + getattr(j, width))]
        if len(addresses) != len(set(addresses)) or any(i < 0 or i >= size for i in addresses):
            raise ValueError(f"overlapping or invalid {field} map")
    actuators = {}
    pairs = list(zip(NATIVE_ACTUATORS, NATIVE_JOINTS))
    if equipped:
        pairs.append((GRIPPER_ACTUATOR, FINGER_JOINTS[0]))
    for name, joint in pairs:
        aid = int(model.actuator(name).id)
        if int(model.actuator_trntype[aid]) != 0 or int(model.actuator_trnid[aid][0]) != joints[joint].id:
            raise ValueError(f"{name}: expected direct joint transmission to {joint}")
        actuators[name] = aid
    if len(set(actuators.values())) != len(actuators) or any(i < 0 or i >= int(model.nu) for i in actuators.values()):
        raise ValueError("overlapping or invalid actuator map")
    sensors = {}
    for joint in NATIVE_JOINTS:
        stem = joint.removesuffix("_joint") if joint in NATIVE_JOINTS[:12] else joint
        for suffix in ("pos", "vel", "torque"):
            name = f"{stem}_{suffix}"
            sid = int(model.sensor(name).id)
            sensors[name] = (sid, int(model.sensor_adr[sid]), int(model.sensor_dim[sid]))
    for name in ("imu_quat", "imu_gyro", "imu_acc", "frame_pos", "frame_vel"):
        sid = int(model.sensor(name).id)
        sensors[name] = (sid, int(model.sensor_adr[sid]), int(model.sensor_dim[sid]))
    if equipped:
        for joint in FINGER_JOINTS:
            for suffix in ("pos", "vel"):
                name = f"{joint}_{suffix}"
                sid = int(model.sensor(name).id)
                sensors[name] = (sid, int(model.sensor_adr[sid]), int(model.sensor_dim[sid]))
    for name, (_, adr, dim) in sensors.items():
        expected = 4 if name == "imu_quat" else 3 if name in ("imu_gyro", "imu_acc", "frame_pos", "frame_vel") else 1
        if dim != expected or adr < 0 or adr + dim > int(model.nsensordata):
            raise ValueError(f"{name}: invalid sensor address/dimension")
    camera_ids = [int(model.camera(name).id) for name in CAMERAS] if equipped else []
    if len(set(camera_ids)) != len(camera_ids):
        raise ValueError("body and wrist cameras cannot alias")
    cameras = dict(zip(CAMERAS, camera_ids)) if equipped else {}
    return NamedMap(int(model.nq), int(model.nv), int(model.nu), joints, actuators, sensors, cameras, int(model.site("x5_ee").id))
