"""Lightweight source-backed native ARX5 kinematics.

This module parses only the arm chain from the delivered URDF.  It is used for
Stage 15 shape placement and FK audits; it has no simulator, policy, or
hardware dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ...trajectory.relative_se3 import _as_pose


def _vec(text: str | None, size: int, default: np.ndarray | None = None) -> np.ndarray:
    if text is None:
        return np.zeros(size, dtype=np.float64) if default is None else np.asarray(default, dtype=np.float64)
    values = np.fromstring(text, sep=" ", dtype=np.float64)
    if values.size != size:
        raise ValueError("URDF vector has unexpected width")
    return values


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw rotation ``Rz(yaw) Ry(pitch) Rx(roll)``."""
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.asarray(rotation, dtype=np.float64)
    out[:3, 3] = np.asarray(translation, dtype=np.float64)
    return out


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    c, s = np.cos(float(angle)), np.sin(float(angle))
    C = 1.0 - c
    return np.asarray(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class JointSpec:
    name: str
    parent: str
    child: str
    joint_type: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None

    @property
    def origin_transform(self) -> np.ndarray:
        return homogeneous(rpy_to_matrix(self.origin_rpy), self.origin_xyz)


@dataclass(frozen=True)
class NativeArmKinematics:
    """The native ``x5_base_link -> x5_link6`` chain plus TCP offset."""

    mount: JointSpec
    joints: tuple[JointSpec, ...]
    tcp_parent_link: str
    tcp_offset_xyz: np.ndarray
    tcp_semantic_id: str = "gripper_center"

    @classmethod
    def from_urdf(
        cls,
        path: str | Path,
        *,
        mount_joint: str = "arm_mount_joint",
        arm_joint_names: tuple[str, ...] = (
            "x5_joint1", "x5_joint2", "x5_joint3",
            "x5_joint4", "x5_joint5", "x5_joint6",
        ),
        tcp_parent_link: str = "x5_link6",
        tcp_offset_xyz: np.ndarray = np.asarray([0.1424, 0.0, 0.0001057]),
    ) -> "NativeArmKinematics":
        root = ET.parse(str(path)).getroot()

        def parse_joint(element: ET.Element) -> JointSpec:
            origin = element.find("origin")
            axis = element.find("axis")
            limit = element.find("limit")
            parent = element.find("parent")
            child = element.find("child")
            if parent is None or child is None:
                raise ValueError("URDF joint is missing parent/child")
            return JointSpec(
                name=str(element.attrib["name"]),
                parent=str(parent.attrib["link"]),
                child=str(child.attrib["link"]),
                joint_type=str(element.attrib.get("type", "fixed")),
                origin_xyz=_vec(None if origin is None else origin.attrib.get("xyz"), 3),
                origin_rpy=_vec(None if origin is None else origin.attrib.get("rpy"), 3),
                axis=_vec(None if axis is None else axis.attrib.get("xyz"), 3, np.asarray([0.0, 0.0, 1.0])),
                lower=None if limit is None or "lower" not in limit.attrib else float(limit.attrib["lower"]),
                upper=None if limit is None or "upper" not in limit.attrib else float(limit.attrib["upper"]),
            )

        joints_by_name = {
            str(e.attrib["name"]): parse_joint(e)
            for e in root.findall("joint")
            if "name" in e.attrib
        }
        try:
            mount = joints_by_name[mount_joint]
            joints = tuple(joints_by_name[name] for name in arm_joint_names)
        except KeyError as exc:
            raise ValueError("required native arm joint is absent: %s" % exc) from exc
        if mount.child != joints[0].parent:
            raise ValueError("mount child does not connect to the first arm joint")
        for previous, current in zip(joints[:-1], joints[1:]):
            if previous.child != current.parent:
                raise ValueError("native arm joint chain is disconnected")
        return cls(
            mount=mount,
            joints=joints,
            tcp_parent_link=tcp_parent_link,
            tcp_offset_xyz=np.asarray(tcp_offset_xyz, dtype=np.float64),
        )

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(j.name for j in self.joints)

    @property
    def joint_limits(self) -> np.ndarray:
        if any(j.lower is None or j.upper is None for j in self.joints):
            raise ValueError("native arm joint limits are incomplete")
        return np.asarray([[j.lower, j.upper] for j in self.joints], dtype=np.float64)

    def fk(self, q: np.ndarray) -> dict[str, np.ndarray]:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (len(self.joints),):
            raise ValueError("native arm q must have shape [6]")
        transforms = {self.mount.parent: np.eye(4, dtype=np.float64)}
        current = self.mount.origin_transform
        transforms[self.mount.child] = current.copy()
        for angle, joint in zip(q, self.joints):
            current = current @ joint.origin_transform
            if joint.joint_type in ("revolute", "continuous"):
                current = current @ homogeneous(axis_angle_to_matrix(joint.axis, float(angle)), np.zeros(3))
            elif joint.joint_type == "prismatic":
                current = current @ homogeneous(np.eye(3), joint.axis * float(angle))
            transforms[joint.child] = current.copy()
        link6 = transforms[self.tcp_parent_link]
        tcp = link6 @ homogeneous(np.eye(3), self.tcp_offset_xyz)
        transforms["__tcp__"] = tcp
        transforms["__link6__"] = link6
        return transforms

    def tcp_pose(self, q: np.ndarray) -> np.ndarray:
        return _as_pose(self.fk(q)["__tcp__"])

    def link6_pose(self, q: np.ndarray) -> np.ndarray:
        return _as_pose(self.fk(q)["__link6__"])

    def finite_difference_jacobian(self, q: np.ndarray, step: float = 1e-6) -> np.ndarray:
        """Return position plus rotation-vector finite differences (6x6)."""
        q = np.asarray(q, dtype=np.float64)
        base = self.tcp_pose(q)
        out = np.zeros((6, len(q)), dtype=np.float64)
        for i in range(len(q)):
            qp, qm = q.copy(), q.copy()
            qp[i] += step
            qm[i] -= step
            pp, pm = self.tcp_pose(qp), self.tcp_pose(qm)
            out[:3, i] = (pp[:3, 3] - pm[:3, 3]) / (2.0 * step)
            rel = pp[:3, :3] @ pm[:3, :3].T
            skew = (rel - rel.T) / 2.0
            out[3:, i] = np.asarray([skew[2, 1], skew[0, 2], skew[1, 0]]) / (2.0 * step)
        del base
        return out
