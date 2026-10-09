"""CPU-only task MJCF overlay and synchronized RGB boundary.

Private inputs stay private. This module returns an overlay; only the single
runtime owner may load it, step physics, calibrate contacts, or render mjData.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from typing import Any, Mapping

import numpy as np

from .mapping import BASE_JOINT, CAMERAS, FINGER_JOINTS, GRIPPER_ACTUATOR, NATIVE_JOINTS, OBJECT_JOINT

GRIPPER_MISSING = ("deployment hardware model/revision identity", "verified operating speed and force limits",
                   "verified grasp-pad contact geometry, material and friction")


def _xml(path: Path) -> ET.Element:
    # ponytail: remove invalid delivered XML comments only in the read copy.
    return ET.fromstring(re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.S))


def _numbers(text: str, size: int) -> tuple[float, ...]:
    values = tuple(float(x) for x in text.split())
    if len(values) != size or not all(math.isfinite(x) for x in values):
        raise ValueError(f"expected {size} finite numbers")
    return values


def _text(values: Any) -> str:
    return " ".join(format(float(x), ".17g") for x in values)


def gripper_spec_from_urdf(path: str | Path) -> dict[str, Any]:
    """Extract delivered model declarations without claiming hardware verification."""
    path = Path(path)
    root = _xml(path)
    joints = {e.get("name"): e for e in root.findall("joint")}
    links = {e.get("name"): e for e in root.findall("link")}
    result: dict[str, Any] = {"status": "PROVISIONAL_GRIPPER", "missing": list(GRIPPER_MISSING),
                              "source_path": str(path.resolve()), "hardware_source": None,
                              "collision_geometry_status": "DELIVERED_PLANNING_SIMPLE_BOXES_UNVERIFIED_PAD_CONTACT",
                              "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "source_kind": "DELIVERED_PLANNING_URDF_MODEL_DECLARATIONS", "fingers": []}
    tcp = joints["gripper_center_fixed"]
    if tcp.find("parent").get("link") != "link6":
        raise ValueError("gripper_center must be fixed to link6")
    origin = tcp.find("origin")
    if _numbers(origin.get("rpy", "0 0 0"), 3) != (0., 0., 0.):
        raise ValueError("rotated TCP requires explicit integration review")
    result["tcp_xyz"] = _numbers(origin.get("xyz", "0 0 0"), 3)
    for side, link_name, joint_name, pad_name in (("left", "link7", "gripper_joint", "left_pad_joint"),
                                                ("right", "link8", "joint8", "right_pad_joint")):
        joint, link = joints[joint_name], links[link_name]
        if joint.get("type") != "prismatic" or joint.find("parent").get("link") != "link6":
            raise ValueError("expected two delivered slide fingers attached to link6")
        limit, inertia = joint.find("limit"), link.find("inertial")
        i = inertia.find("inertia")
        finger = {"body": f"x5_{link_name}", "side": side,
                  "origin": _numbers(joint.find("origin").get("xyz"), 3),
                  "axis": _numbers(joint.find("axis").get("xyz"), 3),
                  "range": (float(limit.get("lower")), float(limit.get("upper"))),
                  "force_model_N": float(limit.get("effort")), "speed_model_m_s": float(limit.get("velocity")),
                  "mass": float(inertia.find("mass").get("value")),
                  "com": _numbers(inertia.find("origin").get("xyz"), 3),
                  "fullinertia": tuple(float(i.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")),
                  "pad_xyz": _numbers(joints[pad_name].find("origin").get("xyz"), 3), "collisions": []}
        for collision in link.findall("collision"):
            geom, transform = collision.find("geometry"), collision.find("origin")
            if geom.find("box") is None:
                raise ValueError("CPU overlay currently accepts delivered simple box finger collisions")
            rpy = _numbers(transform.get("rpy", "0 0 0"), 3) if transform is not None else (0., 0., 0.)
            if rpy != (0., 0., 0.):
                raise ValueError("rotated finger collider requires explicit geometry conversion")
            finger["collisions"].append({"xyz": _numbers(transform.get("xyz", "0 0 0"), 3) if transform is not None else (0., 0., 0.),
                                         "halfsize": tuple(x / 2 for x in _numbers(geom.find("box").get("size"), 3))})
        if not finger["collisions"]:
            raise ValueError("missing delivered finger collision geometry")
        result["fingers"].append(finger)
    mimic = joints["joint8"].find("mimic")
    if mimic is None or mimic.get("joint") != "gripper_joint" or float(mimic.get("multiplier", "1")) != 1 or float(mimic.get("offset", "0")) != 0:
        raise ValueError("expected equal-displacement mechanical finger coupling")
    return result


def provisional_cameras() -> list[dict[str, Any]]:
    """Explicit sim design only; actual extrinsics/FOV still need calibration."""
    return [{"name": CAMERAS[0], "parent": "base_link", "pos": (.25, 0., .15), "xyaxes": (0., -1., 0., .5, 0., 1.), "fovy": 60., "status": "SIM_PROVISIONAL_CAMERA"},
            {"name": CAMERAS[1], "parent": "x5_link6", "pos": (.045, 0., .09), "xyaxes": (0., -1., 0., -.5, 0., 1.), "fovy": 60., "status": "SIM_PROVISIONAL_CAMERA"}]


def _joint_widths(root: ET.Element) -> list[tuple[str, int]]:
    return [(e.get("name"), 7 if e.tag == "freejoint" or e.get("type") == "free" else 4 if e.get("type") == "ball" else 1)
            for e in root.find("worldbody").iter() if e.tag in ("joint", "freejoint")]


def build_task_overlay(robot_xml: str | Path, scene_xml: str | Path, *, layout: Mapping[str, Any],
                       contact_spec: Mapping[str, Any], gripper_spec: Mapping[str, Any],
                       cameras: list[dict[str, Any]] | None = None) -> tuple[str, dict[str, Any]]:
    """Retain original dynamics and masses; append task physics and sensors.

    layout requires table_center_xyz, table_halfsize, cube_xy, cube_yaw,
    zone_xy and selection_source. contact_spec requires friction[3] and
    source; these are frozen task declarations, never success-rate tuning.
    """
    robot_path, scene_path = Path(robot_xml), Path(scene_xml)
    root, scene = _xml(robot_path), _xml(scene_path)
    if root.findall("include"):
        raise ValueError("robot XML includes require an explicit resolved private model")
    includes = scene.findall("include")
    if len(includes) != 1 or (scene_path.parent / includes[0].get("file")).resolve() != robot_path.resolve():
        raise ValueError("scene must include exactly the selected native robot XML")
    bodies = {b.get("name"): b for b in root.iter("body")}
    if not layout.get("selection_source") or not contact_spec.get("source"):
        raise ValueError("layout reach/FOV selection and contact declaration sources required")
    table = _numbers(_text(layout["table_center_xyz"]), 3)
    half = _numbers(_text(layout["table_halfsize"]), 3)
    cube_xy, zone_xy = (_numbers(_text(layout[key]), 2) for key in ("cube_xy", "zone_xy"))
    yaw = float(layout["cube_yaw"])
    friction = _numbers(_text(contact_spec["friction"]), 3)
    if min(half) <= 0 or min(friction) < 0 or not math.isfinite(yaw) or math.dist(cube_xy, zone_xy) < .15:
        raise ValueError("invalid table, friction, yaw or source-goal separation <0.15m")
    cube_radius = .02 * (abs(math.sin(yaw)) + abs(math.cos(yaw)))
    for xy, radius in ((cube_xy, cube_radius), (zone_xy, .08)):
        if any(abs(xy[i] - table[i]) + radius > half[i] for i in (0, 1)):
            raise ValueError("cube and full target zone must lie on the declared table")
    free = bodies["base_link"].find("freejoint")
    if free is None:
        raise ValueError("native base must remain free-floating")
    free.set("name", BASE_JOINT)
    original_joints = _joint_widths(root)
    if set(n for n, _ in original_joints) != {BASE_JOINT, *NATIVE_JOINTS}:
        raise ValueError("expected exactly the native floating base and 18 joints")
    original_controls = [a.get("name") for a in root.find("actuator")]
    inertials_before = {name: ET.tostring(body.find("inertial")) for name, body in bodies.items() if body.find("inertial") is not None}
    tcp = root.find(".//site[@name='x5_ee']")
    if tcp is None or not np.allclose(_numbers(tcp.get("pos"), 3), gripper_spec["tcp_xyz"], atol=1e-10, rtol=0):
        raise ValueError("planning TCP and native x5_ee differ")
    fingers = gripper_spec["fingers"]
    if len(fingers) != 2:
        raise ValueError("exactly two finger declarations required")
    if not np.allclose(fingers[0]["axis"], -np.asarray(fingers[1]["axis"]), atol=1e-10) or not np.isclose(np.linalg.norm(fingers[0]["axis"]), 1):
        raise ValueError("finger slides must have opposing unit axes")
    if fingers[0]["range"] != fingers[1]["range"]:
        raise ValueError("coupled finger stroke ranges must agree")
    for name, spec in zip(FINGER_JOINTS, fingers):
        body, limits = bodies[spec["body"]], spec["range"]
        inertia = body.find("inertial")
        for actual, expected in ((_numbers(body.get("pos"), 3), spec["origin"]),
                                 (_numbers(inertia.get("pos"), 3), spec["com"]),
                                 (_numbers(inertia.get("fullinertia"), 6), spec["fullinertia"])):
            if not np.allclose(actual, expected, atol=1e-10, rtol=0):
                raise ValueError("native finger geometry/inertia differs from delivered URDF")
        if not math.isclose(float(inertia.get("mass")), spec["mass"], abs_tol=1e-10):
            raise ValueError("native finger mass differs; additional mass is forbidden")
        scalars = (*limits, spec["mass"], spec["force_model_N"], spec["speed_model_m_s"])
        if not all(math.isfinite(float(x)) for x in scalars) or body.find("joint") is not None or limits[0] != 0 or min(limits[1], spec["mass"], spec["force_model_N"], spec["speed_model_m_s"]) <= 0:
            raise ValueError("invalid fixed finger or declared physical limits")
        ET.SubElement(body, "joint", name=name, type="slide", axis=_text(spec["axis"]), range=_text(limits), limited="true", damping="0", armature="0", frictionloss="0")
        for index, geom in enumerate(spec["collisions"]):
            size, xyz = _numbers(_text(geom["halfsize"]), 3), _numbers(_text(geom["xyz"]), 3)
            if min(size) <= 0:
                raise ValueError("finger collision halfsize must be positive")
            ET.SubElement(body, "geom", name=f"{name}_collision_{index}", type="box", size=_text(size),
                          pos=_text(xyz), mass="0", contype="1", conaffinity="1", condim="3", friction=_text(friction), group="3")
        ET.SubElement(body, "site", name=f"{name}_pad", pos=_text(spec["pad_xyz"]), size="0.001")
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    for name in FINGER_JOINTS:
        ET.SubElement(sensor, "jointpos", name=f"{name}_pos", joint=name)
        ET.SubElement(sensor, "jointvel", name=f"{name}_vel", joint=name)
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    ET.SubElement(equality, "joint", name="pi0288_finger_coupling", joint1=FINGER_JOINTS[1], joint2=FINGER_JOINTS[0], polycoef="0 1 0 0 0")
    # One driver on the master; the URDF mimic finger is passive, not a second force source.
    force = float(fingers[0]["force_model_N"])
    ET.SubElement(root.find("actuator"), "motor", name=GRIPPER_ACTUATOR, joint=FINGER_JOINTS[0], gear="1", ctrlrange=_text((-force, force)), ctrllimited="true", forcerange=_text((-force, force)), forcelimited="true")
    compiler = root.find("compiler")
    if compiler is not None:
        for key in ("meshdir", "texturedir"):
            if compiler.get(key):
                compiler.set(key, str((robot_path.parent / compiler.get(key)).resolve()))
    for element in scene:
        if element.tag == "include":
            continue
        existing = root.find(element.tag)
        if element.tag in ("worldbody", "asset") and existing is not None:
            existing.extend(copy.deepcopy(list(element)))
        elif existing is None:
            root.append(copy.deepcopy(element))
        else:
            raise ValueError(f"ambiguous native and scene {element.tag} settings")
    world = root.find("worldbody")
    top = table[2] + half[2]
    ET.SubElement(world, "geom", name="pi0288_table", type="box", pos=_text(table), size=_text(half), friction=_text(friction), condim="3", rgba="0.55 0.45 0.35 1")
    cube_pos = (*cube_xy, top + .02)
    cube_quat = (math.cos(yaw / 2), 0., 0., math.sin(yaw / 2))
    cube = ET.SubElement(world, "body", name="pi0288_cube", pos=_text(cube_pos), quat=_text(cube_quat))
    ET.SubElement(cube, "freejoint", name=OBJECT_JOINT)
    ET.SubElement(cube, "geom", name="pi0288_cube_geom", type="box", size=".02 .02 .02", mass=".05", friction=_text(friction), condim="3", rgba="1 0 0 1")
    ET.SubElement(world, "site", name="pi0288_zone", type="box", pos=_text((*zone_xy, top + .0005)), size=".08 .08 .0005", rgba="0 0 1 .65")
    cameras = copy.deepcopy(cameras if cameras is not None else provisional_cameras())
    if {c["name"] for c in cameras} != set(CAMERAS) or len(cameras) != 2:
        raise ValueError("exactly the named body/wrist cameras required")
    for camera in cameras:
        if camera.get("status") not in ("SIM_PROVISIONAL_CAMERA", "CALIBRATED_DEPLOYMENT_CAMERA"):
            raise ValueError("camera provenance status required")
        xyaxes = _numbers(_text(camera["xyaxes"]), 6)
        fovy = float(camera["fovy"])
        if not math.isfinite(fovy) or not 0 < fovy < 180 or np.linalg.norm(np.cross(xyaxes[:3], xyaxes[3:])) < 1e-10:
            raise ValueError("camera FOV and axes must be finite and nondegenerate")
        ET.SubElement(bodies[camera["parent"]], "camera", name=camera["name"], pos=_text(_numbers(_text(camera["pos"]), 3)), xyaxes=_text(xyaxes), fovy=str(fovy))
    defaults = {FINGER_JOINTS[0]: [0.], FINGER_JOINTS[1]: [0.], OBJECT_JOINT: [*cube_pos, *cube_quat]}
    new_joints = _joint_widths(root)
    for key in root.findall("keyframe/key"):
        if key.get("qpos"):
            old = _numbers(key.get("qpos"), sum(width for _, width in original_joints))
            offset, values = 0, {}
            for name, width in original_joints:
                values[name] = old[offset:offset + width]
                offset += width
            values.update(defaults)
            key.set("qpos", _text([value for name, _ in new_joints for value in values[name]]))
        if key.get("ctrl"):
            key.set("ctrl", _text((*_numbers(key.get("ctrl"), len(original_controls)), 0.)))
        if key.get("qvel"):
            raise ValueError("explicit native keyframe qvel needs named expansion")
    inertials_after = {name: ET.tostring(body.find("inertial")) for name, body in bodies.items() if body.find("inertial") is not None}
    if inertials_before != inertials_after:
        raise ValueError("native inertial change is forbidden")
    xml = ET.tostring(root, encoding="unicode")
    eligible = gripper_spec.get("status") == "HARDWARE_SPEC_VERIFIED" and not gripper_spec.get("missing") and bool(gripper_spec.get("hardware_source"))
    manifest = {"schema": "PI0288_TASK_SCENE_V1", "task": "GO2_MJ_PLACE_V1", "status": "CPU_OVERLAY_PREPARED_NOT_EXECUTED",
                "gripper_status": gripper_spec["status"], "formal_data_gate": "NOT_YET_QUALIFIED" if eligible else "BLOCKED_GRIPPER_SPEC",
                "original_inertials_unchanged": True, "additional_tool_mass_kg": 0., "native_control_count": 18,
                "gripper_actuator": GRIPPER_ACTUATOR, "object_freejoint": OBJECT_JOINT, "base_freejoint": BASE_JOINT,
                "cube_edge_m": .04, "cube_mass_kg": .05, "cube_mass_source": "SIM_TASK_DESIGN",
                "zone_side_m": .16, "layout": dict(layout), "contact": dict(contact_spec), "gripper": copy.deepcopy(dict(gripper_spec)),
                "cameras": cameras, "rgb_min_shape": [256, 256, 3], "rgb_rate_Hz": 20, "max_skew_s": .025,
                "native_xml_sha256": hashlib.sha256(robot_path.read_bytes()).hexdigest(), "scene_input_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
                "overlay_sha256": hashlib.sha256(xml.encode()).hexdigest(), "private_asset_paths_in_xml": True}
    return xml, manifest


def independent_gripper_force(*, q_m: float, velocity_m_s: float, requested_m: float, previous_setpoint_m: float,
                              dt_s: float, stroke_m: float, speed_m_s: float, force_N: float,
                              stiffness_N_m: float, calibration_id: str) -> dict[str, Any]:
    """Bounded master-finger command; stiffness is explicit calibrated driver input."""
    values = (q_m, velocity_m_s, requested_m, previous_setpoint_m, dt_s, stroke_m, speed_m_s, force_N, stiffness_N_m)
    if not calibration_id or not all(math.isfinite(v) for v in values) or min(dt_s, stroke_m, speed_m_s, force_N, stiffness_N_m) <= 0:
        raise ValueError("finite state, positive driver limits and calibration identity required")
    if not 0 <= previous_setpoint_m <= stroke_m:
        raise ValueError("previous setpoint is outside declared stroke")
    target = float(np.clip(requested_m, 0, stroke_m))
    accepted = float(np.clip(target, previous_setpoint_m - speed_m_s * dt_s, previous_setpoint_m + speed_m_s * dt_s))
    force = float(np.clip(stiffness_N_m * (accepted - q_m), -force_N, force_N))
    speed_hold = abs(velocity_m_s) >= speed_m_s and force * velocity_m_s > 0
    return {"requested_m": requested_m, "accepted_m": accepted, "force_N": 0. if speed_hold else force,
            "clamped": accepted != requested_m, "speed_hold": speed_hold, "calibration_id": calibration_id}


def validate_rgb_pair(*, body_rgb: Any, wrist_rgb: Any, snapshot_id: str, body_snapshot_id: str,
                      wrist_snapshot_id: str, state_sim_s: float, body_sim_s: float, wrist_sim_s: float,
                      episode_id: str, frame_index: int, ros_s: float, wall_monotonic_s: float,
                      dropped_pairs: int = 0) -> dict[str, Any]:
    """Validate actual renderer output from one owner-controlled state snapshot."""
    times = (state_sim_s, body_sim_s, wrist_sim_s, ros_s, wall_monotonic_s)
    if not episode_id or not snapshot_id or snapshot_id != body_snapshot_id or snapshot_id != wrist_snapshot_id:
        raise ValueError("state and both views must identify the same episode snapshot")
    if not all(math.isfinite(t) and t >= 0 for t in times) or frame_index < 0 or dropped_pairs < 0:
        raise ValueError("invalid raw clock or frame/drop counters")
    if max(times[:3]) - min(times[:3]) > .025 or body_sim_s > state_sim_s or wrist_sim_s > state_sim_s:
        raise ValueError("future RGB or view/state skew above 0.025s")
    result = {"episode_id": episode_id, "snapshot_id": snapshot_id, "frame_index": frame_index,
              "state_sim_s": state_sim_s, "body_sim_s": body_sim_s, "wrist_sim_s": wrist_sim_s,
              "ros_s": ros_s, "wall_monotonic_s": wall_monotonic_s, "dropped_pairs": dropped_pairs,
              "body_age_s": state_sim_s - body_sim_s, "wrist_age_s": state_sim_s - wrist_sim_s}
    for name, pixels in (("body_rgb", body_rgb), ("wrist_rgb", wrist_rgb)):
        image = np.asarray(pixels)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 256:
            raise ValueError("actual uint8 raw RGB at least 256x256 required")
        result[name] = image.copy()
        result[name].setflags(write=False)
    return result


def camera_calibration(model: Any, data: Any, camera_ids: Mapping[str, int], *, width: int = 256, height: int = 256) -> dict[str, Any]:
    """Read actual compiled camera transforms at the same owner-held snapshot."""
    if width < 256 or height < 256 or set(camera_ids) != set(CAMERAS):
        raise ValueError("named dual cameras and raw resolution >=256 required")
    result = {}
    for name, cid in camera_ids.items():
        fovy = float(model.cam_fovy[cid])
        if not math.isfinite(fovy) or not 0 < fovy < 180:
            raise ValueError("finite perspective camera FOV required")
        focal = .5 * height / math.tan(math.radians(fovy) / 2)
        position, rotation = np.asarray(data.cam_xpos[cid], dtype=float), np.asarray(data.cam_xmat[cid], dtype=float).reshape(3, 3)
        if not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError("finite rendered camera world transform required")
        transform = np.eye(4); transform[:3, :3] = rotation; transform[:3, 3] = position
        optical = transform @ np.diag([1., -1., -1., 1.])
        result[name] = {"width": width, "height": height, "fovy_deg": fovy,
                        "K": [[focal, 0., width / 2], [0., focal, height / 2], [0., 0., 1.]],
                        "T_world_camera_mujoco": transform.tolist(), "T_world_optical": optical.tolist(), "parent_body_id": int(model.cam_bodyid[cid]),
                        "parent_pos": np.asarray(model.cam_pos[cid]).tolist(), "parent_quaternion_wxyz": np.asarray(model.cam_quat[cid]).tolist(),
                        "camera_axes": "MuJoCo local +X right,+Y up,-Z forward; K uses image +Y down"}
    return result


def grasp_contact_evidence(*, contacts: list[Mapping[str, Any]], left_geom_ids: set[int], right_geom_ids: set[int],
                           object_geom_ids: set[int], object_in_tcp_xyz: Any, previous_object_in_tcp_xyz: Any,
                           dt_s: float, max_relative_speed_m_s: float) -> dict[str, Any]:
    """Two opposing force-bearing contacts plus measured object/tool relative motion.

    The caller supplies real contact normals pointing geom1->geom2 and real
    positive normal forces. This does not certify pad geometry or accumulate
    the protocol's 0.30s grasp/lift intervals; the scorer handles persistence.
    """
    if not left_geom_ids or not right_geom_ids or not object_geom_ids or left_geom_ids & right_geom_ids or (left_geom_ids | right_geom_ids) & object_geom_ids:
        raise ValueError("distinct nonempty physical pad/object geometry IDs required")
    if not math.isfinite(dt_s) or dt_s <= 0 or not math.isfinite(max_relative_speed_m_s) or max_relative_speed_m_s < 0:
        raise ValueError("explicit positive dt and frozen relative-motion threshold required")
    now, previous = np.asarray(object_in_tcp_xyz, dtype=float), np.asarray(previous_object_in_tcp_xyz, dtype=float)
    if now.shape != (3,) or previous.shape != (3,) or not np.isfinite(now).all() or not np.isfinite(previous).all():
        raise ValueError("finite measured object-relative-TCP positions required")
    pads: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    raw = []
    for contact in contacts:
        a, b, force = int(contact["geom1"]), int(contact["geom2"]), float(contact["normal_force_N"])
        normal = np.asarray(contact["normal_geom1_to_geom2"], dtype=float)
        if normal.shape != (3,) or not np.isfinite(normal).all() or not math.isfinite(force) or force < 0 or not np.isclose(np.linalg.norm(normal), 1, atol=1e-5):
            raise ValueError("real contact must have unit normal and nonnegative normal force")
        for side, ids in (("left", left_geom_ids), ("right", right_geom_ids)):
            if a in ids and b in object_geom_ids:
                inward = normal
            elif b in ids and a in object_geom_ids:
                inward = -normal
            else:
                continue
            raw.append({"side": side, "geom1": a, "geom2": b, "normal_force_N": force, "pad_to_object_normal": inward.tolist()})
            if force > 0:
                pads[side].append(inward)
    dots = [float(np.dot(a, b)) for a in pads["left"] for b in pads["right"]]
    speed = float(np.linalg.norm(now - previous) / dt_s)
    return {"opposing_pad_contact": any(dot < 0 for dot in dots), "relative_motion_valid": speed <= max_relative_speed_m_s,
            "relative_speed_m_s": speed, "max_relative_speed_m_s": max_relative_speed_m_s, "normal_dot_products": dots,
            "left_effective_contacts": len(pads["left"]), "right_effective_contacts": len(pads["right"]), "raw_pad_object_contacts": raw}
