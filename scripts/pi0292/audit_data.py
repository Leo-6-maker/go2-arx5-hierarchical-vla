"""Independent ARX5 raw-data audit. Saved-state forward kinematics only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
import mujoco
from PIL import Image

from native_dataset import DOMAIN, POLICY_KEYS, matrix_from10, pose10, read, sha, validate_samples, write

POINTS = np.asarray(list(itertools.product((-.02, .02), repeat=3)))
OFFSETS = [(0., 0., .0605), (.02, 0., .04), (-.264, 0., 0.), (.245, 0., .056), (.06775, 0., .0845), (.02895, 0., -.0845)]
AXES = [(0., 0., 1.), (0., 1., 0.), (0., -1., 0.), (0., -1., 0.), (0., 0., -1.), (1., 0., 0.)]


def near(left, right, tolerance=1e-8):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        raise ValueError("Shape mismatch in independently reconstructed sensor")
    error = float(np.max(np.abs(left - right)))
    if not np.isfinite(error) or error > tolerance:
        raise ValueError(f"Independent reconstruction error {error} exceeds {tolerance}")
    return error


def independent_fk(joints):
    result = np.eye(4)
    result[:3, 3] = [.05, 0., .50]
    for position, axis, angle in zip(OFFSETS, AXES, joints):
        result[:3, 3] += result[:3, :3] @ position
        x, y, z = axis
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        result[:3, :3] = result[:3, :3] @ (np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew)
    result[:3, 3] += result[:3, :3] @ [.1424, 0., .0001057]
    return result


def sustained(flags):
    # At20Hz, seven uninterrupted real samples span the required300ms.
    run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        if run >= 7:
            return True
    return False


def quat_xyzw_matrices(quaternions):
    quaternions = np.asarray(quaternions, dtype=float)
    quaternions = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(quaternions, -1, 0)
    return np.stack((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
                     2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
                     2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)), axis=-1).reshape(quaternions.shape[:-1] + (3, 3))


def independent_score(physics, card, complete, table_height):
    xyz = physics["object_pose"][:, :3, 3]
    grasp = np.logical_and.reduce([physics[key] for key in ("object_present", "contact_valid", "gripper_state_valid", "native_grasp", "left_object_contact", "right_object_contact")])
    initial = xyz[0]
    safe = bool(physics["safety_valid"].all() and not any(physics[key].any() for key in ("native_abort", "illegal_contact", "joint_limit_penetration", "reset", "takeover")))
    if card["mode"] == "empty_grasp":
        return dict(persistent_success=False, sustained_grasp=sustained(grasp), contact_negative=True)
    corners = xyz[:, None, :] + np.einsum("nij,kj->nki", physics["object_pose"][:, :3, :3], POINTS)
    predicates = {
        "zone_corners": (np.abs(corners[..., :2] - np.asarray(card["goal_xy"])) <= .07).all(axis=(1, 2)),
        "table_height": np.abs(xyz[:, 2] - (table_height + .02)) <= .005,
        "linear_speed": np.linalg.norm(physics["object_linear_velocity"], axis=1) <= .02,
        "angular_speed": np.linalg.norm(physics["object_angular_velocity"], axis=1) <= .10,
        "released": ~grasp & ~physics["native_grasp"],
        "tcp_clear": np.linalg.norm(physics["tcp_world"][:, :3, 3] - xyz, axis=1) >= .08,
        "object_and_contacts_valid": physics["object_present"] & physics["contact_valid"] & physics["gripper_state_valid"],
    }
    hold = np.logical_and.reduce(list(predicates.values()))
    hold_indices = [index for index, tick in enumerate(physics["time_s"]) if 28 - 1e-8 <= tick <= 30 + 1e-8]
    history = dict(sustained_grasp=sustained(grasp), sustained_lift=sustained(grasp & (xyz[:, 2] - initial[2] >= .02)),
                   horizontal_carry=bool(np.any(grasp & (np.linalg.norm(xyz[:, :2] - initial[:2], axis=1) >= .10))))
    passed = int(hold[hold_indices].sum())
    return dict(history=history, hold_observed=len(hold_indices), hold_pass=passed,
                hold_subpredicates={key: int(value[hold_indices].sum()) for key, value in predicates.items()},
                safety_valid=safe, persistent_success=bool(complete and safe and all(history.values()) and len(hold_indices) == 41 and passed == 41))


def negative_gates(samples):
    if len(samples) != 641 or not all(row["supervision"]["valid"] for row in samples[:640]):
        return {"status": "NOT_RUN_ON_INCOMPLETE_OR_CONTACT_NEGATIVE; use complete normal raw fixture"}
    checks = {}
    for fixture in ("object_pose", "teacher_stage", "success", "future_history", "sensor_time"):
        changed = copy.deepcopy(samples)
        if fixture in ("object_pose", "teacher_stage", "success"):
            changed[0]["policy_obs"][fixture] = True
        elif fixture == "future_history":
            changed[0]["causal_history"]["previous_committed"] = 1
        else:
            changed[0]["policy_obs"]["wrist_rgb"]["timestamp_s"] += .05
        try:
            validate_samples(changed)
        except ValueError:
            checks[fixture] = "PASS_REJECTED_IN_MEMORY_MUTATION_OF_ACTUAL_RAW"
        else:
            raise ValueError("Negative failed to reject " + fixture)
    return checks


def audit_episode(root, identifier, engine_class, parameters):
    folder = Path(root) / "episodes" / identifier
    result, card = read(folder / "RESULT.json"), read(folder / "CARD.json")
    if result["domain"] != DOMAIN or card["domain"] != DOMAIN:
        raise ValueError("Wrong data domain")
    files = read(folder / "FILE_MANIFEST.json")
    for row in files:
        path = folder / row["path"]
        if not path.resolve().is_relative_to(folder.resolve()) or not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("Raw manifest path/hash/size mismatch: " + row["path"])
    samples, references = read(folder / "samples.json"), read(folder / "references.json")
    if not samples:
        return dict(id=identifier, status="PRESERVED_EMPTY_NOT_ADMITTED", complete=False, paired_samples=0, physics_steps=result["physics_steps"], manifest_sha256=sha(folder / "FILE_MANIFEST.json"))
    with np.load(folder / "physics.npz", allow_pickle=False) as archive:
        physics = {key: archive[key].copy() for key in archive.files}
    with np.load(folder / "highrate.npz", allow_pickle=False) as archive:
        high = archive["rows"].copy()
        nq, nv = int(archive["nq"]), int(archive["nv"])
    n = len(samples)
    if n != result["paired_samples"] or n != len(physics["time_s"]) or len(high) != result["physics_steps"]:
        raise ValueError("Raw attempt counts disagree")
    near(physics["time_s"], np.arange(n) / 20)
    near(high[:, 0], np.arange(1, len(high) + 1) * .002)
    if not np.isfinite(high).all():
        raise ValueError("Nonfinite high-rate dynamics")
    complete = bool(result["status"] == "VALID_COMPLETE" and n == 641 and len(high) == 16000)
    if card["mode"] == "normal" and complete:
        validate_samples(samples)
    xml = read(folder / "scene.xml.json")
    if hashlib.sha256(xml["xml"].encode()).hexdigest() != xml["sha256"]:
        raise ValueError("Saved scene XML changed")
    # Initial IK is forward-only; all other snapshots come from actual saved states.
    initial = engine_class(card, card["mode"])
    if initial.steps != 0 or initial.data.time != 0 or initial.xml != xml["xml"]:
        raise ValueError("Initial state or saved scene does not match bound physical version")
    model = mujoco.MjModel.from_xml_string(xml["xml"])
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], data.ctrl[:] = initial.data.qpos, initial.data.qvel, initial.data.ctrl
    initial.close()
    if (model.nq, model.nv, model.nu, model.neq, model.nmocap) != (15, 14, 8, 0, 0) or (nq, nv) != (model.nq, model.nv):
        raise ValueError("Expected six-arm/two-finger/free-object domain; no weld/mocap")
    controls = high[:, 1 + nq + nv:1 + nq + nv + 8]
    wrench = high[:, 1 + nq + nv + 8:-1]
    if wrench.shape[1] != 6 or np.any(wrench != 0) or np.any(controls < model.actuator_ctrlrange[:, 0] - 1e-10) or np.any(controls > model.actuator_ctrlrange[:, 1] + 1e-10):
        raise ValueError("Nonzero external wrench or unbounded joint motor")
    qpos, qvel = high[:, 1:1 + nq], high[:, 1 + nq:1 + nq + nv]
    names = [f"x5_joint{index}" for index in range(1, 7)] + ["left_slide", "right_slide"]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
    qi = model.jnt_qposadr[joint_ids]
    vi = model.jnt_dofadr[joint_ids]
    if complete:
        if np.any(high[:, -1]) or np.any(np.abs(qvel[:, vi[:6]]) > np.asarray(parameters["arm_speed_stop_rad_s"])) or np.max(np.abs(qvel[:, vi[6:]])) > parameters["finger_speed_stop_m_s"]:
            raise ValueError("High-rate safety evidence failed")
        for index, jid in enumerate(joint_ids):
            if np.any(qpos[:, qi[index]] < model.jnt_range[jid, 0] - .001) or np.any(qpos[:, qi[index]] > model.jnt_range[jid, 1] + .001):
                raise ValueError("High-rate joint range failed")
    nameid = lambda kind, name: mujoco.mj_name2id(model, kind, name)
    cube = nameid(mujoco.mjtObj.mjOBJ_BODY, "cube")
    tcp_site = nameid(mujoco.mjtObj.mjOBJ_SITE, "x5_ee")
    cube_geom = nameid(mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    pads = [nameid(mujoco.mjtObj.mjOBJ_GEOM, name) for name in ("left_pad", "right_pad")]
    table, floor = [nameid(mujoco.mjtObj.mjOBJ_GEOM, name) for name in ("table", "floor")]
    max_sensor, max_force, max_camera, max_projection, max_fk, max_codec = 0., 0., 0., 0., 0., 0.
    contact_counts = [0, 0]
    stage_pixels = {}
    wrist_pixels, body_pixels = [], []
    wrist_rig, body_rig = None, None
    for index, sample in enumerate(samples):
        tick = index / 20
        snapshot = f"{identifier}:{index * 25}"
        obs, privileged = sample["policy_obs"], sample["privileged"]
        if set(obs) != POLICY_KEYS or not obs["valid"] or obs["instruction"] != card["instruction"] or sample["snapshot_id"] != snapshot:
            raise ValueError("Illegal policy boundary or observation identity")
        near(obs["timestamp_s"], tick)
        expected_history = dict(episode_id=identifier, segment_id=0, previous_samples=index, previous_committed=index, reset=False, abort=False, takeover=False)
        if sample["causal_history"] != expected_history:
            raise ValueError("Future/unordered/reset causal history")
        if index:
            row = high[index * 25 - 1]
            data.qpos[:] = row[1:1 + nq]
            data.qvel[:] = row[1 + nq:1 + nq + nv]
            data.ctrl[:] = row[1 + nq + nv:1 + nq + nv + 8]
        data.time = tick
        data.xfrc_applied[:] = 0.
        data.qfrc_applied[:] = 0.
        mujoco.mj_forward(model, data)
        tcp, obj = np.eye(4), np.eye(4)
        tcp[:3, :3], tcp[:3, 3] = data.site_xmat[tcp_site].reshape(3, 3), data.site_xpos[tcp_site]
        obj[:3, :3], obj[:3, 3] = data.xmat[cube].reshape(3, 3), data.xpos[cube]
        max_sensor = max(max_sensor, near(tcp, obs["actual_tcp"]), near(tcp, physics["tcp_world"][index]), near(obj, physics["object_pose"][index]), near(obj, privileged["object_pose"]))
        max_fk = max(max_fk, near(independent_fk(data.qpos[qi[:6]]), tcp))
        for values, key in ((data.qpos[qi[:6]], "actual_joint_q"), (data.qvel[vi[:6]], "actual_joint_qd"),
                            (data.qpos[qi[6:]], "actual_finger_q"), (data.qvel[vi[6:]], "actual_finger_qd"), (data.actuator_force, "actuator_force")):
            max_sensor = max(max_sensor, near(values, sample["execution"][key]))
        max_sensor = max(max_sensor, near(np.mean(data.qpos[qi[6:]]) / .044, obs["actual_gripper"]), near(np.mean(data.qvel[vi[6:]]) / .044, obs["actual_gripper_speed"]))
        geometric_gap = np.dot(data.geom_xpos[pads[0]] - data.geom_xpos[pads[1]], tcp[:3, 1]) - .006
        near(geometric_gap, parameters["closed_inner_gap"] + data.qpos[qi[6:]].sum())
        points = obj[:3, 3] + POINTS @ obj[:3, :3].T
        near(points, privileged["object_points"])
        cube_velocity, tcp_velocity = np.zeros(6), np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, cube, cube_velocity, 0)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, tcp_site, tcp_velocity, 0)
        near(cube_velocity[3:], physics["object_linear_velocity"][index])
        near(cube_velocity[:3], physics["object_angular_velocity"][index])
        normals, illegal = [[], []], False
        if len(privileged["contacts"]) != data.ncon:
            raise ValueError("Reconstructed contact count differs")
        for number, recorded in enumerate(privileged["contacts"]):
            contact = data.contact[number]
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, number, force)
            if recorded["geom0"] != contact.geom1 or recorded["geom1"] != contact.geom2:
                raise ValueError("Object-specific contact identities differ")
            near(recorded["normal"], contact.frame[:3])
            near(recorded["distance_m"], contact.dist)
            max_force = max(max_force, near(recorded["normal_force_N"], force[0], 1e-5))
            for side, pad in enumerate(pads):
                if {int(contact.geom1), int(contact.geom2)} == {pad, cube_geom} and force[0] >= parameters["minimum_contact_force_N"]:
                    normals[side].append(contact.frame[:3] if contact.geom2 == cube_geom else -contact.frame[:3])
            bodies = [int(model.geom_bodyid[geom]) for geom in (contact.geom1, contact.geom2)]
            arm = any(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body).startswith("x5_") for body in bodies)
            illegal |= bool(contact.dist < -parameters["penetration_stop_m"] or arm and (table in (contact.geom1, contact.geom2) or floor in (contact.geom1, contact.geom2)))
        left, right = map(bool, normals)
        opposing = bool(left and right and any(np.dot(a, b) <= parameters["opposing_contact_dot_max"] for a in normals[0] for b in normals[1]))
        relative = bool(np.linalg.norm(cube_velocity[3:] - tcp_velocity[3:]) <= parameters["relative_grasp_speed_max_m_s"])
        finger_valid = bool(np.isfinite(data.qpos[qi[6:]]).all() and np.all(data.qpos[qi[6:]] >= -.001) and np.all(data.qpos[qi[6:]] <= .045))
        arm_valid = bool(np.all(data.qpos[qi[:6]] >= model.jnt_range[joint_ids[:6], 0] - .001) and np.all(data.qpos[qi[:6]] <= model.jnt_range[joint_ids[:6], 1] + .001))
        grasp = bool(left and right and opposing and relative and finger_valid)
        for key, actual in dict(left_object_contact=left, right_object_contact=right, native_grasp=grasp,
                                gripper_state_valid=finger_valid, illegal_contact=illegal, joint_limit_penetration=not (finger_valid and arm_valid),
                                safety_valid=not illegal and arm_valid and not bool(high[index * 25 - 1, -1]) if index else not illegal and arm_valid).items():
            if bool(physics[key][index]) != actual:
                raise ValueError("Independent physical predicate differs: " + key)
        if opposing != privileged["opposing_contact"] or relative != privileged["relative_motion_valid"] or grasp != privileged["current_grasped"]:
            raise ValueError("Privileged grasp predicate differs from real contacts/velocity")
        contact_counts[0] += left
        contact_counts[1] += right
        for camera in ("body_rgb", "wrist_rgb"):
            image, geometry = obs[camera], sample["geometry"][camera]
            binding = read(Path(root) / "ATTEMPT_PLAN.json")["attempts"][identifier]
            if card["id"] != binding["logical_id"] or card["seed"] != binding["card"]["seed"]:
                raise ValueError("Successor card identity differs from sealed origin")
            geometry_snapshot = f'{card["id"]}:{index * 25}'
            if image["snapshot_id"] != snapshot or geometry["snapshot_id"] != geometry_snapshot or image["frame_index"] != index or image["source"] != "actual_mujoco_renderer":
                raise ValueError("Camera snapshot identity differs")
            near(image["timestamp_s"], tick)
            near(geometry["timestamp_s"], tick)
            path = folder / image["path"]
            if not path.resolve().is_relative_to(folder.resolve()) or sha(path) != image["sha256"]:
                raise ValueError("Original camera file hash differs")
            with Image.open(path) as opened:
                rgb = np.asarray(opened).copy()
            if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8 or float(rgb.std()) <= 8:
                raise ValueError("Blank/wrong real camera frame")
            red = int(np.count_nonzero((rgb[..., 0] > 80) & (rgb[..., 0].astype(float) > 1.5 * rgb[..., 1]) & (rgb[..., 0].astype(float) > 1.5 * rgb[..., 2])))
            (wrist_pixels if camera == "wrist_rgb" else body_pixels).append(red)
            if camera == "wrist_rgb":
                stage_pixels.setdefault(privileged["teacher_stage"], []).append(red)
            cid = nameid(mujoco.mjtObj.mjOBJ_CAMERA, camera)
            transform = np.eye(4)
            transform[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1., -1., -1.])
            transform[:3, 3] = data.cam_xpos[cid]
            max_camera = max(max_camera, near(transform, geometry["T_world_optical"]))
            focal = 256 / (2 * np.tan(np.deg2rad(model.cam_fovy[cid]) / 2))
            intrinsic = np.array([[focal, 0., 128], [0., focal, 128], [0., 0., 1.]])
            near(intrinsic, geometry["K"])
            camera_points = (points - transform[:3, 3]) @ transform[:3, :3]
            projected = camera_points @ intrinsic.T
            uv = projected[:, :2] / projected[:, 2, None]
            max_projection = max(max_projection, near(uv, geometry["object_points_uv"], 1e-6))
            visible = []
            for point, pixel, depth in zip(points, uv, camera_points[:, 2]):
                direction = point - transform[:3, 3]
                direction /= np.linalg.norm(direction)
                hit = np.array([-1], dtype=np.int32)
                distance = mujoco.mj_ray(model, data, transform[:3, 3], direction, None, 1, -1, hit)
                visible.append(bool(depth > 0 and np.all(pixel >= 0) and np.all(pixel < 256) and hit[0] == cube_geom and distance >= 0))
            if visible != geometry["object_point_visible"] or geometry["object_point_ids"] != list(range(8)):
                raise ValueError("Independent keypoint ray visibility differs")
            if camera == "wrist_rgb":
                rig = np.linalg.inv(tcp) @ transform
                if wrist_rig is None:
                    wrist_rig = rig
                near(rig, wrist_rig)
                if model.cam_bodyid[cid] != nameid(mujoco.mjtObj.mjOBJ_BODY, "x5_link6") or geometry["parent"] != "x5_link6":
                    raise ValueError("Wrist camera is not actual arm-attached")
            else:
                if body_rig is None:
                    body_rig = transform
                near(transform, body_rig)
                if model.cam_bodyid[cid] != 0 or geometry["parent"] != "world_fixed_rig":
                    raise ValueError("Body camera parent not honestly world-fixed rig")
        command = sample["supervision"]
        if index == 640:
            if command is not None:
                raise ValueError("Terminal frame must not invent a641st command")
        elif command is not None:
            reference, point = references[index // 8], index % 8
            if command["sequence"] != index or command["point_index"] != point or command["ref_id"] != reference["ref_id"] or not command["actually_committed"]:
                raise ValueError("Causal committed command/reference mismatch")
            near(command["command_time_s"], tick)
            for key in ("desired", "accepted", "installed", "committed"):
                near(command[key], reference["planned_H40"][point])
            near(command["desired_gripper"], reference["gripper_H40"][point])
            near(command["committed_gripper"], reference["gripper_H40"][point])
    for reference in references:
        index = reference["decision_index"] * 8
        if reference["prior_committed_count"] != index or reference["observation_snapshot"] != samples[index]["snapshot_id"]:
            raise ValueError("Reference planned with future or wrong observation")
        near(reference["decision_time_s"], index / 20)
        near(reference["anchor_actual_tcp"], samples[index]["policy_obs"]["actual_tcp"])
        near(reference["offsets_s"], np.arange(40) / 20)
        poses, grips = np.asarray(reference["planned_H40"]), np.asarray(reference["gripper_H40"])
        near(reference["rows10"], pose10(poses, grips))
        near(poses[0], reference["anchor_actual_tcp"])
        quaternions = np.asarray(reference["raw_quaternion_xyzw"])
        near(quat_xyzw_matrices(quaternions), poses[:, :3, :3])
        relative = np.linalg.inv(np.asarray(reference["anchor_actual_tcp"])) @ poses
        coded = pose10(relative, grips)
        restored = np.asarray(reference["anchor_actual_tcp"]) @ matrix_from10(coded)
        max_codec = max(max_codec, near(restored, poses))
        if reference["committed_prefix_count"] != min(8, max(0, min(n, 640) - index)):
            raise ValueError("Uncommitted plan tail presented as executed")
    score = independent_score(physics, card, complete, float(parameters["table_top"]))
    repeated_score = independent_score(physics, card, complete, float(parameters["table_top"]))
    if score != repeated_score:
        raise ValueError("Independent score repeat not identical")
    for key in ("persistent_success", "history", "hold_observed", "hold_pass", "hold_subpredicates", "safety_valid"):
        if key in score and score[key] != result["score"][key]:
            raise ValueError("Independent saved physics score differs: " + key)
    video = None
    if (folder / "ACTUAL_DUAL_RGB.mp4").is_file():
        import imageio.v2 as imageio
        reader = imageio.get_reader(folder / "ACTUAL_DUAL_RGB.mp4")
        actual_frames = sum(1 for _ in reader)
        reader.close()
        if actual_frames != n:
            raise ValueError("Actual decoded video frame count differs from raw observations")
        video = dict(decoded_frames=actual_frames, sha256=sha(folder / "ACTUAL_DUAL_RGB.mp4"))
    key_stages = ("approach", "align", "close", "verify_grasp", "lift", "carry", "lower", "release")
    visibility = dict(minimum_wrist_red_pixels=min(wrist_pixels), minimum_body_red_pixels=min(body_pixels),
                      wrist_zero_red_frames=int(sum(value == 0 for value in wrist_pixels)), initial_wrist_red_pixels=wrist_pixels[0],
                      per_stage={stage: dict(frames=len(values), minimum_red_pixels=min(values), maximum_red_pixels=max(values)) for stage, values in stage_pixels.items()},
                      initial_and_grasp_carry_release_visible=bool(wrist_pixels[0] >= 20 and all(min(values) >= 20 for stage, values in stage_pixels.items() if stage in key_stages)))
    return dict(id=identifier, domain=DOMAIN, status="PASS_FULL_RAW_AUDIT" if complete else "PRESERVED_INCOMPLETE_NOT_ADMITTED", complete=complete,
                mode=card['mode'], negative_fixture=card['mode'] != 'normal', negative_fixture_is_T1_quality_claim=False,
                all_action_supervision_valid=all(row['supervision']['valid'] for row in samples[:min(n,640)] if row['supervision'] is not None),
                formal_BC_admitted=False, admission_boundary='Raw audit is evidence only; commissioning/negatives never formal TRAIN/DEV',
                paired_samples=n, physics_steps=len(high), actual_committed_H40_windows=max(0, min(n - 1, len(high) // 25) - 39),
                file_count=len(files), source_sha=result["source_sha"], physics_sha256=sha(folder / "physics.npz"), samples_sha256=sha(folder / "samples.json"),
                manifest_sha256=sha(folder / "FILE_MANIFEST.json"), saved_scene_sha256=xml["sha256"], max_sensor_error=max_sensor,
                independent_FK_max_error=max_fk, max_camera_extrinsic_error=max_camera, max_keypoint_projection_error_px=max_projection,
                max_reconstructed_contact_force_error_N=max_force, native_codec_roundtrip_max_abs=max_codec,
                contact_samples=dict(left=contact_counts[0], right=contact_counts[1]), camera_visibility=visibility,
                actual_gripper_fraction_range=[min(row['policy_obs']['actual_gripper'] for row in samples), max(row['policy_obs']['actual_gripper'] for row in samples)],
                actual_gripper_clipped=False, actual_gripper_validity='Frozen physical joint tolerance1mm; commands remain[0,1]',
                independent_score=score, score_repeat_identical=True, video=video, privileged_policy_separation=True,
                causality_clock_and_reference_prefix_verified=True, negative_gates=negative_gates(samples),
                integrations=0, render_calls=0, original_raw_files_modified=False, model_loads=0, WBC_runs=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--physics-code-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(affinity[:4]))
    code_root = Path(args.physics_code_root)
    sys.path[:0] = [str(code_root / "src"), str(code_root / "scripts/pi0291_arx5_data")]
    from engine import Engine
    from config import PARAMETERS
    def no_integration(*args, **kwargs):
        raise PermissionError("Auditor forbids all physics integrations")
    mujoco.mj_step = mujoco.mj_step1 = mujoco.mj_step2 = no_integration
    begin = time.monotonic()
    records = []
    for identifier in args.ids:
        try:
            records.append(audit_episode(args.root, identifier, Engine, PARAMETERS))
        except Exception as error:
            records.append(dict(id=identifier, status="AUDIT_FAIL", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), integrations=0))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = dict(status="PASS_RAW_DATA_AUDIT" if all(row["status"] != "AUDIT_FAIL" for row in records) else "RAW_DATA_AUDIT_FAIL",
                  episodes=records, CPU_s=usage.ru_utime + usage.ru_stime + children.ru_utime + children.ru_stime,
                  elapsed_s=time.monotonic() - begin, audit_script_sha256=sha(__file__), integrations=0, model_loads=0, render_calls=0,
                  CPU_includes_children=True, resource_ledger_mutation="Owner must serially charge this measured CPU; auditor does not race physical ledger")
    write(args.output, result)
    print(result["status"], result["CPU_s"], [(row["id"], row["status"], row.get("error")) for row in records])
    if result["status"] != "PASS_RAW_DATA_AUDIT":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

