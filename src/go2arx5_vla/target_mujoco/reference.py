"""PI0288 registered reference panel, native wire contract and evidence scorer.

This is original integration code. Native source stays in the private runtime.
Pose vectors are world TCP [x,y,z,qx,qy,qz,qw], metres and radians.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from go2arx5_vla.trajectory.relative_se3 import pose_from_position_quaternion, pose_quaternion
from go2arx5_vla.trajectory.so3 import geodesic_angle

CONDITIONS = ("hold", "horizontal_0p05m", "vertical_0p05m",
              "local_z_rotation_0p15rad", "vertical_return", "H40_hotswap_every_0p4s")


def pose_matrix(pose):
    value = np.asarray(pose, dtype=float)
    if value.shape != (7,) or not np.isfinite(value).all() or np.linalg.norm(value[3:]) < 1e-8:
        raise ValueError("pose must be finite world TCP xyz + nonzero xyzw quaternion")
    return pose_from_position_quaternion(value[:3], value[3:])


def pose_error(a, b):
    a, b = pose_matrix(a), pose_matrix(b)
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(geodesic_angle(a[:3, :3], b[:3, :3]))


def _smooth(t):
    u = min(1.0, max(0.0, t / 2.0))
    return u * u * (3 - 2 * u)


def panel_pose(condition, initial_pose, t):
    """Fixed world +x/+z motion; rotation right-multiplies tool-local z.

    The registered smooth return is shared by panels 5 and 6. Its largest
    translational derivative is .0375 m/s; rotation reaches .1125 rad/s.
    """
    if condition not in CONDITIONS or not math.isfinite(t) or t < 0:
        raise ValueError("unknown condition or invalid episode time")
    pose = pose_matrix(initial_pose).copy()
    if condition == CONDITIONS[1]:
        pose[0, 3] += .05 * _smooth(t)
    elif condition == CONDITIONS[2]:
        pose[2, 3] += .05 * _smooth(t)
    elif condition == CONDITIONS[3]:
        a = .15 * _smooth(t)
        c, s = math.cos(a), math.sin(a)
        pose[:3, :3] = pose[:3, :3] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    elif condition in CONDITIONS[4:]:
        pose[2, 3] += .05 * (_smooth(t) if t <= 4 else 1 - _smooth(t - 4))
    return [*pose[:3, 3].tolist(), *pose_quaternion(pose).tolist()]


def native_document(condition, initial_pose, *, issue_time_s=0.0, h40=False):
    """Generate only source-validated native keys; receipt metadata stays outside.

    H40 is 40 planned samples at 0..1.95s. The unchanged native manager may
    publish a 41+ point 2s window. Binding clock is observed at runtime, never
    inferred from issue time; scorer exposes any resulting mismatch.
    """
    if not math.isfinite(issue_time_s) or issue_time_s < 0 or issue_time_s >= 10:
        raise ValueError("issue time must be in [0,10)")
    if h40 and condition != CONDITIONS[5]:
        raise ValueError("H40 replacement belongs to registered hotswap panel")
    times = [i * .05 for i in range(40 if h40 else 201)]
    points = []
    for t in times:
        pose = panel_pose(condition, initial_pose, issue_time_s + t)
        points.append({"time": round(t, 10), "position": pose[:3], "quaternion_xyzw": pose[3:]})
    return {"version": 1, "frame": "world", "start_delay": 0.0, "lead_in": 0.0,
            "align_orientation": False, "waypoints": points}


def panel_chunks(condition, initial_pose, episode_id):
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("episode UUID required")
    hot = condition == CONDITIONS[5]
    chunks = []
    for i in range(25 if hot else 1):
        issue = round(i * .4, 10)
        document = native_document(condition, initial_pose, issue_time_s=issue, h40=hot)
        wire = json.dumps(document, separators=(",", ":"), allow_nan=False)
        chunks.append({"episode_id": episode_id, "ref_id": f"{episode_id}:{i}",
                       "sequence": i, "issue_time_s": issue, "document": document,
                       "wire_sha256": hashlib.sha256(wire.encode()).hexdigest(), "wire_json": wire})
    return chunks


def interface_contract():
    """Exact source-read contract; this does not assert runtime acceptance."""
    return {
        "input_topic": "/go2_x5/ee/trajectory_json", "input_type": "std_msgs/msg/String",
        "output_topic": "/mobile_manipulator_mpc_target",
        "version": 1, "units": {"position": "m", "time": "s", "angle": "rad"},
        "pose_order": ["x", "y", "z", "qx", "qy", "qz", "qw"],
        "top_level": {"version": "optional default1", "waypoints": "legacy EE array",
                      "ee": "optional EE object, mutually exclusive with top-level waypoints",
                      "base": "optional base object with same trajectory fields"},
        "trajectory_fields": {"frame": "world(default)|world_offset|ee_start",
                              "start_delay": "finite >=0 default0", "lead_in": "finite >=0 default0",
                              "align_orientation": "boolean default true",
                              "waypoints": "nonempty list of {time,position[3],quaternion_xyzw[4]}"},
        "base_start_alias": "base.frame=base_start maps to ee_start",
        "parser": {"extra_keys": "ignored", "waypoints": "sorted by time; duplicates/negative rejected",
                   "numeric": "finite int/float, booleans rejected for vector/time fields",
                   "quaternion": "norm>=1e-8 then normalized", "version": "Python equality check against1"},
        "binding": {"time": "next MPC observation after complete JSON validation",
                    "world": "absolute world coordinates, orientation preserved",
                    "world_offset": "position offset in world; measured orientation right-multiplied if align_orientation",
                    "ee_start": "position rotated by measured orientation; orientation as world_offset",
                    "first_sample": "t<=bind+start_delay substituted by measured start_pose",
                    "republish": "fixed bound_start; does not reset on repeated 10Hz publication",
                    "replacement": "every new JSON clears bound_start, binds again at next observation"},
        "interpolation": "linear xyz and shortest-path quaternion SLERP",
        "output": "7D EE or14D EE+base pose; each input entry12 zeros",
        "window": "closed [MPC now,now+2s], dt.05, retains knots, 41+points",
        "native_ids_ack_cancel_TTL": "none in JSON; integration receipts required",
        "expiration": "stops republishing after end_time; previous OCS2 target persists",
        "invalid_json": "ValueError/TypeError/OverflowError/RecursionError rejected before replacing active reference",
        "HOME_HOLD": "reference mode0 sends measured pose hold; references dropped on prior enabled transition",
        "status": "SOURCE_READ_RUNTIME_UNVERIFIED"}


def _metrics(errors):
    x = np.asarray(errors)
    return {"RMS": float(np.sqrt(np.mean(x * x))), "P95": float(np.percentile(x, 95)), "max": float(x.max())}


def score_reference(condition, initial_pose, samples, *, issued=(), accepted=(), common_future=(), runtime_evidence=None):
    """Fail closed on missing real observations; synthetic rows are CPU examples.

    Each row carries t and requested/installed/achieved pose7, ref_id, all raw
    layer clocks and safety_ok. All poses must be evaluated on achieved_time_s
    (same simulation domain); issue/accept/install event clocks remain raw.
    """
    if condition not in CONDITIONS:
        raise ValueError("unknown registered panel condition")
    pose_matrix(initial_pose)
    required = {"t", "requested_pose", "installed_pose", "achieved_pose", "ref_id", "sim_time_s",
                "observation_time_s", "install_time_s", "dispatch_time_s", "achieved_time_s", "safety_ok"}
    evidence = runtime_evidence or {}
    reasons = []
    if not samples or any(not required <= x.keys() for x in samples):
        return {"status": "BLOCKED_MISSING_EVIDENCE", "pass": False, "missing": "complete four-layer real trace"}
    times = np.array([x["t"] for x in samples], dtype=float)
    if not np.isfinite(times).all() or times[0] < -1e-9 or times[-1] > 10.075:
        raise ValueError("invalid reference sample times")
    for x in samples:
        for name in ("requested_pose", "installed_pose", "achieved_pose"):
            pose_matrix(x[name])
        clocks = [x[k] for k in ("sim_time_s", "observation_time_s", "install_time_s", "dispatch_time_s", "achieved_time_s")]
        if not np.isfinite(clocks).all():
            raise ValueError("nonfinite layer clock")
    complete = len(times) >= 2 and times[0] <= .025 and times[-1] >= 10 - 1e-9 and np.all(np.diff(times) > 0) and np.max(np.diff(times)) <= .075 + 1e-9
    if not complete:
        reasons.append("INCOMPLETE_10S_TRACE")
    real_keys = ("real_output", "native_abort_record_complete", "joint_bounds_record_complete",
                 "contact_record_complete", "clock_record_complete", "dispatch_record_complete")
    missing = [k for k in real_keys if evidence.get(k) is not True]
    if missing:
        reasons.append("REAL_RUNTIME_EVIDENCE_MISSING")
    if any(x["safety_ok"] is not True for x in samples) or evidence.get("native_abort") is not False:
        reasons.append("NATIVE_ABORT_OR_SAFETY_UNVERIFIED")
    if evidence.get("numerical_bound_warning") is not False or evidence.get("illegal_contact") is not False:
        reasons.append("JOINT_WARNING_OR_CONTACT_UNVERIFIED")
    aligned = all(abs(x["sim_time_s"] - x["achieved_time_s"]) <= .025 and x["ref_id"] for x in samples)
    if not aligned or evidence.get("same_time_pose_alignment") is not True:
        reasons.append("SAME_TIME_ALIGNMENT_UNVERIFIED")
    if any(pose_error(panel_pose(condition, initial_pose, x["t"]), x["requested_pose"])[0] > 1e-5 or
           pose_error(panel_pose(condition, initial_pose, x["t"]), x["requested_pose"])[1] > 1e-4 for x in samples):
        reasons.append("REQUESTED_REGISTERED_PATH_MISMATCH")
    errors = {}
    tail = times >= 8 - 1e-9
    for name in ("requested_pose", "installed_pose"):
        values = np.array([pose_error(x[name], x["achieved_pose"]) for x in samples])
        errors[name] = {"full_position_m": _metrics(values[:, 0]), "full_SO3_rad": _metrics(values[:, 1]),
                        "last2_position_m": _metrics(values[tail, 0]) if tail.any() else None,
                        "last2_SO3_rad": _metrics(values[tail, 1]) if tail.any() else None}
    for name in errors:
        m = errors[name]
        if not tail.any() or m["last2_position_m"]["RMS"] > .015 or m["last2_position_m"]["max"] > .030 or m["last2_SO3_rad"]["RMS"] > .05 or m["last2_SO3_rad"]["max"] > .10:
            reasons.append(f"TRACKING_FAIL_{name}")
    plateau = [x for x in samples if 2 - 1e-9 <= x["t"] <= 4 + 1e-9]
    completion = None
    if condition in CONDITIONS[1:3]:
        axis = 0 if condition == CONDITIONS[1] else 2
        completion = float(np.mean([x["achieved_pose"][axis] - initial_pose[axis] for x in plateau]) / .05) if plateau else 0
        if completion < .8:
            reasons.append("DIRECTIONAL_TRANSLATION_FAIL")
    elif condition == CONDITIONS[3]:
        start = pose_matrix(initial_pose)[:3, :3]
        angles = []
        for x in plateau:
            relative = start.T @ pose_matrix(x["achieved_pose"])[:3, :3]
            angles.append(math.atan2(relative[1, 0], relative[0, 0]))
        completion = float(np.mean(angles)) if angles else 0
        final_angle = pose_error(panel_pose(condition, initial_pose, 10), samples[-1]["achieved_pose"])[1]
        if completion < .12 or final_angle > .03:
            reasons.append("DIRECTED_ROTATION_FAIL")
    jumps = []
    if condition == CONDITIONS[5]:
        ids = [x["ref_id"] for x in issued]
        accepted_ids = [x["ref_id"] for x in accepted]
        if len(ids) != 25 or len(set(ids)) != 25 or len(set(accepted_ids)) < 25 or not set(ids) <= set(accepted_ids):
            reasons.append("HOTSWAP_NOT_25_ISSUED_24_REPLACEMENTS")
        expected = {(a, b) for a, b in zip(ids, ids[1:])}
        observed = set()
        issue_times = {x["ref_id"]: x["issue_time_s"] for x in issued}
        if len(issued) == 25 and any(abs(x["issue_time_s"] - i * .4) > 1e-9 for i, x in enumerate(issued)):
            reasons.append("HOTSWAP_ISSUE_SCHEDULE_MISMATCH")
        for x in common_future:
            if not math.isfinite(x["future_time_s"]):
                raise ValueError("nonfinite common future time")
            old_issue = issue_times.get(x["old_ref_id"])
            new_issue = issue_times.get(x["new_ref_id"])
            if old_issue is None or new_issue is None or not new_issue <= x["future_time_s"] <= old_issue + 1.95 + 1e-9:
                reasons.append("COMMON_FUTURE_OUTSIDE_SHARED_PLANNED_SUPPORT")
            observed.add((x["old_ref_id"], x["new_ref_id"]))
            jumps.append(pose_error(x["old_pose"], x["new_pose"]))
        if not expected or not expected <= observed or not jumps:
            reasons.append("COMMON_FUTURE_REPLACEMENT_EVIDENCE_MISSING")
        elif max(p for p, _ in jumps) > .001 or max(a for _, a in jumps) > .005:
            reasons.append("COMMON_FUTURE_REFERENCE_JUMP")
    latency = []
    for x in accepted:
        if {"issue_time_s", "accept_time_s", "install_time_s"} <= x.keys():
            values = (x["accept_time_s"] - x["issue_time_s"], x["install_time_s"] - x["issue_time_s"])
            if not np.isfinite(values).all() or min(values) < 0:
                raise ValueError("invalid receipt latency")
            latency.append({"ref_id": x["ref_id"], "accept_s": values[0], "install_s": values[1]})
    return {"status": "PASS" if not reasons else ("BLOCKED_MISSING_EVIDENCE" if missing else "REFERENCE_QUALITY_FAIL"),
            "pass": not reasons, "condition": condition, "complete_10s": bool(complete),
            "sample_count": len(samples), "reasons": reasons, "missing_real_evidence": missing,
            "errors": errors, "directed_completion": completion, "latency": latency,
            "common_future_jumps": [{"position_m": p, "SO3_rad": a} for p, a in jumps],
            "reference_v_max_m_s": .05, "reference_w_max_rad_s": .20}
