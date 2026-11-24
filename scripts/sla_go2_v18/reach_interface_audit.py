"""Isolate world-frame re-anchoring, prefix duration and gripper conventions."""

import json
import sys
from pathlib import Path

import numpy as np
from reach_intent_replay import BASE, compile_path, history, read, require, standing
from scripts.pi0292.eval_contract import matrix10
from world_intent_clock import (
    PrefixClock,
    action_fraction_to_width,
    command_width_to_action,
    h40_fraction_to_width,
    sensed_width_to_state,
    world_offset,
)
from compiler import DT, normmax, relative_angle, vectors


def geometry(poses, start, actual, row):
    displacement = poses[:, :3, 3] - poses[0, :3, 3]
    result = {
        "first_distance_from_consumed_mm": 1000
        * np.linalg.norm(poses[0, :3, 3] - start[:3, 3]),
        "length_mm": 1000
        * np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum(),
        "net_length_mm": 1000 * np.linalg.norm(displacement[-1]),
    }
    if row["standing"]:
        d = np.asarray(row["entry"]["direction"])
        red = np.asarray(row["at"]["object_pose"])[:3, 3]
        projected = actual[:3, 3] + (poses[:, :3, 3] - start[:3, 3])
        relative = projected - actual[:3, 3]
        vs_red = np.linalg.norm(projected - red, axis=1)
        result.update(
            final_axial_from_consumed_mm=1000 * relative[-1] @ d,
            peak_lateral_from_consumed_mm=1000
            * normmax(relative - (relative @ d)[:, None] * d),
            first_to_last_red_gain_mm=1000 * (vs_red[0] - vs_red[-1]),
        )
    return result


def limits_on_grid(commands, previous, limit):
    v, w = vectors(np.concatenate((previous[None], commands)), DT)
    values = dict(
        linear_velocity=normmax(v),
        angular_velocity=normmax(w),
        linear_acceleration=normmax(np.diff(v, axis=0) / DT),
        angular_acceleration=normmax(np.diff(w, axis=0) / DT),
    )
    return {
        "observed": values,
        "threshold": limit,
        "pass": all(x <= limit[k] + 1e-9 for k, x in values.items()),
    }


def replay(r, limits):
    at = r["at"]
    actual = np.asarray(at["actual_TCP"])
    ref = matrix10(np.asarray(at["target_endpoint_EEF10"]))
    previous = matrix10(np.asarray(at["installed_EEF10"]))
    origin = r["curve"]
    world = world_offset(origin.knots, actual, ref)
    tool = ref @ np.linalg.inv(actual) @ origin.knots
    require(
        np.allclose(
            world[:, :3, 3] - world[0, :3, 3],
            origin.knots[:, :3, 3] - origin.knots[0, :3, 3],
            rtol=0,
            atol=1e-10,
        ),
        "world direction changed",
    )
    absolute_full = compile_path(
        origin.knots, origin.arm[:, 7], ref, previous, limits, origin
    )
    world_full = compile_path(world, origin.arm[:, 7], ref, previous, limits)
    joint = PrefixClock(lambda i: world_offset(origin.at(i), actual, ref), limits)
    prefix = world[: joint.end_index + 1]
    stop_go = compile_path(prefix, None, ref, previous, limits)
    lead = stop_go.lead
    require(
        np.allclose(stop_go.first, joint.at([0.0])[0], rtol=0, atol=1e-10),
        "prefix start seam",
    )
    t = np.arange(round((lead + joint.duration + 2.0) / DT) + 1) * DT
    lead_commands, _, _ = stop_go.at(t[t < lead - 1e-9])
    commands = np.concatenate((lead_commands, joint.at(t[len(lead_commands) :] - lead)))
    screen = limits_on_grid(
        commands,
        previous,
        {
            k: limits[k]
            for k in (
                "linear_velocity",
                "angular_velocity",
                "linear_acceleration",
                "angular_acceleration",
            )
        },
    )
    angle = np.degrees(relative_angle(ref[:3, :3], actual[:3, :3]))
    budget = 9.0 - 0.002 * (r["tick"] - r["entry"]["tick"]) if r["standing"] else None
    out = {
        "id": r["id"],
        "standing": r["standing"],
        "reference_minus_actual_mm": (1000 * (ref[:3, 3] - actual[:3, 3])).tolist(),
        "reference_minus_actual_angle_deg": float(angle),
        "absolute": geometry(origin.knots, ref, actual, r),
        "body_chart": geometry(tool, ref, actual, r),
        "world_offset": geometry(world, ref, actual, r),
        "actual_archive_shape": {
            "absolute_first_to_last_rotation_deg": float(
                np.degrees(
                    relative_angle(origin.knots[0, :3, :3], origin.knots[-1, :3, :3])
                )
            ),
            "world_offset_first_to_last_rotation_deg": float(
                np.degrees(relative_angle(world[0, :3, :3], world[-1, :3, :3]))
            ),
        },
        "clock": {
            "source_prefix_s": joint.end_index / 30,
            "position_fit_mm": joint.shape_error_mm,
            "orientation_fit_deg": joint.shape_error_deg,
            "absolute_full_lead_s": absolute_full.lead,
            "absolute_full_s": absolute_full.end,
            "world_offset_full_lead_s": world_full.lead,
            "world_offset_full_s": world_full.end,
            "prefix_matched_fixed_open_lead_s": lead,
            "matched_stop_go_motion_s": float(
                stop_go.boundaries[joint.end_index] - lead
            ),
            "joint_motion_s": joint.duration,
            "joint_grid_limits_with_seam": screen,
            "standing_time_remaining_s": budget,
            "joint_finished_by_original_deadline": None
            if budget is None
            else lead + joint.duration <= budget,
            "prefix_geometry": geometry(prefix, ref, actual, r),
        },
    }
    return out


def gripper():
    fractions = np.array([0.0, 0.045454545454545456, 0.5, 0.9545454545454545, 1.0])
    width = h40_fraction_to_width(fractions)
    mapped = action_fraction_to_width(fractions)
    inside = (width >= 0.002) & (width <= 0.042)
    require(
        np.allclose(
            action_fraction_to_width(command_width_to_action(width[inside])),
            width[inside],
        ),
        "action width inverse",
    )
    require(
        np.allclose(sensed_width_to_state(width), fractions),
        "H40 command fraction identity",
    )
    return {
        "H40_fraction": fractions.tolist(),
        "H40_mm": (1000 * width).tolist(),
        "same_scalar_reinterpreted_as_action_mm": (1000 * mapped).tolist(),
        "same_scalar_width_delta_mm": (1000 * (mapped - width)).tolist(),
        "42mm_sensed_code": float(sensed_width_to_state(0.042)),
        "42mm_inverse_of_Go2_action_map": float(command_width_to_action(0.042)),
        "source_hardware_physical_width_verified": False,
        "scope": "Numeric Go2 slide/command conventions. Source physical pad gap not inferred.",
    }


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    limits = read(BASE / "reports/SLA_HIERARCHICAL_V16/V0_EVIDENCE_MATRIX_ACTUAL.json")[
        "limits"
    ]
    local = sys.argv[2:] == ["--standing-only"]
    require(local or not sys.argv[2:], "unknown mode")
    rows = [
        replay(r, limits)
        for group in ((standing,) if local else (history, standing))
        for r in group(root)
    ]
    require(len(rows) == (3 if local else 19), "missing sealed samples")
    print(
        json.dumps(
            {
                "rows": rows,
                "gripper": gripper(),
                "model_forwards": 0,
                "integrations": 0,
                "optimizer_updates": 0,
            },
            indent=2,
            allow_nan=False,
        )
    )
