"""Score a post-R1 constrained fit, same-vision geometry control and null inputs."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from reach_intent_replay import (
    BASE,
    history,
    standing,
    read,
    require,
    compile_path,
    frozen_checks,
)
from task_space_intent import direct_fit, geometric_control, visual_goal


def score(root, row, limits, *, check_scene=True):
    from scripts.pi0292.eval_contract import matrix10

    atom = row["at"]
    ref = matrix10(np.asarray(atom["target_endpoint_EEF10"]))
    prev = matrix10(np.asarray(atom["installed_EEF10"]))
    actual = np.array(atom["actual_TCP"])
    world = row["curve"].knots
    obj = row["obj"]
    result = dict(id=row["id"], condition=row["condition"], profiles={})
    try:
        axis, _, _ = visual_goal(obj, actual)
    except ValueError as e:
        result["vision_refusal"] = str(e)
        return result
    # Synthetic controls are constructed in the same consumed-reference chart.
    relative = ref @ np.linalg.solve(actual, world)
    reverse = relative.copy()
    xyz = reverse[:, :3, 3] - ref[:3, 3]
    reverse[:, :3, 3] -= 2 * np.maximum(xyz @ axis, 0)[:, None] * axis
    controls = dict(
        null=np.repeat(actual[None], 50, axis=0),
        reverse=actual @ np.linalg.solve(ref, reverse),
    )
    for name, raw in controls.items():
        try:
            direct_fit(raw, actual, ref, obj)
        except ValueError as e:
            require(
                str(e) == "no forward component in SLA prefix",
                "unexpected negative control failure",
            )
            result[name] = str(e)
        else:
            raise ValueError("accepted null/reversed proposal")
    try:
        path, stats = direct_fit(world, actual, ref, obj)
    except ValueError as e:
        result["profiles"]["DIRECT"] = dict(refusal=str(e))
        path = None
    geo = geometric_control(actual, ref, obj)
    for name, p in (("DIRECT", path), ("GEOMETRY", geo)):
        if p is None:
            continue
        delta = p[:, :3, 3] - ref[:3, 3]
        val = dict(
            refusal=None,
            geometry_difference_rms_mm=float(
                1000
                * np.sqrt(np.mean(np.sum((p[:, :3, 3] - geo[:, :3, 3]) ** 2, axis=1)))
            ),
            terminal_distance_from_geometry_mm=1000
            * np.linalg.norm(p[-1, :3, 3] - geo[-1, :3, 3]),
        )
        if name == "DIRECT":
            val.update(stats)
        if row["standing"]:
            # Everything below this point scores an already frozen proposal.
            e = row["entry"]
            task = np.array(e["direction"])
            red = np.asarray(atom["object_pose"])[:3, 3]
            start = np.asarray(e["actual_xyz"])
            proposed = actual[:3, 3] + delta
            proj = delta @ task
            val.update(
                local_endpoint_error_mm=1000 * np.linalg.norm(delta[-1] - 0.02 * task),
                local_cross_track_max_mm=1000
                * np.linalg.norm(delta - proj[:, None] * task, axis=1).max(),
                local_axial_mm=1000 * proj[-1],
                local_red_gain_mm=1000
                * (
                    np.linalg.norm(actual[:3, 3] - red)
                    - np.linalg.norm(proposed[-1] - red)
                ),
            )
            comp = compile_path(
                p,
                None if name == "GEOMETRY" else row["curve"].arm[:, 7],
                ref,
                prev,
                limits,
            )
            val["static_preview"] = (
                frozen_checks(root, row, p, comp, ref, prev) if check_scene else None
            )
            budget = 9.0 - 0.002 * (row["tick"] - e["tick"])
            clock = np.arange(round(budget / 0.05) + 1) * 0.05
            cmd, _, phase = comp.at(clock)
            predicted = actual[:3, 3] + (cmd[:, :3, 3] - ref[:3, 3])
            delta_entry = predicted - start
            proj = delta_entry @ task
            rotation = Rotation.from_matrix(
                cmd[:, :3, :3] @ np.asarray(e["consumed"])[:3, :3].T
            ).magnitude()
            val["reference_only_until_original_deadline"] = dict(
                remaining_seconds=budget,
                alignment_s=comp.lead,
                selected_prefix_s_reached=min(0.5, float(phase[-1])),
                projected_terminal_min_mm=1000 * proj[-11:].min(),
                terminal_error_max_mm=1000
                * np.linalg.norm(delta_entry[-11:] - 0.02 * task, axis=1).max(),
                cross_track_max_mm=1000
                * np.linalg.norm(delta_entry - proj[:, None] * task, axis=1).max(),
                orientation_max_deg=np.degrees(rotation.max()),
                terminal_red_gain_min_mm=1000
                * (
                    np.linalg.norm(start - red)
                    - np.linalg.norm(predicted[-11:] - red, axis=1).max()
                ),
            )
        result["profiles"][name] = val
    return result


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    limits = read(BASE / "reports/SLA_HIERARCHICAL_V16/V0_EVIDENCE_MATRIX_ACTUAL.json")[
        "limits"
    ]
    results = []
    geometry_only = sys.argv[2:] == ["--reach-geometry-only"]
    require(geometry_only or not sys.argv[2:], "unknown mode")
    for collection in (standing,) if geometry_only else (history, standing):
        for r in collection(root):
            results.append(score(root, r, limits, check_scene=not geometry_only))
    require(len(results) == (3 if geometry_only else 19), "incomplete panel")
    print(
        json.dumps(
            dict(rows=results, native_starts=0, integrations=0, forwards=0, updates=0),
            indent=2,
            allow_nan=False,
        )
    )
