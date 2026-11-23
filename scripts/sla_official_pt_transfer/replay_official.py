"""Score archived official predictions at their original consumed boundaries. CPU only."""

import argparse
import math
import os
import sys
from functools import partial
from pathlib import Path

from audit_sealed import check, digest, pin, read


class AnchoredIntent:
    """A separate lead-in in world coordinates followed by the unchanged absolute source curve."""

    def __init__(self, intent, anchor, limits):
        import numpy as np
        from scipy.spatial.transform import Rotation, Slerp

        self.intent = intent
        self.identity = intent.identity
        self.anchor = anchor
        self.first = intent.at([0])[0][0]
        self.rotation = Slerp(
            [0, 1], Rotation.from_matrix([anchor[:3, :3], self.first[:3, :3]])
        )
        self.distance = float(np.linalg.norm(self.first[:3, 3] - anchor[:3, 3]))
        self.angle = float(
            Rotation.from_matrix(self.first[:3, :3] @ anchor[:3, :3].T).magnitude()
        )
        # Peak first and second derivatives of 10u^3-15u^4+6u^5.
        duration = max(
            1.875 * self.distance / limits["linear_velocity"],
            1.875 * self.angle / limits["angular_velocity"],
            math.sqrt(
                10 / math.sqrt(3) * self.distance / limits["linear_acceleration"]
            ),
            math.sqrt(10 / math.sqrt(3) * self.angle / limits["angular_acceleration"]),
        )
        self.lead = math.ceil(duration / 0.05) * 0.05
        self.execution_times = self.lead + intent.execution_times

    def at(self, elapsed):
        import numpy as np

        t = np.atleast_1d(elapsed)
        check(np.isfinite(t).all() and np.min(t) >= 0, "execution clock")
        pose, g, phase = self.intent.at(np.maximum(t - self.lead, 0))
        before = t < self.lead
        if np.any(before):
            u = t[before] / self.lead
            h = u**3 * (10 + u * (-15 + 6 * u))
            pose[before, :3, 3] = self.anchor[:3, 3] + h[:, None] * (
                self.first[:3, 3] - self.anchor[:3, 3]
            )
            pose[before, :3, :3] = self.rotation(np.clip(h, 0.0, 1.0)).as_matrix()
        return pose, g, phase


def summarize_gate(gate):
    return {
        "H40_G7": gate["H40_G7_pass"],
        "H40": gate["full_H40"]["original_full_H40_pass"],
        "accepted_join": gate["accepted_join_pass"],
        "issues": gate["issues"],
        "G7": gate["G7"],
    }


def score(root, folder, o0):
    import numpy as np
    from audit import load_episode, make_context
    from b3_shadow import mode_run
    from gripper_ownership import assess
    from reference_manager_v2 import TaskSpaceIntentV2
    from retime import SourcePath, crossing, geometric_progress, twist
    from scipy.spatial.transform import Rotation, Slerp
    from score_target import check as legacy_gate
    from transfer_frames import convert

    from scripts.pi0291_arx5_data.native_dataset import matrix_from10

    p, v = root / "v13/run01", root / "v12"
    sealed = p / "S0_ACTUAL.json"
    pin(sealed, "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9")
    base = read(sealed)
    pins, limits = base["inputs"], base["limits"]
    for name in (
        "TARGET_REGISTRATION.json",
        "A2_O2_DENSE_CAPTURE.json",
        "A4_P2_ABI.json",
    ):
        pin(v / name, pins[str(v / name)])
    pin(
        p / "V13_READOUT.json",
        "bc867cc408a70f050b449f138a6273e94bb8ec295cc6e01036d00b8fc7527409",
    )
    pin(
        p / "S2_ACTUAL.json",
        "e53b719ef09742b623a15d337a6ad1fc45aefd33156cee373b1f956a5c4749d2",
    )
    s2 = {r["id"]: r for r in read(p / "S2_ACTUAL.json")["windows"]}
    archived = {r["window"]: r for r in read(v / "A4_P2_ABI.json")["windows"]}
    readout = {r["id"]: r for r in read(p / "V13_READOUT.json")["windows"]}
    bindings = read(v / "A2_O2_DENSE_CAPTURE.json")["windows"]
    roster = read(root / "evidence/P1_REGISTRATION.json")["target_windows"]
    outputs = {r["id"]: r for r in read(o0)["target"] if not r["repeat"]}
    episodes, result = {}, []
    for w in read(v / "TARGET_REGISTRATION.json")["windows"]:
        key, ep = w["id"], w["episode"]
        old, capture, grip = archived[key], bindings[key], s2[key]
        check(
            old["episode"] == grip["episode"] == ep
            and old["tick"] == grip["consumed_at_tick"] == w["tick"],
            key,
        )
        if ep not in episodes:
            episodes[ep] = load_episode(root, ep, roster[w["case_index"]], {})
        ctx, incoming, _, _, _, query, spec = make_context(root, w, old, episodes[ep])
        _, _, _, anchor, _, _, previous = ctx
        archive_path = v / "A4_ABI" / (key + ".npz")
        pin(archive_path, old["chain_sha256"])
        check(outputs[key]["output_sha256"] == old["raw_sha256"], "output binding")
        raw_path = next(
            Path(k)
            for k, h in pins.items()
            if h == old["raw_sha256"] and k.endswith(".npy")
        )
        pin(raw_path, old["raw_sha256"])
        raw = np.load(raw_path, allow_pickle=False)
        origin, tool = capture["T_W_source_origin"], capture["T_link6_TCP"]
        direct = convert(raw, origin, tool)
        with np.load(archive_path, allow_pickle=False) as z:
            check(
                np.array_equal(raw, z["raw16"])
                and np.array_equal(direct["H40"], z["H40"]),
                key,
            )
            teacher = matrix_from10(z["teacher_H40"])
        direct_gate = legacy_gate(direct["H40"], ctx)
        check(direct_gate == old["student"], "legacy direct reproduction")
        path = SourcePath(raw[:, 8:], origin, tool)
        half = spec["multitask_geometry"]["object_half_sizes"]["red"][0]

        progress = partial(
            geometric_progress,
            path,
            teacher=teacher,
            half_edge=half,
            stage=old["phase"],
        )

        raw_progress = progress(
            np.minimum(np.arange(40) * 0.05, 49 / 30),
            direct["world_H40"],
            grip=direct["H40"][:, 9],
        )
        duration = readout[key]["duration_s"]
        source = TaskSpaceIntentV2(key, raw, origin, tool, np.linspace(0, duration, 50))
        intent = AnchoredIntent(source, anchor, limits)
        # The base stays at the observed query pose for this shadow condition.
        mount = np.repeat(
            np.array(query["actual_state"]["base_pose"])[None], 41, axis=0
        )
        shadow = mode_run(folder / key, intent, mount, "ONE")
        end = intent.execution_times[-1]
        grid = np.arange(math.ceil(end / 0.05) + 1) * 0.05
        poses, g, phase = intent.at(grid)
        check(np.allclose(poses[0], anchor, atol=1e-8, rtol=0), "entry")
        violations = read(folder / key / "FULL40_ENVELOPE.private.json")
        # A preview can span the join. Count by start time only and retain actual channel maxima.
        rejected_before_source = sum(
            f["sequence"] * 0.1 < intent.lead - 1e-9 for f in violations
        )
        source_on_knots = intent.at(intent.execution_times)
        check(
            np.allclose(source_on_knots[0], direct["world_knots"], atol=2e-6, rtol=0)
            and np.allclose(source_on_knots[1], raw[:, 15], atol=1e-12, rtol=0),
            "unchanged source",
        )
        # Evaluate wire interpolation at all source knots. Source progress gives the bridge zero credit.
        emitted_xyz = np.stack(
            [np.interp(intent.execution_times, grid, c) for c in poses[:, :3, 3].T],
            axis=1,
        )
        rot = Slerp(grid, Rotation.from_matrix(poses[:, :3, :3]))(
            intent.execution_times
        )
        xyz_error = np.linalg.norm(
            emitted_xyz - direct["world_knots"][:, :3, 3], axis=1
        )
        angle_error = (
            rot.inv() * Rotation.from_matrix(direct["world_knots"][:, :3, :3])
        ).magnitude()
        wire_progress = progress(phase, poses, grip=g)
        source_progress = progress(phase, path.at(phase), grip=g)
        first_window = progress(phase[:40], path.at(phase[:40]), grip=g[:40])
        width = 0.002 + 0.040 * g
        rate = np.abs(np.diff(np.r_[previous["width_m"], width])) / 0.05
        first_twist = twist(poses[0], poses[1], 0.05)
        seam = [
            float(np.linalg.norm(a - b) / 0.05) for a, b in zip(first_twist, incoming)
        ]
        seam_ok = (
            seam[0] <= limits["linear_acceleration"] + 1e-10
            and seam[1] <= limits["angular_acceleration"] + 1e-10
        )
        dyn = bool(
            not shadow["original_full40_envelope_rejected_windows"]
            and seam_ok
            and rate.max() <= 0.04 + 1e-10
        )
        events_src = crossing(raw[:, 15])
        events_wire = crossing(g)
        event_times = [
            {
                "type": k,
                "source_s": float(i / 30),
                "expected_execution_s": float(intent.lead + i * duration / 49),
            }
            for k, i in events_src
        ]
        emitted_events = [
            {"type": k, "execution_s": float(i * 0.05)} for k, i in events_wire
        ]
        for expected, observed in zip(event_times, emitted_events):
            if expected["type"] == observed["type"]:
                expected["wire_error_s"] = (
                    observed["execution_s"] - expected["expected_execution_s"]
                )
        seen_motion = bool(
            np.any(np.linalg.norm(np.diff(path.raw[:, :3, 3], axis=0), axis=1) > 1e-9)
            or np.any(np.linalg.norm(np.diff(raw[:, 11:15], axis=0), axis=1) > 1e-9)
        )
        check(seen_motion or grip["contact"], "unclassified stationary intent")
        owner = assess(grip)
        check(owner != "SHADOW_MATCH_ONLY", "unexpected gripper authority")

        def labels(dynamics, progress):
            return {
                "WELL_FORMED": True,
                "FRAME_VALID": True,
                "OBS_IN_DOMAIN": False,
                "DYNAMICS_PASS": bool(dynamics),
                "GRIP_SAFE": False,
                "NONTRIVIAL_PROGRESS": bool(progress),
                "TASK_LEGAL": False,
            }

        result.append(
            {
                "id": key,
                "episode": ep,
                "tick": w["tick"],
                "phase": old["phase"],
                "raw_sha256": old["raw_sha256"],
                "DIRECT_OFFICIAL_RAW": {
                    "labels": labels(
                        direct_gate["H40_G7_pass"], raw_progress["motion_pass"]
                    ),
                    "gate": summarize_gate(direct_gate),
                    "progress": raw_progress,
                },
                "OFFICIAL_WITH_CONTINUOUS_REF": {
                    "labels": labels(dyn, source_progress["motion_pass"]),
                    "archived_duration_s": duration,
                    "lead_in_s": intent.lead,
                    "lead_in_m": intent.distance,
                    "lead_in_rad": intent.angle,
                    "source_frame_count": 50,
                    "rejected_starts_before_source": rejected_before_source,
                    "rejected_starts_on_source": len(violations)
                    - rejected_before_source,
                    "world_knot_wire_max_error_m": float(xyz_error.max()),
                    "world_knot_wire_max_error_rad": float(angle_error.max()),
                    "first_P10_source_progress": source_progress,
                    "wire_P10_travel_m": wire_progress["P10_travel_m"],
                    "first_preview_events_pass": first_window["event_pass"],
                    "source_events": event_times,
                    "emitted_events": emitted_events,
                    "full_path_events_pass": source_progress["event_pass"],
                    "source_events_within_first_preview": len(crossing(g[:40])),
                    "grip_seam_m_s": float(rate[0]),
                    "max_grip_rate_m_s": float(rate.max()),
                    "grip_limit_m_s": 0.04,
                    "boundary_accel": seam,
                    "boundary_limits": [
                        limits["linear_acceleration"],
                        limits["angular_acceleration"],
                    ],
                    "boundary_accel_pass": seam_ok,
                    "shadow": shadow,
                },
                "OFFICIAL_WITH_FEEDBACK_ELIGIBILITY": {
                    "status": "A1_DYNAMIC_UNQUALIFIED",
                    "evaluated": 1,
                    "native_controller_runs": 0,
                    "contact": grip["contact"],
                    "time_varying": seen_motion,
                },
                "previous_consumed_grip_m": previous["width_m"],
                "measured_grip_m": grip["measured_m"],
                "first_student_proposal_m": float(width[0]),
                "owner": owner,
                "held": grip["grasp"],
                "held_then_open": bool(
                    (grip["contact"] or grip["grasp"])
                    and width[0] > previous["width_m"] + 1e-10
                ),
            }
        )
    check(len(result) == 16 and len({r["id"] for r in result}) == 16, "denominator")
    return result


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("o0", type=Path)
    parser.add_argument("out", type=Path)
    a = parser.parse_args()
    root = a.root.resolve()
    os.umask(0o077)
    sys.path[:0] = [
        str(Path(__file__).resolve().parents[1] / "sla_transfer_v15"),
        str(root / "v13/src"),
        str(root / "v12"),
        str(root / "v12/CPU_LIB"),
    ]
    from d1_readiness import configure, record

    spec_pins = read(root / "v13/run01/S0_ACTUAL.json")["inputs"]
    spec_file = next(Path(k) for k in spec_pins if k.endswith("/EXECUTION_SPEC.json"))
    pin(spec_file, spec_pins[str(spec_file)])
    configure(read(spec_file))
    a.out.mkdir(exist_ok=False)
    rows = score(root, a.out, a.o0)
    keys = ("DIRECT_OFFICIAL_RAW", "OFFICIAL_WITH_CONTINUOUS_REF")
    counts = {
        m: {k: sum(x[m]["labels"][k] for x in rows) for k in rows[0][m]["labels"]}
        for m in keys
    }
    check(not counts[keys[0]]["DYNAMICS_PASS"], "archived control")
    record(
        a.out / "O1.json",
        {
            "denominator": len(rows),
            "labels": counts,
            "windows": rows,
            "o0_sha256": digest(a.o0),
            "new_gpu_forwards": 0,
            "new_physics_ticks": 0,
            "training_updates": 0,
            "gripper_commands_issued": 0,
            "heldout_episodes": 0,
        },
    )


if __name__ == "__main__":
    main()
