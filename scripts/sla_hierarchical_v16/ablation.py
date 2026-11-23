"""Replay each fixed TRAIN window across the entire native 10-Hz preview schedule."""

import argparse
import json
import math
from pathlib import Path

from sealed_panel import configure_site


def output(path, obj):
    with path.open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False, default=lambda x: x.item())
        f.write("\n")


class SampledClock:
    def __init__(self, key, grid, poses, width, phase, end):
        self.identity = key
        self.execution_times = [0, end]
        self.grid, self.poses, self.width, self.phase = grid, poses, width, phase

    def at(self, t):
        import numpy as np

        i = np.rint(t / 0.05).astype(int)
        if (
            np.any(i < 0)
            or np.max(abs(t - i * 0.05)) > 1e-7
            or max(i) >= len(self.grid)
        ):
            raise ValueError("off-grid preview")
        return self.poses[i], self.width[i], self.phase[i]


def score_method(root, out, c, name, life, lead, at, reference_count):
    import numpy as np
    from b3_shadow import mode_run
    from compiler import DT, END, normmax, relative_angle, vectors
    from retime import crossing
    from scipy.spatial.transform import Rotation, Slerp
    from scipy.optimize import brentq
    from audit_sealed import check, read

    count = max(reference_count, math.ceil(life / 0.1 - 1e-9) + 1)
    horizon = (count - 1) * 0.1
    grid = np.arange(2 * (count - 1) + 41) * DT
    p, g, phase = at(grid)
    check(
        np.min(np.diff(phase)) >= -1e-9 and np.isclose(phase[-1], END),
        "nonmonotone source clock",
    )
    model = c["curve"]
    mounted = np.repeat(
        np.asarray(c["query"]["actual_state"]["base_pose"])[None], 41, axis=0
    )
    sampled = SampledClock(c["key"], grid, p, g, phase, horizon)
    # Only offline callback acknowledgements are constructed. The durable sequence is never consumed.
    shadow = mode_run(out, sampled, mounted, "ONE")
    check(count == shadow["installs"], "publication denominator")
    failures = read(out / "FULL40_ENVELOPE.private.json")
    for r in failures:
        t = 0.1 * r["sequence"]
        r["interval"] = (
            "alignment"
            if t + 1.95 < lead - 1e-9
            else "crosses_entry"
            if t < lead - 1e-9
            else "source_or_terminal"
        )
    last = c["context"][-1]["width_m"]
    width = 0.002 + 0.040 * g
    rates = np.abs(np.diff(np.r_[last, width])) / DT
    g7 = bool(np.max(rates) <= 0.04 + 1e-10)
    limits = c["limits"]
    v, w = vectors(p, DT)
    boundary = [np.linalg.norm(x - y) / DT for x, y in zip((v[0], w[0]), c["incoming"])]
    gap = np.linalg.norm(p[0, :3, 3] - c["context"][3][:3, 3])
    ang = relative_angle(p[:1, :3, :3], c["context"][3][None, :3, :3])[0]
    seam = bool(
        gap / DT <= 0.06 + 1e-10
        and ang / DT <= 0.3 + 1e-10
        and boundary[0] <= limits["linear_acceleration"] + 1e-10
        and boundary[1] <= limits["angular_acceleration"] + 1e-10
    )

    def timestamp(s):
        if s <= 0:
            return lead
        if s >= END:
            return life
        return brentq(
            lambda t: float(at(np.array([t]))[2][0]) - s, lead, life, xtol=1e-11
        )

    dense_phase = np.linspace(0, 49, 49 * 24 + 1)
    dense_time = np.linspace(lead, life, max(101, math.ceil((life - lead) / 0.005) + 1))
    exact, _, s = at(dense_time)
    native = model.at(30 * s)
    wire = np.stack(
        [np.interp(dense_time, grid, axis) for axis in p[:, :3, 3].T], axis=1
    )
    wire_rot = Slerp(grid, Rotation.from_matrix(p[:, :3, :3]))(dense_time).as_matrix()
    knots_t = np.array([timestamp(k / 30) for k in range(50)])
    at_knots = at(knots_t)[0]
    wire_knots = np.stack(
        [np.interp(knots_t, grid, axis) for axis in p[:, :3, 3].T], axis=1
    )
    wire_knots_r = Slerp(grid, Rotation.from_matrix(p[:, :3, :3]))(knots_t).as_matrix()
    geometry = dict(
        max_model_curve_error_m=normmax(exact[:, :3, 3] - native[:, :3, 3]),
        max_model_curve_error_rad=float(
            max(relative_angle(exact[:, :3, :3], native[:, :3, :3]))
        ),
        max_wire_curve_error_m=normmax(wire - native[:, :3, 3]),
        max_wire_curve_error_rad=float(
            max(relative_angle(wire_rot, native[:, :3, :3]))
        ),
        max_model_knot_error_m=normmax(at_knots[:, :3, 3] - model.knots[:, :3, 3]),
        max_wire_knot_error_m=normmax(wire_knots - model.knots[:, :3, 3]),
        max_wire_knot_error_rad=float(
            max(relative_angle(wire_knots_r, model.knots[:, :3, :3]))
        ),
    )
    path = model.at(dense_phase)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path[:, :3, 3], axis=0), axis=1))]
    if not arc[-1] > 0:
        raise ValueError("zero source curve")
    source_arc = np.interp(phase * 30, dense_phase, arc)
    events = []
    for typ, i in crossing(g):
        t = float(i * DT)
        q = float(np.interp(i, np.arange(len(phase)), phase))
        events.append(
            dict(
                kind=typ,
                time_s=t,
                source_progress=float(np.interp(q * 30, dense_phase, arc) / arc[-1]),
            )
        )
    expected = [
        dict(
            kind=k,
            source_s=float(i / 30),
            execution_s=timestamp(i / 30),
            source_progress=float(np.interp(i, dense_phase, arc) / arc[-1]),
        )
        for k, i in model.events
    ]
    ordered = len(expected) == len(events) and all(
        a["kind"] == b["kind"] for a, b in zip(expected, events)
    )
    event_errors = (
        [
            abs(a["source_progress"] - b["source_progress"])
            for a, b in zip(expected, events)
        ]
        if ordered
        else []
    )
    event_ok = ordered and max(event_errors, default=0) <= 0.1
    floor = c["prior"]["DIRECT_OFFICIAL_RAW"]["progress"]["P10_min_m"]
    reaches = np.flatnonzero(source_arc >= floor)
    motion = bool(len(reaches))
    # Evaluate acceleration/jerk along the complete continuous command, excluding artificial rolling pads.
    tick = np.arange(0, life + 0.005, 0.005)
    full = at(tick)[0]
    velocity = vectors(full, 0.005)
    kin = {}
    for k, val in zip(("linear", "angular"), velocity):
        acc = np.diff(val, axis=0) / 0.005
        kin[k] = {
            "speed_peak": normmax(val),
            "acceleration_peak": normmax(acc),
            "jerk_peak": normmax(np.diff(acc, axis=0) / 0.005),
        }
    opportunities = np.arange(0, life + 0.00001, 0.5)
    return dict(
        previews=count,
        original_roster_previews=reference_count,
        extra_previews=count - reference_count,
        original_roster_reject=sum(x["sequence"] < reference_count for x in failures),
        rejected_previews=len(failures),
        rejections=failures,
        pose_envelope_pass=not failures,
        boundary_pass=seam,
        boundary_translation_m=gap,
        boundary_angle_rad=ang,
        boundary_linear_angular_accel=boundary,
        g7_pass=g7,
        g7_limit_m_s=0.04,
        g7_max_m_s=float(max(rates)),
        g7_first_m_s=float(rates[0]),
        dynamics_pass=not failures and seam and g7,
        lead_s=lead,
        lifetime_s=life,
        source_duration_s=END,
        source_progress_0p45_m=source_arc[9],
        source_progress_floor_m=floor,
        first_execution_s_at_motion_floor=float(grid[reaches[0]]) if motion else None,
        progress_after_entry_0p5_m=float(np.interp(lead + 0.5, grid, source_arc)),
        full_source_travel_m=float(arc[-1]),
        nontrivial_source_arc=motion,
        events_source=expected,
        events_wire=events,
        events_preserved=event_ok,
        event_progress_errors=event_errors,
        geometry=geometry,
        kinematics_5ms=kin,
        model_opportunity_s=opportunities.tolist(),
        source_progress_at_opportunity_m=np.interp(
            opportunities, grid, source_arc
        ).tolist(),
        transport_field_sha256=shadow["defined_fields_sha256"],
        receiver_acks_offline=count,
        native_consumption=0,
    )


def run(root, repo, out):
    import numpy as np
    from compiler import Compiler, END
    from sealed_panel import cases
    from audit_sealed import check, pin
    from reference_manager_v2 import TaskSpaceIntentV2
    from replay_official import AnchoredIntent
    from score_target import check as legacy
    from transfer_frames import convert

    out.mkdir(exist_ok=False)
    results = []
    for c in cases(root, repo):
        k = c["key"]
        folder = out / k
        folder.mkdir()
        old = c["prior"]["OFFICIAL_WITH_CONTINUOUS_REF"]
        n = old["shadow"]["installs"]
        curve, ctx = c["curve"], c["context"]
        full = convert(c["arrays"]["raw16"], curve.origin, curve.tool)
        check(np.array_equal(full["H40"], c["arrays"]["H40"]), "original conversion")
        rawgate = legacy(full["H40"], ctx)
        check(rawgate == c["archive"]["student"], "original legacy gate")
        r = c["retime"]
        a = root / "v13/run01/arrays" / r["array_file"]
        pin(a, r["array_sha256"])
        with np.load(a, allow_pickle=False) as z:
            cg = legacy(z["H40"], ctx)
            check(
                cg["H40_G7_pass"] == r["native_gate"]["H40_G7_pass"]
                and cg["full_H40"] == r["native_gate"]["full_H40"],
                "C1 gate",
            )
            arch_phase = z["source_phase_s"]
        d = r["timing"]["duration_s"]
        slope = np.clip(r["timing"]["projected_initial_rate"], 0, min(1, 3 * END / d))

        def control(t, duration=None):
            grip_phase = np.minimum(t, END)
            phase = grip_phase
            if duration is not None:
                z = np.minimum(t / duration, 1.0)
                phase = (
                    END * (3 * z * z - 2 * z**3) + duration * slope * z * (1 - z) ** 2
                )
            return curve.at(phase * 30), curve.grip(grip_phase * 30), phase

        _, _, cp = control(np.arange(40) * 0.05, d)
        check(np.allclose(cp, arch_phase, atol=1e-12, rtol=0), "C1 phase clock")
        paths = {}
        paths["RAW_H40_G7"] = score_method(
            root, folder / "RAW", c, "RAW", END, 0, control, n
        )
        paths["TIME_ONLY_V13_C1"] = score_method(
            root, folder / "TIME", c, "TIME", d, 0, lambda t: control(t, d), n
        )
        ref = TaskSpaceIntentV2(
            k,
            c["arrays"]["raw16"],
            curve.origin,
            curve.tool,
            np.linspace(0, old["archived_duration_s"], 50),
        )
        o = AnchoredIntent(ref, ctx[3], c["limits"])
        paths["O1_CONTINUOUS"] = score_method(
            root, folder / "O1", c, "O1", o.execution_times[-1], o.lead, o.at, n
        )
        check(
            paths["O1_CONTINUOUS"]["rejected_previews"]
            == old["shadow"]["original_full40_envelope_rejected_windows"],
            "historical rolling control",
        )
        for name, b in [("C2_EXACT", False), ("C2_BOUNDED", True)]:
            compiler = Compiler(
                curve,
                ctx[3],
                c["incoming"],
                c["limits"],
                bounded=b,
                protect_all=c["grip"]["contact"] or c["grip"]["grasp"],
            )
            report = score_method(
                root,
                folder / name,
                c,
                name,
                compiler.end,
                compiler.lead,
                compiler.at,
                n,
            )
            report["block_count"] = len(compiler.parts)
            report["spline_blocks"] = sum(x.spline for x in compiler.parts)
            report["committed_entry_twist"] = np.array(c["incoming"]).tolist()
            paths[name] = report
            ts = np.arange(round(compiler.end / 0.05) + 1) * 0.05
            poses, width, phase = compiler.at(ts)
            with (folder / (name + ".private.npz")).open("xb") as f:
                np.savez_compressed(f, t=ts, poses=poses, source_s=phase, grip=width)
        result = dict(
            id=k,
            episode=c["episode"],
            condition=c["archive"]["phase"],
            raw_sha256=c["raw_file_sha256"],
            in_contact=c["grip"]["contact"],
            grasp=c["grip"]["grasp"],
            old_owner=False,
            modes=paths,
            legacy_h40_g7=rawgate["H40_G7_pass"],
            legacy_C1_h40_g7=cg["H40_G7_pass"],
        )
        output(folder / "WINDOW.json", result)
        results.append(result)
    check(len(results) == 16, "population")
    output(
        out / "V1_COMPILER_ABLATION_ACTUAL.json",
        dict(
            windows=results,
            scientific_windows=16,
            new_gpu_forwards=0,
            new_native_requests=0,
            new_physics_ticks=0,
            training_updates=0,
            native_stops=0,
            after_stop_ticks=0,
            lease="none acquired",
            student_task_attempts=0,
            student_success_rate=None,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    a = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    configure_site(a.root.resolve(), repo)
    run(a.root.resolve(), repo, a.out.resolve())
