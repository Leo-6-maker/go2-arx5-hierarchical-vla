"""Replay the crossed H1 handoffs on sealed teacher observations, without starting a plant."""

from pathlib import Path
import sys
import numpy as np

from sealed_panel import configure_site

DT = 0.05


def check(condition, message):
    if not condition:
        raise ValueError(message)


def contact(c):
    return bool(c["grip"]["contact"] or c["grip"]["grasp"])


class ReturnedPreview:
    """Select the available plan at publication time, including its full lookahead."""

    def __init__(self, revised, first_install):
        from revision_wire import NewSourcePhaseView

        self.previous = revised.previous
        self.future = NewSourcePhaseView(revised)
        self.first_install = first_install
        self.execution_times = revised.execution_times

    def at(self, t):
        if len(t) != 41 or not np.allclose(
            t, t[0] + DT * np.arange(41), rtol=0, atol=1e-8
        ):
            raise ValueError("expected one publication window")
        if round(t[0] / DT) >= 2 * self.first_install:
            return self.future.at(t)
        p, g, phase = self.previous.at(t)
        return p, g, np.zeros_like(phase)


def gates(wire, previous_m, count, publications=None):
    from go2arx5_vla.world_model.mobile_wbc_reference_v1 import full_eef_gate, _geometry
    from go2arx5_vla.world_model import fusion_trajectory_adapter as limits
    from scripts.pi0291_arx5_data.native_dataset import pose10
    from retime import crossing

    t = np.arange(2 * (count - 1) + 41) * DT
    poses, g, _ = wire.at(t)
    widths = 0.002 + 0.040 * g
    rates = np.abs(np.diff(np.r_[previous_m, widths])) / DT
    envelope = {
        "maximum_translation_speed_m_s": limits.MAX_TRANSLATION_SPEED_M_S,
        "maximum_angular_speed_rad_s": limits.MAX_ANGULAR_SPEED_RAD_S,
        "maximum_finger_speed_m_s": limits.MAX_FINGER_SPEED_M_S,
        "maximum_discrete_translation_acceleration_m_s2": limits.MAX_TRANSLATION_ACCELERATION_M_S2,
        "maximum_discrete_angular_acceleration_rad_s2": limits.MAX_ANGULAR_ACCELERATION_RAD_S2,
    }
    result = {}
    for k, frac in (("legacy", g), ("physical", widths / 0.044)):
        actions = pose10(poses, frac)
        refused = []
        for i in range(count):
            if publications is None:
                chunk = actions[i * 2 : i * 2 + 40]
                origin = poses[i * 2]
            else:
                p, w, _ = publications.at(t[i * 2 : i * 2 + 41])
                fraction = w if k == "legacy" else (0.002 + 0.040 * w) / 0.044
                chunk = pose10(p[:40], fraction[:40])
                origin = p[0]
            try:
                full_eef_gate(chunk, origin)
            except ValueError as e:
                values = _geometry(chunk, origin)
                exceeded = {
                    k: dict(observed=float(values[k]), limit=float(v))
                    for k, v in envelope.items()
                    if values[k] > v * (1 + 1e-10)
                }
                check(
                    str(e) == "MOBILE_ORIGINAL_FULL40_EEF_GATE" and exceeded,
                    "unidentified refusal",
                )
                refused.append(
                    dict(
                        sequence=i,
                        reason=str(e),
                        limits_exceeded=exceeded,
                    )
                )
        result[k] = refused
    return dict(
        count=count,
        envelopes=result,
        maximum_width_rate_mm_s=float(1000 * np.max(rates)),
        g7=bool(max(rates) <= 0.04 + 1e-10),
        wire_events=[dict(kind=k, time=float(i * DT)) for k, i in crossing(g)],
    )


def initial(root, case, scene, data, geometry, task):
    from compiler import Compiler
    from prefix_replay import WireProgram
    from gripper_alignment import alignment_widths
    from reachability import check_wire

    c = case
    compiler = Compiler(c["curve"], c["context"][3], c["incoming"], c["limits"])
    path = root / "v16/v1-run01" / c["key"] / "C2_EXACT.private.npz"
    with np.load(path, allow_pickle=False) as z:
        t = z["t"]
        p, g, s = compiler.at(t)
        check(
            all(
                np.array_equal(a, z[name])
                for a, name in ((p, "poses"), (g, "grip"), (s, "source_s"))
            ),
            "V1 baseline changed",
        )
    before = c["context"][-1]["width_m"]
    check(
        abs(c["grip"]["previous_command_m"] - before) < 1e-10,
        "native previous width identity",
    )
    old = WireProgram(c["episode"], p, g, s)
    raw = gates(old, before, data["previews"])
    check(
        len(raw["envelopes"]["legacy"]) == data["rejected_previews"]
        and raw["g7"] == data["g7_pass"],
        "historic initial gates",
    )

    def qualify(v, geo):
        return bool(
            geo["pass"]
            and v["g7"]
            and not any(v["envelopes"].values())
            and data["events_preserved"]
            and data["boundary_pass"]
        )

    row = dict(
        id=c["key"],
        grasp_or_contact=contact(c),
        source_sha256=c["raw_file_sha256"],
        phase_proxy=task,
        raw_gates=raw,
        raw_geometry=geometry,
        raw_qualified=qualify(raw, geometry),
    )
    raw_geom = check_wire(scene, c, t, p, g)
    check(raw_geom == geometry, "historical FCL witness changed")
    aligned, entry = alignment_widths(t, s, g, before, compiler.lead, contact(c))
    row["alignment"] = entry
    if aligned is None:
        row["aligned_qualified"] = False
        return row, (old, None)
    plan = WireProgram(c["episode"], p, aligned, s)
    result = gates(plan, before, data["previews"])
    check(
        raw["wire_events"] == result["wire_events"]
        and np.array_equal(
            aligned[t >= compiler.lead - 1e-9], g[t >= compiler.lead - 1e-9]
        ),
        "altered source gripper event",
    )
    geo = check_wire(scene, c, t, p, aligned)
    row.update(
        aligned_gates=result,
        aligned_geometry=geo,
        aligned_qualified=qualify(result, geo),
    )
    return row, (old, plan)


def pair(
    root,
    out,
    old_case,
    new_case,
    previous,
    old_initial,
    calls,
    prior,
    scene,
    protection,
    align,
):
    from b3_shadow import mode_run
    from compiler import relative_angle, END
    from governor import PersistentReference
    from gripper_alignment import alignment_widths
    from prefix_replay import WireProgram
    from reachability import check_wire
    from revision_replay import verify_revision
    from revision_wire import first_publication, RevisionWire

    class SolverHorizonShadow(PersistentReference):
        def protected_end(self, now):
            return now - self.start + 1.05

    class SolverHorizonWire(RevisionWire):
        @staticmethod
        def protected_knot(last_install):
            return 2 * (last_install + 1) + 21

    old, new = calls[old_case["key"]], calls[new_case["key"]]
    interval = (new["tick"] - old["tick"]) * 0.002
    origin = first_publication(old["seconds"])
    publication = first_publication(interval + new["seconds"])
    at = publication - origin
    check(
        at > 0
        and old["episode"] == new["episode"]
        and old["file_sha256"] != new["file_sha256"],
        "chronology",
    )
    row = dict(
        old=old["id"],
        new=new["id"],
        protection=protection,
        alignment=align,
        initial_qualified=old_initial,
        outcome="INITIAL_WIDTH_REJECTED",
        numeric_pass=False,
        geometric_and_numeric_pass=False,
    )
    if previous is None:
        return row
    cls = RevisionWire if protection == "FULL" else SolverHorizonWire
    manager = PersistentReference if protection == "FULL" else SolverHorizonShadow
    revised = cls(
        previous,
        old["file_sha256"],
        new["file_sha256"],
        at - 1,
        new_case["curve"],
        new_case["limits"],
    )
    cut = revised.cut
    if align:
        p, g, s = revised.future.arrays
        last = float(previous.at(np.array([cut * DT]))[1][0])
        new_width, entry = alignment_widths(
            np.arange(len(g)) * DT,
            s,
            g,
            0.002 + 0.040 * last,
            revised.compiler.lead,
            contact(old_case) or contact(new_case),
        )
        row["revision_entry"] = entry
        if new_width is None:
            row["outcome"] = "REVISION_WIDTH_REJECTED"
            return row
        revised.future = WireProgram(revised.identity, p, new_width, s)
    end = round(revised.execution_times[-1] / DT)
    t = np.arange(end + 1) * DT
    pose, g, phase = revised.at(t)
    old_pose, old_g, old_phase = previous.at(t)
    for a, b in zip((pose, g, phase), (old_pose, old_g, old_phase), strict=True):
        check(np.array_equal(a[: cut + 1], b[: cut + 1]), "changed protected prefix")
    diff = (
        (np.linalg.norm(pose[:, :3, 3] - old_pose[:, :3, 3], axis=1) > 1e-10)
        | (relative_angle(pose[:, :3, :3], old_pose[:, :3, :3]) > 1e-9)
        | (np.abs(g - old_g) > 1e-10)
    )
    changed = np.flatnonzero(diff)
    source = np.flatnonzero(
        (np.arange(len(t)) > cut)
        & (phase > 0)
        & (
            np.linalg.norm(pose[:, :3, 3] - new_case["curve"].knots[0, :3, 3], axis=1)
            > 1e-9
        )
    )
    check(
        len(changed)
        and len(source)
        and changed[0] > cut
        and abs(phase[-1] - END) < 1e-8,
        "lost real future",
    )
    folder = out / (
        old["id"]
        + "-"
        + new["id"]
        + "-"
        + protection
        + ("-ALIGNED" if align else "-RAW")
    )
    folder.mkdir()
    wd = folder / "C2_EXACT"
    base = np.repeat(
        np.asarray(old_case["query"]["actual_state"]["base_pose"])[None], 41, axis=0
    )
    publications = ReturnedPreview(revised, at)
    shadow = mode_run(wd, publications, base, "ONE")
    verify_revision(
        previous,
        revised,
        root / "v16/v1-run01" / old["id"] / "C2_EXACT/CALLBACKS.private.jsonl",
        wd / "CALLBACKS.private.jsonl",
        at,
        base,
        shadow["installs"],
        manager_type=manager,
    )
    ga = gates(
        revised, old_case["context"][-1]["width_m"], shadow["installs"], publications
    )
    check(
        len(ga["envelopes"]["legacy"])
        == shadow["original_full40_envelope_rejected_windows"],
        "runtime envelope differs",
    )
    try:
        revised.check_gripper_handoff()
        seam = True
    except ValueError:
        seam = False
    new_events = [e["kind"] for e in ga["wire_events"] if e["time"] > cut * DT + 1e-10]
    source_events = [k for k, _ in new_case["curve"].events]
    from retime import crossing

    expected = [
        (k, (cut + float(i)) * DT) for k, i in crossing(revised.future.arrays[1])
    ]
    observed = [
        (r["kind"], r["time"])
        for r in ga["wire_events"]
        if r["time"] > cut * DT + 1e-10
    ]
    fidelity = (
        new_events == source_events
        and len(expected) == len(observed)
        and all(
            k == q and abs(a - b) <= 1e-9 for (k, a), (q, b) in zip(expected, observed)
        )
    )
    numeric = bool(seam and ga["g7"] and fidelity and not any(ga["envelopes"].values()))
    age = lambda i: float(origin / 10 + t[i] - interval)
    row.update(
        outcome="OFFLINE_CALLBACK_COMPLETE",
        cut_knot=cut,
        first_changed_age_s=age(changed[0]),
        new_source_motion_age_s=age(source[0]),
        source_events=source_events,
        events_after_cut=new_events,
        handoff_brake_s=revised.compiler.brake,
        handoff_align_s=revised.compiler.align,
        handoff_translation_mm=float(
            1000
            * np.linalg.norm(
                revised.compiler.first[:3, 3] - revised.compiler.stopped[:3, 3]
            )
        ),
        expected_wire_events_s=expected,
        observed_wire_events_s=observed,
        events_and_grid_times_match=fidelity,
        seam_pass=seam,
        numeric_pass=numeric,
        gates=ga,
        pose_wire_sha256=shadow["defined_fields_sha256"],
        native_consumptions=shadow["native_consumption_receipts"],
    )
    if numeric:
        geo = check_wire(scene, old_case, t, pose, g)
        row.update(
            revised_geometry=geo,
            geometric_and_numeric_pass=bool(geo["pass"] and old_initial),
        )
    if protection == "FULL" and not align:
        check(
            ga["count"] == prior["full40_previews"]
            and len(ga["envelopes"]["legacy"]) == len(prior["rejected_previews"])
            and ga["g7"] == prior["full_g7_pass"]
            and abs(
                row["new_source_motion_age_s"]
                - prior["age_at_first_new_source_motion_s"]
            )
            < 1e-8
            and row["pose_wire_sha256"] == prior["callback_fields_sha256"],
            "historic H1 pair differs",
        )
    return row


def run(root, repo, out):
    from ablation import output
    from audit_sealed import pin, read
    from frozen_geometry import Scene
    from sealed_panel import cases
    from revision_replay import primary_calls

    folder = repo / "reports/SLA_HIERARCHICAL_V16"
    out.mkdir(parents=True, exist_ok=False)
    v4 = read(folder / "V4_SLA_SKILL_AND_LONG_HORIZON_DIAGNOSTIC.json")
    decision = folder / "V1_DECISION_ACTUAL.json"
    pin(decision, v4["source_v1_decision_sha256"])
    d = read(decision)
    files = d["source_sha256"]
    for k in ("V1_REACH.json", "V1_COMPILER_ABLATION_ACTUAL.json"):
        pin(folder / k, files[k])
    geometry = {
        r["id"]: r["methods"]["C2_EXACT"]
        for r in read(folder / "V1_REACH.json")["records"]
    }
    ablation = {
        r["id"]: r for r in read(folder / "V1_COMPILER_ABLATION_ACTUAL.json")["windows"]
    }
    task = {r["id"]: r["phase_proxy_pass"] for r in v4["windows"]}
    history = read(folder / "H1_REVISION_ACTUAL.json")
    all_cases = {c["key"]: c for c in cases(root, repo)}
    calls = primary_calls(root, all_cases, v4)
    check(
        calls == history["calls"] and list(sorted(all_cases)) == d["population"],
        "wrong revisions",
    )
    rows = {}
    plans = {}
    scenes = {}
    for key, c in all_cases.items():
        f = Path(c["spec"]["model_path"])
        if f not in scenes:
            scenes[f] = Scene(
                f, root / "v16/scene_layout", c["spec"]["frozen_files"][str(f)]
            )
        rows[key], plans[key] = initial(
            root,
            c,
            scenes[f],
            ablation[key]["modes"]["C2_EXACT"],
            geometry[key],
            task[key],
        )
    pairs = []
    for old in history["pairs"]:
        a, b = old["old"], old["new"]
        c = all_cases[a]
        scene = scenes[Path(c["spec"]["model_path"])]
        for protection in ("FULL", "SOLVER"):
            for align in (False, True):
                r = pair(
                    root,
                    out,
                    c,
                    all_cases[b],
                    plans[a][int(align)],
                    rows[a]["aligned_qualified" if align else "raw_qualified"],
                    calls,
                    old,
                    scene,
                    protection,
                    align,
                )
                pairs.append(r)
    check(len(pairs) == 40, "incomplete crossed experiment")
    output(
        out / "HANDOFF_FACTORS_ACTUAL.json",
        dict(
            initial=list(rows.values()),
            pairs=pairs,
            native_starts=0,
            native_consumptions=0,
            own_feedback_updates=0,
            new_forwards=0,
            training_updates=0,
        ),
    )


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parents[2]
    configure_site(root, repo)
    sys.path.insert(0, str(root / "v16/cpu-fcl"))
    run(root, repo, Path(sys.argv[2]).resolve())
