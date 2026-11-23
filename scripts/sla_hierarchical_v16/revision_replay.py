"""Schedule distinct returns from consecutive teacher observations through the frozen callback."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sealed_panel import configure_site


def primary_calls(root, cases, v4):
    from audit_sealed import pin

    calls = {}
    for name, metadata in v4["latency"]["groups"].items():
        folder = root / "v12" / name
        log = folder / "calls.jsonl"
        pin(log, metadata["journal_sha256"])
        for line in log.read_text().splitlines():
            r = json.loads(line)
            if r["event"] != "FORWARD_RETURNED" or r["repeat"]:
                continue
            key = r["id"]
            c = cases[key]
            if (
                key in calls
                or r["episode"] != c["episode"]
                or r["tick"] != c["window"]["tick"]
                or r["seed"] != c["window"]["seed"]
                or r["input_sha256"] != c["capture"]["input_sha256"]
                or r["file_sha256"] != c["raw_file_sha256"]
                or not np.isfinite(r["seconds"])
                or r["seconds"] <= 0
            ):
                raise ValueError("unmatched return " + key)
            pin(folder / r["output_file"], r["file_sha256"])
            calls[key] = {
                k: r[k]
                for k in (
                    "id",
                    "tick",
                    "episode",
                    "seed",
                    "input_sha256",
                    "file_sha256",
                    "seconds",
                )
            }
    if set(calls) != set(cases):
        raise ValueError("missing primary call")
    return calls


def receipts(path):
    pending = None
    with path.open() as stream:
        for line in stream:
            e = json.loads(line)
            if e["kind"] == "accept":
                meta = json.loads(e["raw"])["_pi0294"]
                if meta["operation"] != "install":
                    continue
                if pending is not None:
                    raise ValueError("overlapping receipts")
                pending = e
            elif e["kind"] == "installed" and pending is not None:
                if pending["ref_id"] != e["ref_id"]:
                    raise ValueError("callback receipt mismatch")
                yield pending, e
                pending = None


def verify_revision(
    previous,
    revised,
    old_log,
    new_log,
    install_index,
    base,
    count,
    *,
    manager_type=None,
):
    from governor import PersistentReference
    from ocs2_target_adapter import wire

    original = receipts(old_log)
    received = receipts(new_log)
    epoch = "C2_EXACT"
    m = (manager_type or PersistentReference)(previous, epoch, 30.0)
    for sequence in range(count):
        now = 30.0 + sequence * 0.1
        a, b = next(received)
        if sequence < install_index:
            old_accept, _ = next(original)
            if old_accept["ref_id"] != a["ref_id"]:
                raise ValueError("historical wire changed before revision")
        if sequence == install_index:
            m.revise(revised, now, epoch)
        key, (pose, _, _) = m.window(now)
        if wire(pose, base, epoch, sequence, now) != a["raw"]:
            raise ValueError("different active wire from the acknowledged message")
        m.acknowledge(key, a["ref_id"], a, b)
    if m.consumed is not None or m.revision != 1 or next(received, None) is not None:
        raise ValueError("incorrect revision or callback population")


def pair(root, out, older, newer, calls, admitted, qualified):
    from ablation import output
    from audit_sealed import read
    from b3_shadow import mode_run
    from compiler import Compiler, DT, END, relative_angle
    from prefix_replay import WireProgram
    from revision_wire import first_publication, NewSourcePhaseView, RevisionWire
    from retime import crossing

    left, right = older["key"], newer["key"]
    old, new = calls[left], calls[right]
    delta = (new["tick"] - old["tick"]) * 0.002
    if old["episode"] != new["episode"] or delta <= 0:
        raise ValueError("noncausal observation pair")
    origin = first_publication(old["seconds"])
    publication = first_publication(delta + new["seconds"])
    index = publication - origin
    result = dict(
        old=left,
        new=right,
        observation_interval_s=delta,
        previous_first_install_s=origin / 10,
        new_first_possible_install_s=publication / 10,
        previous_isolated_numeric_geometry_pass=left in admitted,
        new_isolated_numeric_geometry_pass=right in admitted,
        previous_isolated_task_and_geometry=left in qualified,
        archived_plant_owner="teacher",
        student_native_receipts=0,
    )
    if index <= 0:
        return dict(result, status="NO_PRIOR_INSTALLATION")
    if old["file_sha256"] == new["file_sha256"]:
        return dict(result, status="SAME_PREDICTION_BYTES")

    archive = root / "v16/v1-run01" / left
    old_plan = Compiler(
        older["curve"], older["context"][3], older["incoming"], older["limits"]
    )
    with np.load(archive / "C2_EXACT.private.npz", allow_pickle=False) as z:
        values = old_plan.at(z["t"])
        if not all(
            np.array_equal(a, b)
            for a, b in zip(values, (z["poses"], z["grip"], z["source_s"]), strict=True)
        ):
            raise ValueError("old compiler differs from archived schedule")
        previous = WireProgram(older["episode"], *values)
    updated = RevisionWire(
        previous,
        old["file_sha256"],
        new["file_sha256"],
        index - 1,
        newer["curve"],
        newer["limits"],
    )
    end_tick = round(updated.execution_times[-1] / DT)
    ticks = np.arange(end_tick + 1)
    t = ticks * DT
    poses, grip, phase = updated.at(t)
    hashes = updated.source_hash_at(t)
    old_pose, old_grip, old_phase = previous.at(t)
    prefix = slice(0, updated.cut + 1)
    if not all(
        np.array_equal(a[prefix], b[prefix])
        for a, b in zip(
            (poses, grip, phase), (old_pose, old_grip, old_phase), strict=True
        )
    ):
        raise ValueError("changed protected wire")
    new_mask = hashes == new["file_sha256"]
    if np.any(np.diff(phase[new_mask]) < -1e-10) or not np.isclose(phase[-1], END):
        raise ValueError("new local clock incomplete")
    source_motion = np.linalg.norm(
        poses[:, :3, 3] - newer["curve"].knots[0, :3, 3], axis=1
    )
    moving = np.flatnonzero(new_mask & (phase > 0) & (source_motion > 1e-9))
    difference = (
        np.linalg.norm(poses[:, :3, 3] - old_pose[:, :3, 3], axis=1) > 1e-10
    ) | (relative_angle(poses[:, :3, :3], old_pose[:, :3, :3]) > 1e-9)
    width_changed = np.abs(grip - old_grip) > 1e-10
    changed = np.flatnonzero(difference | width_changed)
    if not len(changed) or not len(moving) or changed[0] <= updated.cut:
        raise ValueError("no independently distinguishable tail")
    folder = out / (left + "-" + right)
    folder.mkdir()
    base = np.repeat(
        np.asarray(older["query"]["actual_state"]["base_pose"])[None], 41, axis=0
    )
    wire_dir = folder / "C2_EXACT"
    shadow = mode_run(wire_dir, NewSourcePhaseView(updated), base, "ONE")
    verify_revision(
        previous,
        updated,
        archive / "C2_EXACT/CALLBACKS.private.jsonl",
        wire_dir / "CALLBACKS.private.jsonl",
        index,
        base,
        shadow["installs"],
    )
    failures = read(wire_dir / "FULL40_ENVELOPE.private.json")
    if len(failures) != shadow["original_full40_envelope_rejected_windows"]:
        raise ValueError("missing rejected previews")
    seam = updated.cut + 1
    # V1 G7 uses the target policy's 2..42 mm physical width map.
    widths = 0.002 + 0.040 * grip
    rates = np.abs(np.diff(np.r_[older["context"][-1]["width_m"], widths])) / DT
    source_age = lambda k: float(origin / 10 + k * DT - delta)
    wire_events = [(kind, float(at * DT)) for kind, at in crossing(grip)]
    try:
        updated.check_gripper_handoff()
        gripper_rejection = None
    except ValueError as exc:
        gripper_rejection = str(exc)
    result.update(
        gripper_handoff_rejection=gripper_rejection,
        status="REVISED_IN_SHADOW",
        protected_through_wire_knot=updated.cut,
        first_changed_wire_knot=int(changed[0]),
        age_at_first_wire_change_s=source_age(changed[0]),
        age_at_first_new_source_motion_s=source_age(moving[0]),
        next_source_alignment_s=updated.compiler.lead,
        source_phase_at_old_boundary_s=float(phase[updated.cut]),
        source_phase_at_next_first_knot_s=float(phase[seam]),
        before_next_query=bool(source_age(moving[0]) < 0.5 - 1e-10),
        first_change_within_original_source_horizon=bool(source_age(changed[0]) <= END),
        full40_previews=shadow["installs"],
        rejected_previews=failures,
        seam_g7_mm_s=float(1000 * rates[seam]),
        all_g7_peak_mm_s=float(1000 * max(rates)),
        join_g7_pass=bool(rates[seam] <= 0.04 + 1e-10),
        full_g7_pass=bool(max(rates) <= 0.04 + 1e-10),
        new_source_events=[
            {"kind": kind, "source_s": float(at / 30)}
            for kind, at in newer["curve"].events
        ],
        projected_events=[
            {
                "kind": kind,
                "program_time_s": at,
                "segment": (
                    "old"
                    if at <= updated.cut * DT
                    else "join"
                    if at <= seam * DT
                    else "new"
                ),
            }
            for kind, at in wire_events
        ],
        callback_fields_sha256=shadow["defined_fields_sha256"],
        native_consumption=shadow["native_consumption_receipts"],
    )
    output(folder / "PAIR.json", result)
    return result


def run(root, out):
    from ablation import output
    from audit_sealed import read, pin
    from sealed_panel import cases

    repo = Path(__file__).resolve().parents[2]
    r = repo / "reports/SLA_HIERARCHICAL_V16"
    v4 = read(r / "V4_SLA_SKILL_AND_LONG_HORIZON_DIAGNOSTIC.json")
    decision_file = r / "V1_DECISION_ACTUAL.json"
    pin(decision_file, v4["source_v1_decision_sha256"])
    decision = read(decision_file)
    cohort = {c["key"]: c for c in cases(root, repo)}
    if sorted(cohort) != decision["population"]:
        raise ValueError("incomplete original population")
    calls = primary_calls(root, cohort, v4)
    by_episode = defaultdict(list)
    for c in cohort.values():
        by_episode[c["episode"]].append(c)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for ep in sorted(by_episode):
        ordered = sorted(by_episode[ep], key=lambda c: c["window"]["tick"])
        for older, newer in zip(ordered, ordered[1:]):
            rows.append(
                pair(
                    root,
                    out,
                    older,
                    newer,
                    calls,
                    set(decision["methods"]["C2_EXACT"]["all_numeric_and_geometry"]),
                    set(v4["compiled_and_task_qualified_ids"]["C2_EXACT"]),
                )
            )
    if len(rows) != len(cohort) - len(by_episode):
        raise ValueError("adjacent pair count")
    output(
        out / "H1_REVISION_ACTUAL.json",
        dict(
            calls=calls,
            episodes=len(by_episode),
            pairs=rows,
            original_source_horizon_s=49 / 30,
            real_time_factor_assumption=1,
            new_gpu_forwards=0,
            native_consumed_revisions=0,
            new_native_starts=0,
            new_physics_ticks=0,
            training_updates=0,
        ),
    )


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    configure_site(root, Path(__file__).resolve().parents[2])
    run(root, Path(sys.argv[2]).resolve())
