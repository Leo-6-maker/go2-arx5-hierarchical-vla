"""Qualify and replay a sequence of frozen returns without resetting between adjacent pairs."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from sealed_panel import configure_site

if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parents[2]
    configure_site(root, repo)
    sys.path.insert(0, str(root / "v16/cpu-fcl"))

from demonstrated_grip import read, pin, require
from compiler import DT
from gripper_alignment import alignment_widths
from handoff_factors import contact, gates, initial
from prefix_replay import WireProgram
from revision_wire import RevisionWire, first_publication


def timeline(calls, single_worker):
    origin = calls[0]["tick"]
    busy_until = 0.0
    result = []
    for call in calls:
        captured = (call["tick"] - origin) * 0.002
        start = max(captured, busy_until) if single_worker else captured
        done = start + call["seconds"]
        busy_until = done
        result.append(
            dict(
                id=call["id"],
                capture_s=captured,
                queue_s=start - captured,
                return_s=done,
                publish=first_publication(done),
            )
        )
    return result


def qualify(previous, old, new, sequence, first, scene):
    from reachability import check_wire
    from retime import crossing

    revised = RevisionWire(
        previous,
        old["raw_file_sha256"],
        new["raw_file_sha256"],
        sequence - 1,
        new["curve"],
        new["limits"],
    )
    pos, g, phase = revised.future.arrays
    local_t = np.arange(len(g)) * DT
    cut = revised.cut
    before = float(previous.at(np.array([cut * DT]))[1][0])
    aligned, entry = alignment_widths(
        local_t,
        phase,
        g,
        0.002 + 0.040 * before,
        revised.compiler.lead,
        contact(old) or contact(new),
    )
    evidence = dict(cut=cut, width_alignment=entry)
    if aligned is None:
        return None, dict(**evidence, reason="WIDTH_ALIGNMENT")
    revised.future = WireProgram(revised.identity, pos, aligned, phase)
    revised.check_gripper_handoff()
    stop = revised.execution_times[-1]
    t = np.arange(round(stop / DT) + 1) * DT
    poses, width, phases = revised.at(t)
    prefix = t[: cut + 1]
    for prior, updated in zip(previous.at(prefix), (poses, width, phases), strict=True):
        require(np.array_equal(prior, updated[: cut + 1]), "protected samples differ")
    if isinstance(previous, RevisionWire):
        require(
            np.array_equal(
                revised.source_hash_at(prefix), previous.source_hash_at(prefix)
            ),
            "protected source lineage differs",
        )
    preview_knots = 2 * sequence + np.arange(41)
    ids = revised.source_hash_at(preview_knots * DT)
    labels, counts = np.unique(ids, return_counts=True)
    evidence["first_preview_source_counts"] = dict(zip(labels, map(int, counts)))
    single_parent = np.where(preview_knots > cut, revised.next_hash, revised.prior_hash)
    evidence["first_preview_misattributed_with_single_parent"] = int(
        np.count_nonzero(ids != single_parent)
    )
    count = math.ceil(stop / 0.1 - 1e-10) + 1
    checks = gates(revised, first["context"][-1]["width_m"], count)
    actual = [
        (e["kind"], e["time"])
        for e in checks["wire_events"]
        if e["time"] > cut * DT + 1e-10
    ]
    expected = [(k, (cut + i) * DT) for k, i in crossing(aligned)]
    events_ok = (
        [e[0] for e in actual] == [e[0] for e in new["curve"].events]
        and len(actual) == len(expected)
        and all(
            k == l and abs(x - y) < 1e-8 for (k, x), (l, y) in zip(actual, expected)
        )
    )
    evidence.update(
        h40_previews=count,
        h40_refusals=checks["envelopes"],
        max_width_rate_mm_s=checks["maximum_width_rate_mm_s"],
        g7=checks["g7"],
        events_preserved=events_ok,
        source_events=[e[0] for e in new["curve"].events],
    )
    if not events_ok or not checks["g7"] or any(checks["envelopes"].values()):
        return None, dict(**evidence, reason="EVENT_OR_ENVELOPE")
    geometry = check_wire(scene, first, t, poses, width)
    evidence["geometry"] = geometry
    if not geometry["pass"]:
        return None, dict(**evidence, reason="DISCRETE_GEOMETRY")
    return revised, dict(**evidence, reason=None)


def at_sources(plan, t, root_hash):
    return (
        plan.source_hash_at(t)
        if isinstance(plan, RevisionWire)
        else np.repeat(root_hash, len(t))
    )


def episode_replay(
    folder, queries, schedule, first_plan, first_result, scene, pairs, archive
):
    from clock_replay import episode, GuardedManager
    from governor import PersistentReference
    from ocs2_target_adapter import wire
    from revision_replay import receipts
    from shadow import observe, check as check_message
    from std_msgs.msg import String
    from go2arx5_vla.trajectory.so3 import quat_to_matrix

    folder.mkdir()
    epoch = "C2_EXACT"
    clock = 30.0
    start = schedule[0]["publish"]
    returns = {r["publish"] - start: r["id"] for r in schedule}
    require(
        len(returns) == len(schedule) and list(returns) == sorted(returns),
        "unordered study returns",
    )
    first = queries[schedule[0]["id"]]
    active = first
    rows = {
        r["id"]: dict(
            **r,
            status="NOT_REACHED",
            current_knots=0,
            first_nonzero_source_motion_s=None,
            max_scheduled_source_phase_s=None,
            lookahead_contains_source_motion=False,
        )
        for r in schedule
    }
    root_hash = first["raw_file_sha256"]
    hash_to_key = {c["raw_file_sha256"]: k for k, c in queries.items()}
    proof = hashlib.sha256()
    result = dict(
        rows=rows,
        stop=None,
        callback_installs=0,
        protected_pair_receipt_matches=0,
        terminal_hold=False,
        native_consumptions=0,
    )
    if not first_result["aligned_qualified"]:
        rows[active["key"]].update(
            status="INITIAL_REJECTED", first_width_evidence=first_result["alignment"]
        )
        result["stop"] = "INITIAL_REJECTED"
        return result
    require(first_plan is not None, "accepted missing plan")
    rows[active["key"]]["status"] = "ADMITTED"
    base = np.repeat(
        np.asarray(first["query"]["actual_state"]["base_pose"])[None], 41, axis=0
    )
    mgr = PersistentReference(first_plan, epoch, clock)
    cursor = 0
    i = 0
    has_pair_oracle = False
    with episode(epoch, folder):
        site = GuardedManager()
        while True:
            now = clock + i * 0.1
            if i and i in returns:
                nxt = queries[returns[i]]
                row = rows[nxt["key"]]
                pair = pairs[active["key"], nxt["key"]]
                row["standalone_pair_admitted"] = pair["geometric_and_numeric_pass"]
                candidate, evidence = qualify(mgr.intent, active, nxt, i, first, scene)
                row["revision"] = evidence
                if candidate is None:
                    row["status"] = "REVISION_REJECTED"
                    result["stop"] = "REVISION_REJECTED"
                    break
                mgr.revise(candidate, now, epoch)
                active = nxt
                row["status"] = "ADMITTED"
                if mgr.revision == 1 and pair["geometric_and_numeric_pass"]:
                    old_file = (
                        archive
                        / (first["key"] + "-" + nxt["key"] + "-FULL-ALIGNED")
                        / epoch
                        / "CALLBACKS.private.jsonl"
                    )
                    historical = receipts(old_file)
                    # The earlier, already issued first-source publications have been sent.
                    for _ in range(i):
                        next(historical)
                    has_pair_oracle = True
                else:
                    has_pair_oracle = False
            key, (poses, width, phase) = mgr.window(now)
            raw = wire(poses, base, epoch, key[1], now)
            ident = hashlib.sha256(raw.encode()).hexdigest()
            observe(site, now)
            site.on_trajectory_json(String(data=raw))
            msg = site.target_pub[-1]
            values = np.asarray(check_message(msg, 14))
            require(
                len(values) == 41
                and np.allclose(values[:, 0:3], poses[:, :3, 3], atol=2e-7, rtol=0)
                and np.allclose(values[:, 7:10], base[:, :3, 3], atol=2e-7, rtol=0)
                and np.allclose(
                    quat_to_matrix(values[:, 3:7]), poses[:, :3, :3], atol=7e-7, rtol=0
                )
                and np.allclose(
                    quat_to_matrix(values[:, 10:14]), base[:, :3, :3], atol=7e-7, rtol=0
                )
                and np.allclose(
                    msg.time_trajectory, now + np.arange(41) * DT, atol=1e-8, rtol=0
                ),
                "callback positions or clocks",
            )
            with (folder / "CALLBACKS.private.jsonl").open() as f:
                f.seek(cursor)
                trace = [json.loads(line) for line in f]
                cursor = f.tell()
            ack = next(
                x for x in trace if x["kind"] == "accept" and x["ref_id"] == ident
            )
            installed = next(
                x for x in trace if x["kind"] == "installed" and x["ref_id"] == ident
            )
            mgr.acknowledge(key, ident, ack, installed)
            proof.update(bytes.fromhex(ident))
            if has_pair_oracle:
                h, _ = next(historical)
                require(
                    h["ref_id"] == ident,
                    "continuous wire differs from independent first-pair callbacks",
                )
                result["protected_pair_receipt_matches"] += 1
            site.target_pub.clear()
            elapsed = (2 * i + np.arange(41)) * DT
            sources = at_sources(mgr.intent, elapsed, root_hash)
            end = (
                mgr.intent.execution_times[-1]
                if isinstance(mgr.intent, RevisionWire)
                else DT * (len(first_plan.arrays[0]) - 1)
            )
            for j, (h, p) in enumerate(zip(sources, phase, strict=True)):
                name = hash_to_key[h]
                r = rows[name]
                moved = (
                    p > 0
                    and np.linalg.norm(
                        poses[j, :3, 3] - queries[name]["curve"].knots[0, :3, 3]
                    )
                    > 1e-9
                )
                if moved:
                    r["lookahead_contains_source_motion"] = True
                if j > 1 or elapsed[j] > end + 1e-9:
                    continue
                r["current_knots"] += 1
                r["max_scheduled_source_phase_s"] = max(
                    float(p), r["max_scheduled_source_phase_s"] or 0.0
                )
                if moved and r["first_nonzero_source_motion_s"] is None:
                    r["first_nonzero_source_motion_s"] = float(
                        start / 10 + elapsed[j] - r["capture_s"]
                    )
            result["callback_installs"] += 1
            i += 1
            if i - 1 >= math.ceil(end / 0.1 - 1e-10) and i > max(returns):
                result["stop"] = "END_OF_FINAL_REFERENCE"
                break
        cancel_time = now if result["stop"] == "REVISION_REJECTED" else now + 0.09
        cancel = {
            "_pi0294": dict(
                episode=epoch,
                sequence=i,
                source_mpc_s=cancel_time,
                ttl_s=0.4,
                operation="cancel",
            )
        }
        observe(site, cancel_time)
        site.on_trajectory_json(String(data=json.dumps(cancel)))
        require(
            [m.data for m in site._r4_mode] == ["HOLD"] and mgr.consumed is None,
            "offline terminal HOLD or consumption",
        )
        result.update(
            terminal_hold=True,
            callback_sha256=proof.hexdigest(),
            revisions=mgr.revision,
        )
        mgr.abort()
    return result


def run(root, repo, out):
    from frozen_geometry import Scene
    from revision_replay import primary_calls
    from sealed_panel import cases

    reports = repo / "reports/SLA_HIERARCHICAL_V16"
    h = read(
        pin(
            reports / "HANDOFF_FACTORS_ACTUAL.json",
            "c0b6a25270e793078482ba3416a1c795d2d88277d1cc38a61f447f9fc2ac610b",
        )
    )
    old = read(
        pin(
            reports / "H1_REVISION_ACTUAL.json",
            "855860708527c0a2b0d86b5ad710ea94da7201ba89e02bca5c41e08518ca206c",
        )
    )
    archive = root / "v16/handoff-factors05"
    require(read(archive / "HANDOFF_FACTORS_ACTUAL.json") == h, "site history differs")
    pop = {r["key"]: r for r in cases(root, repo)}
    calls = primary_calls(
        root, pop, read(reports / "V4_SLA_SKILL_AND_LONG_HORIZON_DIAGNOSTIC.json")
    )
    require(len(calls) == 16 and calls == old["calls"], "different model calls")
    pairs = {
        (x["old"], x["new"]): x
        for x in h["pairs"]
        if x["protection"] == "FULL" and x["alignment"]
    }
    initial_rows = {x["id"]: x for x in h["initial"]}
    ablations = {
        r["id"]: r["modes"]["C2_EXACT"]
        for r in read(reports / "V1_COMPILER_ABLATION_ACTUAL.json")["windows"]
    }
    groups = defaultdict(list)
    for call in calls.values():
        groups[call["episode"]].append(call)
    require(len(groups) == 6 and len(pairs) == 10, "unexpected episode denominator")
    out.mkdir()
    result = {}
    scenes = {}
    for ep, group in groups.items():
        group.sort(key=lambda x: x["tick"])
        require(
            all(b["tick"] - a["tick"] == 250 for a, b in zip(group, group[1:])),
            "nonadjacent model observations",
        )
        first = pop[group[0]["id"]]
        path = Path(first["spec"]["model_path"])
        if path not in scenes:
            scenes[path] = Scene(
                path,
                root / "v16/scene_layout",
                first["spec"]["frozen_files"][str(path)],
            )
        original = initial_rows[first["key"]]
        init, (_, plan) = initial(
            root,
            first,
            scenes[path],
            ablations[first["key"]],
            original["raw_geometry"],
            original["phase_proxy"],
        )
        require(init == original, "first query qualification differs")
        result[ep] = {}
        for label, single in (("fifo", True), ("overlap", False)):
            entries = timeline(group, single)
            result[ep][label] = episode_replay(
                out / (ep + "-" + label),
                pop,
                entries,
                plan,
                init,
                scenes[path],
                pairs,
                archive,
            )
    with (out / "ACTUAL.json").open("x") as f:
        json.dump(
            dict(
                episodes=result, new_forwards=0, new_native_starts=0, training_updates=0
            ),
            f,
            indent=2,
            allow_nan=False,
        )
        f.write("\n")


if __name__ == "__main__":
    run(root, repo, Path(sys.argv[2]).resolve())
