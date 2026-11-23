"""Play measured 2 ms contacts against the sealed continuous-return receiver messages."""

import argparse
import hashlib
import json
from pathlib import Path

from demonstrated_grip import pin, read, require, rows
from noncontact_owner import NoncontactOwner
from sealed_panel import configure_site


def history(root, ep, metadata, roster, first, last):
    from go2arx5_vla.world_model.multitask_state_v6 import packets

    info = metadata[ep]
    receipt = read(pin(Path(info["receipt_path"]), info["receipt_sha256"]))
    g = roster[ep]
    ef = pin(Path(g["episode_pin"]["path"]), g["episode_pin"]["sha256"])
    n = ef.parents[3] / "cases" / ep / "native-01"
    require(read(ef)["id"] == receipt["id"] == ep, "wrong contact archive")

    def bound(name):
        return pin(n / name, receipt["source_files"][name])

    with bound("physics.jsonl").open() as f:
        h = json.loads(f.readline())
    require(h["kind"] == "model" and h["timestep"] == 0.002, "native scene clock")
    geo = h["geoms"]
    result = {}
    for p in packets(bound("state.bin.journal")):
        if p["step"] < first:
            continue
        if p["step"] > last:
            break
        require(p["step"] not in result, "duplicate packet")
        result[p["step"]] = {
            k: p[k] for k in ("step", "time_s", "contact_time_s", "contacts")
        }
    return geo, result


def reference_events(folder, old, first):
    result = old["rows"]
    origin = first + min(r["publish"] for r in result.values()) * 50
    if not old["callback_installs"]:
        return origin, {}, origin
    digest = hashlib.sha256()
    messages = {}
    cancel = None
    for r in rows(folder / "CALLBACKS.private.jsonl"):
        if r["kind"] == "accept":
            raw = r["raw"]
            h = hashlib.sha256(raw.encode()).hexdigest()
            m = json.loads(raw)["_pi0294"]
            require(
                r["ref_id"] == h
                and m["sequence"] == len(messages)
                and m["operation"] == "install",
                "archived wire identity",
            )
            t = origin + round((m["source_mpc_s"] - 30) / 0.002)
            require(t == origin + len(messages) * 50, "archived 10 Hz publication")
            messages[t] = (raw, h)
            digest.update(bytes.fromhex(h))
        elif r["kind"] == "lifecycle":
            require(
                r["reason"] == "CANCEL" and cancel is None,
                "unexpected original termination",
            )
            cancel = origin + round((r["observation_time_s"] - 30) / 0.002)
    require(
        digest.hexdigest() == old["callback_sha256"]
        and len(messages) == old["callback_installs"]
        and cancel > max(messages),
        "original receiver stream",
    )
    return origin, messages, cancel


def replay_episode(folder, ep, first, data, geo, frames, events, policy):
    from clock_replay import GuardedManager, episode
    from shadow import observe
    from std_msgs.msg import String

    origin, messages, cancel = events
    folder.mkdir()
    installed = []
    terminal_receipts = []
    owner_epoch = ep
    wire_epoch = "C2_EXACT"

    def time_at(tick):
        return 30 + (tick - origin) * 0.002

    with episode(wire_epoch, folder):
        node = GuardedManager()

        def stop(info):
            at = time_at(info["pause_tick"])
            observe(node, at)
            doc = {
                "_pi0294": dict(
                    episode=wire_epoch,
                    sequence=len(installed),
                    source_mpc_s=at,
                    ttl_s=0.4,
                    operation="cancel",
                )
            }
            node.on_trajectory_json(String(data=json.dumps(doc)))
            log = rows(folder / "CALLBACKS.private.jsonl")
            terminal_receipts.extend(
                [r["reason"] for r in log if r["kind"] == "lifecycle"]
            )

        owner = NoncontactOwner.for_policy(policy.stream, first, geo, stop)
        tick = first

        def audit_pause(start, end):
            # An absent packet closes the owner and is retained in its diagnosis.
            owner.after_pause(
                owner_epoch, end, (frames[i] for i in range(start, end + 1))
            )

        audit_pause(first, first)
        while owner.terminal is None:
            if tick == cancel:
                owner.close(owner_epoch, tick, data["stop"])
                break
            policy.capture(tick)
            policy.deliver(tick, first, data["rows"])
            if tick in messages:
                raw, h = messages[tick]

                def install():
                    before = len(node.target_pub)
                    observe(node, time_at(tick))
                    node.on_trajectory_json(String(data=raw))
                    require(
                        len(node.target_pub) == before + 1
                        and node._r4_ref == h
                        and node.trajectory_started,
                        "reference not installed",
                    )

                owner.dispatch(owner_epoch, tick, install)
                installed.append(h)
            nxt = min(tick + 25, cancel)
            owner.grant(owner_epoch, tick, nxt, lambda: None)
            audit_pause(tick + 1, nxt)
            tick = nxt
        # Archive playback uses the real receiver callbacks and its own idle publication timer.
        # Probe stale effects only through the same owner API that issued the earlier messages.
        prevented = []
        for label in ("policy_return", "eef_install", "gripper_issue", "next_grant"):

            def forbidden():
                raise AssertionError("terminal owner dispatched " + label)

            try:
                if label == "next_grant":
                    owner.grant(owner_epoch, tick, tick + 25, forbidden)
                else:
                    owner.dispatch(owner_epoch, tick, forbidden)
            except ValueError as error:
                require(
                    str(error) == "terminal noncontact epoch",
                    "wrong late effect failure",
                )
                prevented.append(label)
        before = len(node.target_pub)
        observe(node, time_at(tick) + 0.1)
        node.publish()
        after = len(node.target_pub) - before
        hold = [m.data for m in node._r4_mode]
        active = node.trajectory_started
        pending = node._r4_pending is not None
        residual = node.trajectory is not None
        binding_cleared = node.bound_start is None and node._r4_deadline is None
        require(
            terminal_receipts == ["CANCEL"] and hold == ["HOLD"],
            "missing actual cancel/HOLD",
        )
        log = rows(folder / "CALLBACKS.private.jsonl")
        installed_logs = [r["ref_id"] for r in log if r["kind"] == "installed"]
        require(installed_logs[: len(installed)] == installed, "wire prefix mismatch")
    boundary = owner.terminal["pause_tick"]
    effect = {}
    for k, r in data["rows"].items():
        q = first + round(r["capture_s"] / 0.002)
        decision = first + r["publish"] * 50
        motion = r["first_nonzero_source_motion_s"]
        due = None if motion is None else q + round(motion / 0.002)
        effect[k] = dict(
            original_status=r["status"],
            observed_before_stop=q < boundary,
            return_before_stop=(first + r["return_s"] / 0.002) < boundary,
            admitted_before_stop=r["status"] == "ADMITTED" and decision < boundary,
            original_first_source_tick=due,
            source_before_stop=due is not None and due < boundary,
        )
    return dict(
        first_query_tick=first,
        original_install_count=len(messages),
        actual_prefix_installs=len(installed),
        prefix_sha256=hashlib.sha256(
            b"".join(bytes.fromhex(h) for h in installed)
        ).hexdigest(),
        stop=owner.terminal,
        original_terminal_tick=cancel,
        grant_tail_ticks=(boundary - owner.terminal["observed_tick"])
        if "observed_tick" in owner.terminal
        else None,
        stopped_effects=prevented,
        policy=policy.check_terminal(),
        receiver_hold=hold,
        receiver_cancel=terminal_receipts,
        receiver_trajectory_started_after_cancel=active,
        receiver_residual_trajectory=residual,
        receiver_binding_and_deadline_cleared=binding_cleared,
        receiver_pending_after_cancel=pending,
        callback_publications_after_cancel=after,
        installed_receipts_after_prefix=installed_logs[len(installed) :],
        proposals=effect,
    )


def v2_control(root, repo):
    import mujoco
    from verify_grip_card import states

    audit = read(
        pin(
            repo / "reports/SLA_HIERARCHICAL_V16/V2_STUDENT_GRIPPER_OWNER_ACTUAL.json",
            "bbcfef7618d3df51be8261a4fa55e7e66323efd81063f6f9e6ff654a62003125",
        )
    )
    hashes = audit["source_sha256"]
    case = root / "v16/sla-v16-student-gripper-native-01b/V2-GRIP-01B"
    native = (
        case
        / read(pin(case / "RESULT.private.json", hashes["RESULT.private.json"]))[
            "episode"
        ]
    )
    pin(native / "state.bin.journal", hashes["state.bin.journal"])
    spec = read(pin(case / "EXECUTION_SPEC.json", hashes["EXECUTION_SPEC.json"]))
    m = Path(spec["model_path"])
    model = mujoco.MjModel.from_binary_path(str(pin(m, spec["frozen_files"][str(m)])))
    with (native / "physics.jsonl").open() as f:
        geom = json.loads(f.readline())["geoms"]
    require(
        len(geom) == model.ngeom
        and all(
            g["id"] == i
            and g["name"] == (model.geom(i).name or "")
            and g["body"] == (model.body(model.geom_bodyid[i]).name or "")
            for i, g in enumerate(geom)
        ),
        "V2 native geometry map",
    )
    intervals = audit["verified_command_intervals"]
    first = intervals[0]["source_tick"]
    end = intervals[-1]["end_tick"]
    packets = (p for p in states(native) if p["step"] >= first)
    reasons = []
    owner = NoncontactOwner(native.name, first, geom, reasons.append)
    owner.after_pause(native.name, first, [next(packets)])
    for r in intervals:
        a, b = r["source_tick"], r["end_tick"]
        require(
            b == a + 25
            and r["applied_native_steps"] == 25
            and r["receiver_ack_tick"] == a,
            "V2 actual consumption",
        )
        owner.grant(native.name, a, b, lambda: None)
        owner.after_pause(native.name, b, (next(packets) for _ in range(25)))
    require(
        next(packets, None) is None
        and not reasons
        and len(intervals) == 160
        and end == 6503,
        "V2 false stop or gap",
    )
    owner.close(native.name, end, "ORIGINAL_CAP")
    return dict(
        first_tick=first,
        last_tick=end,
        observed_packets=end - first + 1,
        verified_grants=len(intervals),
        false_stops=0,
        original_cap=reasons[0],
    )


def main(root, repo, out):
    from noncontact_policy_return import ArchivedPolicy

    archive = root / "v16/continuous-revisions03"
    record = read(
        pin(
            archive / "ACTUAL.json",
            "eaa766bdaf60aac80e00cb36d2c970f81d768a27104b975acf00ee6a1b78be46",
        )
    )["episodes"]
    s0 = read(
        pin(
            root / "v13/run01/S0_ACTUAL.json",
            "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9",
        )
    )["inputs"]
    capture = root / "v12/A2_O2_DENSE_CAPTURE.json"
    cap = read(pin(capture, s0[str(capture)]))
    meta = {r["id"]: r for r in cap["episodes"]}
    roster = {
        g["episode"]: g
        for g in read(
            pin(
                root / "evidence/P1_REGISTRATION.json",
                "1d496553b347a43c6f5c45c99d736896060ca9d71bf6cd5a72fef3e767b1b900",
            )
        )["target_windows"]
    }
    calls = read(
        pin(
            repo / "reports/SLA_HIERARCHICAL_V16/H1_REVISION_ACTUAL.json",
            "855860708527c0a2b0d86b5ad710ea94da7201ba89e02bca5c41e08518ca206c",
        )
    )["calls"]
    out.mkdir()
    results = {}
    for ep, conditions in record.items():
        first = min(r["tick"] for r in calls.values() if r["episode"] == ep)
        events = {
            k: reference_events(archive / (ep + "-" + k), v, first)
            for k, v in conditions.items()
        }
        geom, frames = history(
            root, ep, meta, roster, first, max(e[2] for e in events.values())
        )
        receipt = read(pin(Path(meta[ep]["receipt_path"]), meta[ep]["receipt_sha256"]))
        results[ep] = {
            k: replay_episode(
                out / (ep + "-" + k),
                ep,
                first,
                v,
                geom,
                frames,
                events[k],
                ArchivedPolicy(root, cap, ep, receipt, calls, s0),
            )
            for k, v in conditions.items()
        }
    result = dict(
        episodes=results,
        closed_gripper_control=v2_control(root, repo),
        native_starts=0,
        new_forwards=0,
        training_updates=0,
    )
    with (out / "ACTUAL.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[2]
    configure_site(root, repo)
    main(root, repo, args.out.resolve())
