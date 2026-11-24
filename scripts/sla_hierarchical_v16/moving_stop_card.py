"""Owner callback for one registered V3 A0 moving-expiry card. CLI runs preflight or scores a closed case."""

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import secrets
import sys
import traceback

BASE = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(BASE / "scripts" / d)
    for d in ("sla_transfer_v14", "sla_transfer_v15", "sla_transfer_feedback")
]
from b2_owner import baseline, check_occupancy, configure, dry_receiver, verify
from identity import read, record, sha
from stop_card_evidence import require

REPORTS = BASE / "reports/SLA_HIERARCHICAL_V16"
CARD = "V3-STOP-01"
FAULT_GRANT = 6453
STOP = 6439


def inputs(root):
    from d1_readiness import pinned
    from d1_factorial_reference_matrix import verify_initial
    from native_known import target
    from fault_reference import motion_from_install
    from go2arx5_vla.world_model.init_trace_reader import audit_trace
    from scripts.wm02 import mobile_boundary_runner_v1 as mobile
    from verify_grip_card import states

    # Fail import/overlay and ABI errors during the no-start entry point.
    require(
        all(
            map(
                callable,
                [
                    verify_initial,
                    target,
                    motion_from_install,
                    audit_trace,
                    mobile.registered,
                    states,
                ],
            )
        ),
        "incomplete runtime",
    )
    base = baseline(root)
    spec = read(
        root
        / "v14/qualification_release/technical_registration/B2_TECHNICAL_REGISTRATION.json"
    )
    runtime = verify(base, spec)
    dry_receiver(base)
    plan = read(REPORTS / "V3_NATIVE_STOP_CARD_CANDIDATE.json")
    require(
        (
            plan["id"],
            plan["max_native_starts"],
            plan["advertised_hard_cap_ticks"],
            plan["maximum_grant_end_tick"],
            plan["planned_stop_tick"],
        )
        == (CARD, 1, 6503, 6453, 6439),
        "card clock drift",
    )
    source = read(REPORTS / "V3_STOP_TRACE_ACTUAL.json")["bridge"]
    receiver = root / "v16/v3-stop-trace02/BRIDGE/receiver/student_grip.so"
    pinned(receiver, source["receiver_sha256"])
    pinned(Path(base["trace_library"]), source["frozen_trace_sha256"])
    match = root / "v15/readiness01/D1_FACTORIAL_4X2_REGISTER.json"
    pinned(match, "c69cea0c14cf4390d369a5fec9efc44ec29ecfe273df44326f2f39fffb42578a")
    match_reg = read(match)
    anchors = sorted(match.parent.glob("*_anchor.private.json"))
    require(
        sorted(map(sha, anchors)) == sorted(match_reg["prior_anchor_sha256"]),
        "both original anchor files required",
    )
    return (
        base,
        runtime,
        receiver,
        [read(p)["initial_features"] for p in anchors],
        match_reg["pairwise_initial_limits"],
    )


def groups_alive(group_ids):
    found = set()
    for p in Path("/proc").glob("[0-9]*/stat"):
        try:
            tail = p.read_text(errors="replace").rsplit(")", 1)[1].split()
            group = int(tail[2])
        except (FileNotFoundError, ProcessLookupError):
            continue
        if group in group_ids:
            found.add((int(p.parent.name), group))
    return sorted(found)


def receiver_adapter(
    base,
    spec,
    case,
    out,
    channel,
    epoch,
    result,
    *,
    grant_cap,
    fault_grant=None,
    fault_expiry=None,
):
    from b2_native import TechnicalAdapter, paused
    from go2arx5_vla.world_model.target_sequential import owned_simulator_pid
    from grip_card import emit

    class Adapter(TechnicalAdapter):
        def launch(self, argv, name, env):
            if name == "sim":
                env = dict(
                    env,
                    V16_GRIP_DIR=str(channel),
                    V16_GRIP_EPOCH=epoch,
                    V16_GRIP_STOP_TICK=str(spec["max_ticks"]),
                )
            super().launch(argv, name, env)

        def advance_to(self, ticks):
            if ticks > 2503:
                require(
                    result["last_requested_tick"] < ticks <= grant_cap,
                    "no subsequent native grant",
                )
                tick = ticks - 25
                seq = (tick - 2503) // 25
                require(
                    tick == 2503 + 25 * seq
                    and not any(
                        (p / "STOP.json").exists()
                        for p in (case.parent, Path(spec["campaign_root"]))
                    ),
                    "stop or illegal step",
                )
                before = paused(self, tick)
                if seq == 0:
                    pid = owned_simulator_pid(
                        next(p.pid for n, p in self.processes if n == "sim"),
                        spec["native_binary"],
                    )
                    result["sim_child_pid"] = pid
                    proc_maps = (
                        (Path("/proc") / str(pid) / "maps").read_text().splitlines()
                    )
                    images = {}
                    for line in proc_maps:
                        fields = line.split(None, 5)
                        if len(fields) == 6 and fields[5].startswith("/"):
                            images[fields[5]] = (fields[3], int(fields[4]))
                    loaded = {}
                    for k in (
                        "outer_library",
                        "trace_library",
                        "observer_library",
                        "mobile_safety_library",
                    ):
                        path = Path(spec[k])
                        st = path.stat()
                        device = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}"
                        require(
                            images.get(str(path)) == (device, st.st_ino),
                            "unverified loaded image: " + k,
                        )
                        loaded[k] = sha(path)
                        require(
                            loaded[k] == spec["frozen_files"][str(path)],
                            "changed loaded library",
                        )
                    require(
                        base["outer_library"] not in images
                        and self.latest["mode"].data == "WBC",
                        "owner or WBC mode",
                    )
                    result["loaded_at_command_reset"] = loaded
                    self.capture_loaded_identity()
                elif seq > 0:
                    with (channel / "receiver.csv").open("rb") as f:
                        f.seek(max(0, os.fstat(f.fileno()).st_size - 1024))
                        tail = f.read().decode().splitlines()[-1].split(",")
                    require(
                        tail[0] == "NATIVE_STEP"
                        and tail[1] == epoch
                        and int(tail[2]) == tick
                        and int(tail[4]) == seq - 1,
                        "missing actual prior gripper consumption",
                    )
                emit(
                    channel,
                    epoch,
                    seq,
                    before["time_s"],
                    opening_m=0.042,
                    expiry_s=fault_expiry
                    if ticks == fault_grant
                    else before["time_s"] + 0.399,
                )
            result["last_requested_tick"] = ticks
            return super().advance_to(ticks)

    return Adapter(spec, out)


def run_case(root, spec, sp, prepared):
    """Called by the allocation owner inside its exclusive world lock and active case lease."""
    from b2_native import paused
    from clock_guard import new_receiver_output
    from d3_known_path_preview import sweep_path
    from go2arx5_vla.world_model.mobile_safety_profile_v1 import binding
    from go2arx5_vla.world_model.mobile_wbc_reference_v1 import full_eef_gate
    from scripts.pi0291_arx5_data.native_dataset import pose10
    from feedback import OPEN
    import numpy as np
    from native_known import atom_receipt, install, read_line, window
    from native_owner import eligibility
    from scripts.wm02 import mobile_boundary_runner_v1 as mobile

    case = sp.parent
    lease = mobile.registered(spec, sp)
    base, pins, receiver, anchors, limits = prepared
    require(
        case.name == CARD
        and spec["b2_card"] == CARD
        and spec["b2_code"] == pins
        and spec["max_ticks"] == 6503
        and Path(spec["outer_library"]) == receiver
        and Path(spec["registered_root"]) == case
        and spec["frozen_files"].get(str(receiver)) == sha(receiver),
        "no matching V3 allocation",
    )
    for k in (
        "native_binary",
        "solver_binary",
        "controller_binary",
        "model_path",
        "mobile_safety_library",
        "observer_library",
        "trace_library",
        "source_sha",
        "source_root",
        "ROS_DOMAIN_ID",
        "physics_dt_s",
    ):
        require(spec[k] == base[k], "runtime differs: " + k)
    require(
        all(spec["frozen_files"].get(p) == v for p, v in base["frozen_files"].items()),
        "freeze manifest differs",
    )
    require(
        not (case / "NATIVE_REQUEST.json").exists(), "existing request on this lease"
    )
    out = new_receiver_output(case)
    channel = case / "student"
    channel.mkdir(mode=0o700)
    epoch = secrets.token_hex(32)
    record(
        channel / "authority.json",
        dict(
            epoch=epoch,
            owner=lease["owner"],
            episode=out.name,
            receiver_sha256=sha(receiver),
            lease_at_start_sha256=sha(case / "control/LEASE.json"),
        ),
    )
    result = dict(
        card=CARD,
        episode=out.name,
        eligible=False,
        outcome="TECHNICAL_STOP",
        last_requested_tick=0,
    )
    record(case / "NATIVE_REQUEST.json", dict(card=CARD, cap=6503, episode=out.name))

    a = receiver_adapter(
        base,
        spec,
        case,
        out,
        channel,
        epoch,
        result,
        grant_cap=FAULT_GRANT,
        fault_grant=FAULT_GRANT,
        fault_expiry=6438.5 * 0.002,
    )
    try:
        a.start()
        parts, sample, seq, _, match = eligibility(a, case, anchors, limits, None)
        result.update(eligible=match["matched"], violations=match["violations"])
        if not match["matched"]:
            paused(a, 5703)
            result["cancel"] = a.cancel_reference()
            a.seal_trace(5703)
            result["outcome"] = "SCREENED_OUT"
        else:
            sweep = sweep_path(
                a.actual_state.model,
                sample,
                parts,
                "translation",
                "slow",
                binding(spec["native_root"]),
            )
            record(case / "SWEEP.private.json", sweep)
            for atom in range(30):
                tick = 5703 + 25 * atom
                if atom % 2 == 0:
                    preview = window(parts["old_ee"], atom * 0.05)
                    full_eef_gate(
                        pose10(preview[:40], np.full(40, OPEN)), sample["tcp"]
                    )
                    action, wire = install(
                        a,
                        preview,
                        parts["old_base"],
                        seq + 1 + atom // 2,
                        tick,
                    )
                    source = tick
                sample, _ = mobile.consume(a, action, sequence=128 + atom)
                last = read_line(out / "MOBILE_SAMPLES.private.jsonl")
                receipt = atom_receipt(out, last, sample, source, wire)
                require(receipt["end"] == tick + 25, "incomplete consumption")
            raise RuntimeError("fault grant completed past stop tick")
    except BaseException:
        # A failed full-atom receipt is expected at the partial grant. Only the independent scorer can qualify it.
        result["interrupt"] = traceback.format_exc()
    finally:
        try:
            a.close()
        except BaseException:
            result["close_error"] = traceback.format_exc()
        if (
            result["outcome"] == "TECHNICAL_STOP"
            and result["last_requested_tick"] == FAULT_GRANT
        ):
            result["outcome"] = "FAULT_GRANT_AWAITING_INDEPENDENT_SCORE"
        result["owned_exits"] = a.exit_records
        groups = {p.pid for _, p in a.processes}
        survivors = groups_alive(groups)
        result["owned_group_survivors"] = survivors
        result["owned_processes_absent"] = (
            not survivors
            and bool(result.get("sim_child_pid"))
            and not (Path("/proc") / str(result["sim_child_pid"])).exists()
        )
        record(case / "RESULT.private.json", result)
    return result


def main():
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("root", type=Path)
    arg.add_argument("--score", type=Path, metavar="CLOSED_SPEC")
    arg.add_argument(
        "--shadow",
        action="store_true",
        help="check the actual issuer and historical ROS payloads on CPU",
    )
    a = arg.parse_args()
    root = a.root.resolve()
    configure(baseline(root))
    if a.score:
        from stop_card_evidence import score

        print(json.dumps(score(root, a.score.resolve()), indent=2, allow_nan=False))
    else:
        from scripts.pi0292.durable import OwnerLock
        from stop_card_evidence import score

        require(callable(score), "scorer missing")
        base, pins, receiver, anchors, _ = inputs(root)
        with closing(OwnerLock(Path(base["mobile_world_lock"]))):
            check_occupancy(base)
        result = dict(
            frozen_receiver_sha256=sha(receiver),
            anchor_count=len(anchors),
            runtime_code_files=len(pins),
            exclusive_check_released=True,
            new_native_starts=0,
            physical_resource_grant=False,
        )
        if a.shadow:
            from stop_card_shadow import run

            result["shadow"] = run(root, BASE)
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
