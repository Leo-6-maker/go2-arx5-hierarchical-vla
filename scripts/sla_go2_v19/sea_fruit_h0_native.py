"""Standing H0 task driver for the registered paused native execution owner."""

import csv
import json
import math

import numpy as np
from scipy.spatial.transform import Rotation

from b2_native import fixed, paused
from d1_factorial_reference_matrix import boundary as startup_boundary
from demonstrated_grip import require
from identity import read, record
from native_known import read_line
from standing_reach import StateMonitor
from sea_fruit_h0_score import StationaryHoldScore

from go2arx5_vla.world_model.fusion_trajectory_adapter import MECHANICAL_TRAVEL_M, WORKING_OPEN_M
from go2arx5_vla.world_model.mobile_boundary_clock_v1 import AcceptedBoundaryIndex
from go2arx5_vla.world_model.mobile_grasp_budget_v3 import MAX_QUERIES, StageClock
from go2arx5_vla.world_model.multitask_state_v6 import read_journal_packet
from go2arx5_vla.world_model.multitask_teacher_v6 import MultitaskTeacher
from scripts.wm02 import mobile_boundary_runner_v1 as mobile

INITIAL_TICK = 5703
STABLE_DEADLINE = 15500
ATOM_TICKS = 25


def grip_at_grant(adapter, tick):
    """Use the real admitted H40 at this paused tick, once per native grant."""
    offset = tick - adapter.mobile_installed_step
    require(offset >= 0 and offset % ATOM_TICKS == 0, "gripper not on the installed 50 ms grid")
    row = offset // ATOM_TICKS
    action = adapter.active_reference["action_EEF10"]
    require(row + 1 < len(action), "gripper grant crosses the admitted reference endpoint")
    fraction = float(action[row][9])
    require(math.isfinite(fraction) and .002 / MECHANICAL_TRAVEL_M <= fraction <=
            WORKING_OPEN_M / MECHANICAL_TRAVEL_M, "working gripper stroke out of range")
    if adapter.mobile_contact_stage == "approach":
        require(fraction == WORKING_OPEN_M / MECHANICAL_TRAVEL_M, "approach requires open gripper")
    return MECHANICAL_TRAVEL_M * fraction


def gripper_receipt(channel, epoch, sequence, start, width):
    with (channel / "receiver.csv").open("rb") as stream:
        size = stream.seek(0, 2)
        stream.seek(max(0, size - 16384))
        if size > 16384:
            stream.readline()
        rows = list(csv.reader(stream.read().decode().splitlines()))
    end = start + ATOM_TICKS
    acks = [r for r in rows if r[0] == "RECEIVER_ACK" and int(r[2]) == start]
    require(len(acks) == 1 and rows[-1][0] == "NATIVE_STEP" and int(rows[-1][2]) == end,
            "native gripper ACK or consumed tick missing")
    for entry in (acks[0], rows[-1]):
        require(entry[1] == epoch and int(entry[4]) == sequence and entry[3] == "1"
                and abs(float(entry[5]) - start*.002) <= 1e-8
                and abs(float(entry[6])*MECHANICAL_TRAVEL_M - width) <= 1e-12,
                "native gripper ACK identity, clock or physical opening mismatch")
    return dict(native_step=end, sequence=sequence, width_m=width,
                two_measured_finger_slide_m=[float(x) for x in rows[-1][10:12]])


def fixed_base_install(adapter, action, fixed_base):
    mobile.install(adapter, action, fixed_base)
    wire = adapter.active_reference["wire_sha256"]
    events = [e for e in adapter.active_reference["receiver_events"]
              if e["kind"] == "installed" and e["ref_id"] == wire]
    verify_fixed_base(events, fixed_base[0])
    return wire


def verify_fixed_base(events, hold):
    require(events, "fixed base reference not installed")
    for event in events:
        rows = np.array(event["poses"])
        times = np.asarray(event["times"])
        # The original 41-knot planner interleaves the live clock into its
        # remaining horizon. The startup and rolled publications can differ in count.
        require(rows.shape == (len(times), 14) and 41 <= len(times) <= 81 and np.isfinite(rows).all()
                and times.shape == (len(rows),) and np.isfinite(times).all()
                and np.all(np.diff(times) > 0) and max(np.diff(times)) <= .05000001
                and np.allclose(rows[:, 7:10], hold[:3, 3], rtol=0, atol=1e-7)
                and np.allclose(Rotation.from_quat(rows[:, 10:14]).as_matrix(),
                                hold[:3, :3], rtol=0, atol=2e-7),
                "installed BASE differs from the original world hold")


def score_through(stream, score, end):
    """Read the exact contiguous measured clock through the already paused atom."""
    while score.last is None or score.last < end:
        packet = read_journal_packet(stream)
        require(packet is not None and (score.last is None or packet["step"] == score.last + 1)
                and packet["step"] <= end, "missing physical HOLD clock")
        score.update_packet(packet)
    require(score.last == end, "score exceeds the paused clock")


def remember_event(events, name, sample, prior):
    if events[name] is None:
        events[name] = dict(tick=sample["native_step"], time_s=sample["time_s"],
                            last_consumed=prior)


def drive(adapter, case, result, *, measured_anchor=False, teacher_class=MultitaskTeacher):
    adapter.prepare_dynamic()
    out = adapter.output
    require(read(out / "GUARD_READY.json") == {
        "episode": out.name, "pid": next(p.pid for n, p in adapter.processes if n == "planner"),
    }, "stationary H0 receiver owner mismatch")
    channel = case / "student"
    epoch = read(channel / "authority.json")["epoch"]
    mount = mobile.arm(adapter)
    adapter.mobile_contact_stage = "approach"
    action, base = fixed(adapter.measured_task_state(), mount)
    sequence = 0
    for _ in range(128):
        if sequence % 7 == 0:
            fixed_base_install(adapter, action, base)
        tick = adapter.observe()["step"]
        width = grip_at_grant(adapter, tick)
        mobile.consume(adapter, action, sequence=sequence)
        grip = gripper_receipt(channel, epoch, sequence, tick, width)
        sequence += 1
    tick = INITIAL_TICK
    paused(adapter, tick)
    samples_path = out / "MOBILE_SAMPLES.private.jsonl"
    parts, _, _ = startup_boundary(out, read_line(samples_path))
    base = np.repeat(parts["old_base"][None], 41, axis=0)
    model = adapter.actual_state.model
    monitor = StateMonitor(model)
    events = {"before_close": None, "first_query_after_grasp": None}
    index = AcceptedBoundaryIndex()
    geometry = adapter.spec["multitask_geometry"]
    consumed = None
    result["observation_events"] = events

    def accepted_knot():
        packet = adapter.observe()
        index.update(adapter.reference_events(), adapter.ros_evidence)
        reference, witness = index.boundary(adapter.active_reference["wire_sha256"], packet["time_s"],
                                            initial_hold=True)
        adapter.reference_history[-1]["pose"] = reference.copy()
        return reference, witness

    with ((out / "state.bin.journal").open("rb") as journal,
          (case / "HOLD_CONSUMPTION.private.jsonl").open("x", buffering=1) as log):
        for i in range(1, INITIAL_TICK + 1):
            packet = read_journal_packet(journal)
            require(packet is not None and packet["step"] == i, "initial journal gap")
        monitor.observe(packet)

        def settle_atom(refresh):
            nonlocal tick, sequence, grip, packet
            require(tick + ATOM_TICKS <= STABLE_DEADLINE, "stationary stable entry deadline")
            if refresh:
                fixed_base_install(adapter, action, base)
            width = grip_at_grant(adapter, tick)
            mobile.consume(adapter, action, sequence=sequence)
            grip = gripper_receipt(channel, epoch, sequence, tick, width)
            sequence += 1
            for _ in range(ATOM_TICKS):
                packet = read_journal_packet(journal)
                require(packet is not None, "missing stable-entry packet")
                monitor.observe(packet)
            tick += ATOM_TICKS
            paused(adapter, tick)

        while not monitor.stable_now():
            settle_atom((sequence - 128) % 7 == 0)
        stance_origin = None
        if measured_anchor:
            from sea_stance_anchor import SETTLING_TICKS, reference as anchored_reference

            prior_ee, prior = accepted_knot()
            adapter.wait_source()
            action, base, anchor = anchored_reference(base[0], prior_ee, mobile.measured_mount(adapter))
            anchor.update(cutover_tick=tick, prior_accepted_knot=prior)
            result["stance_anchor"] = anchor
            record(case / "STATIONARY_ANCHOR.private.json", anchor)
            monitor.entry = monitor.stable[-1]
            stance_origin = monitor.entry[2:5]
            settle_start = tick
            since = sequence
            while tick - settle_start < SETTLING_TICKS or not monitor.stable_now():
                settle_atom((sequence-since) % 7 == 0)
            anchor["settled_tick"] = tick
        hold_base = np.r_[base[0, :3, 3], Rotation.from_matrix(base[0, :3, :3]).as_quat()]
        initial = adapter.measured_task_state()
        require(initial["native_step"] == tick and not initial["robot_object_contact"], "stationary entry")
        reference, witness = accepted_knot()
        consumed = dict(witness=witness, gripper=grip)
        score = StationaryHoldScore(model, geometry, stance_origin=stance_origin)
        score.update_packet(packet)
        stage_clock = StageClock(tick)
        teacher = teacher_class(initial, {"family": "F1", "mode": "M_ASSIST"}, geometry, "HOLD-red")
        result["entry_tick"] = tick
        ended = "H0_BUDGET_EXHAUSTED"
        for query in range(MAX_QUERIES):
            sample = adapter.measured_task_state()
            require(sample["native_step"] == tick, "teacher query does not match the owned pause")
            try:
                plan, evidence = teacher.decide(sample, reference, hold_base)
            except ValueError as error:
                if "ORIGINAL_FULL_PREVIEW_RATE_AND_ACCELERATION_LIMITS" not in str(error):
                    raise
                ended = "H0_REFERENCE_REFUSED"
                result["refusal"] = str(error)
                break
            stage = evidence["stage"]
            if stage_clock.observe(stage, tick)["exhausted"]:
                ended = "H0_STAGE_BUDGET_EXHAUSTED"
                break
            if stage == "CLOSE":
                remember_event(events, "before_close", sample, consumed)
            if sample["native_grasp"]:
                remember_event(events, "first_query_after_grasp", sample, consumed)
            adapter.mobile_contact_stage = "approach" if stage in ("PREGRASP", "DESCEND") else stage.lower()
            adapter.mobile_program = plan
            adapter.mobile_program_start_step = tick
            for atom in range(10):
                action = plan.preview(atom)
                if atom in (0, 7):
                    fixed_base_install(adapter, action, base)
                width = grip_at_grant(adapter, tick)
                sample, _ = mobile.consume(adapter, action, sequence=sequence)
                grip = gripper_receipt(channel, epoch, sequence, tick, width)
                sequence += 1
                tick += ATOM_TICKS
                paused(adapter, tick)
                reference, witness = accepted_knot()
                require(sample["native_step"] == tick, "native task atom ended on wrong tick")
                consumed = dict(query=query, atom=atom, stage=stage, witness=witness, gripper=grip)
                log.write(json.dumps(consumed) + "\n")
                score_through(journal, score, tick)
                if score.first_violation is not None or score.success:
                    break
            result["score"] = score.result()
            if score.first_violation is not None:
                ended = "H0_STANCE_OR_CONTACT_STOP"
                break
            if score.success:
                ended = "H0_HOLD_PASS"
                break
            teacher.commit(plan, tick)
        result.update(end=tick, score=score.result(), stage_transitions=stage_clock.transitions)
    result["cancel"] = adapter.cancel_reference()
    require(result["cancel"]["native_step"] == tick and result["cancel"]["mode"] == "HOLD",
            "terminal independent HOLD missing")
    paused(adapter, tick)
    adapter.seal_trace(tick)
    result["outcome"] = ended
    record(case / "H0_OBSERVATION_EVENTS.private.json", events)
