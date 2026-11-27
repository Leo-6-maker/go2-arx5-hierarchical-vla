"""Read the active MPC command at the frozen controller's actual evaluation call.

Run inside a fresh GDB owning its inferior. This is a diagnostic observer:
hardware breakpoints pause the inferior and are not a production ACK.
"""

import __main__
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time

import gdb


FROZEN = "9e8ce46837d09bc39ae04d4b49f1ce4dd8c8c2d31bea39eac56c5806d108c1f3"
EVALUATE = "_ZN4ocs28MRT_BASE14evaluatePolicyEdRKN5Eigen6MatrixIdLin1ELi1ELi0ELin1ELi1EEERS3_S6_Rm"


def raw(address, n):
    if address <= 0 or n <= 0:
        raise ValueError("A selected-reference field is absent")
    return bytes(gdb.selected_inferior().read_memory(address, n))


def ptr(address):
    return struct.unpack("<Q", raw(address, 8))[0]


def vector(address, stride):
    begin, end, capacity = struct.unpack("<QQQ", raw(address, 24))
    count, remainder = divmod(end - begin, stride)
    if not (begin <= end <= capacity and not remainder and 0 <= count <= 512):
        raise ValueError("Unsupported active trajectory vector bounds")
    return begin, count


def matrices(address):
    start, count = vector(address, 16)
    rows = []
    header = raw(start, 16*count) if count else b""
    for values, width in struct.iter_unpack("<Qq", header):
        if not 0 <= width <= 128:
            raise ValueError("Unexpected Eigen vector width")
        rows.append(list(struct.unpack(f"<{width}d", raw(values, 8*width))) if width else [])
    return rows


def scalars(address, encoding):
    begin, count = vector(address, 8)
    return list(struct.unpack(f"<{count}{encoding}", raw(begin, 8*count))) if count else []


def identity(payload):
    raw_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw_json.encode()).hexdigest()


def solution(address):
    # PrimalSolution and ModeSchedule come from this executable's OCS2 ABI.
    times = scalars(address, "d")
    states, inputs = matrices(address + 24), matrices(address + 48)
    events = scalars(address + 72, "Q")
    event_times, modes = scalars(address + 96, "d"), scalars(address + 120, "Q")
    if not times or len(states) != len(times) or len(inputs) != len(times):
        raise ValueError("Unexpected active OCS2 solution trajectory lengths")
    return dict(active_solution_sha256=identity(dict(times=times, states=states, inputs=inputs,
                                                     post_events=events, event_times=event_times, modes=modes)),
                solution_first_time=times[0], solution_last_time=times[-1], solution_knots=len(times))


def record(reference, selected_at, *, include_solution=False):
    command = ptr(reference + 0x10)
    active_policy = ptr(reference + 0x20)
    if not command or not active_policy:
        raise ValueError("No active OCS2 policy/command at evaluatePolicy")
    target = command + 48
    times = scalars(target, "d")
    n = len(times)
    poses, inputs = matrices(target + 24), matrices(target + 48)
    if not (n and len(poses) == n and len(inputs) in (0, n) and all(a <= b for a,b in zip(times, times[1:]))
            and math.isfinite(selected_at)):
        raise ValueError("Selected target timestamps and values disagree")
    digest = identity(dict(times=times, poses=poses, inputs=inputs))
    observed = struct.unpack("<d", raw(command + 8, 8))[0]
    event = dict(eval_time_s=selected_at, mpc_init_time_s=observed, trajectory_sha256=digest,
                 target_knots=n, first_target_time=times[0], last_target_time=times[-1])
    if include_solution:
        event.update(solution(active_policy))
    return event


def install(output, max_samples, *, fixture=False, fixture_solution=False):
    binary = Path(gdb.current_progspace().filename)
    with binary.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if not fixture and digest != FROZEN:
        raise ValueError("Unknown controller bytes")
    symbol = "fixture_evaluate" if fixture else EVALUATE
    site = int(gdb.parse_and_eval("&" + symbol))
    resume_address = None if fixture else site - 0x2B6C80 + 0x28A589
    if not fixture:
        # Verify the exact direct call instruction at the original wrapper.
        if (raw(resume_address-5, 5) != bytes.fromhex("e8 f7 c6 02 00")
                or raw(resume_address-0x1a, 5) != bytes.fromhex("49 8d 74 24 10")):
            raise ValueError("Unknown frozen WBC caller")
    stream = Path(output).open("x", buffering=1)
    stream.write(json.dumps(dict(schema="OCS2_SELECTED_REFERENCE_BREAKPOINT_02", binary_sha256=digest,
                                 fixture=fixture, hardware_breakpoint=True, max_samples=max_samples,
                                 inferior_pid=gdb.selected_inferior().pid,
                                 solution_fields="TRAJECTORIES_POST_EVENTS_AND_MODE_SCHEDULE"))+"\n")

    class Observer(gdb.Breakpoint):
        def __init__(self):
            super().__init__(f"*{site:#x}",type=gdb.BP_HARDWARE_BREAKPOINT, internal=True)
            self.count = 0

        def stop(self):
            entered = time.monotonic_ns()
            # The return address limits the witness to WbcBridgeCore::Evaluate.
            if resume_address is not None and ptr(int(gdb.parse_and_eval("$rsp"))) != resume_address:
                return False
            v = record(int(gdb.parse_and_eval("$rdi")),float(gdb.parse_and_eval("$xmm0.v2_double[0]")),
                       include_solution=not fixture or fixture_solution)
            if not fixture:
                # RSI is &SystemObservation::state, passed as r12 + 0x10.
                state_address = int(gdb.parse_and_eval("$rsi"))
                if state_address != int(gdb.parse_and_eval("$r12")) + 0x10:
                    raise ValueError("Controller observation does not use the pinned caller ABI")
                source_time = struct.unpack("<d", raw(state_address - 8, 8))[0]
                if not math.isfinite(source_time):
                    raise ValueError("Invalid controller observation clock")
                v["native_observation_time_s"] = source_time
            self.count += 1
            if self.count == max_samples and not fixture:
                v["diagnostic_capture_limit_reached"] = True
            stream.write(json.dumps(dict(v, sequence=self.count, monotonic_ns=entered,
                                         observer_read_ns=time.monotonic_ns()-entered),allow_nan=False)+"\n")
            if self.count == max_samples:
                self.enabled = False
            return False

    return Observer()


def disarm_on_native_signal(observer, output):
    """Remove hardware breakpoints before a normal controller thread teardown."""
    gdb.execute("catch signal SIGINT SIGTERM", to_string=True)
    catchpoint = gdb.breakpoints()[-1]

    def record_disarm():
        if observer.is_valid():
            observer.delete()
            with Path(output).open("x") as stream:
                json.dump(dict(inferior_pid=gdb.selected_inferior().pid, monotonic_ns=time.monotonic_ns(),
                               signal_pending=True, hardware_breakpoint_removed=True), stream)
                stream.write("\n")

    __main__.record_observer_signal_disarm = record_disarm
    gdb.execute(f"commands {catchpoint.number}\nsilent\npython record_observer_signal_disarm()\ncontinue\nend",
                to_string=True)


if "SLA_SELECTED_REFERENCE_TRACE" in os.environ:
    LIMIT = int(os.environ.get("SLA_SELECTED_REFERENCE_TRACE_LIMIT","64"))
    if not 1 <= LIMIT <= 8192:
        raise ValueError("Use a bounded diagnostic trace")
    OBSERVER = install(os.environ["SLA_SELECTED_REFERENCE_TRACE"],LIMIT,
                       fixture=os.environ.get("SLA_SELECTED_REFERENCE_FIXTURE")=="1",
                       fixture_solution=os.environ.get("SLA_SELECTED_REFERENCE_FIXTURE_SOLUTION")=="1")
    disarm_on_native_signal(OBSERVER, os.environ["SLA_SELECTED_REFERENCE_TRACE"] + ".shutdown")
