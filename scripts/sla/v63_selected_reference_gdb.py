"""Read the active MPC command at the frozen controller's actual evaluation call.

Run inside a fresh GDB owning its inferior. This is a diagnostic observer:
hardware breakpoints pause the inferior and are not a production ACK.
"""

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
    for i in range(count):
        values, width = struct.unpack("<Qq", raw(start + 16*i, 16))
        if not 0 <= width <= 128:
            raise ValueError("Unexpected Eigen vector width")
        rows.append(list(struct.unpack(f"<{width}d", raw(values, 8*width))) if width else [])
    return rows


def record(reference, selected_at):
    command = ptr(reference + 0x10)
    active_policy = ptr(reference + 0x20)
    if not command or not active_policy:
        raise ValueError("No active OCS2 policy/command at evaluatePolicy")
    target = command + 48
    start, n = vector(target, 8)
    times = list(struct.unpack(f"<{n}d", raw(start, 8*n)))
    poses, inputs = matrices(target + 24), matrices(target + 48)
    if not (len(poses) == n and len(inputs) in (0, n) and all(a <= b for a,b in zip(times, times[1:]))
            and math.isfinite(selected_at)):
        raise ValueError("Selected target timestamps and values disagree")
    identity = dict(times=times, poses=poses, inputs=inputs)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()
    observed = struct.unpack("<d", raw(command + 8, 8))[0]
    return dict(eval_time_s=selected_at, mpc_init_time_s=observed, trajectory_sha256=digest,
                target_knots=n, first_target_time=times[0], last_target_time=times[-1],
                monotonic_ns=time.monotonic_ns())


def install(output, max_samples, *, fixture=False):
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
        if raw(resume_address-5, 5) != bytes.fromhex("e8 f7 c6 02 00"):
            raise ValueError("Unknown frozen WBC caller")
    stream = Path(output).open("x", buffering=1)
    stream.write(json.dumps(dict(schema="OCS2_SELECTED_REFERENCE_BREAKPOINT_01", binary_sha256=digest,
                                 fixture=fixture, hardware_breakpoint=True, max_samples=max_samples))+"\n")

    class Observer(gdb.Breakpoint):
        def __init__(self):
            super().__init__(f"*{site:#x}",type=gdb.BP_HARDWARE_BREAKPOINT, internal=True)
            self.count = 0

        def stop(self):
            # The return address limits the witness to WbcBridgeCore::Evaluate.
            if resume_address is not None and ptr(int(gdb.parse_and_eval("$rsp"))) != resume_address:
                return False
            v = record(int(gdb.parse_and_eval("$rdi")),float(gdb.parse_and_eval("$xmm0.v2_double[0]")))
            self.count += 1
            stream.write(json.dumps(dict(v,sequence=self.count),allow_nan=False)+"\n")
            if self.count == max_samples:
                self.enabled = False
            return False

    return Observer()


if "SLA_SELECTED_REFERENCE_TRACE" in os.environ:
    LIMIT = int(os.environ.get("SLA_SELECTED_REFERENCE_TRACE_LIMIT","64"))
    if not 1 <= LIMIT <= 1024:
        raise ValueError("Use a bounded diagnostic trace")
    OBSERVER = install(os.environ["SLA_SELECTED_REFERENCE_TRACE"],LIMIT,
                       fixture=os.environ.get("SLA_SELECTED_REFERENCE_FIXTURE")=="1")
