"""Preserve the real controller exit status when GDB detaches at shutdown."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import time


PR_SET_CHILD_SUBREAPER = 36


def supervise(trace, command):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))

    debugger = subprocess.Popen(command)
    gdb_exit = debugger.wait()
    debugger_closed = time.monotonic_ns()
    adopted = {}
    # The debugger is the only child launched by this process. The kernel
    # reparents its detached controller here, even if it has already exited.
    while True:
        try:
            pid, status = os.waitpid(-1, 0)
        except ChildProcessError:
            break
        adopted[pid] = os.waitstatus_to_exitcode(status)

    with trace.open() as stream:
        inferior = json.loads(next(stream))["inferior_pid"]
    disarm = json.loads(trace.with_name(trace.name + ".shutdown").read_bytes())
    exact_child = set(adopted) == {inferior} and disarm["inferior_pid"] == inferior
    protocol_complete = (
        gdb_exit == 0 and exact_child and disarm["hardware_breakpoint_removed"]
        and disarm["signal_pending"] and disarm["monotonic_ns"] <= debugger_closed
    )
    native_exit = adopted.get(inferior)
    with trace.with_name(trace.name + ".detached-exit").open("x") as stream:
        json.dump(dict(gdb_pid=debugger.pid, gdb_exit=gdb_exit, gdb_closed_monotonic_ns=debugger_closed,
                       inferior_pid=inferior, adopted_exits=adopted, native_exit=native_exit,
                       protocol_complete=protocol_complete), stream)
        stream.write("\n")
    if not protocol_complete:
        return 1
    return 128 - native_exit if native_exit < 0 else native_exit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    raise SystemExit(supervise(options.trace, options.command))
