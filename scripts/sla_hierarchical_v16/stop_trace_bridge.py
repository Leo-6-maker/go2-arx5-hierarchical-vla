"""Prove the instrumentation-only build against the unchanged V2 receiver and trace writer."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from build_student_receiver import replace_once
from receiver_fault_replay import read, sha


def checked(path, digest):
    if sha(path) != digest:
        raise ValueError("unexpected artifact " + path.name)


def variant(root, out, spec, command, traced):
    build = read(
        Path(__file__).resolve().parents[2]
        / "reports/SLA_HIERARCHICAL_V16/V2_RECEIVER_BUILD.json"
    )
    old = root / "v16/v2-build03"
    original = Path(spec["outer_library"])
    src = old / original.with_suffix(".cpp").name
    header = old / "student_receiver.hpp"
    checked(src, build["student_source"])
    checked(header, build["student_header"])
    writer = Path(spec["trace_library"])
    checked(writer, spec["frozen_files"][str(writer)])
    writer_spec = read(writer.parent.parent / "EXECUTION_SPEC.json")
    writer_root = Path(writer_spec["source_root"])
    header_key = "src/go2arx5_vla/target_mujoco/init_trace.hpp"
    trace_header = writer_root / header_key
    checked(trace_header, writer_spec["source_files"][header_key])
    built = out / "receiver"
    built.mkdir()
    dest = built / "student_grip.so"
    local = built / src.name
    text = src.read_text()
    shutil.copyfile(header, built / header.name)
    if traced:
        shutil.copyfile(trace_header, built / trace_header.name)
        text = replace_once(
            text,
            '#include "student_receiver.hpp"',
            '#include "student_receiver.hpp"\n#include "init_trace.hpp"',
        )
        marker = '   student.stop("RECEIVER_REJECT_STOP",steps,target,measured_before);'
        text = replace_once(
            text,
            marker,
            "   const bool first_rejection = !student.stopped;\n"
            + marker
            + """
   if (first_rejection) {
     // Marker 1 identifies a gripper packet rejection. Keep the measured simulator clock.
     const float source[] = {1.0f, static_cast<float>(student.last.sequence)};
     WM02InitEmit(wm02_init::STOP, d->time, steps, source, 2);
   }""",
        )
    local.write_text(text)
    argv = [
        str(local)
        if x == str(original.with_suffix(".cpp"))
        else str(dest)
        if x == str(original)
        else x
        for x in command
    ]
    if str(local) not in argv or argv[-2:] != ["-o", str(dest)]:
        raise ValueError("unrecognized frozen build")
    subprocess.run(argv, check=True)
    if not traced:
        checked(dest, build["student_binary"])
    return dest, writer


def score_trace(path, writer, expected):
    spec = read(writer.parent.parent / "EXECUTION_SPEC.json")
    base = Path(spec["source_root"])
    key = "src/go2arx5_vla/world_model/init_trace_reader.py"
    checked(base / key, spec["source_files"][key])
    sys.path.insert(0, str(base / "src"))
    from go2arx5_vla.world_model import init_trace_reader

    checked(Path(init_trace_reader.__file__), spec["source_files"][key])
    rows = init_trace_reader.read_trace(path)
    summary = read(path.with_suffix(".trace.summary"))
    if (
        len(rows) != expected
        or summary["events"] != expected
        or summary["written"] != expected
        or summary["dropped"]
        or summary["sealed_at_explicit_pause"]
        or summary["stop_latched"] != bool(expected)
    ):
        raise ValueError("missing terminal trace or premature seal")
    if expected:
        row = rows[0]
        if (
            row["kind"] != 13
            or row["token"] != 6439
            or abs(row["sim_time_s"] - 6439 * 0.002) > 1e-8
            or list(row["values"]) != [1.0, float((6428 - 2503) // 25)]
        ):
            raise ValueError("STOP not joined to receiver tick, time and command")
    return dict(
        stop_events=expected,
        writer_events=summary["written"],
        stop_sim_s=rows[0]["sim_time_s"] if rows else None,
        stop_token=rows[0]["token"] if rows else None,
        sealed_before_teardown=summary["sealed_at_explicit_pause"],
    )


def main(root, destination):
    from receiver_fault_replay import run

    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    results = [
        run(root, destination / mode, trace_mode=mode) for mode in ("CONTROL", "BRIDGE")
    ]
    for a, b in zip(results[0]["treatments"], results[1]["treatments"], strict=True):
        for k in (
            "method",
            "receiver_log_sha256",
            "recorded_step_counter_sha256",
            "returning_signals",
        ):
            if a[k] != b[k]:
                raise ValueError("instrumentation changed control " + k)
    with (destination / "V3_STOP_TRACE_ACTUAL.json").open("x") as stream:
        json.dump(
            dict(
                control=results[0],
                bridge=results[1],
                new_native_starts=0,
                whole_system_moving_STOP_qualified=False,
            ),
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


if __name__ == "__main__":
    import os

    os.umask(0o077)
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
