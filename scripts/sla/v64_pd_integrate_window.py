"""Replay native PD readiness and sensor consumption from closed warmup traces."""

import argparse
from collections import Counter
from pathlib import Path

import yaml

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v64_warmup_pd_clock import inspect


def clock_contract(case):
    lease = read(case / "control/LEASE.json")
    require(lease["status"] == "CLOSED", "Only a closed native case is eligible")
    spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    bindings = spec["frozen_files"]
    clock_config, = [name for name in bindings if name.endswith("/go2_x5/base.yaml")]
    with pin(clock_config, bindings[clock_config]).open() as stream:
        (settings,) = yaml.safe_load(stream).values()
    control_ns = round(settings["dt"] * 1e9)
    policy_ns = control_ns * settings["decimation"]
    require((control_ns, policy_ns, spec["startup_pause_stride_ticks"]) == (5_000_000, 20_000_000, 25),
            "Expected the two frozen callback clocks and the original pause stride")
    return control_ns, policy_ns


def prepare(case):
    control_ns, policy_ns = clock_contract(case)
    trace = case / "native-01/init.trace"
    outcomes = []
    for row in inspect(trace)[1:]:
        completed = row["tick"]-1
        callback_tick = row["measurement_tick"]
        # The actuator reconstruction on 14,411 events validated both phases.
        sensor_tick = callback_tick if callback_tick % 25 == 0 else callback_tick - 1
        ns = completed * 2_000_000
        callback_ns = callback_tick * 2_000_000
        same_control_bucket = callback_ns//control_ns == ns//control_ns
        outcomes.append(dict(tick=row["tick"], completed_tick=completed, callback_tick=callback_tick,
                             sensor_tick=sensor_tick, control_ready=same_control_bucket,
                             same_policy_bucket=callback_ns//policy_ns == ns//policy_ns))
    require(all(x["control_ready"] and x["same_policy_bucket"] for x in outcomes),
            "Reconstructed frozen clock disagrees with actual native integration")
    return outcomes, sha(trace)


def compare(label, archived, actual):
    a, a_trace = prepare(archived)
    b, b_trace = prepare(actual)
    changed = []
    for ref, run in zip(a, b, strict=True):
        require(ref["tick"] == run["tick"], "Matched native clock expected")
        if ref["sensor_tick"] != run["sensor_tick"]:
            changed.append(dict(tick=ref["tick"], reference_sensor_tick=ref["sensor_tick"],
                                observed_sensor_tick=run["sensor_tick"],
                                reference_callback_tick=ref["callback_tick"],
                                observed_callback_tick=run["callback_tick"]))
    older = sum(x["observed_sensor_tick"] < x["reference_sensor_tick"] for x in changed)
    windows = sorted({(row["tick"] - 1)//25 for row in changed})
    return dict(comparison=label, archive_trace_sha256=a_trace, actual_trace_sha256=b_trace,
                compared_ticks=len(a), control_ready_ticks=len(b), different_sensor_epochs=len(changed),
                actual_older=older, actual_newer=len(changed)-older, affected_windows=len(windows),
                mismatched_next_tick_modulo_25=dict(Counter(row["tick"] % 25 for row in changed)),
                earliest_differences=changed[:8], last_difference=changed[-1] if changed else None)


def phase_fixture(case):
    """Extract one optional callback bit per archived 25-tick window."""
    rows, trace = prepare(case)
    optional = [row for row in rows if row["tick"] % 25 == 3]
    required = []
    for i, row in enumerate(optional):
        pause = i * 25
        newer = row["callback_tick"] == pause+2
        # In the cold first window a callback at tick 1 also reads state 0.
        old_phase = row["callback_tick"] == pause or (i == 0 and row["callback_tick"] == 1)
        require(row["tick"] == pause+3 and (old_phase or newer)
                and row["sensor_tick"] == pause+int(newer), "Unexpected optional phase or sensor input")
        required.append(newer)
    require(len(required) == 100, "Apply the optional phase only to the first five native seconds")
    return dict(trace_sha256=trace, callback_action_tick_within_window=2,
                next_integration_tick_within_window=3, require_callback=required)


def replay_fixture(label, archive, actual):
    fixture = phase_fixture(archive)
    ref, _ = prepare(archive)
    observed, actual_trace = prepare(actual)
    require(fixture["trace_sha256"] != actual_trace, "Independent reference and native comparison required")
    phase = fixture["require_callback"]
    counts = Counter()
    for a, b in zip(ref, observed, strict=True):
        tick = b["tick"]
        new_sensor = b["sensor_tick"]
        if tick % 25 == 3:
            window = (tick-1)//25
            native_optional = b["callback_tick"] == window*25+2
            counts["would_wait"] += phase[window] and not native_optional
            counts["would_suppress"] += native_optional and not phase[window]
            new_sensor = window*25 + phase[window]
        counts["remaining_epoch_mismatches"] += new_sensor != a["sensor_tick"]
    require(not counts["remaining_epoch_mismatches"], "Optional phase doesn't explain the full epoch difference")
    return dict(comparison=label, actual_trace_sha256=actual_trace, phase_fixture=fixture,
                always_require_wrong_windows=phase.count(False), always_suppress_wrong_windows=phase.count(True),
                **counts)


def parse_member(value):
    label, reference, actual = value.split(":", 2)
    return label, Path(reference), Path(actual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("members", type=parse_member, nargs="+")
    parser.add_argument("--stage-fixture", action="store_true")
    args = parser.parse_args()
    comparisons = [compare(*row) for row in args.members]
    fixture = [replay_fixture(*row) for row in args.members] if args.stage_fixture else []
    write(args.output, dict(schema="V64_WARMUP_CLOCK_AND_SENSOR_EPOCH_MATCH", comparisons=comparisons,
                            optional_phase_fixture=fixture,
                            physics_integrations=0, requests=0, worlds_started=0,
                            native_timing_change=False))
