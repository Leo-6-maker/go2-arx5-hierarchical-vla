"""Recover the sensor state used by the frozen PD formula from actual traces."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path

import mujoco
import numpy as np
import yaml

from go2arx5_vla.world_model.init_trace_reader import RECORD
from go2arx5_vla.world_model.native_replay import forbid_integration
from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write


def recorded_prefix(path, tick, *, all_callbacks=False):
    target = None
    callback = None
    states = {}
    callbacks = []
    with path.open("rb") as stream:
        header = stream.read(16)
        require(header[:8] == b"WM02IT01" and int.from_bytes(header[8:12], "little") == RECORD.size,
                "Unexpected actual trace ABI")
        for index, raw in enumerate(iter(lambda: stream.read(RECORD.size), b"")):
            row = RECORD.unpack(raw)
            seq, _, seconds, kind, n = row[:5]
            native_tick = round(seconds / .002)
            require(seq == index, "Original trace order changed")
            if kind not in (9, 10, 11, 12):
                continue
            require(abs(seconds - .002*native_tick) < 1e-8, "Original trace clock changed")
            values = np.asarray(row[10:10+n])
            if kind == 9:
                require(n == 90, "Frozen five by 18 PD target fields are required")
                target = (native_tick, row[7], values)
            if kind == 10:
                require(target is not None and target[:2] == (native_tick, row[7]) and row[7] == row[8],
                        "Every actual callback must bind the same leg output and target")
                callback = (native_tick, row[7], target[2], values)
                if all_callbacks and native_tick >= 4:
                    callbacks.append(callback)
            if kind == 12:
                require(row[9] == native_tick, "Invalid native endpoint")
                states[native_tick] = values
            if kind == 11 and row[9] == tick:
                require(native_tick == tick - 1 and callback is not None, "Missing actual first divergent entry")
                return states, (callbacks if all_callbacks else callback), values
    raise ValueError("No native entry at the registered tick")


def load_inputs(case):
    lease = read(case / "control/LEASE.json")
    require(lease["status"] == "CLOSED", "Only closed case traces may be inspected")
    spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    bindings = spec["frozen_files"]
    model_path = spec["model_path"]
    model = mujoco.MjModel.from_binary_path(str(pin(model_path, bindings[model_path])))
    mappings = []
    for suffix in ("/go2_x5/base.yaml", "/N1_s29_175632/config.yaml"):
        config, = (name for name in bindings if name.endswith(suffix))
        with pin(config, bindings[config]).open() as stream:
            (settings,) = yaml.safe_load(stream).values()
        require(settings["num_of_dofs"] == 18, "Original frozen 18-motor ABI changed")
        mappings.append(settings["joint_mapping"])
    require(mappings[0] == mappings[1] and sorted(mappings[0]) == list(range(18)),
            "Frozen base and leg policy have a common one-to-one motor map")
    mapping = np.asarray(mappings[0])
    require((model.nq, model.nv, model.nu) == (41, 38, 20)
            and np.array_equal(model.sensor_adr[mapping], mapping)
            and np.array_equal(model.sensor_adr[18+mapping], 18+mapping)
            and (model.sensor_type[mapping] == mujoco.mjtSensor.mjSENS_JOINTPOS).all()
            and (model.sensor_type[18+mapping] == mujoco.mjtSensor.mjSENS_JOINTVEL).all()
            and np.array_equal(model.sensor_objid[mapping], model.sensor_objid[18+mapping]),
            "Original SetCommand sensor and actuator positions are required")
    joints = model.sensor_objid[mapping]
    qpos, qvel = model.jnt_qposadr[joints], model.nq + model.jnt_dofadr[joints]
    names = [model.actuator(i).name for i in range(17)]
    return mapping, qpos, qvel, names


def inspect(case, tick):
    mapping, qpos, qvel, names = load_inputs(case)
    trace = case / "native-01/init.trace"
    states, (callback_tick, policy, target, callback_values), entered = recorded_prefix(trace, tick)
    require(len(callback_values) == len(entered) == 96, "Only first 17 actuator controls fit the trace")
    actual = callback_values[79:]
    entered_control = entered[79:]
    np.testing.assert_array_equal(actual, entered_control)
    candidates = {k: pd_command(mapping, qpos, qvel, target, states[k])
                  for k in range(callback_tick-3, callback_tick+1)}
    errors = {k: float(np.max(np.abs(value-actual))) for k, value in candidates.items()}
    matches = [k for k, error in errors.items() if error < 1e-9]
    require(len(matches) == 1 and min(error for k, error in errors.items() if k != matches[0]) > 1e-6,
            "The used sensor state must be unique among the four neighboring states")
    state_tick, = matches
    same_phase = pd_command(mapping, qpos, qvel, target, states[tick-2])
    return dict(trace_sha256=sha(trace), divergent_entry_tick=tick, callback_tick=callback_tick,
                sensor_state_tick=state_tick, native_sensor_age_ms=2*(tick-1-state_tick),
                target_sha256=hashlib.sha256(target.tobytes()).hexdigest(), applied_leg_policy=policy,
                reconstructed_actuators=names, residual_by_candidate_tick=errors,
                actual_control=actual.tolist(), rebuilt_control=candidates[state_tick].tolist(),
                hypothetical_interior_callback_tick=tick-1, hypothetical_sensor_tick=tick-2,
                same_phase_control=same_phase.tolist())


def pd_command(mapping, qpos, qvel, target, sample):
    desired_pos, desired_vel, kp, kd, feedforward = target.reshape(5, 18)
    result = np.empty(18)
    result[mapping] = feedforward + kp*(desired_pos-sample[qpos]) + kd*(desired_vel-sample[qvel])
    return result[:17]


def scan(case):
    mapping, qpos, qvel, _ = load_inputs(case)
    states, callbacks, _ = recorded_prefix(case / "native-01/init.trace", 2500, all_callbacks=True)
    counts = {"25_tick_pause": Counter(), "within_window": Counter()}
    for tick, _, target, values in callbacks:
        actual = values[79:]
        ages = [age for age in range(4)
                if np.max(np.abs(pd_command(mapping, qpos, qvel, target, states[tick-age])-actual)) < 1e-9]
        key = str(ages[0]) if len(ages) == 1 else "ambiguous" if ages else "unmatched"
        counts["25_tick_pause" if tick % 25 == 0 else "within_window"][key] += 1
    return counts


def parse_member(value):
    family, role, tick, case = value.split(":", 3)
    return family, role, int(tick), Path(case)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("members", type=parse_member, nargs="+")
    parser.add_argument("--full-warmup", action="store_true")
    args = parser.parse_args()
    with forbid_integration(mujoco) as effects:
        rows = []
        for family, role, tick, case in args.members:
            row = dict(family=family, topology=role, **inspect(case, tick))
            if args.full_warmup:
                row["warmup_callback_sensor_age_counts_ticks"] = scan(case)
            rows.append(row)
    require(not effects, "Sensor reconstruction must never integrate")
    reference = {row["family"]: row for row in rows if row["topology"] == "archive"}
    comparisons = []
    for row in rows:
        if row["topology"] not in ("N1", "N2") or row["family"] not in reference:
            continue
        archive = reference[row["family"]]
        require(archive["target_sha256"] == row["target_sha256"]
                and archive["hypothetical_sensor_tick"] == archive["sensor_state_tick"]
                and archive["divergent_entry_tick"] == row["divergent_entry_tick"]
                and archive["reconstructed_actuators"] == row["reconstructed_actuators"],
                "The what-if comparison changes only the sensor phase")
        baseline = np.asarray(archive["actual_control"])
        comparisons.append(dict(family=row["family"], topology=row["topology"],
                                actual_error_max=float(np.max(np.abs(baseline-row["actual_control"]))),
                                same_phase_error_max=float(np.max(np.abs(baseline-row["same_phase_control"])))))
    write(args.output, dict(schema="V64_FROZEN_PD_ACTUATOR_REPLAY", records=rows,
                            recorded_state_counterfactual=comparisons, new_native_integrations=0))
