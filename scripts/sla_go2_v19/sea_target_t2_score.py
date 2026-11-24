"""Score T2 predictions against the measured compact H0 flange trajectory, offline."""

import argparse
import os
from pathlib import Path

import numpy as np

from source_custody import BASE, digest, pin, read, require, write_new
from policy_query import tensor_sha
from sea1_archive_readout import alignment, pose_error, pose_rows
from demonstrated_grip import events

DT = .002
HOLD_CLOSE, HOLD_OPEN = .55, .65
TEACHER_OFFSETS = np.rint(np.arange(1, 51) / (30 * DT)).astype(int)


def matrix(position, rotation):
    frame = np.eye(4)
    frame[:3, :3] = rotation.reshape(3, 3)
    frame[:3, 3] = position
    return frame


def teacher(root, ingress, output):
    import mujoco
    from go2arx5_vla.world_model.multitask_state_v6 import packets
    from go2arx5_vla.world_model.native_replay import apply_frame, forbid_integration

    cold = read(ingress)
    report = read(pin(BASE / "reports/SLA_GO2_V193/GRIP_H0/COMPACT_H0_ACTUAL.json",
                      cold["compact_h0_receipt_sha256"]))
    panel = root / "v19" / "V193-COMPACT-FRUIT-H0-01"
    pin(panel / "PANEL_CLOSE.json", report["campaign_sha256"]["PANEL_CLOSE.json"])
    outputs = {}
    require(mujoco.__version__ == "3.2.7" and mujoco.get_mjcb_control() is None, "passive sealed native replay")
    with forbid_integration(mujoco) as integrations:
        for layout, audit in report["rows"].items():
            case = panel / layout
            lease = read(pin(case / "control/LEASE.json", audit["evidence_sha256"]["LEASE.json"]))
            spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
            result = read(pin(case / "RESULT.private.json", audit["evidence_sha256"]["RESULT.private.json"]))
            require(lease["status"] == "CLOSED" and result["outcome"] == "H0_HOLD_PASS",
                    "T2 teacher is unqualified")
            journal = pin(case / result["episode"] / "state.bin.journal",
                          audit["evidence_sha256"]["state.bin.journal"])
            scene = Path(spec["model_path"])
            model = mujoco.MjModel.from_binary_path(str(pin(scene, spec["frozen_files"][str(scene)])))
            data = mujoco.MjData(model)
            rows = {key: row for key, row in cold["rows"].items() if row["layout"] == layout}
            windows = {name: np.r_[row["tick"], row["tick"] + TEACHER_OFFSETS] for name, row in rows.items()}
            wanted = set(np.concatenate(list(windows.values())))
            poses = {}
            for expected, packet in enumerate(packets(journal), 1):
                tick = packet["step"]
                require(tick == expected and abs(packet["time_s"] - tick*DT) < 1e-8, "invalid state journal clock")
                if tick in wanted:
                    apply_frame(mujoco, model, data, packet)
                    flange = data.body("x5_link6")
                    poses[tick] = matrix(flange.xpos, flange.xmat)
            require(set(poses) == wanted, "teacher data does not cover all 50 predicted successors")
            for name, ticks in windows.items():
                row = rows[name]
                origin = np.asarray(row["world_from_diagnostic_flange_frame"])
                actual = np.linalg.inv(origin) @ np.stack([poses[t] for t in ticks])
                state = np.asarray(row["right_observed_channel"])
                require(np.allclose(actual[0], pose_rows(state[None])[0], rtol=0, atol=1e-7),
                        "T2 input flange and measured teacher timeline differ at the query")
                outputs[name] = dict(query_tick=int(ticks[0]), teacher_native_ticks=ticks[1:].tolist(),
                                     actual_flange=actual[1:].tolist(), input_sha256=row["input_sha256"])
    require(not integrations and set(outputs) == set(cold["rows"]), "teacher scoring requested integration")
    write_new(output, dict(status="FOUR_ACTUAL_H0_PATHS_ARCHIVED", cold_sha256=digest(ingress),
                           model_forwards=0, physics_ticks=0, rows=outputs))


def largest_interval(values):
    changes = np.flatnonzero(np.diff(np.r_[False, values, False]))
    return max(0, int(max(changes[1::2] - changes[::2], default=0)) - 1) / 30


def grip_forecast(arm, measured):
    signal = np.r_[measured[7], arm[:, 7]]
    transition = events(np.arange(51) / 30., signal, step=True)
    closed = closed_at_query = bool(signal[0] <= HOLD_CLOSE)
    flags = []
    for aperture in signal[1:]:
        if aperture <= HOLD_CLOSE:
            closed = True
        elif aperture >= HOLD_OPEN:
            closed = False
        flags.append(closed)
    first = next((i for i, value in enumerate(flags) if value), None)
    if first is None:
        elevation = np.full(50, -np.inf)
        close_z = None
    else:
        close_z = float(measured[2] if closed_at_query else arm[first, 2])
        elevation = arm[:, 2] - close_z
        if not closed_at_query:
            elevation[:first] = -np.inf
    above = np.asarray(flags) & (elevation >= .020)
    closed_s = largest_interval(flags)
    above_s = largest_interval(above)
    return dict(started_closed=closed_at_query, first_predicted_closed_row=first,
                remains_closed_across_horizon=all(flags), final_closed=flags[-1],
                contiguous_closed_forecast_s=closed_s, contiguous_elevated_closed_forecast_s=above_s,
                max_flange_Z_above_close_mm=None if close_z is None else 1000*float(elevation.max()),
                reaches_one_second_flange_proxy=above_s >= 1., threshold_events=transition)


def measure(absolute, input_row, teacher_row):
    require(absolute.shape == (50, 16) and np.isfinite(absolute).all(), "incorrect official output")
    arm = absolute[:, 8:]
    qnorm = np.linalg.norm(arm[:, 3:7], axis=1)
    valid_orientation = np.allclose(qnorm, 1, rtol=0, atol=1e-4)
    measured = np.asarray(input_row["right_observed_channel"])
    actual = np.asarray(teacher_row["actual_flange"])
    require(actual.shape == (50, 4, 4) and input_row["input_sha256"] == teacher_row["input_sha256"],
            "wrong offline teacher path")
    pred = pose_rows(arm) if valid_orientation else None
    start = pose_rows(measured[None])[0]
    horizons = {}
    for n in (10, 50):
        delta = arm[n-1, :3] - start[:3, 3]
        control = actual[n-1, :3, 3] - start[:3, 3]
        horizons[str(n)] = dict(
            predicted_delta_mm=(1000*delta).tolist(), actual_h0_delta_mm=(1000*control).tolist(),
            direction_cos=alignment(delta, control),
            flange_error=pose_error(pred[:n], actual[:n]) if valid_orientation else None,
            hold_error=pose_error(np.repeat(start[None], n, axis=0), actual[:n]),
        )
    return dict(horizons=horizons, grip=grip_forecast(arm, measured),
                orientation_unit_quaternion_valid=bool(valid_orientation),
                max_decoded_quaternion_norm_error=float(abs(qnorm-1).max()),
                raw_gripper_fraction_range=[float(arm[:, 7].min()), float(arm[:, 7].max())],
                raw_gripper_fraction_rows_outside_unit_range=int(((arm[:, 7] < 0) | (arm[:, 7] > 1)).sum()),
                physical_gripper_and_tcp_gate="UNSCORED_SOURCE_CODEC_UNAVAILABLE")


def score(run, ingress, control_path, output):
    cold, control = read(ingress), read(control_path)
    start, owner = (read(run / name) for name in ("START.json", "OWNER_EXIT.json"))
    require(start["cold_sha256"] == control["cold_sha256"] == digest(ingress)
            and start["teacher_sha256"] == digest(control_path)
            and not Path("/proc", str(start["owner_pid"])).exists()
            and owner["project_lock_released"] and owner["physics_ticks"] == owner["optimizer_updates"] == 0,
            "T2 owner or scoring inputs not sealed")
    rows = []
    for cell in cold["cells"]:
        name = cell["id"]
        entry, ret = (run / f"{name}.{suffix}.json" for suffix in ("entered", "returned"))
        if not ret.exists():
            rows.append(dict(**cell, status="TECHNICAL_ENTERED" if entry.exists() else "UNSTARTED"))
            continue
        e, r = read(entry), read(ret)
        require(all(e[k] == v for k, v in cell.items()), "returned cell differs from the scientific roster")
        with np.load(run / f"noise-{cell['seed'][1]}.npz", allow_pickle=False) as noise:
            require(tensor_sha(noise["noise"]) == cell["initial_noise_sha256"]
                    and noise["sample_key"].tolist() == e["sample_key"], "actual T2 noise differs")
        artifact = pin(run / f"{name}.npz", r["file_sha256"])
        with np.load(artifact, allow_pickle=False) as z:
            require(z["actions"].shape == (50, 32) and np.isfinite(z["actions"]).all(),
                    "invalid raw sampler artifact")
            absolute = z["absolute"]
            require(tensor_sha(z["actions"]) == r["raw_sha256"]
                    and tensor_sha(absolute) == r["decoded_sha256"], "returned T2 tensor hash differs")
            marks = measure(absolute, cold["rows"][cell["anchor"]], control["rows"][cell["anchor"]])
            rows.append(dict(**cell, status="RETURNED", **marks, output_sha256=r["file_sha256"],
                             raw_sha256=tensor_sha(z["actions"]), raw_dtype=str(z["actions_dtype"]),
                             decoded_sha256=tensor_sha(absolute)))
    counts = {label: sum(r["status"] == label for r in rows)
              for label in ("RETURNED", "TECHNICAL_ENTERED", "UNSTARTED")}
    require(counts["RETURNED"] == owner["returned"]
            and counts["RETURNED"]+counts["TECHNICAL_ENTERED"] == owner["entered"],
            "T2 scientific denominator differs from actual sampler records")
    write_new(output, dict(panel=cold["panel"], counts=counts, rows=rows, owner=owner,
                           physical_gate=cold["physical_action_admission"],
                           new_native_starts=0, physics_ticks=0, optimizer_updates=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("teacher", "score"))
    parser.add_argument("root", type=Path)
    parser.add_argument("ingress", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--control", type=Path)
    a = parser.parse_args()
    os.umask(0o077)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "T2 score uses CPU only")
    if a.mode == "teacher":
        teacher(a.root.resolve(), a.ingress, a.output)
    else:
        score(a.root, a.ingress, a.control, a.output)
