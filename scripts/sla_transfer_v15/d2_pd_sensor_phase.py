"""Compare PD torque against callback state and the preceding native integration's joint sensor phase."""

import argparse
from pathlib import Path

from d1_readiness import B2_PIN, baseline, configure, pinned, read, record, sha


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("root", type=Path)
    cli.add_argument("out", type=Path)
    args = cli.parse_args()
    root = args.root.resolve()
    configure(baseline(root))
    import mujoco
    import numpy as np
    from d1_factorial_reference_matrix import check
    from go2arx5_vla.world_model.init_trace_reader import read_trace

    site = root / "v14/b2-technical-20261123-01"
    manifest = site / "B2_PUBLIC_READOUT.json"
    pinned(manifest, B2_PIN)
    out = {}
    for spec in read(manifest)["cards"]:
        path = site / spec["card"]
        pinned(path / "EXECUTION_SPEC.json", spec["spec_sha256"])
        cfg = read(path / "EXECUTION_SPEC.json")
        p = Path(cfg["model_path"])
        pinned(p, cfg["frozen_files"][str(p)])
        model = mujoco.MjModel.from_binary_path(str(p))
        arm = [model.joint("x5_joint" + str(i)).id for i in range(1, 7)]
        qa = model.jnt_qposadr[arm]
        va = model.jnt_dofadr[arm]
        act = model.actuator_trnid[:, 0]
        motors = [model.actuator("x5_joint" + str(i)).id for i in range(1, 7)]
        check(
            np.array_equal(motors, np.arange(12, 18))
            and np.array_equal(act[motors], arm),
            "motor indexing",
        )
        for i, j in enumerate(arm):
            for address, kind in (
                (12 + i, mujoco.mjtSensor.mjSENS_JOINTPOS),
                (30 + i, mujoco.mjtSensor.mjSENS_JOINTVEL),
            ):
                idx = np.flatnonzero(model.sensor_adr == address)
                check(
                    len(idx) == 1
                    and model.sensor_dim[idx[0]] == 1
                    and model.sensor_objid[idx[0]] == j
                    and model.sensor_type[idx[0]] == kind,
                    "joint sensor ABI",
                )
        tr = path / read(path / "RESULT.private.json")["episode"] / "init.trace"
        pinned(tr, spec["evidence_sha256"]["init.trace"])
        trace = read_trace(tr)
        before = {r["token"]: r for r in trace if r["kind"] == 11}
        after = {r["token"]: r for r in trace if r["kind"] == 12}
        latest = None
        res = {
            k: []
            for k in (
                "callback_state",
                "one_step_old",
                "two_steps_old",
                "old_velocity_only",
                "old_position_only",
            )
        }
        effort = []
        phase_aware_effort = []
        per = []
        for r in trace:
            if r["kind"] == 9:
                latest = r
            if (
                r["kind"] != 10
                or not 11.406 + 1e-8
                < r["sim_time_s"]
                <= spec["actual_ticks"] * 0.002 + 1e-8
            ):
                continue
            tick = round(r["sim_time_s"] / 0.002)
            check(
                abs(tick * 0.002 - r["sim_time_s"]) < 1e-8
                and latest["sim_time_s"] == r["sim_time_s"]
                and latest["monotonic_ns"] <= r["monotonic_ns"],
                "PD pairing",
            )
            check(
                after[tick]["monotonic_ns"] <= r["monotonic_ns"]
                and (
                    tick + 1 not in before
                    or before[tick + 1]["monotonic_ns"] >= r["monotonic_ns"]
                ),
                "integration crossed callback",
            )
            qc, dqc, kp, kd, ff = np.array(latest["values"]).reshape(5, 18)[:, 12:]
            state = np.array(r["values"])
            control = state[66:][motors]
            x0, y0 = state[qa], state[34 + va]
            previous = np.array(before[tick]["values"])
            older = np.array(before[tick - 1]["values"])
            xp, yp = previous[qa], previous[34 + va]
            opts = {
                "callback_state": (x0, y0),
                "one_step_old": (xp, yp),
                "two_steps_old": (older[qa], older[34 + va]),
                "old_velocity_only": (x0, yp),
                "old_position_only": (xp, y0),
            }
            for key, (x, y) in opts.items():
                res[key].append(abs(control - (kp * (qc - x) + kd * (dqc - y) + ff)))
            effort.append([kp * (qc - xp), kd * (dqc - yp), ff, control])
            x, y = (x0, y0) if (tick - 5703) % 25 == 0 else (xp, yp)
            phase_aware_effort.append([kp * (qc - x), kd * (dqc - y), ff, control])
            per.append(tick)
        callback = np.array(res["callback_state"])
        lagged = np.array(res["one_step_old"])
        ticks = np.asarray(per)
        endpoint = (ticks - 5703) % 25 == 0
        last_callback = np.r_[np.diff(ticks) != 0, True]
        res["current_on_25tick_boundary_else_lag1"] = np.where(
            endpoint[:, None], callback, lagged
        )
        res["current_on_last_boundary_callback_else_lag1"] = np.where(
            (endpoint & last_callback)[:, None], callback, lagged
        )
        phase_counts = {}
        for boundary in (False, True):
            flags = endpoint == boundary
            phase_counts[str(boundary)] = {
                "callback_pairs": int(flags.sum()),
                "current_exact": int(
                    ((np.max(callback, axis=1) <= 1e-6) & flags).sum()
                ),
                "lag1_exact": int(((np.max(lagged, axis=1) <= 1e-6) & flags).sum()),
                "neither": int(
                    (
                        (np.max(lagged, axis=1) > 1e-6)
                        & (np.max(callback, axis=1) > 1e-6)
                        & flags
                    ).sum()
                ),
            }
        metrics = {}
        for key, values in res.items():
            residual = np.array(values)
            metrics[key] = {
                "max_abs_Nm": float(residual.max()),
                "RMS_Nm": float(np.sqrt(np.mean(residual**2))),
                "max_by_joint_Nm": residual.max(axis=0).tolist(),
                "all_six_within_1e_minus_6_Nm": int(
                    np.sum(np.max(residual, axis=1) <= 1e-6)
                ),
            }
        out[spec["card"]] = {
            "native_callback_pairs": len(per),
            "distinct_completed_ticks": len(set(per)),
            "sensor_phase_hypotheses": metrics,
            "phase_aware_callback_mean_P_D_ff_actual_by_joint_Nm": np.mean(
                phase_aware_effort, axis=0
            ).tolist(),
            "at_25_tick_boundary": phase_counts,
            "one_step_old_mean_P_D_ff_actual_by_joint_Nm": np.mean(
                effort, axis=0
            ).tolist(),
            "native_trace_sha256": sha(tr),
            "model_sha256": sha(p),
        }
    record(
        args.out,
        {
            "scope": "six native arm servo channels in sealed post-handoff B2 intervals; no directly recorded sensordata",
            "clock": "command and control from paired PD callbacks; lag 1 uses STEP_BEFORE of most recently completed integration (2ms); lag 2 is a negative control",
            "cases": out,
            "new_physics_ticks": 0,
            "new_native_requests": 0,
            "new_model_forwards": 0,
            "new_updates": 0,
        },
    )


if __name__ == "__main__":
    main()
