"""Read closed native records; attribute commands, effort and tracking without stepping a model."""

from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

import mujoco
import numpy as np
from b2_static_bias import effort_evidence, ros_analysis
from d1_factorial_reference_matrix import START, check
from go2arx5_vla.world_model.init_trace_reader import read_trace
from go2arx5_vla.world_model.native_replay import forbid_integration, packets
from identity import read, rows, sha

from go2arx5_vla.trajectory.so3 import geodesic_angle
from scripts.pi0292.eval_contract import matrix10

DT = 0.002
JOINTS = [f"x5_joint{i}" for i in range(1, 7)]


def ranges(v):
    x = np.asarray(v)
    return {
        "min": np.min(x, axis=0).tolist(),
        "mean": np.mean(x, axis=0).tolist(),
        "max": np.max(x, axis=0).tolist(),
    }


def angle(a, b):
    return float(np.degrees(geodesic_angle(a, b)))


def kinematics(model, data, q, tool, body):
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    return (
        data.site_xpos[tool].copy(),
        data.site_xmat[tool].reshape(3, 3).copy(),
        data.xpos[body].copy(),
        data.xmat[body].reshape(3, 3).copy(),
    )


def read_case(folder):
    spec = read(folder / "EXECUTION_SPEC.json")
    result = read(folder / "RESULT.private.json")
    out = folder / result["episode"]
    check(
        result["completed"]
        and read(folder / "control/LEASE.json")["status"] == "CLOSED",
        "only audit completed card here",
    )
    model_path = Path(spec["model_path"])
    check(sha(model_path) == spec["frozen_files"][str(model_path)], "frozen scene")
    state = packets(out / "state.bin.journal")
    trace = read_trace(out / "init.trace")
    atoms = [
        r
        for r in rows(out / "MOBILE_SAMPLES.private.jsonl")
        if r["native_start_step"] >= START
    ]
    end = atoms[-1]["native_end_step"]
    check(
        list(range(1, end + 1)) == [r["step"] for r in state],
        "missing or extra physics tick",
    )
    pd = sorted([r for r in trace if r["kind"] == 9], key=lambda r: r["monotonic_ns"])
    pd_ns = [r["monotonic_ns"] for r in pd]
    before = {r["token"]: r["monotonic_ns"] for r in trace if r["kind"] == 11}
    after = {r["token"]: r["monotonic_ns"] for r in trace if r["kind"] == 12}
    target = matrix10(atoms[0]["target_endpoint_EEF10"])
    check(
        all(
            np.allclose(
                matrix10(r["target_endpoint_EEF10"]), target, rtol=0, atol=1e-10
            )
            for r in atoms
        ),
        "not a fixed D1 target",
    )
    timing = ros_analysis(out, atoms, pd, after, trace)
    request_q = defaultdict(list)
    for r in rows(out / "ROS_CONSUMPTION.private.jsonl"):
        if r["topic_key"] == "arm_request":
            msg = r["message"]
            h = msg["header"]["stamp"]
            ts = h["sec"] + 1e-9 * h["nanosec"]
            request_q[tuple(msg["points"][0]["positions"])].append(ts)
    seen_PD = 0
    unknown_PD = 0
    for r in pd:
        if START * DT + 1e-8 < r["sim_time_s"] <= end * DT + 1e-8:
            seen_PD += 1
            if not any(
                t <= r["sim_time_s"] + 1e-8
                for t in request_q.get(tuple(r["values"][12:18]), [])
            ):
                unknown_PD += 1
    curve = []
    dense_effort = []
    mismatch = 0.0
    contacts = Counter()
    with forbid_integration(mujoco) as calls:
        model = mujoco.MjModel.from_binary_path(str(model_path))
        data = mujoco.MjData(model)
        body = model.body("base_link").id
        tool = model.site("x5_ee").id
        joints = [model.joint(n).id for n in JOINTS]
        qa = model.jnt_qposadr[joints]
        va = model.jnt_dofadr[joints]
        act = [model.actuator(n).id for n in JOINTS]
        check(
            np.array_equal(model.actuator_trnid[act, 0], joints)
            and np.all(model.actuator_gear[act, 0] == 1)
            and np.all(model.actuator_ctrllimited[act]),
            "motor mapping",
        )
        paired = effort_evidence(model, trace, after, end)
        endpoints = {r["native_end_step"]: r for r in atoms}
        for st in state[START - 1 :]:
            tick = st["step"]
            i = bisect_right(pd_ns, before[tick]) - 1
            check(
                i >= 0 and i == bisect_right(pd_ns, after[tick]) - 1,
                "PD changed during integration",
            )
            m = np.array(pd[i]["values"]).reshape(5, 18)[:, 12:]
            q = st["qpos"]
            dq = st["qvel"]
            x, s, b, r = kinematics(model, data, q, tool, body)
            qc = q.copy()
            qc[qa] = m[0]
            xc, _, _, _ = kinematics(model, data, qc, tool, body)
            if tick == START:
                x0 = x.copy()
                xc0 = xc.copy()
                local0 = r.T @ (x - b)
            if tick in endpoints:
                mismatch = max(
                    mismatch,
                    float(
                        np.max(abs(x - np.array(endpoints[tick]["actual_TCP"])[:3, 3]))
                    ),
                )
            offset = 1000 * (x - target[:3, 3])
            delta = 1000 * (x - x0)
            base_only = 1000 * (b + r @ local0 - x0)
            gap = 1000 * (xc - x)
            pdref = 1000 * (xc - target[:3, 3])
            check(np.allclose(offset, pdref - gap, rtol=0, atol=1e-8), "FK closure")
            contacts[len(st["contacts"])] += 1
            # This replay estimates the static gravity requirement. Actual motor inputs come from the native journal.
            data.qpos[:] = q
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            p = m[2] * (m[0] - q[qa])
            d = m[3] * (m[1] - dq[va])
            ctrl = st["ctrl"][act]
            lim = np.where(
                ctrl >= 0,
                model.actuator_ctrlrange[act, 1],
                -model.actuator_ctrlrange[act, 0],
            )
            effort = {
                "q_error_rad": m[0] - q[qa],
                "dq_error_rad_s": m[1] - dq[va],
                "Kp": m[2],
                "Kd": m[3],
                "P_estimate_Nm": p,
                "D_estimate_Nm": d,
                "explicit_ff_Nm": m[4],
                "gravity_zero_speed_Nm": data.qfrc_bias[va].copy(),
                "recorded_input_Nm": ctrl,
                "fraction_of_input_limit": np.abs(ctrl) / lim,
                "observed_minus_estimated_Nm": ctrl - p - d - m[4],
            }
            dense_effort.append(effort)
            curve.append(
                {
                    "tick": tick,
                    "elapsed_ms": 2 * (tick - START),
                    "absolute_error_mm": offset.tolist(),
                    "incremental_error_mm": delta.tolist(),
                    "pd_fk_delta_mm": (1000 * (xc - xc0)).tolist(),
                    "pd_fk_minus_actual_mm": gap.tolist(),
                    "pd_fk_minus_reference_mm": pdref.tolist(),
                    "base_only_delta_mm": base_only.tolist(),
                    "reference_orientation_error_deg": angle(s, target[:3, :3]),
                    "pd_monotonic_ns": pd[i]["monotonic_ns"],
                    "actual_q_rad": q[qa].tolist(),
                    "desired_q_rad": m[0].tolist(),
                    "actual_velocity_rad_s": dq[va].tolist(),
                    "desired_velocity_rad_s": m[1].tolist(),
                    "effort": {k: v.tolist() for k, v in effort.items()},
                    "contacts": len(st["contacts"]),
                }
            )
    check(not calls and mismatch < 1e-10, "offline native FK consistency")
    summary = {}
    for ms in (0, 100, 250, 500, 1000):
        point = curve[ms // 2]
        summary[str(ms)] = {
            k: point[k]
            for k in (
                "absolute_error_mm",
                "incremental_error_mm",
                "pd_fk_delta_mm",
                "pd_fk_minus_actual_mm",
                "pd_fk_minus_reference_mm",
                "base_only_delta_mm",
                "reference_orientation_error_deg",
            )
        }
    for k in ("pd_fk_delta_mm", "incremental_error_mm"):
        summary[k + "_first_minus10mm_ms"] = next(
            (r["elapsed_ms"] for r in curve if r[k][2] <= -10), None
        )

    def rms(key):
        return float(np.sqrt(np.mean([np.dot(r[key], r[key]) for r in curve[1:]])))

    control = {k: ranges([r[k] for r in dense_effort[1:]]) for k in dense_effort[0]}
    return curve, {
        "actual_ticks": end,
        "test_ticks": end - START,
        "milestones": summary,
        "absolute_position_rms_mm": rms("absolute_error_mm"),
        "incremental_position_rms_mm": rms("incremental_error_mm"),
        "orientation_rms_deg": float(
            np.sqrt(
                np.mean([r["reference_orientation_error_deg"] ** 2 for r in curve[1:]])
            )
        ),
        "Y_actual_z_100ms_mm": curve[50]["incremental_error_mm"][2],
        "Y_PD_z_100ms_mm": curve[50]["pd_fk_delta_mm"][2],
        "independent_fk_max_difference_mm": 1000 * mismatch,
        "source_timing": timing,
        "applied_PD_callbacks": seen_PD,
        "applied_PD_q_without_matching_known_request": unknown_PD,
        "selected_MRT_policy_ID": "POLICY_ID_UNOBSERVED",
        "selected_policy_age_ms": None,
        "torque_before_clamp_native": None,
        "actuation_per_tick_summary": control,
        "paired_PD_callback_audit": paired,
        "model_sha256": sha(model_path),
        "joints": JOINTS,
        "gear": model.actuator_gear[act, 0].tolist(),
        "arm_native_ctrlrange": model.actuator_ctrlrange[act].tolist(),
        "seal_sha256": sha(folder / "RESULT.private.json"),
        "inputs_sha256": {
            n: sha(out / n)
            for n in (
                "state.bin.journal",
                "init.trace",
                "MOBILE_SAMPLES.private.jsonl",
                "ROS_CONSUMPTION.private.jsonl",
                "planner.jsonl",
                "physics.jsonl",
                "LOADED_EXECUTOR_00.private.json",
                "CLOCK_RESUME_REQUESTS.jsonl",
            )
        },
    }
