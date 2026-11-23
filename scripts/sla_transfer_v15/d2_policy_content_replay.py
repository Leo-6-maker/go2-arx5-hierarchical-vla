"""Test whether archived solver messages reproduce the position AND velocity of published arm commands."""

import argparse
from pathlib import Path

import numpy as np
from d1_readiness import B2_PIN, baseline, pinned, read, record, rows, sha

SOURCE = {
    "go2_x5_ocs2_node.cpp": "1caf2f24fc18a6210d0ff7c81c46e10e9bf53623248e56d8e0a9ee4a50aa5e68",
    "WbcBridgeCore.cpp": "bc21fd6d32754d3713479205a035e5377e5557c061d009770a28c7b5cf2b1f0e",
    "wbc_bridge_core.hpp": "5ddf926e4105e973cc1c9e4c68585244f9bdc172bd2a420e14a90331da83d79e",
    "FeedforwardController.cpp": "347c7e3746e9b7a476c0afe6c35148c419ee590447910cf24ff13fb0dddd9178",
    "MRT_BASE.cpp": "fddfcd76f9a7b55d21c2b30f07ea625aff4d111ea7c3136e087f5558bc22fe27",
}
TOL = 2e-6


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def interpolate(grid, values, t):
    indexes = np.clip(np.searchsorted(grid, t, side="right") - 1, 0, len(grid) - 2)
    fraction = np.clip(
        (t - grid[indexes]) / (grid[indexes + 1] - grid[indexes]), 0.0, 1.0
    )
    return values[indexes] + fraction[:, None] * (values[indexes + 1] - values[indexes])


def stamp(msg):
    h = msg["header"]["stamp"]
    return h["sec"] + 1e-9 * h["nanosec"]


def percentiles(x):
    return (
        dict(
            zip(
                ("min", "median", "p95", "max"),
                np.percentile(x, [0, 50, 95, 100]).tolist(),
                strict=True,
            )
        )
        if len(x)
        else None
    )


def analyze(folder, receipt, t0, t1):
    requests = []
    policies = []
    obs = []
    for r in rows(receipt):
        m = r["message"]
        k = r["topic_key"]
        if k == "arm_request" and t0 + 1e-8 < stamp(m) <= t1 + 1e-8:
            require(
                m["joint_names"] == ["x5_joint" + str(i) for i in range(1, 7)]
                and len(m["points"]) == 1,
                "arm message ABI",
            )
            pt = m["points"][0]
            requests.append(
                (stamp(m), r["received_monotonic_s"], pt["positions"], pt["velocities"])
            )
        elif (
            k == "policy"
            and m["time_trajectory"]
            and m["time_trajectory"][-1] >= t0
            and m["init_observation"]["time"] <= t1 + 1e-8
        ):
            policies.append((m, r["received_monotonic_s"]))
        elif k == "observation" and t0 - 0.1 <= m["time"] <= t1 + 1e-8:
            obs.append(m["time"])
    t = np.array([r[0] for r in requests])
    actual_q = np.array([r[2] for r in requests])
    actual_v = np.array([r[3] for r in requests])
    require(len(requests) > 0 and len(policies) > 0, "empty window")
    bins = [[] for _ in requests]
    control = [set(), set()]
    position_only = set()
    velocity_only = set()
    duplicates = {
        j: max(v for v in obs if v < time - 1e-8)
        for j, time in enumerate(t)
        if np.count_nonzero(t == time) > 1 and any(v < time - 1e-8 for v in obs)
    }
    earlier = {j: [] for j in duplicates}
    tested = 0
    for serial, (msg, arrival) in enumerate(policies):
        grid = np.asarray(msg["time_trajectory"])
        src = msg["init_observation"]["time"]
        require(
            msg["controller_type"] == 1
            and len(grid) >= 2
            and np.all(np.diff(grid) > 0),
            "unsupported solver serialization",
        )
        q = np.array([a["value"] for a in msg["state_trajectory"]])
        ff = np.array([a["data"] for a in msg["data"]])
        require(
            q.shape == ff.shape == (len(grid), 12), "not the frozen 12D floating model"
        )
        ids = np.flatnonzero(
            (src <= t + 1e-8)
            & (grid[0] <= t + 0.01 + 1e-8)
            & (grid[-1] >= t + 0.01 - 1e-8)
        )
        for j, previous_stamp in duplicates.items():
            if (
                src > previous_stamp + 1e-8
                or not grid[0] - 1e-8 <= previous_stamp + 0.01 <= grid[-1] + 1e-8
            ):
                continue
            query = np.array([previous_stamp])
            pq = np.clip(interpolate(grid, q[:, 6:], query + 0.15), -3.2, 3.7).astype(
                np.float32
            )[0]
            pv = (0.4 * interpolate(grid, ff[:, 6:], query + 0.01)).astype(np.float32)[
                0
            ]
            if np.array_equal(pq, actual_q[j]) and np.array_equal(pv, actual_v[j]):
                earlier[j].append(
                    {
                        "solver_ordinal": serial,
                        "age_at_observation_ms": float(1000 * (previous_stamp - src)),
                    }
                )
        if not len(ids):
            continue
        tt = t[ids]
        tested += len(ids)
        v = (0.4 * interpolate(grid, ff[:, 6:], tt + 0.01)).astype(np.float32)
        qhat = np.clip(interpolate(grid, q[:, 6:], tt + 0.15), -3.2, 3.7).astype(
            np.float32
        )
        ev = np.max(abs(v - actual_v[ids]), axis=1)
        eq = np.max(abs(qhat - actual_q[ids]), axis=1)
        velocity_only.update(ids[ev <= TOL].tolist())
        position_only.update(ids[eq <= TOL].tolist())
        for i in np.flatnonzero((ev <= TOL) & (eq <= TOL)):
            j = ids[i]
            bins[j].append(
                {
                    "solver_message_ordinal": serial,
                    "age_ms": float(1000 * (t[j] - src)),
                    "observer_received_before_request": arrival <= requests[j][1],
                    "q_error_rad": float(eq[i]),
                    "dq_error_rad_s": float(ev[i]),
                }
            )
        for k, preview in enumerate((0.01, 0.30)):
            other = interpolate(grid, q[:, 6:], tt + preview)
            control[k].update(
                ids[
                    (ev <= TOL) & (np.max(abs(other - actual_q[ids]), axis=1) <= TOL)
                ].tolist()
            )
    unique = [b[0] for b in bins if len(b) == 1]
    result = {
        "start_sim_s": t0,
        "end_sim_s": t1,
        "requests": len(requests),
        "solver_messages_examined": len(policies),
        "eligible_message_request_pairs": tested,
        "requests_with_source_observation_at_request_stamp": sum(
            any(abs(v - x) < 1e-8 for v in obs) for x in t
        ),
        "requests_with_position_match": len(position_only),
        "requests_with_velocity_match": len(velocity_only),
        "joint_matches_unique": len(unique),
        "joint_matches_ambiguous": sum(len(b) > 1 for b in bins),
        "joint_matches_missing": sum(not b for b in bins),
        "unique_match_candidate_age_ms": percentiles([x["age_ms"] for x in unique]),
        "unique_receiver_observed_before_request": sum(
            x["observer_received_before_request"] for x in unique
        ),
        "max_unique_q_error_rad": max((x["q_error_rad"] for x in unique), default=None),
        "max_unique_dq_error_rad_s": max(
            (x["dq_error_rad_s"] for x in unique), default=None
        ),
        "negative_control_any_joint_match_by_arm_preview": {
            "10ms": len(control[0]),
            "300ms": len(control[1]),
        },
        "duplicate_stamp_secondary_sensitivity": [
            {
                "request_ordinal": j,
                "header_s": t[j],
                "previous_distinct_observation_s": v,
                "strict_header_matches": len(bins[j]),
                "exact_12_field_matches_at_previous_observation": earlier[j],
            }
            for j, v in duplicates.items()
        ],
        "source_ros_sha256": sha(receipt),
        "selected_policy_id": None,
    }
    record(
        folder / "candidate_matches.private.json",
        {"request_sim_s": t.tolist(), "matches": bins},
    )
    result["full_private_match_sha256"] = sha(folder / "candidate_matches.private.json")
    return result


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("root", type=Path)
    cli.add_argument("out", type=Path)
    args = cli.parse_args()
    root = args.root.resolve()
    s = baseline(root)
    src = Path(s["native_root"]) / "src"
    for n, pin in SOURCE.items():
        pinned(next(src.rglob(n)), pin)
    b2 = root / "v14/b2-technical-20261123-01"
    p = b2 / "B2_PUBLIC_READOUT.json"
    pinned(p, B2_PIN)
    manifest = read(p)
    out = args.out.resolve()
    out.mkdir(exist_ok=False, parents=True)
    cases = {}
    for info in manifest["cards"]:
        folder = b2 / info["card"]
        pinned(folder / "RESULT.private.json", info["result_sha256"])
        r = read(folder / "RESULT.private.json")
        n = folder / r["episode"]
        path = n / "ROS_CONSUMPTION.private.jsonl"
        pinned(path, info["evidence_sha256"][path.name])
        cfg = read(folder / "EXECUTION_SPEC.json")
        pinned(Path(cfg["controller_binary"]), manifest["frozen_sha256"]["controller"])
        commands = read(n / "PROCESS_COMMANDS.private.json")
        argv = next(c["argv"] for c in commands if c["name"] == "controller")
        overrides = [v.split(":=", 1)[0] for v in argv if ":=" in v]
        require(
            "--params-file" not in argv
            and set(overrides) <= {"use_sim_time", "urdfFile", "libFolder", "taskFile"},
            "unexpected parameter override",
        )
        f = out / info["card"]
        f.mkdir()
        require(r["journal_integrations"] == info["actual_ticks"], "sealed count")
        cases[info["card"]] = analyze(
            f, path, 5703 * 0.002, info["actual_ticks"] * 0.002
        )
    record(
        out / "D2_POLICY_CONTENT_REPLAY_ACTUAL.json",
        {
            "scope": "post-hoc source-compatible decode of observed solver messages; IDs and actual internal selection unobserved",
            "source_sha256": SOURCE,
            "controller_sha256": manifest["frozen_sha256"]["controller"],
            "tolerance_q_rad_and_dq_rad_s": TOL,
            "clock": "arm request ROS simulation stamp; solver initial observation time; receipt wall order retained separately",
            "hypothesis": "q is six last state coordinates at +150ms, dq is 0.4 times serialized feedforward at +10ms; test all recorded time-admissible messages",
            "cases": cases,
            "new_native_requests": 0,
            "new_ticks": 0,
            "new_model_forwards": 0,
            "new_updates": 0,
        },
    )


if __name__ == "__main__":
    main()
