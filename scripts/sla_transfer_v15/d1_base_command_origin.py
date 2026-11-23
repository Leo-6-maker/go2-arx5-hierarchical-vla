"""Join actual frozen leg inputs to native WBC base messages during the initial transition."""

import argparse
from bisect import bisect_right
from pathlib import Path

import numpy as np
from d1_gait_replay import TRACE, population, require
from d1_readiness import baseline, pinned, read, record, rows, sha


def inspect(folder, trace_sha, cfg):
    trace = folder / "init.trace"
    pinned(trace, trace_sha)
    with trace.open("rb") as f:
        f.read(16)
        events = [
            r
            for r in TRACE.iter_unpack(f.read())
            if r[3] in (3, 5, 9, 11, 12) and 4.9 <= r[2] < 5.42
        ]
    completed = sorted((r[1] * 1e-9, r[9]) for r in events if r[3] == 12)
    walls, ticks = zip(*completed, strict=True)
    vectors = {}
    above = []
    n = 0
    ros = folder / "ROS_CONSUMPTION.private.jsonl"
    for e in rows(ros):
        wall = e["received_monotonic_s"]
        if not walls[0] <= wall <= walls[-1] or e["topic_key"] != "base_command":
            continue
        x, y, yaw = e["message"]["data"][:3]
        v = (x, y, yaw)
        n += 1
        after = ticks[bisect_right(walls, wall) - 1]
        vectors.setdefault(v, []).append((wall, after))
        if np.linalg.norm(v) >= 0.1:
            above.append(after)
    gait = []
    pd = sorted((e for e in events if e[3] == 9), key=lambda e: e[1])
    pd_times = [e[1] for e in pd]
    step_after = {e[9]: e for e in events if e[3] == 12}
    scale = np.asarray(cfg["action_scale"][:12])
    default = np.asarray(cfg["default_dof_pos"][:12])
    for r in events:
        if r[3] != 3 or r[10 + 45] == 0:
            continue
        key = tuple(np.array(r[10 + 39 : 10 + 42]) / [2, 2, 0.25])
        matches = vectors.get(key, [])
        output = next(e for e in events if e[3] == 5 and e[9] == r[9])
        expected = default + scale * np.array(output[10:22])
        applied = []
        errors = []
        for step in events:
            if step[3] != 11 or step[7] != r[9] or step[9] not in step_after:
                continue
            idx = bisect_right(pd_times, step[1]) - 1
            if (
                idx < 0
                or pd[idx][7] != r[9]
                or idx != bisect_right(pd_times, step_after[step[9]][1]) - 1
            ):
                continue
            errors.append(float(np.max(abs(expected - np.array(pd[idx][10:22])))))
            applied.append(step[9])
        require(applied and max(errors) < 2e-6, "applied leg target mismatch")
        gait.append(
            {
                "leg_policy_token": r[9],
                "matched_integrations": len(applied),
                "first_last_applied_tick": [min(applied), max(applied)],
                "max_PD_target_error_rad": max(errors),
                "sim_s": r[2],
                "unscaled_command_norm": float(np.linalg.norm(key)),
                "exact_three_channel_candidates": len(matches),
                "observer_tick": matches[0][1] if len(matches) == 1 else None,
                "observer_minus_inference_wall_ms": 1000 * (matches[0][0] - r[1] * 1e-9)
                if len(matches) == 1
                else None,
            }
        )
    return {
        "received_base_messages_in_window": n,
        "source_norm_at_or_above_point1": len(above),
        "observer_tick_range_above": [min(above), max(above)] if above else None,
        "gait_input_matches": gait,
        "ROS_sha256": sha(ros),
        "trace_sha256": trace_sha,
    }


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("root", type=Path)
    cli.add_argument("out", type=Path)
    a = cli.parse_args()
    root = a.root.resolve()
    ref = root / "v15/gait-audit01/D1_GAIT_INPUT_REPLAY_ACTUAL.json"
    import yaml

    cfg_path = next(
        Path(k) for k in baseline(root)["frozen_files"] if Path(k).name == "config.yaml"
    )
    previous = read(ref)
    pinned(cfg_path, previous["leg_config_sha256"])
    cfg = next(iter(yaml.safe_load(cfg_path.read_text()).values()))
    cases = []
    for (name, label, n, pin), old in zip(
        population(root), previous["cases"], strict=True
    ):
        require(old["case"] == name and old["group"] == label, "case join")
        row = {"case": name, "group": label}
        cases.append(row)
        if n is None:
            continue
        row.update(inspect(n, pin, cfg))
        require(
            [x["leg_policy_token"] for x in row["gait_input_matches"]]
            == old["walk_tokens_through_boundary_or_stop"],
            "event coverage",
        )
    record(
        a.out,
        {
            "source_gait_replay_sha256": sha(ref),
            "scope": "all observed nonzero frequency events through the boundary fell in this transition; six-component source packets matched on the three planar/yaw channels entering the compiled threshold",
            "observer_window": "from first to last native post-step in [4.9,5.42) simulation seconds or earlier historical stop; base messages have no simulation header",
            "cases": cases,
            "new_native_requests": 0,
            "new_physics_ticks": 0,
            "new_model_forwards": 0,
            "new_updates": 0,
        },
    )


if __name__ == "__main__":
    main()
