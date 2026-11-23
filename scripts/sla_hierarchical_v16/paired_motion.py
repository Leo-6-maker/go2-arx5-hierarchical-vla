"""Compare adjacent archived model proposals on identical future native clock ticks."""

from collections import defaultdict
from pathlib import Path
import argparse
import json

import numpy as np

from demonstrated_grip import pin, read, require
from revision_replay import primary_calls
from sealed_panel import cases, configure_site

REPORTS = Path(__file__).resolve().parents[2] / "reports/SLA_HIERARCHICAL_V16"


def aligned(plan, actual):
    """Keep spatial rotation and translation increments in their original world axes."""
    out = plan.copy()
    out[:, :3, 3] = actual[:3, 3] + (plan[:, :3, 3] - plan[0, :3, 3])
    out[:, :3, :3] = plan[:, :3, :3] @ plan[0, :3, :3].T @ actual[:3, :3]
    return out


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(a @ b / denom, -1, 1)) if denom else None


def path_errors(path, truth):
    from compiler import relative_angle

    dp = (truth[-1, :3, 3] - truth[0, :3, 3]) * 1000
    error = np.linalg.norm(path[1:, :3, 3] - truth[1:, :3, 3], axis=1) * 1000
    rot = np.rad2deg(relative_angle(path[1:, :3, :3], truth[1:, :3, :3]))
    return dict(
        position_rmse_mm=float(np.sqrt(np.mean(error**2))),
        rotation_rmse_deg=float(np.sqrt(np.mean(rot**2))),
        endpoint_position_mm=float(error[-1]),
        endpoint_rotation_deg=float(rot[-1]),
        displacement_cosine=cosine(path[-1, :3, 3] - path[0, :3, 3], dp),
    )


def score_horizon(old, new, truth):
    """Rows start at the same native boundary; only strictly future errors enter RMSE."""
    require(
        old.shape == new.shape == truth.shape
        and old.shape[1:] == (4, 4)
        and len(truth) >= 2
        and np.isfinite([old, new, truth]).all(),
        "invalid common horizon",
    )
    actual = truth[0]
    models = dict(
        old=old,
        new=new,
        old_aligned=aligned(old, actual),
        new_aligned=aligned(new, actual),
        hold=np.broadcast_to(actual, truth.shape),
    )
    dp = (truth[-1, :3, 3] - actual[:3, 3]) * 1000
    out = {name: path_errors(path, truth) for name, path in models.items()}
    return dict(
        future_observations=len(truth) - 1,
        outcomes=out,
        achieved_delta_mm=dp.tolist(),
        achieved_arc_mm=float(
            np.linalg.norm(np.diff(truth[:, :3, 3], axis=0), axis=1).sum() * 1000
        ),
    )


def truth_for_episode(root, rows, meta, pins):
    import mujoco
    from go2arx5_vla.world_model.multitask_state_v6 import TwoObjectState, packets
    from go2arx5_vla.world_model.native_replay import forbid_integration

    case = rows[0]
    name = case["episode"]
    selected = case["spec"]["multitask_geometry"]["objects"]
    require(set(selected) == {"red", "blue"}, "two-object scene required")
    receipt = read(pin(Path(meta["receipt_path"]), meta["receipt_sha256"]))
    require(receipt["id"] == name and receipt["split"] == "TRAIN", "teacher scope")
    sp = next(Path(p) for p in pins if p.endswith("/" + name + "/EXECUTION_SPEC.json"))
    pin(sp, receipt["source_spec_sha256"])
    n = sp.parent / "native-01"
    journal = pin(n / "state.bin.journal", receipt["source_files"]["state.bin.journal"])
    samples = n / "MOBILE_SAMPLES.private.jsonl"
    pin(samples, pins[str(samples)])
    with samples.open() as src:
        end = None
        for line in src:
            a = json.loads(line)
            require(
                (end is None or end == a["native_start_step"])
                and a["native_end_step"] - a["native_start_step"] == 25,
                "teacher interval gap",
            )
            end = a["native_end_step"]
    wanted = {
        c["window"]["tick"] + 50 * k
        for c in rows
        for k in range(12)
        if c["window"]["tick"] + 50 * k <= end
    }
    queries = {c["window"]["tick"] for c in rows}
    dense = root / "v12/DENSE" / name / "frames.jsonl"
    with pin(dense, receipt["frames_sha256"]).open() as f:
        frames = {
            r["tick"]: r for line in f if (r := json.loads(line))["tick"] in queries
        }
    require(frames.keys() == queries, "missing dense query")
    model = mujoco.MjModel.from_binary_path(
        str(pin(Path(case["spec"]["model_path"]), receipt["scene_sha256"]))
    )
    terminal = max(wanted)
    color = case["instruction"].split("-")[1]
    observer = TwoObjectState(model, color)
    states = {}
    prev = 0
    with forbid_integration(mujoco) as integrations:
        for packet in packets(journal):
            t = packet["step"]
            require(
                t == prev + 1 and abs(packet["time_s"] - 0.002 * t) < 1e-8,
                "missing native clock",
            )
            prev = t
            if t in wanted:
                a = observer.update(packet)
                tcp = np.asarray(a["tcp"]).copy()
                if t in queries:
                    require(
                        np.allclose(tcp, frames[t]["T_W_TCP"], rtol=0, atol=1e-9),
                        "sensor TCP disagrees",
                    )
                states[t] = dict(
                    tcp=tcp,
                    objects={
                        k: obj["pose"][:3, 3].copy() for k, obj in a["objects"].items()
                    },
                )
            if t == terminal:
                break
    require(
        not integrations and states.keys() == wanted,
        "incomplete native replay",
    )
    for c in rows:
        t = c["window"]["tick"]
        for k, obj in c["query"]["actual_state"]["objects"].items():
            require(
                np.allclose(
                    states[t]["objects"][k],
                    np.asarray(obj["pose"])[:3, 3],
                    rtol=0,
                    atol=1e-9,
                ),
                "query object disagrees",
            )
    return states, dict(
        episode=name,
        teacher_end_tick=end,
        native_states=len(states),
        query_tcp_crosschecks=len(queries),
        native_journal_sha256=receipt["source_files"]["state.bin.journal"],
        scene_sha256=receipt["scene_sha256"],
        dense_frames_sha256=receipt["frames_sha256"],
        consumed_samples_sha256=pins[str(samples)],
    )


def evaluate_pair(a, b, states, end):
    t0, t1 = (c["window"]["tick"] for c in (a, b))
    require(
        a["episode"] == b["episode"] and t1 - t0 == 250,
        "different episode or wrong offset",
    )
    future = [k for k in range(12) if t1 + 50 * k <= end]
    ticks = np.array(future)
    old = a["curve"].knots[15 + 3 * ticks]
    new = b["curve"].knots[3 * ticks]
    truth = np.stack([states[t1 + 50 * k]["tcp"] for k in future])
    err = (truth[0, :3, 3] - old[0, :3, 3]) * 1000
    revision = (new[0, :3, 3] - old[0, :3, 3]) * 1000
    start = states[t0]
    now = states[t1]
    objects = {}
    for k, pos in start["objects"].items():
        cur = now["objects"][k]
        objects[k] = dict(
            displacement_mm=((cur - pos) * 1000).tolist(),
            tcp_distance_change_mm=float(
                1000
                * (
                    np.linalg.norm(now["tcp"][:3, 3] - cur)
                    - np.linalg.norm(start["tcp"][:3, 3] - pos)
                )
            ),
        )
    return dict(
        old=a["key"],
        new=b["key"],
        episode=b["episode"],
        instruction=b["instruction"],
        phases=[a["archive"]["phase"], b["archive"]["phase"]],
        query_ticks=[t0, t1],
        raw_sha256=[a["raw_file_sha256"], b["raw_file_sha256"]],
        scored_through_tick=t1 + 50 * future[-1],
        past_tcp_delta_mm=((now["tcp"][:3, 3] - start["tcp"][:3, 3]) * 1000).tolist(),
        observed_objects=objects,
        boundary=dict(
            old_position_error_mm=float(np.linalg.norm(err)),
            new_position_error_mm=float(np.linalg.norm(err - revision)),
            revision_mm=revision.tolist(),
            old_to_truth_mm=err.tolist(),
            revision_projection_mm=float(revision @ err / np.linalg.norm(err))
            if np.linalg.norm(err)
            else None,
            revision_cosine=cosine(revision, err),
        ),
        primary=score_horizon(old[:6], new[:6], truth[:6])
        if len(future) >= 6
        else None,
        secondary=score_horizon(old, new, truth) if len(future) == 12 else None,
    )


def contrasted(pairs, horizon, metric, first, second):
    values = [
        r[horizon]["outcomes"][first][metric] - r[horizon]["outcomes"][second][metric]
        for r in pairs
        if r[horizon] is not None
    ]
    if not values:
        return None
    x = np.asarray(values)
    return dict(
        n=len(values),
        lower=int((x < 0).sum()),
        equal=int((x == 0).sum()),
        higher=int((x > 0).sum()),
        median_difference=float(np.median(x)),
    )


def main(root, measurement_file=None):
    reg = read(REPORTS / "PAIRED_MOTION_REGISTRATION.json")
    roster = read(pin(REPORTS / "H1_REVISION_ACTUAL.json", reg["pair_roster_sha256"]))
    v4 = read(REPORTS / "V4_SLA_SKILL_AND_LONG_HORIZON_DIAGNOSTIC.json")
    repo = REPORTS.parents[1]
    data = {r["key"]: r for r in cases(root, repo)}
    require(
        primary_calls(root, data, v4) == roster["calls"], "different forward identities"
    )
    require(
        all(r["seed"] == [0, 0] for r in roster["calls"].values()), "different seed"
    )
    groups = defaultdict(list)
    groupspec = read(root / "evidence/P1_REGISTRATION.json")["target_windows"]
    for c in data.values():
        g = groupspec[c["window"]["case_index"]]
        require(g["episode"] == c["episode"], "task mapping")
        c["instruction"] = g["instruction"]
        groups[c["episode"]].append(c)
    pins = read(
        pin(
            root / "v13/run01/S0_ACTUAL.json",
            "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9",
        )
    )["inputs"]
    cap = root / "v12/A2_O2_DENSE_CAPTURE.json"
    meta = {r["id"]: r for r in read(pin(cap, pins[str(cap)]))["episodes"]}
    cache = {}
    pairs = []
    sources = []
    for ep, cases_in_episode in sorted(groups.items()):
        cs = sorted(cases_in_episode, key=lambda c: c["window"]["tick"])
        states, evidence = truth_for_episode(root, cs, meta[ep], pins)
        sources.append(evidence)
        for a, b in zip(cs, cs[1:]):
            pairs.append(evaluate_pair(a, b, states, evidence["teacher_end_tick"]))
            if measurement_file is not None:
                t = b["window"]["tick"]
                name = a["key"] + "_" + b["key"]
                clock = np.asarray(
                    [
                        t + 50 * k
                        for k in range(12)
                        if t + 50 * k <= evidence["teacher_end_tick"]
                    ]
                )
                cache[name + "_ticks"] = clock
                cache[name + "_tcp"] = np.stack([states[tick]["tcp"] for tick in clock])
    require(
        len(data) == 16
        and len(groups) == 6
        and len(pairs) == 10
        and {(r["old"], r["new"]) for r in pairs}
        == {(r["old"], r["new"]) for r in roster["pairs"]},
        "incomplete pair cohort",
    )

    if measurement_file is not None:
        with measurement_file.open("xb") as f:
            np.savez(f, **cache)

    def summary(rows):
        return {
            h: {
                m: {
                    f"{a}_minus_{b}": contrasted(rows, h, m, a, b)
                    for a, b in [
                        ("new", "old"),
                        ("new_aligned", "old_aligned"),
                        ("new", "hold"),
                        ("new_aligned", "hold"),
                    ]
                }
                for m in ["position_rmse_mm", "rotation_rmse_deg"]
            }
            for h in ["primary", "secondary"]
        }

    return dict(
        pairs=pairs,
        sources=sources,
        summary=summary(pairs),
        by_episode={
            ep: summary([r for r in pairs if r["episode"] == ep]) for ep in groups
        },
        new_forwards=0,
        new_native_starts=0,
        physics_ticks=0,
        student_owned_trajectory_pairs=0,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument(
        "--measurements",
        type=Path,
        help="Save exact native time and TCP rows for subsequent offline comparisons",
    )
    args = p.parse_args()
    root = args.root.resolve()
    configure_site(root, REPORTS.parents[1])
    result = main(root, args.measurements)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
