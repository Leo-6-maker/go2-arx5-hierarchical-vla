"""Score the age of a scheduled source motion against the independent teacher recording."""

from bisect import bisect_left
import json
from pathlib import Path
import sys
import numpy as np

from demonstrated_grip import pin, read, require, rows


def facts(query, color):
    obj = query["actual_state"]["objects"][color]
    return dict(
        tick=query["native_step"],
        phase=query["phase"],
        grasp=obj["native_grasp"],
        hand_contact=obj["hand_contact"],
    )


def audit(root, result, reports):
    pins = read(
        pin(
            root / "v13/run01/S0_ACTUAL.json",
            "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9",
        )
    )["inputs"]
    calls = read(
        pin(
            reports / "H1_REVISION_ACTUAL.json",
            "855860708527c0a2b0d86b5ad710ea94da7201ba89e02bca5c41e08518ca206c",
        )
    )["calls"]
    target = read(
        pin(
            root / "evidence/P1_REGISTRATION.json",
            "1d496553b347a43c6f5c45c99d736896060ca9d71bf6cd5a72fef3e767b1b900",
        )
    )["target_windows"]
    capture = root / "v12/A2_O2_DENSE_CAPTURE.json"
    source_meta = {
        r["id"]: r for r in read(pin(capture, pins[str(capture)]))["episodes"]
    }
    out = {}
    for g in target:
        ep = g["episode"]
        group = result["episodes"][ep]
        epfile = pin(Path(g["episode_pin"]["path"]), g["episode_pin"]["sha256"])
        path = epfile.parents[3] / "cases" / ep / "native-01"
        color = g["instruction"].split("-")[1]
        out[ep] = {}
        for mode, data in group.items():
            out[ep][mode] = {
                k: dict(status=r["status"], comparison="NO_SCHEDULED_SOURCE_MOTION")
                for k, r in data["rows"].items()
            }
        if not any(
            r["first_nonzero_source_motion_s"] is not None
            for d in group.values()
            for r in d["rows"].values()
        ):
            continue

        def checked(name):
            f = path / name
            return pin(f, pins[str(f)])

        queries = read(checked("LIVE_DATA/QUERIES.private.json"))
        ticks = [q["native_step"] for q in queries]
        require(
            all(b - a == 250 for a, b in zip(ticks, ticks[1:])),
            "missing original query",
        )
        atoms = rows(checked("MOBILE_SAMPLES.private.jsonl"))
        boundary = {a["native_end_step"]: a for a in atoms}
        require(len(boundary) == len(atoms), "duplicate boundary")
        meta = source_meta[ep]
        receipt = read(pin(Path(meta["receipt_path"]), meta["receipt_sha256"]))
        require(receipt["id"] == ep, "wrong episode receipt")
        with pin(
            path / "physics.jsonl", receipt["source_files"]["physics.jsonl"]
        ).open() as f:
            joints = {j["name"]: j for j in json.loads(f.readline())["joints"]}
        joint = joints[{"red": "task_cube_free", "blue": "task_blue_cube_free"}[color]]
        require(joint["type"] == 0, "expected recorded object free joint")
        obj = joint["qposadr"]
        fingers = [
            joints[n]["qposadr"] for n in ("task_left_slide", "task_right_slide")
        ]

        def measured(tick):
            row = boundary[tick]
            require(
                row["native_start_step"] + 25 == tick and row["ticks"] == 25,
                "incomplete native atom",
            )
            p = np.array(row["qpos"])
            return np.array(row["actual_TCP"])[:3, 3], p[obj : obj + 3], p[fingers]

        for mode, data in group.items():
            for k, r in data["rows"].items():
                age = r["first_nonzero_source_motion_s"]
                if age is None:
                    continue
                t0 = calls[k]["tick"]
                q = queries[ticks.index(t0)]
                t1 = t0 + round(age / 0.002)
                require(
                    abs((t1 - t0) * 0.002 - age) < 1e-8, "ambiguous first source clock"
                )
                tcp, pos, slides = measured(t0)
                state = q["actual_state"]
                for first, second in (
                    (tcp, np.array(state["tcp"])[:3, 3]),
                    (pos, np.array(state["objects"][color]["pose"])[:3, 3]),
                    (slides, state["finger_q"]),
                ):
                    require(
                        np.allclose(first, second, rtol=0, atol=2e-7),
                        "query state and native atom disagree",
                    )
                if t1 not in boundary:
                    out[ep][mode][k].update(
                        comparison="NO_EXACT_NATIVE_BOUNDARY",
                        query_tick=t0,
                        scheduled_tick=t1,
                    )
                    continue
                b = bisect_left(ticks, t1)
                bracket = (
                    ticks[b : b + 1]
                    if b < len(ticks) and t1 == ticks[b]
                    else ticks[max(0, b - 1) : b + 1]
                )
                tcp1, pos1, slides1 = measured(t1)
                out[ep][mode][k].update(
                    comparison="MEASURED_TEACHER_AT_EXACT_TICK",
                    query_tick=t0,
                    scheduled_tick=t1,
                    age_s=age,
                    tcp_change_mm=float(np.linalg.norm(tcp1 - tcp) * 1000),
                    target_change_mm=float(np.linalg.norm(pos1 - pos) * 1000),
                    target_z_change_mm=float(1000 * (pos1[2] - pos[2])),
                    mean_finger_change_mm=float(1000 * (slides1 - slides).mean()),
                    query=facts(q, color),
                    task_queries_bracketing_later_tick=[
                        facts(queries[ticks.index(t)], color) for t in bracket
                    ],
                )
    require(
        all(sum(len(g[m]) for g in out.values()) == 16 for m in ("fifo", "overlap")),
        "full population",
    )
    return out


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    source = Path(sys.argv[2])
    source_sha = sys.argv[3]
    result = audit(
        root,
        read(pin(source, source_sha)),
        Path(__file__).resolve().parents[2] / "reports/SLA_HIERARCHICAL_V16",
    )
    with Path(sys.argv[4]).open("x") as f:
        json.dump(
            dict(scheduler_sha256=source_sha, teacher_observations=result),
            f,
            indent=2,
            allow_nan=False,
        )
        f.write("\n")
