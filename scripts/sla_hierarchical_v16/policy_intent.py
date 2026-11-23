"""Score frozen official right-arm proposals against scene facts held out from inference."""

import hashlib
import json
from collections import Counter, defaultdict
import os
import sys
from pathlib import Path
import numpy as np
from sealed_panel import configure_site


def features(poses, events):
    xyz = poses[:, :3, 3] * 1000
    disp = xyz - xyz[0]
    moved = float(np.linalg.norm(disp, axis=1).max())
    axis = int(abs(disp[-1]).argmax())
    signed = ("+" if disp[-1, axis] >= 0 else "-") + "XYZ"[axis]
    signature = ",".join(events) or ("STILL" if moved < 1 else signed)
    rotations = poses[:, :3, :3] @ poses[0, :3, :3].T
    angle = np.rad2deg(
        np.arccos(np.clip((np.trace(rotations, axis1=1, axis2=2) - 1) / 2, -1, 1))
    )
    return dict(
        signature=signature,
        events=events,
        delta_mm=disp[-1].tolist(),
        peak_excursion_mm=moved,
        peak_rotation_deg=float(angle.max()),
        source_arc_mm=float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
    )


def evaluate(feat, poses, objects, instruction, phase, zones):
    action, color, *side = instruction.split("-")
    if (
        action not in ("HOLD", "PLACE")
        or set(objects) != {"red", "blue"}
        or color not in objects
    ):
        raise ValueError("unmapped instruction")
    xyz = poses[:, :3, 3] * 1000
    d = {}
    for name, obj in objects.items():
        distance = np.linalg.norm(xyz - 1000 * np.asarray(obj["pose"])[:3, 3], axis=1)
        d[name] = dict(
            start_mm=float(distance[0]),
            end_mm=float(distance[-1]),
            min_mm=float(distance.min()),
            gain_mm=float(distance[0] - distance[-1]),
        )
    delta = np.array(feat["delta_mm"])
    events = feat["events"]
    destination = None
    if action == "PLACE":
        obj = 1000 * np.asarray(objects[color]["pose"])[:2, 3]
        distance = np.linalg.norm(
            1000 * np.asarray(zones[side[0]]["center_world_XY"]) - obj
        )
        if distance <= 0:
            raise ValueError("undefined zone direction")
        destination = float(
            delta[:2]
            @ (1000 * np.asarray(zones[side[0]]["center_world_XY"]) - obj)
            / distance
        )
    if phase == "DESCEND":
        direction = bool(delta[2] < 0 and d[color]["gain_mm"] >= 1)
    elif phase == "CLOSE":
        direction = "close" in events
    elif phase == "LIFT":
        direction = bool(delta[2] >= 1)
    elif phase == "HOLD":
        direction = feat["peak_excursion_mm"] <= 4 and feat["peak_rotation_deg"] <= 2
    elif phase == "CARRY" and destination is not None:
        direction = destination >= 1
    else:
        raise ValueError("unregistered scoring phase")
    return dict(
        phase=phase,
        instruction=instruction,
        object_distances=d,
        zone_projection_mm=destination,
        direction_or_event_proxy=direction,
        phase_proxy_pass=direction and (phase == "DESCEND" or "open" not in events),
    )


def timing(root, windows):
    groups = {}
    all_times = []
    warm = []
    primary = set()
    repeats = []

    def stats(v):
        return dict(
            n=len(v),
            median_s=float(np.median(v)),
            p95_s=float(np.percentile(v, 95)),
            min_s=min(v),
            max_s=max(v),
        )

    for campaign in ("O2_GPU_01", "P2_EXTRA_GPU_01"):
        folder = root / "v12" / campaign
        log = folder / "calls.jsonl"
        lines = [json.loads(s) for s in log.read_text().splitlines()]
        enters = {r["call"]: r for r in lines if r["event"] == "FORWARD_ENTERED"}
        times = []
        for r in lines:
            if r["event"] != "FORWARD_RETURNED":
                continue
            a = enters.pop(r["call"])
            w = windows[r["id"]]
            if (
                not all(
                    r[k] == a[k]
                    for k in ("id", "tick", "input_sha256", "seed", "repeat")
                )
                or r["seed"] != w["seed"]
                or r["tick"] != w["tick"]
                or r["file_sha256"] != w["raw_sha256"]
                or hashlib.sha256((folder / r["output_file"]).read_bytes()).hexdigest()
                != r["file_sha256"]
            ):
                raise ValueError("forward identity")
            if r["repeat"]:
                if r["id"] not in primary or r["same_as_primary"] is not True:
                    raise ValueError("unverified repeated proposal")
                repeats.append(r["id"])
            elif r["id"] in primary:
                raise ValueError("duplicate primary")
            else:
                primary.add(r["id"])
            times.append(float(r["seconds"]))
        if enters or not times or not np.all(np.isfinite(times)) or min(times) <= 0:
            raise ValueError("incomplete latency")
        groups[campaign] = dict(
            journal_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
            calls_s=times,
            first_forward_s=times[0],
            warm=stats(times[1:]),
        )
        all_times += times
        warm += times[1:]
    if primary != set(windows) or len(repeats) != 2:
        raise ValueError("unexpected target population")
    return dict(
        groups=groups, all=stats(all_times), warm=stats(warm), repeat_ids=repeats
    )


def main(root, repo):
    from sealed_panel import cases
    from audit_sealed import digest, pin, read

    p = root / "v13/run01/S0_ACTUAL.json"
    pins = read(p)["inputs"]
    factor = root / "v12/A1_SOURCE_FACTORIAL.json"
    pin(factor, pins[str(factor)])
    factorial = read(factor)
    tasks = read(root / "evidence/P1_REGISTRATION.json")["target_windows"]
    rows = []
    counts = defaultdict(Counter)
    windows = {}
    for c in cases(root, repo):
        q = c["query"]["actual_state"]
        poses = c["curve"].knots
        feat = features(poses, [e[0] for e in c["curve"].events])
        task = tasks[c["window"]["case_index"]]
        if task["episode"] != c["episode"]:
            raise ValueError("query instruction mismatch")
        r = evaluate(
            feat,
            poses,
            q["objects"],
            task["instruction"],
            c["archive"]["phase"],
            c["spec"]["multitask_geometry"]["zones"],
        )
        counts[r["phase"]][feat["signature"]] += 1
        rows.append(
            dict(
                id=c["key"],
                episode=c["episode"],
                raw_sha256=c["raw_file_sha256"],
                **feat,
                **r,
            )
        )
        windows[c["key"]] = dict(
            seed=c["window"]["seed"],
            tick=c["window"]["tick"],
            raw_sha256=c["raw_file_sha256"],
        )
    if len(rows) != 16:
        raise ValueError("incomplete registered panel")
    return dict(
        windows=rows,
        phase_by_action_signature=counts,
        latency=timing(root, windows),
        source_factorial=dict(
            source_sha256=digest(factor),
            windows=factorial["windows"]
            if isinstance(factorial["windows"], int)
            else len(factorial["windows"]),
            aggregation=factorial["aggregation"],
            summary=factorial["summary"],
        ),
        explicit_model_phase_predictions=0,
        own_feedback_replans=0,
        new_gpu_forwards=0,
        student_native_attempts=0,
        heldout_attempts=0,
        decision="V4_LOCAL_PROPOSALS_ONLY",
    )


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parents[2]
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    configure_site(root, repo)
    # configure_site may re-exec the program in the legacy native Python environment.
    report = main(root, repo)
    with Path(sys.argv[2]).open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
