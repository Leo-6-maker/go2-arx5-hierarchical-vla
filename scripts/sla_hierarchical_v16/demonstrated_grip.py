"""Compare archived raw predictions with the original expert commands at the same clock."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE_T = np.arange(50) / 30
PROFILES = ("O0_NATIVE", "O1_VISION_ONLY", "O1_STATE_ONLY", "O1A_MISSING_LEFT")


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def pin(path, digest):
    with path.open("rb") as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    require(h.hexdigest() == digest, f"changed file {path.name}")
    return path


def events(t, widths, *, step):
    t, widths = np.asarray(t), np.asarray(widths)
    require(
        t.ndim == 1
        and len(t)
        and t.shape == widths.shape
        and np.isfinite([t, widths]).all()
        and (np.diff(t) > 0).all(),
        "invalid clock or width",
    )
    found = []
    for i, (a, b) in enumerate(zip(widths, widths[1:])):
        for k, level in (("close", 0.55), ("open", 0.65)):
            if (k == "close" and a > level >= b) or (k == "open" and a < level <= b):
                at = (
                    t[i + 1]
                    if step
                    else t[i] + (level - a) / (b - a) * (t[i + 1] - t[i])
                )
                found.append(dict(kind=k, s=float(at)))
    return found


def compare(pred, expert, expert_events, hold):
    require(
        pred.shape == expert.shape == (50,)
        and np.isfinite([pred, expert]).all()
        and np.isfinite(hold),
        "missing reference",
    )
    proposed = events(SOURCE_T, pred, step=False)
    match = [r["kind"] for r in proposed] == [r["kind"] for r in expert_events]
    errors = (
        [a["s"] - b["s"] for a, b in zip(proposed, expert_events)] if match else None
    )
    return dict(
        proposed=proposed,
        expert=expert_events,
        sequence_equal=match,
        matched_time_errors_s=errors,
        previous_or_measured_fraction=float(hold),
        first_model=float(pred[0]),
        first_expert=float(expert[0]),
        last_model=float(pred[-1]),
        last_expert=float(expert[-1]),
        width_rmse=float(np.sqrt(np.mean((pred - expert) ** 2))),
        hold_rmse=float(np.sqrt(np.mean((hold - expert) ** 2))),
    )


def source(root, repo, pins):
    import h5py

    folder = root / "evidence/O0_PANEL_01"
    o0 = read(repo / "reports/SLA_OFFICIAL_GO2_TRANSFER/O0_ACTUAL.json")["source"]
    calls = {
        r["call"]: r
        for r in rows(folder / "calls.jsonl")
        if r["event"] == "FORWARD_RETURNED"
    }
    require(len(calls) == len(o0) == 8, "source population")
    factorial_path = root / "v12/A1_SOURCE_FACTORIAL.json"
    fac = read(pin(factorial_path, pins[str(factorial_path)]))
    split = root / "v12/A1_SOURCE_01"
    pin(split / "calls.jsonl", fac["journal_sha256"])
    combined = root / "evidence/P1_SOURCE_01"
    combined_pins = read(root / "P1_OBSERVATION_ABLATIONS.json")["source_evidence"]
    ablated = {}
    for directory in (split, combined):
        for r in rows(directory / "calls.jsonl"):
            if r["event"] != "FORWARD_RETURNED" or r["profile"] not in PROFILES:
                continue
            f = directory / r["output_file"]
            pin(f, r["file_sha256"])
            if directory == combined:
                pin(f, combined_pins[str(f.relative_to(root))])
            key = r["o0_call"], r["profile"]
            require(key not in ablated, "duplicate arm")
            ablated[key] = (r, f)
    manifest = read(root / "evidence/P0_SOURCE_FILES.actual.json")["files"]
    files = {
        Path(r["path"]).name: pin(
            root / "data" / Path(r["path"]).name, r["published_lfs_sha256"]
        )
        for r in manifest
        if r["path"].endswith(".hdf5")
    }
    lookup = {(r["episode"], r["frame"]): r for r in fac["windows"]}
    out = []
    for metadata in o0:
        r = calls[metadata["call"]]
        require(
            all(r[k] == metadata[k] for k in ("episode", "frame", "mode")),
            "source join",
        )
        if r["mode"] != "primary":
            continue
        f = pin(folder / r["output_file"], metadata["output_sha256"])
        with np.load(f, allow_pickle=False) as z:
            raw = z["native_absolute_arms"]
            expert = z["expert_absolute_arms"]
            state = z["state"]
        fr = r["frame"]
        with h5py.File(files[r["episode"]], "r") as demo:
            ref = np.concatenate(
                [
                    demo["action/" + s][fr : fr + 50]
                    for s in (
                        "left_ee_poses",
                        "left_ee_joint_states",
                        "right_ee_poses",
                        "right_ee_joint_states",
                    )
                ],
                axis=1,
            )
            require(np.array_equal(ref, expert), "source HDF5 expert differs")
            measured = float(demo["state/right_ee_joint_states"][fr].item())
            require(abs(measured - state[15]) < 1e-6, "source measured width differs")
        checkrow = lookup[r["episode"], fr]
        for profile in PROFILES:
            sample = raw
            digest = metadata["output_sha256"]
            if profile != "O0_NATIVE":
                a, af = ablated[r["call"], profile]
                require(
                    all(
                        a[k] == r[k]
                        for k in ("episode", "frame", "prng_key_before_split")
                    )
                    and checkrow["prng_key"] == r["prng_key_before_split"],
                    "factorial window/seed",
                )
                sample = np.load(af, allow_pickle=False)
                digest = a["file_sha256"]
            require(sample.shape == expert.shape == (50, 16), "shape")
            q = compare(
                sample[:, 15].astype(float),
                expert[:, 15].astype(float),
                events(SOURCE_T, expert[:, 15].astype(float), step=False),
                measured,
            )
            require(
                abs(q["width_rmse"] - checkrow["metrics"][profile]["expert_grip_rmse"])
                < 1e-6,
                "archived grip error",
            )
            out.append(
                dict(
                    episode=r["episode"],
                    frame=fr,
                    phase=r["phase"],
                    profile=profile,
                    file_sha256=digest,
                    **q,
                )
            )
    require(len(out) == 24 and len(ablated) == 18, "incomplete source panel")
    return out


def target(root, repo, pins):
    legacy = read(repo / "reports/SLA_OFFICIAL_GO2_TRANSFER/O1_ACTUAL.json")["windows"]
    motor = read(repo / "reports/SLA_MEMORY_V15/B3_GRIPPER_REPLAY.json")["episodes"]
    episode_records = {r["episode"]: r for r in motor}
    specs = {
        Path(p).parent.name: Path(p) for p in pins if p.endswith("/EXECUTION_SPEC.json")
    }
    rawpaths = {h: Path(p) for p, h in pins.items() if p.endswith(".npy")}
    cache = {}
    out = []
    for w in legacy:
        ep = w["episode"]
        if ep not in cache:
            f = specs[ep].parent / "native-01/MOBILE_SAMPLES.private.jsonl"
            pin(f, episode_records[ep]["commands_sha256"])
            data = rows(pin(f, pins[str(f)]))
            ticks = np.array([c["native_start_step"] for c in data], int)
            width = np.array([0.044 * c["installed_EEF10"][9] for c in data])
            require(
                len(data) == episode_records[ep]["commands"]
                and (np.diff(ticks) == 25).all()
                and all(
                    c["gripper_command"]["sequence"] == i
                    and c["ticks"] == 25
                    and c["native_end_step"] == ticks[i] + 25
                    for i, c in enumerate(data)
                ),
                "teacher clock",
            )
            full_events = events(ticks, (width - 0.002) / 0.040, step=True)
            cache[ep] = (ticks, width, full_events)
        pred = np.load(
            pin(rawpaths[w["raw_sha256"]], w["raw_sha256"]), allow_pickle=False
        )[:, 15].astype(float)
        ticks, width, full_events = cache[ep]
        origin = w["tick"]
        i = ticks.searchsorted(origin)
        require(
            0 < i < len(ticks)
            and ticks[i] == origin
            and abs(width[i - 1] - w["previous_consumed_grip_m"]) < 1e-10,
            "initial previous command",
        )
        previous = (width[i - 1] - 0.002) / 0.040
        info = dict(
            id=w["id"],
            episode=ep,
            phase=w["phase"],
            tick=origin,
            file_sha256=w["raw_sha256"],
            held_at_query=w["held"],
        )
        info["last_teacher_close_relative_s"] = max(
            (
                (e["s"] - origin) * 0.002
                for e in full_events
                if e["kind"] == "close" and e["s"] <= origin
            ),
            default=None,
        )
        info["available_teacher_s"] = float((ticks[-1] + 25 - origin) * 0.002)
        if info["available_teacher_s"] <= SOURCE_T[-1]:
            partial_events = events(
                (ticks[i:] - origin) * 0.002, (width[i:] - 0.002) / 0.040, step=True
            )
            out.append(
                dict(
                    **info,
                    status="CENSORED_TEACHER_END",
                    proposed=events(SOURCE_T, pred, step=False),
                    partial_teacher_events=partial_events,
                    full_sequence_equal=None,
                    query_boundary_events=events(
                        [-0.05, 0.0], [previous, (width[i] - 0.002) / 0.040], step=True
                    ),
                )
            )
            continue
        # Teacher commands are held from their native tick. Rounded integer numerators
        # place coincident 30 Hz and 20 Hz samples on the new command exactly.
        sample_idx = i + (np.arange(50) * 2) // 3
        expert = (width[sample_idx] - 0.002) / 0.040
        j = sample_idx[-1]
        fractions = (width[i : j + 1] - 0.002) / 0.040
        t = (ticks[i : j + 1] - origin) * 0.002
        crossed = events(t, fractions, step=True)
        boundary = events([-0.05, 0.0], [previous, fractions[0]], step=True)
        out.append(
            dict(
                **info,
                status="COMPLETE",
                query_boundary_events=boundary,
                expert_event_ticks=[origin + round(e["s"] / 0.002) for e in crossed],
                **compare(pred, expert, crossed, previous),
            )
        )
    require(len(out) == 16 and len(cache) == 6, "target population")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[2]
    pins = read(
        pin(
            root / "v13/run01/S0_ACTUAL.json",
            "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9",
        )
    )["inputs"]
    result = dict(
        source=source(root, repo, pins),
        target=target(root, repo, pins),
        source_horizon_s=49 / 30,
        new_native_starts=0,
        new_forwards=0,
        training_updates=0,
    )
    with args.out.open("x") as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
