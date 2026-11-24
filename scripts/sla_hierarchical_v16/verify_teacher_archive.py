"""Independently check archived prepared arrays, sample counts and unique execution events."""

import argparse
import hashlib
import io
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
from demonstrated_grip import read, require
from target_labels import checksum
from visual_lift_panel import write_new


def verify(folder, archive, sampler):
    qualification = read(folder / "TRAIN_QUALIFICATION.json")
    ep = qualification["episodes"]
    samples = read(sampler)
    slots = {(x["episode"], x["query"]): x["probability"] for x in samples["samples"]}
    prepared = {
        k: v
        for k, v in ep.items()
        if v["status"] in ("verified_seed", "verified_expansion")
    }
    require(
        len(slots) == len(samples["samples"])
        and all(np.isfinite(w) and w > 0 for w in slots.values())
        and np.isclose(sum(slots.values()), 1, rtol=0, atol=1e-10),
        "sampler weights",
    )
    counters = Counter()
    keys = set()
    sensor_ids = set()
    with tarfile.open(archive, "r:") as tar:
        names = [f.name for f in tar if f.isfile()]
        require(len(names) == len(set(names)), "duplicate archive member")
        for name, row in prepared.items():
            v = row["prepared"]
            base = (
                "v16/target-labels04"
                if row["status"] == "verified_seed"
                else "v17/train-attempt01"
            )
            features = None
            for file, sha in v["artifacts"].items():
                data = tar.extractfile(f"{base}/{name}/{file}").read()
                require(hashlib.sha256(data).hexdigest() == sha, "artifact bytes")
                if file == "sensor_action.npz":
                    with np.load(io.BytesIO(data), allow_pickle=False) as z:
                        n = v["official_rows"]
                        require(
                            z["nominal_action"].shape == (n, 50, 16)
                            and z["state"].shape == (n, 16)
                            and z["history_indices"].shape == (n, 10)
                            and np.array_equal(
                                z["stereo_tick"][z["history_indices"][:, -1]],
                                z["query_tick"],
                            ),
                            "misjoined examples",
                        )
                        require(
                            z["stereo"].dtype == np.uint8
                            and z["stereo"].shape
                            == (len(z["stereo_tick"]), 2, 256, 256, 3),
                            "sensor tensor",
                        )
                        features = z["query_tick"]
                    sensor_ids.add(sha)
                if file == "task_audit.npz":
                    with np.load(io.BytesIO(data), allow_pickle=False) as z:
                        mask = z["confirmed_native_interval_mask"]
                        require(
                            mask.shape == (v["official_rows"], 40)
                            and mask[:, :10].all()
                            and not mask[:, 10:].any(),
                            "consumed tail confused with supervision",
                        )
                        counters["confirmed_intervals"] += int(mask.sum())
            require(
                features is not None and len(features) == v["official_rows"],
                "missing sensor rows",
            )
            keys.update((name, i) for i in range(len(features)))
            counters["official_queries"] += len(features)
            counters["source_action_points"] += len(features) * 50
            ev = samples["episodes"][name]
            for _, kind in ev["consumed_command_threshold_crossings"]:
                counters["consumed_" + kind] += 1
            counters["native_grasp"] += int(ev["target_grasp_tick"] is not None)
            counters["native_lift20mm"] += int(ev["lift20mm_tick"] is not None)
            counters["native_release"] += int(ev["released_tick"] is not None)
    require(
        keys == slots.keys() and len(sensor_ids) == len(prepared),
        "incomplete or duplicate examples",
    )
    return {
        "files": {
            p.name: checksum(p)
            for p in [
                archive,
                sampler,
                folder / "TRAIN_QUALIFICATION.json",
                folder / "RIGHT_ARM_RECIPE_CPU.json",
                folder / "HELD_OUT_ROSTER.json",
            ]
        },
        "status_counts": dict(Counter(v["status"] for v in ep.values())),
        "distinct_admitted_layouts": len({v["layout"] for v in prepared.values()}),
        "teacher_task_counts": dict(Counter(v["task"] for v in prepared.values())),
        "original_failure_counts": dict(
            Counter(
                v["original_reason"] for v in ep.values() if not v["original_admitted"]
            )
        ),
        "training_queries_by_phase": samples["phase_query_counts"],
        "event_and_action_counts": dict(counters),
        "new_network_forwards": 0,
        "new_native_starts": 0,
        "new_physics_ticks": 0,
        "optimizer_updates": 0,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("folder", "archive", "sampler", "out"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    write_new(a.out, verify(a.folder, a.archive, a.sampler))
