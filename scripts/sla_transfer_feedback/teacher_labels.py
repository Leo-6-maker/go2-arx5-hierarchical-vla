"""Audit aligned endpoint target vs actual TCP in six pinned target TRAIN demonstrations."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from feedback import require
from scipy.spatial.transform import Rotation

from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def stats(values):
    a = np.array(values)
    return {
        "atoms": len(a),
        "rms_mm_deg": np.sqrt(np.mean(a * a, axis=0)).tolist(),
        "median_mm_deg": np.median(a, axis=0).tolist(),
        "p95_mm_deg": np.percentile(a, 95, axis=0).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    manifest = root / "v13/run01/S0_ACTUAL.json"
    require(
        digest(manifest)
        == "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9",
        "S0",
    )
    pins = read(manifest)["inputs"]
    roster = root / "v12/A2_O2_DENSE_CAPTURE.json"
    require(digest(roster) == pins[str(roster)], "roster")
    bysegment = defaultdict(list)
    bygrasp = defaultdict(list)
    entries = []
    for e in read(roster)["episodes"]:
        receipt = Path(e["receipt_path"])
        require(
            digest(receipt) == e["receipt_sha256"]
            and read(receipt)["split"] == "TRAIN",
            "not pinned TRAIN",
        )
        spec = next(
            Path(k)
            for k in pins
            if k.endswith("/EXECUTION_SPEC.json") and Path(k).parent.name == e["id"]
        )
        require(digest(spec) == pins[str(spec)], "spec")
        native = spec.parent / "native-01"
        samples = native / "MOBILE_SAMPLES.private.jsonl"
        require(digest(samples) == pins[str(samples)], "atom pins")
        accepted = {}
        for ev in lines(native / "planner.jsonl"):
            if ev["kind"] == "accept":
                raw = ev["raw"]
                h = hashlib.sha256(raw.encode()).hexdigest()
                require(h == ev["ref_id"] and h not in accepted, "wire")
                doc = json.loads(raw)
                require(
                    doc["_pi0294"]["episode"] == native.name
                    and doc["ee"]["frame"] == "world",
                    "episode/frame",
                )
                accepted[h] = doc["ee"]["waypoints"]
        vals = []
        prior = None
        i = -1
        right = None
        for row in lines(samples):
            h = row["installed_reference_sha256"]
            i = 0 if h != prior else i + 1
            require(
                row["ticks"] == 25
                and row["native_end_step"] - row["native_start_step"] == 25
                and (right is None or right == row["native_start_step"])
                and row["solver_exact_installed_targets"]
                and row["safety_valid"],
                "archive_join",
            )
            prior = h
            right = row["native_end_step"]
            action = np.asarray(row["target_endpoint_EEF10"])
            actual = np.asarray(row["actual_TCP"])
            R = rot6d_rows_to_matrix(action[3:9])
            seal = accepted[h][i + 1]
            require(
                np.allclose(seal["position"], action[:3], atol=1e-9, rtol=0)
                and np.allclose(
                    Rotation.from_quat(seal["quaternion_xyzw"]).as_matrix(),
                    R,
                    atol=1e-8,
                    rtol=0,
                ),
                "endpoint_wire",
            )
            error = [
                float(1000 * np.linalg.norm(actual[:3, 3] - action[:3])),
                float(
                    np.degrees(Rotation.from_matrix(actual[:3, :3] @ R.T).magnitude())
                ),
            ]
            vals.append(error)
            bysegment[row["segment"]].append(error)
            bygrasp[str(bool(row["native_grasp"]))].append(error)
        entries.append(
            dict(episode=e["id"], samples_sha256=pins[str(samples)], **stats(vals))
        )
    result = {
        "scope": "Archived TRAIN teacher, endpoint command versus actual world TCP at end of consumed 50 ms interval",
        "audit_sha256": digest(Path(__file__)),
        "episodes": entries,
        "by_segment": {k: stats(v) for k, v in bysegment.items()},
        "by_native_grasp": {k: stats(v) for k, v in bygrasp.items()},
        "new_physics_ticks": 0,
        "new_model_forwards": 0,
        "training_updates": 0,
    }
    with args.output.open("x") as f:
        json.dump(result, f, allow_nan=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
