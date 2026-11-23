"""Locate each frozen acceleration exceedance on the actual wire and preview clock."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def classify(folder, out):
    records = []
    root = folder.resolve().parents[1]
    temporal = {
        r["id"]: r["arms"]["C1"]["timing"]
        for r in json.loads((root / "v13/run01/S3_ACTUAL.json").read_text())["windows"]
        if r["population"] == "TARGET"
    }
    panel = json.loads((folder / "V1_COMPILER_ABLATION_ACTUAL.json").read_text())
    for w in panel["windows"]:
        with np.load(
            root / "v12/A4_ABI" / (w["id"] + ".npz"), allow_pickle=False
        ) as source:
            knots = source["world_knots"]
        translations = np.diff(knots[:, :3, 3], axis=0)
        angles = (
            Rotation.from_matrix(knots[1:, :3, :3])
            * Rotation.from_matrix(knots[:-1, :3, :3]).inv()
        ).as_rotvec()
        for name, subdir in (
            ("RAW_H40_G7", "RAW"),
            ("TIME_ONLY_V13_C1", "TIME"),
            ("O1_CONTINUOUS", "O1"),
            ("C2_EXACT", "C2_EXACT"),
            ("C2_BOUNDED", "C2_BOUNDED"),
        ):
            method = w["modes"][name]
            fail = {r["sequence"]: r for r in method["rejections"]}
            if not fail:
                continue
            with (folder / w["id"] / subdir / "CALLBACKS.private.jsonl").open() as f:
                for line in f:
                    row = json.loads(line)
                    if row["kind"] != "accept":
                        continue
                    wire = json.loads(row["raw"])
                    seq = wire["_pi0294"]["sequence"]
                    if wire["_pi0294"]["operation"] != "install" or seq not in fail:
                        continue
                    entry = fail.pop(seq)
                    pts = wire["ee"]["waypoints"][:40]
                    xyz = np.array([q["position"] for q in pts])
                    r = Rotation.from_quat([q["quaternion_xyzw"] for q in pts])
                    vp = np.diff(xyz, axis=0) / 0.05
                    vr = (r[1:] * r[:-1].inv()).as_rotvec() / 0.05
                    for channel, rule in entry["failures"].items():
                        if "acceleration" not in channel:
                            continue
                        vel = vp if "translation" in channel else vr
                        a = np.linalg.norm(
                            np.diff(np.vstack([np.zeros(3), vel, np.zeros(3)]), axis=0)
                            / 0.05,
                            axis=1,
                        )
                        if not np.isclose(
                            a.max(), rule["observed"], atol=1e-6, rtol=1e-6
                        ):
                            raise ValueError(
                                (w["id"], name, seq, channel, a.max(), rule)
                            )
                        idx = np.flatnonzero(a > rule["limit"] * (1 + 1e-10))
                        for k in idx:
                            t = 0.1 * seq + 0.05 * k
                            # Source phase inside the command interval; classify a turn only if
                            # it actually straddles a raw knot, separately from preview padding.
                            if k in (0, 39):
                                kind = "synthetic_rest_at_preview_boundary"
                            elif abs(t - method["lead_s"]) <= 0.05 + 1e-9:
                                kind = "entry_time_seam"
                            elif t < method["lead_s"]:
                                kind = "alignment_interior"
                            elif abs(t - method["lifetime_s"]) <= 0.05 + 1e-9:
                                kind = "terminal_time_seam"
                            elif t > method["lifetime_s"]:
                                kind = "terminal_hold"
                            else:
                                kind = "source_interior"

                            def source_index(t):
                                end = 49 / 30
                                if name == "RAW_H40_G7":
                                    return np.clip(t * 30, 0, 49)
                                z = np.clip(
                                    (t - method["lead_s"])
                                    / (method["lifetime_s"] - method["lead_s"]),
                                    0,
                                    1,
                                )
                                if name == "O1_CONTINUOUS":
                                    return 49 * z
                                d = method["lifetime_s"]
                                slope = np.clip(
                                    temporal[w["id"]]["projected_initial_rate"],
                                    0,
                                    min(1, 3 * end / d),
                                )
                                return 30 * (
                                    end * (3 * z * z - 2 * z**3)
                                    + d * slope * z * (1 - z) ** 2
                                )

                            lo, hi = source_index(t - 0.05), source_index(t + 0.05)
                            changes = (
                                translations if "translation" in channel else angles
                            )
                            turns = []
                            for j in range(
                                max(1, int(np.ceil(lo))), min(48, int(np.floor(hi))) + 1
                            ):
                                u, v = changes[j - 1 : j + 1]
                                product = np.linalg.norm(u) * np.linalg.norm(v)
                                turns.append(
                                    [
                                        j,
                                        None
                                        if product < 1e-14
                                        else float(
                                            np.rad2deg(
                                                np.arccos(
                                                    np.clip(u @ v / product, -1, 1)
                                                )
                                            )
                                        ),
                                    ]
                                )
                            records.append(
                                [
                                    w["id"],
                                    name,
                                    seq,
                                    int(k),
                                    channel,
                                    float(a[k]),
                                    rule["limit"],
                                    kind,
                                    turns,
                                ]
                            )
            if fail:
                raise ValueError("missing published reference")
    with out.open("x") as f:
        json.dump(
            {
                "fields": [
                    "id",
                    "method",
                    "sequence",
                    "knot",
                    "channel",
                    "observed",
                    "limit",
                    "location",
                    "source_knots_and_direction_change_deg",
                ],
                "exceedances": records,
            },
            f,
            separators=(",", ":"),
            allow_nan=False,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    classify(a.folder, a.output)
