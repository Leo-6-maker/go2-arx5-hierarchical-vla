"""Join V1's three independently produced CPU evidence tables by registered window ID."""

import hashlib
import json
import sys
from pathlib import Path


def summarize(folder):
    def read(name):
        return json.loads((folder / name).read_bytes())

    sources = (
        "V1_COMPILER_ABLATION_ACTUAL.json",
        "V1_REACH.json",
        "V1_PREFIX.json",
        "V1_KNOT_EXCEEDANCES.json.gz",
        "V1_REGISTRATION.json",
    )
    ablation, geom, prefix = (read(k) for k in sources[:3])
    rows = ablation["windows"]
    reach = {r["id"]: r["methods"] for r in geom["records"]}
    ids = {f"T{i:02d}" for i in range(1, 17)}
    if (
        len(rows) != 16
        or len(geom["records"]) != 16
        or {w["id"] for w in rows} != ids
        or set(prefix) != ids
        or set(reach) != ids
    ):
        raise ValueError("incomplete or ambiguous V1 population")
    out = {
        "population": sorted(ids),
        "source_sha256": {
            s: hashlib.sha256((folder / s).read_bytes()).hexdigest() for s in sources
        },
        "legacy_H40_G7": sum(r["legacy_h40_g7"] for r in rows),
        "legacy_C1_H40_G7": sum(r["legacy_C1_h40_g7"] for r in rows),
        "methods": {},
    }
    for method in rows[0]["modes"]:
        modes = [w["modes"][method] for w in rows]
        result = {
            "previews": sum(m["previews"] for m in modes),
            "rejected": sum(m["rejected_previews"] for m in modes),
            "pose_pass": sum(m["pose_envelope_pass"] for m in modes),
            "boundary_pass": sum(m["boundary_pass"] for m in modes),
            "g7_pass": sum(m["g7_pass"] for m in modes),
            "dynamic_pass": [
                w["id"] for w, m in zip(rows, modes) if m["dynamics_pass"]
            ],
            "lifetime_s_min_max": [
                min(m["lifetime_s"] for m in modes),
                max(m["lifetime_s"] for m in modes),
            ],
            "lead_s_min_max": [
                min(m["lead_s"] for m in modes),
                max(m["lead_s"] for m in modes),
            ],
            "events_preserved_windows": sum(m["events_preserved"] for m in modes),
            "source_events": [e["kind"] for m in modes for e in m["events_source"]],
        }
        if method in next(iter(reach.values())):
            result["geometry_pass"] = sorted(k for k in ids if reach[k][method]["pass"])
            result["all_numeric_and_geometry"] = sorted(
                set(result["dynamic_pass"]) & set(result["geometry_pass"])
            )
            result["owner_qualified"] = [
                w["id"]
                for w in rows
                if w["old_owner"] and w["id"] in result["all_numeric_and_geometry"]
            ]
            result["prefix_installs"] = sum(
                p[method]["installs"] for p in prefix.values()
            )
            result["prefix_failures"] = sum(
                p[method]["committed_prefix_failures"] for p in prefix.values()
            )
            result["max_wire_model_distance_mm"] = 1000 * max(
                m["geometry"]["max_wire_curve_error_m"] for m in modes
            )
        out["methods"][method] = result
    out.update(
        decision="V1_FULL_INTERFACE_BLOCKED",
        new_gpu_forwards=0,
        new_physics_ticks=0,
        student_task_attempts=0,
        task_success_rate=None,
    )
    return out


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2, allow_nan=False))
