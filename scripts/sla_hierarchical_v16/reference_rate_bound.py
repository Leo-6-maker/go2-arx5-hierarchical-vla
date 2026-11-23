"""Find necessary A1 reference-shape error using the original cap over every update pair."""

import hashlib
import json
import sys
from pathlib import Path
import numpy as np


def bounds(t, pose, step_mm, step_deg):
    if len(t) < 2 or not np.allclose(np.diff(t), 0.1, rtol=0, atol=1e-8):
        raise ValueError("noncontiguous A1 update grid")
    answer = {}
    for i in range(len(t) - 1):
        dest = pose[i + 1 :]
        n = np.arange(1, len(dest) + 1)
        linear = 1000 * np.linalg.norm(dest[:, :3, 3] - pose[i, :3, 3], axis=1)
        rel = dest[:, :3, :3] @ pose[i, :3, :3].T
        angular = np.rad2deg(
            np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1))
        )
        for k, d, cap in (
            ("position_mm", linear, step_mm),
            ("orientation_deg", angular, step_deg),
        ):
            error = np.maximum(0, (d - cap * n) / 2)
            j = int(error.argmax())
            if k not in answer or error[j] > answer[k]["minimum_uniform_error"]:
                answer[k] = dict(
                    minimum_uniform_error=float(error[j]),
                    start_s=float(t[i]),
                    end_s=float(t[i + 1 + j]),
                    target_separation=float(d[j]),
                    maximum_reference_travel=float(cap * n[j]),
                )
    return answer


def run(root, repo):
    def read(p):
        return json.loads(p.read_bytes())

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    cfg = read(repo / "reports/SLA_FROZEN_FEEDBACK/F0_REGISTRATION.json")["controller"]
    f = root / "v16/v1-run01/V1_COMPILER_ABLATION_ACTUAL.json"
    if (
        sha(f)
        != sha(repo / "reports/SLA_HIERARCHICAL_V16/V1_COMPILER_ABLATION_ACTUAL.json")
        or cfg["cadence_ms"] != 100
    ):
        raise ValueError("frozen source mismatch")
    records = []
    for w in read(f)["windows"]:
        for method in ("C2_EXACT", "C2_BOUNDED"):
            m = w["modes"][method]
            p = f.parent / w["id"] / (method + ".private.npz")
            with np.load(p, allow_pickle=False) as z:
                t = z["t"]
                pose = z["poses"]
                if (
                    not np.allclose(t, np.arange(len(t)) * 0.05, rtol=0, atol=1e-8)
                    or abs(t[-1] - m["lifetime_s"]) > 0.051
                ):
                    raise ValueError("malformed wire samples")
                keep = (
                    (np.arange(len(t)) % 2 == 0)
                    & (t >= m["lead_s"] - 1e-8)
                    & (t <= m["lifetime_s"] + 1e-8)
                )
                x = bounds(t[keep], pose[keep], cfg["max_step_mm"], cfg["max_step_deg"])
            records.append(
                dict(id=w["id"], method=method, compiled_track_sha256=sha(p), **x)
            )
    if len(records) != 32:
        raise ValueError("incomplete population")
    return dict(
        records=records,
        cap_mm_per_100ms=cfg["max_step_mm"],
        cap_deg_per_100ms=cfg["max_step_deg"],
        physics_ticks=0,
        gpu_forwards=0,
    )


if __name__ == "__main__":
    print(
        json.dumps(
            run(Path(sys.argv[1]), Path(__file__).resolve().parents[2]),
            indent=2,
            allow_nan=False,
        )
    )
