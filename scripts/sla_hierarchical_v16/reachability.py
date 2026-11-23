"""Discrete IK and FCL free-space witnesses from frozen scene bytes, without a simulator."""

import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("ablation", type=Path)
    parser.add_argument("out", type=Path)
    a = parser.parse_args()
    from sealed_panel import configure_site, cases

    root = a.root.resolve()
    repo = Path(__file__).resolve().parents[2]
    configure_site(root, repo)
    sys.path.insert(0, str(root / "v16/cpu-fcl"))
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp
    from frozen_geometry import Scene
    from ablation import output

    # Every transmitted segment is checked at both endpoints and at 25 ms midpoints.
    # Include the separately identified measured-to-reference displacement before source time.
    records = []
    cache = {}
    for c in cases(root, repo):
        p = Path(c["spec"]["model_path"])
        if p not in cache:
            cache[p] = Scene(
                p, root / "v16/scene_layout", c["spec"]["frozen_files"][str(p)]
            )
        scene = cache[p]
        row = {"id": c["key"], "methods": {}}
        for mode in ("C2_EXACT", "C2_BOUNDED"):
            with np.load(
                a.ablation / c["key"] / (mode + ".private.npz"), allow_pickle=False
            ) as z:
                t, p, g = z["t"], z["poses"], z["grip"]
            # Include intermediate states in the command-space handoff as well as wire interpolation.
            actual = c["query"]["actual_state"]
            tcp = np.asarray(actual["tcp"])
            seed = np.array([tcp, p[0]])
            d = np.linalg.norm(tcp[:3, 3] - p[0, :3, 3])
            ang = Rotation.from_matrix(tcp[:3, :3] @ p[0, :3, :3].T).magnitude()
            n = max(1, int(np.ceil(max(d / 0.0005, ang / np.deg2rad(0.1)))))
            u = np.arange(n) / n
            ix = np.arange(2 * len(t) - 1) * 0.025

            def interp(clock, poses, query):
                m = np.tile(np.eye(4), (len(query), 1, 1))
                m[:, :3, 3] = np.stack(
                    [np.interp(query, clock, col) for col in poses[:, :3, 3].T], axis=1
                )
                m[:, :3, :3] = Slerp(clock, Rotation.from_matrix(poses[:, :3, :3]))(
                    query
                ).as_matrix()
                return m

            track = np.concatenate(
                (interp([0, 1], seed, u), interp(t, p, np.clip(ix, t[0], t[-1])))
            )
            before = float(c["grip"]["previous_command_m"])
            widths = np.r_[np.full(n, before), 0.002 + 0.040 * np.interp(ix, t, g)]
            evidence = scene.witness(c, np.r_[np.full(n, -0.0), ix], track, widths)
            evidence["entry_sweep_samples"] = n
            row["methods"][mode] = evidence
        records.append(row)
    output(
        a.out,
        {
            "scope": "Discrete, robot arm vs frozen base/limbs and measured stationary objects. Sampled geometric overlap; dynamic, contact and clearance-margin qualification remain separate.",
            "records": records,
            "reader": "standalone MJB offsets, no simulator library",
            "collision_library": "python-fcl 0.7.0.11",
            "physics_ticks": 0,
        },
    )


if __name__ == "__main__":
    main()
