"""Post-hoc CPU readback of the archived RL preview at declared progress zero."""

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import torch

from audit import read, record, sha
from canonical import PIN

CONFIG_SHA = "4d056215bec546f8063e9b83903352cdaf99360e32e31ef6cb4b81eaed4803e6"


def main(run, trajectory_file, config_file, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or sha(trajectory_file) != PIN:
        raise ValueError("Requires CPU execution and the pinned trajectory implementation")
    if sha(config_file) != CONFIG_SHA:
        raise ValueError("Archived preview configuration differs")
    cfg = read(config_file)["resolved"]["wbc"]["goal_reaching"]["trajectory"]
    spec = importlib.util.spec_from_file_location("archived_rl_trajectory", trajectory_file)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    torch.set_num_threads(1)
    rows = []
    canonical = {r["id"]: r for r in read(run / "CANONICAL_ACTUAL.json")["rows"] if r["arm"] == "C1"}
    for window in read(run / "S3_ACTUAL.json")["windows"]:
        if not window["id"].startswith("T"):
            continue
        arm = window["arms"]["C1"]
        candidate = run / "arrays" / arm["array_file"]
        prior = canonical[window["id"]]
        if sha(candidate) != arm["array_sha256"] or prior["candidate_sha256"] != arm["array_sha256"]:
            raise ValueError("C1 input differs from original offline readback")
        with np.load(candidate, allow_pickle=False) as archive:
            poses = archive["world"]
        path, law = native.to_canonical(poses[:, :3, 3], poses[:, :3, :3], 0.05)
        batch = native.TrajectoryBatch(1, cfg["max_gamma_points"], cfg["max_tl_points"], device="cpu")
        if len(path.s_grid) > batch.max_gamma_points or len(law.t_grid) > batch.max_tl_points:
            raise ValueError("Canonical path exceeds archive capacity")
        batch.load(torch.tensor([0]), [path], [law])
        s, _, _, _ = batch.sample_preview(torch.zeros(1), cfg["preview_horizon"], cfg["preview_points"])
        length = float(batch.L[0])
        if length != prior["s_ref"][-1]:
            raise ValueError("Path length disagrees with sealed canonical result")
        rows.append(dict(window=window["id"], length_m_equivalent=length,
                         endpoint_samples=int((s[0] == length).sum()), samples=s.numel()))
    if [row["window"] for row in rows] != [f"T{i:02}" for i in range(1, 17)]:
        raise ValueError("Wrong target TRAIN roster")
    record(output, dict(scope="Post-hoc C1 at s=0, isolated CPU trajectory batch",
                        config_sha256=CONFIG_SHA, trajectory_sha256=PIN,
                        input_receipt_sha256=sha(run / "CANONICAL_ACTUAL.json"),
                        horizon_m_equivalent=cfg["preview_horizon"], rows=rows,
                        all_at_endpoint=sum(x["endpoint_samples"] == x["samples"] for x in rows),
                        live_installs=0, new_native_ticks=0, model_forwards=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "trajectory_file", "config_file", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    main(args.run, args.trajectory_file, args.config_file, args.output)
