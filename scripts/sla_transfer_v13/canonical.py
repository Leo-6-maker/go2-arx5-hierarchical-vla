"""Read the pinned handoff's native Gamma/TimeLaw on CPU without constructing WBC state."""

import argparse, importlib.util, os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from audit import read, record, sha

PIN = "8386841b29b58f324cd2e9afa086439add4183992be2466f688382be8ed9d76a"


def main(out, module):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or sha(module) != PIN:
        raise ValueError("CPU and pinned original trajectory handoff required")
    spec = importlib.util.spec_from_file_location("pinned_native_trajectory", module)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    torch.set_num_threads(1)
    rows = []
    dest = out / "canonical"
    dest.mkdir(exist_ok=False)
    for w in read(out / "S3_ACTUAL.json")["windows"]:
        for arm, log in w["arms"].items():
            if "array_file" not in log:
                continue
            src = out / "arrays" / log["array_file"]
            assert sha(src) == log["array_sha256"]
            with np.load(src, allow_pickle=False) as z:
                raw = z["world"]
                g = z["H40"][:, 9]
            row = dict(id=w["id"], arm=arm, candidate_sha256=sha(src))
            try:
                gamma, clock = original.to_canonical(raw[:, :3, 3], raw[:, :3, :3], 0.05)
                t = torch.arange(40, dtype=clock.t_grid.dtype) * 0.05
                s = clock.s_ref(t)
                sd = clock.sdot_ref(t)
                pos = gamma.p_at(s).numpy()
                rot = gamma.R_at(s).numpy()
                assert (
                    torch.isfinite(s).all()
                    and torch.isfinite(sd).all()
                    and (s.diff() >= 0).all()
                    and (sd >= 0).all()
                )
                h40 = np.column_stack([pos, rot[:, :2, :].reshape(40, 6), g])
                file = dest / (w["id"] + "_" + arm + ".npz")
                with file.open("xb") as f:
                    np.savez_compressed(
                        f,
                        H40=h40,
                        s_ref=s.numpy(),
                        sdot_ref=sd.numpy(),
                        gamma_s=gamma.s_grid.numpy(),
                    )
                row.update(
                    status="OFFLINE_NATIVE_CANONICAL_READBACK",
                    file=file.name,
                    sha256=sha(file),
                    s_ref=s.tolist(),
                    sdot_ref=sd.tolist(),
                    gamma_points=len(gamma.s_grid),
                    t_grid=len(clock.t_grid),
                    pos_deviation_max_m=float(np.linalg.norm(pos - raw[:, :3, 3], axis=1).max()),
                    rotation_deviation_max_rad=float(
                        (Rotation.from_matrix(rot) * Rotation.from_matrix(raw[:, :3, :3]).inv())
                        .magnitude()
                        .max()
                    ),
                )
            except (ValueError, RuntimeError, AssertionError) as e:
                row.update(status="CANONICAL_REJECT", exception=repr(e))
            rows.append(row)
    record(
        out / "CANONICAL_ACTUAL.json",
        dict(
            trajectory_sha256=PIN,
            rows=rows,
            device="cpu",
            dt=0.05,
            native_receiver_installs=0,
            scope="Exact pinned external bridge canonicalizer in isolation. Archived Core executor used waypoint wire with 25-tick receipts. Canonical output has no live receiver ACK or measured WBC progress.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("module", type=Path)
    a = p.parse_args()
    main(a.out, a.module)
