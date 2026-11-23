"""Audit the persistent clock of each C2 intention against actual offline callback receipts."""

import argparse
import json
from pathlib import Path

import numpy as np


class WireProgram:
    def __init__(self, identity, poses, g, phase):
        self.identity = identity
        self.arrays = []
        for a in (poses, g, phase):
            a = np.ascontiguousarray(a, dtype=float)
            self.arrays.append(np.frombuffer(a.tobytes(), dtype=float).reshape(a.shape))
        self.arrays = tuple(self.arrays)
        n = len(poses)
        if (
            not n
            or tuple(x.shape for x in self.arrays) != ((n, 4, 4), (n,), (n,))
            or not all(np.isfinite(x).all() for x in self.arrays)
            or np.min(np.diff(phase), initial=0) < -1e-10
        ):
            raise ValueError("invalid scheduled wire")

    def at(self, t):
        i = np.rint(t / 0.05).astype(int)
        if np.any(i < 0) or np.any(abs(t - 0.05 * i) > 1e-8):
            raise ValueError("invalid wire clock")
        return tuple(a[np.minimum(i, len(a) - 1)] for a in self.arrays)


def audit(path, mode):
    from governor import PersistentReference
    from audit_sealed import check
    from go2arx5_vla.trajectory.so3 import quat_to_matrix

    with np.load(path / (mode + ".private.npz"), allow_pickle=False) as z:
        active = WireProgram(path.name, z["poses"], z["grip"], z["source_s"])
    manager = PersistentReference(active, mode, 30.0)
    count = 0
    last_phase = 0.0
    pending = None
    with (path / mode / "CALLBACKS.private.jsonl").open() as f:
        for line in f:
            e = json.loads(line)
            if e["kind"] == "accept":
                doc = json.loads(e["raw"])
                meta = doc["_pi0294"]
                if meta["operation"] != "install":
                    continue
                now = meta["source_mpc_s"]
                if count and count % 5 == 0:
                    manager.revise(active, now, mode)
                key, (poses, g, phase) = manager.window(now)
                pt = doc["ee"]["waypoints"]
                check(
                    np.allclose(
                        [q["position"] for q in pt], poses[:, :3, 3], rtol=0, atol=2e-7
                    )
                    and np.allclose(
                        quat_to_matrix(np.array([q["quaternion_xyzw"] for q in pt])),
                        poses[:, :3, :3],
                        rtol=0,
                        atol=7e-7,
                    )
                    and phase[0] >= last_phase - 1e-10,
                    "changed scheduled wire",
                )
                last_phase = phase[0]
                pending = key, e
            elif e["kind"] == "installed" and pending is not None:
                key, accept = pending
                manager.acknowledge(key, accept["ref_id"], accept, e)
                count += 1
                pending = None
    check(
        pending is None
        and abs(last_phase - 49 / 30) < 1e-8
        and manager.consumed is None,
        "missing end",
    )
    return {
        "installs": count,
        "model_opportunities_after_start": manager.revision,
        "source_s_at_end": last_phase,
        "committed_prefix_failures": 0,
        "new_policy_predictions": 0,
        "native_consumption_ack": 0,
    }


if __name__ == "__main__":
    from sealed_panel import configure_site

    parser = argparse.ArgumentParser()
    for key in ("root", "folder", "output"):
        parser.add_argument(key, type=Path)
    a = parser.parse_args()
    configure_site(a.root.resolve(), Path(__file__).resolve().parents[2])
    from ablation import output

    out = {
        p.name: {m: audit(p, m) for m in ("C2_EXACT", "C2_BOUNDED")}
        for p in sorted(a.folder.glob("T*"))
        if p.is_dir()
    }
    if len(out) != 16:
        raise ValueError("incomplete panel")
    output(a.output, out)
