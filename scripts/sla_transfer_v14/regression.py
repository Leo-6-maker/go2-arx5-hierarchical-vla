"""Isolated diagnostic correction. The original compiler and numerical thresholds keep their code."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
from canonical_gate import main as original_replay
from go2arx5_vla.world_model import fusion_trajectory_adapter as limits
from go2arx5_vla.world_model import mfree_query_diagnostics_v2 as diagnostics
from go2arx5_vla.world_model.task_learning_executability_v3 import sequence
from identity import read, record, sha


def replay(root, out):
    sealed = root / "v13/run01"
    # The V1.3 replay accepts one directory for input and output. Link immutable
    # inputs into a fresh output directory; the replay only creates a new receipt.
    for name in ("CANONICAL_ACTUAL.json", "canonical"):
        (out / name).symlink_to((sealed / name).resolve())
    original_series = diagnostics.path_series
    changes = []

    def matching_series(rows, *, grip_is_mechanical_fraction=False):
        series = original_series(
            rows, grip_is_mechanical_fraction=grip_is_mechanical_fraction
        )
        rot = limits._poses(rows)[:, :3, :3]
        values = limits.geodesic_angle(rot[:-1], rot[1:]) / limits.ATOM_DT_S
        old = series["angular_velocity"]["peak_value"]
        new = float(values.max())
        if not np.isclose(old, new, atol=1e-9, rtol=1e-8):
            changes.append(
                {
                    "old_rad_s": old,
                    "new_rad_s": new,
                    "abs_difference_rad_s": abs(new - old),
                }
            )
        series["angular_velocity"] = sequence(
            values, limits.MAX_ANGULAR_SPEED_RAD_S, unit="rad/s"
        )
        return series

    with patch.object(diagnostics, "path_series", matching_series):
        original_replay(root, out)
    before = read(sealed / "CANONICAL_GATE.json")["rows"]
    after = read(out / "CANONICAL_GATE.json")["rows"]
    assert len(before) == len(after) == 78
    assert all(
        (a["id"], a["arm"], a["H40"], a["legal"])
        == (b["id"], b["arm"], b["H40"], b["legal"])
        for a, b in zip(before, after)
    )
    assert all("attribution_unavailable" not in r for r in after)
    affected = [
        {"window": r["id"], "arm": r["arm"], "H40": r["H40"]}
        for r in before
        if "attribution_unavailable" in r
    ]
    assert len(affected) == 1 and changes
    record(
        out / "B0_ANGULAR_REGRESSION_ACTUAL.json",
        {
            "scope": "diagnostic scalar angular-speed magnitudes use the same per-interval geodesic operation as the frozen compiler; acceleration retains signed rotation-vector calculus",
            "original_sha256": sha(sealed / "CANONICAL_GATE.json"),
            "new_receipt_sha256": sha(out / "CANONICAL_GATE.json"),
            "prior_exceptions": affected,
            "peak_differences": changes,
            "canonical_replays": 78,
            "changed_gate_decisions": 0,
            "remaining_auxiliary_exceptions": 0,
            "native_requests": 0,
            "new_model_forwards": 0,
            "training_updates": 0,
        },
    )


if __name__ == "__main__":
    import sys

    replay(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
