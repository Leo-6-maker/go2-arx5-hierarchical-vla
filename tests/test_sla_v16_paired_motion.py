from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts/sla_hierarchical_v16")
)
from paired_motion import aligned, contrasted, evaluate_pair, score_horizon


def path(n):
    poses = np.repeat(np.eye(4)[None], n, axis=0)
    poses[:, 0, 3] = np.arange(n) * 0.001
    return poses


def case(key, tick, poses):
    return dict(
        key=key,
        episode="one",
        instruction="HOLD-red",
        window={"tick": tick},
        curve=SimpleNamespace(knots=poses),
        archive={"phase": "HOLD"},
        raw_file_sha256=key,
    )


def test_same_physical_time_excludes_anchor_and_retains_censored_secondary():
    older = path(50)
    newer = path(50)
    newer[:, 0, 3] += 0.015
    o, n = case("old", 100, older), case("new", 350, newer)
    poses = {
        t: dict(
            tcp=older[15 + 3 * i] if t >= 350 else older[0],
            objects={"red": np.zeros(3), "blue": np.ones(3)},
        )
        for i, t in enumerate(range(350, 601, 50))
    }
    poses[100] = dict(tcp=older[0], objects=poses[350]["objects"])
    row = evaluate_pair(o, n, poses, 600)
    p = row["primary"]["outcomes"]
    assert row["primary"]["future_observations"] == 5 and row["secondary"] is None
    assert p["old"]["position_rmse_mm"] == 0 and p["new"][
        "position_rmse_mm"
    ] == pytest.approx(0, abs=1e-12)
    assert p["hold"]["position_rmse_mm"] == pytest.approx(
        np.sqrt(np.mean((3 * np.arange(1, 6)) ** 2))
    )
    assert (
        p["hold"]["displacement_cosine"] is None
        and row["boundary"]["revision_cosine"] is None
    )
    with pytest.raises(ValueError):
        evaluate_pair(o, case("new", 351, newer), poses, 600)


def test_world_rotation_increment_and_translation_are_scored_separately():
    src = path(6)
    src[:, :3, :3] = Rotation.from_euler(
        "y", np.arange(6) * 8, degrees=True
    ).as_matrix()
    actual = Rotation.from_euler("x", 70, degrees=True).as_matrix()
    truth = src.copy()
    truth[:, :3, :3] = src[:, :3, :3] @ actual
    truth[:, 2, 3] += 0.02
    assert np.allclose(aligned(src, truth[0]), truth)
    data = score_horizon(src, truth, truth)["outcomes"]
    assert data["old"]["position_rmse_mm"] == pytest.approx(20)
    assert data["old_aligned"]["rotation_rmse_deg"] == pytest.approx(0, abs=1e-12)
    assert (
        data["new"]["position_rmse_mm"] == 0 and data["new"]["rotation_rmse_deg"] == 0
    )


def test_absolute_improvement_does_not_imply_shape_improvement():
    truth = path(6)
    old = truth.copy()
    old[:, 1, 3] = 0.050
    new = truth.copy()
    new[:, 0, 3] *= -1
    p = score_horizon(old, new, truth)
    rows = [dict(primary=p), dict(primary=None)]
    a = contrasted(rows, "primary", "position_rmse_mm", "new", "old")
    b = contrasted(rows, "primary", "position_rmse_mm", "new_aligned", "old_aligned")
    assert a["n"] == a["lower"] == b["higher"] == 1
    assert p["outcomes"]["new"]["displacement_cosine"] == -1
