from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/sla_hierarchical_v16"))
from compiler import SourceCurve
from prefix_replay import WireProgram
from revision_wire import NewSourcePhaseView, RevisionWire, first_publication


def joined(previous_width, new_width):
    poses = np.repeat(np.eye(4)[None], 180, axis=0)
    phase = np.minimum(np.arange(180) * 0.01, 49 / 30)
    previous = WireProgram("episode", poses, np.full(180, previous_width), phase)
    raw = np.zeros((50, 16))
    raw[:, 11] = 1.0
    raw[:, 8] = np.linspace(0.002, 0.020, 50)
    raw[:, 15] = new_width
    source = SourceCurve(raw, np.eye(4), np.eye(4))
    limits = dict(
        linear_velocity=0.05,
        angular_velocity=0.25,
        linear_acceleration=0.25,
        angular_acceleration=1.0,
    )
    return RevisionWire(previous, "old-return", "new-return", 4, source, limits)


def test_protected_future_and_independent_source_clocks():
    r = joined(0.73, 0.73)
    t = np.arange(r.cut + 4) * 0.05
    assert r.cut == 49
    old = r.previous.at(t)
    future = r.at(t)
    for a, b in zip(old, future, strict=True):
        assert np.array_equal(a[: r.cut + 1], b[: r.cut + 1])
    assert (
        r.source_hash_at(t).tolist()
        == ["old-return"] * (r.cut + 1) + ["new-return"] * 3
    )
    assert old[2][r.cut] > 0 and future[2][r.cut + 1] == 0
    assert NewSourcePhaseView(r).at(t)[2].max() == 0
    r.check_gripper_handoff()
    with pytest.raises(ValueError, match="wire clock"):
        r.at([0.051])


@pytest.mark.parametrize(
    ("a", "b", "reason"),
    [
        (0.551, 0.549, "unattributed"),
        (0.649, 0.651, "unattributed"),
        (0.99, 0.89, "40 mm/s"),
    ],
)
def test_even_a_slow_spurious_event_blocks_the_join(a, b, reason):
    with pytest.raises(ValueError, match=reason):
        joined(a, b).check_gripper_handoff()


def test_arrival_is_never_rounded_to_an_earlier_publication():
    assert first_publication(0.8) == 8
    assert first_publication(0.800001) == 9
    with pytest.raises(ValueError):
        first_publication(float("nan"))
