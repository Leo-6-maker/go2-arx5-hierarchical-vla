from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/sla_hierarchical_v16"))
from gripper_alignment import alignment_widths
from handoff_factors import ReturnedPreview
from prefix_replay import WireProgram


def sample(start, source_end):
    t = np.arange(25) * 0.05
    phase = np.maximum(t - 1, 0)
    raw = np.interp(t, [0, 1, 1.2], [start, start, source_end])
    return t, phase, raw


def test_bridge_preserves_every_source_knot_and_uses_consumed_command():
    t, phase, raw = sample(0.75, 0.32)
    aligned, e = alignment_widths(t, phase, raw, 0.030, 1.0, False)
    assert not e["rejections"]
    assert np.isclose(0.002 + 0.04 * aligned[0], 0.030)
    assert np.all(np.diff(aligned[t <= 1]) >= 0)
    assert np.array_equal(aligned[t >= 1], raw[t >= 1])
    assert e["continuous_peak_mm_s"] == pytest.approx(3.75)


@pytest.mark.parametrize(
    ("previous", "start", "contact", "reason"),
    [
        (0.56, 0.54, False, "UNATTRIBUTED_THRESHOLD_CROSSING"),
        (0.64, 0.66, False, "UNATTRIBUTED_THRESHOLD_CROSSING"),
        (0.80, 0.81, True, "CONTACT_OR_GRASP_AT_HANDOFF"),
        (0.99, 0.40, False, "INSUFFICIENT_EXISTING_ALIGNMENT_TIME"),
        (1.01, 0.98, False, "PREVIOUS_COMMAND_OUTSIDE_WORKING_RANGE"),
    ],
)
def test_bridge_refusals(previous, start, contact, reason):
    t, phase, raw = sample(start, start)
    aligned, e = alignment_widths(t, phase, raw, 0.002 + 0.040 * previous, 1, contact)
    assert aligned is None and reason in e["rejections"]


def test_exact_threshold_and_zero_delta_do_not_create_a_new_event():
    for threshold in (0.55, 0.65):
        t, phase, raw = sample(threshold, threshold)
        aligned, e = alignment_widths(t, phase, raw, 0.002 + 0.040 * threshold, 1, True)
        assert aligned is not None and not e["rejections"]


def test_preview_switches_only_when_prediction_available():
    class Revision:
        identity = "episode"
        execution_times = [0, 4]
        next_hash = "new"

        def __init__(self):
            p = np.repeat(np.eye(4)[None], 100, axis=0)
            self.previous = WireProgram(self.identity, p, np.zeros(100), np.zeros(100))

        def at(self, t):
            p, w, s = self.previous.at(t)
            return p, w + 1, s

        def source_hash_at(self, t):
            return np.repeat(self.next_hash, len(t))

    fixture = ReturnedPreview(Revision(), 5)
    grid = np.arange(41) * 0.05
    assert np.all(fixture.at(0.4 + grid)[1] == 0)
    assert np.all(fixture.at(0.5 + grid)[1] == 1)
    with pytest.raises(ValueError, match="window"):
        fixture.at(grid[:40])
