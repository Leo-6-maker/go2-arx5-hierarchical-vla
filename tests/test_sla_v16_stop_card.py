from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts/sla_hierarchical_v16")
)
from stop_card_readiness import displacement


def test_motion_uses_measured_interval_and_final_tick():
    ticks = np.arange(6414, 6440)
    positions = np.zeros((len(ticks), 3))
    positions[:, 0] = (ticks - 6414) * 0.00001
    result = displacement(positions, ticks, 6439, 25)
    assert result["distance_mm"] == pytest.approx(0.25)
    assert result["last_step_speed_mm_s"] == pytest.approx(5.0)
    assert result["mean_displacement_speed_mm_s"] == pytest.approx(5.0)


@pytest.mark.parametrize(
    "bad", ["hole", "duplicate", "unobserved_endpoint", "nonfinite"]
)
def test_incomplete_or_invalid_motion_sample_cannot_pass(bad):
    ticks = np.arange(6414, 6440)
    xyz = np.zeros((26, 3))
    if bad == "hole":
        ticks = np.delete(ticks, 3)
        xyz = np.delete(xyz, 3, axis=0)
    elif bad == "duplicate":
        ticks[5] = ticks[4]
    elif bad == "unobserved_endpoint":
        ticks[-1] += 1
    else:
        xyz[7, 2] = float("nan")
    with pytest.raises(ValueError):
        displacement(xyz, ticks, 6439, 25)
