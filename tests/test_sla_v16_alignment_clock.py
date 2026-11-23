import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/sla_hierarchical_v16"))
from compiler import CruiseClock, DT


LIMITS = dict(
    linear_velocity=0.06,
    angular_velocity=0.3,
    linear_acceleration=0.16,
    angular_acceleration=0.8,
)


@pytest.mark.parametrize(
    "distance,angle,minimum",
    [
        (0, 0, -0.1),
        (0, 0, 1.031),
        (0.00005, 0, 0),
        (0.037, 0.14, 0),
        (0, np.pi, 0),
        (0.003, 0.01, 4.27),
    ],
)
def test_scheduled_phase_has_stationary_ends_and_retains_both_preview_limits(
    distance, angle, minimum
):
    clock = CruiseClock(distance, angle, LIMITS, minimum)
    assert clock.duration + 1e-12 >= max(minimum, DT)
    assert clock.duration / DT == pytest.approx(round(clock.duration / DT))
    assert clock.phase(np.array([-1, 0, 1, 2])).tolist() == [0, 0, 1, 1]
    # Resolve the ramps separately even on strongly rotation-limited paths.
    r = clock.ramp_fraction
    for lo, hi in ((0, r), (r, 1 - r), (1 - r, 1)):
        if hi - lo < 1e-12:
            continue
        grid = np.linspace(lo, hi, 401)
        dt = (hi - lo) * clock.duration / 400
        speed = np.diff(clock.phase(grid)) / dt
        assert speed.min() >= -1e-9
        acceleration = np.diff(speed) / dt
        for length, key in ((distance, "linear"), (angle, "angular")):
            limit = LIMITS[key + "_acceleration"]
            assert (
                np.max(speed) * length
                <= 0.9 * min(LIMITS[key + "_velocity"], limit * DT) + 1e-9
            )
            assert np.max(np.abs(acceleration)) * length <= 0.9 * limit + 1e-6
    eps = r * 1e-3
    assert clock.phase(eps) < eps * 1e-5
    assert 1 - clock.phase(1 - eps) < eps * 1e-5


@pytest.mark.parametrize("d,a,m", [(-1, 1, 0), (1, np.nan, 0), (1, 0, np.inf)])
def test_rejects_invalid_clock_input(d, a, m):
    with pytest.raises(ValueError, match="invalid alignment"):
        CruiseClock(d, a, LIMITS, m)
