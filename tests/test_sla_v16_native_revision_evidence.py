from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts/sla_hierarchical_v16")
)
from native_revision_evidence import installed_overlap


def document(start, delta, orientation):
    points = [
        dict(
            time=i * 0.05,
            position=[start + i * 0.05 + (delta if i else 0), 0, 0],
            quaternion_xyzw=[0, 0, 0, orientation],
        )
        for i in range(41)
    ]
    return dict(
        _pi0294={"source_mpc_s": start},
        ee={"waypoints": points},
        base={"waypoints": points},
    )


def test_two_common_absolute_grids_with_opposite_quaternion_signs():
    first = document(3.0, 0, 1)
    second = document(3.1, 0, -1)
    actual = installed_overlap(first, second)
    assert actual["ee"]["overlap_knots"] == 39
    assert actual["ee"]["changed"] == 0 and actual["ee"]["first_change_s"] is None
    second = document(3.1, 0.0004, -1)
    actual = installed_overlap(first, second)["ee"]
    assert actual["changed"] == 38 and actual["first_change_s"] == 0.05
    assert np.isclose(actual["at_new_origin_mm"], 0) and np.isclose(
        actual["maximum_translation_mm"], 0.4
    )


def test_off_grid_installation_rejected():
    with pytest.raises(ValueError):
        installed_overlap(document(3.0, 0, 1), document(3.11, 0, 1))
