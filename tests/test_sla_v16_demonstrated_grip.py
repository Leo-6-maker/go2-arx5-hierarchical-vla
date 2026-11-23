import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts/sla_hierarchical_v16")
)
from demonstrated_grip import SOURCE_T, compare, events


def test_command_step_and_interpolated_prediction_cross_at_distinct_times():
    assert events([0, 0.05], [0.60, 0.50], step=True) == [{"kind": "close", "s": 0.05}]
    e = events([0, 0.05], [0.60, 0.50], step=False)
    assert e[0]["kind"] == "close" and e[0]["s"] == pytest.approx(0.025)
    assert events([0, 0.05, 0.10], [0.55, 0.55, 0.50], step=True) == []


def test_equal_empty_sequence_and_mismatch_have_distinct_timing_results():
    static = np.full(50, 0.45)
    matched = compare(static, static, [], 0.45)
    assert matched["sequence_equal"] and matched["matched_time_errors_s"] == []
    proposal = np.linspace(0.4, 0.8, 50)
    mismatch = compare(proposal, static, [], 0.45)
    assert not mismatch["sequence_equal"] and mismatch["matched_time_errors_s"] is None
    assert mismatch["width_rmse"] > mismatch["hold_rmse"]
    assert events(SOURCE_T, proposal, step=False)[0]["kind"] == "open"
