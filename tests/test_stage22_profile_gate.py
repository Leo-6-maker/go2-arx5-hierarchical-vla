from __future__ import annotations

from scripts.stage22_profile_gate import _metric_values


def test_profile_gate_profile_score_metrics_are_frozen_to_five_quantities() -> None:
    payload = {
        "summary": {
            "position_mean_error_m": {"median": 0.01},
            "endpoint_position_error_m": {"median": 0.02},
            "path_length_ratio": {"median": 1.2},
            "mean_speed_ratio": {"median": 0.9},
            "max_speed_ratio": {"median": 1.1},
        }
    }
    assert set(_metric_values(payload)) == {
        "position_mean_error_m",
        "endpoint_position_error_m",
        "path_length_ratio_abs_deviation",
        "mean_speed_ratio_abs_deviation",
        "max_speed_ratio_abs_deviation",
    }
