from __future__ import annotations

from scripts.stage22_aggregate_profile_seeds import _stats


def test_seed_aggregate_stats_use_interpolated_even_median() -> None:
    result = _stats([1.0, 2.0, 3.0, 4.0])
    assert result["median"] == 2.5
    assert result["p95"] == 4.0


def test_seed_aggregate_stats_ignore_missing_values() -> None:
    result = _stats([None, 2.0, 4.0])
    assert result["count"] == 2
    assert result["mean"] == 3.0
