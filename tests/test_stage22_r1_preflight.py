from __future__ import annotations

from scripts.stage22_r1_preflight import (
    _parse_gpu_snapshot,
    _parse_trainable_parameters,
    _parse_train_runtime,
)


def test_r1_preflight_parses_gpu_and_training_telemetry() -> None:
    rows = _parse_gpu_snapshot(
        "0, NVIDIA Test, GPU-abc, 16303, 1234, 7\n"
        "1, NVIDIA Test, GPU-def, 32760, 2345, 9\n"
    )
    assert rows[0]["uuid"] == "GPU-abc"
    assert rows[1]["memory_total_mib"] == 32760
    assert _parse_trainable_parameters("Trainable parameters: 12,345 (0.12%)") == 12345
    assert _parse_train_runtime("'train_runtime': 4.25") == 4.25


def test_r1_preflight_ignores_malformed_gpu_rows() -> None:
    assert _parse_gpu_snapshot("not a csv row\n0, bad") == []
    assert _parse_trainable_parameters("no parameter summary") is None
    assert _parse_train_runtime("no runtime summary") is None
