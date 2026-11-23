from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts/sla_hierarchical_v16")
)
from feedback_policy_panel import arm_diff, call_population
from feedback_gpu import retained_inference


def test_capture_precedes_mutating_decoder_and_restores_original_on_fault():
    def convert(s):
        s["actions"] += 3
        if s["state"][0]:
            raise RuntimeError("decoder failure")
        return s

    class Policy:
        def __init__(self):
            self._output_transform = convert

        def infer(self, obs):
            return self._output_transform(obs)

    policy = Policy()
    for fault in (False, True):
        raw = np.zeros((50, 32), np.float32)
        obs = {"state": np.array([int(fault)]), "actions": raw}
        captured = {}
        if fault:
            with pytest.raises(RuntimeError):
                retained_inference(policy, obs, captured)
        else:
            assert (retained_inference(policy, obs, captured)["actions"] == 3).all()
        assert policy._output_transform is convert and (captured["actions"] == 0).all()
        assert str(captured["actions_dtype"]) == "float32"


def test_call_accounting_keeps_an_interruption_and_rejects_reentry():
    entry = dict(event="FORWARD_ENTERED", id="A", input_sha256="abc", seed=[0, 0])
    ret = dict(entry, event="FORWARD_RETURNED")
    fail = dict(event="FORWARD_FAILED", id="A")
    assert call_population([entry], ["A", "B"]) == ([entry], [], False)
    assert call_population([entry, fail], ["A", "B"]) == ([entry], [], False)
    with pytest.raises(ValueError):
        call_population([entry, fail, entry], ["A", "B"])
    with pytest.raises(ValueError):
        call_population([entry, dict(ret, seed=[0, 1])], ["A", "B"])
    assert call_population(
        [entry, ret, dict(event="PANEL_RETURNED", count=1)], ["A"]
    ) == ([entry], [ret], True)


def test_width_delta_units_and_quaternion_double_cover():
    a = np.zeros((50, 8))
    a[:, 3] = 1
    a[:, 7] = 0.6
    b = a.copy()
    b[:, 3] = -1
    b[:, 7] = 0.55
    metrics = arm_diff(a, b)
    assert metrics["angle_max_deg"] == 0
    assert metrics["width_rms_mm"] == pytest.approx(2)
    assert metrics["width_signed_mean_mm"] == pytest.approx(2)
