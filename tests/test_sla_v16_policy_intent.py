import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts/sla_hierarchical_v16")
)
from policy_intent import evaluate, features


def poses(z):
    p = np.repeat(np.eye(4)[None], len(z), axis=0)
    p[:, 2, 3] = z
    return p


@pytest.mark.parametrize(
    "phase,z,events,expected",
    [
        ("DESCEND", [0.04, 0.03], [], True),
        ("DESCEND", [0.03, 0.04], [], False),
        ("LIFT", [0.02, 0.023], ["open"], False),
        ("LIFT", [0.02, 0.023], [], True),
        ("CLOSE", [0.02, 0.02], ["open"], False),
        ("CLOSE", [0.02, 0.02], ["close"], True),
        ("HOLD", [0.02, 0.026, 0.02], [], False),
    ],
)
def test_offline_direction_and_events(phase, z, events, expected):
    p = poses(z)
    objects = {"red": {"pose": np.eye(4)}, "blue": {"pose": np.eye(4)}}
    f = features(p, events)
    assert (
        evaluate(f, p, objects, "HOLD-red", phase, {})["phase_proxy_pass"] == expected
    )


def test_motion_signature_does_not_use_task_object():
    p = poses([0, 0.003])
    f = features(p, [])
    assert f["signature"] == "+Z"
    assert features(p, ["open"])["signature"] == "open"
