import hashlib
import sys
from pathlib import Path
import pytest
from test_sla_official_gripper_trace import fixture, streams

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts/sla_hierarchical_v16")
)
from owner_session import verify_session

BIND = dict(
    episode="synthetic",
    owner="student",
    epoch="reset-1",
    lease="card-1",
    receiver_sha256="a" * 64,
)


def sample():
    d = fixture()
    end = len(d["state"]["rows"]) - 1
    d["authority"] = dict(
        rows=[dict(event="exclusive_grant", tick=0), dict(event="release", tick=end)]
    )
    d["stop"] = dict(
        rows=[
            dict(event="stop", tick=end),
            dict(event="journal_closed", last_integrated_tick=end, after_stop_ticks=0),
        ]
    )
    for role, log in d.items():
        log.update(role=role, **BIND)
        if role in ("issuer", "receiver", "state"):
            for row in log["rows"]:
                row.update(epoch=BIND["epoch"], lease=BIND["lease"])
    d["receiver"]["rows"][0]["last_consumed_sequence"] = -1
    x = 0.020
    for i, s in enumerate(d["state"]["rows"]):
        if i:
            goal = d["issuer"]["rows"][s["applied_sequence"]]["width_m"]
            x += max(-0.00008, min(0.00008, goal - x))
        s.update(applied_finger_targets_m=[x, x], actual_finger_positions_m=[x, x])
    return d


def verify(d):
    raw = streams(d)
    return verify_session(
        raw, {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()}, **BIND
    )


def test_session_cpu_only():
    assert verify(sample())["post_stop_ticks"] == 0


@pytest.mark.parametrize(
    "role,i,field,value",
    [
        ("authority", 0, "event", "grant"),
        ("authority", 1, "tick", 74),
        ("receiver", 0, "last_consumed_sequence", 0),
        ("receiver", 2, "epoch", "older"),
        ("issuer", 1, "lease", "other"),
        ("state", 26, "epoch", None),
        ("state", 26, "applied_finger_targets_m", [0.019, 0.019]),
        ("state", 1, "actual_finger_positions_m", [0.02]),
        ("stop", 1, "after_stop_ticks", 1),
        ("stop", 1, "last_integrated_tick", 74),
        ("stop", 0, "tick", 74),
    ],
)
def test_reject_even_with_registered_stream(role, i, field, value):
    d = sample()
    d[role]["rows"][i][field] = value
    with pytest.raises(ValueError):
        verify(d)


def test_other_reset_and_duplicate_grant():
    d = sample()
    d["authority"]["rows"].insert(1, d["authority"]["rows"][0])
    with pytest.raises(ValueError):
        verify(d)
    d = sample()
    d["state"]["lease"] = "earlier"
    with pytest.raises(ValueError):
        verify(d)
