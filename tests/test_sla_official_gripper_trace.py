"""Synthetic CPU fixtures for receiver evidence integrity. No physical commands are issued."""

import hashlib
import json
import struct

import pytest

from scripts.sla_official_pt_transfer.gripper_trace import GripperTrace


def packet(seq, t, width):
    fraction = width / 0.044
    raw = struct.pack(
        "<8sQddd", b"GRTGRIP1", seq, 0.002 * t, fraction, 0.002 * (t + 200)
    )
    return hashlib.sha256(raw).hexdigest()


def fixture(widths=(0.020, 0.019, 0.020)):
    logs = {
        k: {"role": k, "episode": "synthetic", "rows": []}
        for k in ("issuer", "receiver", "state")
    }
    log = lambda k: logs[k]["rows"]
    state = {
        "valid": True,
        "contact": False,
        "grasp": False,
        "prohibited_contact": False,
        "actual_width_m": 0.0198,
    }
    log("state").append(dict(state, tick=0))
    log("receiver").append(
        {
            "event": "reset",
            "owner": "student",
            "previous_owner": None,
            "tick": 0,
            "mode": "WBC",
            "initial_finger_targets_m": [widths[0]] * 2,
        }
    )
    for i, width in enumerate(widths):
        t = i * 25
        key = {
            "sequence": i,
            "owner": "student",
            "wire_sha256": packet(i, t, width),
            "width_m": width,
        }
        log("issuer").append(
            dict(
                key,
                event="issue",
                tick=t,
                expiry_tick=t + 200,
                native_fraction=width / 0.044,
                source_s=0.002 * t,
                expiry_s=0.002 * (t + 200),
            )
        )
        log("receiver").extend(
            [dict(key, event="ack", tick=t), dict(key, event="consumed", tick=t + 25)]
        )
        log("state").extend(
            dict(
                state,
                tick=t + j,
                applied_sequence=i,
                applied_wire_sha256=key["wire_sha256"],
            )
            for j in range(1, 26)
        )
    return logs


def streams(logs):
    return {k: json.dumps(v, sort_keys=True).encode() for k, v in logs.items()}


def verifier(stream):
    return GripperTrace(
        "synthetic",
        "student",
        {k: hashlib.sha256(v).hexdigest() for k, v in stream.items()},
    )


def test_noncontact_reset_close_open_cpu_only():
    raw = streams(fixture())
    assert verifier(raw).verify(raw, reset_tick=0)["verified_intervals"] == 3


def test_self_consistent_forged_stream_fails_registration():
    raw = streams(fixture())
    forged = streams(fixture((0.020, 0.021, 0.020)))
    with pytest.raises(ValueError, match="unregistered"):
        verifier(raw).verify(forged, reset_tick=0)


@pytest.mark.parametrize(
    "role,index,field,value",
    [
        ("receiver", 0, "owner", "teacher"),
        ("receiver", 0, "previous_owner", "teacher"),
        ("receiver", 0, "initial_finger_targets_m", [0.0198] * 2),
        ("receiver", 0, "tick", 25),
        ("receiver", 1, "event", "eef_ack"),
        ("receiver", 3, "tick", 38),
        ("receiver", 3, "sequence", 0),
        ("receiver", 2, "event", "ack"),
        ("receiver", 2, "tick", 500),
        ("receiver", 2, "width_m", 0.0198),
        ("issuer", 0, "sequence", False),
        ("issuer", 1, "expiry_tick", 226),
        ("issuer", 1, "source_s", 0.049),
        ("issuer", 2, "tick", 51),
        ("issuer", 1, "wire_sha256", "0" * 64),
        ("issuer", 1, "native_fraction", 0.5),
        ("state", 10, "valid", 1),
        ("state", 26, "contact", True),
        ("state", 34, "grasp", True),
        ("state", 40, "prohibited_contact", True),
        ("state", 36, "applied_sequence", 0),
        ("state", 1, "applied_wire_sha256", "0" * 64),
        ("state", 17, "actual_width_m", float("nan")),
    ],
)
def test_invalid_evidence_even_if_pinned(role, index, field, value):
    logs = fixture()
    logs[role]["rows"][index][field] = value
    raw = streams(logs)
    with pytest.raises(ValueError):
        verifier(raw).verify(raw, reset_tick=0)


def test_rate_from_last_consumed_and_original_threshold():
    raw = streams(fixture((0.020, 0.018, 0.020)))
    assert verifier(raw).verify(raw, reset_tick=0)["verified_intervals"] == 3
    raw = streams(fixture((0.020, 0.01799, 0.020)))
    with pytest.raises(ValueError, match="rate"):
        verifier(raw).verify(raw, reset_tick=0)


def test_missing_actual_tick_and_missing_ack():
    for role in ("state", "receiver"):
        logs = fixture()
        del logs[role]["rows"][1]
        raw = streams(logs)
        with pytest.raises(ValueError):
            verifier(raw).verify(raw, reset_tick=0)


def test_delayed_receiver_ack_preserves_previous_consumption():
    logs = fixture()
    logs["receiver"]["rows"][3]["tick"] = 27
    invalid = streams(logs)
    with pytest.raises(ValueError, match="applied command"):
        verifier(invalid).verify(invalid, reset_tick=0)
    previous = packet(0, 0, 0.020)
    for tick in (26, 27):
        logs["state"]["rows"][tick].update(
            applied_sequence=0, applied_wire_sha256=previous
        )
    valid = streams(logs)
    assert verifier(valid).verify(valid, reset_tick=0)["verified_intervals"] == 3
