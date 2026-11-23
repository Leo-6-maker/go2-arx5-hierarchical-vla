"""Shadow-only gripper handoff checks. All positive receipts here are synthetic."""

from copy import deepcopy

import pytest

from scripts.sla_transfer_v15.gripper_ownership import assess


def boundary(kind):
    previous = "teacher" if kind == "owner_transfer" else "student"
    row = {
        "student_owner": "student",
        "teacher_owner": "teacher",
        "episode": "A",
        "contact_valid": True,
        "prohibited_contact": False,
        "contact": False,
        "grasp": False,
        "delivered_sequence": 12,
        "consumed_at_tick": 301,
        "previous_command_m": 0.02,
        "new_first_m": 0.021,
        "measured_m": 0.035,
    }
    event = {
        "kind": kind,
        "episode": "A",
        "owner": "student",
        "previous_owner": previous,
        "native_tick": 301,
        "contact_valid": True,
        "prohibited_contact": False,
        "contact": False,
        "grasp": False,
        "last_gripper_consumption": {
            "episode": "A",
            "owner": previous,
            "issued_sequence": 12,
            "ack_sequence": 12,
            "consumed_sequence": 12,
            "consumed_tick": 301,
            "issued_width_m": 0.02,
            "consumed_width_m": 0.02,
        },
    }
    return row, event


@pytest.mark.parametrize("kind", ["student_owned_reset", "owner_transfer"])
def test_owner_bound_reset_or_transfer(kind):
    row, event = boundary(kind)
    assert assess(row, receiver_boundary=event) == "SHADOW_MATCH_ONLY"
    # Even a complete self-reported receipt on the policy side is not receiver evidence.
    row.update(
        gripper_receipt=event,
        student_owned_reset=True,
        student_owner_transfer_receipt=event,
    )
    assert assess(row) == "HANDOVER_BLOCKED"


@pytest.mark.parametrize(
    "key",
    [
        "owner",
        "episode",
        "issued_sequence",
        "ack_sequence",
        "consumed_sequence",
        "consumed_tick",
        "issued_width_m",
        "consumed_width_m",
    ],
)
def test_incomplete_or_wrong_command_receipt(key):
    row, event = boundary("owner_transfer")
    for fake in (None, True, "unrelated", 0):
        damaged = deepcopy(event)
        damaged["last_gripper_consumption"][key] = fake
        assert assess(row, receiver_boundary=damaged) != "SHADOW_MATCH_ONLY"
    row.pop("delivered_sequence")
    assert (
        assess(row, receiver_boundary=event) == "CONSUMED_GRIPPER_IDENTITY_UNOBSERVED"
    )


@pytest.mark.parametrize("flag", ["contact", "grasp"])
@pytest.mark.parametrize("width", [0.019, 0.02, 0.021])
def test_transfer_requires_no_contact_even_if_closing(flag, width):
    row, event = boundary("owner_transfer")
    row[flag] = event[flag] = True
    row["new_first_m"] = width
    assert assess(row, receiver_boundary=event) == "CONTACT_HANDOVER_BLOCKED"
    row[flag] = False
    assert assess(row, receiver_boundary=event) == "CONTACT_STATE_BLOCKED"


def test_measured_width_cannot_replace_consumption():
    row, event = boundary("owner_transfer")
    row["measured_m"] = row["new_first_m"] = 0.035
    assert assess(row, receiver_boundary=event) == "GRIPPER_RATE_BLOCKED"
    row["previous_command_m"] = row["measured_m"]
    assert (
        assess(row, receiver_boundary=event) == "CONSUMED_GRIPPER_IDENTITY_UNOBSERVED"
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("new_first_m", float("nan")),
        ("new_first_m", 0.043),
        ("measured_m", None),
        ("new_first_m", True),
        ("consumed_at_tick", 300),
        ("student_owner", "teacher"),
        ("prohibited_contact", None),
        ("contact_valid", 1),
        ("contact", "False"),
    ],
)
def test_invalid_inputs_fail_closed(key, value):
    row, event = boundary("owner_transfer")
    row[key] = value
    assert assess(row, receiver_boundary=event) != "SHADOW_MATCH_ONLY"


def test_open_during_contact_and_exact_seam_limit():
    row, event = boundary("student_owned_reset")
    row["contact"] = event["contact"] = True
    assert assess(row, receiver_boundary=event) == "OPEN_DURING_CONTACT_BLOCKED"
    row["contact"] = event["contact"] = False
    row["new_first_m"] = 0.022
    assert assess(row, receiver_boundary=event) == "SHADOW_MATCH_ONLY"
    row["new_first_m"] += 1e-7
    assert assess(row, receiver_boundary=event) == "GRIPPER_RATE_BLOCKED"
