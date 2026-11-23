"""Audit the first student gripper command at an ownership boundary, in shadow only."""

import math


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def assess(record, *, receiver_boundary=None):
    """The caller must supply independently verified gripper evidence, outside the proposal.

    The historical GRTGRIP1 receiver does not export an owner ACK. Its archived
    windows intentionally have no receiver_boundary. This function never issues a command.
    """
    if (
        record.get("contact_valid") is not True
        or record.get("prohibited_contact") is not False
        or any(type(record.get(k)) is not bool for k in ("contact", "grasp"))
    ):
        return "CONTACT_STATE_BLOCKED"
    if not isinstance(receiver_boundary, dict):
        return "HANDOVER_BLOCKED"
    mode = receiver_boundary.get("kind")
    if mode not in ("student_owned_reset", "owner_transfer"):
        return "HANDOVER_BLOCKED"
    if any(
        receiver_boundary.get(k) is not record[k]
        for k in ("contact_valid", "prohibited_contact", "contact", "grasp")
    ):
        return "CONTACT_STATE_BLOCKED"
    if mode == "owner_transfer" and (record["contact"] or record["grasp"]):
        return "CONTACT_HANDOVER_BLOCKED"

    owner = record.get("student_owner")
    episode = record.get("episode")
    previous_owner = record.get("teacher_owner") if mode == "owner_transfer" else owner
    if (
        any(
            not isinstance(s, str) or not s.strip()
            for s in (owner, previous_owner, episode)
        )
        or (mode == "owner_transfer" and previous_owner == owner)
        or receiver_boundary.get("episode") != episode
        or receiver_boundary.get("owner") != owner
        or receiver_boundary.get("previous_owner") != previous_owner
    ):
        return "HANDOVER_BLOCKED"

    # The last consumption belongs to the prior command issuer, including a teacher
    # at an explicit transfer. EEF ACKs and an observed finger position cannot fill it.
    receipt = receiver_boundary.get("last_gripper_consumption")
    seq, tick = record.get("delivered_sequence"), record.get("consumed_at_tick")
    if (
        not isinstance(receipt, dict)
        or type(seq) is not int
        or seq < 0
        or type(tick) is not int
        or tick <= 0
        or type(receiver_boundary.get("native_tick")) is not int
        or receiver_boundary["native_tick"] != tick
        or receipt.get("episode") != episode
        or receipt.get("owner") != previous_owner
        or any(
            type(receipt.get(k)) is not int or receipt[k] != seq
            for k in ("issued_sequence", "ack_sequence", "consumed_sequence")
        )
        or type(receipt.get("consumed_tick")) is not int
        or receipt["consumed_tick"] != tick
    ):
        return "CONSUMED_GRIPPER_IDENTITY_UNOBSERVED"
    previous = record.get("previous_command_m")
    proposed = record.get("new_first_m")
    if not all(
        number(v)
        for v in (
            previous,
            proposed,
            record.get("measured_m"),
            receipt.get("issued_width_m"),
            receipt.get("consumed_width_m"),
        )
    ):
        return "GRIPPER_NONFINITE"
    if previous != receipt["issued_width_m"] or previous != receipt["consumed_width_m"]:
        return "CONSUMED_GRIPPER_IDENTITY_UNOBSERVED"
    # The receiver compares normalized travel with .002/.044 and .042/.044.
    if any(
        not 0.002 / 0.044 <= width / 0.044 <= 0.042 / 0.044
        for width in (previous, proposed)
    ):
        return "GRIPPER_WIDTH_BLOCKED"
    if (record["contact"] or record["grasp"]) and proposed > previous:
        return "OPEN_DURING_CONTACT_BLOCKED"
    if abs(proposed - previous) / 0.05 > 0.04 + 1e-10:
        return "GRIPPER_RATE_BLOCKED"
    return "SHADOW_MATCH_ONLY"
