"""Shadow handoff audit. Measurements never replace the last consumed command."""

import math


def assess(record):
    if record["prohibited_contact"] or not record["contact_valid"]:
        return "CONTACT_STATE_BLOCKED"
    if (
        not record["student_owned_reset"]
        and not record["student_owner_transfer_receipt"]
    ):
        return "HANDOVER_BLOCKED"
    # These archived flags alone cannot authenticate a student-issued command chain.
    # Native qualification must provide the receipt bound to this owner and episode.
    required = (
        "student_owner",
        "episode",
        "gripper_receipt",
        "consumed_at_tick",
        "previous_command_m",
    )
    if any(record.get(k) is None for k in required):
        return "CONSUMED_GRIPPER_IDENTITY_UNOBSERVED"
    receipt = record["gripper_receipt"]
    if (
        receipt.get("owner") != record["student_owner"]
        or receipt.get("episode") != record["episode"]
        or receipt.get("sequence") != record.get("delivered_sequence")
        or receipt.get("consumed_tick") != record["consumed_at_tick"]
        or receipt.get("consumed_width_m") != record["previous_command_m"]
    ):
        return "CONSUMED_GRIPPER_IDENTITY_UNOBSERVED"
    if any(
        not math.isfinite(record[k])
        for k in ("new_first_m", "previous_command_m", "measured_m")
    ):
        return "GRIPPER_NONFINITE"
    if (record["contact"] or record["grasp"]) and record["new_first_m"] > record[
        "previous_command_m"
    ]:
        return "OPEN_DURING_CONTACT_BLOCKED"
    if abs(record["new_first_m"] - record["previous_command_m"]) / 0.05 > 0.04 + 1e-10:
        return "GRIPPER_RATE_BLOCKED"
    return "SHADOW_MATCH_ONLY"
