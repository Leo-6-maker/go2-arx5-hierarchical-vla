"""Science acceptance checks; an archived teacher receipt cannot transfer command ownership."""

import numpy as np


def handover(ledger):
    if not ledger["contact_valid"] or ledger["prohibited_contact"]:
        return "HANDOVER_BLOCKED_UNKNOWN_OR_ILLEGAL_CONTACT"
    if ledger["contact"] or ledger["grasp"]:
        return "HANDOVER_BLOCKED_CONTACT"
    if ledger["student_owner_transfer_receipt"] is None or not ledger["actual_takeover_flag"]:
        return "HANDOVER_BLOCKED_NO_TRANSFER_ACK"
    if ledger["query_native_s"] > ledger["expiry_native_s"]:
        return "HANDOVER_BLOCKED_EXPIRED"
    if ledger["seam_speed_m_s"] > ledger["threshold_m_s"] + 1e-10:
        return "SEPARATE_ACKNOWLEDGED_TRANSITION_REQUIRED"
    return "REQUIRES_VERIFIED_STUDENT_ISSUE"


def nontrivial(arm, canonical):
    p = arm["progress"]
    return bool(
        arm["native_gate"]["H40_G7_pass"]
        and arm["native_gate"]["accepted_join_pass"]
        and canonical.get("legal")
        and p["motion_pass"]
        and p["event_pass"]
    )


def once_new(output):
    output.mkdir(mode=0o700, exist_ok=False)


def physical_grip_kinematics(g, previous, dt=0.05):
    width = 0.002 + 0.040 * np.asarray(g)
    inner = np.diff(width) / dt
    if previous is None:
        return dict(
            width_m=width.tolist(),
            internal_velocity_m_s=inner.tolist(),
            acceleration_m_s2=(np.diff(inner) / dt).tolist(),
            entry=None,
        )
    v = np.diff(np.r_[previous, width]) / dt
    return dict(
        width_m=width.tolist(),
        internal_velocity_m_s=inner.tolist(),
        acceleration_m_s2=(np.diff(v) / dt).tolist(),
        entry=float(v[0]),
    )
