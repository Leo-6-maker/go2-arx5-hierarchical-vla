"""Offline check of a closed gripper session. Capture identity is supplied out of band."""

import hashlib
import json
import math
from scripts.sla_official_pt_transfer.gripper_trace import GripperTrace


def verify_session(streams, seals, *, episode, owner, epoch, lease, receiver_sha256):
    roles = {"authority", "issuer", "receiver", "state", "stop"}
    if (
        set(streams) != roles
        or set(seals) != roles
        or len(set(seals.values())) != len(roles)
    ):
        raise ValueError("five separately registered capture streams required")
    binding = dict(
        episode=episode,
        owner=owner,
        epoch=epoch,
        lease=lease,
        receiver_sha256=receiver_sha256,
    )
    if (
        any(not isinstance(v, str) or not v.strip() for v in binding.values())
        or len(receiver_sha256) != 64
    ):
        raise ValueError("missing trusted session identity")
    logs = {}
    for role, raw in streams.items():
        if hashlib.sha256(raw).hexdigest() != seals[role]:
            raise ValueError("capture changed")
        record = json.loads(raw)
        if record.get("role") != role or any(
            record.get(k) != v for k, v in binding.items()
        ):
            raise ValueError("stream bound to different session")
        logs[role] = record["rows"]
    state, received = logs["state"], logs["receiver"]
    end = len(state) - 1
    if end <= 0 or logs["authority"] != [
        dict(event="exclusive_grant", tick=0),
        dict(event="release", tick=end),
    ]:
        raise ValueError("missing exclusive grant or overlapping owner")
    if logs["stop"] != [
        dict(event="stop", tick=end),
        dict(event="journal_closed", last_integrated_tick=end, after_stop_ticks=0),
    ]:
        raise ValueError("stop or closed native tick journal mismatch")
    for role in ("issuer", "receiver", "state"):
        if any(
            row.get("epoch") != epoch or row.get("lease") != lease for row in logs[role]
        ):
            raise ValueError("mixed native epoch or lease")
    old_roles = ("issuer", "receiver", "state")
    old = GripperTrace(episode, owner, {r: seals[r] for r in old_roles})
    verified = old.verify({r: streams[r] for r in old_roles}, reset_tick=0)
    if (
        received[0].get("last_consumed_sequence") != -1
        or type(received[0].get("last_consumed_sequence")) != int
    ):
        raise ValueError("dirty native reset")
    last = None
    for tick, row in enumerate(state):
        joints, targets = (
            row.get("actual_finger_positions_m"),
            row.get("applied_finger_targets_m"),
        )
        for pair in (joints, targets):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(
                    type(v) not in (float, int) or not math.isfinite(v) for v in pair
                )
            ):
                raise ValueError("incomplete bilateral measurement")
        if tick == 0 and targets != received[0]["initial_finger_targets_m"]:
            raise ValueError("independent target reset mismatch")
        if last is not None:
            target = logs["issuer"][row["applied_sequence"]]["width_m"]
            for prev, now in zip(last, targets):
                step = max(-0.04 * 0.002, min(0.04 * 0.002, target - prev))
                if not math.isclose(now, prev + step, rel_tol=0, abs_tol=1e-9):
                    raise ValueError("native target disagrees with applied packet")
        last = targets
    return dict(
        **verified,
        epoch_closed=True,
        post_stop_ticks=0,
        authority="offline evidence verification only",
    )
