"""Offline qualification of a student-owned reset and subsequent noncontact gripper receipts."""

import hashlib
import json
import math
import struct


class GripperTrace:
    """Compare issuer, receiver, and measured state from separately sealed evidence streams.

    The independent capture manifest supplies the trusted stream hashes. A proposal cannot supply
    a receipt or add a trusted source. This class has no device or command interface.
    """

    def __init__(self, episode, student, seals):
        self.episode, self.student = episode, student
        if (
            set(seals) != {"issuer", "receiver", "state"}
            or len(set(seals.values())) != 3
        ):
            raise ValueError("independent stream identities required")
        self.seals = dict(seals)

    def verify(self, streams, *, reset_tick):
        if (
            set(streams) != set(self.seals)
            or type(reset_tick) is not int
            or reset_tick != 0
        ):
            raise ValueError("missing streams or reset clock")
        data = {}
        for role, raw in streams.items():
            if hashlib.sha256(raw).hexdigest() != self.seals[role]:
                raise ValueError("unregistered stream")
            log = json.loads(raw)
            if (log["role"], log["episode"]) != (role, self.episode):
                raise ValueError("stream identity mismatch")
            data[role] = log["rows"]
        issued, receiver, state = (data[k] for k in ("issuer", "receiver", "state"))
        if not issued or len(receiver) != 2 * len(issued) + 1 or not state:
            raise ValueError("incomplete issue/ACK/consumption/state sequence")
        reset = receiver[0]
        if (
            reset.get("event") != "reset"
            or type(reset.get("tick")) is not int
            or reset["tick"] != reset_tick
            or reset.get("owner") != self.student
            or "previous_owner" not in reset
            or reset["previous_owner"] is not None
            or reset.get("mode") != "WBC"
        ):
            raise ValueError("student reset unobserved")
        self.check_state(state[0], reset_tick)
        last_tick, previous_width, previous_issue_tick = reset_tick, None, None
        for i, command in enumerate(issued):
            ack, consumed = receiver[2 * i + 1 : 2 * i + 3]
            if any(
                type(x.get("sequence")) is not int
                or x["sequence"] != i
                or x.get("owner") != self.student
                for x in (command, ack, consumed)
            ):
                raise ValueError("wrong sequence or owner")
            if [command.get("event"), ack.get("event"), consumed.get("event")] != [
                "issue",
                "ack",
                "consumed",
            ]:
                raise ValueError("missing receiver events")
            t, expiry, accepted, end = (
                command["tick"],
                command["expiry_tick"],
                ack["tick"],
                consumed["tick"],
            )
            if (
                any(type(x) is not int for x in (t, expiry, accepted, end))
                or t != last_tick
                or not t <= accepted < end <= expiry <= t + 200
                or (i == 0 and accepted != t)
                or accepted - t > 12
            ):
                raise ValueError("stale or discontinuous receiver clock")
            width = command["width_m"]
            if (
                type(width) not in (float, int)
                or not math.isfinite(width)
                or not 0.002 <= width <= 0.042
                or ack.get("width_m") != width
                or consumed.get("width_m") != width
            ):
                raise ValueError("width missing or receiver mismatch")
            wire = command.get("wire_sha256")
            if (
                not isinstance(wire, str)
                or len(wire) != 64
                or any(c not in "0123456789abcdef" for c in wire)
                or ack.get("wire_sha256") != wire
                or consumed.get("wire_sha256") != wire
            ):
                raise ValueError("wrong gripper command identity")
            # Reconstruct the exact issued native packet to check issuer/receiver binding.
            fraction = command.get("native_fraction")
            if (
                type(fraction) not in (int, float)
                or not math.isfinite(fraction)
                or not math.isclose(0.044 * fraction, width, rel_tol=0, abs_tol=1e-12)
            ):
                raise ValueError("native fraction mismatch")
            packet = struct.pack(
                "<8sQddd",
                b"GRTGRIP1",
                i,
                command["source_s"],
                fraction,
                command["expiry_s"],
            )
            if (
                not math.isclose(
                    command["source_s"], 0.002 * t, abs_tol=1e-8, rel_tol=0
                )
                or not math.isclose(
                    command["expiry_s"], 0.002 * expiry, abs_tol=1e-8, rel_tol=0
                )
                or hashlib.sha256(packet).hexdigest() != wire
            ):
                raise ValueError("packet identity or native clock")
            if previous_width is None:
                # No prior student command at reset. Match the independently observed actuator
                # target before enabling motion, across both fingers, not measured finger separation.
                reset_targets = reset.get("initial_finger_targets_m")
                if (
                    not isinstance(reset_targets, list)
                    or len(reset_targets) != 2
                    or any(
                        type(v) not in (int, float)
                        or not math.isclose(width, v, abs_tol=1e-8, rel_tol=0)
                        for v in reset_targets
                    )
                ):
                    raise ValueError("unanchored reset target")
            elif (
                abs(width - previous_width)
                > 0.04 * (t - previous_issue_tick) * 0.002 + 1e-12
            ):
                raise ValueError("consumed command rate above 40 mm/s")
            if end >= len(state):
                raise ValueError("incomplete native state stream")
            for tick in range(t + 1, end + 1):
                self.check_state(state[tick], tick)
                seq = i if tick > accepted else i - 1
                expected_wire = issued[seq]["wire_sha256"]
                if (
                    type(state[tick].get("applied_sequence")) is not int
                    or state[tick]["applied_sequence"] != seq
                    or state[tick].get("applied_wire_sha256") != expected_wire
                ):
                    raise ValueError(
                        "actual native step lacks applied command identity"
                    )
            previous_width, previous_issue_tick, last_tick = width, t, end
        if last_tick != len(state) - 1:
            raise ValueError("unaccounted native state")
        return {
            "verified_intervals": len(issued),
            "scope": "offline noncontact trace only",
        }

    @staticmethod
    def check_state(sample, tick):
        if (
            type(sample.get("tick")) is not int
            or sample["tick"] != tick
            or sample.get("valid") is not True
            or any(
                sample.get(k) is not False
                for k in ("contact", "grasp", "prohibited_contact")
            )
            or type(sample.get("actual_width_m")) not in (float, int)
            or not math.isfinite(sample["actual_width_m"])
        ):
            raise ValueError("invalid or contacted boundary state")
