"""Task-independent world reference adapter for the existing paused, open-gripper receiver."""

from dataclasses import dataclass
from pathlib import Path
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation

BASE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(BASE / "scripts" / name) for name in (
    "sla_go2_v18", "sla_hierarchical_v16", "sla_transfer_feedback",
)]
from compiler import DT, SourceCurve, ease, normmax, vectors
from feedback import OPEN, boundary, pose, require
from known_dynamic_panel import KnownCurve
from reach_intent_replay import compile_path
from reach_student import Student
from standing_reach import ReachProposalRejected
from world_intent_clock import PrefixClock, world_offset


@dataclass(frozen=True)
class PhysicalIntent:
    episode: str
    tick: int
    observation_sha256: str
    provenance: str
    actual_tcp: np.ndarray
    curve: SourceCurve

    def verify(self, episode, tick, actual):
        require(self.provenance in {"RAW_SENSOR", "DEPLOYABLE_SEA", "KNOWN_PHYSICAL"},
                "ORACLE_OR_UNBOUND_PROVENANCE")
        require(
            isinstance(self.curve, SourceCurve) and bool(episode) and self.episode == episode
            and type(self.tick) is int and tick == self.tick
            and bool(re.fullmatch(r"[0-9a-f]{64}", self.observation_sha256))
            and np.allclose(pose(self.actual_tcp), pose(actual), atol=2e-7, rtol=0),
            "STALE_OR_CROSS_EPISODE_INTENT",
        )


def receiver_clock_limits(limits):
    """Bound the first and last 50 ms increments of every H40 publication."""
    from go2arx5_vla.world_model import fusion_trajectory_adapter as native

    require(DT == native.ATOM_DT_S, "FROZEN_RECEIVER_CLOCK_MISMATCH")
    # The receiver appends zero velocity at both edges before checking
    # acceleration. A moving publication edge needs |v| / DT <= a_max.
    return {
        **limits,
        "linear_velocity": min(
            limits["linear_velocity"], native.MAX_TRANSLATION_ACCELERATION_M_S2 * DT
        ),
        "angular_velocity": min(
            limits["angular_velocity"], native.MAX_ANGULAR_ACCELERATION_RAD_S2 * DT
        ),
    }


def grid_report(commands, previous, limits, dt=DT):
    prefix = np.concatenate((previous[None], commands)) if previous is not None else commands
    v, w = vectors(prefix, dt)
    observed = dict(
        linear_velocity=normmax(v), angular_velocity=normmax(w),
        linear_acceleration=normmax(np.diff(v, axis=0)/dt),
        angular_acceleration=normmax(np.diff(w, axis=0)/dt),
    )
    return dict(observed=observed, threshold=limits,
                passed=all(value <= limits[k]+1e-9 for k, value in observed.items()))


class ReferenceSegment:
    """Compile one measured-to-consumed seam. Preserve raw events in the receipt."""

    def __init__(self, curve, actual, consumed, previous, limits):
        self.curve = curve
        self.actual, self.consumed = pose(actual).copy(), pose(consumed).copy()
        self.previous = pose(previous).copy()
        self.limits = limits

        def bound(index):
            return world_offset(curve.at(index), self.actual, self.consumed)

        self.clock = PrefixClock(bound, limits)
        self.selected_source = bound(np.arange(self.clock.end_index+1))
        self.alignment = compile_path(self.selected_source, None, consumed, previous, limits)
        self.lead = self.alignment.lead
        self.brake = self.alignment.brake
        self.end = self.lead + self.clock.duration
        require(np.allclose(self.alignment.first, self.selected_source[0], rtol=0, atol=1e-10),
                "WORLD_PREFIX_START_MISMATCH")
        self.events = [
            dict(kind=kind, source_s=float(i/30), inside_selected_prefix=bool(i<=self.clock.end_index))
            for kind, i in curve.events
        ]

    def at(self, times):
        t = np.atleast_1d(np.asarray(times, dtype=float))
        require(t.ndim == 1 and np.isfinite(t).all() and min(t) >= -1e-9,
                "INVALID_SEA_REFERENCE_CLOCK")
        after = t >= self.lead-1e-9
        command = np.empty((len(t), 4, 4))
        if np.any(~after):
            command[~after] = self.alignment.at(t[~after])[0]
        if np.any(after):
            command[after] = self.clock.at(t[after]-self.lead)
        index = self.clock.end_index * ease((t-self.lead)/self.clock.duration)
        return command, self.curve.grip(index), index/30

    def preview(self, elapsed):
        times = elapsed + np.arange(41)*DT
        commands, _, source = self.at(times)
        region = np.select(
            (times < self.brake-1e-9, times < self.lead-1e-9, times < self.end-1e-9),
            ("BRAKE", "ALIGN", "SOURCE"), default="HOLD",
        )
        commands[-1] = commands[-2]
        source[-1] = source[-2]
        region[-1] = "TERMINAL_PAD"
        return commands, source, region

    def check(self):
        t = np.arange(round((self.end + 2.0)/DT)+1)*DT
        g = grid_report(self.at(t)[0], self.previous, self.limits)
        half = np.arange(round((self.end + DT)/.025)+1)*.025
        grid25 = grid_report(self.at(half)[0], None, self.limits, .025)
        native = np.arange(round((self.end + DT)/.002)+1)*.002
        fine = grid_report(self.at(native)[0], None, self.limits, .002)
        return dict(h40_grid=g, half_grid_25ms=grid25, smooth_curve_2ms_sample=fine)


def verified_seam(episode, receipt, sample, tick):
    measurement = dict(
        episode=episode,
        tick=tick,
        TCP=sample["tcp"],
        safe=sample["safety_valid"] and sample["contact_valid"],
        contact=sample["robot_object_contact"] or sample["native_grasp"],
    )
    return boundary(episode, tick, receipt["consumed"], receipt, measurement)


def admit_intent(intent, receipt, sample, tick, limits):
    """Use the original receipt, native clocks and fixed-open H40 gate."""
    intent.verify(receipt["episode"], tick, sample["tcp"])
    consumed, actual = verified_seam(intent.episode, receipt, sample, tick)
    segment = ReferenceSegment(intent.curve, actual, consumed, receipt["start_pose"], limits)
    checks = segment.check()
    require(all(r["passed"] for r in checks.values()), "SEA_REFERENCE_CLOCK_GATE")
    commands, _, _ = segment.preview(0.)
    KnownCurve.admit(commands, receipt, sample)
    return segment


def motion_projection(measured_start, command_start, measured, command):
    """Compare signed increments along the consumed translation and SO(3) axes."""
    actual_mm = 1000 * (measured[:3, 3] - measured_start[:3, 3])
    target_mm = 1000 * (command[:3, 3] - command_start[:3, 3])
    actual_deg = np.rad2deg(
        Rotation.from_matrix(measured[:3, :3] @ measured_start[:3, :3].T).as_rotvec()
    )
    target_deg = np.rad2deg(
        Rotation.from_matrix(command[:3, :3] @ command_start[:3, :3].T).as_rotvec()
    )

    def axis(actual, target, threshold):
        length = float(np.linalg.norm(target))
        signed = float(actual @ target / length) if length > 1e-12 else None
        return length, signed, length > threshold and signed >= threshold

    mm, signed_mm, translation = axis(actual_mm, target_mm, 1.0)
    deg, signed_deg, rotation = axis(actual_deg, target_deg, .25)
    return dict(
        installed_mm=mm, actual_projection_mm=signed_mm, translation=translation,
        installed_deg=deg, actual_projection_deg=signed_deg, rotation=rotation,
    )


class SEAStandingStudent(Student):
    """Reference producer for standing_reach.drive. The drive retains sole install authority."""

    def __init__(self, root, provider, limits, query_cap):
        super().__init__(root)
        require(type(query_cap) is int and 1 <= query_cap <= 12, "FINITE_QUERY_CAP_REQUIRED")
        self.provider, self.limits, self.query_cap = provider, limits, query_cap
        self.completed_motion = None
        self.episode = None

    def preview(self, adapter, tick, receipt, sample, monitor, anchor, direction):
        require(monitor.last == receipt["end"] == tick, "WRONG_PAUSED_OBSERVATION")
        require(receipt["episode"] == adapter.output.name
                and self.episode in (None, adapter.output.name), "CROSS_OWNER_INTENT")
        self.path = adapter.output.parent / "SEA_REFERENCE.jsonl"
        elapsed = None if self.reference_tick is None else (tick-self.reference_tick)*.002
        # A fresh query requires an actually consumed motion from the current
        # plan. No lead-in or enabled-but-unconsumed reference can refresh it.
        refresh = self.compiled is None or (
            self.motion is not None and self.motion["tick"] > self.reference_tick
            and elapsed >= self.compiled.end-1e-9 and self.calls < self.query_cap
        )
        if refresh:
            try:
                verified_seam(adapter.output.name, receipt, sample, tick)
                intent = self.provider(adapter, tick)
                require(intent.episode == adapter.output.name, "CROSS_OWNER_INTENT")
                segment = admit_intent(intent, receipt, sample, tick, self.limits)
            except (ValueError, RuntimeError) as exc:
                raise ReachProposalRejected(str(exc)) from exc
            if self.calls:
                if intent.observation_sha256 == self.last_origin:
                    raise ReachProposalRejected("UNCHANGED_POST_MOTION_OBSERVATION")
                self.post_motion = True
                self.completed_motion = self.motion
            self.calls += 1
            self.last_origin = intent.observation_sha256
            self.reference_tick = tick
            self.episode = intent.episode
            self.compiled = segment
            self.first_source = None
            self.last_source_tick = None
            self.motion = None
            self.log("RETURN_ADMITTED", tick=tick, input=intent.observation_sha256,
                     predicted_event_metadata=segment.events, gripper_fixed_open=True,
                     plan_lead=segment.lead, plan_end=segment.end)
            elapsed = 0.
        commands, source, tags = self.compiled.preview(elapsed)
        self.log("PREVIEW", tick=tick, source_s=source[:3].tolist(),
                 phase=tags[:3].tolist(), selected_MRT_reference=None)
        # KnownCurve.admit, scene sweep, actual wire and atom_receipt run in
        # the unchanged standing_reach.drive after this returns.
        return commands

    def consumed(self, tick, receipt, sample):
        if self.compiled is None:
            return
        elapsed = (tick - self.reference_tick) * .002
        times = elapsed + np.array([-DT, 0.])
        if times[0] < self.compiled.lead - 1e-9:
            return
        expected, _, phase = self.compiled.at(times)
        if phase[1] <= phase[0]:
            return
        consumed, actual = verified_seam(self.episode, receipt, sample, tick)
        require(
            self.last_origin is not None
            and np.allclose(expected[0], receipt["start_pose"], atol=2e-7, rtol=0)
            and np.allclose(expected[1], consumed, atol=2e-7, rtol=0),
            "SOURCE_PHASE_WITHOUT_EXACT_CONSUMPTION",
        )
        if self.first_source is None:
            self.first_source = (tick, actual.copy(), consumed.copy())
            return
        first, actual0, consumed0 = self.first_source
        values = motion_projection(actual0, consumed0, actual, consumed)
        if self.motion is None and (values["translation"] or values["rotation"]):
            self.motion = dict(start=first, tick=tick, consumed=receipt["consumed"], **values)
            self.log("SOURCE_MOTION", **self.motion)
        self.last_source_tick = tick

    def summary(self):
        result = super().summary()
        result["source_motion"] = self.motion or self.completed_motion
        result["gripper_profile"] = "FIXED_OPEN"
        return result
