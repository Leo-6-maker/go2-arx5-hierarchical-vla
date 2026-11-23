"""Interlock for a paused, noncontact simulation owner. No controller or model I/O."""

import math
from threading import RLock


def contact_at(packet, geometries):
    """Use the native contact list, including weak or zero-force non-support pairs."""
    tick = packet["step"]
    if (
        type(tick) is not int
        or tick < 1
        or not all(
            math.isfinite(packet[key]) and abs(packet[key] - expected) <= 1e-8
            for key, expected in (
                ("time_s", tick * 0.002),
                ("contact_time_s", (tick - 1) * 0.002),
            )
        )
    ):
        raise ValueError("invalid native contact clock")
    first = None
    for c in packet["contacts"]:
        a, b = c["geom0"], c["geom1"]
        if not (
            type(a) is type(b) is int
            and 0 <= a < len(geometries)
            and 0 <= b < len(geometries)
        ):
            raise ValueError("invalid native geometry identity")
        force = float(c["wrench"][0])
        if not math.isfinite(force):
            raise ValueError("invalid native contact force")
        ga, gb = geometries[a], geometries[b]
        names = {ga["name"], gb["name"]}
        support = ("floor" in names and bool(names & {"FL", "FR", "RL", "RR"})) or (
            "task_table" in names
            and bool(names & {"task_cube_geom", "task_blue_cube_geom"})
        )
        if not support and first is None:
            first = dict(
                ids=[a, b],
                geometries=[ga["name"], gb["name"]],
                bodies=[ga["body"], gb["body"]],
                normal_force_N=force,
                contact_time_s=float(packet["contact_time_s"]),
            )
    return first


class NoncontactOwner:
    """One epoch and one outstanding grant. All sends and contact updates share this owner."""

    def __init__(self, epoch, initial_tick, geometries, request_stop):
        if (
            not epoch
            or type(initial_tick) is not int
            or initial_tick < 1
            or not geometries
        ):
            raise ValueError("missing noncontact owner identity")
        self.epoch = epoch
        self.geometry = tuple(dict(g) for g in geometries)
        if any(g["id"] != i for i, g in enumerate(self.geometry)):
            raise ValueError("unbound native geometry table")
        self._lock = RLock()
        self._last = initial_tick - 1
        self._awaiting = initial_tick
        self._stop = request_stop
        self.terminal = None

    @classmethod
    def for_policy(cls, stream, initial_tick, geometries, cancel_reference):
        """Invalidate pending inference and revoke the reference on the same terminal transition."""
        epoch = stream.episode

        def stop(event):
            try:
                stream.stop(epoch)
            finally:
                cancel_reference(event)

        return cls(epoch, initial_tick, geometries, stop)

    def _seal(self, reason, pause, **evidence):
        if self.terminal is None:
            self.terminal = dict(reason=reason, pause_tick=pause, **evidence)
            # Latch before delivery. A failed cancel must never re-enable the owner.
            self._stop(dict(self.terminal))

    def _current(self, epoch):
        if epoch != self.epoch:
            raise ValueError("wrong noncontact epoch")
        if self.terminal is not None:
            raise ValueError("terminal noncontact epoch")

    def after_pause(self, epoch, tick, packets):
        """Check each completed 2 ms interval before allowing a new publication or grant."""
        with self._lock:
            self._current(epoch)
            if self._awaiting is None or tick != self._awaiting:
                self._seal("UNEXPECTED_PAUSE", tick)
                return
            hazard = None
            try:
                for p in packets:
                    if p["step"] != self._last + 1 or p["step"] > tick:
                        raise ValueError("contact stream gap or overlap")
                    value = contact_at(p, self.geometry)
                    if value is not None and hazard is None:
                        hazard = dict(observed_tick=p["step"], contact=value)
                    self._last = p["step"]
                if self._last != tick:
                    raise ValueError("contact stream incomplete at pause")
            except Exception as exc:
                self._seal(
                    "CONTACT_STREAM_INVALID",
                    tick,
                    detail=str(exc),
                    first_contact=hazard,
                )
                return
            self._awaiting = None
            if hazard is not None:
                self._seal("UNEXPECTED_CONTACT", tick, **hazard)

    def dispatch(self, epoch, tick, send):
        """Send a reference, gripper request or return only at a verified paused boundary."""
        with self._lock:
            self._current(epoch)
            if self._awaiting is not None or tick != self._last:
                raise ValueError("native boundary has not been verified")
            try:
                return send()
            except Exception:
                self._seal("DISPATCH_DELIVERY_UNKNOWN", tick)
                raise

    def grant(self, epoch, tick, end, send):
        with self._lock:
            self._current(epoch)
            if (
                self._awaiting is not None
                or tick != self._last
                or type(end) is not int
                or not tick < end <= tick + 25
            ):
                raise ValueError("invalid noncontact grant")
            # The send may succeed before its transport reports a failure.
            self._awaiting = end
            try:
                return send()
            except Exception:
                self._seal("GRANT_DELIVERY_UNKNOWN", tick, requested_end=end)
                raise

    def close(self, epoch, tick, reason):
        with self._lock:
            self._current(epoch)
            if self._awaiting is not None or tick != self._last:
                raise ValueError("native boundary has not been verified")
            self._seal(reason, tick)
