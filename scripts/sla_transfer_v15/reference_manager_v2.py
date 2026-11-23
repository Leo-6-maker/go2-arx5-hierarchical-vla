"""CPU reference manager: source time, execution time and installation clocks remain distinct."""

from dataclasses import dataclass

import numpy as np

from go2arx5_vla.trajectory.so3 import quat_slerp, quat_to_matrix


@dataclass(frozen=True)
class TaskSpaceIntentV2:
    identity: str
    source: np.ndarray
    origin: np.ndarray
    tool: np.ndarray
    execution_times: np.ndarray

    def __post_init__(self):
        raw = np.array(self.source, dtype=float, copy=True)
        times = np.array(self.execution_times, dtype=float, copy=True)
        if (
            raw.shape != (50, 16)
            or times.shape != (50,)
            or not np.isfinite(raw).all()
            or not np.isfinite(times).all()
            or times[0] != 0
            or np.any(np.diff(times) <= 0)
            or not self.identity
        ):
            raise ValueError("source shape or execution clock")
        if np.any(abs(np.linalg.norm(raw[:, 11:15], axis=1) - 1) > 1e-4) or np.any(
            (raw[:, 15] < 0) | (raw[:, 15] > 1)
        ):
            raise ValueError("invalid right arm quaternion or gripper")
        for name in ("origin", "tool"):
            v = np.array(getattr(self, name), dtype=float, copy=True)
            if (
                v.shape != (4, 4)
                or not np.isfinite(v).all()
                or not np.allclose(v[3], [0, 0, 0, 1], atol=1e-9)
                or not np.allclose(v[:3, :3].T @ v[:3, :3], np.eye(3), atol=1e-8)
                or abs(np.linalg.det(v[:3, :3]) - 1) > 1e-8
            ):
                raise ValueError("invalid calibrated frame")
            v.setflags(write=False)
            object.__setattr__(self, name, v)
        for name, v in (("source", raw), ("execution_times", times)):
            v.setflags(write=False)
            object.__setattr__(self, name, v)

    def at(self, elapsed):
        t = np.asarray(elapsed, dtype=float)
        t = np.atleast_1d(t)
        if not np.isfinite(t).all() or np.any(t < -1e-9):
            raise ValueError("invalid elapsed time")
        # Explicit terminal hold in execution time. Never extend the source 30 fps clock.
        source_phase = np.interp(
            t, self.execution_times, np.arange(50, dtype=float) / 30
        )
        index = np.minimum(np.floor(source_phase * 30 + 1e-12).astype(int), 48)
        f = np.clip(source_phase * 30 - index, 0, 1)
        a = self.source[:, 8:]
        q0 = a[index][:, [4, 5, 6, 3]]
        q1 = a[index + 1][:, [4, 5, 6, 3]]
        local = np.repeat(np.eye(4)[None], len(t), axis=0)
        local[:, :3, 3] = (1 - f[:, None]) * a[index, :3] + f[:, None] * a[
            index + 1, :3
        ]
        local[:, :3, :3] = quat_to_matrix(quat_slerp(q0, q1, f))
        return (
            self.origin @ local @ self.tool,
            (1 - f) * a[index, 7] + f * a[index + 1, 7],
            source_phase,
        )


class ReferenceManagerV2:
    def __init__(self, intent, epoch, start, *, native_ledger_hashes=()):
        if not epoch or not np.isfinite(start):
            raise ValueError("invalid start")
        self.intent = intent
        self.epoch = epoch
        self.start = start
        self.revision = 0
        self.last = None
        self.revision_clock = start
        self.sequence = -1
        self.pending = None
        self.installed = None
        self.consumed = None
        self.terminal = False
        self.native_ledger_hashes = frozenset(native_ledger_hashes)

    def abort(self):
        self.terminal = True
        self.pending = self.installed = None

    def revise(self, new, now, epoch):
        if (
            self.terminal
            or epoch != self.epoch
            or new.identity != self.intent.identity
            or not np.isfinite(now)
            or now < self.revision_clock
            or (self.last is not None and now < self.last)
        ):
            self.abort()
            raise ValueError("revision identity or clock")
        end = now - self.start
        # Identical already-entered source segments keep their original time law, including the current boundary.
        k = min(
            50, int(np.searchsorted(self.intent.execution_times, end, side="right")) + 1
        )
        if (
            not np.array_equal(new.source[:k], self.intent.source[:k])
            or not np.array_equal(
                new.execution_times[:k], self.intent.execution_times[:k]
            )
            or not np.array_equal(new.origin, self.intent.origin)
            or not np.array_equal(new.tool, self.intent.tool)
        ):
            raise ValueError("overwriting consumed or current segment")
        self.intent = new
        self.revision += 1
        self.revision_clock = now

    def window(self, now):
        if (
            self.terminal
            or self.pending is not None
            or not np.isfinite(now)
            or now < self.revision_clock
            or (
                self.last is not None
                and (now <= self.last or now - self.last >= 0.4 - 1e-10)
            )
        ):
            self.abort()
            raise ValueError("clock, TTL, or unacknowledged installation")
        self.sequence += 1
        self.last = now
        self.pending = (self.epoch, self.sequence, self.revision)
        return self.pending, self.intent.at(now - self.start + np.arange(41) * 0.05)

    def acknowledge(self, key, wire_hash, accept, installed):
        import hashlib
        import json

        raw = accept.get("raw", "")
        if (
            self.terminal
            or self.pending != key
            or accept.get("kind") != "accept"
            or accept.get("ref_id") != wire_hash
            or hashlib.sha256(raw.encode()).hexdigest() != wire_hash
            or installed.get("kind") != "installed"
            or installed.get("ref_id") != wire_hash
        ):
            self.abort()
            raise ValueError("unmatched receiver receipt")
        meta = json.loads(raw)["_pi0294"]
        if (
            meta["episode"],
            meta["sequence"],
            meta["source_mpc_s"],
            meta["ttl_s"],
            meta["operation"],
        ) != (key[0], key[1], self.last, 0.4, "install"):
            self.abort()
            raise ValueError("wrong installed sequence or source clock")
        self.installed = (key, wire_hash)
        self.pending = None

    def consume(self, key, wire_hash, receipt):
        from pathlib import Path

        from identity import read, sha

        path = receipt.get("native_ledger_path")
        seal = receipt.get("sealed_ledger_sha256")
        if not path or seal not in self.native_ledger_hashes or sha(Path(path)) != seal:
            raise ValueError("missing sealed native consumption ledger")
        evidence = read(Path(path))
        if (
            evidence.get("kind") != "native_applied_reference"
            or evidence.get("epoch") != key[0]
            or evidence.get("sequence") != key[1]
            or evidence.get("wire_sha256") != wire_hash
            or not evidence.get("applied_policy_id")
        ):
            raise ValueError("no applied reference identity")
        source_s = evidence["simulation_time_s"]
        receipt_native_step = evidence["native_step"]
        if (
            self.terminal
            or self.installed != (key, wire_hash)
            or not np.isfinite(source_s)
            or not self.last <= source_s < self.last + 0.4
            or type(receipt_native_step) is not int
            or receipt_native_step <= 0
        ):
            raise ValueError("unmatched native consumption")
        if self.consumed is not None and receipt_native_step <= self.consumed[-1]:
            raise ValueError("replayed consumption")
        self.consumed = (key, wire_hash, source_s, receipt_native_step)
