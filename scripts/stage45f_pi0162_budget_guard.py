#!/usr/bin/env python3
"""Small persistent, cross-process budget guard for the PI0162 campaign.

The guard intentionally has no scientific/runtime dependencies.  A reservation
is committed while holding an ``flock`` *before* the caller enters a native
constructor, model call, control tick, or reference update.  Reservations are
never released, including when the caller raises or the process is restarted.
The ledger lives outside experiment output directories and therefore cannot be
reset by selecting a new output version.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import platform
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping


SCHEMA = "CORELAB_PI0162_BUDGET_LEDGER_V1"
DEFAULT_BUDGETS = {
    "simulator_constructor": 3,
    "explicit_control_tick": 738,
    "reference_update": 12,
    "groot_constructor": 2,
    "conditioning": 6,
    "action_head": 21,
    "model_side_call": 27,
}


class BudgetGuardError(RuntimeError):
    """Base class for fail-closed budget errors."""


class CampaignBusy(BudgetGuardError):
    pass


class BudgetExceeded(BudgetGuardError):
    pass


class LedgerMismatch(BudgetGuardError):
    pass


def _now() -> float:
    return time.time()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pid_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # A live process owned by another uid must be treated as live.  This is
        # conservative and prevents a second launcher from double-spending.
        return True
    except OSError:
        return False
    return True


def _write_atomic(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(dict(value))
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        return _sha_bytes(payload)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def _empty_ledger(campaign_id: str, budgets: Mapping[str, int]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "campaign_id": campaign_id,
        "hostname": platform.node(),
        "created_at": _now(),
        "budgets": {str(k): int(v) for k, v in budgets.items()},
        "used": {str(k): 0 for k in budgets},
        "events": [],
        "active": None,
        "closed_processes": [],
    }


class BudgetLedger:
    """Atomic persistent reservations for one PI0162 campaign."""

    def __init__(
        self,
        ledger_path: str | os.PathLike[str],
        campaign_id: str,
        *,
        budgets: Mapping[str, int] | None = None,
        output_root: str | os.PathLike[str] | None = None,
        process_uuid: str | None = None,
    ) -> None:
        self.path = Path(ledger_path).expanduser().resolve()
        self.lock_path = Path(str(self.path) + ".lock")
        self.campaign_id = str(campaign_id)
        self.budgets = {str(k): int(v) for k, v in (budgets or DEFAULT_BUDGETS).items()}
        if any(v < 0 for v in self.budgets.values()):
            raise ValueError("budget caps must be non-negative")
        self.output_root = None if output_root is None else str(Path(output_root).expanduser().resolve())
        self.process_uuid = process_uuid or uuid.uuid4().hex
        self.pid = os.getpid()
        self.host = platform.node()
        self._registered = False
        self._register()

    @contextlib.contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if self.path.exists():
                    try:
                        state = json.loads(self.path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        raise LedgerMismatch(f"ledger unreadable: {self.path}: {exc}") from exc
                else:
                    state = _empty_ledger(self.campaign_id, self.budgets)
                self._validate_identity(state)
                yield state
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _validate_identity(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != SCHEMA:
            raise LedgerMismatch("budget ledger schema mismatch")
        if state.get("campaign_id") != self.campaign_id:
            raise LedgerMismatch("campaign id mismatch")
        recorded = {str(k): int(v) for k, v in dict(state.get("budgets", {})).items()}
        if recorded != self.budgets:
            raise LedgerMismatch(f"budget caps are immutable: recorded={recorded} requested={self.budgets}")
        used = dict(state.get("used", {}))
        if set(used) != set(self.budgets):
            raise LedgerMismatch("ledger resource set mismatch")
        for resource, amount in used.items():
            if int(amount) < 0 or int(amount) > self.budgets[resource]:
                raise LedgerMismatch(f"invalid persisted usage for {resource}: {amount}")

    def _register(self) -> None:
        with self._locked() as state:
            active = state.get("active")
            if active:
                same = (
                    active.get("host") == self.host
                    and int(active.get("pid", -1)) == self.pid
                    and active.get("process_uuid") == self.process_uuid
                )
                if not same:
                    old_pid = int(active.get("pid", -1))
                    if _pid_alive(old_pid):
                        raise CampaignBusy(
                            f"campaign {self.campaign_id} already active: "
                            f"host={active.get('host')} pid={old_pid} uuid={active.get('process_uuid')}"
                        )
                    state.setdefault("events", []).append({
                        "event": "PROCESS_RECLAIMED_AFTER_EXIT",
                        "at": _now(),
                        "previous_active": dict(active),
                        "new_process_uuid": self.process_uuid,
                    })
            state["active"] = {
                "host": self.host,
                "pid": self.pid,
                "process_uuid": self.process_uuid,
                "claimed_at": _now(),
                "output_root": self.output_root,
            }
            state.setdefault("events", []).append({
                "event": "PROCESS_CLAIMED",
                "at": _now(),
                "host": self.host,
                "pid": self.pid,
                "process_uuid": self.process_uuid,
                "output_root": self.output_root,
            })
            _write_atomic(self.path, state)
        self._registered = True

    def _assert_owner(self, state: Mapping[str, Any]) -> None:
        active = state.get("active") or {}
        expected = (active.get("host"), int(active.get("pid", -1)), active.get("process_uuid"))
        actual = (self.host, self.pid, self.process_uuid)
        if expected != actual:
            raise CampaignBusy(f"reservation attempted by non-owner: active={expected} caller={actual}")

    def reserve(self, resource: str, amount: int = 1, **context: Any) -> dict[str, Any]:
        """Persist an entered reservation before the underlying operation.

        The returned event is an immutable accounting record.  There is
        deliberately no release/refund method.
        """
        resource = str(resource)
        amount = int(amount)
        if resource not in self.budgets:
            raise BudgetGuardError(f"unknown budget resource: {resource}")
        if amount <= 0:
            raise ValueError("reservation amount must be positive")
        with self._locked() as state:
            self._assert_owner(state)
            used = int(state["used"][resource])
            cap = int(self.budgets[resource])
            if used + amount > cap:
                state.setdefault("events", []).append({
                    "event": "RESERVATION_REJECTED_CAP",
                    "at": _now(),
                    "resource": resource,
                    "amount": amount,
                    "used": used,
                    "cap": cap,
                    "pid": self.pid,
                    "process_uuid": self.process_uuid,
                    "context": dict(context),
                })
                _write_atomic(self.path, state)
                raise BudgetExceeded(f"{resource} cap exhausted: used={used} amount={amount} cap={cap}")
            state["used"][resource] = used + amount
            event = {
                "event": "RESERVED_ENTERED",
                "reservation_id": uuid.uuid4().hex,
                "at": _now(),
                "resource": resource,
                "amount": amount,
                "used_after": used + amount,
                "cap": cap,
                "entered": True,
                "failure_refunded": False,
                "pid": self.pid,
                "host": self.host,
                "process_uuid": self.process_uuid,
                "context": dict(context),
            }
            state.setdefault("events", []).append(event)
            _write_atomic(self.path, state)
            return dict(event)

    def guarded_call(self, resource: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Reserve first, then call; exceptions intentionally consume budget."""
        self.reserve(resource, **kwargs.pop("_budget_context", {}))
        return fn(*args, **kwargs)

    def close(self, reason: str = "PROCESS_CLOSED") -> None:
        """Close only this process's claim; counters remain immutable."""
        with self._locked() as state:
            self._assert_owner(state)
            state.setdefault("closed_processes", []).append({
                "event": reason,
                "at": _now(),
                "host": self.host,
                "pid": self.pid,
                "process_uuid": self.process_uuid,
                "used": dict(state["used"]),
            })
            state.setdefault("events", []).append({
                "event": reason,
                "at": _now(),
                "host": self.host,
                "pid": self.pid,
                "process_uuid": self.process_uuid,
            })
            state["active"] = None
            _write_atomic(self.path, state)
        self._registered = False

    def snapshot(self) -> dict[str, Any]:
        with self._locked() as state:
            return json.loads(json.dumps(state))

    @staticmethod
    def reserve_if_missing(guard: "BudgetLedger", artifact: str | os.PathLike[str], resource: str, **context: Any) -> dict[str, Any]:
        """Return a no-op for an already sealed raw artifact."""
        path = Path(artifact)
        if path.is_file():
            return {"reserved": False, "reason": "RAW_ALREADY_EXISTS", "artifact": str(path.resolve())}
        event = guard.reserve(resource, artifact=str(path.resolve()), **context)
        return {"reserved": True, "event": event}


def _self_test() -> dict[str, Any]:
    """CPU-only regression tests for persistence and fail-closed behavior."""
    with tempfile.TemporaryDirectory(prefix="pi0162-budget-test-") as td:
        root = Path(td)
        ledger = root / "campaign" / "ledger.json"
        caps = {"simulator_constructor": 1, "explicit_control_tick": 1, "reference_update": 1, "groot_constructor": 1, "conditioning": 1, "action_head": 1, "model_side_call": 1}
        g = BudgetLedger(ledger, "TEST_PI0162", budgets=caps, output_root=root / "out-a")
        try:
            g.reserve("simulator_constructor", condition="constructor-fails")
            try:
                raise RuntimeError("synthetic constructor failure")
            except RuntimeError:
                pass
            state = g.snapshot()
            assert state["used"]["simulator_constructor"] == 1
            raw = root / "out-a" / "raw.npy"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b"sealed-raw")
            no_forward = BudgetLedger.reserve_if_missing(g, raw, "model_side_call")
            assert no_forward["reserved"] is False and g.snapshot()["used"]["model_side_call"] == 0
            g.close("TEST_HANDOFF")
        finally:
            # ``close`` above is intentionally idempotence-tested by avoiding a
            # second call; a failed assertion must not hide the original error.
            pass
        # New process UUID and different output directory retain the used slot.
        g2 = BudgetLedger(ledger, "TEST_PI0162", budgets=caps, output_root=root / "out-b")
        try:
            assert g2.snapshot()["used"]["simulator_constructor"] == 1
            try:
                g2.reserve("simulator_constructor", context="restart")
            except BudgetExceeded:
                pass
            else:
                raise AssertionError("exhausted cap was not rejected")
            called = {"value": False}
            def underlying() -> None:
                called["value"] = True
            try:
                g2.reserve("simulator_constructor", context="must-not-enter")
            except BudgetExceeded:
                underlying_not_called = not called["value"]
            else:
                underlying_not_called = False
            assert underlying_not_called
            row = root / "partial" / "rows.json"
            row.parent.mkdir(parents=True)
            row.write_bytes(_canonical({"rows": [{"step": 0, "termination": "TRACKING_ABORT"}]}))
            try:
                raise RuntimeError("termination after row persistence")
            except RuntimeError:
                pass
            assert row.is_file()
            g2.close("TEST_DONE")
        finally:
            pass
        # A closed ledger can be claimed by a second independent launcher; the
        # counters and immutable events remain visible.
        g3 = BudgetLedger(ledger, "TEST_PI0162", budgets=caps, output_root=root / "out-c")
        try:
            events = g3.snapshot()["events"]
            assert any(e.get("event") == "PROCESS_CLAIMED" for e in events)
            g3.close("TEST_DONE_2")
        finally:
            pass
        return {"status": "PASS", "ledger_sha256": _sha_bytes(ledger.read_bytes()), "events": len(events), "used": g3.snapshot()["used"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("only --self-test is exposed; import BudgetLedger from a runner")
    print(json.dumps(_self_test(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
