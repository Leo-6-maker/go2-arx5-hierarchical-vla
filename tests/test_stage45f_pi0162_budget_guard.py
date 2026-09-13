"""CPU-only contract tests for the PI0162 guarded continuation."""

from __future__ import annotations

import multiprocessing
import tempfile
import unittest
from pathlib import Path

from stage45f_pi0162_budget_guard import BudgetLedger, CampaignBusy, BudgetExceeded


CAPS = {
    "simulator_constructor": 1,
    "explicit_control_tick": 1,
    "reference_update": 1,
    "groot_constructor": 1,
    "conditioning": 1,
    "action_head": 1,
    "model_side_call": 1,
}


def _claim_from_second_process(path: str, queue: multiprocessing.queues.Queue) -> None:
    try:
        BudgetLedger(path, "TEST_CONCURRENT", budgets=CAPS, output_root=Path(path).parent / "other")
    except CampaignBusy:
        queue.put("BUSY")
    except Exception as exc:  # pragma: no cover - a useful diagnostic if it fails
        queue.put(type(exc).__name__)
    else:
        queue.put("CLAIMED")


class PI0162BudgetGuardTests(unittest.TestCase):
    def test_concurrent_process_cannot_claim_active_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger_path = str(Path(directory) / "fixed" / "ledger.json")
            owner = BudgetLedger(ledger_path, "TEST_CONCURRENT", budgets=CAPS, output_root=Path(directory) / "owner")
            try:
                context = multiprocessing.get_context("fork")
                queue = context.Queue()
                child = context.Process(target=_claim_from_second_process, args=(ledger_path, queue))
                child.start()
                child.join(10)
                self.assertEqual(child.exitcode, 0)
                self.assertEqual(queue.get(timeout=2), "BUSY")
            finally:
                owner.close("TEST_CLOSE")

    def test_constructor_failure_and_restart_consume_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            owner = BudgetLedger(path, "TEST_RESTART", budgets=CAPS, output_root=Path(directory) / "a")
            owner.reserve("simulator_constructor", phase="synthetic-failure")
            owner.close("TEST_HANDOFF")
            restarted = BudgetLedger(path, "TEST_RESTART", budgets=CAPS, output_root=Path(directory) / "b")
            try:
                self.assertEqual(restarted.snapshot()["used"]["simulator_constructor"], 1)
                with self.assertRaises(BudgetExceeded):
                    restarted.reserve("simulator_constructor", phase="must-reject-before-call")
            finally:
                restarted.close("TEST_CLOSE")

    def test_raw_exists_does_not_reserve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            guard = BudgetLedger(path, "TEST_RAW", budgets=CAPS, output_root=Path(directory) / "out")
            try:
                raw = Path(directory) / "out" / "raw.npz"
                raw.parent.mkdir()
                raw.write_bytes(b"sealed")
                event = BudgetLedger.reserve_if_missing(guard, raw, "model_side_call")
                self.assertFalse(event["reserved"])
                self.assertEqual(guard.snapshot()["used"]["model_side_call"], 0)
            finally:
                guard.close("TEST_CLOSE")


if __name__ == "__main__":
    unittest.main()
