import json
import tempfile
import unittest
from pathlib import Path

from scripts.pi0243_checkpoint_persistence import (
    PersistenceError,
    checkpoint_record,
    publish_checkpoint,
    verify_receipt,
)


class PI0243CheckpointPersistenceTests(unittest.TestCase):
    def _checkpoint(self, root: Path, step: int = 1000, max_steps: int = 5000) -> Path:
        root.mkdir(parents=True)
        (root / "config.json").write_text("{}\n", encoding="utf-8")
        (root / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
        for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
            (root / name).write_bytes(name.encode("utf-8"))
        (root / "trainer_state.json").write_text(json.dumps({"global_step": step, "max_steps": max_steps}) + "\n", encoding="utf-8")
        return root

    def test_publish_verify_and_reject_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = self._checkpoint(base / "source" / "checkpoint-1000")
            receipt = publish_checkpoint(
                source,
                durable_host=None,
                durable_root=base / "durable",
                expected_step=1000,
                expected_max_steps=5000,
                family="TEST",
                receipt_root=base / "receipts",
                job_id="test",
            )
            self.assertEqual(receipt["status"], "DURABLE_VERIFIED")
            receipt_path = base / "receipts" / "checkpoint-1000.durable_receipt.json"
            verify_receipt(base / "durable" / "training" / "checkpoint-1000", receipt_path, expected_step=1000, expected_max_steps=5000, family="TEST")
            (base / "durable" / "training" / "checkpoint-1000" / "optimizer.pt").write_bytes(b"truncated")
            with self.assertRaises(PersistenceError):
                verify_receipt(base / "durable" / "training" / "checkpoint-1000", receipt_path, expected_step=1000, expected_max_steps=5000, family="TEST")

    def test_wrong_step_and_max_steps_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._checkpoint(Path(temporary) / "checkpoint-1000")
            with self.assertRaises(PersistenceError):
                checkpoint_record(checkpoint, expected_step=2500, expected_max_steps=5000)
            with self.assertRaises(PersistenceError):
                checkpoint_record(checkpoint, expected_step=1000, expected_max_steps=10000)


if __name__ == "__main__":
    unittest.main()
