"""CPU-only regressions for the Vanda Stage45E N2 evaluation surface."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import stage45e_n2_train24_eval as evaluator  # noqa: E402
import stage45e_vanda_n2_eval_audit as audit  # noqa: E402


class Stage45EVandaN2AuditTests(unittest.TestCase):
    def test_static_audit_binds_first_science_commit_and_eval_surface(self) -> None:
        result = audit.build_static_audit()
        self.assertEqual(result["status"], "PASS_STATIC_NOT_EXECUTED")
        self.assertEqual(result["return_token"], audit.RETURN_TOKEN)
        self.assertEqual(result["manifest"]["active_scalar_count"], 190)
        self.assertEqual(result["manifest"]["window_count"], 24)
        self.assertEqual(result["science_commit_audit"]["n2_first_science_commit"], audit.N2_FIRST_SCIENCE_COMMIT)
        self.assertTrue(result["science_commit_audit"]["first_science_commit_only_xyz_stats_override"])
        self.assertTrue(result["evaluator_source_audit"]["durable_raw_writer"])
        self.assertEqual(result["execution_boundary"]["gpu"], 0)
        self.assertEqual(result["execution_boundary"]["pbs_submissions"], 0)
        self.assertFalse(result["execution_boundary"]["training"])

    def test_manifest_rejects_boundary_mutation(self) -> None:
        manifest = json.loads(audit.MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["locked_boundaries"]["model_forward"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(audit.N2EvalAuditError):
                audit._validate_manifest(path)

    def test_durable_raw_writer_is_atomic_and_non_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw" / "00.npy"
            expected = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
            evaluator._write_npy_durable(path, expected, np_module=np)
            np.testing.assert_array_equal(np.load(path, allow_pickle=False), expected)
            self.assertFalse(list(path.parent.glob("*.tmp-*")))
            with self.assertRaises(RuntimeError):
                evaluator._write_npy_durable(path, expected, np_module=np)


if __name__ == "__main__":
    unittest.main()
