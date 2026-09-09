"""CPU/static tests for the Corelab Stage45C H200 runtime adapter."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45c_h200_eval_runtime_adapter as adapter  # noqa: E402


PROTOCOL = ROOT / "manifests/STAGE45C_TRAIN24_EVALUATION_PROTOCOL_FROZEN_V1.json"
MAPPING = ROOT / "manifests/STAGE45C_CHECKPOINT_RUNTIME_MAPPING_V1.json"
BINDING = ROOT / "manifests/STAGE45C_H200_EVAL_RUNTIME_BINDING_TEMPLATE_V1.json"
PBS = ROOT / "hpc/pbs/stage45c_train24_checkpoint_eval_h200.pbs"


class TestStage45CH200EvalRuntimeAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mapping_summary = adapter.audit_protocol_and_mapping(PROTOCOL, MAPPING)

    def test_frozen_protocol_and_mapping_audit_passes(self) -> None:
        self.assertEqual(self.mapping_summary["status"], "PASS")
        self.assertEqual(self.mapping_summary["checkpoint_order"], list(adapter.CHECKPOINT_ORDER))
        self.assertEqual(self.mapping_summary["windows_per_checkpoint"], 24)
        self.assertEqual(self.mapping_summary["planned_forwards_total"], 96)
        self.assertEqual(self.mapping_summary["historical_materialization"], "PENDING_VANDA_RECOVERY")
        self.assertFalse(self.mapping_summary["scientific_contract_changed"])

    def test_pathless_template_is_a_valid_late_binding(self) -> None:
        binding = json.loads(BINDING.read_text(encoding="utf-8"))
        result = adapter.validate_runtime_binding(binding, self.mapping_summary, require_paths=False)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(value is None for value in result["paths"].values()))
        self.assertEqual(binding["historical_comparator"]["runtime_materialization_status"], "PENDING_VANDA_RECOVERY")
        self.assertIsNone(binding["historical_comparator"]["materialization_path"])

    def test_runtime_binding_requires_all_paths_for_execution(self) -> None:
        binding = json.loads(BINDING.read_text(encoding="utf-8"))
        with self.assertRaises(adapter.AdapterError):
            adapter.validate_runtime_binding(binding, self.mapping_summary, require_paths=True)

    def test_mapping_tampering_fails_closed(self) -> None:
        mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
        mapping["checkpoint_order"][0] = "FULL24@2000"
        temporary = ROOT / "tests/.stage45c_mapping_tamper.json"
        try:
            temporary.write_text(json.dumps(mapping), encoding="utf-8")
            with self.assertRaises(adapter.AdapterError):
                adapter.audit_protocol_and_mapping(PROTOCOL, temporary)
        finally:
            temporary.unlink(missing_ok=True)

    def test_plan_is_zero_exposure(self) -> None:
        plan = adapter._plan(self.mapping_summary)
        boundary = plan["execution_boundary"]
        self.assertFalse(boundary["model_construction"])
        self.assertFalse(boundary["model_forward"])
        self.assertFalse(boundary["gpu"])
        self.assertFalse(boundary["training"])
        self.assertEqual(boundary["pbs_submissions_by_corelab"], 0)
        self.assertEqual(boundary["wbc_steps"], 0)
        self.assertEqual(plan["sampler"]["development_seed"], 4242)
        self.assertEqual(plan["sampler"]["timestep_ids"], [0, 250, 500, 750])
        self.assertEqual(plan["return_token"], "CORELAB_STAGE45C_H200_EVAL_RUNTIME_ADAPTER_READY")

    def test_static_guard_has_no_top_level_runtime_import(self) -> None:
        guard = adapter._static_guard()
        self.assertEqual(guard["status"], "PASS")
        self.assertEqual(guard["top_level_forbidden_imports"], [])
        self.assertEqual(guard["forbidden_call_hits"], [])

    def test_source_declares_raw_first_and_v2_bridge(self) -> None:
        source = (ROOT / "scripts/stage45c_h200_eval_runtime_adapter.py").read_text(encoding="utf-8")
        self.assertIn("np.save(raw_path", source)
        self.assertIn('"decoded": False', source)
        self.assertIn("_decode_v2", source)
        self.assertIn("_decode_base", source)
        self.assertIn("evaluate_raw_prediction", source)
        self.assertIn("same_noise_across_checkpoints", source)
        self.assertNotIn("model.get_action", source)
        self.assertNotIn("import gr00t", source)

    def test_pbs_envelope_calls_only_adapter(self) -> None:
        source = PBS.read_text(encoding="utf-8")
        self.assertIn("stage45c_h200_eval_runtime_adapter.py", source)
        self.assertNotIn("stage45c_run_train24_checkpoint_eval.py", source)
        self.assertNotIn("STAGE45C_STATS_PATH", source)
        self.assertNotIn("STAGE45C_RAW_PREDICTIONS", source)
        self.assertNotIn("legacy-output", source)

    def test_committed_runtime_files_do_not_contain_node_local_paths(self) -> None:
        for path in (MAPPING, BINDING):
            text = path.read_text(encoding="utf-8").lower()
            self.assertNotIn("/scratch/", text)
            self.assertNotIn("/var/cache/", text)
            self.assertNotIn("/home/", text)


if __name__ == "__main__":
    unittest.main()
