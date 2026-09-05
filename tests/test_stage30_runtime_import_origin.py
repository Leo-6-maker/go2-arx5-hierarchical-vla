from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/stage30_runtime_import_origin.py"
WRAPPER = REPO / "hpc/stage30_p0_import_origin.sh"
SPEC = importlib.util.spec_from_file_location("stage30_p0", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RuntimeImportOriginTests(unittest.TestCase):
    def test_required_module_set_is_complete(self) -> None:
        expected = {
            "gr00t",
            "gr00t.policy.policy",
            "gr00t.policy.gr00t_policy",
            "gr00t.model.gr00t_n1d7.gr00t_n1d7",
            "gr00t.configs.model.gr00t_n1d7",
            "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7",
            "gr00t.data.state_action.state_action_processor",
            "gr00t.data.utils",
            "gr00t.eval.open_loop_eval",
        }
        self.assertEqual(set(MODULE.CRITICAL_MODULES), expected)

    def test_classification_is_fail_closed(self) -> None:
        inside = [{"inside_pinned_source_tree": True, "byte_exact_to_pinned": True}]
        outside = [{"inside_pinned_source_tree": False, "byte_exact_to_pinned": True}]
        conflict = [{"inside_pinned_source_tree": True, "byte_exact_to_pinned": False}]
        self.assertEqual(MODULE.classify_import_records(inside), "PASS")
        self.assertEqual(MODULE.classify_import_records(outside), "PASS_WITH_PROVENANCE_NOTE")
        self.assertEqual(MODULE.classify_import_records(conflict), "MATERIAL_CONFLICT")

    def test_import_audit_has_no_model_execution_path(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        wrapper = WRAPPER.read_text(encoding="utf-8")
        for forbidden in ("AutoModel", ".from_pretrained(", ".get_action(", "torch.load", "qsub"):
            self.assertNotIn(forbidden, source)
            self.assertNotIn(forbidden, wrapper)
        self.assertNotIn("--nv", wrapper)
        self.assertIn("PYTHONPATH=\"${STAGE29_EXACT_PYTHONPATH}\"", wrapper)


if __name__ == "__main__":
    unittest.main()
