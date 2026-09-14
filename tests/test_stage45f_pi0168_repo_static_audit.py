from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


REPO = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO / "scripts/stage45f_pi0168_repo_static_audit.py"
SPEC = importlib.util.spec_from_file_location("stage45f_pi0168_repo_static_audit", AUDIT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


class PI0168RepositoryStaticAuditTests(unittest.TestCase):
    def test_negative_control_detects_top_level_runtime_import(self) -> None:
        evidence = AUDIT.analyze_source("import numpy as np\n\ndef f():\n    return 1\n")
        self.assertEqual(["numpy"], [item["module"] for item in evidence["top_level_forbidden_imports"]])
        self.assertEqual([], evidence["function_local_forbidden_imports"])

    def test_negative_control_detects_function_local_runtime_import(self) -> None:
        evidence = AUDIT.analyze_source("def f():\n    import torch\n    return torch\n")
        self.assertEqual(["torch"], [item["module"] for item in evidence["function_local_forbidden_imports"]])
        self.assertEqual([], evidence["top_level_forbidden_imports"])

    def test_known_pi0167_pre_admission_contamination_is_reported(self) -> None:
        report = AUDIT.build_report(REPO)
        blocker = report["import_purity"]["known_pi0167_blocker"]
        self.assertTrue(blocker["present"])
        self.assertEqual("KNOWN_PI0167_BLOCKER", blocker["classification"])
        self.assertFalse(blocker["new_p0"])
        self.assertEqual(0, report["scope"]["validation_e_payloads_read"])
        self.assertEqual(0, report["scope"]["holdout_f_payloads_read"])

    def test_audit_process_declares_no_runtime_imports(self) -> None:
        report = AUDIT.build_report(REPO)
        self.assertTrue(report["import_purity"]["audit_process_project_runtime_imports_false"])
        self.assertTrue(report["quality"]["audit_process_uses_only_stdlib_and_git_metadata"])


if __name__ == "__main__":
    unittest.main()
