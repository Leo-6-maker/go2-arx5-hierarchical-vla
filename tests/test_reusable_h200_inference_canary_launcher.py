"""CPU regressions for the committed reusable H200 canary launcher."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "reusable_h200_inference_canary.py"
ADAPTER = ROOT / "scripts" / "stage45c_h200_eval_runtime_adapter.py"
BOOTSTRAP = ROOT / "scripts" / "stage45c_h200_eval_bootstrap.py"


class ReusableH200CanaryLauncherTests(unittest.TestCase):
    def _run(self, foreign_cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = ""
        environment["PYTHONNOUSERSITE"] = "1"
        return subprocess.run(
            [sys.executable, str(LAUNCHER), *arguments],
            cwd=foreign_cwd,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_absolute_foreign_cwd_empty_pythonpath_self_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            foreign_cwd = Path(temporary) / "unrelated-cwd"
            foreign_cwd.mkdir()
            result = self._run(
                foreign_cwd,
                "--repo-root",
                str(ROOT),
                "--self-check",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["repository_root"], str(ROOT.resolve()))
        self.assertEqual(payload["src_root"], str((ROOT / "src").resolve()))
        self.assertEqual(payload["import_roots"], {"repository": True, "src": True})
        self.assertEqual(payload["scripts_dependency"], "scripts.stage45c_h200_eval_runtime_adapter")
        self.assertEqual(payload["scripts_dependency_origin"], str(ADAPTER.resolve()))
        self.assertTrue(payload["scripts_dependency_imported"])
        self.assertEqual(payload["admission_boundary"], "REACHED_FAIL_CLOSED")
        self.assertEqual(payload["model_modules_present"], [])
        self.assertFalse(payload["torch_imported"])
        self.assertFalse(payload["gr00t_imported"])
        self.assertFalse(payload["transformers_imported"])
        self.assertFalse(payload["cuda_entered"])
        self.assertFalse(payload["model_construction"])
        self.assertFalse(payload["model_forward"])
        self.assertFalse(payload["training"])
        self.assertEqual(payload["cwd"], str(foreign_cwd))
        self.assertEqual(payload["pythonpath"], "")

    def test_wrong_explicit_repo_root_fails_closed_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            foreign_cwd = Path(temporary) / "unrelated-cwd"
            wrong_root = Path(temporary) / "wrong-root"
            foreign_cwd.mkdir()
            wrong_root.mkdir()
            result = self._run(
                foreign_cwd,
                "--repo-root",
                str(wrong_root),
                "--self-check",
            )

        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        payload = json.loads(result.stderr)
        self.assertEqual(payload["status"], "BLOCKED")
        self.assertIn("repo root does not match launcher checkout", payload["error"])
        self.assertEqual(payload["model_modules_present"], [])
        self.assertNotIn("scripts_dependency_imported", payload)

    def test_path_bootstrap_precedes_namespace_dependency_and_model_imports_are_deferred(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        root_offset = source.index("REPO_ROOT = Path(__file__).resolve().parents[1]")
        dependency_offset = source.index('"scripts.stage45c_h200_eval_runtime_adapter"')
        self.assertLess(root_offset, dependency_offset)
        self.assertIn("SRC_ROOT = REPO_ROOT / \"src\"", source)
        self.assertIn("if _import_root_text not in sys.path:", source)

        tree = ast.parse(source)
        top_level_imports: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level_imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level_imports.append(node.module)
        forbidden = ("torch", "gr00t", "transformers", "safetensors")
        self.assertEqual(
            [name for name in top_level_imports if name == forbidden or name.startswith(tuple(f + "." for f in forbidden))],
            [],
        )
        run_source = source[source.index("def run_canary") :]
        admission_offset = run_source.index("validate_h200_inference_admission(")
        model_entry_offset = run_source.index("observations = extract_observations")
        torch_import_offset = run_source.index("import torch")
        self.assertLess(admission_offset, model_entry_offset)
        self.assertLess(admission_offset, torch_import_offset)

    def test_reusable_support_entrypoints_keep_both_import_roots(self) -> None:
        bootstrap_source = BOOTSTRAP.read_text(encoding="utf-8")
        bootstrap_root_offset = bootstrap_source.index("REPO_ROOT = Path(__file__).resolve().parents[1]")
        bootstrap_runpy_offset = bootstrap_source.index("runpy.run_path")
        self.assertLess(bootstrap_root_offset, bootstrap_runpy_offset)
        self.assertIn("SRC_ROOT = REPO_ROOT / \"src\"", bootstrap_source)
        self.assertIn("for _import_root in (REPO_ROOT, SRC_ROOT):", bootstrap_source)

        adapter_source = ADAPTER.read_text(encoding="utf-8")
        adapter_root_offset = adapter_source.index("REPO = Path(__file__).resolve().parents[1]")
        adapter_namespace_offset = adapter_source.index("``scripts``")
        self.assertLess(adapter_root_offset, adapter_namespace_offset)
        self.assertIn("for _import_root in (REPO, SRC):", adapter_source)


if __name__ == "__main__":
    unittest.main()
