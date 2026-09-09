#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_PATH = ROOT / "scripts" / "stage45c_h200_eval_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("stage45c_h200_eval_bootstrap_tested", BOOTSTRAP_PATH)
assert SPEC is not None and SPEC.loader is not None
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOOTSTRAP)


class Stage45CBootstrapTests(unittest.TestCase):
    def _fake_gr00t_modules(self, calls: list[tuple[str, dict]]) -> dict[str, types.ModuleType]:
        gr00t = types.ModuleType("gr00t")
        model = types.ModuleType("gr00t.model")
        n1d7 = types.ModuleType("gr00t.model.gr00t_n1d7")
        processing = types.ModuleType("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7")
        gr00t.__path__ = []
        model.__path__ = []
        n1d7.__path__ = []

        def original(model_name: str, loading_kwargs: dict):
            calls.append((model_name, dict(loading_kwargs)))
            return model_name, dict(loading_kwargs)

        processing.build_processor = original
        n1d7.processing_gr00t_n1d7 = processing
        return {
            "gr00t": gr00t,
            "gr00t.model": model,
            "gr00t.model.gr00t_n1d7": n1d7,
            "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7": processing,
        }

    def test_processor_binding_rewrites_only_canonical_cosmos_repo_id(self) -> None:
        calls: list[tuple[str, dict]] = []
        fake_modules = self._fake_gr00t_modules(calls)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(sys.modules, fake_modules, clear=False):
            snapshot = Path(temporary) / "snapshot"
            snapshot.mkdir()
            expected_snapshot = str(snapshot.resolve())
            record = BOOTSTRAP._bind_model_owned_processor(snapshot)
            processing = fake_modules["gr00t.model.gr00t_n1d7.processing_gr00t_n1d7"]
            kwargs = {"local_files_only": True, "cache_dir": "/sealed/cache"}
            result_cosmos = processing.build_processor(BOOTSTRAP.COSMOS_REPO_ID, kwargs)
            result_other = processing.build_processor("Qwen/other", kwargs)

        self.assertEqual(record["status"], "PASS")
        self.assertEqual(record["canonical_repo_id"], BOOTSTRAP.COSMOS_REPO_ID)
        self.assertEqual(record["model_config_repo_id_rewritten"], False)
        self.assertEqual(record["backbone_dispatch_changed"], False)
        self.assertEqual(record["loading_kwargs_changed"], False)
        self.assertEqual(result_cosmos[0], expected_snapshot)
        self.assertEqual(result_other[0], "Qwen/other")
        self.assertEqual(calls[0][1], kwargs)
        self.assertEqual(calls[1][1], kwargs)

    def test_rebinding_requires_exact_same_snapshot(self) -> None:
        calls: list[tuple[str, dict]] = []
        fake_modules = self._fake_gr00t_modules(calls)
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(sys.modules, fake_modules, clear=False):
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            BOOTSTRAP._bind_model_owned_processor(first)
            same = BOOTSTRAP._bind_model_owned_processor(first)
            self.assertEqual(same["binding_state"], "ALREADY_BOUND_SAME_SNAPSHOT")
            with self.assertRaises(RuntimeError):
                BOOTSTRAP._bind_model_owned_processor(second)

    def test_runtime_binding_resolves_exact_pinned_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            groot = root / "groot"
            cosmos = root / "cosmos"
            snapshot = cosmos / "snapshots" / BOOTSTRAP.COSMOS_REVISION
            groot.mkdir()
            snapshot.mkdir(parents=True)
            refs = cosmos / "refs"
            refs.mkdir()
            (refs / "main").write_text(BOOTSTRAP.COSMOS_REVISION + "\n", encoding="utf-8")
            binding = root / "binding.json"
            binding.write_text(
                json.dumps({"paths": {"groot_source": str(groot), "cosmos_root": str(cosmos)}}),
                encoding="utf-8",
            )
            observed_groot, observed_snapshot = BOOTSTRAP._load_execution_paths(binding)
            self.assertEqual(observed_groot, groot.resolve())
            self.assertEqual(observed_snapshot, snapshot.resolve())

    def test_pbs_launches_bootstrap_not_adapter_directly(self) -> None:
        pbs = (ROOT / "hpc" / "pbs" / "stage45c_train24_checkpoint_eval_h200.pbs").read_text(encoding="utf-8")
        self.assertIn('"${STAGE45C_REPO}/scripts/stage45c_h200_eval_bootstrap.py"', pbs)
        self.assertNotIn('"${STAGE45C_REPO}/scripts/stage45c_h200_eval_runtime_adapter.py" \\\n  --execute', pbs)


if __name__ == "__main__":
    unittest.main()
