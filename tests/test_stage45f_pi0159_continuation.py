from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import unittest


sys.path.insert(0, "scripts")
import stage45f_pi0159_validation_e_continuation as continuation


def _config(tmp_path: Path) -> dict[str, object]:
    return {
        "schema": "STAGE45F_PI0159_VALIDATION_E_CONTINUATION_CONFIG_V1",
        "pi_gate": continuation.PI_GATE_COMMENT,
        "authority_token": continuation.AUTHORITY_TOKEN,
        "pr_number": continuation.PR_NUMBER,
        "branch": continuation.EXPECTED_BRANCH,
        "max_allocations": 2,
        "prior_model_load_attempts": 1,
        "max_model_loads_total": 8,
        "max_new_model_loads": 7,
        "max_new_model_side_calls": 210,
        "expected_head": "c7d541dec6828d00f91ee4917c9cd8730b4945a6",
        "evaluation_code_commit": "522ea718e99ff8b1c4b2aa08bec680ed5adfcae5",
        "cosmos_revision": continuation.EXPECTED_COSMOS_REVISION,
        "cosmos_tree_sha256": continuation.EXPECTED_COSMOS_TREE_SHA256,
        "stats_sha256": continuation.EXPECTED_STATS_SHA256,
        "noise_seed": 4242,
        "timestep_ids": [0, 250, 500, 750],
        "t_values": [0.0, 0.25, 0.5, 0.75],
        "nominal_calls": continuation.EXPECTED_NOMINAL_CALLS,
        "holdout_f_read": False,
        "training": False,
        "output_root": str(tmp_path / "output"),
    }


def _cosmos_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    hf_home = tmp_path / "hf_home"
    hub = hf_home / "hub"
    cosmos = hub / continuation.EXPECTED_MODEL_ID
    (cosmos / "refs").mkdir(parents=True)
    (cosmos / "snapshots" / continuation.EXPECTED_COSMOS_REVISION).mkdir(parents=True)
    (cosmos / "refs" / "main").write_text(continuation.EXPECTED_COSMOS_REVISION + "\n", encoding="utf-8")
    (cosmos / "snapshots" / continuation.EXPECTED_COSMOS_REVISION / "config.json").write_text("{}\n", encoding="utf-8")
    observed, _ = continuation.cosmos_tree_digest(cosmos)
    config = {
        "cosmos_root": str(cosmos),
        "hf_home": str(hf_home),
        "cosmos_revision": continuation.EXPECTED_COSMOS_REVISION,
        "cosmos_tree_sha256": observed,
    }
    return hf_home, cosmos, config


class PI0159ContinuationTests(unittest.TestCase):
    def test_static_guard_is_cpu_only(self) -> None:
        guard = continuation.static_guard()
        self.assertEqual(guard["status"], "PASS", guard)
        self.assertEqual(guard["top_level_forbidden_imports"], [])
        self.assertEqual(guard["forbidden_call_hits"], [])
        self.assertEqual(guard["h200"], 0)
        self.assertEqual(guard["model_import"], 0)

    def test_pi0159_execution_order_is_seed_interleaved(self) -> None:
        self.assertEqual(continuation.EXPECTED_ARM_KEYS, (
            ("Legacy", 2222), ("L1", 2222),
            ("Legacy", 2223), ("L1", 2223),
            ("Legacy", 2224), ("L1", 2224),
        ))

    def test_offline_namespace_links_only_accepted_cosmos(self) -> None:
        _, cosmos, config = _cosmos_fixture(self._tmpdir())
        output = Path(config["output_root"]) if "output_root" in config else Path(cosmos).parent.parent / "output"
        output.mkdir()
        old = {key: os.environ.get(key) for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HUB_OFFLINE", "GROOT_HF_LOCAL_FIRST")}
        old_tree_sha = continuation.EXPECTED_COSMOS_TREE_SHA256
        try:
            continuation.EXPECTED_COSMOS_TREE_SHA256 = str(config["cosmos_tree_sha256"])
            record = continuation.bind_offline_runtime(config, output)
            entry = output / "runtime_namespace" / "huggingface" / "hub" / continuation.EXPECTED_MODEL_ID
            self.assertTrue(entry.is_symlink())
            self.assertEqual(entry.resolve(), cosmos.resolve())
            self.assertEqual(record["cosmos_tree_sha256"], config["cosmos_tree_sha256"])
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(os.environ["HF_HUB_CACHE"], str(output / "runtime_namespace" / "huggingface" / "hub"))
            self.assertEqual(os.environ["GROOT_HF_LOCAL_FIRST"], "1")
            self.assertEqual(os.environ["GROOT_PATCH_MISTRAL"], "1")
        finally:
            continuation.EXPECTED_COSMOS_TREE_SHA256 = old_tree_sha
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_offline_namespace_rejects_hub_entry_drift(self) -> None:
        root = self._tmpdir()
        _, cosmos, config = _cosmos_fixture(root)
        wrong = root / "wrong"
        wrong.mkdir()
        config["cosmos_root"] = str(wrong)
        output = root / "output"
        output.mkdir()
        with self.assertRaisesRegex(continuation.ContinuationError, "HF Hub entry"):
            continuation.bind_offline_runtime(config, output)

    def test_config_enforces_pi0159_budget(self) -> None:
        root = self._tmpdir()
        value = _config(root)
        value["max_model_loads_total"] = 9
        path = root / "config.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(continuation.ContinuationError, "model-load ceiling"):
            continuation._config(path)

    def test_pbs_surface_binds_offline_bridge_and_absolute_entrypoint(self) -> None:
        source = Path("hpc/pbs/stage45f_pi0159_validation_e_continuation_h200.pbs").read_text(encoding="utf-8")
        self.assertIn("HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1", source)
        self.assertIn("GROOT_HF_LOCAL_FIRST=1 GROOT_PATCH_MISTRAL=1", source)
        self.assertIn("unset PYTHONPATH", source)
        self.assertIn("stage45f_pi0159_validation_e_continuation.py", source)
        self.assertIn('cd "${TMPDIR:-/tmp}"', source)
        self.assertNotIn("qsub", source)

    @staticmethod
    def _tmpdir() -> Path:
        import tempfile

        path = Path(tempfile.mkdtemp(prefix="pi0159-test-"))
        return path


if __name__ == "__main__":
    unittest.main()
