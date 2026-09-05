from __future__ import annotations

import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/stage29_c2_free_running_transfer.py"
SPEC = importlib.util.spec_from_file_location("stage29_c2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _noise_row(seed: int, checkpoint: str, digest: str) -> dict:
    return {"seed": seed, "checkpoint": checkpoint, "initial_noise": {"raw_sha256": digest}}


class Stage29ContractTests(unittest.TestCase):
    def test_protocol_is_exactly_six_paired_official_sampler_calls(self) -> None:
        plan = MODULE.protocol_plan()
        MODULE.assert_protocol_static(plan)
        self.assertEqual(plan["experiment"]["seeds"], [2223, 2224, 2225])
        self.assertEqual(plan["experiment"]["checkpoints"], ["BASE", "H2@2000"])
        self.assertEqual(plan["experiment"]["maximum_sampler_calls"], 6)
        self.assertEqual(plan["sampler"]["steps"], 4)
        self.assertEqual(plan["sampler"]["continuous_time_grid"], [0.0, 0.25, 0.5, 0.75])

    def test_execution_boundary_keeps_training_wbc_and_locked_data_false(self) -> None:
        execution = MODULE.protocol_plan()["execution"]
        for key in (
            "training", "backward", "optimizer_step", "scheduler_step", "resume",
            "checkpoint_write", "wbc", "hardware", "validation_e", "holdout_f",
        ):
            self.assertIs(execution[key], False)

    def test_runner_has_no_training_or_checkpoint_write_call(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertNotIn("backward", called_attributes)
        self.assertNotIn("save_pretrained", called_attributes)
        self.assertNotIn("train", called_attributes)
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(".get_action(**call_inputs)", source)
        self.assertIn("np.savez_compressed", source)

    def test_no_eager_torch_or_transformers_import(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        top_imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                top_imports.append(node.module or "")
        self.assertNotIn("torch", top_imports)
        self.assertNotIn("transformers", top_imports)

    def test_paired_noise_gate_requires_exact_hash_equality(self) -> None:
        records = [
            _noise_row(seed, checkpoint, f"{seed:064x}")
            for seed in MODULE.SEEDS
            for checkpoint in MODULE.CHECKPOINTS
        ]
        self.assertEqual(MODULE.validate_paired_noise(records)["status"], "MATCH")
        records[-1]["initial_noise"]["raw_sha256"] = "f" * 64
        with self.assertRaisesRegex(MODULE.Stage29Error, "paired initial-noise mismatch"):
            MODULE.validate_paired_noise(records)

    def test_output_must_not_overlap_immutable_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            immutable = root / "checkpoint"
            immutable.mkdir()
            with self.assertRaisesRegex(MODULE.Stage29Error, "overlaps immutable"):
                MODULE._ensure_external_output(immutable / "nested", (immutable,))
            output = root / "stage29-output"
            MODULE._ensure_external_output(output, (immutable,))
            self.assertTrue(output.is_dir())

    def test_s0_report_and_source_hash_contract_are_committed(self) -> None:
        report = REPO / "reports/STAGE29_C2_SOURCE_AUDIT.md"
        text = report.read_text(encoding="utf-8")
        self.assertIn("PASS_UNAMBIGUOUS_OFFICIAL_SAMPLER", text)
        self.assertIn("51d4c89f72fda44cbf77285c6a8114b52676b8a1", text)
        self.assertIn("No second semantically plausible stock PyTorch sampler path", text)
        self.assertEqual(len(MODULE.SOURCE_FILES), 8)

    def test_pbs_wrapper_is_h200_evaluation_only(self) -> None:
        wrapper = (REPO / "hpc/pbs/stage29_c2_free_running_transfer.pbs").read_text(encoding="utf-8")
        self.assertIn("#PBS -P CFP03-CF-004", wrapper)
        self.assertIn("#PBS -l select=1:ngpus=1", wrapper)
        self.assertIn("--mode execute", wrapper)
        self.assertIn("--device cuda:0", wrapper)
        self.assertIn("PI_GATE_STATE_0009", wrapper)
        self.assertNotIn("qsub", wrapper)


if __name__ == "__main__":
    unittest.main()
