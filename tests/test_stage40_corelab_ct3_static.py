from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


NATIVE_SOURCE = ROOT / "scripts" / "stage40_corelab_native_hold.py"
MODEL_SOURCE = ROOT / "scripts" / "stage40_corelab_model_hold.py"
LAUNCHER = ROOT / "hpc" / "stage40_corelab_ct3_runtime.sh"


class Stage40CorelabCt3StaticTest(unittest.TestCase):
    def test_native_constructor_trajectory_is_non_scientific_and_finite(self) -> None:
        source = NATIVE_SOURCE.read_text(encoding="utf-8")
        self.assertIn("def _construction_trajectory", source)
        self.assertIn("construction-only chunk source", source)
        self.assertIn("np.asarray([0.0, 2.0], dtype=np.float64)", source)
        self.assertIn("np.asarray([[0.30, 0.10, 0.58], [0.30, 0.10, 0.58]]", source)
        self.assertIn("np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]", source)

    def test_cutover_token_requires_one_exact_line_and_selected_uuid(self) -> None:
        uuid = "GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad"
        source = NATIVE_SOURCE.read_text(encoding="utf-8")
        self.assertIn("lines[0] in CUTOVER_TOKENS", source)
        self.assertIn("CUDA_VISIBLE_DEVICES", source)
        self.assertIn("--gpu-uuid", source)

    def test_model_hold_has_no_forward_or_decode_call(self) -> None:
        source = MODEL_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("get_action", source)
        self.assertNotIn("_run_one", source)
        self.assertIn("AutoModel.from_pretrained", source)
        self.assertIn("VANDA_STAGE40_A40_QUEUE_RELEASED_CORELAB_PRIMARY", source)
        self.assertIn("VANDA_STAGE40_A40_EXECUTION_ENDED_CORELAB_PRIMARY", source)

    def test_native_hold_has_no_policy_step_or_reference_update_call(self) -> None:
        source = NATIVE_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("_policy_step(", source)
        self.assertNotIn("load_reference_chunk(", source)
        self.assertNotIn("initial_bind_reference(", source)
        self.assertIn("_build_native", source)

    def test_launcher_is_direct_and_fail_closed(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn("#PBS", source)
        self.assertNotIn("Apptainer", source)
        self.assertNotIn("singularity", source)
        self.assertIn("VANDA_STAGE40_A40_QUEUE_RELEASED_CORELAB_PRIMARY", source)
        self.assertIn("VANDA_STAGE40_A40_EXECUTION_ENDED_CORELAB_PRIMARY", source)
        self.assertIn("CUDA_VISIBLE_DEVICES=\"${GPU_UUID}\"", source)
        self.assertIn("nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used", source)
        self.assertIn("test \"${GPU_TOTAL_MIB}\" -ge 30000", source)
        self.assertIn("stage40_corelab_native_hold.py", source)
        self.assertIn("stage40_corelab_model_hold.py", source)

    def test_launcher_requires_model_and_native_paths_before_gpu_start(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        preflight = source.split("nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used", 1)[0]
        for variable in (
            "STAGE40_CT3_BUNDLE",
            "STAGE40_CT3_NATIVE_PYTHON",
            "STAGE40_CT3_MODEL_PYTHON",
            "STAGE40_CT3_MODEL_PATH",
            "STAGE40_CT3_COSMOS_SNAPSHOT",
            "STAGE40_CUTOVER_TOKEN_FILE",
        ):
            self.assertIn(variable, preflight)

    def test_ct3_preparation_manifest_is_explicitly_unexecuted(self) -> None:
        manifest = json.loads(
            (ROOT / "manifests" / "CORELAB_STAGE40_CT3_RUNTIME_PREPARATION_20260908.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["pi_gate"], "PI_GATE_STATE_0065")
        self.assertEqual(manifest["issue_comment"], 5577233814)
        self.assertEqual(manifest["status"], "PREPARED_NOT_EXECUTED")
        self.assertFalse(manifest["cutover_token_seen"])
        self.assertEqual(len(manifest["cutover_tokens"]), 2)
        self.assertEqual(manifest["protocol"]["gr00t_forward_budget"], 0)
        self.assertEqual(manifest["protocol"]["wbc_step_budget"], 0)
        self.assertEqual(manifest["protocol"]["reference_update_budget"], 0)
        self.assertEqual(manifest["protocol"]["scientific_rows"], 0)


if __name__ == "__main__":
    unittest.main()
