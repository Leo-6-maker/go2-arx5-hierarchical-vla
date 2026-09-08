import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Stage45BHandoffTest(unittest.TestCase):
    def test_manifest_contract_and_recipe_are_frozen(self):
        manifest = json.loads(
            (ROOT / "manifests/STAGE45B_FULL24_FRESH_BASE_EXECUTION_MANIFEST_V1.json").read_text()
        )
        self.assertEqual(manifest["schema"], "CORELAB_STAGE45B_FULL24_FRESH_BASE_EXECUTION_MANIFEST_V1")
        self.assertEqual(manifest["status"], "CORELAB_STAGE45B_H200_ADMISSION_READY")
        self.assertEqual(manifest["intervention"], "B_FRESH_BASE_FULL24")
        self.assertEqual(manifest["dataset"]["window_count"], 24)
        self.assertIn("structured_stats_artifact", manifest["runtime_paths"])
        self.assertEqual(manifest["structured_stats"]["sha256"], "49c7139dff09d5795e31205c7e72cb11e5801ce5f176572ebc1ec8d91f310c3d")
        transport = manifest["structured_stats"]["embedded_transport"]
        self.assertEqual(transport["decoded_sha256"], manifest["structured_stats"]["sha256"])
        self.assertTrue((ROOT / transport["path"]).is_file())
        self.assertEqual(manifest["recipe"]["seed"], 2222)
        self.assertEqual(manifest["recipe"]["max_steps"], 2000)
        self.assertEqual(manifest["recipe"]["save_steps"], [1000, 2000])
        self.assertFalse(manifest["recipe"]["resume_from_checkpoint"])
        self.assertTrue(manifest["forbidden"]["option_a_warm_start"])
        self.assertTrue(manifest["forbidden"]["validation_e_read"])
        self.assertTrue(manifest["forbidden"]["holdout_f_read"])
        self.assertEqual(manifest["execution_boundary"]["optimizer_steps"], 0)

    def test_runner_and_pbs_do_not_offer_alternate_experiment(self):
        runner = (ROOT / "scripts/stage45b_full24_fresh_base_train.py").read_text()
        pbs = (ROOT / "hpc/pbs/stage45b_full24_fresh_base.pbs").read_text()
        self.assertNotIn("STRUCTURED@1000", runner)
        self.assertNotIn("--resume", pbs)
        self.assertNotIn("--eval", pbs)
        self.assertEqual(len(re.findall(r"checkpoint-1000", runner)), 1)
        self.assertEqual(len(re.findall(r"checkpoint-2000", runner)), 1)


if __name__ == "__main__":
    unittest.main()
