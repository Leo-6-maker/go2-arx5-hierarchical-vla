from __future__ import annotations

import inspect
import json
from tempfile import TemporaryDirectory
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pi0246_gate_pass_train as train


class TRuntimeBindingTest(unittest.TestCase):
    def test_functional_preflight_uses_real_dataset_path_without_model_setup(self) -> None:
        source = inspect.getsource(train.functional_preflight)
        self.assertIn("build_runtime_config", source)
        self.assertIn("pipeline._create_dataset", source)
        self.assertNotIn("pipeline.setup(", source)
        self.assertNotIn("AutoModel", source)
        self.assertNotIn("run_official(", source)

    def test_corelab_wrapper_binds_verified_ffmpeg6_before_training(self) -> None:
        wrapper = Path(__file__).resolve().parents[1] / "scripts" / "pi0246_corelab_t_train.sh"
        source = wrapper.read_text(encoding="utf-8")
        for key in (
            "ffmpeg6_root", "ffmpeg6_manifest", "ffmpeg6_manifest_sha256",
            "ffmpeg6_bootstrap_record", "ffmpeg6_bootstrap_record_sha256",
            "ffmpeg6_capsule_sha256", "torchcodec_library_root", "torch_library_root",
        ):
            self.assertIn(f".paths.{key}", source)
        self.assertIn('export LD_LIBRARY_PATH="$ffmpeg_root/lib:$ffmpeg_root/compat:$torchcodec_root:$torch_library_root:', source)
        self.assertLess(source.index('export LD_LIBRARY_PATH='), source.index('exec "$PI0246_T_PYTHON"'))
        self.assertIn('--functional-preflight|--train', source)
        self.assertIn('"$mode" --run-card "$PI0246_T_RUN_CARD"', source)

    def test_h200_wrapper_runs_no_model_preflight_before_training(self) -> None:
        wrapper = Path(__file__).resolve().parents[1] / "hpc" / "pbs" / "pi0246_gate_pass_t_logged_h200.pbs"
        source = wrapper.read_text(encoding="utf-8")
        preflight = source.index("--functional-preflight --run-card")
        training = source.index("--train --run-card")
        self.assertLess(preflight, training)

    def test_h200_wrapper_checks_user_quota_root_not_nfs_global_free_space(self) -> None:
        wrapper = Path(__file__).resolve().parents[1] / "hpc" / "pbs" / "pi0246_gate_pass_t_logged_h200.pbs"
        source = wrapper.read_text(encoding="utf-8")
        self.assertIn('quota_root="/scratch/$(id -un)"', source)
        self.assertIn("stat -f -c '%a %s' \"$quota_root\"", source)
        self.assertNotIn("stat -f -c '%a %s' \"$output\"", source)

    def test_h200_wrapper_does_not_require_parent_delta_at_campaign_root(self) -> None:
        wrapper = Path(__file__).resolve().parents[1] / "hpc" / "pbs" / "pi0246_gate_pass_t_logged_h200.pbs"
        source = wrapper.read_text(encoding="utf-8")
        self.assertIn('test -r "$runtime/pi0245_physical_delta.py"', source)
        self.assertNotIn('test -r "$campaign/pi0245_physical_delta.py"', source)
        self.assertIn('test -r "$campaign/pi0245_p1_stock.py"', source)

    def test_t_runtime_seal_binds_staged_code_and_frozen_card(self) -> None:
        with TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            campaign = tmp_path / "campaign"
            runtime = tmp_path / "runtime"
            code_root = tmp_path / "code"
            p1_root = tmp_path / "pi0245" / "p1"
            output = tmp_path / "T"
            campaign.mkdir()
            runtime.mkdir()
            (code_root / "scripts").mkdir(parents=True)
            for name in ("pi0241_fixed16_learning.py", "pi0228_h200_large_adapt_v1.py"):
                (code_root / "scripts" / name).write_text(name, encoding="utf-8")
            (p1_root.parent / "code").mkdir(parents=True)
            (p1_root.parent / "code" / "pi0245_physical_delta.py").write_text("frozen helper\n", encoding="utf-8")
            (campaign / "pi0246_gate_pass_train.py").write_text("parent trainer\n", encoding="utf-8")
            (campaign / "pi0245_native_overlay.py").write_text("parent overlay\n", encoding="utf-8")
            (campaign / "pi0245_p1_stock.py").write_text("parent p1 runner\n", encoding="utf-8")
            (runtime / "pi0245_physical_delta.py").write_text("frozen helper\n", encoding="utf-8")
            (runtime / "pi0246_gate_pass_train.py").write_text("staged trainer\n", encoding="utf-8")
            (runtime / "pi0245_native_overlay.py").write_text("staged overlay\n", encoding="utf-8")
            (runtime / "pi0246_reconcile_c_update.py").write_text("staged reconciler\n", encoding="utf-8")
            wrapper = runtime / "train.pbs"
            wrapper.write_text("staged wrapper\n", encoding="utf-8")
            card = {
                "schema": "PI0246_ARM_RUN_CARD_V1", "status": "SEALED_READY_NOT_STARTED",
                "arm": "T", "family": train.ARMS["T"][0], "loss": train.ARMS["T"][1],
                "seed": train.SEED, "resolved_config": {"max_steps": train.STEPS},
                "paths": {"campaign_code_root": str(campaign), "code_root": str(code_root),
                          "p1_root": str(p1_root), "output_root": str(output)},
            }
            card_path = tmp_path / "T_run_card.json"
            card_path.write_text(json.dumps(card), encoding="utf-8")
            card_path.with_name("run_card.seal.json").write_text(
                json.dumps({"sha256": train.sha256_file(card_path)}), encoding="utf-8"
            )

            sealed = train.seal_t_runtime(card_path, runtime, wrapper)
            verified = train.verify_t_runtime(
                card_path, runtime, wrapper, runtime / "T_runtime_binding.json", sealed["binding_sha256"]
            )
            self.assertEqual(verified["status"], "PASS_T_RUNTIME_BINDING")
            self.assertEqual(sealed["binding"]["physical_delta_sha256"], sealed["binding"]["parent_physical_delta_sha256"])

            (runtime / "pi0245_native_overlay.py").write_text("changed after seal\n", encoding="utf-8")
            with self.assertRaisesRegex(train.CampaignError, "binding, card, wrapper"):
                train.verify_t_runtime(
                    card_path, runtime, wrapper, runtime / "T_runtime_binding.json", sealed["binding_sha256"]
                )
if __name__ == "__main__":
    unittest.main()
