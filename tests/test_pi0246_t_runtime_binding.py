from __future__ import annotations

import json
from tempfile import TemporaryDirectory
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pi0246_gate_pass_train as train


class TRuntimeBindingTest(unittest.TestCase):
    def test_t_runtime_seal_binds_staged_code_and_frozen_card(self) -> None:
        with TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            campaign = tmp_path / "campaign"
            runtime = tmp_path / "runtime"
            code_root = tmp_path / "code"
            output = tmp_path / "T"
            campaign.mkdir()
            runtime.mkdir()
            (code_root / "scripts").mkdir(parents=True)
            for name in ("pi0241_fixed16_learning.py", "pi0228_h200_large_adapt_v1.py"):
                (code_root / "scripts" / name).write_text(name, encoding="utf-8")
            (campaign / "pi0245_physical_delta.py").write_text("frozen helper\n", encoding="utf-8")
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
                "paths": {"campaign_code_root": str(campaign), "code_root": str(code_root), "output_root": str(output)},
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
