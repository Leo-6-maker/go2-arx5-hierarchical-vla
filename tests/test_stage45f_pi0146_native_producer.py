from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45f_corelab_pi0145_v2_adapter as adapter


PINNED = adapter.PI0146_VANDA_PRODUCER_HEAD


class PI0146NativeProducerIntegrationTest(unittest.TestCase):
    def _load_pinned_vanda_module(self, root: Path):
        source = subprocess.check_output(
            ["git", "show", f"{PINNED}:scripts/stage45e_h200_unified_replication.py"],
            cwd=ROOT,
        )
        source_path = root / "pinned_vanda_stage45e.py"
        source_path.write_bytes(source)
        spec = importlib.util.spec_from_file_location("pinned_vanda_stage45e", source_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_native_cpu_closure_reaches_shared_consumer_and_reports_producer_gap(self) -> None:
        contract = adapter.validate_pi0146_producer_contract()
        self.assertEqual(contract["producer_head"], PINNED)
        with tempfile.TemporaryDirectory(prefix="stage45f-pi0146-native-") as directory:
            root = Path(directory)
            vanda = self._load_pinned_vanda_module(root)
            campaign = root / "campaign"
            (campaign / "specs").mkdir(parents=True)
            lineages = {route: {} for route in vanda.ROUTES}
            for route in vanda.ROUTES:
                for seed in vanda.ALL_SEEDS:
                    lineages[route][seed] = {
                        "route": route,
                        "seed": seed,
                        "root": str(root / "lineages" / route / f"seed{seed}"),
                        "checkpoints": {str(step): {"tree_sha256": "a" * 64} for step in (1000, 2000)},
                        "training_source_kind": vanda.TRAINING_SOURCE_STANDARD,
                        "training_lineage_head": "b" * 40,
                        "training_lineage_branch": "native-test-lineage",
                    }
            phase_specs = vanda.build_phase_specs(
                lineages,
                campaign,
                evaluator={"role": "EVAL", "head": "c" * 40},
                orchestration={"role": "ORCH", "head": "d" * 40},
                synthetic_test_only=True,
            )
            phase_roots = {phase: {} for phase in vanda.DIAGNOSTIC_PHASES}
            for phase in vanda.DIAGNOSTIC_PHASES:
                for route in vanda.ROUTES:
                    attempt_root = root / "attempts" / phase / route
                    attempt_root.mkdir(parents=True)
                    inner = attempt_root / "inner.json"
                    inner.write_text('{"status":"PASS"}\n', encoding="utf-8")
                    rows = attempt_root / "rows.json"
                    rows.write_text('{"rows":1}\n', encoding="utf-8")
                    seal = attempt_root / "unified_phase_seal.json"
                    seal.write_text(
                        json.dumps(
                            {
                                "status": "PASS",
                                "phase": phase,
                                "route": route,
                                "inner_seal": str(inner),
                                "row_bundle": [str(rows)],
                                "stats_sha256": vanda.N2_STATS_SHA if route == "n2" else vanda.RUNTIME_STATS_SHA,
                                "stats_payload_sha256": vanda.N2_STATS_PAYLOAD_SHA if route == "n2" else vanda.RUNTIME_STATS_PAYLOAD_SHA,
                            },
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    phase_roots[phase][route] = attempt_root
            # These are the producer's own native CPU serialization functions,
            # not Corelab's producer-shaped fixture builder.
            native = vanda.build_external_evidence_closure(
                campaign,
                phase_roots,
                phase_specs,
                lineages,
                evaluator={"role": "EVAL", "head": "c" * 40},
                orchestration={"role": "ORCH", "head": "d" * 40},
            )
            self.assertEqual(len(native["descriptors"]), 9)
            native_descriptor = json.loads((campaign / "evidence_units/E4_legacy.json").read_text(encoding="utf-8"))
            self.assertNotIn("train24_contract", native_descriptor)
            with self.assertRaises(adapter.V2AdapterError) as caught:
                adapter.validate_external_inventory(
                    Path(native["inventory_path"]), campaign_root=campaign, allow_synthetic=True
                )
            self.assertIn("PRODUCER_PREREQUISITE_MISSING:", str(caught.exception))
            self.assertIn(".train24_contract", str(caught.exception))
            self.assertFalse((campaign / "final_result.json").exists())


if __name__ == "__main__":
    unittest.main()
