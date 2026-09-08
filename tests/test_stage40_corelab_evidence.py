"""CPU-only tests for the Stage40 Corelab bundle and predecessor seal chain."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage40_corelab_evidence as EVIDENCE  # noqa: E402


HEAD = "a" * 40
LINEAGE = {"runner_sha256": "1" * 64, "source_sha256": "2" * 64}


class ChainHarness:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.seals = {}
        self.infos = {}

    @staticmethod
    def counts(stage_id):
        if stage_id == "RTX-runtime":
            return (0, 0, 0, [], [], [])
        if stage_id in {"RTX-Q0", "RTX-bind"}:
            return (1, 0, 0, [1], [], [])
        if stage_id == "M1":
            return (2, 11, 1, [1, 1], [5, 6], [1])
        return (4, 23, 3, [1, 1, 1, 1], [5, 6, 6, 6], [1, 1, 1])

    @staticmethod
    def budgets(stage_id):
        forwards, steps, updates, _, _, _ = ChainHarness.counts(stage_id)
        return {
            "gr00t_forward_count": forwards,
            "wbc_step_count": steps,
            "reference_update_count": updates,
        }

    def prepare(self, stage_id, predecessor_info=None, mutate=None):
        stage_root = self.root / stage_id
        stage_root.mkdir(parents=True, exist_ok=True)
        forwards, steps, updates, requests, segments, references = self.counts(stage_id)
        before = {name: 100 for name in EVIDENCE.COUNTER_NAMES}
        after = {
            "gr00t_forward_count": 100 + forwards,
            "wbc_step_count": 100 + steps,
            "reference_update_count": 100 + updates,
        }
        counters = EVIDENCE.build_counter_observation(
            before,
            after,
            gr00t_requests=requests,
            wbc_segments=segments,
            reference_updates=references,
        )
        role_paths = {}
        for role in sorted(EVIDENCE.REQUIRED_EVIDENCE_ROLES[stage_id]):
            relative = role + ".json"
            role_paths[role] = relative
            payload = counters if role == "counter_observations" else {
                "schema": "TEST_RAW_EVIDENCE_V1",
                "stage_id": stage_id,
                "role": role,
            }
            EVIDENCE.write_canonical_json(stage_root / relative, payload)
        manifest = EVIDENCE.build_evidence_manifest(stage_id, stage_root, role_paths)
        predecessor = None
        if predecessor_info is not None:
            predecessor = {
                "stage_id": predecessor_info["stage_id"],
                "seal_sha256": predecessor_info["seal_sha256"],
                "repository_head": predecessor_info["repository_head"],
                "execution_class": predecessor_info["execution_class"],
                "dev_seed": predecessor_info["dev_seed"],
                "status": predecessor_info["status"],
            }
        bundle_predecessor = None if predecessor is None else {
            "stage_id": predecessor["stage_id"],
            "seal_sha256": predecessor["seal_sha256"],
        }
        bundle = {
            "schema": "CORELAB_STAGE40_EXECUTION_BUNDLE_V1",
            "stage_id": stage_id,
            "pi_gate": EVIDENCE.PI_GATE,
            "classification": EVIDENCE.CLASSIFICATION,
            "execution_class": EVIDENCE.EXECUTION_CLASS[stage_id],
            "dev_seed": EVIDENCE.DEV_SEED,
            "repository_head": HEAD,
            "executable_lineage": copy.deepcopy(LINEAGE),
            "artifact_identities": {"structured_checkpoint_tree": "3" * 64},
            "budgets": self.budgets(stage_id),
            "forbidden_operations": EVIDENCE.expected_forbidden_operations(),
            "predecessor": bundle_predecessor,
        }
        result = {
            "schema": EVIDENCE.RESULT_SCHEMA[stage_id],
            "stage_id": stage_id,
            "pi_gate": EVIDENCE.PI_GATE,
            "classification": EVIDENCE.CLASSIFICATION,
            "status": "PASS",
            "dev_seed": EVIDENCE.DEV_SEED,
            "repository_head": HEAD,
            "execution_class": EVIDENCE.EXECUTION_CLASS[stage_id],
            "executable_lineage": copy.deepcopy(LINEAGE),
            "execution_bundle_sha256": EVIDENCE.sha256_bytes(EVIDENCE.canonical_json_bytes(bundle)),
            "predecessor": predecessor,
            "evidence_manifest": manifest,
            "evidence_manifest_sha256": EVIDENCE.sha256_bytes(EVIDENCE.canonical_json_bytes(manifest)),
            "counters": {
                "gr00t_forward_count": forwards,
                "wbc_step_count": steps,
                "reference_update_count": updates,
            },
            "safety_gate_decisions": ["ACCEPT"] * EVIDENCE.SAFETY_DECISION_COUNT[stage_id],
            "forbidden_operations": EVIDENCE.expected_forbidden_operations(),
        }
        if mutate is not None:
            mutate(bundle, result, stage_root)
        bundle_path = stage_root / (stage_id + "_bundle.json")
        result_path = stage_root / (stage_id + "_result.json")
        seal_path = stage_root / (stage_id + "_result.seal.json")
        EVIDENCE.write_canonical_json(bundle_path, bundle)
        EVIDENCE.write_canonical_json(result_path, result)
        return stage_root, bundle_path, result_path, seal_path

    def add(self, stage_id):
        expected = EVIDENCE.PREDECESSOR[stage_id]
        predecessor_info = None if expected is None else self.infos[expected]
        stage_root, bundle, result, seal = self.prepare(stage_id, predecessor_info)
        predecessor_seal = None if expected is None else self.seals[expected]
        info = EVIDENCE.seal_result(result, bundle, stage_root, seal, predecessor_seal)
        self.seals[stage_id] = seal
        self.infos[stage_id] = {
            **info,
            "repository_head": HEAD,
            "execution_class": EVIDENCE.EXECUTION_CLASS[stage_id],
            "dev_seed": EVIDENCE.DEV_SEED,
        }
        return stage_root, bundle, result, seal


class Stage40CorelabEvidenceTest(unittest.TestCase):
    def test_build_result_from_hash_bound_counter_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = ChainHarness(Path(directory))
            stage_root, bundle, expected_result, _ = harness.prepare("RTX-runtime")
            expected = EVIDENCE.read_json(expected_result)
            manifest_path = stage_root / "RTX-runtime_evidence_manifest.json"
            EVIDENCE.write_canonical_json(manifest_path, expected["evidence_manifest"])
            built_path = stage_root / "rebuilt_result.json"
            info = EVIDENCE.build_result_document(
                bundle,
                manifest_path,
                stage_root,
                built_path,
                safety_gate_decisions=[],
            )
            self.assertEqual(info["status"], "PASS")
            self.assertEqual(EVIDENCE.read_json(built_path), expected)

    def test_positive_chain_runtime_q0_bind_m1_m2(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = ChainHarness(Path(directory))
            for stage_id in EVIDENCE.STAGES:
                stage_root, _, _, seal = harness.add(stage_id)
                predecessor = EVIDENCE.PREDECESSOR[stage_id]
                info = EVIDENCE.verify_stage_seal(
                    seal,
                    stage_root,
                    expected_stage=stage_id,
                    expected_head=HEAD,
                    predecessor_seal=None if predecessor is None else harness.seals[predecessor],
                )
                self.assertEqual(info["status"], "PASS")
                self.assertTrue(Path(str(seal) + ".sha256").is_file())

    def test_missing_predecessor_seal_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = ChainHarness(Path(directory))
            harness.add("RTX-runtime")
            stage_root, bundle, result, seal = harness.prepare("RTX-Q0", harness.infos["RTX-runtime"])
            with self.assertRaisesRegex(EVIDENCE.Stage40EvidenceError, "predecessor seal is required"):
                EVIDENCE.seal_result(result, bundle, stage_root, seal)

    def test_bad_seal_sha_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = ChainHarness(Path(directory))
            stage_root, _, _, seal = harness.add("RTX-runtime")
            Path(str(seal) + ".sha256").write_text("0" * 64 + "  " + seal.name + "\n", encoding="utf-8")
            with self.assertRaisesRegex(EVIDENCE.Stage40EvidenceError, "seal SHA mismatch"):
                EVIDENCE.verify_stage_seal(seal, stage_root)

    def test_wrong_stage_seed_head_and_execution_class_fail(self):
        mutations = {
            "stage": lambda bundle, result, root: result.__setitem__("stage_id", "M2"),
            "seed": lambda bundle, result, root: result.__setitem__("dev_seed", 7),
            "class": lambda bundle, result, root: result.__setitem__("execution_class", "WRONG"),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                harness = ChainHarness(Path(directory))
                stage_root, bundle, result, seal = harness.prepare("RTX-runtime", mutate=mutation)
                with self.assertRaises(EVIDENCE.Stage40EvidenceError):
                    EVIDENCE.seal_result(result, bundle, stage_root, seal)
        with tempfile.TemporaryDirectory() as directory:
            harness = ChainHarness(Path(directory))
            stage_root, _, _, seal = harness.add("RTX-runtime")
            with self.assertRaisesRegex(EVIDENCE.Stage40EvidenceError, "unexpected repository HEAD"):
                EVIDENCE.verify_stage_seal(seal, stage_root, expected_head="b" * 40)

    def test_evidence_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = ChainHarness(Path(directory))
            stage_root, _, _, seal = harness.add("RTX-runtime")
            (stage_root / "telemetry.json").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(EVIDENCE.Stage40EvidenceError, "evidence .* mismatch"):
                EVIDENCE.verify_stage_seal(seal, stage_root)

    def test_counter_forgery_fails_closed(self):
        def forge(bundle, result, stage_root):
            path = stage_root / "counter_observations.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["counters"]["gr00t_forward_count"]["delta"] = 1
            EVIDENCE.write_canonical_json(path, value)
            result["evidence_manifest"] = EVIDENCE.build_evidence_manifest(
                "RTX-runtime",
                stage_root,
                {role: role + ".json" for role in EVIDENCE.REQUIRED_EVIDENCE_ROLES["RTX-runtime"]},
            )
            result["evidence_manifest_sha256"] = EVIDENCE.sha256_bytes(
                EVIDENCE.canonical_json_bytes(result["evidence_manifest"])
            )

        with tempfile.TemporaryDirectory() as directory:
            harness = ChainHarness(Path(directory))
            stage_root, bundle, result, seal = harness.prepare("RTX-runtime", mutate=forge)
            with self.assertRaisesRegex(EVIDENCE.Stage40EvidenceError, "forged/non-monotonic"):
                EVIDENCE.seal_result(result, bundle, stage_root, seal)

    def test_m2_rejects_anything_except_m1_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = ChainHarness(Path(directory))
            for stage_id in ("RTX-runtime", "RTX-Q0", "RTX-bind", "M1"):
                harness.add(stage_id)
            stage_root, bundle_path, result_path, seal = harness.prepare("M2", harness.infos["M1"])
            with self.assertRaisesRegex(EVIDENCE.Stage40EvidenceError, "wrong execution bundle stage"):
                EVIDENCE.seal_result(
                    result_path,
                    bundle_path,
                    stage_root,
                    seal,
                    predecessor_seal=harness.seals["RTX-Q0"],
                )

    def test_bundle_budget_and_forbidden_operation_are_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = ChainHarness(Path(directory))

            def bad_budget(bundle, result, root):
                bundle["budgets"]["gr00t_forward_count"] = 2
                result["execution_bundle_sha256"] = EVIDENCE.sha256_bytes(EVIDENCE.canonical_json_bytes(bundle))

            stage_root, bundle, result, seal = harness.prepare("RTX-Q0", {"stage_id": "RTX-runtime", "seal_sha256": "4" * 64, "repository_head": HEAD, "execution_class": EVIDENCE.EXECUTION_CLASS["RTX-runtime"], "dev_seed": EVIDENCE.DEV_SEED, "status": "PASS"}, mutate=bad_budget)
            with self.assertRaisesRegex(EVIDENCE.Stage40EvidenceError, "wrong GR00T forward budget"):
                EVIDENCE.seal_result(result, bundle, stage_root, seal, stage_root / "missing")

            def forbidden(bundle, result, root):
                bundle["forbidden_operations"]["training"] = True
                result["execution_bundle_sha256"] = EVIDENCE.sha256_bytes(EVIDENCE.canonical_json_bytes(bundle))

            stage_root, bundle, result, seal = harness.prepare("RTX-runtime", mutate=forbidden)
            with self.assertRaisesRegex(EVIDENCE.Stage40EvidenceError, "forbidden operation"):
                EVIDENCE.seal_result(result, bundle, stage_root, seal)


if __name__ == "__main__":
    unittest.main()
