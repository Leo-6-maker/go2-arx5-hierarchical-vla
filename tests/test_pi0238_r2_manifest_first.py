from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pi0238_audit_to_sim as mature  # noqa: E402
import pi0238_r1_engineering_recovery as wrapper  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ManifestFirstTests(unittest.TestCase):
    def test_checkpoint_resolver_accepts_nested_manifest_and_rejects_bad_entries(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "fresh"
            checkpoints = root / "training"
            for step in (2500, 5000, 7500, 10000):
                (checkpoints / f"checkpoint-{step}").mkdir(parents=True)
            family = {
                "job_id": mature.FRESH_JOB,
                "root": str(root),
                "checkpoints": [
                    {
                        "checkpoint_step": step,
                        "path": str(checkpoints / f"checkpoint-{step}"),
                        "selection_candidate": True,
                        "tree_sha256": f"{step:064x}",
                    }
                    for step in (2500, 5000, 7500, 10000)
                ],
            }
            resolved = mature.resolve_sealed_checkpoints(family, expected_root=root)
            self.assertEqual(resolved["checkpoints"]["2500"]["path"], str((checkpoints / "checkpoint-2500").resolve()))

            duplicate = json.loads(json.dumps(family))
            duplicate["checkpoints"][1]["checkpoint_step"] = 2500
            with self.assertRaises(mature.DiagnosticBlocker):
                mature.resolve_sealed_checkpoints(duplicate, expected_root=root)

            string_candidate = json.loads(json.dumps(family))
            string_candidate["checkpoints"][0]["selection_candidate"] = "true"
            with self.assertRaises(mature.DiagnosticBlocker):
                mature.resolve_sealed_checkpoints(string_candidate, expected_root=root)

    def test_inventory_is_cwd_independent_and_rejects_empty(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            evidence = root / "evidence"
            other_cwd = root / "cwd with spaces"
            evidence.mkdir()
            other_cwd.mkdir()
            artifact = evidence / "artifact.txt"
            artifact.write_text("manifest-relative\n", encoding="utf-8")
            inventory = evidence / "inventory.sha256"
            inventory.write_text(f"{digest(artifact)}  artifact.txt\n", encoding="utf-8")
            old_cwd = Path.cwd()
            try:
                # The parser must use the declared evidence base, not cwd.
                import os
                os.chdir(other_cwd)
                result = mature.inventory_parse(inventory, evidence, [evidence])
            finally:
                os.chdir(old_cwd)
            self.assertEqual(result["rows"][0]["resolved_path"], str(artifact.resolve()))

            empty = evidence / "empty.sha256"
            empty.write_text("\n", encoding="utf-8")
            with self.assertRaises(mature.DiagnosticBlocker):
                mature.inventory_parse(empty, evidence, [evidence])

    def test_real_wrapper_dispatch_calls_mature_g0_on_nested_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            p237 = root / "p237"
            evidence = p237 / "evidence"
            evidence.mkdir(parents=True)
            train = root / "train"
            calib = root / "calib"
            for dataset in (train, calib):
                (dataset / "meta").mkdir(parents=True)
                (dataset / "meta" / "pi0227_train_only_split_lock.json").write_text(
                    json.dumps({"e_source_read_count": 0, "f_source_read_count": 0, "target_xyz_in_model_input": False}),
                    encoding="utf-8",
                )
            prep = root / "prep"
            (prep / "evidence").mkdir(parents=True)
            fresh = root / "fresh"
            for step in (2500, 5000, 7500, 10000):
                (fresh / "training" / f"checkpoint-{step}").mkdir(parents=True)

            run_path = root / "run" / "pi0236_train_calib_evaluation_run.json"
            run_path.parent.mkdir()
            run_path.write_text(json.dumps({"execution_status": "COMPLETE"}), encoding="utf-8")
            old_json = {
                "pi0236_s3_provenance_lock.json": {},
                "pi0236_s4_selection_freeze.json": {},
                "pi0236_s5_diagnostics.json": {},
                "pi0236_execution_state.json": {},
                "pi0228_train_calib_target_authority.json": {},
            }
            for name, payload in old_json.items():
                (evidence / name).write_text(json.dumps(payload), encoding="utf-8")
            cohort = {
                "train": [{"source_rollout_id": f"train-{i}"} for i in range(64)],
                "calib": [{"source_rollout_id": f"calib-{i}"} for i in range(64)],
            }
            cohort_json = evidence / "pi0237_cohort.json"
            cohort_csv = evidence / "pi0237_cohort.csv"
            cohort_json.write_text(json.dumps(cohort), encoding="utf-8")
            cohort_csv.write_text("source_rollout_id\n", encoding="utf-8")
            (evidence / "pi0228_train_calib_seal.json").write_text(
                json.dumps({"status": "PASS", "train": {"tree_sha256": mature.EXPECTED_TREES["train"]}, "train_calib": {"tree_sha256": mature.EXPECTED_TREES["calib"]}}),
                encoding="utf-8",
            )
            artifact = evidence / "artifact.txt"
            artifact.write_text("inventory artifact\n", encoding="utf-8")
            (evidence / "pi0237_evidence_inventory.sha256").write_text(
                f"{digest(artifact)}  artifact.txt\n", encoding="utf-8"
            )
            s2 = {
                "fresh_family": {
                    "job_id": mature.FRESH_JOB,
                    "root": str(fresh),
                    "checkpoints": [
                        {"checkpoint_step": step, "path": str(fresh / "training" / f"checkpoint-{step}"), "selection_candidate": True, "tree_sha256": f"{step:064x}"}
                        for step in (2500, 5000, 7500, 10000)
                    ],
                }
            }
            (evidence / "pi0236_s2_evaluator_admission.json").write_text(json.dumps(s2), encoding="utf-8")

            protocol = root / "protocol.md"
            original_protocol = root / "original.md"
            modality = root / "go2_arx5_modality.py"
            modality.write_text("CONFIG = {}\n", encoding="utf-8")
            for path in (protocol, original_protocol):
                path.write_text("protocol\n", encoding="utf-8")
            paths = {
                "output_root": root / "output",
                "cache_root": root / "cache",
                "tmp_root": root / "tmp",
                "p237_root": p237,
                "run_root": run_path.parent,
                "train_dataset": train,
                "calib_dataset": calib,
                "prep_root": prep,
                "vla_source": root / "vla",
                "stage10_root": root / "stage10",
                "groot_source": root / "groot",
                "baseline": root / "baseline",
                "fresh_root": fresh,
                "cosmos_snapshot": root / "cosmos",
                "cosmos_cache_root": root / "cosmos-cache",
                "modality_config": modality,
                "protocol": protocol,
                "original_protocol": original_protocol,
                "script": root / "runner.py",
                "pbs": root / "runner.pbs",
                "launch_config": root / "launch.json",
            }
            for name, path in paths.items():
                if name in {"output_root", "cache_root", "tmp_root", "launch_config"}:
                    continue
                if path.suffix:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch(exist_ok=True)
                else:
                    path.mkdir(parents=True, exist_ok=True)
            config = {
                "schema": "PI0238_R2_RESOLVED_LAUNCH_V1",
                "protocol_commit": wrapper.R2_PROTOCOL_COMMIT,
                "audited_parent": wrapper.R2_AUDITED_PARENT,
                "paths": {key: str(value) for key, value in paths.items()},
            }
            config_path = paths["launch_config"]
            config_path.write_text(json.dumps(config), encoding="utf-8")

            old_sha = mature.sha256_file
            sha_map = {
                str(run_path): "4b9ceff21be16a8a2c15b789cdde7b5ecc598af1cb9f9112f295d4481f2236e4",
                str(cohort_json): mature.EXPECTED_OLD["cohort_json"],
                str(cohort_csv): mature.EXPECTED_OLD["cohort_csv"],
            }
            for name, expected in {
                "pi0236_s3_provenance_lock.json": mature.EXPECTED_OLD["s3"],
                "pi0236_s4_selection_freeze.json": mature.EXPECTED_OLD["s4"],
                "pi0236_s5_diagnostics.json": mature.EXPECTED_OLD["s5"],
                "pi0236_execution_state.json": mature.EXPECTED_OLD["state"],
            }.items():
                sha_map[str(evidence / name)] = expected

            def fake_sha(path: Path) -> str:
                return sha_map.get(str(path), old_sha(path))

            def fake_profiles(_args: object) -> dict[int, dict[str, int]]:
                return {step: {"forward_count": 1088} for step in mature.PROFILE_STEPS}

            def fake_metadata(dataset: Path, split: str, _expected: int) -> dict[str, dict[str, str]]:
                return {"only": {"source_rollout_id": split}}

            def fake_tree(_root: Path, prefix: str) -> tuple[str, int, int]:
                return mature.EXPECTED_TREES["train" if prefix == "train" else "calib"], 1, 1

            def fake_capture(command: list[str]) -> dict[str, object]:
                return {"argv": command, "returncode": 0, "stdout": "", "stderr": ""}

            calls: list[dict[str, object]] = []
            original_resolver = mature.resolve_sealed_checkpoints

            def spy_resolver(*args: object, **kwargs: object) -> dict[str, object]:
                calls.append({"args": args, "kwargs": kwargs})
                return original_resolver(*args, **kwargs)

            with mock.patch.object(mature, "sha256_file", side_effect=fake_sha), \
                mock.patch.object(mature, "profile_records", side_effect=fake_profiles), \
                mock.patch.object(mature, "full_metadata", side_effect=fake_metadata), \
                mock.patch.object(mature, "tree_sha256", side_effect=fake_tree), \
                mock.patch.object(mature, "run_capture", side_effect=fake_capture), \
                mock.patch.object(mature, "resolve_sealed_checkpoints", side_effect=spy_resolver), \
                mock.patch.object(sys, "argv", ["runner", "--phase", "g0", "--launch-config", str(config_path)]):
                self.assertEqual(wrapper.main(), 0)

            self.assertTrue(calls, "wrapper dispatch did not invoke mature resolver")
            result = json.loads((paths["output_root"] / "evidence" / "g0_identity_and_access.json").read_text())
            self.assertEqual(result["fresh_family"]["checkpoints"]["2500"]["path"], str((fresh / "training" / "checkpoint-2500").resolve()))
            self.assertEqual(result["evidence_inventory"]["rows"][0]["resolved_path"], str(artifact.resolve()))


if __name__ == "__main__":
    unittest.main()
