from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

import sys

sys.path.insert(0, "scripts")
import stage45f_pi0163_ffmpeg_capsule_bootstrap as bootstrap


class PI0163SupportTests(unittest.TestCase):
    def _allocation8_raw_fixture(self, root: Path) -> Path:
        import numpy as np

        raw_root = root / "raw" / "Legacy_seed2222_at_2000"
        raw_root.mkdir(parents=True)
        records = []
        for index, window_id in enumerate(
            f"stage22_expert_red_target_holdout_e:expert-{index:04d}" for index in range(6)
        ):
            array = np.full((1, 40, 132), float(index), dtype=np.float32)
            path = raw_root / f"window-{index:02d}.npy"
            np.save(path, array, allow_pickle=False)
            raw_bytes = path.read_bytes()
            records.append({
                "schema": "STAGE45F_PI0154_RAW_PREDICTION_V1",
                "checkpoint": {"label": "Legacy(seed2222)@2000", "route": "Legacy", "seed": 2222, "tier": 2000},
                "window_id": window_id,
                "path": str(path.relative_to(root)),
                "file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "array_sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
                "shape": [1, 40, 132],
                "dtype": "float32",
                "timestep_ids": [0, 250, 500, 750],
                "action_head_calls_observed": 4,
                "decoded": False,
                "noise_sha256": f"noise-{index}",
            })
        (root / "raw_predictions.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
        return root

    def _aggregate_rows(self, recovery: object) -> list[dict[str, object]]:
        fields = {
            "position_rmse_m": 1.0,
            "position_max_abs_m": 1.0,
            "endpoint_error_m": 1.0,
            "prediction_path_length_m": 1.0,
            "expert_path_length_m": 1.0,
            "eef_rot6d_rmse": 1.0,
            "eef_rot6d_max_abs": 1.0,
            "gripper_rmse": 1.0,
            "gripper_max_abs": 1.0,
            "full_action_rmse_mixed_units": 1.0,
            "roughness_ratio": 1.0,
        }
        target = {
            "initial_target_distance_m": 1.0,
            "minimum_target_distance_m": 0.5,
            "final_target_distance_m": 0.25,
            "absolute_progress_m": 0.75,
            "best_progress_m": 0.75,
            "path_length_m": 1.0,
            "progress_sign": "POSITIVE",
            "endpoint_target_vector_cosine": 1.0,
            "roughness_position_second_difference_m": {"mean": 0.0},
        }
        return [
            {
                "window_id": window_id,
                "checkpoint": {"label": "fixture", "sha256": "checkpoint"},
                "model": {"label": "fixture", "sha256": "model"},
                "metrics": {**fields, "target": target},
                "raw_prediction_artifact": {"array_sha256": f"raw-{index}"},
                "decoded_action": {"array_sha256": f"decoded-{index}"},
                "safety_gate_decision": "ACCEPT",
            }
            for index, window_id in enumerate(recovery.EXPECTED_WINDOW_IDS)
        ]

    def _fixture(self, root: Path, *, escaping: bool = False) -> tuple[Path, str, str, str]:
        lib_bytes = b"abc"
        compat_bytes = b"xyz"
        manifest: dict[str, object] = {
            "schema": "STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1",
            "status": "SEALED_IMMUTABLE",
            "artifact_id": bootstrap.EXPECTED_RUNTIME_ID,
            "payload_tree_sha256": bootstrap.EXPECTED_RUNTIME_ID,
            "payload_entries": [
                {
                    "path": "lib/libavutil.so.58.29.100",
                    "kind": "regular_file",
                    "bytes": len(lib_bytes),
                    "sha256": hashlib.sha256(lib_bytes).hexdigest(),
                },
                {
                    "path": "lib/libavutil.so.58",
                    "kind": "symlink",
                    "target": "../outside" if escaping else "libavutil.so.58.29.100",
                    "sha256": None,
                },
                {
                    "path": "compat/libpython3.12.so.1.0",
                    "kind": "regular_file",
                    "bytes": len(compat_bytes),
                    "sha256": hashlib.sha256(compat_bytes).hexdigest(),
                },
            ],
        }
        manifest_bytes = json.dumps(manifest, sort_keys=True).encode() + b"\n"
        checksum_bytes = b"manifest\n"
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        checksum_sha = hashlib.sha256(checksum_bytes).hexdigest()
        archive = root / "capsule.tar"
        with tarfile.open(archive, "w") as handle:
            for directory in ("lib", "compat"):
                info = tarfile.TarInfo(directory)
                info.type = tarfile.DIRTYPE
                # Match the production capsule's non-writable payload roots;
                # extraction must defer restoring this mode until children
                # have been materialized.
                info.mode = 0o555
                handle.addfile(info)
            for name, data in (
                ("lib/libavutil.so.58.29.100", lib_bytes),
                ("compat/libpython3.12.so.1.0", compat_bytes),
                (bootstrap.SOURCE_MANIFEST, manifest_bytes),
                (bootstrap.SOURCE_CHECKSUM_FILE, checksum_bytes),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = 0o644
                handle.addfile(info, io.BytesIO(data))
            link = tarfile.TarInfo("lib/libavutil.so.58")
            link.type = tarfile.SYMTYPE
            link.linkname = str(manifest["payload_entries"][1]["target"])
            handle.addfile(link)
        return archive, hashlib.sha256(archive.read_bytes()).hexdigest(), manifest_sha, checksum_sha

    def test_production_capsule_constants_are_pinned(self) -> None:
        self.assertEqual(len(bootstrap.EXPECTED_RUNTIME_ID), 64)
        self.assertEqual(len(bootstrap.EXPECTED_MANIFEST_SHA256), 64)
        self.assertEqual(len(bootstrap.EXPECTED_CHECKSUM_FILE_SHA256), 64)
        self.assertEqual(bootstrap.EXPECTED_ROOTS, {"lib", "compat"})

    def test_fixture_materializes_and_rehashes_exact_entries(self) -> None:
        old_manifest_sha = bootstrap.EXPECTED_MANIFEST_SHA256
        old_checksum_sha = bootstrap.EXPECTED_CHECKSUM_FILE_SHA256
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, archive_sha, manifest_sha, checksum_sha = self._fixture(root)
            bootstrap.EXPECTED_MANIFEST_SHA256 = manifest_sha
            bootstrap.EXPECTED_CHECKSUM_FILE_SHA256 = checksum_sha
            try:
                result = bootstrap.materialize(
                    capsule=archive,
                    capsule_sha256=archive_sha,
                    destination=root / "destination",
                    record=root / "record.json",
                    expected_runtime_id=bootstrap.EXPECTED_RUNTIME_ID,
                    expected_manifest_sha256=manifest_sha,
                )
            finally:
                bootstrap.EXPECTED_MANIFEST_SHA256 = old_manifest_sha
                bootstrap.EXPECTED_CHECKSUM_FILE_SHA256 = old_checksum_sha
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["payload_entry_count"], 3)
            self.assertTrue((root / "destination/lib/libavutil.so.58").is_symlink())
            self.assertEqual((root / "destination/lib").stat().st_mode & 0o7777, 0o555)
            self.assertEqual((root / "destination/compat").stat().st_mode & 0o7777, 0o555)
            self.assertEqual(json.loads((root / "record.json").read_text())["capsule_sha256"], archive_sha)

    def test_fixture_rejects_escaping_symlink(self) -> None:
        old_manifest_sha = bootstrap.EXPECTED_MANIFEST_SHA256
        old_checksum_sha = bootstrap.EXPECTED_CHECKSUM_FILE_SHA256
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, archive_sha, manifest_sha, checksum_sha = self._fixture(root, escaping=True)
            bootstrap.EXPECTED_MANIFEST_SHA256 = manifest_sha
            bootstrap.EXPECTED_CHECKSUM_FILE_SHA256 = checksum_sha
            try:
                with self.assertRaises(bootstrap.BootstrapError):
                    bootstrap.materialize(
                        capsule=archive,
                        capsule_sha256=archive_sha,
                        destination=root / "destination",
                        record=root / "record.json",
                        expected_runtime_id=bootstrap.EXPECTED_RUNTIME_ID,
                        expected_manifest_sha256=manifest_sha,
                    )
            finally:
                bootstrap.EXPECTED_MANIFEST_SHA256 = old_manifest_sha
                bootstrap.EXPECTED_CHECKSUM_FILE_SHA256 = old_checksum_sha

    def test_pbs_has_single_direct_entry_and_no_sleep(self) -> None:
        source = Path("hpc/pbs/stage45f_pi0163_validation_e_gen1_h200.pbs").read_text(encoding="utf-8")
        self.assertIn("stage45f_pi0163_ffmpeg_capsule_bootstrap.py", source)
        self.assertIn("stage45f_pi0163_runtime_import_probe.py", source)
        self.assertIn("stage45f_pi0159_validation_e_continuation.py", source)
        self.assertIn("stage45f_pi0163_validation_e_recovery.py", source)
        self.assertIn("PI0163_RECOVERY_MODE", source)
        self.assertIn("--pre-submit", source)
        self.assertNotIn("sleep", source)
        self.assertNotIn("qsub", source)

    def test_recovery_plan_is_cpu_static_and_uses_mapping_compatibility(self) -> None:
        import stage45f_pi0163_validation_e_recovery as recovery

        plan = recovery.plan()
        self.assertEqual(plan["status"], "PASS_STATIC_NOT_EXECUTED")
        self.assertEqual(plan["new_calls"], {"conditioning": 30, "action_head": 120, "model_side_total": 150})
        self.assertEqual(plan["reused_raw_queries"], 6)
        self.assertEqual(plan["training"], 0)
        self.assertEqual(plan["holdout_f_read"], 0)
        self.assertFalse(plan["raw_inventory_verified"])

    def test_recovery_aggregate_behaviorally_accepts_six_and_rejects_count_or_order(self) -> None:
        import stage45f_pi0163_validation_e_recovery as recovery
        from go2arx5_vla.evaluation.checkpoint_eval import EvaluationContractError, aggregate_evaluation_rows

        rows = self._aggregate_rows(recovery)
        aggregate = aggregate_evaluation_rows(rows, expected_window_ids=recovery.EXPECTED_WINDOW_IDS)
        self.assertEqual(aggregate["window_count"], 6)
        self.assertEqual(aggregate["window_ids"], list(recovery.EXPECTED_WINDOW_IDS))
        with self.assertRaises(EvaluationContractError):
            aggregate_evaluation_rows(rows[:-1], expected_window_ids=recovery.EXPECTED_WINDOW_IDS)
        wrong_order = list(rows)
        wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
        with self.assertRaises(EvaluationContractError):
            aggregate_evaluation_rows(wrong_order, expected_window_ids=recovery.EXPECTED_WINDOW_IDS)

    def test_raw_inventory_self_verifies_six_preserved_and_thirty_missing(self) -> None:
        import stage45f_pi0163_validation_e_recovery as recovery

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = recovery.raw_inventory(self._allocation8_raw_fixture(root))
            inventory = recovery.validate_allocation8_inventory(inventory)
            self.assertEqual(inventory["preserved_raw_count"], 6)
            self.assertEqual(inventory["missing_raw_count"], 30)
            self.assertEqual(inventory["preserved_cells"][0]["arm"], recovery.EXPECTED_REUSED_ARM)
            self.assertEqual(len(inventory["missing_cells"]), 30)

    def test_isolated_raw_verification_keeps_parent_clean_before_frozen_admission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._allocation8_raw_fixture(root)
            probe = r'''
import json
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
import stage45f_pi0163_validation_e_recovery as recovery

assert "numpy" not in sys.modules
inventory = recovery.validate_allocation8_inventory(recovery.raw_inventory_isolated(Path(sys.argv[2])))
assert "numpy" not in sys.modules
import stage45f_vanda_validation_e_gen1 as evaluator
before = evaluator.admission.model_imports_present()
try:
    evaluator.validate_artifacts({}, {}, "")
except evaluator.ValidationExecutionError as exc:
    frozen_error = str(exc)
else:
    raise AssertionError("frozen validate_artifacts unexpectedly accepted an empty config")
print(json.dumps({
    "preserved_raw_count": inventory["preserved_raw_count"],
    "missing_raw_count": inventory["missing_raw_count"],
    "before_validate_artifacts": before,
    "numpy_in_parent_after_verification": "numpy" in sys.modules,
    "frozen_error": frozen_error,
}, sort_keys=True))
'''
            completed = subprocess.run(
                [sys.executable, "-c", probe, "scripts", str(root)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result["preserved_raw_count"], 6)
            self.assertEqual(result["missing_raw_count"], 30)
            self.assertEqual(result["before_validate_artifacts"], [])
            self.assertFalse(result["numpy_in_parent_after_verification"])
            self.assertEqual(result["frozen_error"], "execution config is missing evaluation code commit")

    def test_frozen_guard_rejects_deliberate_parent_numpy_contamination(self) -> None:
        probe = r'''
import json
import sys

sys.path.insert(0, "scripts")
import numpy
import stage45f_vanda_validation_e_gen1 as evaluator

present = evaluator.admission.model_imports_present()
try:
    evaluator.validate_artifacts({}, {}, "")
except evaluator.ValidationExecutionError as exc:
    frozen_error = str(exc)
else:
    raise AssertionError("frozen validate_artifacts accepted deliberate NumPy contamination")
print(json.dumps({"present": present, "frozen_error": frozen_error}, sort_keys=True))
'''
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertIn("numpy", result["present"])
        self.assertEqual(result["frozen_error"], "model-side module already imported before execution admission")

    def test_fully_preserved_first_arm_is_model_skipped_behaviorally(self) -> None:
        import stage45f_pi0163_validation_e_recovery as recovery

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = recovery.validate_allocation8_inventory(recovery.raw_inventory(self._allocation8_raw_fixture(root)))
            execution_plan = recovery.recovery_execution_plan(inventory)
            self.assertEqual(execution_plan["model_load_attempts"], 5)
            self.assertEqual(execution_plan["new_queries"], 30)
            self.assertEqual(execution_plan["model_loads_by_arm"][recovery.EXPECTED_REUSED_ARM], 0)
            self.assertEqual(execution_plan["forward_cells_by_arm"][recovery.EXPECTED_REUSED_ARM], 0)
            self.assertEqual(execution_plan["forward_cells_by_arm"]["L1(seed2222)@2000"], 6)

    def test_raw_inventory_rejects_extra_array_without_manifest_record(self) -> None:
        import stage45f_pi0163_validation_e_recovery as recovery

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._allocation8_raw_fixture(root)
            extra = root / "raw" / "Legacy_seed2222_at_2000" / "window-06.npy"
            extra.write_bytes(b"extra")
            with self.assertRaises(Exception):
                recovery.raw_inventory(root)

    def test_raw_inventory_rejects_duplicate_manifest_record(self) -> None:
        import stage45f_pi0163_validation_e_recovery as recovery

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._allocation8_raw_fixture(root)
            manifest = root / "raw_predictions.jsonl"
            first = manifest.read_text(encoding="utf-8").splitlines()[0]
            manifest.write_text(manifest.read_text(encoding="utf-8") + first + "\n", encoding="utf-8")
            with self.assertRaises(Exception):
                recovery.raw_inventory(root)

    def test_raw_inventory_rejects_manifest_file_missing_from_directory(self) -> None:
        import stage45f_pi0163_validation_e_recovery as recovery

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._allocation8_raw_fixture(root)
            (root / "raw" / "Legacy_seed2222_at_2000" / "window-05.npy").unlink()
            with self.assertRaises(Exception):
                recovery.raw_inventory(root)

    def test_raw_inventory_rejects_symlink_entry(self) -> None:
        import stage45f_pi0163_validation_e_recovery as recovery

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._allocation8_raw_fixture(root)
            link = root / "raw" / "Legacy_seed2222_at_2000" / "window-06.npy"
            link.symlink_to("window-00.npy")
            with self.assertRaises(Exception):
                recovery.raw_inventory(root)

    def test_raw_inventory_rejects_extra_directory(self) -> None:
        import stage45f_pi0163_validation_e_recovery as recovery

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._allocation8_raw_fixture(root)
            (root / "raw" / "unexpected").mkdir()
            with self.assertRaises(Exception):
                recovery.raw_inventory(root)

    def test_snapshot_dataclass_is_bridged_without_changing_metrics(self) -> None:
        import stage45f_pi0163_validation_e_recovery as recovery

        class Snapshot:
            physical_action = [[1.0]]

        case = self

        class Runtime:
            def _evaluation_row(self, raw, physical, details, window, checkpoint, runtime_sha, raw_path, raw_sha):
                self_snapshot = window["snapshot"]
                case.assertIsInstance(self_snapshot, dict)
                case.assertEqual(self_snapshot["expert_physical_action"], [[1.0]])
                return {"ok": True}

        runtime = Runtime()
        recovery._install_snapshot_mapping_compat(runtime)
        row = runtime._evaluation_row(
            None,
            None,
            {},
            {"snapshot": Snapshot(), "anchor_world_m": [0.0, 0.0, 0.0], "target_world_m": [1.0, 1.0, 1.0]},
            {},
            "runtime",
            "raw.npy",
            "raw-sha",
        )
        self.assertEqual(row, {"ok": True})

    def test_import_probe_has_no_top_level_model_import(self) -> None:
        source = Path("scripts/stage45f_pi0163_runtime_import_probe.py").read_text(encoding="utf-8")
        self.assertNotIn("import torch\n", source)
        self.assertNotIn("from torch", source)
        self.assertIn('importlib.import_module("torch")', source)
        self.assertIn("local_files_only=True", source)


if __name__ == "__main__":
    unittest.main()
