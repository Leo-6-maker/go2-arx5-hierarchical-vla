from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import sys

sys.path.insert(0, "scripts")
import stage45f_pi0163_ffmpeg_capsule_bootstrap as bootstrap


class PI0163SupportTests(unittest.TestCase):
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
        self.assertEqual(plan["new_calls"], {"conditioning": 35, "action_head": 140, "model_side_total": 175})
        self.assertEqual(plan["reused_raw_queries"], 1)
        self.assertEqual(plan["training"], 0)
        self.assertEqual(plan["holdout_f_read"], 0)
        self.assertFalse(plan["raw_inventory_verified"])

    def test_recovery_aggregate_binds_six_validation_e_windows(self) -> None:
        source = Path("scripts/stage45f_pi0163_validation_e_recovery.py").read_text(encoding="utf-8")
        self.assertIn(
            "aggregate_evaluation_rows(action_rows, expected_window_ids=evaluator.WINDOW_IDS)",
            source,
        )

    def test_raw_inventory_self_verifies_one_preserved_and_thirty_five_missing(self) -> None:
        import stage45f_pi0163_validation_e_recovery as recovery

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / recovery.EXPECTED_RAW_RELATIVE_PATH
            raw_path.parent.mkdir(parents=True)
            raw_path.write_bytes(b"raw-fixture")
            record = {
                "schema": recovery.EXPECTED_RAW_SCHEMA,
                "checkpoint": {"label": recovery.EXPECTED_REUSED_ARM, "route": "Legacy", "seed": 2222, "tier": 2000},
                "window_id": recovery.EXPECTED_REUSED_WINDOW,
                "path": recovery.EXPECTED_RAW_RELATIVE_PATH,
                "file_sha256": hashlib.sha256(b"raw-fixture").hexdigest(),
                "shape": list(recovery.EXPECTED_RAW_SHAPE),
                "dtype": "float32",
                "timestep_ids": list(recovery.EXPECTED_TIMESTEPS),
                "action_head_calls_observed": 4,
                "decoded": False,
            }
            (root / "raw_predictions.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            inventory = recovery.raw_inventory(root)
            self.assertEqual(inventory["preserved_raw_count"], 1)
            self.assertEqual(inventory["missing_raw_count"], 35)
            self.assertEqual(inventory["preserved_cells"][0]["arm"], recovery.EXPECTED_REUSED_ARM)
            self.assertEqual(len(inventory["missing_cells"]), 35)

    def test_raw_inventory_rejects_extra_array_without_manifest_record(self) -> None:
        import stage45f_pi0163_validation_e_recovery as recovery

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / recovery.EXPECTED_RAW_RELATIVE_PATH
            raw_path.parent.mkdir(parents=True)
            raw_path.write_bytes(b"raw-fixture")
            extra = root / "raw" / "Legacy_seed2222_at_2000" / "window-01.npy"
            extra.write_bytes(b"extra")
            (root / "raw_predictions.jsonl").write_text("\n", encoding="utf-8")
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
