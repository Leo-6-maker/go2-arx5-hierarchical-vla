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
        self.assertNotIn("sleep", source)
        self.assertNotIn("qsub", source)

    def test_import_probe_has_no_top_level_model_import(self) -> None:
        source = Path("scripts/stage45f_pi0163_runtime_import_probe.py").read_text(encoding="utf-8")
        self.assertNotIn("import torch\n", source)
        self.assertNotIn("from torch", source)
        self.assertIn('importlib.import_module("torch")', source)
        self.assertIn("local_files_only=True", source)


if __name__ == "__main__":
    unittest.main()
