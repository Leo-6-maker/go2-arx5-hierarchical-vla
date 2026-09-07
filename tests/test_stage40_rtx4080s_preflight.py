from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "stage40_rtx4080s_preflight.py"
SPEC = importlib.util.spec_from_file_location("stage40_rtx4080s_preflight", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


class Stage40Rtx4080SPreflightTest(unittest.TestCase):
    def test_canonical_tree_matches_stage38_record_format(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_bytes(b"{}\n")
            (root / "nested").mkdir()
            (root / "nested" / "weights.bin").write_bytes(b"weights")
            expected_entries = []
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    payload = path.read_bytes()
                    expected_entries.append(
                        (path.relative_to(root).as_posix(), len(payload), hashlib.sha256(payload).hexdigest())
                    )
            digest = hashlib.sha256()
            for relative, size, sha256 in expected_entries:
                digest.update(
                    f"{PREFLIGHT.STRUCTURED_CHECKPOINT_TREE_PREFIX}/{relative}\0{size}\0{sha256}\n".encode()
                )
            observed, entries = PREFLIGHT.canonical_tree_sha256(root)
            self.assertEqual(observed, digest.hexdigest())
            self.assertEqual(len(entries), 2)

    def test_checkpoint_symlink_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "real").write_bytes(b"immutable")
            try:
                (root / "link").symlink_to(root / "real")
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            result = PREFLIGHT.audit_checkpoint_tree(root)
            self.assertEqual(result["status"], "BLOCKED")
            self.assertIn("symlink", result["error"])

    def test_missing_checkpoint_and_statistics_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = PREFLIGHT.run_preflight(
                checkpoint=root / "checkpoint-1000",
                structured_stats=root / "structured_action_statistics.json",
                wbc_archive=root / "wbc.zip",
                groot_source=root / "groot",
            )
            self.assertEqual(result["status"], "BLOCKED_PROVENANCE")
            self.assertFalse(result["execution"]["model_constructed"])
            self.assertFalse(result["execution"]["forward"])
            self.assertFalse(result["execution"]["gpu"])
            self.assertFalse(result["checks"]["tiny1_dataset"])
            self.assertFalse(result["checks"]["stage22_modality"])
            self.assertFalse(result["checks"]["cosmos_cache"])

    def test_cosmos_archive_is_not_accepted_as_materialized_runtime_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "cosmos_cache.tar"
            archive.write_bytes(b"archive-only")
            result = PREFLIGHT.audit_cosmos_cache(archive)
            self.assertEqual(result["status"], "BLOCKED")
            self.assertEqual(result["error"], "archive_only_materialization_required")

    def test_modality_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            modality = Path(temporary) / "go2_arx5_modality.py"
            modality.write_text("# wrong bytes\n", encoding="utf-8")
            result = PREFLIGHT.audit_modality(modality)
            self.assertEqual(result["status"], "BLOCKED")
            self.assertFalse(result["hash_match"])

    def test_preflight_source_has_no_model_or_simulator_import(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotRegex(source, r"(?m)^\s*(?:from|import)\s+(?:torch|transformers|isaacgym)\b")
        self.assertNotIn("AutoModel.from_pretrained", source)

    def test_frozen_contract_is_explicit(self) -> None:
        contract = PREFLIGHT.frozen_contract()
        self.assertEqual(contract["representation"], "StructuredActionV1")
        self.assertEqual(contract["geometry"], "K16")
        self.assertEqual(contract["timing"], "T16")
        self.assertEqual(contract["horizon"], 40)
        self.assertEqual(contract["universal_action_dim"], 132)
        self.assertEqual(contract["development_seed"], 4242)


if __name__ == "__main__":
    unittest.main()
