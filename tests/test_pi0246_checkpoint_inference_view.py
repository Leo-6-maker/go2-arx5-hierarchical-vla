import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.pi0246_checkpoint_inference_view import make_view
from scripts.pi0245_restore_inference_model import validate_model_index


class CheckpointInferenceViewTests(unittest.TestCase):
    def test_model_only_inference_view_accepts_four_file_checkpoint_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"layer.a": "model-00001-of-00002.safetensors",
                               "layer.b": "model-00002-of-00002.safetensors"}
            }))
            entries = [
                {"path": "config.json"},
                {"path": "model.safetensors.index.json"},
                {"path": "model-00001-of-00002.safetensors"},
                {"path": "model-00002-of-00002.safetensors"},
            ]
            validate_model_index(root, entries)

    def test_model_index_rejects_a_missing_shard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"layer.a": "model-00001-of-00002.safetensors",
                               "layer.b": "model-00002-of-00002.safetensors"}
            }))
            with self.assertRaisesRegex(RuntimeError, "exactly the indexed model shards"):
                validate_model_index(root, [
                    {"path": "config.json"},
                    {"path": "model.safetensors.index.json"},
                    {"path": "model-00001-of-00002.safetensors"},
                ])

    def test_view_binds_archived_model_files_without_mutating_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint-1000"
            checkpoint.mkdir()
            files = {
                "config.json": b"{}",
                "embodiment_id.json": b"{}",
                "processor_config.json": b"{}",
                "statistics.json": b"{}",
                "model.safetensors.index.json": b"{}",
                "model-00001.safetensors": b"one",
                "model-00002.safetensors": b"two",
            }
            entries = []
            for name, content in files.items():
                (checkpoint / name).write_bytes(content)
                entries.append({"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
            entries.sort(key=lambda row: row["path"])
            tree = hashlib.sha256()
            for row in entries:
                tree.update(f"pi0241/checkpoint-1000/{row['path']}\0{row['bytes']}\0{row['sha256']}\n".encode())
            manifest = {
                "family": "PI0245_C_NATIVE_M16_2K_V1",
                "run_card_sha256": "a" * 64,
                "checkpoints": {"1000": {"entries": entries, "file_count": len(entries), "tree_sha256": tree.hexdigest()}},
            }
            manifest_path = root / "training_manifest.json"
            manifest_path.write_text(json.dumps(manifest))

            view = root / "views/C1000"
            record = make_view(checkpoint, manifest_path, 1000, view)

            self.assertEqual(record["status"], "PASS_CHECKPOINT_ARCHIVE_READBACK")
            self.assertEqual(record["inference_files"], len(files))
            self.assertTrue((view / "model-00001.safetensors").is_symlink() or
                            os.path.samefile(view / "model-00001.safetensors", checkpoint / "model-00001.safetensors"))
            self.assertTrue(os.path.samefile(view / "model-00001.safetensors", checkpoint / "model-00001.safetensors"))
            restore = json.loads((view / "pi0245_inference_restore.json").read_text())
            self.assertEqual(restore["schema"], "PI0246_VERIFIED_FULL_TRAINER_CHECKPOINT_INFERENCE_VIEW_V1")
            self.assertFalse((checkpoint / "pi0245_inference_restore.json").exists())


if __name__ == "__main__":
    unittest.main()
