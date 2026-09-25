import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pi0246_recover_pt_closeout.py"
SPEC = importlib.util.spec_from_file_location("pi0246_recover_pt_closeout", SCRIPT)
CLOSEOUT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLOSEOUT)


def tree_hash(step, entries):
    digest = hashlib.sha256()
    for row in sorted(entries, key=lambda item: item["path"]):
        digest.update(f"pi0243/PT_FAMILY/checkpoint-{step}/{row['path']}\0{row['bytes']}\0{row['sha256']}\n".encode())
    return digest.hexdigest()


class PTCloseoutTests(unittest.TestCase):
    def test_accepts_complete_nine_file_trainer_archive_and_checks_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = CLOSEOUT.REQUIRED_TRAINER_FILES | {
                "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
            }
            index = {"weight_map": {
                "weight.a": "model-00001-of-00002.safetensors",
                "weight.b": "model-00002-of-00002.safetensors",
            }}
            (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
            entries = []
            for name in sorted(names):
                path = root / name
                if name != "model.safetensors.index.json":
                    path.write_bytes(name.encode())
                entries.append({"path": name, "bytes": path.stat().st_size, "sha256": CLOSEOUT.sha256(path)})
            record = {"global_step": 1000, "checkpoint_step": 1000, "max_steps": 2000,
                      "inventory": {"file_count": 9, "total_bytes": sum(row["bytes"] for row in entries),
                                    "tree_sha256": tree_hash(1000, entries), "entries": entries}}
            self.assertEqual(CLOSEOUT.validate_full_trainer_inventory(
                record, 1000, root / "model.safetensors.index.json", tree_hash,
            )["file_count"], 9)
            record["inventory"]["tree_sha256"] = "0" * 64
            with self.assertRaises(CLOSEOUT.CloseoutError):
                CLOSEOUT.validate_full_trainer_inventory(
                    record, 1000, root / "model.safetensors.index.json", tree_hash,
                )

    def test_retired_checkpoint_allows_only_identical_nontrainer_sidecars(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "processor" / "processor_config.json"
            source.parent.mkdir()
            source.write_text("{\"ok\":true}\n", encoding="utf-8")
            residual = root / "checkpoint-1000"
            residual.mkdir()
            (residual / "processor_config.json").write_bytes(source.read_bytes())
            kept = CLOSEOUT.validate_retired_payload(root, 1000, {"optimizer.pt"})
            self.assertEqual([row["path"] for row in kept], ["processor_config.json"])
            (residual / "optimizer.pt").write_text("payload", encoding="utf-8")
            with self.assertRaises(CLOSEOUT.CloseoutError):
                CLOSEOUT.validate_retired_payload(root, 1000, {"optimizer.pt"})


if __name__ == "__main__":
    unittest.main()
