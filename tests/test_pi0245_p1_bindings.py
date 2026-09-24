import inspect
import json
import tempfile
import unittest
from pathlib import Path

from scripts.pi0245_p1_stock import one, resolve_path_overlay, resolve_tensor_path


class P1BindingTests(unittest.TestCase):
    def test_recorded_tensor_path_resolves_inside_fixed_cell_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cells = root / "screen/step-10000/cells"
            cells.mkdir(parents=True)
            (cells / "train-1.npz").touch()
            resolved = resolve_tensor_path(root, "/scratch/old-root/cells/train-1.npz")
            self.assertEqual(resolved, cells / "train-1.npz")

    def test_rejects_tensor_path_without_cells_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(AssertionError):
                resolve_tensor_path(Path(temporary), "/scratch/old-root/train-1.npz")

    def test_path_overlay_is_card_bound_and_path_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            overlay_path = Path(temporary) / "overlay.json"
            overlay = {"schema": "PI0245_PATH_OVERLAY_V1", "source_run_card_sha256": "a" * 64,
                       "paths": {"code_root": "/corelab/code"}}
            overlay_path.write_text(json.dumps(overlay))
            card = {"_sha256": "a" * 64, "paths": {"code_root": "/hopper/code", "train_dataset": "/hopper/train"}}
            self.assertEqual(resolve_path_overlay(card, overlay_path),
                             {"code_root": "/corelab/code", "train_dataset": "/hopper/train"})
            overlay["paths"] = {"seed": "22301"}
            overlay_path.write_text(json.dumps(overlay))
            with self.assertRaises(AssertionError):
                resolve_path_overlay(card, overlay_path)

    def test_stock_cell_uses_explicitly_bound_paths(self):
        self.assertIn("paths", inspect.signature(one).parameters)

    def test_hopper_wrapper_uses_unix_line_endings(self):
        wrapper = Path(__file__).resolve().parents[1] / "hpc" / "pbs" / "pi0245_p1_stock_h200.pbs"
        contents = wrapper.read_bytes()
        self.assertTrue(contents.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r", contents)

    def test_p1_wrapper_uses_compute_visible_inputs_and_one_model_per_job(self):
        wrapper = Path(__file__).resolve().parents[1] / "hpc" / "pbs" / "pi0245_p1_stock_h200.pbs"
        text = wrapper.read_text()
        self.assertIn("#PBS -W group_list=HPC_Users_3104", text)
        self.assertNotIn("/Project_Storage", text)
        self.assertIn("p1_noise_bank", text)
        self.assertIn("${PI0245_MODEL_NAME:?set PI0245_MODEL_NAME=R0 or R2}", text)
        self.assertEqual(text.count("--model-name \"${MODEL_NAME}\" --output-root \"${OUT}\" --source-index 0 --noise-name N0 --all"), 1)
        self.assertIn("PI0245_P1_ARCHIVE_PENDING", text)


if __name__ == "__main__":
    unittest.main()
