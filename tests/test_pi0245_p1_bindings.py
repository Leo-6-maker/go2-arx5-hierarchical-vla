import inspect
import json
import tempfile
import unittest
from pathlib import Path

from scripts.pi0245_p1_stock import NOISES, SOURCES, close_p1, one, resolve_path_overlay, resolve_tensor_path, verify_model_files


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
                       "paths": {"code_root": "/corelab/code", "r0_root": "/hopper/input",
                                 "noise_root": "/hopper/noise", "model_base_root": "/hopper/models",
                                 "p1_runner": "/hopper/code/p1.py", "output_root": "/hopper/output"}}
            overlay_path.write_text(json.dumps(overlay))
            card = {"_sha256": "a" * 64, "paths": {"code_root": "/hopper/code", "train_dataset": "/hopper/train"}}
            self.assertEqual(resolve_path_overlay(card, overlay_path),
                             {"code_root": "/corelab/code", "train_dataset": "/hopper/train",
                              "r0_root": "/hopper/input", "noise_root": "/hopper/noise",
                              "model_base_root": "/hopper/models", "p1_runner": "/hopper/code/p1.py",
                              "output_root": "/hopper/output"})
            overlay["paths"] = {"seed": "22301"}
            overlay_path.write_text(json.dumps(overlay))
            with self.assertRaises(AssertionError):
                resolve_path_overlay(card, overlay_path)

    def test_stock_cell_uses_explicitly_bound_paths(self):
        self.assertIn("paths", inspect.signature(one).parameters)

    def test_model_restore_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "weights.bin").write_bytes(b"actual bytes")
            (root / "pi0245_inference_restore.json").write_text(json.dumps({
                "optimizer_restored": False,
                "files": [{"path": "weights.bin", "sha256": "0" * 64}],
            }))
            with self.assertRaisesRegex(AssertionError, "model source drift"):
                verify_model_files(root)

    def test_model_restore_missing_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pi0245_inference_restore.json").write_text(json.dumps({
                "optimizer_restored": False,
                "files": [{"path": "missing.safetensors", "sha256": "0" * 64}],
            }))
            with self.assertRaisesRegex(AssertionError, "model source drift"):
                verify_model_files(root)

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
        self.assertIn(".paths.noise_root", text)
        self.assertIn(".paths.p1_runner", text)
        self.assertIn("${PI0245_MODEL_NAME:?set PI0245_MODEL_NAME=R0 or R2}", text)
        self.assertIn("PATH_OVERLAY=${PI0245_PATH_OVERLAY:-/scratch/e1192548/pi0245_physical_temporal_pilot_20260924/p1_path_overlay_r1_hf_cache_v3.json}", text)
        self.assertIn("RUN_ROOT=\"$(jq -er '.paths.r0_root' \"${PATH_OVERLAY}\")\"", text)
        self.assertIn("MODEL_BASE=\"$(jq -er '.paths.model_base_root' \"${PATH_OVERLAY}\")\"", text)
        self.assertIn('test -r "${MODEL_ROOT}/pi0245_inference_restore.json"', text)
        self.assertNotIn('${MODEL_BASE}/R0/pi0245_inference_restore.json', text)
        self.assertNotIn('${MODEL_BASE}/R2/pi0245_inference_restore.json', text)
        self.assertIn('--path-overlay "${PATH_OVERLAY}"', text)
        self.assertEqual(text.count('--path-overlay "${PATH_OVERLAY}"'), 2)
        preflight_only = text.index('if [[ "${PREFLIGHT_ONLY}" == "1" ]]')
        self.assertLess(preflight_only, text.index('nvidia-smi --query-gpu=name'))
        self.assertIn('PI0245_P1_PREFLIGHT_ONLY must be 0 or 1', text)
        self.assertEqual(text.count("--model-name \"${MODEL_NAME}\" --output-root \"${OUT}\" --source-index 0 --noise-name N0 --all"), 1)
        self.assertIn("PI0245_P1_ARCHIVE_PENDING", text)

    def test_single_model_runtime_binding_does_not_require_sibling_restore_manifest(self):
        source = inspect.getsource(__import__("scripts.pi0245_p1_stock", fromlist=["main"]).main)
        self.assertNotIn("model_restore_manifests", source)
        self.assertNotIn("args.model_root.parent", source)

    def test_final_p1_closure_rejects_single_model_even_with_all_its_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows = [{"cell_key": f"R0__{source}__{noise}"} for source in SOURCES for noise in NOISES]
            (output / "queries.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(AssertionError, "identity/order mismatch"):
                close_p1(type("Args", (), {"output_root": output})())


if __name__ == "__main__":
    unittest.main()
