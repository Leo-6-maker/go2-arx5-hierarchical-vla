"""The model-only HF restore must admit only the requested checkpoint step."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class RestoreStepTest(unittest.TestCase):
    def test_requested_step_is_required(self):
        class ReachedDownload(Exception):
            pass

        hub = types.ModuleType("huggingface_hub")
        hub.hf_hub_download = lambda **_kwargs: (_ for _ in ()).throw(ReachedDownload())
        script = Path(__file__).resolve().parents[1] / "scripts/pi0245_restore_inference_model.py"
        spec = importlib.util.spec_from_file_location("pi0245_restore_step_test", script)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"huggingface_hub": hub}):
            spec.loader.exec_module(module)
        names = ("config.json", "embodiment_id.json", "processor_config.json", "statistics.json",
                 "model.safetensors.index.json", "model-1.safetensors", "model-2.safetensors",
                 "experiment_cfg/config.json")
        receipt = {"status": "DURABLE_VERIFIED", "checkpoint_step": 2000,
                   "durable_path": "owner/repo/family/checkpoint-2000",
                   "hf_manifest_commit_id": "a" * 40,
                   "source_record": {"inventory": {"entries": [{"path": name} for name in names]}}}
        with tempfile.TemporaryDirectory() as root, patch.object(Path, "read_text", return_value=json.dumps(receipt)):
            args = ["restore", "--receipt", str(Path(root) / "receipt.json"),
                    "--model-root", str(Path(root) / "model"), "--cache-root", str(Path(root) / "cache")]
            with patch.object(sys, "argv", args), self.assertRaises(AssertionError):
                module.main()
            with patch.object(sys, "argv", args + ["--expected-step", "2000"]), self.assertRaises(ReachedDownload):
                module.main()


if __name__ == "__main__":
    unittest.main()
