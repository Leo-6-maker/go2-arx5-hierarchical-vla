import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


CARD_SHA = "12e09718d5499aece46800b348f733d1bb506400d0468cb128f233f571f75191"


class P1WrapperDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.name != "posix" or not shutil.which("bash"):
            raise unittest.SkipTest("run this dispatch regression in Linux/WSL")
        cls.repo = Path(__file__).resolve().parents[1]
        cls.wrapper = cls.repo / "hpc/pbs/pi0245_p1_stock_h200.pbs"

    def executable(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        path.chmod(0o755)
        return path

    def make_case(self, model_name, fault=None):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        bin_dir = root / "bin"
        run_root = root / "input"
        model_base = root / "models"
        model_root = model_base / model_name
        noise_root = root / "noise"
        stage10 = root / "stage10"
        snapshot = root / "cache/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/rev-test"
        paths = {
            "cosmos_snapshot": str(snapshot), "r0_root": str(run_root),
            "noise_root": str(noise_root), "model_base_root": str(model_base),
            "output_root": str(root / "output"), "code_root": str(root / "code"),
            "stage10_root": str(stage10), "groot_source": str(root / "groot"),
            "p1_runner": str(root / "runner.py"),
            "processor_base_model": str(root / "processor"),
            "train_dataset": str(root / "train"), "modality_config": str(root / "modality.json"),
        }
        for key, value in paths.items():
            path = Path(value)
            if key == "modality_config":
                path.write_text("{}")
            elif key != "p1_runner":
                path.mkdir(parents=True, exist_ok=True)
        (run_root / "pi0243_run_card.json").write_text("{}")
        (run_root / "pi0243_run_card.seal.json").write_text("{}")
        (run_root / "screen/step-10000").mkdir(parents=True)
        (run_root / "screen/step-10000/records.jsonl").write_text("{}\n")
        (noise_root / "manifest.json").write_text("{}")
        (snapshot / "config.json").write_text('{"model_type":"qwen3_vl"}')
        cache_ref = root / "cache/hub/models--nvidia--Cosmos-Reason2-2B/refs/main"
        cache_ref.parent.mkdir(parents=True, exist_ok=True)
        cache_ref.write_text("rev-test")
        runtime = stage10 / "runtime/venv/bin/python"
        self.executable(runtime, "#!/usr/bin/env bash\nset -e\nif [[ \"$1\" == \"-c\" ]]; then echo 'COSMOS_STUB_PASS'; exit 0; fi\nif [[ \"$1\" == \"-u\" ]]; then shift; script=$1; shift; exec python3 \"$script\" \"$@\"; fi\nexit 97\n")

        manifest_files = []
        for name in ("config.json", "model.safetensors.index.json", "model-00001.safetensors", "model-00002.safetensors"):
            payload = ("fixture:" + name).encode()
            path = model_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(payload).hexdigest()
            manifest_files.append({"path": name, "sha256": digest})
            if name == "model-00001.safetensors" and fault == "missing":
                continue
            path.write_bytes(payload)
        if fault == "wrong_hash":
            manifest_files[0]["sha256"] = "0" * 64
        restore = {"optimizer_restored": False, "revision": "fixture-revision", "files": manifest_files}
        (model_root / "pi0245_inference_restore.json").write_text(json.dumps(restore))

        runner = r'''import argparse, hashlib, json, sys
from pathlib import Path
p=argparse.ArgumentParser()
for n in ("r0-root","run-card","path-overlay","noise-root","model-root","model-name","output-root","source-index","noise-name"): p.add_argument("--"+n,required=True)
g=p.add_mutually_exclusive_group(required=True)
for n in ("model-preflight","bind-preflight","label-preflight","preflight","close-p1","all"): g.add_argument("--"+n,action="store_true")
a=p.parse_args(); root=Path(a.model_root)
if root.name != a.model_name: raise SystemExit("model root/name mismatch")
if a.model_preflight:
 m=json.loads((root/"pi0245_inference_restore.json").read_text()); checked=[]
 for row in m["files"]:
  f=root/row["path"]
  if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=row["sha256"]: raise SystemExit("model source drift: "+str(f))
  checked.append(row["path"])
 if "config.json" not in checked or "model.safetensors.index.json" not in checked or sum(x.endswith(".safetensors") for x in checked)<2: raise SystemExit("incomplete model files")
 print("PASS_CURRENT_MODEL_ONLY",a.model_name)
elif a.bind_preflight or a.label_preflight: print("PASS_STUB_NO_MODEL")
'''
        runner_path = Path(paths["p1_runner"])
        runner_path.write_text(runner)
        sha_tool = r'''#!/usr/bin/env python3
import hashlib,os,sys
p=sys.argv[-1]
d=os.environ["TEST_CARD_PATH"]
h=os.environ["TEST_CARD_SHA"] if p==d else hashlib.sha256(open(p,"rb").read()).hexdigest()
print(h,p)
'''
        self.executable(bin_dir / "sha256sum", sha_tool)
        jq_tool = r'''#!/usr/bin/env python3
import json,sys
a=sys.argv[1:]
if a and a[0].startswith("-"): a=a[1:]
key=None
if "--arg" in a:
 i=a.index("--arg"); key=a[i+2]; expr=a[i+3]; path=a[i+4]
else: expr,path=a[0],a[1]
data=json.load(open(path,"r"))
if key is not None: value=data["paths"][key]
elif expr.startswith(".paths."): value=data["paths"][expr.split(".",2)[2]]
elif expr==".source_run_card_sha256": value=data["source_run_card_sha256"]
else: raise SystemExit("unsupported jq expression: "+expr)
print(value)
'''
        self.executable(bin_dir / "jq", jq_tool)
        marker = root / "nvidia-smi-was-called"
        self.executable(bin_dir / "nvidia-smi", f"#!/usr/bin/env bash\ntouch '{marker}'\nexit 99\n")
        self.executable(bin_dir / "mkdir", "#!/usr/bin/env bash\nargs=()\nfor arg in \"$@\"; do [[ \"$arg\" == /scratch/e1192548/pi0245_physical_temporal_pilot_20260924/logs ]] || args+=(\"$arg\"); done\nexec /bin/mkdir \"${args[@]}\"\n")
        overlay = root / "overlay.json"
        overlay.write_text(json.dumps({"source_run_card_sha256": CARD_SHA, "paths": paths}))
        env = os.environ.copy()
        env.update({
            "PATH": str(bin_dir) + os.pathsep + "/usr/bin:/bin",
            "PI0245_MODEL_NAME": model_name, "PI0245_P1_PREFLIGHT_ONLY": "1",
            "PI0245_PATH_OVERLAY": str(overlay), "TEST_CARD_PATH": str(run_root / "pi0243_run_card.json"),
            "TEST_CARD_SHA": CARD_SHA,
        })
        return temporary, env, marker

    def run_case(self, model_name, fault=None):
        temporary, env, marker = self.make_case(model_name, fault)
        try:
            result = subprocess.run(["bash", str(self.wrapper)], cwd=self.repo, env=env, text=True, capture_output=True)
            return result, marker.exists()
        finally:
            temporary.cleanup()

    def test_preflight_dispatch_accepts_only_current_model_manifest(self):
        for model_name in ("R0", "R2"):
            with self.subTest(model=model_name):
                result, marker = self.run_case(model_name)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"PASS_CURRENT_MODEL_ONLY {model_name}", result.stdout)
                self.assertIn("PI0245_P1_PREFLIGHT_ONLY_PASS", result.stdout)
                self.assertFalse(marker)

    def test_preflight_fails_closed_for_missing_file_and_wrong_hash(self):
        for fault in ("missing", "wrong_hash"):
            with self.subTest(fault=fault):
                result, marker = self.run_case("R0", fault)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("model source drift", result.stdout + result.stderr)
                self.assertFalse(marker)


if __name__ == "__main__":
    unittest.main()
