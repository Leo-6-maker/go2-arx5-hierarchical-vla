import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
PBS = ROOT / "hpc/pbs/stage45b_full24_fresh_base.pbs"
CPU_PBS = ROOT / "hpc/pbs/stage45b_premodel_cpu_smoke.pbs"
H200_PBS = ROOT / "hpc/pbs/stage45b_premodel_h200_smoke.pbs"
SMOKE = ROOT / "scripts/stage45b_premodel_runtime_smoke.py"


class Stage45BPremodelQualificationTest(unittest.TestCase):
    def test_smoke_scripts_are_bound_to_exact_python_and_no_science(self):
        for path in (CPU_PBS, H200_PBS, PBS):
            text = path.read_text()
            self.assertIn("STAGE45B_PYTHON", text)
            self.assertNotIn("python3 scripts/stage45b_full24_fresh_base_train.py", text)
        smoke = SMOKE.read_text()
        for forbidden in (
            "from gr00t.model",
            "gr00t_run(",
            "optimizer.step",
            "backward(",
            "from_pretrained(",
        ):
            self.assertNotIn(forbidden, smoke)
        self.assertIn("VANDA_HOPPER_STAGE45B_PREMODEL_CPU_PBS_QUALIFIED", CPU_PBS.read_text())
        self.assertIn("VANDA_HOPPER_STAGE45B_PREMODEL_H200_RUNTIME_QUALIFIED", H200_PBS.read_text())

    def test_smoke_resource_bounds_and_no_qsub_v(self):
        cpu = CPU_PBS.read_text()
        h200 = H200_PBS.read_text()
        self.assertRegex(cpu, r"select=1:ngpus=0:ncpus=2:mem=8GB")
        self.assertRegex(cpu, r"walltime=00:10:00")
        self.assertRegex(h200, r"select=1:ngpus=1:ncpus=2:mem=16GB")
        self.assertRegex(h200, r"walltime=00:10:00")
        self.assertNotIn("qsub -V", cpu + h200 + PBS.read_text())

    def test_manifest_and_runner_gate_binding_are_refrozen(self):
        manifest = json.loads(
            (ROOT / "manifests/STAGE45B_FULL24_FRESH_BASE_EXECUTION_MANIFEST_V1.json").read_text()
        )
        self.assertEqual(manifest["pi_gate"], "PI_GATE_STATE_0079")
        self.assertEqual(manifest["repository"]["branch"], "stage45b/full24-fresh-base-execution")
        self.assertIn("stage45b_python", manifest["runtime_paths"])
        self.assertIn("stage45b_premodel_runtime_smoke.py", " ".join(
            str(item["path"]) for item in manifest["execution_files"].values()
        ))
        runner = (ROOT / "scripts/stage45b_full24_fresh_base_train.py").read_text()
        self.assertIn('PI_GATE = "PI_GATE_STATE_0079"', runner)
        self.assertEqual(len(re.findall(r"checkpoint-1000", runner)), 1)
        self.assertEqual(len(re.findall(r"checkpoint-2000", runner)), 1)
