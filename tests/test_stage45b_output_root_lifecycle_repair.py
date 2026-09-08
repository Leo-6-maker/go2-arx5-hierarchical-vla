
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PBS = ROOT / "hpc/pbs/stage45b_full24_fresh_base.pbs"


class Stage45BOutputRootLifecycleRepairTest(unittest.TestCase):
    """NON-SCIENTIFIC / STUB tests for the PBS-to-runner lifecycle boundary."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="stage45b-lifecycle-stub-", dir=ROOT)
        self.sandbox = Path(self.temp_dir.name)
        self.fake_bin = self.sandbox / "fake-bin"
        self.fake_repo = self.sandbox / "fake-repo"
        self.fake_bin.mkdir()
        self.fake_repo.mkdir()
        self.observation = self.sandbox / "stub-observation.json"
        self._write_executable(
            self.fake_bin / "git",
            """#!/bin/sh
set -eu
case "$1" in
  branch)
    test "$2" = "--show-current"
    printf '%s\\n' 'stage45b/full24-fresh-base-execution'
    ;;
  status)
    test "$2" = "--porcelain=v1"
    ;;
  *)
    echo "unexpected fake git invocation: $*" >&2
    exit 90
    ;;
esac
""",
        )
        self._write_executable(
            self.fake_bin / "nvidia-smi",
            """#!/bin/sh
set -eu
printf '%s\\n' 'NVIDIA H200'
""",
        )
        self._write_executable(
            self.fake_bin / "python3",
            """#!/bin/sh
set -eu
test "$#" -ge 1
test "$1" = "scripts/stage45b_full24_fresh_base_train.py"
test ! -e "$STAGE45B_OUTPUT_ROOT"
test -d "$TMPDIR"
case "$TMPDIR/" in
  "$STAGE45B_OUTPUT_ROOT/"*)
    echo "bootstrap TMPDIR is inside output root" >&2
    exit 91
    ;;
esac
test "$TMPDIR" != "$STAGE45B_OUTPUT_ROOT/pbs-tmp"
printf '%s\\n' '{"schema":"STAGE45B_NON_SCIENTIFIC_STUB_V1","mode":"NON_SCIENTIFIC_STUB","output_root_exists":false,"bootstrap_tmp_exists":true,"bootstrap_tmp_outside":true}' > "$STUB_OBSERVATION"
""",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_executable(self, path, content):
        path.write_text(content)
        path.chmod(0o755)

    def _environment(self, output_root, job_id):
        env = os.environ.copy()
        env.update(
            {
                "PATH": os.pathsep.join([str(self.fake_bin), env.get("PATH", os.defpath)]),
                "STAGE45B_REPO": str(self.fake_repo),
                "STAGE45B_BUNDLE": str(self.sandbox / "manifest.json"),
                "STAGE45B_OUTPUT_ROOT": str(output_root),
                "STAGE45B_BASE_MODEL_PATH": str(self.sandbox / "base"),
                "STAGE45B_DATASET_DIR": str(self.sandbox / "dataset"),
                "STAGE45B_GROOT_SOURCE": str(self.sandbox / "groot"),
                "STAGE45B_COSMOS_ROOT": str(self.sandbox / "cosmos"),
                "STAGE45B_MODALITY_CONFIG": str(self.sandbox / "modality.py"),
                "STAGE45B_CONFIG_PATH": str(self.sandbox / "config.yaml"),
                "STAGE45B_STATS_ARTIFACT": str(self.sandbox / "stats.json"),
                "PBS_JOBID": job_id,
                "STUB_OBSERVATION": str(self.observation),
            }
        )
        return env

    def _run_pbs(self, output_root, job_id):
        return subprocess.run(
            ["bash", str(PBS)],
            cwd=self.sandbox,
            env=self._environment(output_root, job_id),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_positive_runner_boundary_uses_external_bootstrap_tmp(self):
        output_root = self.sandbox / "positive-output"
        result = self._run_pbs(output_root, "stub-positive")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(output_root.exists())
        bootstrap = self.sandbox / ".stage45b-pbs-tmp-stub-positive"
        self.assertTrue(bootstrap.is_dir())
        self.assertFalse(bootstrap.is_relative_to(output_root))
        self.assertTrue(self.observation.is_file())
        observation = json.loads(self.observation.read_text())
        self.assertEqual(observation["mode"], "NON_SCIENTIFIC_STUB")
        self.assertFalse(observation["output_root_exists"])
        self.assertTrue(observation["bootstrap_tmp_exists"])
        self.assertTrue(observation["bootstrap_tmp_outside"])
        self.assertIn("HOPPER_STAGE45B_FULL24_FRESH_BASE_TRAINING_COMPLETE", result.stdout)

    def test_negative_preexisting_output_root_fails_before_python3_stub(self):
        output_root = self.sandbox / "negative-output"
        output_root.mkdir()
        sentinel = output_root / "historical-sentinel"
        sentinel.write_text("must remain untouched")
        result = self._run_pbs(output_root, "stub-negative")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_text(), "must remain untouched")
        self.assertFalse(self.observation.exists())
        self.assertFalse((self.sandbox / ".stage45b-pbs-tmp-stub-negative").exists())

    def test_repaired_pbs_preserves_fail_closed_lifecycle_text(self):
        pbs = PBS.read_text()
        dollar_brace = "$" + "{"
        self.assertIn('test ! -e "' + dollar_brace + 'STAGE45B_OUTPUT_ROOT}"', pbs)
        self.assertIn(
            'BOOTSTRAP_TMPDIR="' + dollar_brace + 'OUTPUT_PARENT}/.stage45b-pbs-tmp-' + dollar_brace + 'PBS_JOBID}"',
            pbs,
        )
        self.assertIn('export TMPDIR="' + dollar_brace + 'BOOTSTRAP_TMPDIR}"', pbs)
        self.assertNotIn('mkdir -p "' + dollar_brace + 'STAGE45B_OUTPUT_ROOT}"', pbs)
        self.assertNotIn(
            'export TMPDIR="' + dollar_brace + 'STAGE45B_OUTPUT_ROOT}/pbs-tmp"',
            pbs,
        )


if __name__ == "__main__":
    unittest.main()
