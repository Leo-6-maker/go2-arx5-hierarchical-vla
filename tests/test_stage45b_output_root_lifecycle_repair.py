
import hashlib
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
        self.exact_python = self.sandbox / "exact-python3"
        self.observation = self.sandbox / "stub-observation.json"
        self.video_qualification = self.sandbox / "stub-video-qualification.json"
        self.video_qualification.write_text("stub-video-qualification\n")
        self.ffmpeg_runtime = self.sandbox / "STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1__stub-runtime-id"
        (self.ffmpeg_runtime / "lib").mkdir(parents=True)
        (self.ffmpeg_runtime / "compat").mkdir()
        self.ffmpeg_manifest = self.ffmpeg_runtime / "STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1.json"
        self.ffmpeg_manifest.write_text("stub-runtime\n")
        self.ffmpeg_manifest_sha256 = hashlib.sha256(self.ffmpeg_manifest.read_bytes()).hexdigest()
        self.torchcodec_root = self.sandbox / "torchcodec"
        self.torchcodec_root.mkdir()
        self.torch_library_root = self.sandbox / "torch-lib"
        self.torch_library_root.mkdir()
        self.runtime_dataset_qualification = self.sandbox / "runtime-dataset-qualification.json"
        self.runtime_dataset_qualification.write_text("stub-runtime-dataset\n")
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
  rev-parse)
    test "$2" = "HEAD"
    printf '%s\\n' 'stub-head'
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
echo ambient-python3-must-not-run >&2
exit 90
""",
        )
        self._write_executable(
            self.exact_python,
            """#!/bin/sh
set -eu
case "$1" in
  */scripts/stage45b_production_release.py)
    shift
    mode=""
    out=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --mode) shift; mode="$1" ;;
        --evidence-path) shift; out="$1" ;;
      esac
      shift
    done
    test "$mode" = "verify"
    test -n "$out"
    printf '%s\n' '{"schema":"STAGE45B_B2_TRAINING_RELEASE_V2","status":"PASS","token":"VANDA_STAGE45B_B2_PRODUCTION_TRAINING_RELEASE_V2_QUALIFIED"}' > "$out"
    ;;
  */scripts/stage45b_groot_source_snapshot.py)
    shift
    mode=""
    out=""
    destination=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --mode) shift; mode="$1" ;;
        --json-output) shift; out="$1" ;;
        --destination) shift; destination="$1" ;;
      esac
      shift
    done
    test "$mode" = "extract"
    test -n "$out"
    test -n "$destination"
    mkdir -p "$destination"
    printf '%s\n' '{"schema":"STAGE45B_GROOT_SOURCE_SNAPSHOT_V2","status":"PASS","stub":true}' > "$out"
    ;;
  */scripts/stage45b_video_decoder_runtime.py)
    shift
    out=""
    qualification_root=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --evidence) shift; out="$1" ;;
        --qualification-root) shift; qualification_root="$1" ;;
      esac
      shift
    done
    test -n "$out"
    test -n "$qualification_root"
    mkdir -p "$qualification_root"
    printf '%s\n' '{"schema":"VANDA_STAGE45B_FULL24_VIDEO_DECODER_RUNTIME_QUALIFIED_V1","status":"PASS","token":"VANDA_STAGE45B_FULL24_VIDEO_DECODER_RUNTIME_QUALIFIED","stub":true}' > "$out"
    ;;
  */scripts/stage45b_premodel_runtime_smoke.py)
    shift
    mode=""
    out=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --mode) shift; mode="$1" ;;
        --evidence-path) shift; out="$1" ;;
      esac
      shift
    done
    test "$mode" = "scientific"
    test -n "$out"
    test ! -e "$STAGE45B_OUTPUT_ROOT"
    test -d "$TMPDIR"
    printf '%s\\n' '{"schema":"VANDA_STAGE45B_PREMODEL_RUNTIME_SMOKE_V1","status":"PASS","mode":"scientific","token":"VANDA_STAGE45B_SCIENTIFIC_PREMODEL_RUNTIME_PASS"}' > "$out"
    ;;
  scripts/stage45b_full24_fresh_base_train.py)
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
    ;;
  *)
    echo "unexpected exact Python invocation: $*" >&2
    exit 92
    ;;
esac
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
                "STAGE45B_GROOT_SNAPSHOT_ARCHIVE": str(self.sandbox / "groot-snapshot.tar"),
                "STAGE45B_GROOT_SNAPSHOT_MANIFEST": str(self.sandbox / "groot-snapshot.manifest.json"),
                "STAGE45B_GROOT_SNAPSHOT_ARCHIVE_SHA256": "0" * 64,
                "STAGE45B_GROOT_SNAPSHOT_MANIFEST_FILE_SHA256": "1" * 64,
                "STAGE45B_GROOT_SNAPSHOT_MANIFEST_PAYLOAD_SHA256": "2" * 64,
                "STAGE45B_GROOT_SNAPSHOT_PAYLOAD_TREE_SHA256": "3" * 64,
                "STAGE45B_PRODUCTION_RELEASE": str(self.sandbox / "production-release.json"),
                "STAGE45B_PRODUCTION_RELEASE_SHA256": "4" * 64,
                "STAGE45B_VIDEO_RUNTIME_QUALIFICATION": str(self.video_qualification),
                "STAGE45B_VIDEO_RUNTIME_QUALIFICATION_SHA256": hashlib.sha256(self.video_qualification.read_bytes()).hexdigest(),
                "STAGE45B_FFMPEG_RUNTIME_ROOT": str(self.ffmpeg_runtime),
                "STAGE45B_FFMPEG_RUNTIME_ID": "stub-runtime-id",
                "STAGE45B_FFMPEG_RUNTIME_MANIFEST_SHA256": self.ffmpeg_manifest_sha256,
                "STAGE45B_TORCHCODEC_PACKAGE_ROOT": str(self.torchcodec_root),
                "STAGE45B_TORCH_LIBRARY_ROOT": str(self.torch_library_root),
                "STAGE45B_RUNTIME_DATASET_QUALIFICATION": str(self.runtime_dataset_qualification),
                "STAGE45B_COSMOS_ROOT": str(self.sandbox / "cosmos"),
                "STAGE45B_MODALITY_CONFIG": str(self.sandbox / "modality.py"),
                "STAGE45B_CONFIG_PATH": str(self.sandbox / "config.yaml"),
                "STAGE45B_STATS_ARTIFACT": str(self.sandbox / "stats.json"),
                "STAGE45B_PYTHON": str(self.exact_python),
                "STAGE45B_PYTHON_REALPATH": str(self.exact_python),
                "STAGE45B_EXECUTION_HEAD": "stub-head",
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
