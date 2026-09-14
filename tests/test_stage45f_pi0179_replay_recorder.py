import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "stage45f_pi0179_replay_recorder.py"
SPEC = importlib.util.spec_from_file_location("stage45f_pi0179_replay_recorder", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PI0179ReplayRecorderTests(unittest.TestCase):
    def test_plan_is_execution_free_and_declares_required_outputs(self):
        payload = MODULE.plan()
        self.assertFalse(payload["execution"])
        self.assertFalse(payload["simulator_executed"])
        self.assertFalse(payload["wbc_executed"])
        self.assertEqual(payload["smoke_classification"], "ENGINEERING_REPLAY_SMOKE")
        self.assertIn("third_person.mp4", payload["required_outputs"])
        self.assertIn("predicted_h40", payload["third_person_semantics"])

    def test_frame_requires_exact_h40_and_validates_source_metrics(self):
        with self.assertRaisesRegex(ValueError, "exactly 40"):
            MODULE.ReplayFrameV1(
                timestamp_s=0.0,
                body_frame="body.ppm",
                wrist_frame="wrist.ppm",
                achieved_eef_xyz=(0.0, 0.0, 0.0),
                target_xyz=(1.0, 0.0, 0.0),
                predicted_h40_xyz=((0.0, 0.0, 0.0),),
                model_update_index=0,
                safety_gate="PASS",
            )

    def test_synthetic_smoke_emits_all_outputs_and_false_execution_flags(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg is required for MP4 smoke")
        with tempfile.TemporaryDirectory(prefix="pi0179_test_") as directory:
            output = Path(directory) / "replay"
            frames = MODULE._synthetic_frames(Path(directory) / "source", count=3)
            manifest = MODULE.ReplayRecorderV1(MODULE.ReplayRecorderConfigV1(
                output_dir=output,
                source_identity={"kind": "TEST_SYNTHETIC"},
                synthetic_fixture=True,
                ffmpeg_bin="ffmpeg",
            )).render(frames)
            for name in ("third_person.mp4", "body_camera.mp4", "wrist_camera.mp4", "side_by_side.mp4", "trajectory_overlay.json", "frame_metrics.jsonl", "manifest.json"):
                self.assertTrue((output / name).is_file(), name)
            self.assertEqual(manifest["classification"], "ENGINEERING_REPLAY_SMOKE")
            self.assertFalse(manifest["simulator_executed"])
            self.assertFalse(manifest["wbc_executed"])
            self.assertEqual(len((output / "frame_metrics.jsonl").read_text().splitlines()), 3)
            overlay = json.loads((output / "trajectory_overlay.json").read_text())
            self.assertEqual(len(overlay["frames"]), 3)
            self.assertEqual(len(overlay["frames"][0]["predicted_h40_xyz"]), 40)

    def test_source_timestamp_order_is_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix="pi0179_test_") as directory:
            source = Path(directory)
            frames = list(MODULE._synthetic_frames(source / "source", count=2))
            frames[1] = MODULE.ReplayFrameV1(
                timestamp_s=frames[0].timestamp_s,
                body_frame=frames[1].body_frame,
                wrist_frame=frames[1].wrist_frame,
                achieved_eef_xyz=frames[1].achieved_eef_xyz,
                target_xyz=frames[1].target_xyz,
                predicted_h40_xyz=frames[1].predicted_h40_xyz,
                model_update_index=frames[1].model_update_index,
                safety_gate=frames[1].safety_gate,
            )
            recorder = MODULE.ReplayRecorderV1(MODULE.ReplayRecorderConfigV1(
                output_dir=source / "out", source_identity={"kind": "TEST"}, ffmpeg_bin="ffmpeg"
            ))
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                recorder.render(frames)
