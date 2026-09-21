import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pi0243", ROOT / "scripts" / "pi0243_staged_learning_to_sim.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PI0243ScheduleTests(unittest.TestCase):
    def test_stage_specs_are_frozen(self):
        self.assertEqual(MODULE.stage_spec("a")["max_steps"], 5000)
        self.assertEqual(MODULE.stage_spec("b")["view"], "64")
        self.assertEqual(MODULE.stage_spec("c")["checkpoints"], [2500, 5000, 7500, 10000])

    def test_schedule_is_deterministic_and_outcome_independent(self):
        rows = [
            {"episode_index": i, "source_episode_id": f"source-{i}", "target_family": "a", "phase": 0}
            for i in range(16)
        ]
        left = MODULE.exposure_schedule(rows, 1000)
        right = MODULE.exposure_schedule(rows, 1000)
        self.assertEqual(left, right)
        self.assertEqual(len(left), 1000)
        self.assertEqual({row["source_episode_id"] for row in left}, {f"source-{i}" for i in range(16)})
        self.assertTrue(set(__import__("collections").Counter(row["fixed_index"] for row in left).values()) <= {62, 63})

    def test_image_conversion_accepts_chw_and_hwc(self):
        import numpy as np

        chw = np.zeros((3, 8, 9), dtype=np.uint8)
        hwc = np.zeros((8, 9, 3), dtype=np.uint8)
        self.assertEqual(MODULE._image_hwc(chw).shape, (8, 9, 3))
        self.assertEqual(MODULE._image_hwc(hwc).shape, (8, 9, 3))

    def test_runtime_card_maps_only_sealed_stage_root(self):
        card = {"paths": {"output_root": "/sealed/stage"}, "cohort": {"path": "/sealed/stage/stage_cohort.json"}, "schedule": {"path": "/sealed/stage/stage_exposure_schedule.json"}}
        old_metadata = os.environ.get("PI0243_METADATA_MIRROR_ROOT")
        old_output = os.environ.get("PI0243_OUTPUT_MIRROR_ROOT")
        os.environ["PI0243_METADATA_MIRROR_ROOT"] = "/home/meta/stage"
        os.environ["PI0243_OUTPUT_MIRROR_ROOT"] = "/scratch/run"
        try:
            mapped = MODULE.runtime_card(card)
        finally:
            if old_metadata is None:
                os.environ.pop("PI0243_METADATA_MIRROR_ROOT", None)
            else:
                os.environ["PI0243_METADATA_MIRROR_ROOT"] = old_metadata
            if old_output is None:
                os.environ.pop("PI0243_OUTPUT_MIRROR_ROOT", None)
            else:
                os.environ["PI0243_OUTPUT_MIRROR_ROOT"] = old_output
        self.assertEqual(Path(mapped["cohort"]["path"]), Path("/home/meta/stage/stage_cohort.json"))
        self.assertEqual(mapped["paths"]["output_root"], "/scratch/run")
        self.assertEqual(card["paths"]["output_root"], "/sealed/stage")


if __name__ == "__main__":
    unittest.main()
