from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/stage29_c2_finalize_existing.py"
SPEC = importlib.util.spec_from_file_location("stage29_finalize", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _metrics(value: float) -> dict:
    summary = {"count": 1, "min": value, "median": value, "mean": value, "max": value}
    profile = {
        "path_length_m": value,
        "linear_speed_mps": summary,
        "second_difference_m": summary,
        "adjacent_direction_cosine": summary,
        "back_and_forth_rate": value,
        "first_difference_dft": {"high_fraction": value},
    }
    return {
        "all_finite": True,
        "physical_action_mse_active_10d": value,
        "eef_position_mse": value,
        "endpoint_position_error_m": value,
        "rotation_error_rad": summary,
        "absolute_gripper_error": summary,
        "predicted_profile": profile,
        "expert_profile": profile,
    }


class ExistingRecordClosureTests(unittest.TestCase):
    def test_exact_six_record_order_is_required(self) -> None:
        rows = []
        for index, (seed, checkpoint) in enumerate(MODULE.EXPECTED_ORDER, start=1):
            rows.append({
                "seed": seed,
                "checkpoint": checkpoint,
                "sampler_call_index": index,
                "conditioning_sha256": "a" * 64,
                "initial_noise": {"raw_sha256": f"{seed:064x}", "finite": True},
                "sampler": {"steps": 4, "rtc": False},
                "metrics": _metrics(float(index)),
                "artifact": {"path": "/does/not/exist", "bytes": 0, "sha256": "b" * 64},
            })
        payload = {"schema": MODULE.stage29.SCHEMA, "status": "PARTIAL_EVIDENCE", "record_count": 6, "records": rows}
        with self.assertRaises(FileNotFoundError):
            MODULE.validate_existing_records(payload)
        payload["records"] = list(reversed(rows))
        with self.assertRaisesRegex(MODULE.FinalizeError, "order drifted"):
            MODULE.validate_existing_records(payload)

    def test_finalizer_has_no_ml_or_execution_entrypoint(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("import transformers", source)
        self.assertNotIn(".get_action(", source)
        self.assertNotIn('["qsub"', source)
        self.assertNotIn("backward(", source)
        self.assertIn("posthoc_model_calls", source)


if __name__ == "__main__":
    unittest.main()
