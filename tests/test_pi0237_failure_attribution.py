from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from scripts.pi0237_failure_attribution import (
        DiagnosticBlocker,
        build_cohort,
        parse_rollout_identity,
        _numeric_stats,
    )
except ModuleNotFoundError:  # direct staged-file execution on Hopper
    from pi0237_failure_attribution import (  # type: ignore
        DiagnosticBlocker,
        build_cohort,
        parse_rollout_identity,
        _numeric_stats,
    )


class PI0237PureDiagnosticsTests(unittest.TestCase):
    def test_rollout_identity_is_explicit_and_train_only(self) -> None:
        parsed = parse_rollout_identity(
            "pi0227_persistent_train_a_j07_xp010_yp010_s04_ym050_r03_r03_227156"
        )
        self.assertEqual(parsed["target_family"], "a")
        self.assertEqual(parsed["jitter_family"], "j07_xp010_yp010")
        self.assertEqual(parsed["initial_state"], "s04_ym050")
        with self.assertRaises(DiagnosticBlocker):
            parse_rollout_identity("pi0227_persistent_final_f_j00_nominal_s00_c0_r00")

    def test_fixed_cohort_is_deterministic_and_unique(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "meta").mkdir()
            rows = []
            episode_index = 0
            jitter_names = ("j00_nominal", "j01_xp015", "j02_xm015", "j03_yp015")
            for target in "abcd":
                for rollout_index, jitter_name in enumerate(jitter_names):
                    rollout = (
                        f"pi0227_persistent_train_{target}_{jitter_name}_"
                        f"s00_c0_r0{rollout_index}_r0{rollout_index}_{episode_index:06d}"
                    )
                    for offset in (0, 1, 2, 3):
                        rows.append(
                            {
                                "episode_index": episode_index,
                                "length": 40,
                                "source_episode_id": f"{rollout}:obs-{offset:04d}",
                            }
                        )
                        episode_index += 1
            (root / "meta" / "episodes.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            first, _ = build_cohort(root, "TRAIN", expected_rollouts=16)
            second, _ = build_cohort(root, "TRAIN", expected_rollouts=16)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 64)
            self.assertEqual(len({row["source_episode_id"] for row in first}), 64)

    def test_missing_numeric_values_remain_null(self) -> None:
        summary = _numeric_stats([None, float("nan")])
        self.assertEqual(summary["count"], 0)
        self.assertIsNone(summary["p50"])


if __name__ == "__main__":
    unittest.main()
