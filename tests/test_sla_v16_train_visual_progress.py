import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts/sla_hierarchical_v16")
)
from train_visual_progress import aggregate, matched_hold_targets, summarize_color


def test_unknown_true_frames_and_false_early_detection_are_retained():
    predictions = []
    queries = []
    for tick, (height, flag) in enumerate(
        ((0, False), (0.019, True), (0.022, None), (0.025, True))
    ):
        pose = np.eye(4)
        pose[2, 3] = height
        queries.append(
            {
                "native_step": tick,
                "actual_state": {
                    "native_step": tick,
                    "objects": {"red": {"pose": pose}},
                },
            }
        )
        predictions.append(
            {"tick": tick, "colors": {"red": {"seen": flag, "status": "measured"}}}
        )
    result = summarize_color(predictions, queries, "red")
    assert result["first_actual"] == 2 and result["first_visual"] == 1
    assert result["false_positive_frames"] == [
        {"tick": 1, "actual_height_delta_mm": 19.0}
    ]
    assert result["confusion"]["abstain_positive"] == 1
    group = aggregate([{"target": result}, {}], "target")
    assert group["episodes"] == 2 and group["scored"] == 1 and group["frames"] == 4
    assert (
        group["detected_while_lifted"]
        == group["first_visual_precedes_lift_or_no_lift"]
        == 1
    )


def test_matched_layout_requires_same_scene_seed_and_reset():
    item = {
        "group": "other_admitted",
        "family": "F1",
        "layout": "A",
        "scene": "scene",
        "reset": 2,
        "seed": 3,
        "target": {},
    }
    data = {"b": dict(item, task="HOLD-blue"), "r": dict(item, task="HOLD-red")}
    assert matched_hold_targets(data)[0]["episodes"] == {
        "HOLD-blue": "b",
        "HOLD-red": "r",
    }
    for key, other in (("scene", "B"), ("reset", 8), ("seed", 12), ("layout", "C")):
        assert matched_hold_targets(dict(data, r={**data["r"], key: other})) == []
