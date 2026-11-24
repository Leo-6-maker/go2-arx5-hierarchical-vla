"""Score the #489 one-variable left wrist mask experiment with retained sampler and geometry."""

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from reach_crossed_seeds import DIGESTS, REPORT, case_inputs, file_sha
from demonstrated_grip import pin, read, require
from feedback_policy_panel import arm_diff, official_action_codec
from reach_intent_replay import standing
from reach_pair_readout import geometry
from compiler import SourceCurve


def score(root):
    out = root / "v18/Z1_MASK_489"
    close = read(out / "EXIT.json")
    require(close["entered"] == close["returned"] == 2, "mask sample accounting")
    original = case_inputs(root)[0][1]
    earlier = read(REPORT / "Z1_ATTRIBUTION/CROSSED_CUSTODY.json")
    feedback = pin(
        root / "v18/Z1_CROSSED_486/cross-1.npz", earlier["members"]["cross-1.npz"]
    )
    rows = {}
    _, _, decode = official_action_codec(root)
    for i, (candidate, base) in enumerate(
        zip(
            (r for r in standing(root) if r["id"].startswith("G01")),
            (original, feedback),
            strict=True,
        )
    ):
        r = read(out / f"returned-{i}.json")
        m = read(out / f"mask-{i}.json")
        require(
            r["input_sha256"] == DIGESTS[i]
            and r["seed"] == [0, 0]
            and m
            == {
                "base_0_rgb": True,
                "left_wrist_0_rgb": False,
                "right_wrist_0_rgb": True,
            },
            "wrong treatment",
        )
        files = (base, pin(out / f"cross-{i}.npz", r["output_sha256"]))
        raw = []
        state = []
        physical = []
        metrics = []
        fixed = candidate["curve"]
        red = np.asarray(candidate["at"]["object_pose"])[:3, 3]
        actual = np.asarray(candidate["at"]["actual_TCP"])[:3, 3]
        axis = np.asarray(candidate["entry"]["direction"])
        for path in files:
            with np.load(path, allow_pickle=False) as z:
                a = z["actions"].astype(str(z["actions_dtype"]))
                s = z["sampler_state"].astype(str(z["sampler_state_dtype"]))
                p = decode({"state": s.copy(), "actions": a.copy()})["actions"].astype(
                    np.float32
                )
                require(np.array_equal(p, z["absolute"]), "decoder output")
            raw.append(a)
            state.append(s)
            physical.append(p)
            delta = SourceCurve(p, fixed.origin, fixed.tool).knots[:16, :3, 3].copy()
            distance = np.linalg.norm(delta - red, axis=1)
            shift = delta - delta[0]
            proj = shift @ axis
            metrics.append(
                {
                    "sha256": file_sha(path),
                    "complete": geometry(
                        p, fixed.origin, fixed.tool, axis, red, actual
                    ),
                    "prefix_0p50_s": {
                        "red_gain_mm": float(1000 * (distance[0] - distance[-1])),
                        "end_axial_mm": float(1000 * proj[-1]),
                        "max_lateral_from_first_mm": float(
                            1000
                            * np.linalg.norm(shift - proj[:, None] * axis, axis=1).max()
                        ),
                    },
                }
            )
        require(np.array_equal(*state), "masked state changed")
        rows[candidate["id"]] = {
            "raw_equal": bool(np.array_equal(*raw)),
            "difference": arm_diff(physical[1][:, 8:], physical[0][:, 8:]),
            "baseline": metrics[0],
            "left_wrist_masked": metrics[1],
        }
    return {
        "rows": rows,
        "frozen_forwards": 2,
        "native_starts": 0,
        "integrations": 0,
        "updates": 0,
    }


if __name__ == "__main__":
    with redirect_stdout(sys.stderr):
        result = score(Path(sys.argv[1]).resolve())
    print(json.dumps(result, indent=2, allow_nan=False))
