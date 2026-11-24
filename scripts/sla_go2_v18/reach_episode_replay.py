"""Replay measured entry and genuine successor inputs through the fixed episode goal."""

import json
import sys
from pathlib import Path

import numpy as np
from reach_intent_replay import standing
from scripts.pi0292.eval_contract import matrix10
from task_space_intent import ReachEpisode


def replay(root):
    sessions = {}
    results = []
    for r in standing(root):
        episode, at = r["episode"], r["at"]
        obj = r["obj"]
        actual = np.asarray(at["actual_TCP"])
        ref = matrix10(np.asarray(at["target_endpoint_EEF10"]))
        if episode not in sessions:
            if r["tick"] != r["entry"]["tick"]:
                raise ValueError("missing actual entry")
            sessions[episode] = ReachEpisode(episode, actual, ref, obj)
        controller = sessions[episode]
        outputs = {}
        for method in ("GEOMETRY", "DIRECT"):
            try:
                p = controller.plan(
                    episode,
                    actual,
                    ref,
                    obj,
                    method=method,
                    model_world=r["curve"].knots if method == "DIRECT" else None,
                )
                outputs[method] = {
                    "status": "CANDIDATE",
                    "peak_axial_since_entry_mm": float(
                        1000
                        * np.max(
                            (p[:, :3, 3] - controller.ref[:3, 3]) @ controller.axis
                        )
                    ),
                    "endpoint_from_entry_ref_mm": (
                        1000 * (p[-1, :3, 3] - controller.ref[:3, 3])
                    ).tolist(),
                }
            except ValueError as e:
                outputs[method] = {"status": "REFUSED", "reason": str(e)}
        results.append(
            {
                "id": r["id"],
                "entry_tick": r["entry"]["tick"],
                "query_tick": r["tick"],
                "profiles": outputs,
            }
        )
    if len(results) != 3 or len(sessions) != 2:
        raise ValueError("incomplete original native return panel")
    return {"rows": results, "integrations": 0, "new_forwards": 0, "updates": 0}


if __name__ == "__main__":
    print(json.dumps(replay(Path(sys.argv[1]).resolve()), indent=2, allow_nan=False))
