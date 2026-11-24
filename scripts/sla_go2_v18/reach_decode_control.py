"""Re-decode sealed sampler outputs on both observed states, on CPU with the original transforms."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sla_hierarchical_v16"))
from demonstrated_grip import pin, read, require
from feedback_policy_panel import arm_diff, official_action_codec


def compare(root):
    from checkpoint_inputs import configured_transforms
    from openpi import transforms
    from policy_query import input_identity

    _, _, decode = official_action_codec(root)
    _, data, stats = configured_transforms(root)
    prepare = transforms.compose(
        [
            *data.data_transforms.inputs,
            transforms.Normalize(stats, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ]
    )
    pairs = []
    cases = root / "v18/Z1_REACH_483"
    require((cases / "PANEL_CLOSE.json").exists(), "panel still open")

    def apply(st, actions):
        return decode({"state": st.copy(), "actions": actions.copy()})[
            "actions"
        ].astype(np.float32)

    for slot in ("G00-A", "G01-A"):
        case = (cases if slot == "G00-A" else root / "v18/Z1_REACH_483_G01") / slot
        outputs = []
        measured = []
        for q in sorted(case.glob("query-*")):
            with np.load(q / "INPUT.npz", allow_pickle=False) as z:
                obs = {
                    "state": z["state"],
                    "prompt": str(z["prompt"]),
                    "images": {
                        "cam_high": z["head"],
                        "cam_right_wrist": z["wrist"],
                        "cam_left_wrist": np.zeros((480, 640, 3), np.uint8),
                    },
                }
                require(
                    input_identity(obs)[0] == read(q / "REQUEST.json")["input_sha256"],
                    "unbound observed input",
                )
                measured.append(prepare(obs)["state"])
            if not (q / "RETURN.json").exists():
                continue
            binding = read(q / "RETURN.json")
            with np.load(
                pin(q / "OUTPUT.npz", binding["output_sha256"]), allow_pickle=False
            ) as z:
                raw = z["actions"].astype(str(z["actions_dtype"]))
                state = z["sampler_state"].astype(str(z["sampler_state_dtype"]))
                absolute = z["absolute"].copy()
            require(
                np.array_equal(state, measured[-1].astype(state.dtype)),
                "incorrect sampler state",
            )
            reproduced = apply(state, raw)
            require(
                np.array_equal(reproduced, absolute),
                "unmatched official output transform",
            )
            outputs.append((raw, state, absolute))
        row = {
            "slot": slot,
            "exact_native_decodes": len(outputs),
            "matched_state_control": None,
        }
        if len(measured) == 2 and outputs:
            raw0, st0, abs0 = outputs[0]
            baseline = apply(measured[1].astype(st0.dtype), raw0)
            row["old_raw_redecoded_on_second_observation"] = {
                "right_arm_change_against_original": arm_diff(
                    abs0[:, 8:], baseline[:, 8:]
                ),
                "second_observation_has_fresh_forward": len(outputs) == 2,
            }
        if len(outputs) == 2:
            raw0, _, _ = outputs[0]
            raw1, st1, real = outputs[1]
            fixed = apply(st1, raw0)
            row["matched_state_control"] = {
                "equal_raw": bool(np.array_equal(raw0, raw1)),
                "right_arm_same_state": arm_diff(fixed[:, 8:], real[:, 8:]),
                "sampling_note": "Seeds follow the registered [0,0] then [0,1] sequence. This comparison includes observation and sampling changes.",
            }
        pairs.append(row)
    return {
        "official_output_transform": "CPU, exact own-state reconstruction required before counterfactual",
        "cases": pairs,
        "new_forwards": 0,
        "new_ticks": 0,
    }


if __name__ == "__main__":
    out = compare(Path(sys.argv[1]).resolve())
    with Path(sys.argv[2]).open("x") as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write("\n")
