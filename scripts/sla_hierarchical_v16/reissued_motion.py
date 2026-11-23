"""Cross the old reconstructed action sequence with the new real state and calibration."""

import argparse
import hashlib
from pathlib import Path
import numpy as np

from compiler import SourceCurve
from demonstrated_grip import pin, read, require
from feedback_policy_panel import arm_diff, official_action_codec, write_new
from policy_query import tensor_sha
from paired_motion import REPORTS, contrasted, path_errors


def run(root, measurements):
    from openpi import transforms

    registration = read(REPORTS / "PAIRED_MOTION_DECODER_REGISTRATION.json")
    first = read(
        pin(REPORTS / "PAIRED_MOTION_ACTUAL.json", registration["first_readout_sha256"])
    )
    cfg, enc, dec = official_action_codec(root)
    require(cfg.model.action_dim == 32, "unexpected sampler width")
    pad = transforms.PadStatesAndActions(cfg.model.action_dim)
    index = read(
        pin(
            root / "v13/run01/S0_ACTUAL.json",
            "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9",
        )
    )["inputs"]
    capturefile = root / "v12/A2_O2_DENSE_CAPTURE.json"
    capture = read(pin(capturefile, index[str(capturefile)]))["windows"]
    abi = root / "v12/A4_P2_ABI.json"
    calls = {r["window"]: r for r in read(pin(abi, index[str(abi)]))["windows"]}
    cameras = {
        k: np.zeros((4, 4, 3), np.uint8)
        for k in ("cam_high", "cam_left_wrist", "cam_right_wrist")
    }
    records = {}
    sources = {}

    def decode(state, actions):
        return dec(dict(state=state.copy(), actions=actions.copy()))["actions"].astype(
            np.float32
        )

    for k, cap in capture.items():
        with np.load(
            pin(root / "v12/STUDENT_INPUT" / (k + ".npz"), cap["input_sha256"]),
            allow_pickle=False,
        ) as z:
            state = z["state"]
        with np.load(
            pin(root / "v12/A4_ABI" / (k + ".npz"), calls[k]["chain_sha256"]),
            allow_pickle=False,
        ) as z:
            raw = z["raw16"]
        code = pad(
            enc(
                dict(
                    state=state.copy(),
                    actions=raw.copy(),
                    images=cameras,
                    prompt="offline transform check",
                )
            )
        )
        st, actions = (
            code["state"].astype(np.float32),
            code["actions"].astype(np.float32),
        )
        require(st.shape == (32,) and actions.shape == (50, 32), "padded ABI")
        same = decode(st, actions)
        errors = [arm_diff(same[:, j : j + 8], raw[:, j : j + 8]) for j in (0, 8)]
        maximum = {
            metric: max(e[metric] for e in errors)
            for metric in ("position_max_mm", "angle_max_deg", "width_max_abs")
        }
        require(
            maximum["position_max_mm"] < 0.001
            and maximum["angle_max_deg"] < 0.001
            and maximum["width_max_abs"] < 1e-6,
            "failed roundtrip",
        )
        records[k] = dict(state=st, actions=actions, same=same, original=raw)
        sources[k] = dict(
            input_sha256=cap["input_sha256"],
            raw_sha256=calls[k]["raw_sha256"],
            reconstructed_actions_sha256=tensor_sha(actions),
            roundtrip_max=maximum,
        )

    def poses(raw, k):
        c = capture[k]
        return SourceCurve(raw, c["T_W_source_origin"], c["T_link6_TCP"]).knots

    pairs = []
    with np.load(measurements, allow_pickle=False) as z:
        require(len(z) == 2 * len(first["pairs"]), "wrong cache population")
        for row in first["pairs"]:
            a, b = row["old"], row["new"]
            clock = z[a + "_" + b + "_ticks"]
            truth = z[a + "_" + b + "_tcp"]
            require(
                capture[a]["tick"] + 250 == capture[b]["tick"] == clock[0]
                and list(row["raw_sha256"])
                == [sources[k]["raw_sha256"] for k in (a, b)]
                and np.array_equal(
                    clock, np.arange(clock[0], row["scored_through_tick"] + 1, 50)
                )
                and truth.shape == (len(clock), 4, 4),
                "mismatched truth or action identities",
            )
            i = np.arange(len(clock)) * 3
            old, new = records[a], records[b]
            crossed = decode(new["state"], old["actions"])
            original_old = poses(old["original"], a)[15 + i]
            original_new = poses(new["original"], b)[i]
            paths = dict(
                A=poses(old["same"], a)[15 + i],
                B=poses(old["same"], a)[i],
                C=poses(old["same"], b)[i],
                D=poses(crossed, b)[i],
                E=poses(new["same"], b)[i],
                hold=np.broadcast_to(truth[0], truth.shape),
            )
            result = dict(
                old=a,
                new=b,
                episode=row["episode"],
                phase=row["phases"][1],
                crossed_right_action_sha256=tensor_sha(crossed[:, 8:]),
                query_error_mm={
                    k: float(1000 * np.linalg.norm(v[0, :3, 3] - truth[0, :3, 3]))
                    for k, v in paths.items()
                },
            )
            for label, n in (("primary", 6), ("secondary", 12)):
                result[label] = None
                if len(i) < n:
                    require(row[label] is None, "dropped original horizon")
                    continue
                current = truth[:n]
                for k, raw in (("old", original_old), ("new", original_new)):
                    recovered = path_errors(raw[:n], current)
                    for metric in (
                        "position_rmse_mm",
                        "rotation_rmse_deg",
                        "endpoint_position_mm",
                        "endpoint_rotation_deg",
                    ):
                        require(
                            abs(recovered[metric] - row[label]["outcomes"][k][metric])
                            < 1e-8,
                            "first readout differs",
                        )
                result[label] = dict(
                    outcomes={k: path_errors(v[:n], current) for k, v in paths.items()}
                )
            pairs.append(result)
    comparisons = [
        ("B", "A"),
        ("C", "B"),
        ("D", "C"),
        ("E", "D"),
        ("E", "A"),
        ("D", "hold"),
        ("E", "hold"),
    ]
    summary = {
        h: {
            m: {f"{a}_minus_{b}": contrasted(pairs, h, m, a, b) for a, b in comparisons}
            for m in ("position_rmse_mm", "rotation_rmse_deg")
        }
        for h in ("primary", "secondary")
    }
    return dict(
        prior_readout_sha256=registration["first_readout_sha256"],
        native_measurements_sha256=hashlib.sha256(
            measurements.read_bytes()
        ).hexdigest(),
        sources=sources,
        pairs=pairs,
        summary=summary,
        new_forwards=0,
        new_integrations=0,
        training_updates=0,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("measurements", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    write_new(args.output, run(args.root.resolve(), args.measurements))
