"""Frozen V4 policy panel on already closed V2 observations. No actuator interface."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from policy_query import input_identity, tensor_sha as tensor_sha
from demonstrated_grip import events, pin, read, require, SOURCE_T

REPORT = Path(__file__).resolve().parents[2] / "reports/SLA_HIERARCHICAL_V16"
REG = REPORT / "FEEDBACK_POLICY_PANEL_REGISTRATION.json"
INPUTS = REPORT / "OWNED_FEEDBACK_ACTUAL.json"
DATA = "v16/feedback-observation02"


def write_new(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def inputs(root):
    plan = read(REG)
    history = read(pin(INPUTS, plan["feedback_sha256"]))
    require(
        history["terminal_query_refused"] and len(history["windows"]) == 4,
        "incomplete owned session",
    )
    windows = {w["tick"]: w for w in history["windows"]}
    parked = np.float32(windows[5703]["measured_input_fraction"])
    prepared = {}
    for key, tick, mode in plan["cohort"]:
        r = windows[tick]
        with np.load(
            pin(root / DATA / r["file"], r["input_sha256"]), allow_pickle=False
        ) as z:
            require(
                set(z) == {"head", "right_wrist", "state", "prompt"},
                "unexpected observation channel",
            )
            state = z["state"].copy()
            require(
                state.shape == (16,)
                and state.dtype == np.float32
                and state[15] == np.float32(r["measured_input_fraction"])
                and str(z["prompt"]) == history["task_text"],
                "wrong feedback or task",
            )
            if mode == "parked_width":
                require(state[15] != parked, "empty width counterfactual")
                state[15] = parked
            else:
                require(mode == "observed", "unknown condition")
            obs = dict(
                images=dict(
                    cam_high=z["head"],
                    cam_left_wrist=np.zeros((480, 640, 3), np.uint8),
                    cam_right_wrist=z["right_wrist"],
                ),
                state=state,
                prompt=str(z["prompt"]),
            )
        ident, fields = input_identity(obs)
        prepared[key] = (
            obs,
            dict(
                id=key,
                tick=tick,
                mode=mode,
                input_sha256=ident,
                archive_sha256=r["input_sha256"],
                field_sha256=fields,
            ),
        )
    require(
        len(prepared)
        == len({s["input_sha256"] for _, s in prepared.values()})
        == plan["max_forwards"]
        == 6,
        "wrong scientific population",
    )
    for label, other in (
        ("CLOSED", "CLOSED_PARKED_WIDTH"),
        ("CLOSED_HOLD", "HOLD_PARKED_WIDTH"),
    ):
        a, sa = prepared[label]
        b, sb = prepared[other]
        require(
            np.flatnonzero(a["state"] != b["state"]).tolist() == [15]
            and [
                k
                for k in sa["field_sha256"]
                if sa["field_sha256"][k] != sb["field_sha256"][k]
            ]
            == ["state"],
            "nonisolated width",
        )
    return prepared


def arm_diff(a, b):
    """First argument minus second, both 50x8 absolute arm trajectories."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    require(
        a.shape == b.shape == (50, 8) and np.isfinite([a, b]).all(), "non-finite arm"
    )
    for v in (a, b):
        require(
            np.max(abs(np.linalg.norm(v[:, 3:7], axis=1) - 1)) <= 1e-4
            and np.all((v[:, 7] >= 0) & (v[:, 7] <= 1)),
            "invalid raw output",
        )
    pos = 1000 * np.linalg.norm(a[:, :3] - b[:, :3], axis=1)
    qa, qb = (Rotation.from_quat(v[:, [4, 5, 6, 3]]) for v in (a, b))
    ang = np.rad2deg((qa * qb.inv()).magnitude())
    d = a[:, 7] - b[:, 7]

    def rms(v):
        return float(np.sqrt(np.mean(v**2)))

    return dict(
        position_rms_mm=rms(pos),
        position_max_mm=float(pos.max()),
        angle_rms_deg=rms(ang),
        angle_max_deg=float(ang.max()),
        width_rms=rms(d),
        width_max_abs=float(max(abs(d))),
        width_rms_mm=40 * rms(d),
        width_max_abs_mm=40 * float(max(abs(d))),
        width_signed_mean_mm=40 * float(d.mean()),
        width_signed_min_mm=40 * float(d.min()),
        width_signed_max_mm=40 * float(d.max()),
        first_events=events(SOURCE_T, a[:, 7], step=False),
        second_events=events(SOURCE_T, b[:, 7], step=False),
    )


def official_action_codec(root):
    from checkpoint_inputs import configured_transforms
    from openpi import transforms

    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("JAX_PLATFORMS") == "cpu",
        "CPU only",
    )
    cfg, data, stats = configured_transforms(root)
    norm = transforms.Normalize(stats, use_quantiles=data.use_quantile_norm)
    encode = transforms.compose([*data.data_transforms.inputs, norm])
    decode = transforms.compose(
        [
            *data.model_transforms.outputs,
            transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm),
            *data.data_transforms.outputs,
        ]
    )
    return cfg, encode, decode


def cpu_control(root, output):
    from openpi import transforms

    cfg, encode, decode = official_action_codec(root)
    plans = inputs(root)
    pins = read(
        pin(
            root / "v13/run01/S0_ACTUAL.json",
            "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9",
        )
    )["inputs"]
    path = root / "v12/A4_P2_ABI.json"
    samples = read(pin(path, pins[str(path)]))["windows"]
    cap = root / "v12/A2_O2_DENSE_CAPTURE.json"
    live = read(pin(cap, pins[str(cap)]))["windows"]
    controls = []
    for entry in samples:
        k = entry["window"]
        with np.load(
            pin(root / "v12/A4_ABI" / (k + ".npz"), entry["chain_sha256"]),
            allow_pickle=False,
        ) as z:
            native = z["raw16"].copy()
        with np.load(
            pin(root / "v12/STUDENT_INPUT" / (k + ".npz"), live[k]["input_sha256"]),
            allow_pickle=False,
        ) as z:
            state = z["state"]
        cams = {n: np.zeros((4, 4, 3), np.uint8) for n in plans["PARKED"][0]["images"]}
        enc = encode(
            dict(
                images=cams,
                state=state,
                actions=native.copy(),
                prompt="offline transform check",
            )
        )

        def inverse(s, actions):
            return decode(dict(state=s.copy(), actions=actions.copy()))["actions"]

        actual = inverse(enc["state"], enc["actions"])
        rt = [arm_diff(actual[:, j : j + 8], native[:, j : j + 8]) for j in (0, 8)]
        require(
            all(
                d["position_max_mm"] < 0.001
                and d["angle_max_deg"] < 0.001
                and d["width_max_abs"] < 1e-6
                for d in rt
            ),
            "archived action roundtrip",
        )
        # Exercise the deployed action_dim and float32 boundary as well as the
        # native 14-D double precision roundtrip. These are reconstructed controls.
        padded = transforms.PadStatesAndActions(cfg.model.action_dim)(
            dict(state=enc["state"].copy(), actions=enc["actions"].copy())
        )
        st = padded["state"].astype(np.float32)
        acts = padded["actions"].astype(np.float32)
        decoded32 = inverse(st, acts).astype(np.float32)
        error32 = [
            arm_diff(decoded32[:, j : j + 8], native[:, j : j + 8]) for j in (0, 8)
        ]
        require(
            all(
                d["position_max_mm"] < 0.001
                and d["angle_max_deg"] < 0.001
                and d["width_max_abs"] < 1e-6
                for d in error32
            ),
            "padded sampler ABI roundtrip",
        )
        conditions = []
        for label in ("PARKED", "CLOSED", "CLOSED_HOLD", "REOPEN"):
            s = state.copy()
            s[15] = plans[label][0]["state"][15]
            other = encode(
                dict(images=cams, state=s, prompt="offline transform check")
            )["state"]
            require(
                np.allclose(other[:13], enc["state"][:13], rtol=0, atol=0),
                "changed other model state",
            )
            decoded = inverse(other, enc["actions"])
            require(
                np.array_equal(decoded, actual), "decoder introduced a width effect"
            )
            altered = st.copy()
            altered[13] = other[13]
            require(
                np.array_equal(inverse(altered, acts).astype(np.float32), decoded32),
                "padded decoder introduced a width effect",
            )
            conditions.append(
                dict(
                    state=label,
                    normalized_width=float(other[13]),
                    identical_output=True,
                )
            )
        controls.append(
            dict(
                id=k,
                archived_raw_sha256=entry["raw_sha256"],
                roundtrip=rt,
                float32_padded_roundtrip=error32,
                conditions=conditions,
            )
        )
    require(len(controls) == 16, "incomplete transform archive")
    write_new(
        output,
        dict(
            inputs=[v[1] for v in plans.values()],
            controls=controls,
            sites=16,
            knots_per_trajectory=50,
            decode_conditions_per_window=4,
            model_action_dim=cfg.model.action_dim,
            new_forwards=0,
            native_starts=0,
            training_updates=0,
        ),
    )


def call_population(log, ordered):
    entered, returned = [], []
    pending = None
    failed = False
    complete = False
    for r in log:
        event = r["event"]
        require(not complete and not failed, "event after terminal outcome")
        if event == "FORWARD_ENTERED":
            require(
                pending is None
                and len(entered) < len(ordered)
                and r["id"] == ordered[len(entered)],
                "unexpected call",
            )
            pending = r
            entered.append(r)
        elif event in ("FORWARD_RETURNED", "FORWARD_FAILED"):
            require(
                pending is not None and r["id"] == pending["id"], "unmatched return"
            )
            if event == "FORWARD_RETURNED":
                require(
                    all(r[k] == pending[k] for k in ("input_sha256", "seed")),
                    "return identity changed",
                )
                returned.append(r)
            else:
                failed = True
            pending = None
        elif event == "PANEL_RETURNED":
            require(
                pending is None and len(returned) == len(ordered) == r["count"],
                "early panel close",
            )
            complete = True
        else:
            require(
                event in ("LOAD_ENTERED", "LOAD_RETURNED") and not entered,
                "unexpected lifecycle event",
            )
    return entered, returned, complete


def inspect_results(root, folder, output):
    from checkpoint_inputs import configured_transforms
    from openpi import transforms

    plans = inputs(root)
    owner_exit = read(folder / "OWNER_EXIT.json")
    start = read(folder / "START.json")
    require(
        start["registration_sha256"] == hashlib.sha256(REG.read_bytes()).hexdigest(),
        "wrong run registration",
    )
    path = folder / "calls.jsonl"
    log = (
        [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    )
    enters, returns, complete = call_population(log, list(plans))
    require(not complete or owner_exit["exit_code"] == 0, "inconsistent owner exit")
    cfg, data, stats = configured_transforms(root)
    norm = transforms.Normalize(stats, use_quantiles=data.use_quantile_norm)
    prepare = transforms.compose(
        [*data.data_transforms.inputs, norm, *data.model_transforms.inputs]
    )
    decode = transforms.compose(
        [
            *data.model_transforms.outputs,
            transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm),
            *data.data_transforms.outputs,
        ]
    )
    vals = {}
    summaries = {}
    for r in returns:
        require(
            r["input_sha256"] == plans[r["id"]][1]["input_sha256"]
            and r["seed"] == read(REG)["seed"],
            "call identity",
        )
        with np.load(
            pin(folder / r["file"], r["file_sha256"]), allow_pickle=False
        ) as z:
            # Importing the official config registers JAX's numerical dtypes.
            raw = z["actions"].astype(str(z["actions_dtype"]))
            absolute = z["absolute"]
            state = z["sampler_state"].astype(str(z["sampler_state_dtype"]))
            require(
                raw.ndim == 2
                and raw.shape[0] == 50
                and raw.shape[1] >= 14
                and absolute.shape == (50, 16)
                and np.isfinite(raw).all()
                and np.isfinite(state).all(),
                "unscorable return",
            )
            require(
                raw.shape == (50, cfg.model.action_dim)
                and state.shape == (cfg.model.action_dim,),
                "sampler ABI",
            )
            reference_state = prepare(plans[r["id"]][0])["state"]
            require(
                np.array_equal(state, reference_state.astype(state.dtype)),
                "sampler saw different state",
            )
            # The official external adapter converts the returned arm arrays to float32.
            reproduced = decode(dict(state=state.copy(), actions=raw.copy()))[
                "actions"
            ].astype(np.float32)
            require(
                np.array_equal(reproduced, absolute),
                "cannot reproduce physical actions from the retained sampler output",
            )
            vals[r["id"]] = (raw, absolute)
            summaries[r["id"]] = dict(
                input_sha256=r["input_sha256"],
                file_sha256=r["file_sha256"],
                raw_shape=list(raw.shape),
                measured_input=float(plans[r["id"]][0]["state"][15]),
                first_opening=float(absolute[0, 15]),
                last_opening=float(absolute[-1, 15]),
                events=events(SOURCE_T, absolute[:, 15], step=False),
                seconds=r["seconds"],
            )
    pairs = []
    for a, b, kind in (
        ("CLOSED", "CLOSED_PARKED_WIDTH", "width_only"),
        ("CLOSED_HOLD", "HOLD_PARKED_WIDTH", "width_only"),
        ("CLOSED", "PARKED", "all_observed_channels"),
        ("CLOSED_HOLD", "CLOSED", "all_observed_channels"),
        ("REOPEN", "CLOSED_HOLD", "all_observed_channels"),
    ):
        if a not in vals or b not in vals:
            continue
        raw, absolute = vals[a]
        ref, babs = vals[b]
        pairs.append(
            dict(
                first=a,
                second=b,
                kind=kind,
                sampler_bitwise_equal=np.array_equal(raw, ref),
                normalized_right_grip_max_abs=float(
                    np.max(abs(raw[:, 13].astype(float) - ref[:, 13].astype(float)))
                ),
                **arm_diff(absolute[:, 8:], babs[:, 8:]),
            )
        )
    write_new(
        output,
        dict(
            calls=summaries,
            pairs=pairs,
            entered=len(enters),
            returned=len(returns),
            episode_count=1,
            registered=len(plans),
            never_entered=len(plans) - len(enters),
            entered_without_return=len(enters) - len(returns),
            completed=complete,
            owner_exit=owner_exit,
            new_forwards=len(enters),
            new_physics_ticks=0,
            training_updates=0,
            student_sla_action_episodes=0,
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("mode", choices=["control", "score"])
    p.add_argument("paths", nargs="+", type=Path)
    a = p.parse_args()
    if a.mode == "control" and len(a.paths) == 1:
        cpu_control(a.root.resolve(), a.paths[0])
    elif a.mode == "score" and len(a.paths) == 2:
        inspect_results(a.root.resolve(), *a.paths)
    else:
        p.error("control needs output; score needs run directory and output")


if __name__ == "__main__":
    main()
