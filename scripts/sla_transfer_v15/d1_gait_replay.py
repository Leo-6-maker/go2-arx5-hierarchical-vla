"""Replay archived frozen leg actions and isolate the current-frame gait inputs on CPU."""

import argparse
import os
import struct
from pathlib import Path

import numpy as np
from d1_readiness import B2_PIN, baseline, pinned, read, record, sha

TRACE = struct.Struct("<QQdIIQQQQQ96d")
SNAPSHOT = struct.Struct("<5Q")
START = 2503
END = 5703
TOL = 2e-5


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def population(root):
    histogram = root / "v15/prefix02/D1_PREFIX_AUDIT_ACTUAL.json"
    pinned(
        histogram, "e86688b55525e857b4d0e15a06918d56c474d7c21d6c1e54e67d99526fb22e96"
    )
    h = read(histogram)["historical_panel"]
    prior = root / "v14/history_qualification/run04/HISTORICAL_TRACKING_ACTUAL.json"
    pinned(prior, h["prior_manifest_sha256"])
    ref = read(root / "v13/run01/S0_ACTUAL.json")
    spec = read(
        next(Path(k) for k in ref["inputs"] if k.endswith("/EXECUTION_SPEC.json"))
    )
    last = Path(spec["mobile_interface_qualification"]["path"]).parents[2]
    panels = [last.with_name("mobile-wbc-sla-boundary-v1-m0-20261115-01"), last]
    base = root / "v14/b2-technical-20261123-01"
    manifest = base / "B2_PUBLIC_READOUT.json"
    pinned(manifest, B2_PIN)
    for row in read(manifest)["cards"]:
        p = base / row["card"]
        pinned(p / "RESULT.private.json", row["result_sha256"])
        yield (
            row["card"],
            "B2 anchor",
            p / read(p / "RESULT.private.json")["episode"],
            row["evidence_sha256"]["init.trace"],
        )
    folder = root / "v15/sla-v15-d1-native-20261123-01/wave_A/wave_A-D1-OO"
    w = root / "v15/readout01/D1_WAVE_A_ACTUAL.json"
    pinned(w, "f7d3a15c394860fe7755dc267702b1d5df9a1ca993bac251baabefe9856572c2")
    oo = read(w)["records"][0]
    yield (
        "D1 OO",
        "D1 initial mismatch",
        folder / read(folder / "RESULT.private.json")["episode"],
        oo["input_sha256"]["init.trace"],
    )
    manifest = read(prior)
    index = {}
    for p, seal in zip(panels, manifest["batch_sha256"], strict=True):
        pinned(p / "RESULT.private.json", seal)
        for f in (p / "runs").glob("*/RESULT.private.json"):
            index[sha(f)] = f.parent
    for row, old in zip(h["cases"], manifest["cases"], strict=True):
        require(row["condition"] == old["condition"], "M0 join")
        c = row["condition"]
        name = "M0 " + c["axis"] + " " + c["eef_mode"]
        if not row["historic_ticks"]:
            yield name, "withheld", None, None
            continue
        n = index[old["result_sha256"]] / "native-01"
        label = (
            ("M0 matched" if row["matched"] else "M0 mismatched")
            if "matched" in row
            else "M0 stopped early"
        )
        yield name, label, n, row.get("trace_sha256", sha(n / "init.trace"))


def evidence(folder, pin):
    path = folder / "init.trace"
    pinned(path, pin)
    stream = {}
    snap = {}
    with path.open("rb") as f:
        header = f.read(16)
        require(
            header[:8] == b"WM02IT01"
            and int.from_bytes(header[8:12], "little") == TRACE.size,
            "trace ABI",
        )
        for r in TRACE.iter_unpack(f.read()):
            if r[3] in (3, 4, 5) and r[2] <= 11.406001:
                token = r[9]
                entry = stream.setdefault(token, {"t": r[2]})
                require(r[3] not in entry, "duplicate inference token")
                entry[r[3]] = r[10 : 10 + r[4]] if r[4] else r[10:13]
    with (folder / "init.trace.snapshots").open("rb") as f:
        while h := f.read(SNAPSHOT.size):
            generation, slot, token, kind, n = SNAPSHOT.unpack(h)
            data = f.read(4 * n)
            require(len(data) == 4 * n, "truncated history")
            if (
                kind == 4
                and token in stream
                and 4.98 - 1e-8 <= stream[token]["t"] <= 5.406 + 1e-8
            ):
                require(
                    stream[token][4] == (slot, generation, n) and n == 2640,
                    "history descriptor",
                )
                snap[token] = np.frombuffer(data, dtype="<f4").reshape(30, 88)
    return stream, snap


def evaluate(net, history):
    import torch

    with torch.inference_mode():
        action = net(torch.from_numpy(history.reshape(1, 2640).copy()))
    y = action.detach().numpy().ravel()
    require(y.shape == (12,) and np.isfinite(y).all(), "invalid leg action")
    return np.clip(y, -10, 10)


def case(net, cfg, name, label, n, pin):
    if n is None:
        return {
            "case": name,
            "group": label,
            "native_boundary": False,
            "CPU_leg_forwards": 0,
        }
    stream, histories = evidence(n, pin)
    rawscale = np.array(cfg["dog_commands_scale"][:3], dtype=np.float32)
    action_scale = np.array(cfg["action_scale"][:12])
    stand_clock = np.sin(
        np.float32(2 * np.pi)
        * np.float32(0.25)
        * np.float32(0.5)
        / np.float32(cfg["gait_duration"])
    )
    active = []
    frames = []
    checked = 0
    maxerr = 0.0
    ablations = []
    for token, entry in sorted(stream.items()):
        t = entry["t"]
        frame = np.asarray(entry[3], dtype=np.float32)
        require(frame.shape == (88,), "current frame missing")
        mag = float(np.linalg.norm(frame[39:42] / rawscale))
        walk = mag >= 0.1
        require(
            np.isclose(frame[45], cfg["gait_frequency"] if walk else 0, atol=1e-6),
            "observed frequency vs compiled gate",
        )
        if not walk:
            require(
                np.allclose(frame[59:63], stand_clock, rtol=0, atol=2e-6),
                "standing clock mismatch",
            )
        if walk:
            active.append(token)
        if not 4.98 - 1e-8 <= t <= 5.406 + 1e-8:
            continue
        hist = histories[token]
        native = np.asarray(entry[5], dtype=np.float32)
        require(
            np.array_equal(
                hist,
                np.array(
                    [stream[i][3] for i in range(token - 29, token + 1)],
                    dtype=np.float32,
                ),
            ),
            "snapshot and inline history differ",
        )
        actual = evaluate(net, hist)
        err = float(np.max(abs(actual - native)))
        require(err <= TOL, "original CPU replay differs")
        checked += 1
        maxerr = max(err, maxerr)
        frames.append(
            {
                "leg_inference_token": token,
                "sim_s": t,
                "scaled_back_command_norm": mag,
                "effective_frequency_Hz": float(frame[45]),
            }
        )
        if not walk:
            continue
        out = {(1, 1): actual}
        for freq, clock in ((0, 0), (0, 1), (1, 0)):
            alt = hist.copy()
            if not freq:
                alt[-1, 45] = 0
            if not clock:
                alt[-1, 59:63] = stand_clock
            out[freq, clock] = evaluate(net, alt)

        def angular(v):
            return float(np.degrees(np.max(abs(v * action_scale))))

        ablations.append(
            {
                "token": token,
                "joint_target_max_abs_deg": {
                    "both_walk_vs_both_stand": angular(actual - out[0, 0]),
                    "frequency_only_at_stand_clock": angular(out[1, 0] - out[0, 0]),
                    "clock_only_at_stand_frequency": angular(out[0, 1] - out[0, 0]),
                    "factor_interaction": angular(
                        out[1, 1] - out[0, 1] - out[1, 0] + out[0, 0]
                    ),
                },
            }
        )
    require(
        histories.keys() == {r["leg_inference_token"] for r in frames},
        "unused or missing history",
    )
    return {
        "case": name,
        "group": label,
        "native_boundary": label != "M0 stopped early",
        "physical_trace_sha256": pin,
        "snapshots_sha256": sha(n / "init.trace.snapshots"),
        "walk_tokens_through_boundary_or_stop": active,
        "replayed_frames": frames,
        "CPU_leg_forwards": checked + 3 * len(ablations),
        "natural_replays": checked,
        "max_native_action_abs_error": maxerr,
        "current_frame_ablations": ablations,
    }


def main():
    import torch
    import yaml

    cli = argparse.ArgumentParser()
    cli.add_argument("root", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    root = args.root.resolve()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU only")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    s = baseline(root)
    frozen = s["frozen_files"]
    files = {Path(k).name: Path(k) for k in frozen}
    for k in ("policy.pt", "config.yaml"):
        pinned(files[k], frozen[str(files[k])])
    cfg = next(iter(yaml.safe_load(files["config.yaml"].read_text()).values()))
    require(
        cfg["num_observations"] == 88
        and cfg["use_dynamic_gait"]
        and cfg["dog_commands_scale"][:3] == [2, 2, 0.25]
        and cfg["arm_num_commands"] == 9
        and cfg["observations_history"] == list(range(29, -1, -1)),
        "wrong layout",
    )
    spec = read(Path(s["native_binary"]).parents[1] / "EXECUTION_SPEC.json")
    objs = Path(spec["baseline_candidate"])
    ident = objs / "CANDIDATE.private.json"
    pinned(ident, "61b1ae2a77c6ff0d38c5c43b0db8fc9a67415d34d4fa732ea83be680fab361f7")
    sources = read(ident)["candidate_sources"]
    for f in ("rl_sim_mujoco.cpp", "rl_sdk.cpp"):
        pinned(objs / f, sources[f])
    pinfile = Path(s["native_binary"]).parent / "RESULT.json"
    pinned(pinfile, "460214e35fed0c067dc78433b18c0b92866135c2ac4ec0f47148044de8263b9a")
    require(
        read(pinfile)["original_controller_and_leg_SDK_objects_reused"]
        and read(pinfile)["binary_sha256"] == sha(Path(s["native_binary"])),
        "runtime provenance",
    )
    net = torch.jit.load(str(files["policy.pt"]), map_location="cpu").eval()
    out = [case(net, cfg, *r) for r in population(root)]
    record(
        args.output,
        {
            "scope": "retrospective input sensitivity, exact recorded histories; effective-frequency and four clock slots of last frame independently replaced by compiled standing values",
            "leg_weights_sha256": sha(files["policy.pt"]),
            "leg_config_sha256": sha(files["config.yaml"]),
            "source_sha256": {
                k: sources[k] for k in ("rl_sim_mujoco.cpp", "rl_sdk.cpp")
            },
            "CPU_runtime": torch.__version__,
            "tolerance_native_replay": TOL,
            "sim_time_replay_s": [4.98, 5.406],
            "new_SLA_forwards": 0,
            "new_optimizer_updates": 0,
            "new_native_requests": 0,
            "new_physics_ticks": 0,
            "new_cpu_leg_policy_evaluations": sum(r["CPU_leg_forwards"] for r in out),
            "cases": out,
        },
    )


if __name__ == "__main__":
    main()
