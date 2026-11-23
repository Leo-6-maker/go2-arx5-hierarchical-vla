"""Form causal policy inputs after known V2 student commands on a closed native session."""

import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np

from demonstrated_grip import read, require, pin
from sealed_panel import configure_site

PANEL = {
    5703: "parked",
    5903: "close_finished",
    6103: "closed_hold_finished",
    6303: "reopen_finished",
}


def run(root, repo, out):
    import mujoco
    from OpenGL import GL
    from go2arx5_vla.world_model.init_trace_reader import read_trace
    from go2arx5_vla.world_model.native_camera_transfer_v1 import FixedBodyRenderer
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from observation_stream import ObservationStream
    from observation_visibility import observe_packet, digest
    from verify_grip_card import score, states

    case = root / "v16/sla-v16-student-gripper-native-01b/V2-GRIP-01B"
    previous = read(
        repo / "reports/SLA_HIERARCHICAL_V16/V2_STUDENT_GRIPPER_OWNER_ACTUAL.json"
    )
    validated = score(root, case)
    require(
        validated == previous
        and validated["lease_closed"]
        and validated["ack_count"] == 160,
        "original qualification differs",
    )
    ledger = {r["end_tick"]: r for r in validated["verified_command_intervals"]}
    spec = read(case / "EXECUTION_SPEC.json")
    native = case / read(case / "RESULT.private.json")["episode"]
    authority = read(case / "student/authority.json")
    epoch = authority["epoch"]
    with (case / "student/receiver.csv").open() as f:
        stops = [r for r in csv.reader(f) if "STOP" in r[0]]
    require(
        len(stops) == 1
        and stops[0][0] == "RECEIVER_STOP_AT_CAP"
        and stops[0][1] == epoch,
        "wrong terminal event",
    )
    stop = int(stops[0][2])
    require(stop == 6503, "wrong stop")
    tfile = repo / "reports/SLA_HIERARCHICAL_V16/OBSERVATION_VISIBILITY_ACTUAL.json"
    t02 = next(r for r in read(tfile)["target"] if r["id"] == "T02")
    with np.load(
        pin(root / "v12/STUDENT_INPUT/T02.npz", t02["input_sha256"]), allow_pickle=False
    ) as z:
        prompt = str(z["prompt"])
        virtual_left = z["state"][:8].copy()
    p = root / "v12/TARGET_REGISTRATION.json"
    pins = read(
        pin(
            root / "v13/run01/S0_ACTUAL.json",
            "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9",
        )
    )["inputs"]
    reg = read(pin(p, pins[str(p)]))
    stream = ObservationStream(
        epoch, prompt, virtual_left, reg["source_origin"], reg["T_link6_TCP"]
    )
    need = {0, *[tick - offset * 500 for tick in PANEL for offset in range(9)]}
    with (native / "initialization.jsonl").open() as f:
        init = json.loads(f.readline())
    first = {k: np.asarray(init[k]) for k in ("qpos", "qvel", "ctrl")}
    before = next(x for x in read_trace(native / "init.trace") if x["kind"] == 11)
    require(
        before["sim_time_s"] == init["time"] == 0
        and init["kind"] == "actual_startup_V1"
        and np.array_equal(before["values"][:66], np.r_[first["qpos"], first["qvel"]]),
        "unverified reset pose",
    )
    first.update(step=0, time_s=0.0)
    os.umask(0o077)
    out.mkdir(mode=0o700, exist_ok=False)
    scene = Path(spec["model_path"])
    writer = FixedBodyRenderer(pin(scene, spec["frozen_files"][str(scene)]))
    hashes = {}
    windows = []
    terminal = False
    try:
        driver = GL.glGetString(GL.GL_RENDERER).decode()
        require(
            "llvmpipe" in driver.lower() or "softpipe" in driver.lower(),
            "expected CPU rendering",
        )
        with forbid_integration(mujoco) as calls:
            for packet in itertools.chain([first], states(native)):
                t = packet["step"]
                if t == stop:
                    stream.stop(epoch)
                    try:
                        stream.policy_input()
                    except ValueError as ex:
                        require(str(ex) == "episode stopped", "unrelated refusal")
                    else:
                        raise ValueError("policy input after registered stop")
                    terminal = True
                if t not in need:
                    continue
                rgb, fingers = observe_packet(writer, stream, packet)
                hashes[t] = rgb
                if t not in PANEL:
                    continue
                obs = stream.policy_input()
                r = ledger[t]
                require(
                    r["applied_native_steps"] == 25
                    and r["source_tick"] == r["receiver_ack_tick"] == t - 25
                    and np.allclose(
                        fingers * 1000, r["measured_slide_end_mm"], rtol=0, atol=1e-10
                    ),
                    "unmatched completed consumption",
                )
                expected = [0, *[t - 500 * k for k in range(8, 0, -1)], t]
                images = obs["images"]
                require(
                    [digest(a) for a in images["cam_high"]]
                    == [hashes[k][0] for k in expected]
                    and digest(images["cam_right_wrist"]) == rgb[1]
                    and obs["state"][15] == np.float32(fingers.mean() / 0.044),
                    "mismatched observation clock",
                )
                f = out / f"{t}.npz"
                with f.open("xb") as sink:
                    np.savez_compressed(
                        sink,
                        head=images["cam_high"],
                        right_wrist=images["cam_right_wrist"],
                        state=obs["state"],
                        prompt=np.array(obs["prompt"]),
                    )
                windows.append(
                    dict(
                        tick=t,
                        boundary=PANEL[t],
                        file=f.name,
                        input_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                        previous_sequence=r["sequence"],
                        previous_width_mm=r["width_m"] * 1000,
                        previous_wire_sha256=r["wire_sha256"],
                        measured_slides_mm=(fingers * 1000).tolist(),
                        measured_input_fraction=float(obs["state"][15]),
                        newest_image_sha256=rgb,
                        head_ticks=expected,
                    )
                )
            require(
                not calls
                and terminal
                and len(windows) == 4
                and len(hashes) == len(need),
                "incomplete receipt replay",
            )
    finally:
        writer.renderer.close()
    result = dict(
        source_denominator=validated["full_denominator"],
        source_files=validated["source_sha256"],
        epoch=epoch,
        renderer=driver,
        scene_sha256=spec["frozen_files"][str(scene)],
        rendered_states=len(hashes),
        task_text=prompt,
        task_text_provenance="fixed T02 red HOLD task attached retrospectively",
        windows=windows,
        terminal_tick=stop,
        terminal_query_refused=terminal,
        original_policy_owned_actions=0,
        new_forwards=0,
        new_native_integrations=0,
    )
    (out / "FEEDBACK_OBSERVATIONS.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


def preprocess(root, folder, output):
    from checkpoint_inputs import inspection_tools

    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("JAX_PLATFORMS") == "cpu",
        "CPU preprocessing",
    )
    inspect, _, meta = inspection_tools(root)
    require(meta["discrete_state_input"], "unexpected token contract")
    captures = read(folder / "FEEDBACK_OBSERVATIONS.json")
    samples = []
    for row in captures["windows"]:
        with np.load(
            pin(folder / row["file"], row["input_sha256"]), allow_pickle=False
        ) as z:
            obs = dict(
                images=dict(
                    cam_high=z["head"],
                    cam_left_wrist=np.zeros((480, 640, 3), np.uint8),
                    cam_right_wrist=z["right_wrist"],
                ),
                state=z["state"],
                prompt=str(z["prompt"]),
            )
        transformed = inspect(obs)
        require(
            transformed["prompt"] == captures["task_text"]
            and transformed["state"][13] == obs["state"][15]
            and not transformed["prompt_truncated"],
            "feedback dropped",
        )
        s = transformed["normalized"][13]
        # Mirrors the pinned PaligemmaTokenizer discrete-state input, prior to SentencePiece.
        code = int(np.digitize(s, np.linspace(-1, 1, 257)[:-1]) - 1)
        if not samples:
            parked_width = obs["state"][15]
        fixed = obs["state"].copy()
        fixed[15] = parked_width
        same_wrist = inspect(dict(obs, state=fixed))
        samples.append(
            dict(
                **row,
                normalized_gripper=s,
                gripper_bin=code,
                tokens=transformed["tokens_used"],
                tokens_sha256=transformed["tokens_sha256"],
                network_image_sha256=transformed["network_image_sha256"],
                parked_width_only_counterfactual_changes_tokens=same_wrist[
                    "tokens_sha256"
                ]
                != transformed["tokens_sha256"],
            )
        )
    require(
        [r["tick"] for r in samples] == list(PANEL)
        and captures["terminal_query_refused"],
        "incomplete panel",
    )
    captures.update(windows=samples, token_capacity=meta["token_capacity"])
    with output.open("x") as f:
        f.write(json.dumps(captures, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    if sys.argv[2] == "render":
        repo = Path(__file__).resolve().parents[2]
        configure_site(root, repo)
        run(root, repo, Path(sys.argv[3]))
    elif sys.argv[2] == "process":
        preprocess(root, Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        raise ValueError("expected render or process")
