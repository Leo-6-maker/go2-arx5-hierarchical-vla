"""Build the frozen policy's exact Reach input from a single paused two-cube episode."""

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sla_hierarchical_v16"))
from demonstrated_grip import pin, read, require
from policy_query import input_identity

PROMPT = "Reach toward the red cube and stop before touching it."


def sensor_input(root, case, native, tick):
    import mujoco
    from go2arx5_vla.world_model.multitask_camera_v6 import TwoObjectBodyRenderer
    from go2arx5_vla.world_model.multitask_state_v6 import packets
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from observation_stream import ObservationStream
    from observation_visibility import observe_packet
    from OpenGL import GL

    spec = read(case / "EXECUTION_SPEC.json")
    require(
        tick >= 4000
        and native.parent == case
        and read(case / "NATIVE_REQUEST.json")["episode"] == native.name,
        "wrong episode or incomplete memory",
    )
    scene = Path(spec["model_path"])
    pins = read(
        pin(
            root / "v13/run01/S0_ACTUAL.json",
            "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9",
        )
    )["inputs"]
    path = root / "v12/TARGET_REGISTRATION.json"
    calibration = read(pin(path, pins[str(path)]))
    report = (
        Path(__file__).resolve().parents[2]
        / "reports/SLA_HIERARCHICAL_V16/OBSERVATION_VISIBILITY_ACTUAL.json"
    )
    t02 = next(row for row in read(report)["target"] if row["id"] == "T02")
    with np.load(
        pin(root / "v12/STUDENT_INPUT/T02.npz", t02["input_sha256"]), allow_pickle=False
    ) as src:
        left = src["state"][:8].copy()
    stream = ObservationStream(
        native.name,
        PROMPT,
        left,
        calibration["source_origin"],
        calibration["T_link6_TCP"],
    )
    writer = TwoObjectBodyRenderer(pin(scene, spec["frozen_files"][str(scene)]))
    try:
        driver = GL.glGetString(GL.GL_RENDERER).decode()
        require(
            any(k in driver.lower() for k in ("llvmpipe", "softpipe")),
            "expected CPU rendering",
        )
        wanted = [0, *range(tick - 4000, tick + 1, 500)]
        with (native / "initialization.jsonl").open() as src:
            init = json.loads(next(src))
        require(
            init["time"] == 0 and init["kind"] == "actual_startup_V1", "reset identity"
        )
        first = {k: np.asarray(init[k]) for k in ("qpos", "qvel", "ctrl")}
        first.update(step=0, time_s=0.0)
        observed = {}
        previous = -1
        with forbid_integration(mujoco) as calls:
            for pkt in itertools.chain([first], packets(native / "state.bin.journal")):
                t = pkt["step"]
                if t == previous == 0:
                    require(
                        all(
                            np.array_equal(pkt[k], first[k])
                            for k in ("qpos", "qvel", "ctrl")
                        ),
                        "ambiguous reset",
                    )
                    continue
                require(t == previous + 1, "journal gap")
                previous = t
                if t in wanted:
                    hashes, finger = observe_packet(writer, stream, pkt)
                    observed[t] = hashes
                if t == tick:
                    break
            require(
                not calls and set(observed) == set(wanted),
                "incomplete causal sensor history",
            )
            obs = stream.policy_input()
            require(
                set(obs) == {"state", "images", "prompt"}
                and obs["state"][15] == np.float32(np.mean(finger) / 0.044),
                "unbound measured gripper",
            )
            digest, fields = input_identity(obs)
        return obs, {
            "tick": tick,
            "head_ticks": wanted,
            "input_sha256": digest,
            "fields": fields,
            "image_tick_sha256": observed,
            "actual_finger_mm": (1000 * finger).tolist(),
            "world_origin": stream._world_origin.tolist(),
            "link6_tcp": stream.link6_tcp.tolist(),
            "renderer": driver,
            "sim_integrations": 0,
            "scene_sha256": spec["frozen_files"][str(scene)],
        }
    finally:
        writer.close()


def main(args):
    root, case, out = args.root.resolve(), args.case.resolve(), args.out
    os.umask(0o077)
    if args.mode == "render":
        sys.path.insert(
            0, str(Path(__file__).resolve().parents[1] / "sla_transfer_v14")
        )
        from b2_owner import configure

        configure(read(case / "EXECUTION_SPEC.json"))
        sys.path[:0] = [str(root / "v12"), str(root / "v12/CPU_LIB")]
        native = case / read(case / "NATIVE_REQUEST.json")["episode"]
        out.mkdir(mode=0o700)
        rows = []
        for tick in (5703, 5753):
            obs, r = sensor_input(root, case, native, tick)
            np.savez_compressed(
                out / f"{tick}.npz",
                head=obs["images"]["cam_high"],
                wrist=obs["images"]["cam_right_wrist"],
                state=obs["state"],
                prompt=obs["prompt"],
            )
            rows.append(r)
        (out / "SENSOR.json").write_text(json.dumps(rows, indent=2) + "\n")
    else:
        from checkpoint_inputs import inspection_tools

        require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and os.environ.get("JAX_PLATFORMS") == "cpu",
            "CPU only",
        )
        inspect, _, meta = inspection_tools(root)
        rows = read(out / "SENSOR.json")
        for r in rows:
            with np.load(out / f"{r['tick']}.npz", allow_pickle=False) as z:
                obs = {
                    "state": z["state"],
                    "prompt": str(z["prompt"]),
                    "images": {
                        "cam_high": z["head"],
                        "cam_left_wrist": np.zeros((480, 640, 3), np.uint8),
                        "cam_right_wrist": z["wrist"],
                    },
                }
                require(input_identity(obs)[0] == r["input_sha256"], "input changed")
                tr = inspect(obs)
            r.update(
                tokens_used=tr["tokens_used"],
                tokens_capacity=meta["token_capacity"],
                prompt_truncated=tr["prompt_truncated"],
                tokens_sha256=tr["tokens_sha256"],
                network_rgb_sha256=tr["network_image_sha256"],
                history_sha256=tr["history_images_sha256"],
            )
            require(
                len(r["history_sha256"]) == 9 and not r["prompt_truncated"],
                "missing memory or task tokens",
            )
        with (out / "PROCESSOR.json").open("x") as dest:
            json.dump(
                {"prompt": PROMPT, "samples": rows, "frozen_forwards": 0, "updates": 0},
                dest,
                indent=2,
            )
            dest.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("render", "process"))
    for name in ("root", "case", "out"):
        parser.add_argument(name, type=Path)
    main(parser.parse_args())
