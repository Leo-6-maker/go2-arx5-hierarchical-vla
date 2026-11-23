"""Replay live-format observations and score native geometry visibility using archived states."""

import itertools
import hashlib
import os
from pathlib import Path
import sys

import numpy as np

from demonstrated_grip import read, rows, require, pin
from sealed_panel import configure_site


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def observe_packet(writer, stream, packet):
    from dense_replay import pose

    m, d, rr = writer.model, writer.data, writer.renderer
    body, _, _, _ = writer.render(packet, audit_masks=False)
    rr.update_scene(d, camera="wrist_rgb")
    wrist = rr.render().copy()
    fingers = [
        m.jnt_qposadr[m.joint("task_" + k + "_slide").id] for k in ("left", "right")
    ]
    actual = d.qpos[fingers].copy()
    arm, flange, tip = (
        m.body("x5_base_link").id,
        m.body("x5_link6").id,
        m.site("x5_ee").id,
    )
    stream.observe(
        stream.episode,
        packet["step"],
        packet["time_s"],
        body,
        wrist,
        pose(d.xpos[arm], d.xmat[arm]),
        pose(d.xpos[flange], d.xmat[flange]),
        pose(d.site_xpos[tip], d.site_xmat[tip]),
        actual,
    )
    return [digest(body), digest(wrist)], actual


def visible(renderer, model, data, camera):
    import mujoco

    renderer.update_scene(data, camera=camera)
    seg = renderer.render()
    out = {}
    for label, name in (("red", "task_cube_geom"), ("blue", "task_blue_cube_geom")):
        y, x = np.where(
            (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
            & (seg[..., 0] == model.geom(name).id)
        )
        out[label] = dict(
            pixels=len(x),
            bbox_xyxy=[int(x.min()), int(y.min()), int(x.max()), int(y.max())]
            if len(x)
            else None,
        )
    return out


def run(root, repo, *, on_query=None):
    from observation_stream import ObservationStream
    import mujoco
    from OpenGL import GL
    from go2arx5_vla.world_model.multitask_camera_v6 import TwoObjectBodyRenderer
    from go2arx5_vla.world_model.multitask_state_v6 import packets
    from go2arx5_vla.world_model.native_replay import forbid_integration

    pins = read(
        pin(
            root / "v13/run01/S0_ACTUAL.json",
            "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9",
        )
    )["inputs"]
    v = root / "v12"
    cap_path = v / "A2_O2_DENSE_CAPTURE.json"
    cap = read(pin(cap_path, pins[str(cap_path)]))
    roster = v / "TARGET_REGISTRATION.json"
    reg = read(pin(roster, pins[str(roster)]))
    # Virtual arm and missing wrist are the frozen inputs chosen for this checkpoint.
    left = read(v / "A1_SOURCE_01/call_02_input.json")["state"][:8]
    left_rgb = read(v / "A1_SOURCE_01/call_01_input.json")["wrists_sha256"][
        "cam_left_wrist"
    ]
    meta = {r["id"]: r for r in cap["episodes"]}
    out = []
    for group in read(root / "evidence/P1_REGISTRATION.json")["target_windows"]:
        name = group["episode"]
        ep_file = pin(
            Path(group["episode_pin"]["path"]), group["episode_pin"]["sha256"]
        )
        ep = read(ep_file)
        rec = read(pin(Path(meta[name]["receipt_path"]), meta[name]["receipt_sha256"]))
        require(ep["id"] == name == rec["id"] and rec["split"] == "TRAIN", "episode")
        case = ep_file.parents[3] / "cases" / name
        spec = read(pin(case / "EXECUTION_SPEC.json", rec["source_spec_sha256"]))
        n = case / "native-01"
        entries = {
            w["tick"]: w for w in cap["windows"].values() if w["episode"] == name
        }
        need = {t for w in entries.values() for t in w["head_ticks"]}
        fpath = v / "DENSE" / name / "frames.jsonl"
        frozen = {
            r["tick"]: r
            for r in rows(pin(fpath, rec["frames_sha256"]))
            if r["tick"] in need
        }
        require(need == frozen.keys(), "missing dense frame")
        trace = n / "state.bin.journal"
        pin(trace, rec["source_files"][trace.name])
        firstpath = n / "initialization.jsonl"
        with pin(firstpath, rec["source_files"][firstpath.name]).open() as f:
            import json

            init = json.loads(f.readline())
        require(
            init["time"] == 0 and init["kind"] == "actual_startup_V1",
            "missing actual reset",
        )
        first = {k: np.asarray(init[k]) for k in ("qpos", "qvel", "ctrl")}
        first.update(time_s=0.0, step=0)
        stream = ObservationStream(
            name, ep["language"], left, reg["source_origin"], reg["T_link6_TCP"]
        )
        writer = TwoObjectBodyRenderer(
            pin(Path(spec["model_path"]), rec["scene_sha256"])
        )
        m, d, rr = writer.model, writer.data, writer.renderer
        masks = {}
        sites = 0
        try:
            driver = GL.glGetString(GL.GL_RENDERER).decode()
            require(
                "llvmpipe" in driver.lower() or "softpipe" in driver.lower(),
                "software renderer required",
            )

            def make_input(a):
                tick = a["step"]
                rgb, _ = observe_packet(writer, stream, a)
                require(rgb == frozen[tick]["rgb_sha256"], "different camera bytes")
                try:
                    rr.enable_segmentation_rendering()
                    masks[tick] = {
                        k: visible(rr, m, d, k) for k in ("body_rgb", "wrist_rgb")
                    }
                finally:
                    rr.disable_segmentation_rendering()

            with forbid_integration(mujoco) as attempts:
                last = None
                for a in itertools.chain([first], packets(trace)):
                    tick = a["step"]
                    if last == 0 and tick == 0:
                        require(
                            all(
                                np.array_equal(first[k], a[k])
                                for k in ("qpos", "qvel", "ctrl")
                            ),
                            "divergent reset",
                        )
                        continue
                    require(last is None or tick == last + 1, "native state gap")
                    last = tick
                    if tick not in need:
                        continue
                    make_input(a)
                    sites += 1
                    if tick not in entries:
                        continue
                    w = entries[tick]
                    obs = stream.policy_input()
                    f = v / "STUDENT_INPUT" / (w["id"] + ".npz")
                    with np.load(pin(f, w["input_sha256"]), allow_pickle=False) as z:
                        require(
                            np.array_equal(obs["images"]["cam_high"], z["head"])
                            and np.array_equal(
                                obs["images"]["cam_right_wrist"], z["right_wrist"]
                            )
                            and np.array_equal(obs["state"], z["state"])
                            and obs["prompt"] == str(z["prompt"])
                            and digest(obs["images"]["cam_left_wrist"]) == left_rgb,
                            "stream differs from model-bound input",
                        )
                    if on_query is not None:
                        on_query(stream, w)
                    out.append(
                        dict(
                            id=w["id"],
                            episode=name,
                            tick=tick,
                            input_sha256=w["input_sha256"],
                            prompt=obs["prompt"],
                            body_history=[
                                masks[k]["body_rgb"] for k in w["head_ticks"]
                            ],
                            wrist=masks[tick]["wrist_rgb"],
                            sensor_state=obs["state"].tolist(),
                            renderer=driver,
                        )
                    )
                    if tick == max(need):
                        break
                require(
                    not attempts and sites == len(need),
                    "incomplete replay/physics",
                )
            for r in out:
                if r["episode"] == name:
                    r["unique_episode_snapshots"] = sites
        finally:
            rr.close()
    require(len(out) == len({r["id"] for r in out}) == 16, "incomplete target cohort")
    return dict(
        target=sorted(out, key=lambda r: r["id"]), native_integrations=0, new_forwards=0
    )


if __name__ == "__main__":
    import json

    root = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parents[2]
    configure_site(root, repo)
    require(os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1", "software CPU renderer")
    result = run(root, repo)
    with Path(sys.argv[2]).open("x") as f:
        f.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
