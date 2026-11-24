"""Check sealed Reach inputs against native RGB, scene geometry and model coordinates on CPU."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sla_hierarchical_v16"))
from demonstrated_grip import pin, read, require
from observation_visibility import visible
from policy_query import input_identity
from visual_objects import VIEWS, estimate_objects


def frame(xyz, rotation):
    out = np.eye(4)
    out[:3, 3] = xyz
    out[:3, :3] = rotation.reshape(3, 3)
    return out


def poses(rows):
    out = np.tile(np.eye(4), (len(rows), 1, 1))
    out[:, :3, 3] = rows[:, :3]
    out[:, :3, :3] = Rotation.from_quat(rows[:, [4, 5, 6, 3]]).as_matrix()
    return out


def gain(xyz, center):
    return float(
        1000 * (np.linalg.norm(xyz[0] - center) - np.linalg.norm(xyz[-1] - center))
    )


def angle(u, v):
    return float(
        np.degrees(
            np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))
        )
    )


def run(root):
    import mujoco
    from OpenGL import GL
    from go2arx5_vla.world_model.multitask_camera_v6 import TwoObjectBodyRenderer
    from go2arx5_vla.world_model.multitask_state_v6 import packets
    from go2arx5_vla.world_model.native_replay import forbid_integration

    publication = read(
        Path(__file__).resolve().parents[2]
        / "reports/SLA_GO2_V18/Z1_NATIVE/ARCHIVE.json"
    )
    manifest = read(
        pin(root / "v18/sealed-Z1/members.json", publication["member_manifest_sha256"])
    )

    def sealed(path):
        return pin(path, manifest[str(path.relative_to(root / "v18"))])

    pins = read(
        pin(
            root / "v13/run01/S0_ACTUAL.json",
            "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9",
        )
    )["inputs"]
    p = root / "v12/TARGET_REGISTRATION.json"
    reg = read(pin(p, pins[str(p)]))
    tool = np.asarray(reg["T_link6_TCP"])
    base_origin = frame(reg["source_origin"], np.eye(3))
    output = []
    for panel, slot in (("Z1_REACH_483", "G00-A"), ("Z1_REACH_483_G01", "G01-A")):
        case = root / "v18" / panel / slot
        entry = read(sealed(case / "REACH_ENTRY.json"))
        direction = np.asarray(entry["direction"])
        spec = read(sealed(case / "EXECUTION_SPEC.json"))
        native = case / read(sealed(case / "NATIVE_REQUEST.json"))["episode"]
        journal = sealed(native / "state.bin.journal")
        requests = {
            r["tick"]: (q, r)
            for q in sorted(case.glob("query-*"))
            for r in [read(sealed(q / "REQUEST.json"))]
        }
        with sealed(native / "MOBILE_SAMPLES.private.jsonl").open() as stream:
            records = {
                r["native_end_step"]: r
                for r in map(json.loads, stream)
                if r["native_end_step"] in requests
            }
        require(
            set(records) == set(requests) == {6203, 8203}, "missing registered sample"
        )
        scene = Path(spec["model_path"])
        writer = TwoObjectBodyRenderer(pin(scene, spec["frozen_files"][str(scene)]))
        try:
            driver = GL.glGetString(GL.GL_RENDERER).decode()
            require(
                "llvmpipe" in driver.lower() or "softpipe" in driver.lower(),
                "CPU renderer",
            )
            with forbid_integration(mujoco) as steps:
                for pkt in packets(journal):
                    t = pkt["step"]
                    if t not in requests:
                        continue
                    q, req = requests[t]
                    body, _, cal, _ = writer.render(pkt, audit_masks=False)
                    rr, data, model = writer.renderer, writer.data, writer.model
                    rr.update_scene(data, camera="wrist_rgb")
                    wrist = rr.render().copy()
                    with np.load(sealed(q / "INPUT.npz"), allow_pickle=False) as z:
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
                            input_identity(obs)[0] == req["input_sha256"]
                            and req["episode"] == native.name,
                            "policy input binding",
                        )
                        require(
                            np.array_equal(body, z["head"][-1])
                            and np.array_equal(wrist, z["wrist"]),
                            "mismatched native RGB",
                        )
                        state = z["state"]
                    rgb_objects = estimate_objects(
                        body, wrist, cal, pkt["time_s"], reject_spanning=True
                    )
                    rr.enable_segmentation_rendering()
                    try:
                        masks = {v: visible(rr, model, data, v) for v in VIEWS}
                    finally:
                        rr.disable_segmentation_rendering()
                    red = np.asarray(records[t]["object_pose"])[:3, 3]
                    tcp = np.asarray(records[t]["actual_TCP"])
                    base = model.body("x5_base_link").id
                    origin = frame(data.xpos[base], data.xmat[base]) @ base_origin
                    state_tcp = origin @ poses(state[None, 8:])[0] @ tool
                    fk_error = np.linalg.norm(tcp[:3, 3] - state_tcp[:3, 3]) * 1000
                    rot_error = np.degrees(
                        Rotation.from_matrix(
                            tcp[:3, :3].T @ state_tcp[:3, :3]
                        ).magnitude()
                    )
                    require(
                        fk_error < 0.001 and rot_error < 0.001,
                        "state to measured TCP discrepancy",
                    )
                    for key, geom in (
                        ("red", "task_cube_geom"),
                        ("blue", "task_blue_cube_geom"),
                    ):
                        obj = rgb_objects[key]
                        truth = data.geom_xpos[model.geom(geom).id]
                        if key == "red":
                            require(
                                np.linalg.norm(truth - red) < 1e-8,
                                "native red mismatch",
                            )
                        obj["center_error_mm"] = (
                            None
                            if obj["center_world_m"] is None
                            else float(
                                1000
                                * np.linalg.norm(
                                    np.asarray(obj["center_world_m"]) - truth
                                )
                            )
                        )
                    red_image = rgb_objects["red"]["center_world_m"]
                    truth_goal = red - tcp[:3, 3]
                    row = dict(
                        slot=slot,
                        tick=t,
                        input_sha256=req["input_sha256"],
                        source_seed=req["seed"],
                        current_rgb_exact=True,
                        tcp_error_mm=float(fk_error),
                        tcp_rotation_error_deg=float(rot_error),
                        native_masks=masks,
                        image_objects=rgb_objects,
                        source_forecast=None,
                        task_axis_vs_current_red_deg=angle(direction, truth_goal),
                        task_axis_vs_current_image_red_deg=None
                        if red_image is None
                        else angle(direction, red_image - tcp[:3, 3]),
                    )
                    if (q / "RETURN.json").exists():
                        ret = read(sealed(q / "RETURN.json"))
                        with np.load(
                            pin(sealed(q / "OUTPUT.npz"), ret["output_sha256"]),
                            allow_pickle=False,
                        ) as z:
                            flange = origin @ poses(z["absolute"][:, 8:])
                        knots = (flange @ tool)[:, :3, 3]
                        # Hold the first tool orientation for this geometric decomposition only.
                        translation_only = flange[:, :3, 3] + (
                            knots[0] - flange[0, :3, 3]
                        )
                        delta = knots[-1] - knots[0]
                        row["source_forecast"] = dict(
                            first_to_last_red_gain_mm=gain(knots, red),
                            first_to_last_rgb_red_gain_mm=None
                            if red_image is None
                            else gain(knots, red_image),
                            translation_only_red_gain_mm=gain(translation_only, red),
                            source_first_knot_offset_from_actual_mm=float(
                                1000 * np.linalg.norm(knots[0] - tcp[:3, 3])
                            ),
                            first_to_last_axial_mm=float(1000 * delta @ direction),
                            first_to_last_transverse_mm=float(
                                1000
                                * np.linalg.norm(
                                    delta - (delta @ direction) * direction
                                )
                            ),
                        )
                    output.append(row)
                    if t == max(requests):
                        break
                require(not steps, "integration during replay")
        finally:
            writer.close()
    require(
        len(output) == 4 and sum(r["source_forecast"] is not None for r in output) == 3,
        "sample accounting",
    )
    return dict(
        source_archive_sha256=publication["archive_sha256"],
        renderer=driver,
        samples=output,
        native_starts=0,
        simulation_ticks=0,
        model_forwards=0,
        updates=0,
    )


if __name__ == "__main__":
    out = run(Path(sys.argv[1]).resolve())
    print(json.dumps(out, indent=2, allow_nan=False))
