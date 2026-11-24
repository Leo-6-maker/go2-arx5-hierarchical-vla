"""Check off-center trial projection and preserve the sealed Go2 centered rasters."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from OpenGL import GL
from scipy.spatial.transform import Rotation

from crosspaint import replace_appearance
from pinhole_camera import PinholeCamera
from source_custody import BASE, pin, read, require, target_case, target_obs, write_new
from demonstrated_grip import rows
from go2arx5_vla.world_model.native_replay import forbid_integration
from sea1_target_preflight import hash_array, measured_robot, render_candidate
from twin_render import Arm, SIZE, SOURCE_SPACING_M, SourceRenderer, TargetRenderer, quat, solve, words

SEA = BASE / "reports/SLA_GO2_V19"
BODY, WRIST = "body_rgb", "wrist_rgb"


def point_projection():
    k = PinholeCamera(640, 480, 593.4, 438.7, 297.3, 256.7)
    # Turn and translate the camera so projection tests pose as well as K.
    rotation = Rotation.from_euler("xyz", [15, -19, 31], degrees=True).as_matrix()
    position = np.array([.12, -.3, .55])
    xyz = np.array([[0, 0, .6], [.11, -.17, .83], [-.17, .08, .91],
                    [.19, .07, 1.25], [.02, -.28, 1.2], [-.25, -.13, 1.37]])
    world = (xyz * [1, -1, -1]) @ rotation.T + position
    xml = '<mujoco><worldbody><camera name="sensor" pos="{}" quat="{}"/>'.format(
        words(position), words(quat(rotation)))
    xml += "".join(f'<geom name="p{i}" pos="{words(p)}" type="sphere" size=".0035"/>'
                   for i, p in enumerate(world))
    model = mujoco.MjModel.from_xml_string(xml + "</worldbody></mujoco>")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=k.height, width=k.width) as renderer:
        driver = GL.glGetString(GL.GL_RENDERER).decode()
        require(any(s in driver.lower() for s in ("llvmpipe", "softpipe")), "CPU renderer required")
        with k.bind(model, "sensor", renderer):
            renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera="sensor")
            seg = renderer.render().copy()
    expected = xyz[:, :2] / xyz[:, 2, None] * [k.fx, k.fy] + [k.cx, k.cy]
    found = []
    for i in range(len(expected)):
        support = np.argwhere((seg[..., 0] == i) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM))
        require(len(support) >= 3, "known marker was not observed by the renderer")
        found.append(support[:, ::-1].mean(axis=0))
    error = np.linalg.norm(np.array(found) - expected, axis=1)
    require(float(max(error)) < .65, "calibrated projection failed the known-point check")
    return dict(driver=driver, calibration=vars(k), fixture_points=len(found), expected_px=expected.tolist(),
                measured_px=np.array(found).tolist(), max_error_px=float(max(error)))


def anchored_views(root, assets):
    archive = read(BASE / "reports/SLA_GO2_V18/Z1_NATIVE/ARCHIVE.json")
    members = read(pin(root / "v18/sealed-Z1/members.json", archive["member_manifest_sha256"]))

    def sealed(path):
        return pin(path, members[str(path.relative_to(root / "v18"))])

    bundle = read(assets / "ASSETS.json")
    require(bundle["scope"] == "PUBLIC_SIMULATION_X5_VISUAL_ASSETS", "unexpected X5 asset type")
    for name, member in bundle["files"].items():
        pin(assets / name, member["sha256"])
    inventory = Arm(assets / "X5A.urdf", 8)
    history = read(SEA / "SEA1/TARGET_BRIDGE_ACTUAL.json")
    captures = {row["id"]: row for row in history["slots"]}
    result, extra = [], None
    for anchor in read(SEA / "SEA0/SEA0_PAIRED_ROSTER.json")["go2"]:
        label = anchor["id"]
        tick = anchor["tick"]
        case = target_case(root, anchor["layout"])
        spec = read(sealed(case / "EXECUTION_SPEC.json"))
        scene = pin(Path(spec["model_path"]), anchor["scene_sha256"])
        native = case / read(sealed(case / "NATIVE_REQUEST.json"))["episode"]
        at = next(row for row in rows(sealed(native / "MOBILE_SAMPLES.private.jsonl")) if row["native_end_step"] == tick)
        file = case / f"query-{int(label[-2:]):02d}" / "INPUT.npz"
        obs = target_obs(pin(sealed(file), anchor["input_file_sha256"]))["images"]
        audit = captures[label]
        target, source = TargetRenderer(scene), SourceRenderer(inventory, assets)
        try:
            robot_q = measured_robot(target, at["qpos"])
            for view, original, previous in (
                (BODY, obs["cam_high"][-1], audit["head"][-1]),
                (WRIST, obs["cam_right_wrist"], audit["wrist"]),
            ):
                rendered, camera_params, _ = render_candidate(target, source, robot_q, view)
                camera, fovy = camera_params
                predicted = replace_appearance(original, *rendered)
                require(hash_array(predicted.image) == previous["candidate_rgb_sha256"]
                        and hash_array(original) == previous["raw_rgb_sha256"], "legacy input or pixels changed")
                centered = PinholeCamera.centered(SIZE, SIZE, fovy)
                calibrated_rgb, calibrated_mask = target.image(view, intrinsics=centered)
                left_mount = target.mount().copy()
                left_mount[:3, 3] += left_mount[:3, 1] * SOURCE_SPACING_M
                mount = target.mount()
                q = solve(target.flange(), target.data.qpos[target.indices], source.source.bounds,
                          lambda q: mount @ source.source.forward(q)["link6"])
                lq = np.zeros(6)
                ls = np.array([.022, .022])
                rs = np.clip(robot_q[target.slides], 0, .044)
                calibrated_src, calibrated_src_mask = source.image_calibrated(
                    left_mount, mount, lq, q, ls, rs, camera, centered)
                remapped = replace_appearance(original, calibrated_rgb, calibrated_src,
                                              calibrated_mask, calibrated_src_mask)
                result.append(dict(id=label, camera=view, prior_candidate_reproduced=True,
                                   target_mask_xor_pixels=int(np.count_nonzero(calibrated_mask ^ rendered[2])),
                                   x5_mask_xor_pixels=int(np.count_nonzero(calibrated_src_mask ^ rendered[3])),
                                   explicit_k_candidate_changed_pixels=int(np.count_nonzero(
                                       np.any(remapped.image != predicted.image, axis=2)))))
                repeat = target.image(view)
                require(all(np.array_equal(x, y) for x, y in zip(repeat, rendered[::2], strict=True)),
                        "calibrated view changed the legacy camera configuration")
                if extra is None:
                    k = PinholeCamera(640, 480, 477., 401., 308., 242.)
                    wide = SourceRenderer(inventory, assets, width=k.width, height=k.height)
                    try:
                        rgb, mask = wide.image_calibrated(left_mount, mount, lq, q, ls, rs, camera, k)
                        require(rgb.shape == (480, 640, 3) and int(mask.sum()) > 0,
                                "off-center X5 render lost the commissioned geometry")
                        extra = dict(raster=list(rgb.shape), x5_geometry_pixels=int(mask.sum()),
                                     rgb_sha256=hash_array(rgb), mask_sha256=hash_array(mask))
                    finally:
                        wide.draw.close()
        finally:
            target.draw.close()
            source.draw.close()
    require(len(result) == 8 and extra is not None, "fixed camera preflight denominator changed")
    return result, extra


def run(root, assets, output):
    with forbid_integration(mujoco) as integrations:
        test = point_projection()
        archive, extra = anchored_views(root, assets)
        require(not integrations, "native integration in camera replay")
    write_new(output, dict(point_control=test, archive=archive, approximate_off_axis_x5=extra,
                           new_source_calibrations=0, comparable_scientific_cells=0,
                           native_ticks=0, sampler_forwards=0, training_updates=0))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "assets", "output"):
        cli.add_argument(k, type=Path)
    args = cli.parse_args()
    run(args.root.resolve(), args.assets.resolve(), args.output.resolve())
