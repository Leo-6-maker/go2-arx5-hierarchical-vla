"""Verify a known lens model and audit the warped layers on four sealed Go2 states."""

import argparse
import os
from pathlib import Path

import numpy as np

from crosspaint import replace_appearance, replace_distorted_appearance
from lens_remap import LensRemap
from pinhole_camera import PinholeCamera
from source_custody import BASE, pin, read, require, target_case, target_obs, write_new

SEA = BASE / "reports/SLA_GO2_V19"
TEST_LENS = (-.19, .051, .004, -.002, -.005, .027, -.019, .002)
BODY, WRIST = "body_rgb", "wrist_rgb"


def opencv_reference():
    import cv2

    ideal = PinholeCamera(640, 480, 350., 360., 319.5, 239.5)
    capture = PinholeCamera(352, 272, 270., 280., 170.5, 141.5)
    k = np.array([[capture.fx, 0, capture.cx], [0, capture.fy, capture.cy], [0, 0, 1.]])
    cases = []
    for coefficients in (TEST_LENS[:4], TEST_LENS[:5], TEST_LENS):
        lens = LensRemap(ideal, capture, coefficients)
        vv, uu = np.where(lens.valid)
        rays = np.column_stack(((lens.x[vv, uu]-ideal.cx)/ideal.fx,
                                (lens.y[vv, uu]-ideal.cy)/ideal.fy, np.ones(len(vv))))
        projected, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), k, np.array(coefficients))
        error = np.linalg.norm(projected[:, 0] - np.column_stack((uu, vv)), axis=1)
        require(len(error) == capture.width*capture.height and float(error.max()) < 1e-5,
                "rectified rays failed the OpenCV projection control")
        cases.append(dict(coefficient_count=len(coefficients), samples=len(error),
                          max_error_px=float(error.max())))

    # A fold and a smaller ideal raster exercise abstention independently of photometry.
    crop = PinholeCamera(24, 16, 20., 20., 11.5, 7.5)
    small_capture = PinholeCamera(32, 24, 20., 20., 15.5, 11.5)
    unreachable = LensRemap(crop, small_capture, (-2., 0., 0., 0., 0.))
    require(not unreachable.valid[0, 0] and unreachable.valid[11, 15], "invalid rays were not isolated")
    toy = np.full((crop.height, crop.width, 3), 13, dtype=np.uint8)
    solid = np.ones(toy.shape[:2], dtype=bool)
    observation = np.full((small_capture.height, small_capture.width, 3), 13, dtype=np.uint8)
    result = replace_distorted_appearance(observation, (toy, solid), (toy+10, solid), unreachable)
    require(not result.replaced[0, 0] and result.unresolved[0, 0]
            and np.array_equal(result.image[0, 0], observation[0, 0]), "missing lens support edited measured RGB")
    return dict(opencv_version=cv2.__version__, tested=cases, tested_lens=list(TEST_LENS),
                folded_or_outside_rays=int((~unreachable.valid).sum()))


def verify_view(observed, layers, camera, truth):
    target, source, target_mask, source_mask = layers
    legacy = replace_appearance(observed, *layers)
    identity = LensRemap(camera, camera, (0., 0., 0., 0.))
    original_layers = (target, target_mask), (source, source_mask)
    unchanged = replace_distorted_appearance(observed, *original_layers, identity)
    require(np.array_equal(legacy.image, unchanged.image) and np.array_equal(legacy.replaced, unchanged.replaced)
            and np.array_equal(legacy.unresolved, unchanged.unresolved), "zero lens changed a sealed control")

    distorted = LensRemap(camera, camera, TEST_LENS)
    # The known lens is imposed on the sealed, already verified simulator RGB.
    observed_in_lens = distorted.warp_rgb(observed)
    result = replace_distorted_appearance(observed_in_lens, *original_layers, distorted)
    actual_robot, cube = truth[:2]
    oracle_support = distorted.robot_layer(observed, actual_robot).robot
    any_cube_donor = np.logical_or.reduce([value & (weight > 0) for value, weight in distorted.sample(cube)])
    edited = (result.image != observed_in_lens).any(axis=2)
    require(edited.any() and not (edited & ~result.replaced).any() and not (edited & ~oracle_support).any()
            and not (edited & any_cube_donor).any(), "lens candidate changed outside measured robot support")
    a, b = [distorted.robot_layer(*layer) for layer in original_layers]
    absent = ~a.valid | ~b.valid
    require((~absent | result.unresolved).all() and not (result.replaced & absent).any(),
            "mixed silhouette or absent donor was not declared unresolved")
    return dict(zero_coefficients_identical=True, projected_rays=int(distorted.valid.sum()),
                no_rectified_support_pixels=int((~distorted.valid).sum()),
                mixed_silhouette_support_pixels=int((distorted.valid & absent).sum()),
                measured_rgb_pixels_edited=int(edited.sum()),
                edited_outside_visible_robot_pixels=int((edited & ~oracle_support).sum()),
                edited_with_task_object_donor_pixels=int((edited & any_cube_donor).sum()),
                max_supported_inverse_error_px=float(distorted.inverse_error_px[distorted.valid].max()))


def archive_controls(root, assets):
    import mujoco
    from OpenGL import GL

    from go2arx5_vla.world_model.multitask_camera_v6 import TwoObjectBodyRenderer
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from sea1_target_preflight import audit_frame, measured_robot, render_candidate
    from twin_render import Arm, SIZE, SourceRenderer, TargetRenderer, packet

    archive = read(BASE / "reports/SLA_GO2_V18/Z1_NATIVE/ARCHIVE.json")
    members = read(pin(root / "v18/sealed-Z1/members.json", archive["member_manifest_sha256"]))

    def sealed(path):
        return pin(path, members[str(path.relative_to(root / "v18"))])

    bundle = read(assets / "ASSETS.json")
    require(bundle["scope"] == "PUBLIC_SIMULATION_X5_VISUAL_ASSETS", "unexpected X5 mesh provenance")
    for name, entry in bundle["files"].items():
        pin(assets / name, entry["sha256"])
    arm = Arm(assets / "X5A.urdf", 8)
    observations = []
    with forbid_integration(mujoco) as integrated:
        for a in read(SEA / "SEA0/SEA0_PAIRED_ROSTER.json")["go2"]:
            case = target_case(root, a["layout"])
            scene = pin(Path(read(sealed(case / "EXECUTION_SPEC.json"))["model_path"]), a["scene_sha256"])
            native = case / read(sealed(case / "NATIVE_REQUEST.json"))["episode"]
            sealed(native / "state.bin.journal")
            state = packet(native, a["tick"])
            query = case / f"query-{int(a['id'][-2:]):02d}" / "INPUT.npz"
            archived = target_obs(pin(sealed(query), a["input_file_sha256"]))["images"]
            measured = {BODY: archived["cam_high"][-1], WRIST: archived["cam_right_wrist"]}
            evaluator = TwoObjectBodyRenderer(scene)
            target = TargetRenderer(scene)
            source = SourceRenderer(arm, assets)
            try:
                driver = GL.glGetString(GL.GL_RENDERER).decode()
                require(any(s in driver.lower() for s in ("llvmpipe", "softpipe")), "CPU renderer required")
                canonical, truth = audit_frame(evaluator, state, (BODY, WRIST), target.robot_geoms)
                robot_q = measured_robot(target, state["qpos"])
                for view, image in measured.items():
                    require(np.array_equal(canonical[view], image), "camera audit and sealed RGB differ")
                    layers, (_, fovy), _ = render_candidate(target, source, robot_q, view)
                    k = PinholeCamera.centered(SIZE, SIZE, fovy)
                    observations.append(dict(id=a["id"], view=view, **verify_view(image, layers, k, truth[view])))
            finally:
                evaluator.close()
                target.draw.close()
                source.draw.close()
        require(len(observations) == 8 and not integrated, "sealed camera replay denominator changed")
    return dict(driver=driver, applied_synthetic_distortion=list(TEST_LENS), views=observations,
                new_calibrated_source_views=0, native_ticks=0, sampler_forwards=0, updates=0)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("--site", type=Path, help="Read-only site root for the Go2 archive replay")
    args = cli.parse_args()
    if args.site:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU render required")
        result = archive_controls(args.site, args.site / "v19/sim-x5-assets")
    else:
        result = opencv_reference()
    write_new(args.output, result)
