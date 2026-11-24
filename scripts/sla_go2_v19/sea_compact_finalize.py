"""Independently check the two selected scenes, hold geometry and camera frusta."""

import argparse
import json
from pathlib import Path

import numpy as np

from sea_compact_preflight import evaluate, startup
from sea_compact_workspace import PROFILE
from sea_fruit_camera import fruit_body_model
from sea_fruit_h0_cold import pin, read, require, sha
from sea_stance_preflight import reached_case
from sea_three_view import H, W, camera_copy
from go2arx5_vla.world_model.native_replay import forbid_integration


def pinhole(model, data, camera, target, height, width):
    cam = data.cam_xmat[camera].reshape(3, 3).T @ (target-data.cam_xpos[camera])
    depth = -cam[2]
    if depth <= 0:
        return dict(target_center_in_frame=False, depth_mm=float(depth*1000), center_px=None)
    focal = (height/2)/np.tan(np.radians(model.cam_fovy[camera])/2)
    pixel = [width/2+focal*cam[0]/depth, height/2-focal*cam[1]/depth]
    return dict(target_center_in_frame=bool(0 <= pixel[0] < width and 0 <= pixel[1] < height),
                depth_mm=float(depth*1000), center_px=[float(v) for v in pixel])


def optical_views(file, initial):
    import mujoco as mj

    body = fruit_body_model(file)
    aux, camera = camera_copy(file)
    state = []
    for model in (body, aux):
        data = mj.MjData(model)
        data.qpos[:] = initial["qpos"]
        mj.mj_fwdPosition(model, data)
        state.append(data)
    main, extra = state
    fruit = main.geom("task_cube_geom").xpos.copy()
    return dict(
        body=pinhole(body, main, body.camera("body_rgb").id, fruit, 256, 256),
        wrist=pinhole(body, main, body.camera("wrist_rgb").id, fruit, 256, 256),
        auxiliary=pinhole(aux, extra, camera, fruit, H, W),
    )


def run(site, folder):
    import mujoco

    screen_path = folder / "SCREEN.json"
    screen = read(screen_path)
    require(screen["status"] == "TWO_STATIC_LAYOUTS_SELECTED" and set(screen["selected"]) == {"G00", "G01"},
            "no selected frozen pair")
    rows = {}
    with forbid_integration(mujoco) as calls:
        for role, item in screen["selected"].items():
            row = evaluate(site, folder, role, item)
            require(row["close_geometry"] == item["close_geometry"]
                    and row["launch_state_fixture_replayed"] == item["launch_state_fixture_replayed"]
                    and row["maximum_TCP_excursion_mm"] == item["maximum_TCP_excursion_mm"],
                    "final compact model differs from screened result")
            _, entry, archive, _ = reached_case(site, role)
            file = pin(folder / f"{role}.mjb", item["scene_sha256"])
            native = mujoco.MjModel.from_binary_path(str(file))
            initial, _ = startup(native, archive["state.bin.journal"], entry)
            views = optical_views(file, initial)
            rows[role] = dict(row, scene_sha256=item["scene_sha256"], camera_frusta=views)
    require(not calls, "visual geometry proof must not integrate")
    return dict(status="STATIC_COMPACT_HOLD_AND_OPTICS_VERIFIED", profile=PROFILE,
                screen_sha256=sha(screen_path), slots=rows, new_rendered_frames=0,
                frozen_model_forwards=0, new_integrations=0,
                visibility_is_projection_only=True, native_qualified=False)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("site", type=Path)
    cli.add_argument("folder", type=Path)
    a = cli.parse_args()
    print(json.dumps(run(a.site.resolve(), a.folder.resolve()), indent=2))
