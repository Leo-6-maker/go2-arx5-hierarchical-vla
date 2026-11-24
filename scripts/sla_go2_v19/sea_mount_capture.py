"""Capture the selected fixed auxiliary mount on two sealed pre-close packets."""

import argparse
import os
from pathlib import Path
import socket
import time

import numpy as np
from scipy.spatial.transform import Rotation

from sea_visual_support import EVIDENCE, FLIP, H0_DIR, H0_SHA, frozen_rgb_support, transform
from source_custody import digest, pin, read, require, write_new
from policy_query import tensor_sha

PANEL = "SEA193-AUX-MOUNT-RECORDED-01"
SLOTS = ("G00-before_close", "G01-before_close")
SCREEN_SHA = "6eef2d3b234f307a26866672f6ea2767ed01181843d5d178446934ee3fc56d66"


def candidate(screen_path):
    screen = read(pin(screen_path, SCREEN_SHA))
    picked = screen["selected"]
    require(screen["candidate_count"] == 36 and picked in screen["side_front_fixed_bracket_offset_grid"]
            and picked["acceptable"] and picked["offset_m"] == [.04, 0, .12], "unregistered mount")
    return picked


def load(source, destination, screen_path, *, grant_path=None, grant_sha=None):
    import mujoco as mj
    from go2arx5_vla.world_model.fusion_source_camera_capture import _same_model_except_external
    from go2arx5_vla.world_model.native_replay import apply_frame, forbid_integration
    from sea_fruit_h0_capture import samples
    from sea_three_view import H, W, calibration, native_sample_identity, rigid

    picked = candidate(screen_path)
    fixed = rigid(picked["base_from_camera"])
    h0 = read(pin(EVIDENCE / "GRIP_H0/COMPACT_H0_ACTUAL.json", H0_SHA))
    root = source / "v19" / H0_DIR
    pin(root / "PANEL_CLOSE.json", h0["campaign_sha256"]["PANEL_CLOSE.json"])
    K = np.asarray(calibration()["K"])
    render = grant_path is not None
    if render:
        grant = read(pin(grant_path, grant_sha))
        require(grant["status"] == "APPROVED" and grant["panel"] == PANEL
                and bool(grant["authorization_id"]) and grant["slots"] == list(SLOTS) and grant["max_rgb"] == 2
                and grant["screen_sha256"] == SCREEN_SHA and grant["capture_code_sha256"] == digest(Path(__file__))
                and grant["model_forwards"] == grant["physics_ticks"] == grant["optimizer_updates"] == 0
                and socket.gethostname().split(".")[0] == "corelab-4080s"
                and time.time() < grant["valid_until_epoch"], "missing mount capture allocation")
        destination.mkdir(mode=0o700)
    records, rendered = {}, 0
    try:
        with forbid_integration(mj) as calls:
            for name in SLOTS:
                layout = name[:3]+"-H0"
                case = root / layout
                row = h0["rows"][layout]
                seals = row["evidence_sha256"]
                lease = read(pin(case / "control/LEASE.json", seals["LEASE.json"]))
                result = read(pin(case / "RESULT.private.json", seals["RESULT.private.json"]))
                require(lease["status"] == "CLOSED" and row["qualified"] and result["absent"], "unclosed H0 case")
                spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
                native = case / result["episode"]
                pin(native / "state.bin.journal", seals["state.bin.journal"])
                folder = case / "THREE_VIEW_H0.private"
                mark = read(pin(folder / "MANIFEST.json", row["capture"]["manifest_sha256"]))["before_close"]
                at = result["observation_events"]["before_close"]
                pkt = samples(native, {"before_close": at})[mark["tick"]]
                require(native_sample_identity(native.name, pkt) == mark["current_packet_sha256"],
                        "candidate is not the same paired actual observation")
                scene = Path(spec["model_path"])
                pin(scene, spec["frozen_files"][str(scene)])
                model = mj.MjModel.from_binary_path(str(scene))
                original = mj.MjModel.from_binary_path(str(scene))
                eye = model.camera("body_rgb").id
                require(model.cam_mode[eye] == mj.mjtCamLight.mjCAMLIGHT_FIXED
                        and model.cam_bodyid[eye] == model.body("base_link").id
                        and model.vis.global_.offheight >= H and model.vis.global_.offwidth >= W,
                        "incompatible original render camera")
                model.cam_pos[eye] = fixed[:3, 3]
                model.cam_quat[eye] = Rotation.from_matrix(fixed[:3, :3]).as_quat()[[3, 0, 1, 2]]
                model.cam_fovy[eye] = calibration()["fovy_deg"]
                _same_model_except_external(original, model)
                data = mj.MjData(model)
                apply_frame(mj, model, data, pkt)
                world = transform(data.cam_xpos[eye], data.cam_xmat[eye])
                base = data.body("base_link")
                require(np.allclose(world, transform(base.xpos, base.xmat) @ fixed, rtol=0, atol=1e-9),
                        "optics do not follow the fixed candidate mount")
                records[name] = dict(input_native_state_sha256=mark["current_packet_sha256"], tick=mark["tick"],
                                     old_rgb_sha256=mark["aux_sha256"], base_from_camera=fixed.tolist(),
                                     old_anchor_sha256=row["capture"]["events"]["before_close"]["artifact_sha256"],
                                     scene_sha256=spec["frozen_files"][str(scene)])
                if render:
                    from OpenGL import GL

                    require(time.time() < grant["valid_until_epoch"] and not (destination / "STOP.json").is_file(),
                            "mount capture stopped or expired")
                    with mj.Renderer(model, height=H, width=W) as renderer:
                        driver = GL.glGetString(GL.GL_RENDERER).decode()
                        require(any(x in driver.lower() for x in ("llvmpipe", "softpipe")), "CPU renderer required")
                        write_new(destination / (name+".RGB_ENTERED.json"), dict(id=name, tick=mark["tick"]))
                        renderer.update_scene(data, camera=eye)
                        rgb = renderer.render()
                        rendered += 1
                    _, support = frozen_rgb_support(model, data, world @ FLIP, K, rgb)
                    require(support["first_hit_fruit_px"] == picked["views"][name]["first_hit_fruit_px"]
                            and support["actual_rgb_tint_share"] >= .70
                            and tensor_sha(rgb) != mark["aux_sha256"], "actual auxiliary RGB readback failed")
                    file = destination / (name+".aux.private.npy")
                    with file.open("xb") as handle:
                        np.save(handle, rgb, allow_pickle=False)
                    records[name].update(rgb_sha256=tensor_sha(rgb), rgb_file_sha256=digest(file),
                                         support=support, renderer=driver)
        require(not calls and len(records) == 2, "same-state render cannot advance physics")
    finally:
        report = dict(panel=PANEL, screen_sha256=SCREEN_SHA, code_sha256=digest(Path(__file__)), results=records,
                      new_physics_integrations=0, model_forwards=0, optimizer_updates=0, rendered=rendered)
        if render:
            write_new(destination / "CAPTURE_CLOSE.json", dict(
                report, allocation_sha256=grant_sha,
                entered=len(list(destination.glob("*.RGB_ENTERED.json")))))
    if not render:
        write_new(destination, dict(report, status="FIXED_OPTICS_TWO_ACTUAL_STATES_CPU_READY"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for label in ("root", "destination", "screen"):
        p.add_argument(label, type=Path)
    p.add_argument("--allocation", type=Path)
    p.add_argument("--allocation-sha")
    a = p.parse_args()
    os.umask(0o077)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU archived optics only")
    load(a.root.resolve(), a.destination, a.screen, grant_path=a.allocation, grant_sha=a.allocation_sha)
