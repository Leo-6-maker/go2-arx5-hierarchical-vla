"""No-step, robot-only SEA-0 flange and visibility probe on immutable captures."""

import argparse
import hashlib
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from go2arx5_vla.world_model.multitask_state_v6 import packets
from go2arx5_vla.world_model.native_replay import forbid_integration

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts/sla_hierarchical_v16"))
from demonstrated_grip import pin, read, require
from visual_lift_panel import write_new

URDF_SHA = "13b4a47a689afb03afd3a5f08577f0afd79559bbf4193b25072b053d0c20737b"
VIEW = "body_rgb"
SIZE = 256
SOURCE_SPACING_M = 0.6
SIMULATED_NEUTRAL_SLIDE_M = 0.022


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def transform(position, rotation):
    out = np.eye(4)
    out[:3, 3], out[:3, :3] = position, rotation
    return out


def pose(p, r):
    return transform(np.asarray(p), np.asarray(r).reshape(3, 3))


def values(text):
    return np.fromstring(text, sep=" ")


def origin(node):
    return transform(values(node["xyz"]), Rotation.from_euler("xyz", values(node["rpy"])).as_matrix())


def words(array):
    return " ".join(format(float(x), ".17g") for x in array)


def quat(matrix):
    return Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]


def pose_error(expect, got):
    translation = got[:3, 3] - expect[:3, 3]
    rotation = Rotation.from_matrix(expect[:3, :3].T @ got[:3, :3]).as_rotvec()
    return translation, rotation


def residual(expect, got):
    dp, dr = pose_error(expect, got)
    return np.r_[dp, .15*dr]


def solve(goal, guess, bounds, forward):
    x0 = np.clip(guess, bounds[0], bounds[1])
    fit = least_squares(lambda q: residual(goal, forward(q)), x0, bounds=bounds,
                        max_nfev=160, ftol=1e-10, gtol=1e-10, xtol=1e-10)
    return fit.x


class Arm:
    def __init__(self, path, count):
        document = ET.parse(path)
        joints = [document.find(f"./joint[@name='joint{i}']") for i in range(1, count+1)]
        self.rows = [(row.find("parent").get("link"), row.find("child").get("link"),
                      origin(row.find("origin").attrib), values(row.find("axis").get("xyz")),
                      row.get("type")) for row in joints]
        self.bounds = np.array([[float(row.find("limit").get(key)) for row in joints[:6]]
                                for key in ("lower", "upper")])
        self.visuals = {}
        if count == 8:
            self.slide_bounds = np.array([
                [float(row.find("limit").get(key)) for row in joints[6:]]
                for key in ("lower", "upper")
            ])
            names = ["base_link", *(row[1] for row in self.rows)]
            for name in names:
                visual = document.find(f"./link[@name='{name}']/visual")
                mesh = visual.find("geometry/mesh")
                color = visual.find("material/color").get("rgba")
                self.visuals[name] = (mesh.get("filename"), origin(visual.find("origin").attrib), color)

    def forward(self, q):
        frames = {"base_link": np.eye(4)}
        for value, (parent, child, offset, axis, kind) in zip(q, self.rows):
            motion = np.eye(4)
            if kind == "revolute":
                motion[:3, :3] = Rotation.from_rotvec(value*axis).as_matrix()
            else:
                require(kind == "prismatic", "unsupported X5 joint type")
                motion[:3, 3] = value*axis
            frames[child] = frames[parent] @ offset @ motion
        return frames


class SourceRenderer:
    def __init__(self, source, assets):
        model = ET.Element("mujoco")
        ET.SubElement(model, "compiler", angle="radian")
        asset, world = ET.SubElement(model, "asset"), ET.SubElement(model, "worldbody")
        ET.SubElement(world, "light", pos="0 -.5 2", dir="0 0 -1")
        ET.SubElement(world, "camera", name=VIEW, pos="0 0 1")
        files = {}
        for name, (mesh, geom, color) in source.visuals.items():
            filename = Path(mesh).name
            ET.SubElement(asset, "mesh", name=name, file=filename)
            files[filename] = (assets / mesh).read_bytes()
            for side in ("left", "right"):
                body = ET.SubElement(world, "body", name=f"{side}_{name}", mocap="true")
                ET.SubElement(body, "geom", type="mesh", mesh=name, pos=words(geom[:3, 3]),
                              quat=words(quat(geom[:3, :3])), rgba=color, contype="0", conaffinity="0")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(model, encoding="unicode"), assets=files)
        self.data = mujoco.MjData(self.model)
        self.source = source
        self.draw = mujoco.Renderer(self.model, height=SIZE, width=SIZE)

    def image(self, mount, left, right, left_slide, right_slide, camera, fovy):
        for side, q, slide in (("left", left, left_slide), ("right", right, right_slide)):
            require(np.all(slide >= self.source.slide_bounds[0]) and np.all(slide <= self.source.slide_bounds[1]),
                    "simulation-only finger range exceeded")
            base = mount.copy()
            if side == "left":
                base[:3, 3] += mount[:3, 1]*SOURCE_SPACING_M
            for name, local in self.source.forward(np.r_[q, slide]).items():
                body = self.model.body(f"{side}_{name}")
                idx = int(self.model.body_mocapid[body.id])
                mat = base @ local
                self.data.mocap_pos[idx] = mat[:3, 3]
                self.data.mocap_quat[idx] = quat(mat[:3, :3])
        cam = self.model.camera(VIEW)
        cam.pos[:] = camera[:3, 3]
        cam.quat[:] = quat(camera[:3, :3])
        cam.fovy[0] = fovy
        mujoco.mj_forward(self.model, self.data)
        return render_robot(self.draw, self.data, self.model, np.arange(self.model.ngeom))


def render_robot(renderer, data, model, robot_geoms, option=None, *, view=VIEW):
    renderer.update_scene(data, camera=view, scene_option=option)
    rgb = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=view, scene_option=option)
    seg = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    mask = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 0], robot_geoms)
    rgb[~mask] = 24
    return rgb, mask


class TargetRenderer:
    def __init__(self, path):
        from go2arx5_vla.world_model.native_camera_transfer_v1 import (
            FOVY, POSITION, camera_rotation,
        )

        self.model = mujoco.MjModel.from_binary_path(str(path))
        body_camera = self.model.camera(VIEW)
        require(body_camera.mode[0] == mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
                "registered body camera must be fixed to the base")
        self.model.cam_bodyid[body_camera.id] = self.model.body("base_link").id
        body_camera.pos[:] = POSITION
        body_camera.quat[:] = quat(camera_rotation())
        body_camera.fovy[0] = FOVY
        self.data = mujoco.MjData(self.model)
        self.indices = np.array([self.model.joint(f"x5_joint{i}").qposadr[0] for i in range(1, 7)])
        ids = [self.model.joint(f"x5_joint{i}").id for i in range(1, 7)]
        require(all(self.model.jnt_limited[ids]), "unbounded target joint in frozen scene")
        self.bounds = self.model.jnt_range[ids].T.copy()
        self.slides = np.array([self.model.joint(f"task_{side}_slide").qposadr[0] for side in ("left", "right")])
        n = int(self.model.body("base_link").id)
        bodies = {n}
        for i, parent in enumerate(self.model.body_parentid):
            if int(parent) in bodies:
                bodies.add(i)
        self.robot_geoms = np.flatnonzero(np.isin(self.model.geom_bodyid, list(bodies)))
        outside = ~np.isin(np.arange(self.model.ngeom), self.robot_geoms)
        self.model.geom_group[outside] = 5
        self.options = mujoco.MjvOption()
        self.options.geomgroup[5] = 0
        self.draw = mujoco.Renderer(self.model, height=SIZE, width=SIZE)

    def apply(self, sample):
        self.data.qpos[:] = sample["qpos"]
        self.data.qvel[:] = sample["qvel"]
        self.data.ctrl[:] = sample["ctrl"]
        self.data.time = sample["time_s"]
        mujoco.mj_forward(self.model, self.data)

    def flange(self):
        body = self.data.body("x5_link6")
        return pose(body.xpos, body.xmat)

    def arm(self, q):
        self.data.qpos[self.indices] = q
        mujoco.mj_forward(self.model, self.data)
        return self.flange()

    def camera(self, view=VIEW):
        cam = self.data.camera(view)
        return pose(cam.xpos, cam.xmat), float(self.model.camera(view).fovy[0])

    def set_camera(self, world_pose, fovy):
        cam = self.model.camera(VIEW)
        body = self.data.body(int(self.model.cam_bodyid[cam.id]))
        local = np.linalg.inv(pose(body.xpos, body.xmat)) @ world_pose
        cam.pos[:] = local[:3, 3]
        cam.quat[:] = quat(local[:3, :3])
        cam.fovy[0] = fovy
        mujoco.mj_forward(self.model, self.data)

    def mount(self):
        base = self.data.body("x5_base_link")
        return pose(base.xpos, base.xmat)

    def image(self, view=VIEW):
        return render_robot(self.draw, self.data, self.model, self.robot_geoms, self.options, view=view)


def packet(path, tick):
    for row in packets(Path(path) / "state.bin.journal"):
        if row["step"] == tick:
            require(abs(row["time_s"] - tick*.002) <= 1e-8, "target archived clock changed")
            return row
    raise ValueError("archived paused target state not found")


def projection(a, b, camera, fovy):
    # This is a common rendering-camera check on four flange landmarks.
    focal = SIZE/2 / np.tan(np.deg2rad(fovy)/2)
    basis = np.column_stack((np.zeros(3), .025*np.eye(3)))
    coords = []
    visible = []
    for mat in (a, b):
        points = mat[:3, :3] @ basis + mat[:3, 3, None]
        local = camera[:3, :3].T @ (points - camera[:3, 3, None])
        depth = -local[2]
        vis = depth > 0
        p = np.column_stack((SIZE/2+focal*local[0]/np.where(vis, depth, 1),
                             SIZE/2-focal*local[1]/np.where(vis, depth, 1)))
        visible.append(vis & ((p>=0) & (p<SIZE)).all(axis=1))
        coords.append(p)
    valid = visible[0] & visible[1]
    error = np.linalg.norm((coords[0]-coords[1])[valid], axis=1)
    return dict(common_in_frame_flange_landmarks=int(valid.sum()),
                median_px=float(np.median(error)) if len(error) else None,
                p95_px=float(np.percentile(error, 95)) if len(error) else None)



def inspection_camera(mount):
    eye = np.array([.78, -.9, .65])
    look = np.array([.24, .23, .06])
    z = eye - look
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return mount @ transform(eye, np.column_stack((x, y, z)))


def run(custody, assets, reference, out):
    record = read(custody / "SEA0_PAIRED_ROSTER.json")
    original = Arm(pin(reference, URDF_SHA), 6)
    official = read(assets / "ASSETS.json")
    for name, item in official["files"].items():
        pin(assets / name, item["sha256"])
    require(official["scope"] == "PUBLIC_SIMULATION_X5_VISUAL_ASSETS", "unknown visual assets")
    authored = Arm(assets / "X5A.urdf", 8)
    src_data = read(custody / "SOURCE_GEOMETRY_INPUTS.private.json")
    tgt_data = read(custody / "TARGET_GEOMETRY_INPUTS.private.json")
    available = {row["id"]: row for row in record["go2"]}
    require(len(available) == 4, "expected four fixed target developer states")
    out.mkdir(mode=0o700)
    diagnostics, inspection_count, wrist_count = [], 0, 0
    source_renderer = SourceRenderer(authored, assets)
    from OpenGL import GL

    driver = GL.glGetString(GL.GL_RENDERER).decode()
    require("llvmpipe" in driver.lower() or "softpipe" in driver.lower(), "CPU renderer required")
    with forbid_integration(mujoco) as attempts:
        for anchor in record["source"] + record["go2"]:
            key = anchor["id"]
            source_case = key in src_data
            row = available["G00-00"] if source_case else anchor
            info = tgt_data[row["id"]]
            scene = pin(Path(info["scene"]), row["scene_sha256"])
            target = TargetRenderer(scene)
            try:
                target.apply(packet(info["native"], row["tick"]))
                mount = target.mount()
                if source_case:
                    source_q = np.asarray(src_data[key]["right"]["joint_q"][-1])
                    left_q = src_data[key]["left"]["joint_q"][-1]
                    desired = mount @ original.forward(source_q)["link6"]
                    fit = solve(desired, source_q, target.bounds, target.arm)
                    target_frame = target.arm(fit)
                    slide = np.array([SIMULATED_NEUTRAL_SLIDE_M]*2)
                    target.data.qpos[target.slides] = slide
                    mujoco.mj_forward(target.model, target.data)
                else:
                    target_frame = target.flange()
                    desired = target_frame
                    source_q = target.data.qpos[target.indices].copy()
                    left_q = [0]*6
                    slide = target.data.qpos[target.slides].copy()
                camera, fovy = target.camera()
                source_fit = solve(desired, source_q, authored.bounds,
                                   lambda q: mount @ authored.forward(q)["link6"])
                source_frame = mount @ authored.forward(source_fit)["link6"]
                source_rgb, source_mask = source_renderer.image(
                    mount, left_q, source_fit, np.array([SIMULATED_NEUTRAL_SLIDE_M]*2),
                    np.clip(slide, 0, .044), camera, fovy,
                )
                target_rgb, target_mask = target.image()
                dp, dr = pose_error(target_frame, source_frame)
                target_dp, target_dr = pose_error(desired, target_frame)
                def mm(v):
                    return float(1000*np.linalg.norm(v))
                def deg(v):
                    return float(np.rad2deg(np.linalg.norm(v)))

                f = out / (key + ".private.npz")
                with f.open("xb") as stream:
                    np.savez_compressed(stream, source=source_rgb, target=target_rgb,
                                        source_robot_mask=source_mask, target_robot_mask=target_mask)
                inspection = wrist = None
                if not source_case:
                    wrist_camera, wrist_fovy = target.camera("wrist_rgb")
                    s_wrist, sw_mask = source_renderer.image(
                        mount, left_q, source_fit, np.array([SIMULATED_NEUTRAL_SLIDE_M]*2),
                        np.clip(slide, 0, .044), wrist_camera, wrist_fovy,
                    )
                    t_wrist, tw_mask = target.image("wrist_rgb")
                    wf = out / (key + "-wrist.private.npz")
                    with wf.open("xb") as stream:
                        np.savez_compressed(stream, source=s_wrist, target=t_wrist,
                                            source_robot_mask=sw_mask, target_robot_mask=tw_mask)
                    wrist = dict(archive_sha256=sha(wf), source_mask_pixels=int(sw_mask.sum()),
                                 go2_mask_pixels=int(tw_mask.sum()),
                                 shared_mask_pixels=int((sw_mask & tw_mask).sum()),
                                 projection=projection(source_frame, target_frame, wrist_camera, wrist_fovy))
                    wrist_count += 2
                    bench_view = inspection_camera(mount)
                    target.set_camera(bench_view, 50)
                    check_pose, check_fovy = target.camera()
                    require(np.allclose(check_pose, bench_view, atol=1e-8, rtol=0)
                            and check_fovy == 50, "diagnostic view pose did not bind")
                    s_img, s_mask = source_renderer.image(
                        mount, left_q, source_fit, np.array([SIMULATED_NEUTRAL_SLIDE_M]*2),
                        np.clip(slide, 0, .044), bench_view, 50,
                    )
                    t_img, t_mask = target.image()
                    require(s_mask.any() and t_mask.any(), "diagnostic camera cannot see both rigs")
                    second = out / (key + "-inspection.private.npz")
                    with second.open("xb") as stream:
                        np.savez_compressed(stream, source=s_img, target=t_img,
                                            source_robot_mask=s_mask, target_robot_mask=t_mask)
                    inspection = dict(archive_sha256=sha(second), source_mask_pixels=int(s_mask.sum()),
                                      go2_mask_pixels=int(t_mask.sum()), separate_geometry_diagnostic=True)
                    inspection_count += 2
                diagnostics.append(dict(
                    id=key, pair_archive_sha256=sha(f), target_frozen_scene_sha256=row["scene_sha256"],
                    pose_origin="real_X5_joint_archive" if source_case else "real_Go2_paused_sim_packet",
                    source_to_target_flange_mm=mm(dp), source_to_target_flange_deg=deg(dr),
                    target_to_requested_flange_mm=mm(target_dp), target_to_requested_flange_deg=deg(target_dr),
                    diagnostic_flange_match=mm(dp) <= 1 and deg(dr) <= .5
                    and mm(target_dp) <= 1 and deg(target_dr) <= .5,
                    robot_pixels=[int(source_mask.sum()), int(target_mask.sum())],
                    body_camera_source_visible=bool(source_mask.any()), side_view_inspection=inspection,
                    original_go2_wrist=wrist,
                    projection=projection(source_frame, target_frame, camera, fovy),
                    working_gripper_slide_known=False,
                    scientific_observation_pair_qualified=False,
                ))
            finally:
                target.draw.close()
    source_renderer.draw.close()
    require(not attempts and len(diagnostics) == len(record["source"])+4,
            "render accounting or integration audit failed")
    write_new(out / "SEA0_ROBOT_GEOMETRY.json", dict(
        status="APPROX_SOURCE_TWIN_ROBOT_ONLY",
        shared_view="Go2 runtime-patched native body camera",
        included_views=["body_rgb", "wrist_rgb", "separate_inspection"],
        real_source_camera_matched=False, real_source_tool_matched=False,
        history_rendered=False, renderer_cpu_verified=True, source_assets=official,
        source_kinematic_ref_sha256=URDF_SHA, rows=diagnostics,
        production_matched_observation_cells=0, geometry_only_pairs=len(diagnostics),
        rendered_diagnostic_frames=2*len(diagnostics)+inspection_count+wrist_count,
        body_camera_empty_source_masks=sum(not r["body_camera_source_visible"] for r in diagnostics),
        separate_inspection_frames=inspection_count, original_go2_wrist_frames=wrist_count,
        labelled_training_examples=0,
        new_model_forwards=0, integration_ticks=0, optimizer_updates=0,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("custody", "assets", "reference", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.custody, args.assets, args.reference, args.output)
