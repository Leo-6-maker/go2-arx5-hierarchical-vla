"""Audit the sealed H0 target's geometric support in its actual policy pixels."""

import argparse
from collections import Counter
import os
from pathlib import Path

import numpy as np

from source_custody import BASE, digest, pin, read, require, write_new
from policy_query import tensor_sha

H0_SHA = "d2d7b94ea4ac4d33fdfa15082d12edd0c87276710ffc380c3e2ecd83a81fe443"
T2_SHA = "3f62c6e3ac79e7ef38da130423b92e0d616302c2ffd1db96d5d13f49c4d1e102"
H0_DIR = "V193-COMPACT-FRUIT-H0-01"
EVIDENCE = BASE / "reports/SLA_GO2_V193"
FLIP = np.diag([1., -1., -1., 1.])
CAMERAS = {"body_rgb": "base_0_rgb", "wrist_rgb": "right_wrist_0_rgb", "aux_left_rgb": "left_wrist_0_rgb"}
BOX_SIGNS = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])


def transform(p, r):
    t = np.eye(4)
    t[:3, :3] = r.reshape(3, 3)
    t[:3, 3] = p
    return t


def optics(model, data, name):
    camera = data.camera(name)
    f = 128/np.tan(np.deg2rad(model.camera(name).fovy[0])/2)
    return transform(camera.xpos, camera.xmat) @ FLIP, np.array([[f, 0, 128], [0, f, 128], [0, 0, 1]])


def rays_to_target(center, orient, radius, world_from_camera, K):
    """Sample the isolated analytic ellipsoid silhouette at pixel centres."""
    eye, axes = world_from_camera[:3, 3], world_from_camera[:3, :3]
    box = (center + (BOX_SIGNS*radius) @ orient.T - eye) @ axes
    require(box[:, 2].min() > 0, "fruit intersects or lies behind the camera plane")
    projected = (box[:, :2] / box[:, 2, None]) @ K[:2, :2].T + K[:2, 2]
    left, top = np.floor(projected.min(axis=0)).astype(int)
    right, bottom = np.ceil(projected.max(axis=0)).astype(int)
    require((right-left)*(bottom-top) <= 1_000_000, "fruit projection outside expected task scale")
    x, y = (p.reshape(-1) for p in np.meshgrid(np.arange(left, right), np.arange(top, bottom)))
    xy = (np.column_stack((x + .5, y + .5)) - K[:2, 2]) / K.diagonal()[:2]
    directions = np.column_stack((xy, np.ones(len(x)))) @ axes.T
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    local_eye = (eye-center) @ orient / radius
    local_rays = directions @ orient / radius
    a = np.sum(local_rays**2, axis=1)
    b = local_rays @ local_eye
    c = local_eye @ local_eye - 1
    disc = b*b - a*c
    keep = (disc >= 0) & (b < 0)
    return x[keep], y[keep], directions[keep], (-b[keep]-np.sqrt(disc[keep]))/a[keep]


def geometric_support(model, data, world_from_camera, K, shape):
    import mujoco as mj

    target = model.geom("task_cube_geom")
    item = data.geom(target.id)
    require(target.type[0] == mj.mjtGeom.mjGEOM_ELLIPSOID, "expected the sealed fruit ellipsoid")
    h, w = shape
    x, y, vec, fruit_dist = rays_to_target(item.xpos, item.xmat.reshape(3, 3), target.size[:3],
                                         world_from_camera, K)
    frame = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    eye = np.ascontiguousarray(world_from_camera[:3, 3])
    vec, x_in, y_in = np.ascontiguousarray(vec[frame]), x[frame], y[frame]
    first = np.full(len(vec), -1, dtype=np.int32)
    dist = np.full(len(vec), -1., dtype=np.float64)
    options = mj.MjvOption()
    for j, direction in enumerate(vec):
        dist[j] = mj.mj_ray(model, data, eye, direction, options.geomgroup, 1, -1, first[j:j+1])
    visible = first == target.id
    require(np.allclose(dist[visible], fruit_dist[frame][visible], rtol=0, atol=1e-7), "fruit-ray geometry differs")
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y_in[visible], x_in[visible]] = 255
    support_x, support_y = x_in[visible], y_in[visible]
    bounds = [int(support_x.min()), int(support_y.min()), int(support_x.max())+1, int(support_y.max())+1] if len(support_x) else None
    blockers = Counter(int(i) for i in first[~visible])
    names = {model.geom(i).name or f"geom_{i}": n for i, n in blockers.most_common() if i >= 0}
    return mask, dict(
        raster_hw=[h, w], isolated_silhouette_px=len(frame), in_raster_silhouette_px=int(frame.sum()),
        first_hit_fruit_px=int(visible.sum()), first_hit_bbox_xyxy=bounds,
        occluders_px=names, no_visible_geom_hits=int((first == -1).sum()),
    )


def tint_share(rgb, support, channel):
    colors = np.asarray(rgb)[support].astype(np.int16)
    if not len(colors):
        return None
    return float(np.mean(colors[:, channel] - np.delete(colors, channel, axis=1).max(axis=1) >= 30))


def frozen_rgb_support(model, data, world_from_camera, K, rgb):
    require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, "invalid sealed RGB")
    mask, proof = geometric_support(model, data, world_from_camera, K, rgb.shape[:2])
    channel = int(np.argmax(model.geom("task_cube_geom").rgba[:3]))
    return mask, dict(proof, dominant_geom_rgb_channel=channel, actual_rgb_tint_share=tint_share(rgb, mask > 0, channel))


def fixed_mount(position, look_at):
    front = look_at - position
    front /= np.linalg.norm(front)
    right = np.cross(front, [0., 0., 1.])
    right /= np.linalg.norm(right)
    return transform(position, np.column_stack((right, np.cross(right, front), -front)))


def screen_mounts(states, baseline):
    import mujoco as mj
    from itertools import product
    from sea_three_view import H, W, FIXED_SIGHT_M, POSITION_M, calibration, camera_local

    offsets = list(product((-.04, 0., .04), (0., .04, .08), (0., .04, .08, .12)))
    K = np.asarray(calibration()["K"])
    options = mj.MjvOption()
    directions = np.r_[np.eye(3), -np.eye(3)]
    rows = []
    old = np.asarray(POSITION_M)
    require(np.allclose(fixed_mount(old, np.asarray(FIXED_SIGHT_M)), camera_local(), atol=1e-14, rtol=0),
            "zero-offset bracket differs from the original optics")
    for offset in offsets:
        pos = old + offset
        mount = fixed_mount(pos, np.asarray(FIXED_SIGHT_M))
        observations = {}
        for name, model, data, root_frame in states:
            pose = root_frame @ mount @ FLIP
            eye = np.ascontiguousarray(pose[:3, 3])
            target_id = np.empty(1, dtype=np.int32)
            near = [d for ray in directions if (d := mj.mj_ray(model, data, eye, ray, options.geomgroup, 1, -1, target_id)) >= 0]
            _, proof = geometric_support(model, data, pose, K, (H, W))
            observations[name] = dict(
                first_hit_fruit_px=proof["first_hit_fruit_px"],
                in_raster_silhouette_px=proof["in_raster_silhouette_px"],
                entirely_in_raster=proof["in_raster_silhouette_px"] == proof["isolated_silhouette_px"],
                nearest_axial_surface_m=min(near, default=2.),
            )
        acceptable = all(
            r["first_hit_fruit_px"] >= baseline[name]["current"]["aux_left_rgb"]["first_hit_fruit_px"]
            and r["entirely_in_raster"] and r["nearest_axial_surface_m"] >= .020
            for name, r in observations.items()
        )
        counts = [r["first_hit_fruit_px"] for name, r in observations.items() if name.endswith("-before_close")]
        rows.append(dict(offset_m=list(offset), base_from_camera=mount.tolist(), views=observations,
                         minimum_preclose_support_px=min(counts), acceptable=acceptable))
    control = next(r for r in rows if r["offset_m"] == [0, 0, 0])
    require(all(r["first_hit_fruit_px"] == baseline[k]["current"]["aux_left_rgb"]["first_hit_fruit_px"]
                for k, r in control["views"].items()), "zero-offset screen failed to reproduce archived support")
    eligible = [r for r in rows if r["acceptable"]]
    selected = max(eligible, key=lambda r: (r["minimum_preclose_support_px"], -np.linalg.norm(r["offset_m"]))) if eligible else None
    return dict(side_front_fixed_bracket_offset_grid=rows, selected=selected,
                candidate_count=len(rows), accepted_count=len(eligible),
                selection="maximize minimum pre-close visible support; prefer shortest offset on ties",
                camera_enclosure_and_live_mount_qualified=False, new_rgb_renders=0)


def geometry(root, destination):
    import mujoco as mj
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from sea_fruit_camera import fruit_body_model
    from sea_fruit_h0_capture import samples
    from sea_three_view import calibration, camera_local, native_sample_identity

    h0 = read(pin(EVIDENCE / "GRIP_H0/COMPACT_H0_ACTUAL.json", H0_SHA))
    campaign = root / "v19" / H0_DIR
    require(mj.__version__ == "3.2.7" and mj.get_mjcb_control() is None, "passive pinned geometry runtime")
    pin(campaign / "PANEL_CLOSE.json", h0["campaign_sha256"]["PANEL_CLOSE.json"])
    reports, masks, snapshots = {}, {}, []
    with forbid_integration(mj) as attempted:
        for layout, item in h0["rows"].items():
            case = campaign / layout
            seals = item["evidence_sha256"]
            closed = read(pin(case / "control/LEASE.json", seals["LEASE.json"]))
            result = read(pin(case / "RESULT.private.json", seals["RESULT.private.json"]))
            spec = read(pin(case / "EXECUTION_SPEC.json", closed["spec_sha256"]))
            require(closed["status"] == "CLOSED" and item["qualified"] and result["absent"], "unqualified archive")
            native = case / result["episode"]
            pin(native / "state.bin.journal", seals["state.bin.journal"])
            folder = case / "THREE_VIEW_H0.private"
            manifest = read(pin(folder / "MANIFEST.json", item["capture"]["manifest_sha256"]))
            states = samples(native, result["observation_events"])
            scene = Path(spec["model_path"])
            model = fruit_body_model(pin(scene, spec["frozen_files"][str(scene)]))
            data = mj.MjData(model)

            def at(tick):
                data.qpos[:] = states[tick]["qpos"]
                mj.mj_fwdPosition(model, data)

            for event, obs in manifest.items():
                name, tick = layout[:3] + "-" + event, obs["tick"]
                mark = item["capture"]["events"][event]
                source = pin(folder / (event + ".npz"), mark["artifact_sha256"])
                require(obs["status"] == "SEALED" and obs["aux_calibration"] == calibration()
                        and obs["current_packet_sha256"] == native_sample_identity(native.name, states[tick]),
                        "sealed optical packet changed")
                results = reports[name] = dict(tick=tick, raw_artifact_sha256=mark["artifact_sha256"], current={}, history=[])
                with np.load(source, allow_pickle=False) as z:
                    head, wrist, aux = z["head"], z["wrist"], z["aux"]
                    require([tensor_sha(f) for f in head] == obs["history_frame_sha256"]
                            and tensor_sha(wrist) == obs["wrist_sha256"] and tensor_sha(aux) == obs["aux_sha256"],
                            "archived RGB bytes changed")
                    for i, historical in enumerate(obs["history_ticks"][:-1]):
                        at(historical)
                        key = f"{name}-history{i}"
                        masks[key], rec = frozen_rgb_support(model, data, *optics(model, data, "body_rgb"), head[i])
                        results["history"].append(dict(tick=historical, mask_key=key, **rec))
                    at(tick)
                    snapshot = mj.MjData(model)
                    snapshot.qpos[:] = states[tick]["qpos"]
                    mj.mj_fwdPosition(model, snapshot)
                    b = snapshot.body("base_link")
                    snapshots.append((name, model, snapshot, transform(b.xpos, b.xmat)))
                    for view, rgb in (("body_rgb", head[-1]), ("wrist_rgb", wrist), ("aux_left_rgb", aux)):
                        if view == "aux_left_rgb":
                            base = data.body("base_link")
                            pose = transform(base.xpos, base.xmat) @ camera_local() @ FLIP
                            K = np.asarray(obs["aux_calibration"]["K"])
                        else:
                            pose, K = optics(model, data, view)
                            recorded = obs["body_wrist_calibration"][view]
                            require(np.allclose(pose, recorded["T_world_optical"], rtol=0, atol=1e-8)
                                    and np.allclose(K, recorded["K"], rtol=0, atol=1e-8),
                                    "camera optical reconstruction differs from capture seal")
                        key = f"{name}-{view}"
                        masks[key], rec = frozen_rgb_support(model, data, pose, K, rgb)
                        results["current"][view] = dict(mask_key=key, **rec)
        screen = screen_mounts(snapshots, reports)
    require(not attempted and len(reports) == 4 and len(masks) == 48, "incorrect passive visual scope")
    destination.mkdir(mode=0o700)
    with (destination / "SUPPORT.private.npz").open("xb") as f:
        np.savez_compressed(f, **masks)
    write_new(destination / "GEOMETRY.json", dict(
        status="ARCHIVED_RGB_FIRST_SURFACE_SUPPORT", compact_h0_receipt_sha256=H0_SHA,
        mask_support_sha256=digest(destination / "SUPPORT.private.npz"), rows=reports,
        new_rgb_renders=0, new_physics_integrations=0, model_forwards=0, optimizer_updates=0,
    ))
    write_new(destination / "MOUNT_SCREEN.json", screen)


def processor(root, geometry_path, output):
    import jax
    import jax.numpy as jnp
    from openpi.shared.image_tools import resize_with_pad
    from sea1b_panel import fields_at_device, registration, verified_processor
    from sea_target_t2 import CONDITION, anchors_and_payloads

    require(jax.default_backend() == "cpu", "official image transform uses CPU")
    frozen, _ = registration()
    prepare, _, identity = verified_processor(root, frozen)
    rows, _, prepared = anchors_and_payloads(root, prepare)
    cold = read(pin(EVIDENCE / "TARGET_T2/COLD.json", T2_SHA))
    require(identity == cold["processor"], "different policy image processor")
    record = read(geometry_path / "GEOMETRY.json")
    require(record["compact_h0_receipt_sha256"] == H0_SHA and set(rows) == set(record["rows"]),
            "different geometry input population")
    with np.load(pin(geometry_path / "SUPPORT.private.npz", record["mask_support_sha256"]), allow_pickle=False) as z:
        for name, entry in record["rows"].items():
            obs = prepared[name]
            require(entry["raw_artifact_sha256"] == rows[name]["archive_sha256"]
                    and fields_at_device(obs) == cold["fields"][name][CONDITION],
                    "actual processor pixels or observation seal changed")
            pairs = [(entry["current"][view], obs["image"][slot]) for view, slot in CAMERAS.items()]
            pairs.extend(zip(entry["history"], obs["extra_images"], strict=True))
            for item, rgb in pairs:
                shape = item["raster_hw"]
                raw = z[item["mask_key"]]
                require(list(raw.shape) == shape and np.count_nonzero(raw) == item["first_hit_fruit_px"]
                        and rgb.shape == (224, 224, 3), "mask or policy pixel layout differs")
                mask = np.asarray(resize_with_pad(jnp.asarray(raw[..., None]), 224, 224))[..., 0]
                core = mask >= 128
                item["processor_equivalent_visible_px"] = float(mask.sum()/255)
                item["processor_majority_visible_px"] = int(core.sum())
                item["processor_rgb_tint_share"] = tint_share(rgb, core, item["dominant_geom_rgb_channel"])
    write_new(output, dict(record, status="ARCHIVE_VISUAL_SUPPORT_AND_OFFICIAL_RESIZE_VERIFIED",
                           t2_cold_sha256=T2_SHA, raw_geometry_sha256=digest(geometry_path / "GEOMETRY.json"),
                           publicly_exports_masks=False, real_source_optical_calibration=False))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    modes = cli.add_subparsers(dest="mode", required=True)
    modes.add_parser("geometry").add_argument("output", type=Path)
    p = modes.add_parser("processor")
    for label in ("geometry", "output"):
        p.add_argument(label, type=Path)
    a = cli.parse_args()
    os.umask(0o077)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU readback only")
    if a.mode == "geometry":
        geometry(a.root.resolve(), a.output)
    else:
        processor(a.root.resolve(), a.geometry, a.output)
