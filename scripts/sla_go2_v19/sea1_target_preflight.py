"""Cross-paint fixed Go2 developer observations and score support in a separate oracle."""

import argparse
import itertools
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from OpenGL import GL

from crosspaint import replace_appearance
from twin_render import (
    BASE, Arm, SourceRenderer, TargetRenderer, pin, pose_error, read,
    require, sha, solve, write_new,
)
from go2arx5_vla.world_model.multitask_camera_v6 import TwoObjectBodyRenderer
from go2arx5_vla.world_model.multitask_state_v6 import packets
from go2arx5_vla.world_model.native_replay import forbid_integration
from policy_query import input_identity

BODY = "body_rgb"
WRIST = "wrist_rgb"


def measured_robot(renderer, qpos):
    """Extract free base, legs, right arm and measured fingers by body ancestry."""
    model = renderer.model
    owners = {model.body("base_link").id}
    for body, parent in enumerate(model.body_parentid):
        if int(parent) in owners and body:
            owners.add(body)
    joints = np.flatnonzero(np.isin(model.jnt_bodyid, list(owners)))
    addresses = np.r_[model.jnt_qposadr, model.nq]
    indices = np.concatenate([np.arange(addresses[j], addresses[j + 1]) for j in joints])
    require(np.array_equal(indices, np.arange(27)), "unexpected recorded robot joint ABI")
    return np.asarray(qpos)[indices].copy()


def render_candidate(target, source, robot_q, view):
    """The copied renderer receives measured robot coordinates only."""
    require(robot_q.shape == (27,) and np.isfinite(robot_q).all(), "invalid robot signal")
    model, data = target.model, target.data
    mujoco.mj_resetData(model, data)
    data.qpos[:27] = robot_q
    mujoco.mj_forward(model, data)
    camera, fovy = target.camera(view)
    rgb, mask = target.image(view)
    mount = target.mount()
    flange = target.flange()
    fitted = solve(flange, data.qpos[target.indices], source.source.bounds,
                   lambda q: mount @ source.source.forward(q)["link6"])
    candidate_flange = mount @ source.source.forward(fitted)["link6"]
    dp, dr = pose_error(flange, candidate_flange)
    error = [float(1000*np.linalg.norm(dp)), float(np.rad2deg(np.linalg.norm(dr)))]
    require(error[0] <= 1.0 and error[1] <= .5, "approximate virtual flange match failed")
    # The source geometry uses a simulation-only finger convention. Propagate
    # the known approximation in the enclosing scientific admission receipt.
    source_rgb, source_mask = source.image(
        mount, np.zeros(6), fitted, np.array([.022, .022]),
        np.clip(robot_q[target.slides], 0.0, .044), camera, fovy,
    )
    layer = (rgb, source_rgb, mask, source_mask)
    return layer, (camera, fovy), error


def audit_frame(writer, packet, views, robot_geoms):
    """World geometry exists only in the separate, archived evaluator."""
    body, _, _, _ = writer.render(packet, audit_masks=False)
    buf, model, data = writer.renderer, writer.model, writer.data
    raw = {BODY: body}
    if WRIST in views:
        buf.update_scene(data, camera=WRIST)
        raw[WRIST] = buf.render().copy()
    truth = {}
    buf.enable_segmentation_rendering()
    try:
        for view in views:
            buf.update_scene(data, camera=view)
            ids = buf.render().copy()
            is_geom = ids[..., 1] == mujoco.mjtObj.mjOBJ_GEOM
            cam_id = model.camera(view).id
            truth[view] = (
                is_geom & np.isin(ids[..., 0], robot_geoms),
                is_geom & np.isin(ids[..., 0], [
                    model.geom("task_cube_geom").id, model.geom("task_blue_cube_geom").id
                ]),
                data.cam_xpos[cam_id].copy(), data.cam_xmat[cam_id].reshape(3, 3).copy(),
                float(model.cam_fovy[cam_id]),
            )
    finally:
        buf.disable_segmentation_rendering()
    return raw, truth


def compare(result, observed, layer, cal, truth):
    target_rgb, _, target, source = layer
    visible, task_objects, pos, rotation, fovy = truth
    cam, angle = cal
    require(np.allclose(cam[:3, 3], pos, atol=1e-10, rtol=0)
            and np.allclose(cam[:3, :3], rotation, atol=1e-10, rtol=0)
            and angle == fovy, "robot-only and observation camera differ")
    edited = np.any(result.image != observed, axis=2)
    require(np.all(~edited | result.replaced), "unexpected edit outside certified pixels")
    return dict(
        target_geometry_pixels=int(target.sum()), source_geometry_pixels=int(source.sum()),
        common_geometry_pixels=int((target & source).sum()),
        target_photometric_match_pixels=int((target & (target_rgb == observed).all(axis=-1)).sum()),
        accepted_support_pixels=int(result.replaced.sum()), actual_edits=int(edited.sum()),
        unresolved_pixels=int(result.unresolved.sum()), oracle_visible_robot_pixels=int(visible.sum()),
        oracle_supported_nonrobot_pixels=int((result.replaced & ~visible).sum()),
        oracle_edited_nonrobot_pixels=int((edited & ~visible).sum()),
        oracle_edited_task_object_pixels=int((edited & task_objects).sum()),
        oracle_source_only_would_overwrite_visible_nonrobot=int((source & ~target & ~visible).sum()),
        raw_rgb_sha256=hash_array(observed), candidate_rgb_sha256=hash_array(result.image),
        support_mask_sha256=hash_array(result.replaced), camera_verified_against_archive=True,
    )


def hash_array(value):
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def raw_observation(file):
    with np.load(file, allow_pickle=False) as z:
        return dict(
            state=z["state"], prompt=str(z["prompt"]),
            images=dict(cam_high=z["head"], cam_right_wrist=z["wrist"],
                        cam_left_wrist=np.zeros((480, 640, 3), dtype=np.uint8)),
        )


def frames(episode, sealed, last):
    with sealed(episode / "initialization.jsonl").open() as file:
        init = json.loads(next(file))
    require(init["time"] == 0 and init["kind"] == "actual_startup_V1", "reset provenance")
    first = {k: np.asarray(init[k]) for k in ("qpos", "qvel", "ctrl")}
    first.update(step=0, time_s=0.0)
    previous = -1
    for row in itertools.chain([first], packets(sealed(episode / "state.bin.journal"))):
        t = row["step"]
        if t == previous == 0:
            require(all(np.array_equal(row[k], first[k]) for k in ("qpos", "qvel", "ctrl")),
                    "different reset packet")
            continue
        require(t == previous + 1 and abs(row["time_s"] - t*.002) <= 1e-8, "native clock gap")
        previous = t
        yield row
        if t == last:
            return
    raise ValueError("query capture missing from the native journal")


def build_case(root, out, roster, assets, sealed):
    slots = {z["layout"] for z in roster}
    require(len(slots) == 1 and len(roster) == 2, "wrong paired episode scope")
    slot = slots.pop()
    parent = "Z1_REACH_483" if slot == "G00" else "Z1_REACH_483_G01"
    case = root / "v18" / parent / f"{slot}-A"
    spec = read(sealed(case / "EXECUTION_SPEC.json"))
    episode = case / read(sealed(case / "NATIVE_REQUEST.json"))["episode"]
    scene = pin(Path(spec["model_path"]), roster[0]["scene_sha256"])
    received = []
    for i, item in enumerate(roster):
        query = case / f"query-{i:02d}"
        request = read(sealed(query / "REQUEST.json"))
        obs = raw_observation(sealed(query / "INPUT.npz"))
        require(
            request["tick"] == item["tick"] and request["episode"] == episode.name
            and input_identity(obs)[0] == request["input_sha256"] == item["input_sha256"]
            and item["history_ticks"] == [0, *range(item["tick"]-4000, item["tick"]+1, 500)],
            "archived inputs changed",
        )
        received.append((item, obs))
    head = {}
    wrist = {}
    for item, obs in received:
        for tick, im in zip(item["history_ticks"], obs["images"]["cam_high"]):
            require(tick not in head or np.array_equal(head[tick], im), "shared history conflict")
            head[tick] = im
        wrist[item["tick"]] = obs["images"]["cam_right_wrist"]

    evaluator = TwoObjectBodyRenderer(scene)
    target = TargetRenderer(scene)
    src = SourceRenderer(Arm(assets / "X5A.urdf", 8), assets)
    results, poison_cases = {}, 0
    try:
        driver = GL.glGetString(GL.GL_RENDERER).decode()
        require(any(k in driver.lower() for k in ("llvmpipe", "softpipe")),
                "CPU renderer required")
        for row in frames(episode, sealed, max(head)):
            tick = row["step"]
            if tick not in head:
                continue
            views = [BODY, WRIST] if tick in wrist else [BODY]
            rgb, oracle = audit_frame(evaluator, row, views, target.robot_geoms)
            q = measured_robot(target, row["qpos"])
            for view in views:
                observed = (head if view == BODY else wrist)[tick]
                require(np.array_equal(rgb[view], observed), "archived camera RGB differs")
                layer, camera, fit = render_candidate(target, src, q, view)
                result = replace_appearance(observed, *layer)
                audit = compare(result, observed, layer, camera, oracle[view])
                if tick in wrist:
                    poison = np.r_[q, np.full(14, np.nan)]
                    replay, _, _ = render_candidate(target, src, measured_robot(target, poison), view)
                    require(all(np.array_equal(a, b) for a, b in zip(layer, replay)),
                            "non-robot state leaked into candidate pixels")
                    poison_cases += 1
                results[tick, view] = (audit, result, fit)
    finally:
        evaluator.close()
        target.draw.close()
        src.draw.close()

    outputs = []
    for item, raw in received:
        ts = item["history_ticks"]
        history = [results[t, BODY] for t in ts]
        current_wrist = results[item["tick"], WRIST]
        diagnostics = dict(
            images=dict(cam_high=np.stack([z[1].image for z in history]),
                        cam_right_wrist=current_wrist[1].image,
                        cam_left_wrist=raw["images"]["cam_left_wrist"]),
            state=raw["state"], prompt=raw["prompt"],
        )
        f = out / (item["id"] + ".private.npz")
        with f.open("xb") as fp:
            np.savez_compressed(fp, head=diagnostics["images"]["cam_high"],
                                wrist=diagnostics["images"]["cam_right_wrist"],
                                head_paint=np.stack([z[1].replaced for z in history]),
                                wrist_paint=current_wrist[1].replaced)
        outputs.append(dict(
            id=item["id"], tick=item["tick"], original_input_sha256=item["input_sha256"],
            diagnostic_input_sha256=input_identity(diagnostics)[0], archive_sha256=sha(f),
            head=[dict(tick=t, **z[0], approximate_flange_error=z[2]) for t,z in zip(ts, history)],
            wrist=dict(tick=item["tick"], **current_wrist[0], approximate_flange_error=current_wrist[2]),
            original_state_and_prompt_preserved=True, original_left_placeholder_preserved=True,
            scientific_observation_pair_qualified=False,
        ))
    return outputs, len(results), poison_cases, driver


def synthetic_control():
    obs = np.zeros((10, 10, 3), dtype=np.uint8)
    target = np.zeros((10, 10), dtype=bool)
    target[1:8, 1:8] = True
    source = target.copy()
    source[8:, 8:] = True
    observed = obs.copy()
    observed[4, 4] = [0, 30, 0]
    result = replace_appearance(observed, obs, np.full_like(obs, 80), target, source)
    require(result.replaced.sum() == 24 and result.unresolved[8, 8]
            and np.array_equal(result.image[4, 4], observed[4, 4])
            and np.array_equal(result.image[8, 8], observed[8, 8]), "paint/control behavior")
    return True


def run(root, roster, assets, out):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only target replay")
    manifest_report = read(BASE / "reports/SLA_GO2_V18/Z1_NATIVE/ARCHIVE.json")
    manifest = read(pin(root / "v18/sealed-Z1/members.json", manifest_report["member_manifest_sha256"]))

    def sealed(path):
        return pin(path, manifest[str(path.relative_to(root / "v18"))])

    asset_records = read(assets / "ASSETS.json")
    require(asset_records["scope"] == "PUBLIC_SIMULATION_X5_VISUAL_ASSETS", "source asset scope")
    for name, item in asset_records["files"].items():
        pin(assets / name, item["sha256"])
    fixed = read(roster)["go2"]
    require([x["id"] for x in fixed] == ["G00-00", "G00-01", "G01-00", "G01-01"],
            "wrong target roster")
    out.mkdir(mode=0o700)
    checks, cases, poison, drivers = [], 0, 0, set()
    with forbid_integration(mujoco) as steps:
        for layout in ("G00", "G01"):
            batch, n, mutated, driver = build_case(root, out, [x for x in fixed if x["layout"] == layout],
                                           assets, sealed)
            checks.extend(batch)
            cases += n
            poison += mutated
            drivers.add(driver)
    require(not steps and poison == 8 and len(checks) == 4, "unexpected replay evidence")
    write_new(out / "TARGET_BRIDGE_ACTUAL.json", dict(
        status="SAME_VIEW_APPEARANCE_CPU_DIAGNOSTIC",
        target_camera="archived runtime-patched body; unchanged native wrist",
        source_camera_calibrated=False, source_physical_tool_calibrated=False,
        source_geometry_asset_sha256=sha(assets / "ASSETS.json"),
        scientific_observation_pairs_qualified=0, new_native_starts=0,
        new_model_forwards=0, physics_ticks=0, updates=0,
        replayed_slots=sum(len(x["head"])+1 for x in checks), distinct_camera_frames=cases,
        renderer=sorted(drivers), non_robot_state_poison_checks=poison,
        synthetic_support_control_passed=synthetic_control(), slots=checks,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "roster", "assets", "out"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.roster.resolve(), args.assets.resolve(), args.out.resolve())
