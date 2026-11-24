"""Conservative continuous extension clearance at every recorded H0 state."""

import argparse
from itertools import chain
import json
import os
from pathlib import Path

import numpy as np

from sea_mount_capture import SCREEN_SHA, candidate
from sea_visual_support import EVIDENCE, H0_DIR, H0_SHA
from source_custody import pin, read, require, write_new

HEAD_RADIUS = .030
EXTENSION_RADIUS = .012
SAMPLES = 17


def geometry_boxes(model):
    import mujoco as mj

    bounds = model.geom_aabb.copy()
    meshes = np.flatnonzero(model.geom_type == mj.mjtGeom.mjGEOM_MESH)
    for geom in meshes:
        mesh = model.geom_dataid[geom]
        start, size = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start+size]
        # Include both the collision broad-phase bounds and every compiled visual vertex.
        lower = np.minimum(bounds[geom, :3]-bounds[geom, 3:], vertices.min(axis=0))
        upper = np.maximum(bounds[geom, :3]+bounds[geom, 3:], vertices.max(axis=0))
        bounds[geom] = np.r_[(lower+upper)/2, (upper-lower)/2]
    require((bounds[:, 3:] > 0).all() and np.isfinite(bounds).all(), "unsupported frozen geometry")
    return bounds, model.geom_type == mj.mjtGeom.mjGEOM_PLANE


def point_bounds(points, data, boxes, planes):
    delta = points[:, None, :] - data.geom_xpos
    local = np.einsum("pgi,gij->pgj", delta, data.geom_xmat.reshape(-1, 3, 3))
    edge = abs(local - boxes[:, :3]) - boxes[:, 3:]
    distance = np.linalg.norm(np.maximum(edge, 0), axis=-1) + np.minimum(edge.max(axis=-1), 0)
    distance[:, planes] = local[:, planes, 2]
    return distance


def scan(model, states, start_local, target_local, end):
    import mujoco as mj

    data = mj.MjData(model)
    boxes, planes = geometry_boxes(model)
    require(planes.any(), "original floor plane missing")
    centers = np.linspace(start_local, target_local, SAMPLES)
    inflation = np.linalg.norm(target_local-start_local)/(2*(SAMPLES-1))
    base = data.body("base_link")
    worst = {"head": {"clearance_mm": float("inf")},
             "extension": {"clearance_mm": float("inf")}}
    last = None
    for tick, state in enumerate(states):
        require(state["step"] == tick <= end and len(state["qpos"]) == model.nq, "native motion ledger mismatch")
        data.qpos[:] = state["qpos"]
        mj.mj_kinematics(model, data)
        pts = centers @ base.xmat.reshape(3, 3).T + base.xpos
        bound = point_bounds(pts, data, boxes, planes)
        extension = bound - (EXTENSION_RADIUS+inflation)
        head = (bound[-1] - HEAD_RADIUS)[None]
        for kind, distances in (("head", head), ("extension", extension)):
            sample, geom = np.unravel_index(distances.argmin(), distances.shape)
            clearance = 1000*float(distances[sample, geom])
            if clearance < worst[kind]["clearance_mm"]:
                item = model.geom(geom)
                worst[kind] = dict(clearance_mm=clearance, tick=tick, geom_id=int(geom),
                                   geom=item.name or None, geom_body=model.body(item.bodyid[0]).name,
                                   offset_index=int(sample) if kind == "extension" else SAMPLES-1)
        if tick == 0:
            p = data.geom_xpos[planes][0] - data.geom_xmat[planes][0].reshape(3, 3)[:, 2]*.05
            require(np.min(point_bounds(p[None], data, boxes, planes)) < -.0499,
                    "below-floor negative control was not rejected")
        last = tick
    require(last == end, "native motion replay stopped early")
    return dict(recorded_2ms_ticks=last, reset_sample_checks=1, worst=worst,
                conservative_proxy_clear=all(x["clearance_mm"] > 0 for x in worst.values()))


def run(root, screen, output):
    import mujoco as mj
    from go2arx5_vla.world_model.multitask_state_v6 import packets
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from sea_three_view import POSITION_M

    scope = read(pin(EVIDENCE / "GRIP_H0/COMPACT_H0_ACTUAL.json", H0_SHA))
    panel = root / "v19" / H0_DIR
    pin(panel / "PANEL_CLOSE.json", scope["campaign_sha256"]["PANEL_CLOSE.json"])
    old, target = np.asarray(POSITION_M), np.asarray(candidate(screen)["base_from_camera"])[:3, 3]
    report = {}
    with forbid_integration(mj) as attempted:
        for layout, row in scope["rows"].items():
            case = panel / layout
            seals = row["evidence_sha256"]
            lease = read(pin(case / "control/LEASE.json", seals["LEASE.json"]))
            actual = read(pin(case / "RESULT.private.json", seals["RESULT.private.json"]))
            spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
            require(lease["status"] == "CLOSED" and row["qualified"] and actual["absent"], "open H0 owner")
            scene = Path(spec["model_path"])
            native = case / actual["episode"]
            journal = pin(native / "state.bin.journal", seals["state.bin.journal"])
            model = mj.MjModel.from_binary_path(str(pin(scene, spec["frozen_files"][str(scene)])))
            with (native / "initialization.jsonl").open() as file:
                initial = json.loads(next(file))
            require(initial["kind"] == "actual_startup_V1" and initial["time"] == 0., "original reset absent")
            result = scan(model, chain([dict(step=0, qpos=initial["qpos"])], packets(journal)),
                          old, target, row["physical_ticks"])
            report[layout] = dict(result, journal_sha256=seals["state.bin.journal"],
                                  scene_sha256=spec["frozen_files"][str(scene)])
    require(not attempted, "clearance replay made a physics step")
    write_new(output, dict(
        status="RECORDED_H0_MOUNT_EXTENSION_CLEARANCE", compact_h0_sha256=H0_SHA, screen_sha256=SCREEN_SHA,
        fixed_center_before_m=old.tolist(), fixed_center_after_m=target.tolist(), rows=report,
        camera_optical_center_sphere_radius_mm=1000*HEAD_RADIUS, extension_capsule_radius_mm=1000*EXTENSION_RADIUS,
        extension_samples=SAMPLES, interpolation_between_native_packets_proven=False,
        complete_hardware_bracket_qualified=False, new_rgb_renders=0, new_physics_integrations=0,
        optimizer_updates=0, model_forwards=0,
    ))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("root", "screen", "output"):
        p.add_argument(arg, type=Path)
    a = p.parse_args()
    os.umask(0o077)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU clearance only")
    run(a.root.resolve(), a.screen, a.output)
