"""Evaluate archived startup clearance and compact grasp/lift geometry without physics."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from sea_fruit_h0_cold import pin, read, require, sha
from sea_compact_workspace import LIFT_TRAVEL_M, PROFILE, build
from sea_compact_teacher import CompactWorkspaceTeacher
from sea_stance_preflight import reached_case
from go2arx5_vla.world_model.mobile_continuous_teacher_v2 import TRANSLATION_SPEED
from go2arx5_vla.world_model.multitask_contact_v6 import observer_permitted, preview_permitted
from go2arx5_vla.world_model.multitask_state_v6 import packets
from go2arx5_vla.world_model.native_replay import forbid_integration
from scripts.pi0294.task_path_model import certify_chunk

PADS = ("task_left_pad", "task_right_pad")
TARGET = "task_cube_geom"
WORKING_OPEN = .042
WORKING_CLOSED = .002


def update_robot(m, d, packet, object_reset=None):
    passive = m.qpos0[27:] if object_reset is None else object_reset
    d.qpos[:] = np.r_[packet["qpos"][:27], passive]
    d.qvel[:] = np.r_[packet["qvel"][:26], np.zeros(12)]
    d.ctrl[:] = packet["ctrl"]
    d.time = packet["time_s"]


def startup(m, journal, tick, *, object_reset=None):
    import mujoco

    d = mujoco.MjData(m)
    names = [m.geom(i).name for i in range(m.ngeom)]
    object_geoms = [m.geom(g).id for g in (TARGET, "task_blue_cube_geom")]
    task = set(object_geoms) | {m.geom("task_table").id}
    root = m.body("base_link").id
    robot = []
    for i in range(m.ngeom):
        b = int(m.geom_bodyid[i])
        while b and b != root:
            b = int(m.body_parentid[b])
        if b == root and (m.geom_contype[i] or m.geom_conaffinity[i]):
            robot.append(i)
    clearance = float("inf")
    all_ticks = 0
    history = packets(journal) if isinstance(journal, Path) else journal
    for p in history:
        all_ticks += 1
        require(p["step"] == all_ticks <= tick, "startup journal does not cover the native clock")
        update_robot(m, d, p, object_reset)
        mujoco.mj_fwdPosition(m, d)
        for c in d.contact:
            a, b = map(int, c.geom)
            if a in task or b in task:
                names_pair = names[a], names[b]
                require(preview_permitted(*names_pair, float(c.dist), "approach", "red")
                        and observer_permitted(*names_pair, float(c.dist), "red"),
                        f"new scene startup contact tick {all_ticks}: {names_pair} {float(c.dist)} m")
        if all_ticks >= tick-500:
            for r in robot:
                for item in object_geoms:
                    clearance = min(clearance, mujoco.mj_geomDistance(m, d, r, item, .25, None))
        if all_ticks == tick:
            break
    require(all_ticks == tick and clearance >= .015,
            f"stable-entry geometry misses the original 15 mm clearance: {clearance:.6f} m")
    pose = np.eye(4)
    pose[:3, :3] = d.site("x5_ee").xmat.reshape(3, 3)
    pose[:3, 3] = d.site("x5_ee").xpos
    obj = np.eye(4)
    obj[:3, 3] = d.geom(TARGET).xpos
    mount = np.eye(4)
    mount[:3, :3] = d.body("x5_base_link").xmat.reshape(3, 3)
    mount[:3, 3] = d.body("x5_base_link").xpos
    return dict(qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=d.ctrl.copy(), tcp=pose,
                object_pose=obj, base_pose=mount, native_step=tick), dict(
                archived_startup_ticks=all_ticks, last_501_min_robot_object_clearance_mm=clearance*1000,
                forbidden_scene_contacts=0)


def h40_segments(a, b):
    """Use the original teacher speed and a smooth start/end for the dry sweep."""
    distance = np.linalg.norm(b[:3, 3]-a[:3, 3])
    steps = max(39, math.ceil(distance*1.875/(TRANSLATION_SPEED*.05)))
    samples = np.arange(39*math.ceil(steps/39)+1)
    u = np.clip(samples/steps, 0., 1.)
    ease = u**3*(10+u*(-15+6*u))
    poses = np.repeat(a[None], len(samples), axis=0)
    poses[:, :3, 3] += ease[:, None]*(b[:3, 3]-a[:3, 3])
    return [poses[i:i+40] for i in range(0, len(samples)-1, 39)]


def certify_stage(model, sample, a, b, stage, hypothesis=None):
    def permit(names, distance, phase):
        return observer_permitted(*names, distance, "red") and preview_permitted(*names, distance, phase, "red")

    proofs = [certify_chunk(model, sample, path, stage=stage, contact_rule=permit,
                            planning_object_motion=hypothesis) for path in h40_segments(a, b)]
    return dict(h40_windows=len(proofs), static_samples=sum(p["points"] for p in proofs),
                maximum_joint_speed_rad_s=max(p["max_joint_speed_rad_s"] for p in proofs),
                maximum_joint_acceleration_rad_s2=max(p["max_joint_acceleration_rad_s2"] for p in proofs))


def opposing_geometry(model, sample, target):
    import mujoco

    frozen = []

    def witness(_model, data):
        if not frozen:
            frozen.append(data.qpos.copy())

    certify_stage(model, sample, target, target, "approach", witness)
    d = mujoco.MjData(model)
    d.qpos[:] = frozen[0]
    joint = [int(model.jnt_qposadr[model.joint(f"task_{s}_slide").id]) for s in ("left", "right")]
    gripper = [model.geom(name).id for name in PADS]
    object_id = model.geom(TARGET).id
    for opening in np.linspace(WORKING_OPEN, WORKING_CLOSED, 161):
        d.qpos[joint] = opening
        mujoco.mj_fwdPosition(model, d)
        distances = [float(mujoco.mj_geomDistance(model, d, p, object_id, .08, None)) for p in gripper]
        for c in d.contact:
            names = [model.geom(int(i)).name for i in c.geom]
            require(observer_permitted(*names, float(c.dist), "red"),
                    f"forbidden close geometry {names}: {float(c.dist)} m at {opening} m")
        if all(x <= 0 for x in distances):
            require(all(x >= -.004 for x in distances), "opposing geometric witness exceeds original contact depth")
            return d.qpos.copy(), dict(nominal_per_finger_slide_mm=float(opening*1000),
                                       pad_signed_distances_mm=[1000*x for x in distances])
    raise ValueError("both pad contacts never close on selected fruit within original contact depth")


def test_geometry(m, initial, descriptor):
    import mujoco

    with forbid_integration(mujoco) as calls:
        teacher = CompactWorkspaceTeacher(initial, {"family": "F1", "mode": "M_ASSIST"},
                                          descriptor["scoring_geometry"], "HOLD-red")
        start = initial["tcp"]
        hover, close, lift = [start.copy() for _ in range(3)]
        hover[:3, 3], close[:3, 3] = teacher.hover(), teacher.grasp_goal()
        lift[:3, 3] = close[:3, 3] + [0., 0., LIFT_TRAVEL_M]
        geometry = dict(approach=certify_stage(m, initial, start, hover, "approach"),
                        descent=certify_stage(m, initial, hover, close, "approach"))
        closed, grasp = opposing_geometry(m, initial, close)
        obj_z = int(m.jnt_qposadr[m.joint("task_cube_free").id])+2
        # Move the passive fruit by the same vertical distance as the hypothetical
        # TCP. This is only a carried-geometry check on private MjData.
        def nominal_carried(_model, data):
            data.qpos[obj_z] = closed[obj_z] + data.site("x5_ee").xpos[2]-close[2, 3]

        geometry["lift"] = certify_stage(m, dict(initial, qpos=closed), close, lift, "lift", nominal_carried)
        geometry["hold"] = certify_stage(m, dict(initial, qpos=closed), lift, lift, "hold", nominal_carried)
    require(not calls, "frozen-clone qualification attempted dynamics")
    excursion = max(float(np.linalg.norm(p[:3, 3]-start[:3, 3])*1000) for p in (hover, close, lift))
    return dict(poses_world_m={k: p[:3, 3].tolist() for k, p in
                               (("start", start), ("hover", hover), ("close", close), ("lift", lift))},
                maximum_TCP_excursion_mm=excursion, geometry=geometry,
                opposing_contacts_are_geometric_only=True, close_geometry=grasp)


def evaluate(site, out, role, descriptor):
    import mujoco

    _, entry, inputs, is_fruit = reached_case(site, role)
    journal = inputs["state.bin.journal"]
    m = mujoco.MjModel.from_binary_path(str(pin(out / f"{role}.mjb", descriptor["scene_sha256"])))
    with forbid_integration(mujoco) as calls:
        initial, warmup = startup(m, journal, entry)
    require(not calls, "startup qualifications attempted dynamics")
    return dict(status="STATIC_WORKSPACE_PASSES", parent_role_is_fruit=is_fruit, entry_tick=entry,
                source_journal_sha256=sha(journal), launch_state_fixture_replayed=warmup,
                **test_geometry(m, initial, descriptor))


def run(site, scene_archive, parent, out):
    compiled = build(scene_archive, parent, out)
    tested = {}
    for role, descriptor in compiled["geometry"].items():
        try:
            tested[role] = evaluate(site, out, role, descriptor)
        except ValueError as error:
            tested[role] = dict(status="STATIC_WORKSPACE_REJECTED", reason=str(error))
    return dict(profile=PROFILE, geometry=compiled["geometry"], tests=tested,
                native_launch_ready=False, integrations=0, model_forwards=0, training_updates=0)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("site", "archive", "parent", "out"):
        cli.add_argument(f"--{key}", type=Path, required=True)
    a = cli.parse_args()
    result = run(a.site.resolve(), a.archive.resolve(), a.parent.resolve(), a.out.resolve())
    (a.out / "PREFLIGHT.json").write_text(json.dumps(result, indent=2) + "\n")
