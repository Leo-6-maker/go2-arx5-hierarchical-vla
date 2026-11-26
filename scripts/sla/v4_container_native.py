"""Bind the V4 CONTAINER teacher qualification to closed source assets and the standing executor."""

import argparse
from functools import partial
import inspect
import numpy as np
import os
import time
from pathlib import Path

from compiler import Compiler as BoundCompiler
from sea_fruit_h0_score import StationaryHoldScore
from reach_native import configure, sites
from reach_pair import environment
from sea_fruit_h0_native import drive, grip_at_grant
from sea_fruit_h0_owner import COMPACT_CAMPAIGN, site_binding
from sea_near_source_scene import PRIMARY_BOWL
from sea_near_source_task import BowlPlacement, RELEASE_OPEN_M, bowl_retreat, bowl_segments, contact_contract, qualify
from sea_near_source_centered_native import COMPILER_SHA256, FinishClockReplay
from sea_pt_nominal_codec import CLOSED_M
from sea_pt_nominal_reference import Compiler
from sea_v197_loaded import segment
from sea_v195_owner import run_card
from sea_v196_owner import CAMPAIGN as WIRE_SCOPE
from source_custody import BASE, digest, pin, read, require, write_new


OLD_CONTACT = "sea-near-source-01/E1-S25-CENTERED-NATIVE-02/E1-KNOWN-PICK-TO-BLUE-BOWL"
REGISTRY_SHA256 = "86cde1a83f2cedefbba7321b528435304f8f2ec7b884b3981ed4527686a9f9d1"
CAP = 50_000


def replay_finish(mujoco, model, site_case, geometry):
    from go2arx5_vla.world_model.multitask_state_v6 import packets
    from go2arx5_vla.world_model.native_replay import forbid_integration

    prior = read(site_case / "RESULT.private.json")
    require(prior["score"]["task_success"] and prior["score"]["target_grasp"],
            "The original held endpoint must be an actual successful grasp")
    for frame in packets(site_case / prior["episode"] / "state.bin.journal"):
        last = frame
    require(last["step"] == prior["score"]["native_step"], "Expected exact last paused state")
    state = mujoco.MjData(model)
    state.qpos[:] = last["qpos"]
    with forbid_integration(mujoco) as effects:
        mujoco.mj_fwdPosition(model, state)
        grasp = np.eye(4)
        grasp[:3, :3] = state.site("x5_ee").xmat.reshape(3, 3)
        grasp[:3, 3] = state.site("x5_ee").xpos
        target = state.geom("task_cube_geom").xpos.copy()
        motion = FinishClockReplay(last["step"], grasp, CLOSED_M)
        stages = bowl_segments(geometry, grasp, target, CLOSED_M, direct_lift=False)
        parts = (*stages, ("RETREAT", bowl_retreat(stages), RELEASE_OPEN_M, 250))
        checks = {}
        for name, pose, width, dwell in parts:
            motion.windows.clear()
            motion.jaw_seam_m_s = 0.
            segment(motion, None, pose, width, name, dwell=dwell, mapping=None, observe=lambda *_: None)
            require(np.allclose(motion.current, pose, atol=2e-7, rtol=0.)
                    and abs(motion.gripper["width_m"] - width) <= 1e-12,
                    "The frozen reference compiler did not reach the stage end")
            checks[name] = {"windows": len(motion.windows), "terminal_tick": motion.tick,
                            "max_jaw_seam_m_s": motion.jaw_seam_m_s}
    require(not effects and motion.tick <= CAP, "Compiled finish exceeded the original endpoint")
    return dict(stages=checks, assumes_nominal_tracking=True, integrations=0)


def bind(site, cold, registry, role):
    require("after_entry" in inspect.signature(drive).parameters,
            "The actual standing driver cannot enter the bowl qualification")
    require(Compiler is BoundCompiler, "The motion driver did not reuse the bound reference compiler")
    compiler = pin(Path(inspect.getfile(BoundCompiler)), COMPILER_SHA256)
    card = {"G00": "CONTAINER_Q_D01", "G01": "CONTAINER_Q_D02"}[role]
    plan = read(pin(registry, REGISTRY_SHA256))
    cell, = [row for row in plan["conditions"] if row["id"] == card]
    require(cell["block"] == "NEW_QDEV" and cell["skill"] == "CONTAINER", "Expected fixed V4 teacher roster")
    report_path = cold / "RESULT.json"
    report = read(report_path)
    require(report["kind"] == "V4_CONTAINER_COMPACT_COLD"
            and report["native_integrations"] == report["model_forwards"] == report["optimizer_updates"] == 0,
            "Unexpected cold scene work")
    # Use the first passing member in the prospective cold grid.
    selected = next(row for row in report["rows"][role]["cases"] if row["result"] == "COLD_PASS")
    scene = pin(cold / selected["scene_file"], selected["scene_sha256"])
    first = sites(site)[0][1]
    env = environment(first, site / "v19/fruit-h0-owner-environment")
    env["PYTHONPATH"] = os.pathsep.join((str(BASE), str(site / "v12"), str(site / "v12/CPU_LIB"),
                                        os.environ["PYTHONPATH"], env["PYTHONPATH"]))
    configure(dict(first, native_environment=env))
    prior = site / "v19" / COMPACT_CAMPAIGN / (role + "-H0")
    prior_lease = read(prior / "control/LEASE.json")
    require(prior_lease["status"] == "CLOSED", "Original compact owner must be closed")
    prior_spec = read(pin(prior / "EXECUTION_SPEC.json", prior_lease["spec_sha256"]))
    original, templates, _, pins = site_binding(
        site, site / "v19/SEA193_FRUIT_H0_COLD_01", Path(prior_spec["outer_library"]).parent,
        compact=Path(prior_spec["model_path"]).parent)
    _, base, old_scene, descriptor = next(row for row in templates if row[0] == role + "-H0")
    require(digest(old_scene) == report["rows"][role]["pickup_source_scene_sha256"]
            and str(old_scene) == prior_spec["model_path"]
            and descriptor["scoring_geometry"] == prior_spec["multitask_geometry"],
            "Cold pickup and native compact carrier identities differ")
    archive = site / "v19" / OLD_CONTACT
    lease = read(archive / "control/LEASE.json")
    require(lease["status"] == "CLOSED", "The bowl receiver predecessor must be closed")
    verified = read(pin(archive / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    identities = {name: verified["frozen_files"][verified[name]]
                  for name in ("outer_library", "observer_library", "trace_library")}
    receiver = pin(Path(verified["outer_library"]), identities["outer_library"])
    observer = pin(Path(verified["observer_library"]), identities["observer_library"])
    for key in ("trace_library", "controller_binary", "native_binary", "solver_binary"):
        require(base["frozen_files"][base[key]] == verified["frozen_files"][verified[key]],
                f"Frozen standing contract differs at {key}")
    import mujoco
    from go2arx5_vla.world_model.multitask_state_v6 import TwoObjectState

    model = mujoco.MjModel.from_binary_path(str(scene))
    with contact_contract((PRIMARY_BOWL,)):
        state = TwoObjectState(model, "red")
    require(model.body(PRIMARY_BOWL).id not in state.robot_bodies
            and model.body("base_link").id in state.robot_bodies,
            "Task state confused the stationary vessel with the original robot")
    geometry = selected["geometry"]
    finish = replay_finish(mujoco, model, prior, geometry)
    ready = dict(original, receiver_sha256=identities["outer_library"])
    base = dict(base, observer_library=str(observer),
                frozen_files={**base["frozen_files"], str(observer): identities["observer_library"]})
    descriptor = dict(descriptor, scene_sha256=selected["scene_sha256"])
    actual = dict(finish_clock=finish, compiler_sha256=digest(compiler), driver_sha256=digest(Path(inspect.getfile(drive))), card=card, scope="V4_CONTAINER", registry_sha256=digest(registry),
                  cold_sha256=digest(report_path), geometry=geometry, scene_sha256=digest(scene),
                  contact_predecessor_spec_sha256=lease["spec_sha256"], frozen_contact_assets=identities,
                  native_tick_cap=CAP, physics_dt_s=.002, model_forwards_max=0, training_updates_max=0,
                  controller_binding_sha256=base["frozen_files"][base["controller_binary"]],
                  legal_contact=("target fruit and its pads in the registered phases",
                                 "target fruit and designated bowl under original penetration cap"))
    return (ready, {role: (base, scene, descriptor)}, receiver, pins, None), actual


def replay_scoring(case, package, binding, role):
    import mujoco
    from go2arx5_vla.world_model.multitask_state_v6 import packets
    from go2arx5_vla.world_model.native_replay import forbid_integration

    summary = read(case / "RESULT.private.json")
    lease = read(case / "control/LEASE.json")
    closed = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    require(lease["status"] == "CLOSED" and summary["absent"] and summary["outcome"] == "TECHNICAL_STOP"
            and closed["frozen_files"][closed["model_path"]] == binding["scene_sha256"],
            "Replay only the matching settled technical interruption")
    begin = summary["entry_tick"]
    handoff = summary["qualification"]["teacher_carry_path_start_tick"]
    ending = summary["last_requested_tick"]
    _, scene, shape = package[1][role]
    model = mujoco.MjModel.from_binary_path(str(scene))
    journal = case / summary["episode"] / "state.bin.journal"
    with forbid_integration(mujoco) as calls, contact_contract((PRIMARY_BOWL,)):
        initial = None
        placement = None
        prefix = StationaryHoldScore(model, shape["scoring_geometry"])
        for packet in packets(journal):
            tick = packet["step"]
            if tick < begin:
                continue
            if tick <= handoff:
                prefix.update_packet(packet)
                if initial is None:
                    initial = prefix.last_sample
                if tick == handoff:
                    require(prefix.success, "Archived actual HOLD did not pass")
                    placement = BowlPlacement(prefix, binding["geometry"], initial, held=True)
            else:
                require(placement is not None, "Missing actual handoff")
                placement.update_packet(packet)
        require(not calls and placement is not None and placement.latest["tick"] == ending
                and placement.first_violation is None and placement.latest["native_grasp"]
                and placement.first_lift is not None and placement.first_grasp is not None,
                "Recovered readout failed to cover the consumed physical handoff")
    return dict(archived_input_sha256=digest(journal), entry_tick=begin, handoff_tick=handoff,
                terminal_tick=ending, recovered_samples=ending-handoff,
                result=placement.result(), new_integrations=0)


def run(site, cold, registry, role, destination, execute, handoff_replay):
    package, binding = bind(site, cold, registry, role)
    if handoff_replay is not None:
        binding["observed_handoff_replay"] = replay_scoring(handoff_replay, package, binding, role)
    if not execute:
        print(binding)
        return
    destination.mkdir(mode=0o700)
    record = destination / "REGISTRATION.json"
    write_new(record, binding)
    base = package[1][role][0]
    with contact_contract((PRIMARY_BOWL,)):
        result = run_card(destination, binding["card"], role, package,
                          driver=partial(qualify, geometry=binding["geometry"], retreat_after_release=True),
                          opening=grip_at_grant, maximum=CAP, scope=WIRE_SCOPE,
                          deadline=time.time() + base["wall_limit_s"] + 60,
                          metadata=dict(experiment="V4_CONTAINER", registration_sha256=digest(record)))
    write_new(destination / "CLOSED.json", result)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("site", "cold", "registry"):
        cli.add_argument(key, type=Path)
    cli.add_argument("role", choices=("G00", "G01"))
    cli.add_argument("destination", type=Path)
    cli.add_argument("--execute", action="store_true")
    cli.add_argument("--handoff-replay", type=Path)
    run(**vars(cli.parse_args()))
