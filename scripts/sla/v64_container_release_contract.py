"""Verify Mobile Container support against real release evidence and compiled bowls."""

import argparse
from dataclasses import dataclass
from itertools import product
import json
import math
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write

PLACE_RESULT = "ec7bd3b90eb3d38b50581220b590761431b4a540db3a3ed0d9158cfa11c022a1"
SUPPORT_PROBE = "67aadb05d419bbede485c04613a08347df8df3ebeea859464b813343f384e3f2"
BOWL_FLOOR = "ef5_static_bowl_base"


@dataclass(frozen=True)
class Release:
    # The completed interval supplies the OPEN acknowledgment. Neither a plan
    # nor a proposal may establish the release.
    grasp_tick: int
    lift_tick: int
    open_issue_tick: int
    open_ack_tick: int
    separation_tick: int

    def observed(self, tick):
        return (0 <= self.grasp_tick <= self.lift_tick <= self.open_issue_tick
                and self.open_issue_tick + 25 == self.open_ack_tick
                and self.open_issue_tick < self.separation_tick <= tick
                and self.open_ack_tick <= tick)


class ReleaseTracker:
    """Bind a physical release to one actual increasing OPEN atom."""

    def __init__(self, start, initial_z, selected):
        if selected not in ("red", "blue") or not math.isfinite(initial_z):
            raise ValueError("A finite initial position of the selected object is required")
        self.selected = selected
        self.initial_z = initial_z
        self.last = start - 1
        self.last_width_fraction = None
        self.grasp_tick = None
        self.lift_tick = None
        self.open_interval = None
        self.open_wire = None
        self.separation_tick = None
        self.separation_ticks = []
        self.recontact_ticks = []
        self.was_touching = False
        self.fault = None

    def invalidate(self, tick, reason):
        if self.fault is None:
            self.fault = (tick, reason)

    def consumed(self, atom, accepted, is_open):
        start = atom["native_start_step"]
        end = atom["native_end_step"]
        width = atom["installed_EEF10"][9]
        wire = atom["installed_reference_sha256"]
        if (start != self.last or end != start + 25 or atom["ticks"] != 25
                or atom["safety_valid"] is not True
                or not 0 <= width <= 1
                or not math.isclose(atom["gripper_command"]["source_native_s"], .002 * start, rel_tol=0, abs_tol=1e-8)
                or accepted["accepted_wire_sha256"] != wire
                or accepted["exact_receiver_knot"] is not True
                or not math.isclose(accepted["native_time_s"], .002 * end, rel_tol=0, abs_tol=1e-8)
                or not math.isclose(accepted["receiver_knot_time_s"], .002 * end, rel_tol=0, abs_tol=1e-8)):
            self.invalidate(self.last, "INVALID_ACTUAL_OPEN_RECEIPT")
            raise ValueError("A complete actual atom and exact accepted boundary are required")
        if (is_open and self.open_interval is None and self.lift_tick is not None
                and self.was_touching and self.last_width_fraction is not None
                and width > self.last_width_fraction + 1e-8):
            self.open_interval = start, end
            self.open_wire = wire
        elif self.open_interval is not None and width < self.last_width_fraction - 1e-8:
            self.invalidate(start, "NATIVE_GRIP_COMMAND_REVERSED_AFTER_OPEN")
        self.last_width_fraction = width

    def observe(self, sample):
        from go2arx5_vla.world_model.multitask_score_v6 import LIFT

        tick = sample["native_step"]
        target = sample["objects"][self.selected]
        z = target["pose"][2, 3]
        if tick != self.last + 1 or not math.isfinite(z):
            self.invalidate(tick, "NONCONTIGUOUS_OR_NONFINITE_STATE")
        if not sample["safety_valid"] or not sample["gripper_state_valid"]:
            self.invalidate(tick, "INVALID_SAFETY_OR_GRIPPER_STATE")
        self.last = tick
        if target["native_grasp"] and target["hand_contact"]:
            if self.grasp_tick is None:
                self.grasp_tick = tick
            if self.lift_tick is None and z - self.initial_z >= LIFT:
                self.lift_tick = tick
        touching = target["hand_contact"]
        if self.lift_tick is not None and self.was_touching and not touching:
            if self.open_interval is None or tick <= self.open_interval[0]:
                self.invalidate(tick, "SEPARATION_BEFORE_OPEN")
            elif self.separation_tick is None:
                self.separation_tick = tick
                self.separation_ticks.append(tick)
        if self.separation_tick is not None and touching:
            self.recontact_ticks.append(tick)
            self.separation_tick = None
        self.was_touching = touching

    def result(self):
        if self.fault is not None or self.separation_tick is None:
            return None
        release = Release(self.grasp_tick, self.lift_tick, *self.open_interval, self.separation_tick)
        return release if release.observed(self.last) else None


def contact_permitted(geoms, distance_m, selected, tick, release, *, inside, touching_robot):
    """Add only post-release contact with the exact static bowl floor."""
    from go2arx5_vla.world_model.multitask_contact_v6 import OBJECT_GEOMS, PENETRATION_LIMIT, observer_permitted

    if len(geoms) != 2 or not math.isfinite(distance_m):
        return False
    if observer_permitted(*geoms, distance_m, selected):
        return True
    return (frozenset(geoms) == {OBJECT_GEOMS[selected], BOWL_FLOOR}
            and distance_m >= -PENETRATION_LIMIT and inside and not touching_robot
            and release is not None and release.observed(tick))


def release_readback(case):
    import mujoco
    import numpy as np

    from go2arx5_vla.world_model.multitask_state_v6 import TwoObjectState, packets
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from v64_recovery_seal import accepted_rows, actual_queries

    lease = read(case / "control/LEASE.json")
    require(lease["status"] == "CLOSED", "The actual paired PLACE must be closed")
    spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    result = read(pin(case / "RESULT.private.json", PLACE_RESULT))
    detail = result["detail"]
    events = detail["score"]["events"]
    require(result["completed"] and result["shutdown_clean"] and not result["unsafe"]
            and detail["score"]["task_success"] and not result["optimizer_updates"], "Original complete PLACE required")
    capture, queries = actual_queries(case, detail, spec)
    accepted_rows(case, queries, result["actual_native_integrations"])
    path = case / "native-01/MOBILE_SAMPLES.private.jsonl"
    with path.open() as stream:
        cmd = {row["native_start_step"]: row for row in map(json.loads, stream)}
    first = next(q for q in queries if q["phase"] == "OPEN")
    opening_start = first["native_step"]
    prior = cmd[opening_start-25]["installed_EEF10"][9]
    open_atoms = [cmd[atom["start_native_step"]] for q in queries if q["phase"] == "OPEN" for atom in q["intervals"]]
    opening = next(atom for atom in open_atoms if atom["installed_EEF10"][9] > prior + 1e-8)
    lift_tick = events["target_lift20mm"]["native_step"]
    grasp_tick = events["target_grasp"]["native_step"]
    release_tick = events["released"]["native_step"]
    state_path = case / "native-01/state.bin.journal"
    expected = {grasp_tick, lift_tick, opening_start, opening["native_end_step"], release_tick-1, release_tick}
    model_path = spec["model_path"]
    model = mujoco.MjModel.from_binary_path(str(pin(model_path, spec["frozen_files"][model_path])))
    reader = TwoObjectState(model, "red")
    evidence = {}
    with forbid_integration(mujoco) as integrations:
        for packet in packets(state_path):
            tick = packet["step"]
            if tick > release_tick:
                break
            if tick not in expected:
                continue
            sample = reader.update(packet)
            target = sample["objects"]["red"]
            evidence[tick] = dict(grasp=target["native_grasp"], touching_robot=target["hand_contact"],
                                  opening=sample["gripper_open_fraction"], height_m=float(target["pose"][2,3]),
                                  safe=sample["safety_valid"], linear_speed_m_s=float(np.linalg.norm(
                                      target["linear_velocity"])))
    require(not integrations and set(evidence) == expected and all(item["safe"] for item in evidence.values())
            and evidence[grasp_tick]["grasp"] and evidence[lift_tick]["grasp"]
            and evidence[opening["native_end_step"]]["touching_robot"]
            and evidence[release_tick-1]["touching_robot"] and not evidence[release_tick]["touching_robot"],
            "Require an actual held-at-ACK state followed by the recorded physical separation")
    release = Release(grasp_tick, lift_tick, opening["native_start_step"], opening["native_end_step"], release_tick)
    return release, dict(wave_result_sha256=PLACE_RESULT, real_contact_ticks=evidence,
                         observations_sha256=sha(state_path), commands_sha256=sha(path),
                         capture_labels_sha256=capture["labels"]["sha256"],
                         consumed_increasing_open_atoms=sum(row["installed_EEF10"][9] > prior + 1e-8 for row in open_atoms))


def bowl_geometry(factory, holds, samples):
    import mujoco as mj
    import numpy as np

    from go2arx5_vla.world_model.native_replay import forbid_integration
    from v5_mobile_container_corridor import COLD, OBJECTS, SCORES, terminal_pose, wall_top

    compiled = factory / "SCENE_FACTORY_CONTAINER-PROJECT-01"
    layouts = read(pin(compiled / "ACTUAL.json", COLD))["rows"]
    corners = np.asarray(list(product((-1.,1.),repeat=3)))
    observed = {}
    with forbid_integration(mj) as effects:
        for layout, hold, score in zip(layouts, holds, SCORES, strict=True):
            identity = layout["id"]
            original, witness = terminal_pose(hold, score)
            model = mj.MjModel.from_binary_path(str(pin(compiled / f"{identity}.mjb", layout["compiled_sha256"])))
            state = mj.MjData(model)
            floor = model.geom(BOWL_FLOOR).id
            bowl = model.body("ef5_static_bowl").id
            require(model.geom_bodyid[floor] == bowl and model.body_jntnum[bowl] == 0
                    and model.body_parentid[bowl] == 0 and model.geom_type[floor] == mj.mjtGeom.mjGEOM_CYLINDER,
                    "The compiled floor must be the designated fixed bowl rather than a robot geom")
            objects = OBJECTS[layout["target"]]
            geom = model.geom(objects+"_geom").id
            q = model.joint(objects+"_free").qposadr[0]
            state.qpos[q+3:q+7] = original[q+3:q+7]
            mj.mj_fwdPosition(model,state)
            half = model.geom_size[geom][:3]
            rotation = state.geom_xmat[geom].reshape(3,3)
            relative_corners = (corners*half) @ rotation.T
            floor_z = wall_top(mj,model,state,floor)
            radius = model.geom_size[floor][0]
            center = state.geom_xpos[floor][:2].copy()
            for row in (s for s in samples if s["id"] == identity):
                require(witness == row["source_evidence"] and layout["compiled_sha256"] == row["scene_sha256"],
                        "Object orientation and bowl bytes must match the original contact fixtures")
                dx = np.asarray(row["offset_xy_mm"])*.001
                state.qpos[q:q+3] = (*tuple(center+dx), floor_z-relative_corners[:,2].min()-.0005)
                mj.mj_fwdPosition(model,state)
                margin = float(radius-np.linalg.norm(relative_corners[:,:2]+dx,axis=1).max())
                contacts = [c for c in state.contact if {c.geom[0],c.geom[1]} == {floor,geom} and c.dist < 0]
                require(len(contacts) == 1 and margin > 0 and abs(contacts[0].frame[2]) >= .99,
                        "Require contained footprint and vertical support on the designated floor")
                observed[(identity,*row["offset_xy_mm"])] = dict(footprint_margin_mm=1e3*margin,
                    support_normal_vertical_abs=float(abs(contacts[0].frame[2])))
    require(not effects and len(observed) == len(samples), "Expect two immutable layouts and zero physics steps")
    return observed


def run(case, support, output, factory, hold_red, hold_blue):
    from go2arx5_vla.world_model.multitask_contact_v6 import OBJECT_GEOMS, observer_permitted

    release, replay = release_readback(case)
    static = read(pin(support, SUPPORT_PROBE))
    require(static["native_integrations"] == 0 and static["floor_contact_samples"] == 18
            and static["admitted_by_existing_observer"] == 0, "The two compiled bowl fixtures are required")
    geometry = bowl_geometry(factory,(hold_red,hold_blue),static["samples"])
    samples = []
    for row in static["samples"]:
        contact, = row["contacts"]
        selected = "red" if row["id"] == "EF5_Q_CONTAINER_01" else "blue"
        other = "blue" if selected == "red" else "red"
        tick = release.separation_tick
        d = -.001*contact["penetration_mm"]
        pair = contact["geoms"]
        shape = geometry[(row["id"],*row["offset_xy_mm"])]

        def compare(geoms, dist=d, when=tick, ack=release, inside=True, robot=False):
            return contact_permitted(geoms,dist,selected,when,ack,inside=inside,touching_robot=robot)

        actual = dict(
            allowed_after_release=compare(pair, inside=shape["footprint_margin_mm"] > 0),
            allowed_reverse_order=compare(pair[::-1]),
            before_native_separation=compare(pair, when=tick-1),
            before_open_ack=compare(pair, when=release.open_ack_tick-1),
            missing_ack=compare(pair, ack=None),
            no_prior_lift=compare(pair, ack=Release(-1, -1, release.open_issue_tick, release.open_ack_tick, tick)),
            outside_bowl=compare(pair, inside=False),
            robot_still_touching=compare(pair, robot=True),
            deep_floor_penetration=compare(pair, dist=-.0041),
            wrong_cube=compare((OBJECT_GEOMS[other], BOWL_FLOOR)),
            arm_with_bowl=compare(("x5_link6", BOWL_FLOOR)),
            finger_with_bowl=compare(("task_left_pad", BOWL_FLOOR)),
            wall_with_selected_cube=compare(("ef5_static_bowl_wall_0", OBJECT_GEOMS[selected])),
        )
        positive = {"allowed_after_release","allowed_reverse_order"}
        require({key for key, value in actual.items() if value} == positive
                and not observer_permitted(*pair,d,selected),
                "Unexpected result from the task-only support rule")
        samples.append(dict(id=row["id"], scene_sha256=row["scene_sha256"],
                            offset_xy_mm=row["offset_xy_mm"], compiled_geometry=shape, checks=actual))
    write(output, dict(schema="V64_MOBILE_CONTAINER_COLD_RELEASE_SUPPORT", release=vars(release),
                       real_tabletop_place_replay=replay, static_bowl_contact_checks=samples,
                       scope="COLD_ONLY", new_integrations=0, container_successes=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "support", "output"):
        parser.add_argument(name, type=Path)
    for name in ("factory","hold-red","hold-blue"):
        parser.add_argument("--"+name, type=Path, required=True)
    run(**vars(parser.parse_args()))
