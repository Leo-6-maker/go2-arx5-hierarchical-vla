"""Score measured post-release support, retreat and dwell on the native clock."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v64_container_release_contract import BOWL_FLOOR, PLACE_RESULT, Release


@dataclass(frozen=True)
class TerminalSample:
    tick: int
    time_s: float
    safe: bool
    inside: bool
    support: bool
    separated: bool
    opening: float
    retreat_m: float
    speed_m_s: float
    approach_m: float


class TerminalScore:
    """Use the original PLACE thresholds on the selected support surface."""

    def __init__(self, release, first_tick):
        self.release = release
        self.first_tick = first_tick
        self.last_tick = None
        self.stable_since = None
        self.stable_ticks = 0
        self.longest_ticks = 0
        self.first_success_tick = None
        self.first_support_tick = None
        self.unsafe = 0
        self.gaps = 0
        self.samples = 0
        self.success = False

    def update(self, sample):
        from go2arx5_vla.world_model.multitask_score_v6 import BASE_MOTION, DT, HOLD_SECONDS, OPEN_FRACTION, RETREAT, STABLE_SPEED

        expected = self.first_tick if self.last_tick is None else self.last_tick + 1
        require(type(sample.tick) is int and (self.last_tick is None or sample.tick > self.last_tick)
                and np.isclose(sample.time_s,sample.tick*DT,rtol=0,atol=1e-8)
                and np.isfinite([sample.opening,sample.retreat_m,sample.speed_m_s,sample.approach_m]).all(),
                "Native terminal scoring needs finite measurements and monotone 2 ms packets")
        self.gaps += sample.tick != expected
        self.unsafe += not sample.safe
        self.samples += 1
        self.last_tick = sample.tick
        release_seen = self.release is not None and self.release.observed(sample.tick)
        supported = release_seen and sample.inside and sample.support and sample.separated
        if supported and self.first_support_tick is None:
            self.first_support_tick = sample.tick
        stable = (supported and sample.opening >= OPEN_FRACTION and sample.retreat_m >= RETREAT
                  and sample.speed_m_s < STABLE_SPEED and not self.gaps and not self.unsafe)
        if stable:
            if self.stable_since is None:
                self.stable_since = sample.tick
            self.stable_ticks = sample.tick - self.stable_since
            self.longest_ticks = max(self.longest_ticks,self.stable_ticks)
        else:
            self.stable_since = None
            self.stable_ticks = 0
        self.success = bool(stable and self.stable_ticks >= round(HOLD_SECONDS/DT) and sample.approach_m >= BASE_MOTION)
        if self.success and self.first_success_tick is None:
            self.first_success_tick = sample.tick

    def result(self):
        return dict(task_success=self.success, observed_states=self.samples, first_support_tick=self.first_support_tick,
                    first_success_tick=self.first_success_tick, longest_stable_s=.002*self.longest_ticks,
                    unsafe_ticks=self.unsafe, missing_intervals=self.gaps)


def bowl_support(model, data, packet, target_geom):
    """Require full XY containment and a measured upward contact on the bound floor."""
    from itertools import product
    import mujoco as mj

    floor = model.geom(BOWL_FLOOR).id
    require(model.geom_type[floor] == mj.mjtGeom.mjGEOM_CYLINDER
            and model.geom_bodyid[floor] == model.body("ef5_static_bowl").id
            and model.geom_type[target_geom] == mj.mjtGeom.mjGEOM_BOX, "Exact configured container and target geometry required")
    half = model.geom_size[target_geom][:3]
    corners = np.asarray(list(product((-1.,1.),repeat=3))) * half
    rotation = data.geom_xmat[target_geom].reshape(3,3)
    xy = (corners @ rotation.T + data.geom_xpos[target_geom])[:,:2] - data.geom_xpos[floor,:2]
    inside = bool(np.linalg.norm(xy,axis=1).max() <= model.geom_size[floor][0])
    support = any({c["geom0"],c["geom1"]} == {floor,target_geom} and c["wrench"][0] >= .05
                  and (c["normal"][2] if c["geom1"] == target_geom else -c["normal"][2]) >= .99
                  for c in packet["contacts"])
    return inside, support


def verify_bowl_surface(factory, holds, proof):
    import mujoco as mj

    from go2arx5_vla.world_model.native_replay import forbid_integration
    from v5_mobile_container_corridor import COLD, OBJECTS, SCORES, terminal_pose, wall_top

    compiled = factory/"SCENE_FACTORY_CONTAINER-PROJECT-01"
    layouts = read(pin(compiled/"ACTUAL.json",COLD))["rows"]
    evidence = []
    with forbid_integration(mj) as effects:
        for layout, hold, result in zip(layouts,holds,SCORES,strict=True):
            identity = layout["id"]
            model = mj.MjModel.from_binary_path(str(pin(compiled/f"{identity}.mjb",layout["compiled_sha256"])))
            data = mj.MjData(model)
            obj = OBJECTS[layout["target"]]
            floor, target = model.geom(BOWL_FLOOR).id, model.geom(obj+"_geom").id
            q = model.joint(obj+"_free").qposadr[0]
            held, _ = terminal_pose(hold,result)
            data.qpos[q+3:q+7] = held[q+3:q+7]
            mj.mj_fwdPosition(model,data)
            center = data.geom_xpos[floor,:2].copy()
            z = wall_top(mj,model,data,floor)
            vertical_half = abs(data.geom_xmat[target].reshape(3,3)[2]) @ model.geom_size[target][:3]
            for row in (x for x in proof["static_bowl_contact_checks"] if x["id"] == identity):
                offset = 1e-3*np.asarray(row["offset_xy_mm"])
                data.qpos[q:q+3] = (*tuple(center+offset),z+vertical_half-.0005)
                mj.mj_forward(model,data)
                contacts = []
                for index, contact in enumerate(data.contact):
                    if {contact.geom1,contact.geom2} != {floor,target}:
                        continue
                    wrench = np.zeros(6)
                    mj.mj_contactForce(model,data,index,wrench)
                    contacts.append(dict(geom0=contact.geom1,geom1=contact.geom2,normal=contact.frame[:3].copy(),
                                         wrench=wrench))
                inside,support = bowl_support(model,data,{"contacts":contacts},target)
                inside_empty,support_empty = bowl_support(model,data,{"contacts":[]},target)
                require(inside and support and inside_empty and not support_empty,
                        "The floor adapter must distinguish actual solved contact force from geometry only")
                evidence.append(dict(id=identity,offset_xy_mm=row["offset_xy_mm"],
                                     static_normal_load_n=sum(float(c["wrench"][0]) for c in contacts)))
    require(not effects and len(evidence) == 18, "Expect the original 18 cold states and no integration")
    return evidence


def replay(case, release_proof, output, factory, hold_red, hold_blue):
    from dataclasses import replace
    import mujoco as mj

    from go2arx5_vla.world_model.multitask_score_v6 import TABLE_TOLERANCE, footprint_inside
    from go2arx5_vla.world_model.multitask_state_v6 import TwoObjectState, packets
    from go2arx5_vla.world_model.native_replay import forbid_integration

    lease = read(case/"control/LEASE.json")
    require(lease["status"] == "CLOSED", "The observed positive control must be closed")
    spec = read(pin(case/"EXECUTION_SPEC.json",lease["spec_sha256"]))
    recorded = read(pin(case/"RESULT.private.json",PLACE_RESULT))
    proof = read(pin(release_proof,"3654c07cd2e3663b2e670ce6b5d7c41ee3cbb7d1ca6975084f8dd9419777dc2e"))
    require(proof["real_tabletop_place_replay"]["wave_result_sha256"] == PLACE_RESULT, "The release belongs to this run")
    release = Release(**proof["release"])
    geometry = spec["multitask_geometry"]
    direction = np.asarray(geometry["direction_world_XY"])
    source = case/"native-01/state.bin.journal"
    model = mj.MjModel.from_binary_path(str(pin(spec["model_path"],spec["frozen_files"][spec["model_path"]])))
    reader = TwoObjectState(model,"red")
    first = release.separation_tick-1
    switches = ("measured_table_control","missing_bowl_support","missing_ack",
                "robot_still_touching","no_retreat","unstable_velocity","unsafe","missing_tick")
    checks = {key:TerminalScore(None if key == "missing_ack" else release, first) for key in switches}
    with forbid_integration(mj) as effects:
        origin = None
        for packet in packets(source):
            tick = packet["step"]
            if tick == 5703:
                origin = reader.update(packet)["base_pose"][:2,3].copy()
            if tick < first:
                continue
            sample = reader.update(packet)
            target = sample["objects"]["red"]
            pose = target["pose"]
            half = np.asarray(geometry["object_half_sizes"]["red"])
            bottom = pose[2,3] - np.sum(abs(pose[2,:3])*half)
            captured = TerminalSample(tick, sample["time_s"], sample["safety_valid"],
                footprint_inside(pose,half,geometry["zones"]["right"]),
                abs(bottom-geometry["table_top_world_z"]) <= TABLE_TOLERANCE and target["table_contact"],
                not (target["native_grasp"] or target["hand_contact"]),sample["gripper_open_fraction"],
                float(np.linalg.norm(sample["tcp"][:3,3]-pose[:3,3])),
                float(np.linalg.norm(target["linear_velocity"])),
                float((sample["base_pose"][:2,3]-origin) @ direction))
            checks["measured_table_control"].update(captured)
            checks["missing_bowl_support"].update(replace(captured,support=False))
            checks["missing_ack"].update(captured)
            checks["robot_still_touching"].update(replace(captured,separated=False))
            checks["no_retreat"].update(replace(captured,retreat_m=0.))
            checks["unstable_velocity"].update(replace(captured,speed_m_s=.03))
            checks["unsafe"].update(replace(captured,safe=False) if tick == first else captured)
            if tick != first+1:
                checks["missing_tick"].update(captured)
    actual = {k:v.result() for k,v in checks.items()}
    post = actual["measured_table_control"]
    old = recorded["detail"]["score"]["longest_stable_released_retreat_s"]
    require(not effects and post["task_success"] and abs(post["longest_stable_s"]-old) < 1e-8
            and not any(r["task_success"] for k,r in actual.items() if k != "measured_table_control"),
            "Positive control and required terminal rejection cases must agree with the actual trace")
    support = verify_bowl_surface(factory,(hold_red,hold_blue),proof)
    write(output, dict(schema="V64_CONTAINER_TERMINAL_COLD_REPLAY", native_integrations=0, container_successes=0,
                       actual_table_control=PLACE_RESULT, release_proof=sha(release_proof),
                       actual_original_stable_s=old, verification=actual, cold_solved_floor_load=support))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case","release_proof","output"):
        parser.add_argument(name,type=Path)
    for name in ("factory","hold-red","hold-blue"):
        parser.add_argument("--"+name,type=Path,required=True)
    replay(**vars(parser.parse_args()))
