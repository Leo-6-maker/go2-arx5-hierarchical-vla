"""Use the registered Container role at the original paused Teacher boundaries."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from go2arx5_vla.world_model.multitask_score_v6 import LIFT
from go2arx5_vla.world_model.multitask_state_v6 import TwoObjectState, score_through
from v64_container_release_contract import BOWL_FLOOR, ReleaseTracker, contact_permitted, publish_release
from v64_container_teacher import QUALIFICATION, ContainerTeacher
from v64_container_terminal import TerminalSample, TerminalScore, bowl_support


class ReleaseWindow:
    def __init__(self, condition, output):
        self.condition = condition
        self.path = Path(output) / "CONTAINER_RELEASE.private"
        if self.path.exists():
            raise PermissionError("The Container world must start without a prior release receipt")
        self.start = None
        self.release = None

    def at(self, tick):
        if self.start is not None and self.start < tick <= self.start + 25:
            return self.release
        return None

    def advance(self, tick, tracker):
        active = publish_release(self.path, self.condition, tick, tracker)
        self.start = tick
        self.release = tracker.result() if active else None


class ContainerState(TwoObjectState):
    """Retain the stock state decoder and recognise the authorised static support."""

    def __init__(self, model, selected, window):
        super().__init__(model, selected)
        self.window = window
        self.bowl_body = model.body("ef5_static_bowl").id
        self.floor = model.geom(BOWL_FLOOR).id
        self.robot_bodies.discard(self.bowl_body)
        self.foreign_contact_tick = -1

    def update(self, packet):
        sample = super().update(packet)
        step = sample["native_step"]
        target = self.cube_geom
        pairs = [(c["geom0"], c["geom1"], c["distance_m"]) for c in packet["contacts"]]
        foreign = any(target in (a, b) and self.floor not in (a, b) for a, b, _ in pairs)
        if foreign:
            self.foreign_contact_tick = max(self.foreign_contact_tick, step)
        inside, support = bowl_support(self.model, self.data, packet, target)
        sample.update(container_inside=inside, container_support=support)
        release = self.window.at(step)
        if (release is not None and release.separation_tick > self.foreign_contact_tick
                and any({a, b} == {target, self.floor} for a, b, _ in pairs)):
            allowed = all(contact_permitted((self.model.geom(a).name, self.model.geom(b).name), distance,
                                           self.selected, step, release, inside=inside, touching_robot=foreign)
                          for a, b, distance in pairs)
            sample.update(safety_valid=allowed, illegal_contact=not allowed)
        return sample


class ContainerScore:
    def __init__(self, initial, geometry, color):
        self.start = initial["native_step"]
        self.last = None
        self.color = color
        self.z = initial["objects"][color]["pose"][2, 3]
        self.base = initial["base_pose"][:2, 3].copy()
        self.direction = np.asarray(geometry["direction_world_XY"])
        self.tracker = ReleaseTracker(self.start, self.z, color)
        self.terminal = TerminalScore(None, self.start)
        self.peak_lift = 0.
        self.keep_since = None
        self.longest_hold = 0

    def update(self, sample):
        tick = sample["native_step"]
        self.last = tick
        self.tracker.observe(sample)
        target = sample["objects"][self.color]
        if target["native_grasp"]:
            lift = target["pose"][2, 3] - self.z
            self.peak_lift = max(self.peak_lift, lift)
            if lift >= LIFT:
                if self.keep_since is None:
                    self.keep_since = tick
                self.longest_hold = max(self.longest_hold, tick - self.keep_since)
            else:
                self.keep_since = None
        else:
            self.keep_since = None
        self.terminal.release = self.tracker.result()
        self.terminal.update(TerminalSample(
            tick=tick, time_s=sample["time_s"], safe=sample["safety_valid"] and sample["gripper_state_valid"],
            inside=sample["container_inside"], support=sample["container_support"],
            separated=not (target["native_grasp"] or target["hand_contact"]),
            opening=sample["gripper_open_fraction"],
            retreat_m=float(np.linalg.norm(sample["tcp"][:3, 3] - target["pose"][:3, 3])),
            speed_m_s=float(np.linalg.norm(target["linear_velocity"])),
            approach_m=float((sample["base_pose"][:2, 3] - self.base) @ self.direction)))

    def result(self):
        release = self.tracker.result()
        return dict(self.terminal.result(), native_step=self.last, ever_target_grasp=self.tracker.grasp_tick is not None,
                    maximum_lift_m=float(self.peak_lift), longest_grasp_lift_hold_s=.002 * self.longest_hold,
                    release=asdict(release) if release is not None else None,
                    release_fault=self.tracker.fault)


class ContainerEpisode:
    def __init__(self, adapter):
        self.adapter = adapter
        self.identity = adapter.spec["panel_entry_id"]
        self.window = ReleaseWindow(self.identity, adapter.output)

    def state_reader(self, model):
        return ContainerState(model, self.adapter.task.color, self.window)

    def initialize(self, initial):
        spec = self.adapter.spec
        model = self.adapter.actual_state.model
        data = self.adapter.actual_state.data
        bowl = model.body("ef5_static_bowl").id
        center = data.xpos[bowl, :2].copy()
        rim = bowl_rim_z(model, data, bowl)
        geometry = spec["multitask_geometry"]
        return (ContainerTeacher(initial, spec["mobile_condition"], geometry, self.identity, center, rim),
                ContainerScore(initial, geometry, self.adapter.task.color), self.state_reader(model))

    def after_atom(self, stream, reader, score, sample, witness, stage):
        """Process the finished atom before writing authority for the next one."""
        tick = sample["native_step"]
        log = self.adapter.output / "MOBILE_SAMPLES.private.jsonl"
        with log.open("rb") as source:
            source.seek(max(0, log.stat().st_size - 16384))
            atom = json.loads(source.read().splitlines()[-1])
        score.tracker.consumed(atom, witness, stage == "OPEN")
        score_through(stream, reader, score, tick)
        self.window.advance(tick, score.tracker)


def bowl_rim_z(model, data, bowl):
    import mujoco as mj

    heights = []
    for g in range(model.ngeom):
        if not model.geom_contype[g] or model.geom_bodyid[g] != bowl:
            continue
        if g == model.geom(BOWL_FLOOR).id:
            continue
        if model.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
            raise ValueError("The compiled high-release layout requires the original mesh bowl walls")
        mesh = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        rotation = data.geom_xmat[g].reshape(3, 3)
        heights.append(float(data.geom_xpos[g, 2] + (model.mesh_vert[start:start+count] @ rotation[2]).max()))
    if len(heights) != 24:
        raise ValueError("The exact 24 compiled bowl-wall pieces are required")
    return max(heights)


def bound_episode(adapter):
    cached = getattr(adapter, "_container_episode", None)
    if cached is None:
        cached = adapter._container_episode = ContainerEpisode(adapter)
    return cached


def run(adapter):
    from go2arx5_vla.world_model.multitask_teacher_episode_v6 import episode

    spec = adapter.spec
    family, color, side = QUALIFICATION[spec["panel_entry_id"]]
    if (spec["collection_split"] != "QUAL" or spec["expert_training_eligible"]
            or spec["mobile_condition"]["family"] != family or adapter.task.instruction != f"PLACE-{color}-{side}"):
        raise PermissionError("The independent Container qualification and transport must match")
    result = episode(adapter, container=bound_episode(adapter))
    passed = result["task_success"]
    return dict(result, schema="V64_MFREE_CONTAINER_QUAL_ACTUAL", task_success=False,
                container_qualification_success=passed, qualification_instruction=f"CONTAINER-{color}-{side}",
                training_admission=False)
