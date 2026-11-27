"""Exercise the wired observer and Teacher boundary with archived native evidence."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco as mj
import numpy as np

from go2arx5_vla.world_model.multitask_adapter_v6 import MultitaskAdapter
from go2arx5_vla.world_model.multitask_scope_v6 import validate_frozen_assets
from go2arx5_vla.world_model.multitask_score_v6 import Task
from go2arx5_vla.world_model.multitask_state_v6 import TwoObjectState, packets, score_through
from go2arx5_vla.world_model.native_replay import forbid_integration
from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v64_container_episode import ContainerScore, bound_episode
from v64_container_release_contract import PLACE_RESULT, Release
from v64_container_runtime import GUARD, NATIVE, PLAN, SCENES
from v64_container_terminal import verify_bowl_surface
from v64_recovery_seal import accepted_rows, actual_queries


class TabletopControl(TwoObjectState):
    def update(self, packet):
        measured = super().update(packet)
        measured.update(container_inside=False, container_support=False)
        return measured


def boundary_replay(case, spec, terminal, folder):
    _, queries = actual_queries(case, terminal["detail"], spec)
    accepted_rows(case, queries, terminal["actual_native_integrations"])
    directory = case / "native-01"
    path = pin(spec["model_path"], spec["frozen_files"][spec["model_path"]])
    model = mj.MjModel.from_binary_path(str(path))
    reader = TabletopControl(model, "red")
    start = queries[0]["native_step"]
    snapshots = directory / "state.bin.journal"
    initial = reader.update(next(row for row in packets(snapshots) if row["step"] == start))
    score = ContainerScore(initial, spec["multitask_geometry"], "red")
    episode = bound_episode(SimpleNamespace(spec={"panel_entry_id": "EF5_Q_CONTAINER_01"}, output=folder))
    receipt = episode.window.path
    copied = folder / "MOBILE_SAMPLES.private.jsonl"
    published = []
    with (directory / "MOBILE_SAMPLES.private.jsonl").open() as source:
        atoms = {row["native_start_step"]: row for row in map(json.loads, source) if row["native_start_step"] >= start}
    stopped_at = None
    with snapshots.open("rb") as stream, copied.open("x") as observed:
        score_through(stream, reader, score, start)
        for query in queries:
            for interval in query["intervals"]:
                atom = atoms[interval["start_native_step"]]
                end = interval["end_native_step"]
                observed.write(json.dumps(atom) + "\n")
                observed.flush()
                episode.after_atom(stream, reader, score, {"native_step": end}, interval["boundary"], query["phase"])
                if receipt.is_file():
                    require(episode.window.at(end) is None and episode.window.at(end+1) is not None
                            and episode.window.at(end+26) is None, "The active receipt covers only the next atom")
                    published.append(end)
        try:
            episode.after_atom(stream, reader, score, {"native_step": end}, interval["boundary"], query["phase"])
        except ValueError:
            stopped_at = score.last
        else:
            raise ValueError("A replayed prior consume receipt must never advance the boundary")
    final = score.result()
    require(1220 == len(atoms) and final["observed_states"] == 30501 and published[0] == 33203
            and stopped_at == terminal["actual_native_integrations"] and not final["task_success"]
            and score.tracker.recontact_ticks == [33196, 33198, 33202], "Boundary replay changed the native evidence")
    return dict(actual_atoms=len(atoms), first_release_publication_tick=published[0],
                last_release_publication_tick=published[-1], original_tabletop_scored_as_container=False,
                repeated_atom_rejected_at=stopped_at, observed_states=final["observed_states"])


def measured_guard(factory, holds, baseline, observer, guard, release_proof, output):
    validate_frozen_assets(baseline)
    observer_ref = dict(path=str(pin(observer, NATIVE)), sha256=NATIVE)
    guard_ref = dict(path=str(pin(guard, GUARD)), sha256=GUARD)
    outcomes = []
    fixtures = read(release_proof)
    ticks = Release(**fixtures["release"])
    # Use the known uninterrupted separation from the paused release qualification.
    live = Release(ticks.grasp_tick, ticks.lift_tick, ticks.open_issue_tick, ticks.open_ack_tick, 33203)

    def check(layout, model, data, target, floor):
        identity = layout["id"]
        color = layout["target"]
        side = "left" if color == "red" else "right"
        path = factory / "SCENE_FACTORY_CONTAINER-PROJECT-01" / f"{identity}.mjb"
        directory = output / identity
        if not directory.exists():
            directory.mkdir()
            write(directory / "binding.json", dict(plan=PLAN, id=identity, scene=dict(path=str(path), sha256=SCENES[identity]),
                  candidate_observer=observer_ref, observer_guard=guard_ref))
        binding_path = directory / "binding.json"
        binding = dict(path=str(binding_path), sha256=sha(binding_path))
        spec = dict(baseline, panel_entry_id=identity, collection_split="QUAL", expert_training_eligible=False,
                    task_instruction_id=f"PLACE-{color}-{side}",
                    model_path=str(path), observer_library=str(observer), v64_container_qualification=binding,
                    frozen_files={**baseline["frozen_files"], str(path): SCENES[identity], str(observer): NATIVE},
                    mobile_condition={**baseline["mobile_condition"], "family": layout["family"]})
        validate_frozen_assets(spec)
        adapter = object.__new__(MultitaskAdapter)
        adapter.spec, adapter.output, adapter.task = spec, directory, Task.parse(spec["task_instruction_id"])
        environment = adapter.bind_simulator_environment({})
        window = bound_episode(adapter).window
        require(environment["WM02_V64_CONTAINER_RELEASE"] == str(window.path)
                and environment["WM02_V64_CONTAINER_CASE"] == identity, "One world and its reader must share the release path")
        reader = adapter.make_task_state(model)
        contacts = []
        for index, item in enumerate(data.contact):
            wrench = np.zeros(6)
            mj.mj_contactForce(model, data, index, wrench)
            contacts.append(dict(geom0=int(item.geom1), geom1=int(item.geom2), distance_m=float(item.dist),
                                 normal=item.frame[:3].copy(), wrench=wrench))
        packet = dict(step=33204, time_s=33204*.002, contact_time_s=33203*.002,
                      qpos=data.qpos.copy(), qvel=data.qvel.copy(), ctrl=data.ctrl.copy(), contacts=contacts)
        native = reader.update(packet)
        require(not native["safety_valid"] and native["container_support"], "Missing release preserves the stock refusal")
        window.start, window.release = 33203, live
        allowed = reader.update(packet)
        require(allowed["safety_valid"] and allowed["container_inside"] and allowed["container_support"]
                and not allowed["objects"][color]["hand_contact"], "Only released support may enter the measured state")
        foreign = dict(geom0=target, geom1=model.geom("task_left_pad").id, distance_m=-.0001,
                       normal=np.array([0., 0., 1.]), wrench=np.array([.1, 0., 0., 0., 0., 0.]))
        touched = reader.update(dict(packet, step=33205, time_s=33205*.002, contacts=contacts+[foreign]))
        retook = reader.update(dict(packet, step=33206, time_s=33206*.002))
        require(not touched["safety_valid"] and not retook["safety_valid"],
                "A recontact must invalidate the old permit in the measured reader")
        outcomes.append(identity)

    verify_bowl_surface(factory, holds, fixtures, check_contact=check)
    require(len(outcomes) == 18, "Expected all original compiled supporting states")
    return dict(compiled_states=len(outcomes), bound_wbc_preload=True, stored_permit_recontact_rejected=18)


def run(case, factory, hold_red, hold_blue, observer, guard, release_proof, output):
    output.mkdir(mode=0o700)
    closed = read(case / "control/LEASE.json")
    require(closed["status"] == "CLOSED", "Only the original closed archive may be read")
    spec = read(pin(case / "EXECUTION_SPEC.json", closed["spec_sha256"]))
    terminal = read(pin(case / "RESULT.private.json", PLACE_RESULT))
    with forbid_integration(mj) as effects:
        native = boundary_replay(case, spec, terminal, output)
        geometry = measured_guard(factory, (hold_red, hold_blue), spec, observer, guard, release_proof, output)
    require(not effects, "Qualification replay must never integrate")
    path = output / "ACTUAL.json"
    write(path, dict(teacher_after_atom=native, measured_contact_reader=geometry,
                    native_integrations=0, native_requests=0, training_admissions=0))
    print(sha(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "factory", "hold_red", "hold_blue", "observer", "guard", "release_proof", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
