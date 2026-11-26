"""Screen compact standing fruit-to-container scenes on closed MuJoCo state records."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from go2arx5_vla.world_model.multitask_scene_v6 import overlay_xml
from go2arx5_vla.world_model.multitask_state_v6 import packets
from go2arx5_vla.world_model.native_replay import forbid_integration
from sea_compact_workspace import DEV_LAYOUTS, compact_xml, fruit_xml, source, verify_frozen
from sea_near_source_cold_route import segment
from sea_near_source_scene import COLORS, FRUIT, PRIMARY_BOWL, bowl, verify_robot
from sea_near_source_shortlist import opening_envelope
from sea_near_source_task import contact_contract
from sea_near_source_workspace import release_retreat
from source_custody import digest, pin, read, require, write_new


POSITIONS_M = ((0., -.080), (.055, -.060), (-.045, -.067))
OPPOSITE_M = ((.055, .060), (.075, .050), (.080, 0.))
RIM_RADIUS_M = .046
RIM_HEIGHT_M = .016


def candidate_scene(mj, xml, old, center, top):
    table = old.geom("task_table")
    require(np.all(np.abs(center - table.pos[:2]) + RIM_RADIUS_M <= table.size[:2]),
            "Bowl footprint extends beyond the original tabletop")
    root = ET.fromstring(xml)
    geometry = bowl(root.find("worldbody"), root.find("asset"), name=PRIMARY_BOWL,
                    xy=center, top=top, bottom_radius=.031, rim_radius=RIM_RADIUS_M,
                    height=RIM_HEIGHT_M, color=COLORS["blue"])
    model = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    verify_robot(old, model)
    return model, {"blue_bowl": geometry}


def load_sample(mj, model, packet):
    data = mj.MjData(model)
    for key in ("qpos", "qvel", "ctrl"):
        getattr(data, key)[:] = packet[key]
    mj.mj_fwdPosition(model, data)
    return data


def check_prefix(mj, model, states):
    data = mj.MjData(model)
    bowl_id = model.body(PRIMARY_BOWL).id
    bowl_geoms = set(np.flatnonzero(model.geom_bodyid == bowl_id))
    for row in states:
        data.qpos[:] = row["qpos"]
        mj.mj_fwdPosition(model, data)
        for contact in data.contact:
            ids = set(map(int, contact.geom))
            if ids & bowl_geoms:
                names = [model.geom(i).name for i in contact.geom]
                raise ValueError(f"New bowl contacts {names} at archived tick {row['step']}")
    return len(states)


def finish(mj, model, state, geometry):
    d = load_sample(mj, model, state)
    object_position = d.geom(FRUIT).xpos.copy()
    current = np.eye(4)
    current[:3, :3] = d.site("x5_ee").xmat.reshape(3, 3)
    current[:3, 3] = d.site("x5_ee").xpos
    offset = object_position - current[:3, 3]
    sample = dict(state, tcp=current, native_step=state["step"])
    goal = np.asarray(geometry["blue_bowl"]["center_world_xy_m"])
    rise = max(.014, geometry["blue_bowl"]["rim_z_m"] + .047 - object_position[2])
    require(rise <= .040, "Additional post-grasp rise exceeds existing bowl path envelope")
    lift = current.copy()
    lift[2, 3] += rise
    place = lift.copy()
    place[:2, 3] += goal - object_position[:2]
    raised, lift_check = segment(model, sample, current, lift, "lift", attached_offset=offset)
    carried, carry_check = segment(model, raised, lift, place, "hold", attached_offset=offset)
    opening = opening_envelope(mj, model, carried["qpos"], goal, geometry)
    require(opening["entire_static_opening_passed"], "Nominal carried fruit or opening misses bowl geometry")
    route = [{"intended_tool_m": row[:3, 3]} for row in (lift, place)]
    retreat = release_retreat(mj, model, carried["qpos"], goal, geometry, route)
    require(retreat is None, f"Nominal retreat prohibited: {retreat}")
    reset = mj.MjData(model)
    reset.qpos[27:30] = [*goal, geometry["blue_bowl"]["floor_z_m"] + model.geom(FRUIT).size[2] - .0003]
    mj.mj_fwdPosition(model, reset)
    contacts = [{model.geom(i).name for i in c.geom} for c in reset.contact]
    require({FRUIT, PRIMARY_BOWL + "_base"} in contacts and {FRUIT, "task_table"} not in contacts,
            "Released fruit lacks distinct container support")
    return dict(lift=lift_check, carry=carry_check, opening=opening, support_surface=PRIMARY_BOWL+"_base",
                additional_rise_m=rise, carry_xy_m=float(np.linalg.norm(goal-object_position[:2])),
                nominal_retreat_m=.020)


def run(site, archive, output, opposite):
    import mujoco as mj

    require(mj.__version__ == "3.2.7", "Expected the original CPU MuJoCo version")
    v19 = site / "v19"
    previous = v19 / "V193-COMPACT-FRUIT-H0-01"
    compact = v19 / "SEA193_COMPACT_SCREEN_02"
    chosen = read(compact / "SCREEN.json")["selected"]
    original, template, roster, _ = source(archive, v19 / "SEA193_FRUIT_H0_COLD_01")
    output.mkdir(mode=0o700)
    rows = {}
    with forbid_integration(mj) as calls, contact_contract():
        for role, name in zip(("G00", "G01"), DEV_LAYOUTS, strict=True):
            if opposite and role != "G01":
                continue
            case = previous / (role + "-H0")
            lease = read(case / "control/LEASE.json")
            summary = read(case / "RESULT.private.json")
            require(lease["status"] == "CLOSED" and summary["absent"] and summary["score"]["task_success"],
                    "Expected a settled, real standing fruit hold")
            spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
            record = chosen[role]
            old_path = pin(Path(spec["model_path"]), record["scene_sha256"])
            old = mj.MjModel.from_binary_path(str(old_path))
            journal = case / summary["episode"] / "state.bin.journal"
            states = list(packets(journal))
            require(states[-1]["step"] == summary["score"]["native_step"]
                    and all(p["step"] == n for n, p in enumerate(states, 1)),
                    "Full actual positive pickup trajectory must be continuous")
            xml, base = overlay_xml(template, original, roster[name])
            xyz, top = record["positions_world_m"], record["table_top_m"]
            xml = compact_xml(fruit_xml(xml), base["table_top_world_z"], xyz, top=top)
            verify_frozen(old, mj.MjModel.from_xml_string(xml))
            results = []
            for offset in OPPOSITE_M if opposite else POSITIONS_M:
                center = np.asarray(xyz["red"][:2]) + offset
                row = dict(offset_m=offset, bowl_xy_m=center.tolist())
                results.append(row)
                stage = "scene"
                try:
                    m, geometry = candidate_scene(mj, xml, old, center, top)
                    stage = "archived_pickup"
                    row["replayed_pickup_ticks"] = check_prefix(mj, m, states)
                    stage = "lift_transport_release_retreat"
                    row["geometry"] = geometry
                    row["finish"] = finish(mj, m, states[-1], geometry)
                except (ValueError, AssertionError) as exc:
                    row.update(result="REJECTED", stage=stage, reason=str(exc))
                else:
                    path = output / f"{role}-{len(results)}.mjb"
                    mj.mj_saveModel(m, str(path), None)
                    row.update(result="COLD_PASS", scene_file=path.name, scene_sha256=digest(path))
            rows[role] = dict(cases=results, pickup_source_scene_sha256=digest(old_path),
                              pickup_archive_sha256=digest(journal))
    require(not calls, "Cold inspection advanced physics")
    result = dict(kind="V4_CONTAINER_COMPACT_COLD", rows=rows, native_integrations=0,
                  model_forwards=0, optimizer_updates=0, native_qualification=False)
    write_new(output / "RESULT.json", result)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("site", "archive", "output"):
        cli.add_argument(key, type=Path)
    cli.add_argument("--opposite", action="store_true")
    run(**vars(cli.parse_args()))
