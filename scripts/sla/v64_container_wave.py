"""Register the two previously screened Container scenes in the original N1 owner."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from v64_container_runtime import GUARD, NATIVE, PLAN, SCENES, install_runtime
from v64_container_teacher import QUALIFICATION


PRIOR = "9cb51f8ae4ccee4d7d6fe7b74bf3da335cbd5d5018e80cd5bafd4d93bff4764d"
COLD = "3d9ad98d7b92e36b4ca10eaf7232370af9f9ff2d2908b169ac6b1b465aa87cd3"
S1 = "f10b07348ad2746876a3e3ba1399a4b3b950ae5875c965f7c1d909d07b50df39"
REPLAY = "3d6fef4621af5edd49448a9d16899b6d14a6c0e465b2cd287272fd10ec07d8d3"
CAP = 50703


def rows(population, roster):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin

    proof = read_pin(roster["boundary_replay"])
    selected = population["conditions"]["QUAL"]
    if (roster["plan_v64"] != PLAN or roster["split"] != "QUAL" or roster["physics_requests"] != 2
            or roster["integrations"] != 2*CAP or roster["boundary_replay"]["sha256"] != REPLAY
            or proof["native_integrations"] or proof["teacher_after_atom"]["actual_atoms"] != 1220
            or proof["measured_contact_reader"]["compiled_states"] != 18
            or population["source_geometry_sha256"] != COLD
            or list(QUALIFICATION) != [item["id"] for item in selected]
            or {item["id"]: item["scene"]["sha256"] for item in population["clusters"]} != SCENES
            or set(roster["case_bindings"]) != set(QUALIFICATION)):
        raise PermissionError("Only the two screened Container qualification scenes are registered")
    for item in selected:
        family, color, side = QUALIFICATION[item["id"]]
        if (item["family"] != family or item["instruction"] != f"PLACE-{color}-{side}"
                or item["split"] != "QUAL" or item["native_max_ticks"] != CAP):
            raise PermissionError("The Container qualification cannot use alternate conditions")
    return selected


def bind(case, selection):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin

    roster = read_pin(selection)
    if roster["variant"] != "CONTAINER_QUAL":
        return
    ref = roster["case_bindings"][case["panel_entry_id"]]
    bound = read_pin(ref)
    case["v64_container_qualification"] = ref
    observer = bound["candidate_observer"]
    case["observer_library"] = observer["path"]
    case["frozen_files"][observer["path"]] = observer["sha256"]
    _, color, _ = QUALIFICATION[case["panel_entry_id"]]
    case["language"] = f"Pick up the {color} cube, carry it over the bowl, release it into the bowl and withdraw."


PATCHES = {
    "v62_mobile_qualification.py": [
        ('    split, identities, geometry = SCOPES[variant]\n    if variant == "DIVERSE_PLACE_WAVE7":',
         '    if variant == "CONTAINER_QUAL":\n'
         '        from scripts.sla.v64_container_wave import rows as container_rows\n'
         '        return container_rows(population,roster)\n'
         '    split, identities, geometry = SCOPES[variant]\n    if variant == "DIVERSE_PLACE_WAVE7":'),
        ('def capacity_file(root):\n',
         'def capacity_file(root):\n    if root.name.startswith("CONTAINER-QUAL-"):\n'
         '        return root.parent / "CONTAINER-QUAL-CAPACITY.json"\n'),
    ],
    "core_hopper_expert.py": [
        ('    request_cap, tick_cap = (1, 50703) if binding else (264, 12065592)\n',
         '    request_cap, tick_cap = (1, 50703) if binding else (264, 12065592)\n'
         '    if binding and read_pin(binding)["variant"] == "CONTAINER_QUAL":\n'
         '        request_cap, tick_cap = 2, 101406\n'),
    ],
    "core_hopper_register.py": [
        ("    spec=dict(interface,source_sha=source_sha,",
         "    if selection and read_pin(selection)['variant']=='CONTAINER_QUAL':\n"
         "        env['PYTHONPATH']+=os.pathsep+str(source/'scripts/sla')\n"
         "    spec=dict(interface,source_sha=source_sha,"),
        ("            case['v62_mobile_qualification']=selection\n        validate(case)\n",
         "            case['v62_mobile_qualification']=selection\n"
         "            from scripts.sla.v64_container_wave import bind\n"
         "            bind(case,selection)\n        validate(case)\n"),
    ],
    "core_hopper_collection.py": [
        ("            case['v62_mobile_qualification']=selection\n        require_unstarted_cases",
         "            case['v62_mobile_qualification']=selection\n"
         "            from scripts.sla.v64_container_wave import bind\n"
         "            bind(case,selection)\n        require_unstarted_cases"),
        ("            records.append(rec)\n",
         "            if case.get('v64_container_qualification'):\n"
         "                rec['container_qualification_success']=bool((native.get('detail') or {}).get('container_qualification_success'))\n"
         "            records.append(rec)\n"),
    ],
}


def stage(prior, layouts, s1, replay, guard, closed_place, destination, root, revision):
    from go2arx5_vla.world_model.storage import digest
    from scripts.pi0292.durable import read, write
    from scripts.wm02.multitask_stage_cards_v6 import pin
    from v64_n1_observation import patched

    checks = ((prior, PRIOR), (layouts, COLD), (s1, S1), (replay, REPLAY), (guard, GUARD))
    if any(digest(path) != expected for path, expected in checks):
        raise PermissionError("Container allocation requires the exact archived inputs")
    if (root.exists() or root.parent.stat().st_uid != os.getuid() or
            len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision)):
        raise PermissionError("Use the unopened Core world and full producer revision")
    config = read(prior)
    original = Path(config["source_root"])
    members = read(config["source_members"]["path"])
    if digest(config["source_members"]["path"]) != config["source_members"]["sha256"] or any(
            digest(original / name) != expected for name, expected in members.items()):
        raise ValueError("The exact original producer closure must be preserved")
    closed = read(closed_place / "control/LEASE.json")
    actual = read(closed_place / "RESULT.private.json")
    if (closed["status"] != "CLOSED" or closed["spec_sha256"] != digest(closed_place / "EXECUTION_SPEC.json")
            or not actual["completed"] or actual["unsafe"] or actual["technical_failure"]
            or not actual["shutdown_clean"] or not 0 < actual["actual_native_integrations"] <= CAP):
        raise ValueError("A closed complete N1 PLACE is required for actual storage measurement")

    destination.mkdir(mode=0o700)
    source, delta = destination / "source", destination / "delta"
    shutil.copytree(original, source, copy_function=os.link, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    install_runtime(original, delta)
    for path in delta.rglob("*.py"):
        member = path.relative_to(delta)
        os.replace(path, source / member)
        members[str(member)] = digest(source / member)
    relative = Path("scripts/sla") / Path(__file__).name
    shutil.copyfile(__file__, source / relative)
    members[str(relative)] = digest(source / relative)
    for name, edits in PATCHES.items():
        relative = Path("scripts/sla") / name
        patched(source / relative, edits)
        members[str(relative)] = digest(source / relative)

    bindings, clusters, conditions = {}, [], []
    earlier = {item["id"]: item for item in read(s1)["clusters"]}
    cold = read(layouts)
    for index, layout in enumerate(cold["rows"], 1):
        identity = layout["id"]
        family, color, side = QUALIFICATION[identity]
        scene_path = layouts.parent / f"{identity}.mjb"
        scene = pin(scene_path)
        if scene["sha256"] != SCENES[identity] or layout["status"] != "COLD_PASS" or layout["target"] != color:
            raise ValueError("The screened bowl geometry must remain exact")
        detector = pin(guard.parent / "observer.so")
        if detector["sha256"] != NATIVE:
            raise ValueError("The compiled released-floor observer changed")
        binding = destination / f"{identity}.json"
        write(binding, dict(plan=PLAN, id=identity, scene=scene, candidate_observer=detector, observer_guard=pin(guard)))
        bindings[identity] = pin(binding)
        reset = 702700 + index
        cluster = earlier[f"EF5_Q_MOBILE_{index:02}"]
        clusters.append(dict(cluster, id=identity, reset=reset, scene=scene))
        conditions.append(dict(id=identity, condition_id=identity, scene_family=identity, cluster=identity,
            family=family, scene_seed=reset, reset=reset, instruction=f"PLACE-{color}-{side}", split="QUAL",
            native_max_ticks=CAP, max_queries_or_generations=180))
    roster = destination / "ROSTER.json"
    write(roster, dict(variant="CONTAINER_QUAL", plan_v64=PLAN, split="QUAL", physics_requests=2,
         integrations=2*CAP, boundary_replay=pin(replay), case_bindings=bindings))
    population = destination / "POPULATION.json"
    write(population, dict(plan=read(config["population"]["path"])["plan"], source_geometry_sha256=COLD,
         clusters=clusters, conditions={"QUAL": conditions}))
    expected_bytes = 6 * sum(p.stat().st_blocks*512 for p in closed_place.rglob("*") if p.is_file())
    expected_bytes = (expected_bytes*CAP + actual["actual_native_integrations"]-1)//actual["actual_native_integrations"]
    free = shutil.disk_usage(root.parent).free
    if free <= expected_bytes:
        raise OSError("The original measured N1 reserve does not fit")
    with tempfile.TemporaryFile(dir=root.parent) as stream:
        os.posix_fallocate(stream.fileno(), 0, 64*1024**2)
        os.fsync(stream.fileno())
    capacity_path = root.parent / "CONTAINER-QUAL-CAPACITY.json"
    if capacity_path.exists():
        raise PermissionError("A prior Container wave already owns the reserve")
    write(capacity_path, dict(free_bytes=free, upcoming_writes_bytes=expected_bytes, writable_probe_bytes=64*1024**2))
    members_path = destination / "SOURCE_MEMBERS.json"
    write(members_path, members)
    staged = dict(config, root=str(root), source_root=str(source), source_sha=revision, population=pin(population),
                  source_members=pin(members_path), v62_mobile_qualification=pin(roster))
    manifest = destination / "CONFIG.json"
    write(manifest, staged)
    spec = read(closed_place / "EXECUTION_SPEC.json")
    env = dict(os.environ, **spec["native_environment"])
    env["PYTHONPATH"] = os.pathsep.join((str(source), str(source / "src"), str(source / "scripts/sla"), env["PYTHONPATH"]))
    command = ('from pathlib import Path; from scripts.pi0292.durable import read; '
               'from scripts.sla.core_hopper_register import prepare; import sys; '
               'prepare(dict(read(Path(sys.argv[1])), root=sys.argv[2]), execute=False)')
    subprocess.run([spec["python"], "-B", "-c", command, str(manifest), str(destination / "CPU-PREFLIGHT")],
                   check=True, timeout=90, cwd=source, env=env)
    print(json.dumps(pin(manifest)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prior", "layouts", "s1", "replay", "guard", "closed_place", "destination", "root"):
        parser.add_argument(name, type=Path)
    parser.add_argument("revision")
    stage(**vars(parser.parse_args()))
