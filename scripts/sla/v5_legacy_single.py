"""Continue the original single-world Teacher population with its last unused F1 red HOLD."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


CONDITION = "CHV1-TRAIN-F1-HOLD-red-19"
PLAN = "2cbce375a4f5a01bedc4f065d5ad6581fe7cbb05"
BASE_SOURCE = "7090e2ad238e76cd6278cf65405f7785604abfdb"
POP = "d92fe34f122637bb6dc29bb93d9f7a3d60464f65c5474b00962432bbc07b4914"
SCENE = "3201c91a9ea23ce94f3e40e340ccb4fb64a527174abc6bb9e2a75fb03afce2b1"
HOLD_TICKS = 35703
GIB = 1024 ** 3


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ref(path):
    return dict(path=str(path), sha256=digest(path))


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def rows(population, binding):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin

    spec = read_pin(binding)
    if spec != dict(plan=PLAN, id=CONDITION, scene_sha256=SCENE, one_world=True,
                    native_tick_cap=HOLD_TICKS, generator_calls=0, reserve_gib=58, ram_gib=13):
        raise ValueError("The single-world continuation registration changed")
    item, = (r for r in population["conditions"]["TRAIN"] if r["id"] == CONDITION)
    scene, = (s for s in population["clusters"] if s["id"] == item["scene_family"])
    if (item["family"], item["instruction"], item["native_max_ticks"], scene["scene"]["sha256"]) != (
            "F1", "HOLD-red", HOLD_TICKS, SCENE):
        raise ValueError("Expected the original unstarted TRAIN layout and cap")
    return [item]


def capacity(root):
    """Use the existing absolute reservation for this bounded deployment."""
    with Path("/proc/meminfo").open() as stream:
        memory = int(next(row.split()[1] for row in stream if row.startswith("MemAvailable:"))) * 1024
    return memory >= 13 * GIB and shutil.disk_usage(root).free >= 58 * GIB


def stage(predecessor, catalog, destination, root, source_sha):
    spec_file = predecessor / "EXECUTION_SPEC.json"
    lease = read(predecessor / "control/LEASE.json")
    original = read(spec_file)
    terminal = read(predecessor / "RESULT.private.json")
    if (lease["status"] != "CLOSED" or lease["spec_sha256"] != digest(spec_file)
            or original["source_sha"] != BASE_SOURCE or original["compiled_population"]["sha256"] != POP
            or terminal["source_sha"] != BASE_SOURCE or root.exists()):
        raise ValueError("Expected the terminal original N1 producer and a fresh destination")
    admitted = read(catalog)
    if SCENE in admitted["excluded_dev_scenes"] or any(r["episode"] == CONDITION for r in admitted["episodes"]):
        raise ValueError("Selected case overlaps the signed data or protected scenes")
    original_source = Path(original["source_root"])
    source = destination / "source"
    public = original["source_files"].copy()
    scope = "scripts/sla/"
    patches = {
        "core_hopper_expert.py": [
            ("    elif authority['native_request_cap']!=264", "    elif authority.get('v5_legacy_single'):\n"
             "        from scripts.sla.v5_legacy_single import rows\n"
             "        selected = rows(population, authority['v5_legacy_single'])\n"
             "        if (case.get('v5_legacy_single') != authority['v5_legacy_single']\n"
             "                or case['panel_entry_id'] != selected[0]['id']\n"
             "                or authority['native_request_cap'] != 1\n"
             "                or authority['native_tick_cap'] != selected[0]['native_max_ticks']):\n"
             "            raise PermissionError('Expected the one V5 legacy Teacher card')\n"
             "    elif authority['native_request_cap']!=264")],
        "core_hopper_register.py": [
            ("def prepare(config):", "def prepare(config, *, execute=True):"),
            ("    if not repair:\n        lock=OwnerLock", "    if not repair and execute:\n        lock=OwnerLock"),
            ("    closure=read_pin(config['source_members'])", "    selection=config.get('v5_legacy_single')\n"
             "    if selection:\n        from scripts.sla.v5_legacy_single import rows as legacy_rows\n"
             "        rows=legacy_rows(pop,selection)\n    closure=read_pin(config['source_members'])"),
            ("**({'carry_repair':repair} if repair else {})))", "**({'carry_repair':repair} if repair else {}),\n"
             "        **({'v5_legacy_single':selection} if selection else {})))"),
            ("    if repair:\n        spec['carry_repair']=repair", "    if repair:\n        spec['carry_repair']=repair\n"
             "    if selection:\n        spec['v5_legacy_single']=selection"),
            ("    register(spec,[(root,spec)])", "    if not execute:\n        return\n    register(spec,[(root,spec)])")],
        "core_hopper_collection.py": [
            ("    series=bool(repair", "    selection=spec.get('v5_legacy_single')\n"
             "    capacity_for_scope=capacity\n    if selection:\n"
             "        from scripts.sla.v5_legacy_single import capacity as capacity_for_scope\n"
             "    series=bool(repair"),
            ("            rows=order(pop)\n", "            rows=order(pop)\n            if selection:\n"
             "                from scripts.sla.v5_legacy_single import rows as legacy_rows\n"
             "                rows=legacy_rows(pop,selection)\n"),
            ("not capacity(root)", "not capacity_for_scope(root)")],
    }
    shutil.copytree(original_source, source, copy_function=os.link, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, edits in patches.items():
        relative = scope + name
        path = source / relative
        if digest(path) != public[relative]:
            raise ValueError(f"Original source changed in {name}")
        text = path.read_text()
        for before, after in edits:
            if text.count(before) != 1:
                raise ValueError(f"Original dispatch anchor changed in {name}")
            text = text.replace(before, after)
        compile(text, str(path), "exec")
        path.unlink()
        path.write_text(text)
        public[relative] = digest(path)
    addition = source / scope / Path(__file__).name
    shutil.copy2(__file__, addition)
    public[str(addition.relative_to(source))] = digest(addition)
    members = destination / "SOURCE_MEMBERS.json"
    binding = destination / "ROSTER.json"
    write(members, public)
    write(binding, dict(plan=PLAN, id=CONDITION, scene_sha256=SCENE, one_world=True,
          native_tick_cap=HOLD_TICKS, generator_calls=0, reserve_gib=58, ram_gib=13))
    source_authority = original["sim_only_project_authorization"]
    mesh = read(source_authority["path"])
    if ref(source_authority["path"]) != source_authority or mesh["frozen_producer"] != BASE_SOURCE:
        raise ValueError("Cannot bind the original site authorization window")
    mount = destination / "MOUNT.json"
    write(mount, dict(plan=mesh["plan_commit"], T0_epoch=mesh["T0_epoch"], deadline_epoch=mesh["deadline_epoch"]))
    config = dict(root=str(root), source_root=str(source), mount=ref(mount), interface=original["interface_predecessor"],
                  population=original["compiled_population"], source_members=ref(members), source_sha=source_sha,
                  admission_python=original["admission_python"], statistics=original["C20_statistics_path"],
                  v5_legacy_single=ref(binding))
    write(destination / "CONFIG.json", config)
    sys.path[:0] = [str(source), str(source / "src")]
    from scripts.sla.core_hopper_register import prepare

    prepare(dict(config, root=str(destination / "CPU-PREFLIGHT")), execute=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("predecessor", "catalog", "destination", "root"):
        parser.add_argument(field, type=Path)
    parser.add_argument("source_sha")
    stage(**vars(parser.parse_args()))
