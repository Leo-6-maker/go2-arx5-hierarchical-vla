"""Stage two existing V5 mobile-grasp scenes for isolated native Teacher qualification."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


PLAN = "90efbfb8785dbdf7fbb1219fbe179e049ec41e31"
PRIOR_CONFIG = "a07cde3a0b644609b6fed3b67c40c6aa0d66f2691b23121e8303180bcd566d19"
GEOMETRY = "f10b07348ad2746876a3e3ba1399a4b3b950ae5875c965f7c1d909d07b50df39"
TICKS = 35703
IDS = ("EF5_Q_MOBILE_01", "EF5_Q_MOBILE_02")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def plain(path):
    return json.loads(Path(path).read_bytes())


def pinned(ref):
    path = Path(ref["path"])
    if digest(path) != ref["sha256"]:
        raise ValueError(f"Pinned input changed: {path.name}")
    return plain(path)


def ref(path):
    return dict(path=str(path), sha256=digest(path))


def write(path, data):
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def rows(population, binding):
    roster = pinned(binding)
    selected = population["conditions"]["QUAL"]
    if (population["plan"] != PLAN or population["source_geometry_sha256"] != GEOMETRY
            or roster["plan"] != PLAN or roster["physics_requests"] != 2
            or roster["integrations"] != 2*TICKS or roster["previously_qualified_worlds"] != 1
            or [x["id"] for x in selected] != list(IDS) or set(roster["cards"]) != set(IDS)
            or any(x["split"] != "QUAL" or x["native_max_ticks"] != TICKS for x in selected)):
        raise ValueError("Only the original two preparation scenes may be qualified")
    by_id = {cluster["id"]: cluster["scene"]["sha256"] for cluster in population["clusters"]}
    if by_id != roster["cards"]:
        raise ValueError("Compiled scenes differ from the qualification allocation")
    return selected


def capacity(root):
    planned = plain(root.parent / "QUAL-CAPACITY.json")
    memory = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                      if line.startswith("MemAvailable:"))) * 1024
    return shutil.disk_usage(root).free > planned["upcoming_writes_bytes"] and memory >= 13*1024**3


def build_population(compiled, old):
    clusters = {x["id"]: x for x in compiled["clusters"]}
    historical = {x["scene"]["sha256"] for x in old["clusters"]}
    resets = {x["reset"] for x in old["clusters"]}
    cards = []
    for name in IDS:
        q, = (x for x in compiled["conditions"]["QUAL"] if x["id"] == name)
        cluster = clusters[name]
        if (q["skill"] != "MOBILE_GRASP" or q["physics_eligible"] or not q["original_2ms"]
                or cluster["scene"]["sha256"] in historical or cluster["reset"] in resets
                or digest(cluster["scene"]["path"]) != cluster["scene"]["sha256"]):
            raise ValueError("Unexpected mobile pilot or geometry overlaps historical data")
        instruction = f"HOLD-{q['target']}"
        original = next(x for x in old["conditions"]["TRAIN"]
                        if x["family"] == q["family"] and x["instruction"] == instruction)
        cards.append(dict(original, id=name, condition_id=name, split="QUAL", instruction=instruction,
                          cluster=name, scene_family=name, reset=q["reset"], reset_seed=q["reset"],
                          scene_seed=q["reset"], native_max_ticks=TICKS))
    return dict(plan=PLAN, source_geometry_sha256=GEOMETRY, clusters=[clusters[x] for x in IDS],
                conditions={"QUAL": cards})


def patch(anchor, path, expected, edits):
    if digest(anchor) != expected:
        raise ValueError(f"Qualified launcher bytes changed: {anchor.name}")
    text = anchor.read_text()
    for before, after in edits:
        if text.count(before) != 1:
            raise ValueError(f"Expected one dispatch match for {anchor.name}")
        text = text.replace(before, after)
    text = text.replace("v5_legacy_single", "v62_mobile_qualification")
    compile(text, str(path), "exec")
    path.unlink()
    path.write_text(text)


def stage(prior, compiled, original_output, destination, root, revision):
    if digest(prior / "CONFIG.json") != PRIOR_CONFIG or digest(compiled) != GEOMETRY:
        raise ValueError("The original single world launcher and frozen scene compiler are required")
    history = plain(original_output / "RESULT.private.json")
    if (history["paid_requests"] != 1 or history["exit_code"] != 0
            or plain(original_output / "control/LEASE.json")["status"] != "CLOSED"):
        raise ValueError("Read the real previously successful one-world control first")
    old = plain(prior / "CONFIG.json")
    old_source = Path(old["source_root"])
    source = destination / "source"
    symbols = pinned(old["source_members"])
    population = build_population(plain(compiled), pinned(old["population"]))
    destination.mkdir(mode=0o700)
    shutil.copytree(old_source, source, copy_function=os.link,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    scope = "scripts/sla/"
    overlays = {
        "core_hopper_register.py": [
            ("    if config['population']['sha256']!=POP:", "    selection=config.get('v5_legacy_single')\n"
             "    if not selection and config['population']['sha256']!=POP:"),
            ("    pop=read_pin(config['population']); rows=order(pop)\n    selection=config.get('v5_legacy_single')",
             "    pop=read_pin(config['population']); rows=[] if selection else order(pop)"),
            ("compiled_population_sha256=POP,hardware_allowed=False",
             "compiled_population_sha256=config['population']['sha256'],hardware_allowed=False"),
            ("population_sha256=POP,plan=PLAN", "population_sha256=config['population']['sha256'],plan=PLAN"),
        ],
        "core_hopper_expert.py": [
            ("(1, 35703) if binding else (264, 12065592)", "(2, 71406) if binding else (264, 12065592)"),
            ("    if (ref['sha256'] != POP or authority['compiled_population_sha256'] != POP",
             "    if ((not binding and ref['sha256'] != POP) or authority['compiled_population_sha256'] != ref['sha256']"),
            ("    selected = order(population)\n", "    selected = [] if binding else order(population)\n"),
        ],
        "core_hopper_collection.py": [
            ("scheduled, capacity_for_scope = order(pop), capacity", "scheduled, capacity_for_scope = "
             "([] if selection else order(pop)), capacity"),
            ("if outcome['task_and_mobile_success'] and outcome['completed'] and not outcome['hard_STOP']:",
             "if row['split'] != 'QUAL' and outcome['task_and_mobile_success'] and outcome['completed'] and not outcome['hard_STOP']:"),
        ],
    }
    for name, edits in overlays.items():
        relative = scope + name
        patch(old_source / relative, source / relative, symbols[relative], edits)
        symbols[relative] = digest(source / relative)
    relative = scope + Path(__file__).name
    shutil.copy2(__file__, source / relative)
    symbols[relative] = digest(source / relative)
    pop_file = destination / "POPULATION.json"
    registry_file = destination / "ROSTER.json"
    source_file = destination / "SOURCE_MEMBERS.json"
    write(pop_file, population)
    write(registry_file, dict(plan=PLAN, previously_qualified_worlds=1, physics_requests=2, integrations=2*TICKS,
                              cards={c["id"]: c["scene"]["sha256"] for c in population["clusters"]}))
    write(source_file, symbols)
    use = shutil.disk_usage(root.parent)
    demonstrated = int(sum(p.stat().st_blocks*512 for p in original_output.rglob("*") if p.is_file()))
    planned = demonstrated * 6
    if use.free <= planned:
        raise OSError("Insufficient real capacity for both world outputs and complete terminal seal")
    with tempfile.TemporaryFile(dir=root.parent) as f:
        os.posix_fallocate(f.fileno(), 0, 64*1024**2)
        os.fsync(f.fileno())
    write(root.parent / "QUAL-CAPACITY.json", dict(free_bytes=use.free, previous_trace_bytes=demonstrated,
               upcoming_writes_bytes=planned, writable_probe_bytes=64*1024**2, device=root.parent.stat().st_dev))
    config = dict(old, root=str(root), source_root=str(source), source_sha=revision,
                  population=ref(pop_file), source_members=ref(source_file))
    del config["v5_legacy_single"]
    config["v62_mobile_qualification"] = ref(registry_file)
    write(destination / "CONFIG.json", config)
    sys.path[:0] = [str(source), str(source / "src")]
    from scripts.sla.core_hopper_register import prepare

    prepare(dict(config, root=str(destination / "CPU-PREFLIGHT")), execute=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prior", "compiled", "original_output", "destination", "root"):
        parser.add_argument(name, type=Path)
    parser.add_argument("revision")
    stage(**vars(parser.parse_args()))
