"""Stage fixed Scene Factory waves through the qualified single-world Teacher."""

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
TRAIN_GEOMETRY = "d36c9da25c6c927134fe578be1a378c60d3a7851ea9fa2b97e099e77eefe38e4"
F2_GEOMETRY = "0e99e023f4348139b8123d5b1c8b4a7da7f4acf2b48d2312f665dba913eb4621"
F2_REPLAY = "9d0d116a768cfb00f24b42e57e4cb0df1d7ef1e9cf139479ed3b1b6668c697a1"
F1_RED_LEGACY = "f0015701b5070b877dce21d2292cefb41b9aa13d17ba1e9fc669f7eaa6648f92"
DIVERSE_GEOMETRY = "838cd05519316ee5fc16d97a051d9a707e9203fdf4d452ba842f9885c8da723b"
DIVERSE_WAVE2_GEOMETRY = "25c15ab27d2fa528dca803432f7de2f4d0c3a423f13255c427da7099ad9bd6fe"
DIVERSE_WAVE3_GEOMETRY = "a6d6a35dd5f9fc15b6f587c17eed3bcb355b044b68776494a915d602546b301e"
DIVERSE_WAVE4_GEOMETRY = "7a9e350319006ce628a718f533de2dd520222dc8c3042a7b738697957762418e"
DIVERSE_WAVE5_GEOMETRY = "664aee826dcb87082ec6b26ef17d700dd07775afff60ce4a090cb12a6eca4d22"
DIVERSE_WAVE2_TERMINAL = "fd4ab7afadf92a45d097e638c23cc488a27d6916c890705be265513c9a62e3e0"
DIVERSE_WAVE3_TERMINAL = "6c86071bb00828c6a891be72f6be2c48aeea1b3f3d9a9ceb61306a7caa0526f1"
DIVERSE_WAVE4_TERMINAL = "ad40b4a7dbbd410061b861f64014f9a33bff35b74a15b0cf786b8d907aa7ec82"
F2_ACTUAL_PREFIX = "bc56a16a451df2dc89aea9b725eabccc0f5be0b92442a71f69569d0635827aed"
DIVERSE_LEGACY = "05b145f759de77ff7cf00c82db6aee4fdafe21b826c8fa15a410e88c07ebf154"
DIVERSE_LEGACY_LEFT = "414fa12a17d0527c7ac91ff8eec6254b1714e8ecb00a447a0d57c284b1beebeb"
TICKS = 35703
PLACE_TICKS = 50703
IDS = ("EF5_Q_MOBILE_01", "EF5_Q_MOBILE_02")
FIRST_TRAIN = ("EF5_T_MOBILE_01", "EF5_T_MOBILE_05")
F1_RED_TRAIN = ("EF5_T_MOBILE_03", "EF5_T_MOBILE_07")
DIVERSE_PLACE = {
    "EF5_DIVERSE_PLACE_001": ("F1", "PLACE-red-right", "CHV1-TRAIN-F1-PLACE-red-right-02"),
    "EF5_DIVERSE_PLACE_002": ("F2", "PLACE-blue-right", "CHV1-TRAIN-F2-PLACE-blue-right-02"),
}
DIVERSE_PLACE_WAVE2 = {
    "EF5_DIVERSE_PLACE_003": ("F1", "PLACE-red-right", "CHV1-TRAIN-F1-PLACE-red-right-02"),
    "EF5_DIVERSE_PLACE_004": ("F2", "PLACE-red-left", "CHV1-TRAIN-F2-PLACE-red-left-02"),
}
DIVERSE_PLACE_WAVE3 = {
    "EF5_DIVERSE_PLACE_005": ("F1", "PLACE-blue-left", "MEV1-TRAIN-F1-NEAR_NEG-00-PLACE-blue-left"),
    "EF5_DIVERSE_PLACE_006": ("F2", "PLACE-red-left", "CHV1-TRAIN-F2-PLACE-red-left-02"),
}
DIVERSE_PLACE_WAVE4 = {
    "EF5_DIVERSE_PLACE_007": ("F1", "PLACE-blue-left", "MEV1-TRAIN-F1-NEAR_NEG-00-PLACE-blue-left"),
    "EF5_DIVERSE_PLACE_008": ("F1", "PLACE-red-right", "CHV1-TRAIN-F1-PLACE-red-right-02"),
}
DIVERSE_PLACE_WAVE5 = {
    "EF5_DIVERSE_PLACE_009": DIVERSE_PLACE_WAVE4["EF5_DIVERSE_PLACE_007"],
    "EF5_DIVERSE_PLACE_010": DIVERSE_PLACE_WAVE4["EF5_DIVERSE_PLACE_008"],
}
PLACE_WAVES = {"DIVERSE_PLACE": DIVERSE_PLACE, "DIVERSE_PLACE_WAVE2": DIVERSE_PLACE_WAVE2,
               "DIVERSE_PLACE_WAVE3": DIVERSE_PLACE_WAVE3, "DIVERSE_PLACE_WAVE4": DIVERSE_PLACE_WAVE4,
               "DIVERSE_PLACE_WAVE5": DIVERSE_PLACE_WAVE5}
SCOPES = {"QUAL": ("QUAL", IDS, GEOMETRY), "TRAIN": ("TRAIN", FIRST_TRAIN, TRAIN_GEOMETRY),
          "F2_REPAIR": ("QUAL", ("EF5_Q_MOBILE_F2_RED_R2",), F2_GEOMETRY),
          "F1_RED_TRAIN": ("TRAIN", F1_RED_TRAIN, TRAIN_GEOMETRY),
          "DIVERSE_PLACE": ("TRAIN", tuple(DIVERSE_PLACE), DIVERSE_GEOMETRY),
          "DIVERSE_PLACE_WAVE2": ("TRAIN", tuple(DIVERSE_PLACE_WAVE2), DIVERSE_WAVE2_GEOMETRY),
          "DIVERSE_PLACE_WAVE3": ("TRAIN", tuple(DIVERSE_PLACE_WAVE3), DIVERSE_WAVE3_GEOMETRY),
          "DIVERSE_PLACE_WAVE4": ("TRAIN", tuple(DIVERSE_PLACE_WAVE4), DIVERSE_WAVE4_GEOMETRY),
          "DIVERSE_PLACE_WAVE5": ("TRAIN", tuple(DIVERSE_PLACE_WAVE5), DIVERSE_WAVE5_GEOMETRY)}


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
    variant = roster.get("variant", roster.get("split", "QUAL"))
    split, identities, geometry = SCOPES[variant]
    ticks = PLACE_TICKS if variant in PLACE_WAVES else TICKS
    selected = population["conditions"][split]
    if (population["plan"] != PLAN or population["source_geometry_sha256"] != geometry
            or roster["plan"] != PLAN or roster["physics_requests"] != len(identities)
            or roster["integrations"] != len(identities)*ticks or roster["previously_qualified_worlds"] != 1
            or [x["id"] for x in selected] != list(identities) or set(roster["cards"]) != set(identities)
            or any(x["split"] != split or x["native_max_ticks"] != ticks for x in selected)):
        raise ValueError("Only the registered scene wave may run")
    by_id = {cluster["id"]: cluster["scene"]["sha256"] for cluster in population["clusters"]}
    if by_id != roster["cards"]:
        raise ValueError("Compiled scenes differ from the qualification allocation")
    if variant in PLACE_WAVES:
        history = pinned(roster["qualification"])
        if roster["qualification"]["sha256"] != DIVERSE_LEGACY:
            raise ValueError("Diverse PLACE requires the original consumed Teacher archive")
        approved = {r["episode"]: r for r in history["episodes"]}
        if variant in ("DIVERSE_PLACE_WAVE2", "DIVERSE_PLACE_WAVE3"):
            left = pinned(roster["left_qualification"])
            if roster["left_qualification"]["sha256"] != DIVERSE_LEGACY_LEFT:
                raise ValueError("The second wave requires the original complete F2 red-left witness")
            for admission in left["admissions"]:
                episode = admission["episode"]
                if episode["split"] == "TRAIN" and episode["individual_pass"]:
                    approved[episode["id"]] = admission
        for row in selected:
            family, instruction, witness = PLACE_WAVES[variant][row["id"]]
            proof = approved[witness]
            if ((row["family"], row["instruction"]) != (family, instruction) or proof["task"] != instruction
                    or not all(proof["phase_windows"].get(phase, 0) > 0 for phase in ("CARRY", "LOWER", "OPEN", "WITHDRAW"))):
                raise ValueError("Diverse PLACE requires a complete same-family task and actual release support")
        if variant == "DIVERSE_PLACE_WAVE3":
            terminal = pinned(roster["previous_place_result"])
            failed, = (r for r in terminal["records"] if r["condition"] == "EF5_DIVERSE_PLACE_004")
            if (roster["previous_place_result"]["sha256"] != DIVERSE_WAVE2_TERMINAL
                    or failed["first_failure"] != "ILLEGAL_CONTACT" or failed["post_STOP_ticks"] != 0):
                raise ValueError("The archived F2 STOP must remain complete and in the denominator")
        if variant == "DIVERSE_PLACE_WAVE4":
            terminal = pinned(roster["previous_place_result"])
            passed, failed = terminal["records"]
            prefix = pinned(roster["stop_prefix"])
            if (roster["previous_place_result"]["sha256"] != DIVERSE_WAVE3_TERMINAL
                    or passed["condition"] != "EF5_DIVERSE_PLACE_005" or not passed["task_and_mobile_success"]
                    or failed["condition"] != "EF5_DIVERSE_PLACE_006"
                    or failed["first_failure"] != "ILLEGAL_CONTACT" or failed["post_STOP_ticks"] != 0
                    or roster["stop_prefix"]["sha256"] != F2_ACTUAL_PREFIX
                    or prefix[-1]["condition"] != failed["condition"]
                    or prefix[-1]["terminal_tick"] != 18846):
                raise ValueError("F1-only production requires the closed preceding wave and actual F2 prefix")
        if variant == "DIVERSE_PLACE_WAVE5":
            terminal = pinned(roster["previous_place_result"])
            preceding = terminal["records"]
            if (roster["previous_place_result"]["sha256"] != DIVERSE_WAVE4_TERMINAL
                    or terminal["paid_requests"] != 2 or terminal["exit_code"] != 0
                    or [row["condition"] for row in preceding] != list(DIVERSE_PLACE_WAVE4)
                    or any(not row["task_and_mobile_success"] or row["hard_STOP"] or row["technical_failure"]
                           for row in preceding)):
                raise ValueError("The new F1 layouts require both original successful terminal records")
    elif split == "TRAIN":
        confirmation = pinned(roster["qualification"])
        expected, instruction = IDS[0], "HOLD-blue"
        if variant == "F1_RED_TRAIN":
            if roster["qualification"]["sha256"] != F1_RED_LEGACY or confirmation["paid_requests"] != 1:
                raise ValueError("The closed F1 red mobile-grasp episode is required")
            expected, instruction = "CHV1-TRAIN-F1-HOLD-red-19", "HOLD-red"
        matches = [r for r in confirmation["records"] if r["condition"] == expected]
        if (len(matches) != 1 or not matches[0]["task_and_mobile_success"]
                or matches[0]["hard_STOP"] or matches[0]["technical_failure"]
                or matches[0]["post_STOP_ticks"] not in (None, 0)
                or any((x["family"], x["instruction"]) != ("F1", instruction) for x in selected)):
            raise ValueError("TRAIN requires a closed successful same-family, same-target mobile-grasp qualification")
    if variant == "F2_REPAIR":
        proof = pinned(roster["repair"])
        confirmation = pinned(roster["qualification"])
        previous, = (r for r in confirmation["records"] if r["condition"] == IDS[1])
        if (roster["repair"]["sha256"] != F2_REPLAY or previous["result"]["sha256"] != proof["result_sha256"]
                or not previous["hard_STOP"] or previous["technical_failure"] or previous["post_STOP_ticks"] != 0
                or (selected[0]["family"], selected[0]["instruction"]) != ("F2", "HOLD-red")):
            raise ValueError("The new F2 case is limited to the evidenced distractor displacement")
    return selected


def capacity_file(root):
    if root.name in ("DIVERSE-PLACE-TRAIN-05", "DIVERSE-PLACE-OVERLAY-05"):
        return root.parent / "DIVERSE-PLACE-CAPACITY-05.json"
    if root.name in ("DIVERSE-PLACE-TRAIN-04", "DIVERSE-PLACE-OVERLAY-04"):
        return root.parent / "DIVERSE-PLACE-CAPACITY-04.json"
    if root.name in ("DIVERSE-PLACE-TRAIN-03", "DIVERSE-PLACE-OVERLAY-03"):
        return root.parent / "DIVERSE-PLACE-CAPACITY-03.json"
    if root.name in ("DIVERSE-PLACE-TRAIN-02", "DIVERSE-PLACE-OVERLAY-02"):
        return root.parent / "DIVERSE-PLACE-CAPACITY-02.json"
    name = "DIVERSE-PLACE-CAPACITY.json" if root.name.startswith("DIVERSE-PLACE-") else "QUAL-CAPACITY.json"
    return root.parent / name


def capacity(root):
    planned = plain(capacity_file(root))
    memory = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                      if line.startswith("MemAvailable:"))) * 1024
    return shutil.disk_usage(root).free > planned["upcoming_writes_bytes"] and memory >= 13*1024**3


def build_population(compiled, old, variant):
    clusters = {x["id"]: x for x in compiled["clusters"]}
    historical = {x["scene"]["sha256"] for x in old["clusters"]}
    resets = {x["reset"] for x in old["clusters"]}
    cards = []
    split, identities, geometry = SCOPES[variant]
    place = variant in PLACE_WAVES
    for name in identities:
        q, = (x for x in compiled["conditions"][split] if x["id"] == name)
        cluster = clusters[name]
        if (q["skill"] != ("DIVERSE_PLACE" if place else "MOBILE_GRASP")
                or q["physics_eligible"] or not q["original_2ms"]
                or cluster["scene"]["sha256"] in historical or cluster["reset"] in resets
                or digest(cluster["scene"]["path"]) != cluster["scene"]["sha256"]):
            raise ValueError("Unexpected mobile pilot or geometry overlaps historical data")
        instruction = q["instruction"] if place else f"HOLD-{q['target']}"
        original = next(x for x in old["conditions"]["TRAIN"]
                        if x["family"] == q["family"] and x["instruction"] == instruction)
        cards.append(dict(original, id=name, condition_id=name, split=split, instruction=instruction,
                          cluster=name, scene_family=name, reset=q["reset"], reset_seed=q["reset"],
                          scene_seed=q["reset"], native_max_ticks=PLACE_TICKS if place else TICKS))
    return dict(plan=PLAN, source_geometry_sha256=geometry, clusters=[clusters[x] for x in identities],
                conditions={split: cards})


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


def stage(prior, compiled, original_output, destination, root, revision, qualification=None, repair=None,
          legacy_red_train=False, place_catalog=None, place_left_proof=None, previous_place_wave=None,
          stop_prefix=None, after_clean_place_wave=False):
    if repair and not qualification:
        raise ValueError("Read the original F2 terminal before staging a repair")
    if legacy_red_train and (qualification or repair):
        raise ValueError("F1 red uses only the preserved legacy qualification")
    if place_catalog and (qualification or repair or legacy_red_train):
        raise ValueError("Diverse PLACE requires its own legacy task witness")
    if place_left_proof and not place_catalog:
        raise ValueError("The second PLACE wave also requires the original right-placement witness")
    if after_clean_place_wave and (not (previous_place_wave and place_catalog) or stop_prefix or place_left_proof):
        raise ValueError("The fifth wave uses the sealed two-card F1 terminal")
    if previous_place_wave and (not (place_left_proof or stop_prefix or after_clean_place_wave)
            or plain(previous_place_wave / "control/LEASE.json")["status"] != "CLOSED"):
        raise ValueError("The successor wave requires closed production and its earlier evidence")
    if stop_prefix and (not (place_catalog and previous_place_wave) or place_left_proof):
        raise ValueError("The fourth wave requires the previous terminal and the two actual contact prefixes")
    variant = ("DIVERSE_PLACE_WAVE5" if after_clean_place_wave else
               "DIVERSE_PLACE_WAVE4" if stop_prefix else
               "DIVERSE_PLACE_WAVE3" if previous_place_wave else
               "DIVERSE_PLACE_WAVE2" if place_left_proof else "DIVERSE_PLACE" if place_catalog else
               "F1_RED_TRAIN" if legacy_red_train else
               "F2_REPAIR" if repair else "TRAIN" if qualification else "QUAL")
    split, identities, geometry = SCOPES[variant]
    ticks = PLACE_TICKS if place_catalog else TICKS
    if digest(prior / "CONFIG.json") != PRIOR_CONFIG or digest(compiled) != geometry:
        raise ValueError("The original single world launcher and frozen scene compiler are required")
    history = plain(original_output / "RESULT.private.json")
    if (history["paid_requests"] != 1 or history["exit_code"] != 0
            or plain(original_output / "control/LEASE.json")["status"] != "CLOSED"):
        raise ValueError("Read the real previously successful one-world control first")
    old = plain(prior / "CONFIG.json")
    old_source = Path(old["source_root"])
    source = destination / "source"
    symbols = pinned(old["source_members"])
    population = build_population(plain(compiled), pinned(old["population"]), variant)
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
            ("(1, 35703) if binding else (264, 12065592)",
             f"({len(identities)}, {len(identities)*ticks}) if binding else (264, 12065592)"),
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
    registered = dict(plan=PLAN, previously_qualified_worlds=1, physics_requests=len(identities), integrations=len(identities)*ticks,
                      cards={c["id"]: c["scene"]["sha256"] for c in population["clusters"]})
    if qualification:
        result = qualification / "RESULT.private.json"
        if (plain(qualification / "control/LEASE.json")["status"] != "CLOSED"
                or plain(result)["paid_requests"] != 2 or plain(result)["exit_code"] != 0):
            raise ValueError("TRAIN requires the original mobile qualification wave to close")
        registered.update(split=split, qualification=ref(result))
    if repair:
        registered.update(variant=variant, repair=ref(repair))
    if legacy_red_train:
        registered.update(variant=variant, split=split, qualification=ref(original_output / "RESULT.private.json"))
    if place_catalog:
        registered.update(variant=variant, split=split, qualification=ref(place_catalog))
    if place_left_proof:
        registered["left_qualification"] = ref(place_left_proof)
    if previous_place_wave:
        registered["previous_place_result"] = ref(previous_place_wave / "RESULT.private.json")
    if stop_prefix:
        registered["stop_prefix"] = ref(stop_prefix)
    write(registry_file, registered)
    write(source_file, symbols)
    use = shutil.disk_usage(root.parent)
    demonstrated = int(sum(p.stat().st_blocks*512 for p in original_output.rglob("*") if p.is_file()))
    planned = demonstrated * 6
    if place_catalog:
        require_ticks = history["verified_ticks"]
        if not 0 < require_ticks <= TICKS:
            raise ValueError("No valid native byte-rate witness for the next wave")
        planned = (planned * ticks + require_ticks - 1) // require_ticks
    if use.free <= planned:
        raise OSError("Insufficient real capacity for both world outputs and complete terminal seal")
    with tempfile.TemporaryFile(dir=root.parent) as f:
        os.posix_fallocate(f.fileno(), 0, 64*1024**2)
        os.fsync(f.fileno())
    reservation = capacity_file(root)
    if variant == "QUAL" or variant in PLACE_WAVES:
        write(reservation, dict(free_bytes=use.free, previous_trace_bytes=demonstrated,
              upcoming_writes_bytes=planned, writable_probe_bytes=64*1024**2, device=root.parent.stat().st_dev))
    elif plain(reservation)["upcoming_writes_bytes"] != planned:
        raise ValueError("The same native output reservation must cover this two-card TRAIN wave")
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
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--repair", type=Path)
    parser.add_argument("--legacy-red-train", action="store_true")
    parser.add_argument("--place-catalog", type=Path)
    parser.add_argument("--place-left-proof", type=Path)
    parser.add_argument("--previous-place-wave", type=Path)
    parser.add_argument("--stop-prefix", type=Path)
    parser.add_argument("--after-clean-place-wave", action="store_true")
    stage(**vars(parser.parse_args()))
