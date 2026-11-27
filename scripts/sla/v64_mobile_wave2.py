"""Stage independent two-card F1 Mobile Grasp waves with the existing N1 Teacher."""
import argparse
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import re


COMPILER = "68821b29e04a9f4dea7f68bb272ebdb16e9ff82f522587271e8e84d9fea1340d"
DISPATCHER = "57b26f851f937cc13427b04cc5657b50e5896fe48f4cf1666935691f525f1cab"
BLUE_DISPATCHER = "83ecd1d887252ca2ffcc2ee5fbab653f4aace68e449fef3c24f463143cafc4b8"
QUALIFIED_RED = "a68732134b2fe0c508dadd05d66fb518e37835ac949c1b514ab065d64e5022ed"
QUALIFIED_BLUE = "8a2b3a54d8e96d2332d3ec22ae1663b1abdccb7e87ad952cc8cc42951ac4e259"
OLD_MOBILE = "d36c9da25c6c927134fe578be1a378c60d3a7851ea9fa2b97e099e77eefe38e4"
LEGACY_POP = "d92fe34f122637bb6dc29bb93d9f7a3d60464f65c5474b00962432bbc07b4914"
PROTECTED_HISTORY = (
    ("SCENE-FACTORY-MOBILE-TRAIN-01", OLD_MOBILE),
    ("SCENE-FACTORY-MOBILE-F2-REPAIR-01", "0e99e023f4348139b8123d5b1c8b4a7da7f4acf2b48d2312f665dba913eb4621"),
    ("SCENE-FACTORY-DIVERSE-PLACE-01", "838cd05519316ee5fc16d97a051d9a707e9203fdf4d452ba842f9885c8da723b"),
    ("SCENE-FACTORY-DIVERSE-PLACE-02", "25c15ab27d2fa528dca803432f7de2f4d0c3a423f13255c427da7099ad9bd6fe"),
    ("SCENE-FACTORY-DIVERSE-PLACE-03", "a6d6a35dd5f9fc15b6f587c17eed3bcb355b044b68776494a915d602546b301e"),
    ("SCENE-FACTORY-DIVERSE-PLACE-04", "7a9e350319006ce628a718f533de2dd520222dc8c3042a7b738697957762418e"),
    ("SCENE-FACTORY-DIVERSE-PLACE-05", "664aee826dcb87082ec6b26ef17d700dd07775afff60ce4a090cb12a6eca4d22"),
)
ROSTER = (
    ("EF5_T_MOBILE_09", "F1", "MOBILE_GRASP", "red", (6, 10), (-8, 1), 702309),
    ("EF5_T_MOBILE_10", "F1", "MOBILE_GRASP", "red", (2, 5), (-12, 5), 702310),
)
BLUE_ROSTER = (
    ("EF5_T_MOBILE_11", "F1", "MOBILE_GRASP", "blue", (-9, 6), (6, -9), 702311),
    ("EF5_T_MOBILE_12", "F1", "MOBILE_GRASP", "blue", (4, -6), (-6, 5), 702312),
)
FINAL_BLUE_ROSTER = (("EF5_T_MOBILE_13", "F1", "MOBILE_GRASP", "blue", (-6, 11), (9, -8), 702313),)
BLUE_WAVE2 = "9b6a01177d8a99102011f086b666e66e1d1570aa5240054edcd33a1495b68b14"
BLUE_WAVE2_TERMINAL = "4c2b89eb69fa3bfdfa5826134702a7ab8bcd856da611458ad0e6d797dc1550d2"
RECOVERY_TRAIN = "e9b7552d4c10779f4c87a04ef23c252ee2f599a69a24df5d8f94624166ad6cba"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Fixed input changed: {path.name}")
    return path


def load(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def module(path):
    spec = spec_from_file_location(path.stem, path)
    loaded = module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def prepare(site, base_input, train_manifest, qualification, prior, legacy, work, revision, *,
            blue=False, final_blue=False):
    if work.exists() or not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("A fresh CPU staging root and exact publication commit are required")
    blue = blue or final_blue
    variant = "BLUE" if blue else "RED"
    wave = "03" if final_blue else "02"
    qualified = site / ("MOBILE-QUAL-01" if blue else "MOBILE-F1-RED-TRAIN-01")
    result_path = check(qualified / "RESULT.private.json", QUALIFIED_BLUE if blue else QUALIFIED_RED)
    result = load(result_path)
    reference_rows = ([row for row in result["records"] if row["condition"] == "EF5_Q_MOBILE_01"]
                      if blue else result["records"])
    if (load(qualified / "control/LEASE.json")["status"] != "CLOSED"
            or result["exit_code"] != 0 or result["paid_requests"] != 2
            or len(reference_rows) != (1 if blue else 2)
            or any(row["hard_STOP"] or row["technical_failure"] or not row["task_and_mobile_success"]
                   for row in reference_rows)
            or (not blue and result["admitted_train"] != 2)):
        raise ValueError("Require the closed actual same-target F1 Teacher qualification")
    compiler_path = check(site / "v5_scene_factory_mobile_train.py", COMPILER)
    anchor = "MOBILE-TRAIN-OVERLAY-01" if blue else "MOBILE-F1-RED-TRAIN-OVERLAY-01"
    previous = check(site / anchor / "source/scripts/sla/v62_mobile_qualification.py",
                     BLUE_DISPATCHER if blue else DISPATCHER)
    legacy_pop = Path(load(prior / "CONFIG.json")["population"]["path"])
    sources = [check(legacy_pop, LEGACY_POP)]
    sources.extend(check(site / name / "COMPILED_POPULATION.private.json", digest) for name, digest in PROTECTED_HISTORY)
    if blue:
        sources.extend(check(site / name / "COMPILED_POPULATION.private.json", digest) for name, digest in (
            ("SCENE-FACTORY-MOBILE-F1-RED-02", "090b104d5a8fdc270a39307513ebbf855c12cff48c583ed22ad75db994a40535"),
            ("SCENE-FACTORY-DIVERSE-PLACE-06", "7a5f0a2466cc139c589112b186bdfcd7f76c2fd6b610fc84b43359ea1a15b001"),
        ))
    if final_blue:
        earlier_root = site / "MOBILE-F1-BLUE-TRAIN-02"
        terminal = load(check(earlier_root / "RESULT.private.json", BLUE_WAVE2_TERMINAL))
        if (load(earlier_root / "control/LEASE.json")["status"] != "CLOSED"
                or terminal["paid_requests"] != 2 or terminal["admitted_train"] != 2):
            raise ValueError("The two preceding blue training cards must be sealed")
        sources.append(check(site / "SCENE-FACTORY-MOBILE-F1-BLUE-02" / "COMPILED_POPULATION.private.json",
                             BLUE_WAVE2))
    protected = [row for source in sources for row in load(source)["clusters"]]
    held = load(check(site / "RECOVERY-TRAIN-ROSTER-01.json", RECOVERY_TRAIN))["cases"]
    if blue:
        held += load(check(site / "RECOVERY-TRAIN-ROSTER-02.json",
                           "4018539bb1f083322450b2fd0bdfdc76380f41a91aa926225e47c2a17ed77959"))["cases"]
    protected.extend(dict(id=row["id"], reset=row["reset"], scene=dict(sha256=row["scene_sha256"])) for row in held)
    work.mkdir(mode=0o700)
    exclude = work / "PROTECTED_POPULATION.json"
    write(exclude, dict(clusters=protected))
    factory = module(compiler_path)
    if tuple(factory.MOBILE_TRAIN[2][0:4]) != ("EF5_T_MOBILE_03", "F1", "MOBILE_GRASP", "red"):
        raise ValueError("The original mobile compiler roster changed")
    roster = FINAL_BLUE_ROSTER if final_blue else BLUE_ROSTER if blue else ROSTER
    factory.MOBILE_TRAIN = roster
    stage_version = f"{wave}-R2" if final_blue else wave
    compiled = site / f"SCENE-FACTORY-MOBILE-F1-{variant}-{stage_version}"
    factory.build(base_input, train_manifest, compiled, qualification=qualification, protected_population=exclude)
    population = compiled / "COMPILED_POPULATION.private.json"
    # Preserve the original launcher and change only its fixed TRAIN population.
    body = previous.read_text()
    earlier = '("EF5_T_MOBILE_01", "EF5_T_MOBILE_05")' if blue else '("EF5_T_MOBILE_03", "EF5_T_MOBILE_07")'
    replacements = [(OLD_MOBILE, sha(population)), (earlier, repr(tuple(row[0] for row in roster)))]
    if final_blue:
        replacements.extend((
            ('roster["physics_requests"] != 2', 'roster["physics_requests"] != 1'),
            ('roster["integrations"] != 2*TICKS', 'roster["integrations"] != TICKS'),
            ("(2, 71406) if binding else (264, 12065592)", "(1, 35703) if binding else (264, 12065592)"),
            ("physics_requests=2, integrations=2*TICKS", "physics_requests=1, integrations=TICKS"),
        ))
    for before, after in replacements:
        if body.count(before) != 1:
            raise ValueError("Existing Mobile Grasp dispatcher seam changed")
        body = body.replace(before, after)
    dispatch = work / previous.name
    compile(body, str(dispatch), "exec")
    with dispatch.open("x") as stream:
        stream.write(body)
    overlay = site / f"MOBILE-F1-{variant}-TRAIN-OVERLAY-{stage_version}"
    root = site / f"MOBILE-F1-{variant}-TRAIN-{wave}"
    options = dict(qualification=qualified) if blue else dict(legacy_red_train=True)
    module(dispatch).stage(prior, population, legacy, overlay, root, revision, **options)
    write(work / "ACTUAL.json", dict(stage="V64_F1_MOBILE_GRASP_FOLLOWUP_PREFLIGHT",
          teacher_qualification_sha256=sha(result_path), scene_population_sha256=sha(population),
          roster_sha256=sha(overlay / "ROSTER.json"), config_sha256=sha(overlay / "CONFIG.json"),
          protected_sources_sha256=[sha(path) for path in sources], exclusion_rows=len(protected),
          fixed_conditions=[row[0] for row in roster], actual_native_requests=0, actual_native_integrations=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("site", "base_input", "train_manifest", "qualification", "prior", "legacy", "work"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("revision")
    parser.add_argument("--blue", action="store_true")
    parser.add_argument("--final-blue", action="store_true")
    prepare(**vars(parser.parse_args()))
