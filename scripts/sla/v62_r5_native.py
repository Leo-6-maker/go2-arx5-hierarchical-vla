"""Stage the two preregistered R5 release cases on the original V4 PLACE executor."""

import argparse
import json
import os
from pathlib import Path
import shutil

from m38_gpu_preflight_v1 import pin, read, require, sha
from v4_long_watch import candidate


ROLE = "R5_RELEASE_0200"
PLAN = "90efbfb8785dbdf7fbb1219fbe179e049ec41e31"
AUDIT = "f63869a7dda6f47a93985becd6811cfb88e80d967cbafb506546b3c24421563c"
PARENT_PANEL = "ec4455850009b9f79a9ffabbafbe93d22ac53091a19d0182868387cece1cb108"
R4_EXECUTOR = "7415d5a6436a8c2ab670bdc7b428fb66307fe221829b31302b88a8262a0ec496"
WATCH = "8ad51b043db337a79ac0e58bbafcfc4c56bd6b777f2672592523ce6874b16f80"
R5_SEAL = "e44934ce604dff1aef3c2d3e94b4c2235efd5a15e9883c9ceba6265bb8cc61d2"
TWO = ("LONG_PARENT_P01_701", "LONG_PARENT_P02_701")
SOURCE = Path("scripts/sla")


def receipts(path):
    return dict(path=str(path), sha256=sha(path))


def checked_manifest(reference, parent, audit):
    proof_file = pin(audit["path"], AUDIT)
    require(audit["sha256"] == AUDIT, "Unexpected R5 CPU restore receipt")
    proof = read(proof_file)
    folder = Path(reference["path"])
    model = candidate(folder, parent, 200, proof_file)
    require(reference["sha256"] == proof["provider_manifest_sha256"]
            and proof["role"] == "R5_RELEASE_PHASE_REPLAY" and proof["receipt_sha256"] == R5_SEAL
            and proof["rng_restored"] and proof["complete_training_endpoint"]
            and proof["adam_counters"] == dict(shared=200, target=200, robodojo=200, libero=200)
            and not any(model[k] for k in ("native_admission", "hardware_allowed", "physics")),
            "Use the exact restored R5 diagnostic weights")
    return model


def initial_rows(original):
    rows = [row for row in original if row["id"] in TWO]
    require(tuple(row["id"] for row in rows) == TWO
            and all(row["noise_label"] == 701 and row["link"] == "L1"
                    and row["max_ticks"] == 45703 and row["policy_queries"] == 160 for row in rows),
            "The original two matching R5 test conditions must be preserved")
    return [dict(row, id=row["id"].replace("LONG_PARENT_", ROLE + "_"),
                 pair_id=ROLE, within_pair_order=i)
            for i, row in enumerate(rows, 1)]


def initial_registry(conditions):
    rows = [row for row in conditions if row["id"] in TWO]
    require(tuple(row["id"] for row in rows) == TWO, "The original R5 registry inputs changed")
    return [dict(row, id=row["id"].replace("LONG_PARENT_", ROLE + "_")) for row in rows]


def launcher(path):
    text = pin(path, R4_EXECUTOR).read_text()
    edits = [
        ('    if registration["v4_model_role"] == "R4_DEV_0200":\n        from v6_r4_native_deploy',
         '    if registration["v4_model_role"] == "R5_RELEASE_0200":\n'
         '        from v62_r5_native import checked_manifest\n'
         '        return checked_manifest(reference, expected, registration["r5_audit"])\n'
         '    if registration["v4_model_role"] == "R4_DEV_0200":\n        from v6_r4_native_deploy'),
        ('    if role == "R4_DEV_0200":', '    if role in ("R4_DEV_0200", "R5_RELEASE_0200"):'),
        ('    require(len(conditions) == (4 if phase == "LONG_PLACE" else 2)',
         '    if registration["v4_model_role"] == "R5_RELEASE_0200":\n'
         '        from v62_r5_native import initial_registry\n'
         '        require(phase == "LONG_PLACE", "R5 starts with its original PLACE pair")\n'
         '        return initial_registry(conditions)\n'
         '    require(len(conditions) == (4 if phase == "LONG_PLACE" else 2)'),
        ('        registration = dict(additional, plan=plan)\n        model = comparison_manifest(registration)',
         '        r5 = binding["role"] == "R5_RELEASE_0200"\n'
         '        if r5:\n'
         '            from v62_r5_native import PLAN as r5_plan\n'
         '            require(phase == "LONG_PLACE" and binding["v62_plan"] == r5_plan, "R5 changed its first-pair scope")\n'
         '            scope = "SLA_R5_RELEASE_0200_PLACE"\n'
         '            additional.update(r5_audit=binding["r5_audit"], v62_plan=r5_plan)\n'
         '        registration = dict(additional, plan=plan)\n'
         '        model = comparison_manifest(registration)'),
        ('"PARENT" if r4 else binding["role"])','"PARENT" if r4 or r5 else binding["role"])'),
        ('            places, holds = comparison_rows(phase, places, holds)\n',
         '            places, holds = comparison_rows(phase, places, holds)\n'
         '        if r5:\n'
         '            from v62_r5_native import initial_rows\n'
         '            places, holds = initial_rows(places), []\n'),
    ]
    for before, after in edits:
        require(text.count(before) == 1, "The pinned PLACE launcher entry differs")
        text = text.replace(before, after)
    compile(text, str(path), "exec")
    return text


def stage(base, retention, overlay):
    r4 = base / "R4-DEV-0200-PLACE-OVERLAY-01"
    original = base / "v4-long-eval-noise-01/PARENT_PREFLIGHT_V2.json"
    panel = read(pin(original, PARENT_PANEL))
    parent = panel["candidate"]
    related = retention / "R5_STEP0200_PROVIDER"
    reference = dict(path=str(related), sha256=sha(related / "CHECKPOINT.json"))
    audit = receipts(retention / "R5_STEP0200_PROVIDER_READBACK.json")
    model = checked_manifest(reference, read(pin(Path(parent["path"])/"CHECKPOINT.json", parent["sha256"])), audit)
    code = launcher(r4 / SOURCE / "m38_a_native_v1.py")
    shutil.copytree(r4, overlay, copy_function=os.link, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    file = overlay / SOURCE / "m38_a_native_v1.py"
    file.unlink()
    file.write_text(code)
    shutil.copy2(__file__, overlay / SOURCE)
    shutil.copy2(pin(retention / "code/v4_long_watch.py", WATCH), overlay / SOURCE)
    panel_path = overlay / "R5_PLACE_PREFLIGHT.json"
    with panel_path.open("x") as stream:
        json.dump(dict(panel, role=ROLE, v62_plan=PLAN, candidate=reference, checkpoint_sha256=model["checkpoint_sha256"],
                 place=initial_rows(panel["place"]), hold=[], r5_audit=audit, parent_panel=receipts(original)),
                 stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(sha(panel_path), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "retention", "overlay"):
        parser.add_argument(key, type=Path)
    stage(**vars(parser.parse_args()))
