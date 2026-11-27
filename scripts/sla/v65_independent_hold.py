"""Bind V6.5 to the four independent HOLD comparisons from V6.1."""

import argparse
import os
from pathlib import Path
import shutil

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v61_hold_pair import rows as original_rows
from v62_source_native import SLA, V65, checked_manifest, source_role


BASE_ENTRY = "904f54907cf46b19cd025c24294ab98b3fcd259ded248df2edf8a2ffd04a0bae"
ROLE = "RECOVERY1000"


def rows(roster):
    return [dict(row, id=row["id"].replace("R4_DEV_0200", ROLE))
            for row in original_rows(roster, "R4_DEV_0200")]


def launcher(path):
    text = pin(path, BASE_ENTRY).read_text()
    for old, new in (
        ('    if registration["v4_model_role"] == "R4_DEV_0200":\n        from v6_r4_native_deploy',
         '    if "source_audit" in registration:\n'
         '        from v62_source_native import checked_manifest\n'
         '        return checked_manifest(reference, expected, registration)\n'
         '    if registration["v4_model_role"] == "R4_DEV_0200":\n        from v6_r4_native_deploy'),
        ('        from v61_hold_pair import rows\n        return rows(registration["v61_roster"], registration["v4_model_role"])',
         '        from v65_independent_hold import rows\n        return rows(registration["v61_roster"])'),
        ('            additional["v61_roster"] = binding["v61_roster"]\n',
         '            from v62_source_native import V65, source_role\n'
         '            require(binding["v62_plan"] == V65 and binding["role"] == source_role(binding["source_step"], V65),\n'
         '                    "Wrong independent V6.5 course")\n'
         '            additional.update(v61_roster=binding["v61_roster"], v62_plan=V65,\n'
         '                              source_step=binding["source_step"], source_audit=binding["source_audit"])\n'),
        ('qualification, "PARENT" if r4 else binding["role"])',
         'qualification, "PARENT" if r4 or binding.get("v61_roster") else binding["role"])'),
        ('            from v61_hold_pair import rows\n            places, holds = [], rows(binding["v61_roster"], binding["role"])',
         '            from v65_independent_hold import rows\n            places, holds = [], rows(binding["v61_roster"])'),
    ):
        require(text.count(old) == 1, "The original independent HOLD launcher seam changed")
        text = text.replace(old, new)
    compile(text, str(path), "exec")
    return text


def stage(prior, standard, overlay):
    original = read(prior / "V61_R4_DEV_0200_PREFLIGHT.json")
    regular = read(standard / "RECOVERY_HOLD_PREFLIGHT.json")
    require(original["hold"] == original_rows(original["v61_roster"], "R4_DEV_0200")
            and original["noise"]["hold"] == regular["noise"]["hold"]
            and regular["role"] == source_role(1000, V65), "The qualified HOLD archive and V6.5 are unmatched")
    registration = dict(v4_model_role=ROLE, **{k: regular[k] for k in ("v62_plan", "source_step", "source_audit")})
    own = regular["parent_panel"]
    old_parent = read(pin(own["path"], own["sha256"]))["candidate"]
    parent = read(pin(Path(old_parent["path"]) / "CHECKPOINT.json", old_parent["sha256"]))
    model = checked_manifest(regular["candidate"], parent, registration)
    entry = SLA / "m38_a_native_v1.py"
    source = launcher(prior / entry)
    shutil.copytree(prior, overlay, copy_function=os.link, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("v62_source_native", "v62_source_endpoint", "v62_source_admission", "v61_source_course", "v61_source_bank",
                 "v6_dual_course", "v65_recovery_endpoint", "v65_recovery_course", "v65_recovery_data", "v63_adapt_course",
                 "v63_adapt_data", "v64_a78_delta", "v64_a87_increment", "v64_a93_increment", "v64_go2_delta4"):
        relative = SLA / f"{name}.py"
        if (overlay / relative).exists():
            require(sha(overlay / relative) == sha(standard / relative), f"Mismatched evaluation dependency: {name}")
        else:
            shutil.copy2(standard / relative, overlay / relative)
    (overlay / entry).unlink()
    (overlay / entry).write_text(source)
    shutil.copy2(__file__, overlay / SLA)
    write(overlay / "RECOVERY_INDEPENDENT_HOLD_PREFLIGHT.json",
          dict(regular, v61_roster=original["v61_roster"], hold=rows(original["v61_roster"]), place=[],
               checkpoint_sha256=model["checkpoint_sha256"]))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("prior", "standard", "overlay"):
        cli.add_argument(key, type=Path)
    stage(**vars(cli.parse_args()))
