"""Bind two independent HOLD layouts to the previously qualified L1 launch path."""

import argparse
import os
from pathlib import Path
import shutil

from m38_gpu_preflight_v1 import pin, read, require, sha


PLAN = "221cc714f7648bb6f596745f9af1aaf1cae530d1"
ROSTER = "c2af53f5b70340d1543082adbdbb769ea8e0e66231ac63d56059485136669605"
LAUNCHER = "c7fbb1975cf0ad0edce15ef0b0f1a48ddcabbef260cd0a906175c9fad4a25c09"
CONFIRMATION = "5c6725e4857f8f94ccca4e554b985a58b5278c8607bd0b426927d5ae7b4cffd1"


def registered(reference):
    require(reference["sha256"] == ROSTER, "Use the sealed V6.1 paired layouts")
    roster = read(pin(reference["path"], ROSTER))
    require(roster["plan"] == PLAN and roster["task"] == "HOLD-blue" and roster["link"] == "L1"
            and roster["interval_s"] == .002 and roster["integrations"] == roster["model_forwards"] == 0,
            "Unexpected V6.1 task, link or accounting")
    return roster


def rows(reference, role):
    require(role in ("PARENT", "R4_DEV_0200"), "Unexpected V6.1 comparison model")
    roster = registered(reference)
    # N03's two MAIN3.8K completions already exist, including its safety stops.
    work = [cell for cell in roster["rows"] if role != "PARENT" or cell["slot"] != "N03"]
    return [
        dict(id=f"V61_{role}_{cell['slot']}_{cell['noise']}", task="HOLD_BLUE", link="L1",
             pair_id="V61_INDEPENDENT_HOLD", within_pair_order=i, layout_slot=cell["slot"],
             policy_seed=cell["policy_seed"], noise_label=cell["noise"], enabled=True, native_started=False,
             max_ticks=cell["tick_cap"], policy_queries=cell["query_cap"], wall_limit_s=1800)
        for i, cell in enumerate(work, 1)
    ]


def teacher(reference, slot, proof):
    roster = registered(reference)
    require(roster["original_teacher_qualification"]["sha256"] == proof["sha256"]
            and slot in roster["slots"], "Teacher and prior qualification must match the sealed panel")
    return roster["slots"][slot]


def changed(path, expected, replacements):
    source = pin(path, expected).read_text()
    for before, after in replacements:
        require(source.count(before) == 1, f"Qualified launcher anchor changed in {path.name}: {before.strip()}")
        source = source.replace(before, after)
    compile(source, str(path), "exec")
    path.unlink()
    path.write_text(source)


def stage(base, overlay):
    from scripts.pi0292.durable import write

    anchored = base / "R4-DEV-0200-HOLD-OVERLAY-01"
    roster_path = base / "v61-corrected-retention-20261126-01/PAIRED_HOLD_ROSTER.json"
    roster_ref = dict(path=str(roster_path), sha256=ROSTER)
    roster = registered(roster_ref)
    previous = base / "V2-CONFIRM-01"
    parent_bank = read(previous / "COMPARISON.json")
    require(parent_bank["gpu_noise_sha256"] == roster["noise"]["sha256"], "Pair noise mismatch")
    for noise in (201, 202):
        case = previous / f"CONFIRM_N03_{noise}"
        spec_file = case / "EXECUTION_SPEC.json"
        spec = read(spec_file)
        result = read(case / "RESULT.private.json")
        lease = read(case / "control/LEASE.json")
        row, = [x for x in roster["rows"] if x["slot"] == "N03" and x["noise"] == noise]
        require(spec["comparison_policy_seed"] == row["policy_seed"]
                and spec["frozen_files"][spec["model_path"]] == roster["slots"]["N03"]["scene"]["sha256"]
                and spec["interface_comparison_link"] == "L1" and result["physical_requests"] == 1
                and result["shutdown_clean"] and not result["technical_failure"]
                and lease["status"] == "CLOSED" and lease["spec_sha256"] == sha(spec_file),
                "The reused MAIN3.8K case is not a valid terminal matched case")
    scripts = Path("scripts/sla")
    shutil.copytree(anchored, overlay, copy_function=os.link,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    launcher = overlay / scripts / "m38_a_native_v1.py"
    changed(launcher, LAUNCHER, (
        ('    from v4_long_eval_noise import REGISTRY_SHA256\n',
         '    if registration.get("v61_roster"):\n'
         '        from v61_hold_pair import rows\n'
         '        return rows(registration["v61_roster"], registration["v4_model_role"])\n'
         '    from v4_long_eval_noise import REGISTRY_SHA256\n'),
        ('        registration = dict(additional, plan=plan)\n',
         '        if binding.get("v61_roster"):\n'
         '            require(phase == "LONG_HOLD", "V6.1 applies to the independent HOLD panel")\n'
         '            scope = "SLA_V61_INDEPENDENT_HOLD_" + binding["role"]\n'
         '            additional["v61_roster"] = binding["v61_roster"]\n'
         '        registration = dict(additional, plan=plan)\n'),
        ('        if r4:\n            places, holds = [], hold_rows(holds)\n',
         '        if binding.get("v61_roster"):\n'
         '            from v61_hold_pair import rows\n'
         '            places, holds = [], rows(binding["v61_roster"], binding["role"])\n'
         '        elif r4:\n'
         '            places, holds = [], hold_rows(holds)\n'),
        ('bind_confirmation_fields(original_case,confirmation_qualification,row)',
         'bind_confirmation_fields(original_case,confirmation_qualification,row,'
         'v61=additional.get("v61_roster"))'),
    ))
    changed(overlay / scripts / "v2_m38_confirmation_student.py", CONFIRMATION, (
        ('def bind_confirmation_fields(original,proof,row):\n',
         'def bind_confirmation_fields(original,proof,row,*,v61=None):\n'),
        ('    evidence = layouts[row["layout_slot"]]\n',
         '    if v61:\n'
         '        from v61_hold_pair import teacher\n'
         '        evidence = teacher(v61, row["layout_slot"], proof)\n'
         '    else:\n'
         '        evidence = layouts[row["layout_slot"]]\n'),
        ('    require(registered\n',
         '    v61 = registration.get("v61_roster")\n'
         '    require((registered or (v61 and registration["plan"]==V4_PLAN\n'
         '                and registration["v4_block"]=="LONG_HOLD"))\n'),
        ('bind_confirmation_fields(original,registration["confirmation_qualification"],row)',
         'bind_confirmation_fields(original,registration["confirmation_qualification"],row,v61=v61)'),
    ))
    shutil.copy2(__file__, overlay / scripts / Path(__file__).name)
    candidates = {
        "PARENT": read(base / "v4-long-eval-noise-01/PARENT_PREFLIGHT_V2.json"),
        "R4_DEV_0200": read(anchored / "R4_HOLD_PREFLIGHT.json"),
    }
    for role, panel in candidates.items():
        require(panel["role"] == role and panel["noise"]["hold"] == roster["noise"], "Candidate identity drift")
        write(overlay / f"V61_{role}_PREFLIGHT.json",
              dict(panel, place=[], hold=rows(roster_ref, role), v61_roster=roster_ref))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("overlay", type=Path)
    stage(**vars(parser.parse_args()))
