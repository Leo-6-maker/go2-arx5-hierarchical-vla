"""Stage matched R4-200 developmental comparisons on the qualified V4 runner."""

import argparse
import json
from pathlib import Path
import shutil

from m38_gpu_preflight_v1 import pin, read, require, sha


ROLE = "R4_DEV_0200"
LAUNCHER = "be6528e1501f391113d3a949be99187cd0cf375ef9fbeda8ea0f9fcd7fa91aec"
PARENT_PANEL = "ec4455850009b9f79a9ffabbafbe93d22ac53091a19d0182868387cece1cb108"
AUDIT = "820b1ef46077355da86c85c448bd83e2b552ab199f138439d057e82f53f79741"
SEAM = "bbd4da750339e01b757692cb95f82872ebdeda8556f4e2df6035dddd392d6c31"
HOLD_READBACK = "db77327c1ff1bd3d1dd759d51afb4b7ada4c5d230b0da937aa893d0d8fbb243d"


def checked_manifest(ref, parent, audit_ref, replay_ref):
    proof = read(pin(audit_ref["path"], AUDIT))
    replay = read(pin(replay_ref["path"], SEAM))
    require(audit_ref["sha256"] == AUDIT and replay_ref["sha256"] == SEAM
            and tuple(r["id"] for r in replay["cases"]) == ("CONFIRM_N02_201", "CONFIRM_N02_202")
            and all(r["r4_l1"]["status"] == "L1_RETIMED" for r in replay["cases"]), "Wrong R4 evidence")
    folder = Path(ref["path"])
    model = read(pin(folder / "CHECKPOINT.json", ref["sha256"]))
    require(proof["complete_training_endpoint"] and proof["rng_restored"]
            and not proof["formal"]
            and proof["adam_counters"] == dict(shared=200, target=200, robodojo=200, libero=200)
            and proof["role"] == "R4_REEXPRESSED_COMMANDS" and proof["step"] == 200
            and proof["provider_manifest_sha256"] == ref["sha256"]
            and proof["parent_sha256"] == parent["checkpoint_sha256"] == model["previous_checkpoint_sha256"]
            and proof["contract_sha256"] == model["training_contract_sha256"]
            and proof["receipt_sha256"] == model["training_receipt_sha256"], "Wrong training endpoint or ancestry")
    require(model["schema"] == "SLA_V4_MAIN_MULTISOURCE_INFERENCE" and model["role"] == "DIAGNOSTIC"
            and model["phase"] == 200 and model["frozen_wbc"] and model["original_l1"]
            and not any(model[k] for k in ("formal_inference_eligible", "native_admission", "hardware_allowed", "physics"))
            and model["source_codec"] == "DENSE_REFERENCE_G7_V6"
            and all(model[k] == parent[k] for k in ("foundation_sha256", "frozen_model",
                    "action_sha256", "state_sha256", "tokenizer_sha256", "partition_sha256")), "Go2 provider ABI changed")
    for new, old in zip(model["arrays"], parent["arrays"], strict=True):
        require(all(new[k] == old[k] for k in ("file", "key", "shape", "dtype")), "Go2 provider leaf ABI changed")
        pin(folder / new["file"], new["sha256"])
    return model


def checked_hold(hold_ref, checkpoint):
    proof = read(pin(hold_ref["path"], HOLD_READBACK))
    require(hold_ref["sha256"] == HOLD_READBACK and proof["model_checkpoint_sha256"] == checkpoint
            and proof["registered"] == proof["valid"] == proof["full_success"] == 2
            and proof["gpu_and_native_leases_closed"] and not any(a["unsafe"] for a in proof["arms"]),
            "R4 PLACE must use the independently reconciled HOLD candidate")


def comparison_rows(phase, places, holds):
    if phase == "LONG_PLACE":
        rows = places
        roster = [("P01_701", "P01", 701), ("P02_702", "P02", 702), ("P02_701", "P02", 701), ("P01_702", "P01", 702)]
        cap = (45703, 160)
    else:
        require(phase == "LONG_HOLD", "Unknown R4 phase")
        rows = holds
        roster = [("N02_A", "N02", 201), ("N02_B", "N02", 202)]
        cap = (35703, 120)
    require([(r["id"], r["layout_slot"], r["noise_label"]) for r in rows]
            == [("LONG_PARENT_" + name, layout, noise) for name, layout, noise in roster]
            and all(r["link"] == "L1" and r.get("enabled", True) and (r["max_ticks"], r["policy_queries"]) == cap
                    for r in rows), "R4 uses the original matched conditions and finite bounds")
    tag = phase.removeprefix("LONG_")
    selected = [dict(r, id=r["id"].replace("LONG_PARENT_", ROLE + "_"), pair_id=ROLE + "_" + tag) for r in rows]
    return (selected, []) if phase == "LONG_PLACE" else ([], selected)


def launcher_source(path):
    original = pin(path, LAUNCHER).read_text()
    for before, after in (
        ('    return manifest(reference, expected, registration["v4_model_role"])',
         '    if registration["v4_model_role"] == "R4_DEV_0200":\n'
         '        from v6_r4_native_deploy import checked_hold, checked_manifest\n'
         '        candidate = checked_manifest(reference, expected, registration["r4_audit"], registration["r4_replay"])\n'
         '        if registration["v4_block"] == "LONG_PLACE":\n'
         '            checked_hold(registration["r4_hold"], candidate["checkpoint_sha256"])\n'
         '        return candidate\n'
         '    return manifest(reference, expected, registration["v4_model_role"])'),
        ('    conditions = [c for c in plan["conditions"]',
         '    role = registration["v4_model_role"]\n'
         '    if role == "R4_DEV_0200":\n'
         '        role = "PARENT"\n'
         '    conditions = [c for c in plan["conditions"]'),
        ('and c["model"] == registration["v4_model_role"] and c["task"] == task]',
         'and c["model"] == role and c["task"] == task]\n'
         '    if registration["v4_model_role"] == "R4_DEV_0200":\n'
         '        conditions = [dict(c, id=c["id"].replace("LONG_PARENT_", "R4_DEV_0200_")) for c in conditions]'),
        ('        registration = dict(additional, plan=plan)\n        model = comparison_manifest(registration)',
         '        r4 = binding["role"] == "R4_DEV_0200"\n'
         '        if r4:\n'
         '            from v6_r4_native_deploy import comparison_rows\n'
         '            require(phase == binding["r4_phase"], "R4 bound phase changed")\n'
         '            scope = "SLA_R4_DEV_0200_" + phase.removeprefix("LONG_")\n'
         '            additional.update(r4_audit=binding["r4_audit"], r4_replay=binding["r4_replay"])\n'
         '            if phase == "LONG_PLACE":\n'
         '                additional["r4_hold"] = binding["r4_hold"]\n'
         '        registration = dict(additional, plan=plan)\n'
         '        model = comparison_manifest(registration)'),
        ('                                      v2_registry, qualification, binding["role"])',
         '                                      v2_registry, qualification, "PARENT" if r4 else binding["role"])\n'
         '        if r4:\n'
         '            places, holds = comparison_rows(phase, places, holds)'),
    ):
        require(original.count(before) == 1, "Qualified V4 launcher seam changed")
        original = original.replace(before, after)
    compile(original, str(path), "exec")
    return original


def reference(path):
    return dict(path=str(path), sha256=sha(path))


def stage(base, overlay, phase):
    source = base / "v4-long-eval-source-01"
    previous = base / "v4-long-eval-noise-01/PARENT_PREFLIGHT_V2.json"
    panel = read(pin(previous, PARENT_PANEL))
    parent_pin = panel["candidate"]
    parent = read(pin(Path(parent_pin["path"]) / "CHECKPOINT.json", parent_pin["sha256"]))
    retention = base / "v6-retention-20261126-01"
    candidate = retention / "R4_STEP0200_PROVIDER"
    audit = reference(retention / "R4_STEP0200_PROVIDER_READBACK.json")
    replay = reference(retention / "R4_STEP0200_HOLD_SEAM_REPLAY_V2.json")
    deployment = dict(path=str(candidate), sha256=sha(candidate / "CHECKPOINT.json"))
    model = checked_manifest(deployment, parent, audit, replay)
    places, holds = comparison_rows(phase, panel["place"], panel["hold"])
    new_panel = dict(panel, role=ROLE, candidate=deployment, checkpoint_sha256=model["checkpoint_sha256"],
                     place=places, hold=holds, r4_audit=audit, r4_replay=replay,
                     parent_panel=reference(previous), r4_phase=phase)
    if phase == "LONG_PLACE":
        hold = reference(base / "R4-DEV-0200-HOLD-NATIVE-01/INDEPENDENT_PANEL_READBACK.json")
        checked_hold(hold, model["checkpoint_sha256"])
        new_panel["r4_hold"] = hold
    entry = Path("scripts/sla/m38_a_native_v1.py")
    launcher = launcher_source(source / entry)
    shutil.copytree(source, overlay, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (overlay / entry).write_text(launcher)
    shutil.copy2(__file__, overlay / entry.parent)
    with (overlay / f"R4_{phase.removeprefix('LONG_')}_PREFLIGHT.json").open("x") as stream:
        json.dump(new_panel, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base", type=Path)
    cli.add_argument("overlay", type=Path)
    cli.add_argument("--phase", choices=("LONG_HOLD", "LONG_PLACE"), default="LONG_HOLD")
    stage(**vars(cli.parse_args()))
