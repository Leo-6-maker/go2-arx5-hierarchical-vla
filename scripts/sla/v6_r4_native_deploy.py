"""Stage the R4-200 developmental HOLD comparison on the qualified V4 runner."""

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


def hold_rows(rows):
    require([(r["id"], r["layout_slot"], r["noise_label"], r["link"], r["enabled"], r["max_ticks"],
              r["policy_queries"]) for r in rows] == [
                  ("LONG_PARENT_N02_A", "N02", 201, "L1", True, 35703, 120),
                  ("LONG_PARENT_N02_B", "N02", 202, "L1", True, 35703, 120),
              ], "R4 uses the two original N02 HOLD conditions")
    return [dict(r, id=r["id"].replace("LONG_PARENT_", "R4_DEV_0200_"), pair_id="R4_DEV_0200_HOLD") for r in rows]


def launcher_source(path):
    original = pin(path, LAUNCHER).read_text()
    for before, after in (
        ('    return manifest(reference, expected, registration["v4_model_role"])',
         '    if registration["v4_model_role"] == "R4_DEV_0200":\n'
         '        from v6_r4_native_deploy import checked_manifest\n'
         '        return checked_manifest(reference, expected, registration["r4_audit"], registration["r4_replay"])\n'
         '    return manifest(reference, expected, registration["v4_model_role"])'),
        ('    conditions = [c for c in plan["conditions"]',
         '    role = registration["v4_model_role"]\n'
         '    if role == "R4_DEV_0200":\n'
         '        require(phase == "LONG_HOLD", "R4 developmental admission is HOLD only")\n'
         '        role = "PARENT"\n'
         '    conditions = [c for c in plan["conditions"]'),
        ('and c["model"] == registration["v4_model_role"] and c["task"] == task]',
         'and c["model"] == role and c["task"] == task]\n'
         '    if registration["v4_model_role"] == "R4_DEV_0200":\n'
         '        conditions = [dict(c, id=c["id"].replace("LONG_PARENT_", "R4_DEV_0200_")) for c in conditions]'),
        ('        registration = dict(additional, plan=plan)\n        model = comparison_manifest(registration)',
         '        r4 = binding["role"] == "R4_DEV_0200"\n'
         '        if r4:\n'
         '            from v6_r4_native_deploy import hold_rows\n'
         '            require(phase == "LONG_HOLD", "R4 developmental admission is HOLD only")\n'
         '            scope = "SLA_R4_DEV_0200_HOLD"\n'
         '            additional.update(r4_audit=binding["r4_audit"], r4_replay=binding["r4_replay"])\n'
         '        registration = dict(additional, plan=plan)\n'
         '        model = comparison_manifest(registration)'),
        ('                                      v2_registry, qualification, binding["role"])',
         '                                      v2_registry, qualification, "PARENT" if r4 else binding["role"])\n'
         '        if r4:\n'
         '            places, holds = [], hold_rows(holds)'),
    ):
        require(original.count(before) == 1, "Qualified V4 launcher seam changed")
        original = original.replace(before, after)
    compile(original, str(path), "exec")
    return original


def reference(path):
    return dict(path=str(path), sha256=sha(path))


def stage(base, overlay):
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
    entry = Path("scripts/sla/m38_a_native_v1.py")
    launcher = launcher_source(source / entry)
    shutil.copytree(source, overlay, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (overlay / entry).write_text(launcher)
    shutil.copy2(__file__, overlay / entry.parent)
    new_panel = dict(panel, role=ROLE, candidate=deployment, checkpoint_sha256=model["checkpoint_sha256"],
                     place=[], hold=hold_rows(panel["hold"]), r4_audit=audit, r4_replay=replay,
                     parent_panel=reference(previous))
    with (overlay / "R4_HOLD_PREFLIGHT.json").open("x") as stream:
        json.dump(new_panel, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("base", type=Path)
    cli.add_argument("overlay", type=Path)
    stage(**vars(cli.parse_args()))
