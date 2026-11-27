"""Bind Recovery TRAIN cases to the original M38 native owner and CUDA worker."""
import argparse
import hashlib
from pathlib import Path

from v64_recovery_cohorts import FIRST_NOISE, FIRST_ROSTER, NOISE, read_roster, roster_ref
from v64_recovery_followup import FIRST_RESULT
from v64_recovery_roster import PLAN, QUERIES, TERMINAL, TICKS, fixed, sha


EXECUTOR = "7415d5a6436a8c2ab670bdc7b428fb66307fe221829b31302b88a8262a0ec496"
R4_COMPARISON = "1e19b5cdce0d957d9da4b976d2f3b80cca0b3c972c372b683f665465d6687050"
SCOPE = "SLA_V64_RECOVERY_TRAIN"


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def checked_ref(reference, expected):
    if reference["sha256"] != expected:
        raise ValueError("A sealed Recovery dependency changed")
    return fixed(Path(reference["path"]), expected)


def registration_inputs(roster_path, teacher_wave, parent_path, noise_path):
    roster, cohort = read_roster(roster_path)
    parent = fixed(parent_path, R4_COMPARISON)
    noise = fixed(noise_path, NOISE[cohort])
    cards = roster["cases"]
    expected_noise = [(card["id"], query, (card["policy_seed"] + query) % 2**31)
                      for card in cards for query in range(QUERIES)]
    actual_noise = [(row["case"], row["query"], row["noise_id"]) for row in noise["tensors"]]
    if (roster["plan"] != noise["plan"] or noise["plan"] != PLAN
            or roster["split"] != noise["split"] or noise["split"] != "TRAIN"
            or parent["comparison_model"] != roster["student_provider"] or parent["v4_model_role"] != "R4_DEV_0200"
            or parent["gpu_noise_sha256"] != noise["prior_noise_manifest_sha256"]
            or actual_noise != expected_noise or len({row["gpu_sha256"] for row in noise["tensors"]}) != 2 * QUERIES
            or noise["roster_sha256"] != cohort or noise["matched_archived_cuda_sentinels"] != 3):
        raise ValueError("The Recovery cohort, retained R4 model, or realized noise bank changed")
    if cohort != FIRST_ROSTER:
        closed = checked_ref(roster["prior_terminal"], FIRST_RESULT)
        if (not closed["complete_reconciliation"] or closed["terminal_cases"] != 2 or closed["train_recovery_count"] != 1
                or noise["prior_recovery_noise_sha256"] != FIRST_NOISE
                or roster["max_native_requests_cumulative"] != 20
                or closed["terminal_cases"] != roster["previous_recovery_native_requests"]):
            raise ValueError("The followup requires the entire actual first cohort")
    teachers = {}
    for card in cards:
        key = card["original_teacher"]
        if cohort == FIRST_ROSTER:
            wave = teacher_wave
            provenance = fixed(wave / "RESULT.private.json", TERMINAL)
        else:
            scope = roster["teacher_sources"][card["id"]]
            wave = Path(scope["teacher_wave"])
            if wave.parent != teacher_wave or Path(scope["terminal"]["path"]) != wave / "RESULT.private.json":
                raise ValueError("The followup sources must be the actual closed independent TRAIN waves")
            provenance = fixed(wave / "RESULT.private.json", scope["terminal"]["sha256"])
        row, = (entry for entry in provenance["records"] if entry["condition"] == key)
        case = wave / "cases" / key
        if not (row["admitted"] and row["task_and_mobile_success"] and row["split"] == "TRAIN"
                and not row["technical_failure"] and not row["hard_STOP"]
                and Path(row["result"]["path"]) == case / "RESULT.private.json"):
            raise ValueError("Recovery geometry requires the actually admitted Teacher condition")
        fixed(case / "RESULT.private.json", row["result"]["sha256"])
        lease = case / "control/LEASE.json"
        record = fixed(lease, sha(lease))
        path = case / "EXECUTION_SPEC.json"
        if record["status"] != "CLOSED" or sha(path) != record["spec_sha256"]:
            raise ValueError("Recovery origins require an unchanged closed Teacher lease")
        teachers[card["id"]] = dict(path=str(path), sha256=record["spec_sha256"])
    return roster, parent, teachers


def case_fields(original, registration, row):
    roster, _ = roster_ref(registration["v64_recovery_train"])
    selected, = (card for card in roster["cases"] if card["id"] == row["id"])
    witness = registration["teacher_specs"][row["id"]]
    teacher = fixed(Path(witness["path"]), witness["sha256"])
    scene = Path(selected["scene_path"])
    if (sha(scene) != selected["scene_sha256"] or teacher["model_path"] != str(scene)
            or teacher["frozen_files"][str(scene)] != selected["scene_sha256"]
            or selected["scene_sha256"] in roster["protected_scene_sha256"]
            or teacher["collection_split"] != "TRAIN" or teacher["task_instruction_id"] != selected["task"]
            or teacher["multitask_geometry"] != selected["geometry"]
            or teacher["policy_from_world"] != original["policy_from_world"]
            or teacher["mobile_condition"]["family"] != selected["family"]
            or selected["max_ticks"] != TICKS or selected["maximum_queries"] != QUERIES
            or row["policy_seed"] != selected["policy_seed"] or row["instruction"] != selected["task"]
            or row["link"] != "L1"
            or original["max_ticks"] != TICKS or original["policy_queries"] != QUERIES):
        raise ValueError("Use the fixed disjoint TRAIN geometry, task, seed, and original native budget")
    condition = dict(teacher["mobile_condition"], method=original["mobile_condition"]["method"],
                     reset=selected["reset"], seed=selected["reset"], collection_episode=selected["id"])
    frozen = dict(original["frozen_files"])
    del frozen[original["model_path"]]
    frozen[str(scene)] = selected["scene_sha256"]
    item = dict(id=row["id"], instruction=selected["task"], role="STUDENT", arm=original["arm"],
                condition=condition, geometry=selected["geometry"], scene=ref(scene))
    fields = dict(v64_recovery_train=registration["v64_recovery_train"], collection_split="TRAIN",
                  evaluation_split="TRAIN_RECOVERY", panel_entry_id=row["id"], mobile_condition=condition, model_path=str(scene), frozen_files=frozen,
                  task_instruction_id=selected["task"], language=teacher["language"],
                  multitask_geometry=selected["geometry"])
    return fields, item


def verify_case(spec, registration, row, original):
    fields, item = case_fields(original, registration, row)
    panel = spec["registered_panel"]
    untouched = ("source_sha", "source_files", "student_proof", "packages", "arm", "policy_from_world",
                 "global_task_budget_s", "stage_budget_s", "mobile_safety_binding", "execution_clock_profile",
                 "physics_dt_s", "WBC_changes", "safety_changes", "hardware_connections_allowed",
                 "expert_training_eligible", "max_ticks", "policy_queries", "new_optimizer_updates_budget")
    if (any(spec[key] != value for key, value in fields.items())
            or any(spec[key] != original[key] for key in untouched)
            or fixed(Path(panel["path"]), panel["sha256"]) != [item]
            or spec["comparison_model"] != registration["comparison_model"]
            or spec["interface_comparison_link"] != "L1" or spec.get("comparison_jaw_bridge") is not None):
        raise ValueError("Recovery modified the original frozen control, safety, camera or scope")


def episode_hook(original):
    from scripts.sla.m38_native_link_v1 import episode_source
    from scripts.sla.v64_recovery_native import student_source

    return student_source(episode_source(original))


def adapt_executor(source, destination):
    """Change only the registration and episode seams of an exact executor."""
    if sha(source) != EXECUTOR:
        raise ValueError("The original native executor bytes changed")
    body = source.read_text()
    edits = (
        ('CURRENT_PLAN = ', 'from v64_recovery_launcher import (PLAN as RECOVERY_PLAN, SCOPE as RECOVERY_SCOPE,\n'
         '    R4_COMPARISON, ref, checked_ref, registration_inputs, case_fields, verify_case, episode_hook)\n\n'
         'CURRENT_PLAN = '),
        ('    if registration["plan"] != V4_PLAN:', '    if registration["plan"] == RECOVERY_PLAN:\n'
         '        old = checked_ref(registration["r4_registration"], R4_COMPARISON)\n'
         '        require(old["comparison_model"] == reference, "Wrong Recovery student model")\n'
         '        return comparison_manifest(old, original)\n'
         '    if registration["plan"] != V4_PLAN:'),
        ('reg["plan"] in (CURRENT_PLAN,V2_PLAN,V3_PLAN,V4_PLAN)', 'reg["plan"] in (CURRENT_PLAN,V2_PLAN,V3_PLAN,V4_PLAN,RECOVERY_PLAN)'),
        ('    # The archived guard includes its routing IDs', '    if reg["plan"] == RECOVERY_PLAN:\n'
         '        require(selected[0]["checkpoint"] == comparison_manifest(reg)["checkpoint_sha256"],\n'
         '                "The armed Recovery worker must serve exactly R4-200")\n\n'
         '    # The archived guard includes its routing IDs'),
        ('    if spec.get("comparison_place_layout"):', '    if reg["plan"] == RECOVERY_PLAN:\n'
         '        layout_validator = verify_case\n'
         '    elif spec.get("comparison_place_layout"):'),
        ('        if layout_validator:\n            require(load_code', '        if layout_validator and reg["plan"] != RECOVERY_PLAN:\n'
         '            require(load_code'),
        ('or (reg["plan"]==V4_PLAN and reg["v4_block"] in V4_PHASES),',
         'or (reg["plan"]==V4_PLAN and reg["v4_block"] in V4_PHASES) or reg["plan"]==RECOVERY_PLAN,'),
        ('load_code(spec,V6,episode_source,episode)', 'load_code(spec,V6,episode_hook if reg["plan"]==RECOVERY_PLAN else episode_source,episode)'),
        ('reg["plan"] in (V3_PLAN, V4_PLAN) and self.candidate', 'reg["plan"] in (V3_PLAN, V4_PLAN, RECOVERY_PLAN) and self.candidate'),
        ('          v4_registry=None, v4_panel=None):', '          v4_registry=None, v4_panel=None, recovery_roster=None, recovery_teacher=None, recovery_parent=None):'),
        ('    new_noise = phase in ', '    recovery = recovery_roster is not None\n'
         '    if recovery:\n'
         '        require(phase == "RECOVERY_TRAIN" and through is None and recovery_teacher and recovery_parent\n'
         '                and all(x is None for x in (v2_registry,v3_registry,v4_registry,v4_panel,study_evidence,trained_candidate)),\n'
         '                "Use exactly the separate Recovery registration")\n'
         '        roster,r4,teachers = registration_inputs(recovery_roster,recovery_teacher,recovery_parent,noise)\n'
         '    new_noise = recovery or phase in '),
        ('    noise_identity = read(noise)["old_gpu_manifest_sha256"] if new_noise else sha(noise)',
         '    noise_identity = None if recovery else (read(noise)["old_gpu_manifest_sha256"] if new_noise else sha(noise))\n'
         '    if recovery:\n'
         '        old_panel = checked_ref(r4["v4_panel"], r4["v4_panel"]["sha256"])\n'
         '        old_noise = checked_ref(old_panel["noise"]["place"], r4["gpu_noise_sha256"])\n'
         '        noise_identity = old_noise["old_gpu_manifest_sha256"]'),
        ('    if v4_panel is not None:\n        from v4_long_native_binding', '    if recovery:\n'
         '        plan,scope = RECOVERY_PLAN,RECOVERY_SCOPE\n'
         '        opener = pin(Path(base["source_root"])/REQUEST_POLICY,base["source_files"][REQUEST_POLICY])\n'
         '        additional = dict(v64_recovery_train=ref(recovery_roster), teacher_specs=teachers,\n'
         '            r4_registration=ref(recovery_parent), comparison_model=roster["student_provider"],\n'
         '            hold_request_sha256=hashlib.sha256(hold_request_source(opener.read_bytes()).encode()).hexdigest())\n'
         '        template, = [r for r in receipt["schedule"] if r["task"]=="PLACE_RED_RIGHT" and r["link"]=="L1"]\n'
         '        schedule = [dict(template,id=c["id"],within_pair_order=i+1,pair_id="V64_RECOVERY_TRAIN",\n'
         '            policy_seed=c["policy_seed"],instruction=c["task"],checkpoint=roster["checkpoint_sha256"]) for i,c in enumerate(roster["cases"])]\n'
         '    elif v4_panel is not None:\n        from v4_long_native_binding'),
        ('    write(reg,dict(plan=plan,schedule=schedule,zero_training=True,case_router_sha256=receipt[\n'
         '        "compiled_m38_case_router_sha256"],episode_sha256=receipt["compiled_m38_episode_router_sha256"],',
         '    episode_hash = receipt["compiled_m38_episode_router_sha256"]\n'
         '    if recovery:\n'
         '        original_episode = pin(source/V6,base["source_files"][V6])\n'
         '        episode_hash = hashlib.sha256(episode_hook(original_episode.read_bytes()).encode()).hexdigest()\n'
         '    write(reg,dict(plan=plan,schedule=schedule,zero_training=True,case_router_sha256=receipt[\n'
         '        "compiled_m38_case_router_sha256"],episode_sha256=episode_hash,'),
        ('    if v3_registry is not None or v4_panel is not None:\n        changes', '    if recovery or v3_registry is not None or v4_panel is not None:\n        changes'),
        ('worker["comparison_noise_cardinality"] = 240 if phase=="M_HOLD" else 320',
         'worker["comparison_noise_cardinality"] = len(read(noise)["tensors"]) if recovery else (240 if phase=="M_HOLD" else 320)'),
        ('        if place_layouts:\n            updates,item', '        if recovery:\n'
         '            updates,item = case_fields(original_case,read(reg),row)\n'
         '        elif place_layouts:\n            updates,item'),
        ('        if place_layouts or confirmation_qualification or new_confirmation:', '        if recovery or place_layouts or confirmation_qualification or new_confirmation:'),
        ('    parser.add_argument("--study-evidence",type=Path)', '    parser.add_argument("--study-evidence",type=Path)\n'
         '    for flag in ("recovery-roster", "recovery-teacher", "recovery-parent"):\n'
         '        parser.add_argument("--"+flag,type=Path)'),
    )
    for before, after in edits:
        if body.count(before) != 1:
            raise ValueError(f"Original Recovery extension seam changed: {before[:70]}")
        body = body.replace(before, after)
    compile(body, str(destination), "exec")
    with destination.open("x") as stream:
        stream.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    adapt_executor(**vars(parser.parse_args()))
