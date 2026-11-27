"""Bind a restored V6.2 SOURCE checkpoint to the original six Go2 evaluation cases."""

import argparse
from collections import Counter
from importlib import import_module
import os
from pathlib import Path
import shutil

from m38_gpu_preflight_v1 import pin, read, require, sha
from v62_source_admission import V62
from v62_source_endpoint import BANK_RECEIPT, ENDPOINTS, GROUPS, MAIN, PLAN, R4, SOURCE_VISITS, VISITS


PARENT_PANEL = "ec4455850009b9f79a9ffabbafbe93d22ac53091a19d0182868387cece1cb108"
EXECUTOR = "7415d5a6436a8c2ab670bdc7b428fb66307fe221829b31302b88a8262a0ec496"
PARENT_RELEASE = "0896270957c478197e586a71f2d2d3a55464a5bb7436cf3d390fa14804926433"
R4_PANELS = {
    "HOLD": "db77327c1ff1bd3d1dd759d51afb4b7ada4c5d230b0da937aa893d0d8fbb243d",
    "PLACE": "ce7077c88ed98d00277340098a3851fff1acfe6e35dc529d0b69e65c96e9806d",
}
FEEDBACK_ROSTER = "a6e24bfd71bcd7034f18a9a5007cc0061f74190950ae1f4d080a4f466ddd698a"
SLA = Path("scripts/sla")


def reference(path):
    return dict(path=str(path), sha256=sha(path))


def source_role(step, plan=V62):
    require(type(step) is int and step in ENDPOINTS, "Unknown SOURCE evaluation milestone")
    if plan != V62:
        from v64_adapt_endpoint import END, PLAN as ADAPT_PLAN

        require(plan == ADAPT_PLAN and step == END, "Unknown ADAPT evaluation milestone")
        return f"ADAPT{step:04d}"
    return f"SOURCE{step:04d}"


def checked_manifest(provider, parent, registration):
    step = registration["source_step"]
    plan = registration["v62_plan"]
    require(registration["v4_model_role"] == source_role(step, plan), "Unregistered evaluation role")
    audit = registration["source_audit"]
    proof = read(pin(audit["path"], audit["sha256"]))
    require(proof["science_plan"] == plan and proof["parent_sha256"] == MAIN and proof["initial_r4_receipt_sha256"] == R4
            and proof["step"] == step and proof["rng_restored"] and proof["complete_training_endpoint"]
            and proof["adam_counters"] == dict.fromkeys(GROUPS, step)
            and proof["provider_manifest_sha256"] == provider["sha256"], "Incomplete restore or wrong course")
    if plan == V62:
        require(proof["bank_receipt_sha256"] == BANK_RECEIPT
                and proof["actual_visits"] == {name: n*step for name, n in VISITS.items()}
                and proof["actual_source_visits"] == {name: n*step for name, n in SOURCE_VISITS.items()},
                "Incomplete SOURCE training exposure")
    else:
        from v64_adapt_endpoint import BANK, GO2, VISITS as ADAPT_VISITS

        require(proof["formal"] and proof["bank_receipt_sha256"] == BANK
                and proof["actual_visits"] == {name: n*step for name, n in ADAPT_VISITS.items()}
                and proof["actual_go2_population"] == {name: n*step for name, n in GO2.items()},
                "Incomplete ADAPT training exposure")
    folder = Path(provider["path"])
    model = read(pin(folder / "CHECKPOINT.json", provider["sha256"]))
    require(model["previous_checkpoint_sha256"] == parent["checkpoint_sha256"] == MAIN
            and model["training_receipt_sha256"] == proof["receipt_sha256"] == proof["training_terminal_sha256"]
            and model["training_contract_sha256"] == proof["contract_sha256"]
            and model["phase"] == step and model["role"] == "DIAGNOSTIC"
            and model["schema"] == "SLA_V4_MAIN_MULTISOURCE_INFERENCE"
            and model["source_codec"] == "DENSE_REFERENCE_G7_V6" and model["frozen_wbc"] and model["original_l1"]
            and not any(model[k] for k in ("formal_inference_eligible", "native_admission", "hardware_allowed", "physics"))
            and all(model[k] == parent[k] for k in ("foundation_sha256", "frozen_model", "action_sha256",
                    "state_sha256", "tokenizer_sha256", "partition_sha256")), "SOURCE provider ABI or provenance changed")
    for old, new in zip(parent["arrays"], model["arrays"], strict=True):
        require(all(old[k] == new[k] for k in ("file", "key", "shape", "dtype")), "SOURCE provider shape changed")
        pin(folder / new["file"], new["sha256"])
    return model


def comparison_rows(phase, places, holds, step, plan=V62):
    from v6_r4_native_deploy import ROLE, comparison_rows as matched

    role = source_role(step, plan)
    places, holds = matched(phase, places, holds)
    return tuple([dict(row, id=row["id"].replace(ROLE, role), pair_id=row["pair_id"].replace(ROLE, role))
                  for row in rows] for rows in (places, holds))


def launcher(path):
    text = pin(path, EXECUTOR).read_text()
    for before, after in (
        ('    if registration["v4_model_role"] == "R4_DEV_0200":\n        from v6_r4_native_deploy',
         '    if "source_step" in registration:\n'
         '        from v62_source_native import checked_manifest\n'
         '        return checked_manifest(reference, expected, registration)\n'
         '    if registration["v4_model_role"] == "R4_DEV_0200":\n        from v6_r4_native_deploy'),
        ('    if role == "R4_DEV_0200":', '    if role == "R4_DEV_0200" or "source_step" in registration:'),
        ('    require(len(conditions) == (4 if phase == "LONG_PLACE" else 2)',
         '    if "source_step" in registration:\n'
         '        from v62_source_native import source_role\n'
         '        role = source_role(registration["source_step"], registration["v62_plan"])\n'
         '        conditions = [dict(c, id=c["id"].replace("LONG_PARENT_", role + "_")) for c in conditions]\n'
         '    require(len(conditions) == (4 if phase == "LONG_PLACE" else 2)'),
        ('        registration = dict(additional, plan=plan)\n        model = comparison_manifest(registration)',
         '        source_step = binding.get("source_step")\n'
         '        if source_step is not None:\n'
         '            from v62_source_native import source_role\n'
         '            require(binding["role"] == source_role(source_step, binding["v62_plan"]) and phase == binding["source_phase"],\n'
         '                    "The evaluation role or phase changed")\n'
         '            scope = "SLA_" + binding["role"] + "_" + phase.removeprefix("LONG_")\n'
         '            additional.update(source_step=source_step, v62_plan=binding["v62_plan"],\n'
         '                              source_audit=binding["source_audit"])\n'
         '        registration = dict(additional, plan=plan)\n        model = comparison_manifest(registration)'),
        ('"PARENT" if r4 else binding["role"])','"PARENT" if r4 or source_step is not None else binding["role"])'),
        ('            places, holds = comparison_rows(phase, places, holds)\n',
         '            places, holds = comparison_rows(phase, places, holds)\n'
         '        if source_step is not None:\n'
         '            from v62_source_native import comparison_rows as source_rows\n'
         '            places, holds = source_rows(phase, places, holds, source_step, binding["v62_plan"])\n'),
    ):
        require(text.count(before) == 1, "The qualified native launcher seam changed")
        text = text.replace(before, after)
    compile(text, str(path), "exec")
    return text


def stage(base, endpoint, overlay, step, *, adapt=False):
    from scripts.pi0292.durable import write

    prior = base / "R4-DEV-0200-PLACE-OVERLAY-01"
    prior_panel = base / "v4-long-eval-noise-01/PARENT_PREFLIGHT_V2.json"
    panel = read(pin(prior_panel, PARENT_PANEL))
    parent = panel["candidate"]
    source_provider = endpoint / "PROVIDER"
    provider = dict(path=str(source_provider), sha256=sha(source_provider / "CHECKPOINT.json"))
    plan = V62
    if adapt:
        from v64_adapt_endpoint import PLAN as plan
    role = source_role(step, plan)
    registration = dict(v4_model_role=role, source_step=step, v62_plan=plan,
                        source_audit=reference(endpoint / "CPU-ENDPOINT-READBACK.json"))
    model = checked_manifest(provider, read(pin(Path(parent["path"]) / "CHECKPOINT.json", parent["sha256"])), registration)
    code = launcher(prior / SLA / "m38_a_native_v1.py")
    shutil.copytree(prior, overlay, copy_function=os.link, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    entry = overlay / SLA / "m38_a_native_v1.py"
    entry.unlink()
    entry.write_text(code)
    shutil.copy2(__file__, overlay / SLA)
    modules = ("v62_source_admission", "v62_source_endpoint", "v61_source_course", "v61_source_bank", "v6_dual_course")
    if adapt:
        modules += ("v64_adapt_endpoint", "v63_adapt_course", "v63_adapt_data")
    for name in modules:
        member = Path(import_module(name).__file__)
        target = overlay / SLA / member.name
        if target.exists():
            require(sha(target) == sha(member), f"Conflicting native evaluation module: {name}")
        else:
            shutil.copy2(member, target)
    prefix = "ADAPT" if adapt else "SOURCE"
    for phase in ("LONG_HOLD", "LONG_PLACE"):
        places, holds = comparison_rows(phase, panel["place"], panel["hold"], step, plan)
        registered = dict(registration, role=role, candidate=provider, source_phase=phase,
                          checkpoint_sha256=model["checkpoint_sha256"], place=places, hold=holds,
                          parent_panel=reference(prior_panel))
        write(overlay / f"{prefix}_{phase.removeprefix('LONG_')}_PREFLIGHT.json", dict(panel, **registered))


def read_phase(root, phase, step):
    from m38_a_native_v1 import bind_native, comparison_manifest, v4_panel_rows
    from m38_a_readout_v1 import read_arm

    require(all(read(root / part / "control/LEASE.json")["status"] == "CLOSED" for part in (".", "worker")),
            "Readout requires the original owner to close")
    registration = read(root / "COMPARISON.json")
    require(registration["source_step"] == step and registration["v4_block"] == "LONG_" + phase,
            "Wrong SOURCE evaluation group")
    model = comparison_manifest(registration)
    table = v4_panel_rows(registration)
    cases = registration["schedule"]
    require([row["id"] for row in table] == [row["id"] for row in cases]
            and all(row["checkpoint"] == model["checkpoint_sha256"] for row in cases), "Native roster changed")
    binding = registration["v4_panel"]
    config = read(pin(binding["path"], binding["sha256"]))
    noise = config["noise"][phase.lower()]
    require(noise["sha256"] == registration["gpu_noise_sha256"], "Native CUDA bank changed")
    expected = {row["noise_id"]: row["gpu_sha256"] for row in read(pin(noise["path"], noise["sha256"]))["tensors"]}
    scores = []
    for row in cases:
        case = root / row["id"]
        request = read(case / "COMPARISON_REQUEST.json")
        lease = read(case / "control/LEASE.json")
        require(lease["status"] == "CLOSED" and lease["spec_sha256"] == request["spec_sha256"], "Unclosed native request")
        spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
        require(spec["comparison_policy_seed"] == row["policy_seed"] and spec["interface_comparison_link"] == "L1",
                "Native seed or transport changed")
        bind_native(spec)
        score = read_arm(case, expected)[0]
        score["policy_seed"] = spec["comparison_policy_seed"]
        require(score["native_ticks"] <= row["max_ticks"] and score["forwards"] <= row["policy_queries"],
                "Unreconciled native budgets")
        if "grasp" not in score:
            terminal = read(pin(case / "RESULT.private.json", score["result_sha256"]))
            events = terminal["physics_stops"]
            score.update(grasp=False, task_success=False,
                         actual_stop=events[0] if events else terminal["mobile_safety_stop"])
        scores.append(score)
    worker = read(root / "worker/RESULT.private.json")
    require(sum(s["forwards"] for s in scores) == worker["completed_generations"], "Unreconciled frozen forwards")
    return dict(checkpoint=model["checkpoint_sha256"], arms=scores, valid=sum(s["valid_science"] for s in scores),
                audit=registration["source_audit"])


def settle(base, parent_release, hold, place, step, output):
    from scripts.pi0292.durable import write
    from v61_native_gate import stop_mechanism

    release = read(pin(parent_release, PARENT_RELEASE))
    require(release["released"] and release["r4_receipt_sha256"] == R4, "The R4 parent was not eligible")
    panels = {"HOLD": read_phase(hold, "HOLD", step), "PLACE": read_phase(place, "PLACE", step)}
    h, p = panels.values()
    require(h["checkpoint"] == p["checkpoint"] and h["audit"] == p["audit"], "The panels used different weights")
    failures = Counter()
    for phase, panel in panels.items():
        old = read(pin(base / f"R4-DEV-0200-{phase}-NATIVE-01/INDEPENDENT_PANEL_READBACK.json", R4_PANELS[phase]))
        archived = {row["id"].removeprefix("R4_DEV_0200_"): row for row in old["arms"]}
        for row in panel["arms"]:
            prior = archived[row["id"].removeprefix(source_role(step) + "_")]
            stop = stop_mechanism(row)
            if stop is not None and stop != stop_mechanism(prior):
                failures[stop] += 1
    hold_success = sum(row["valid_science"] and row.get("task_success", False) for row in h["arms"])
    grasps = sum(row["valid_science"] and row.get("grasp", False) for row in p["arms"])
    repeated = any(n >= 2 for n in failures.values())
    write(output, dict(plan=V62, source_step=step, model_checkpoint_sha256=h["checkpoint"],
                       audit=h["audit"], n02_valid=h["valid"], n02_success=hold_success,
                       place_valid=p["valid"], place_grasp=grasps, new_repeated_pregrasp_stop=repeated,
                       native_pass=h["valid"] == hold_success == 2 and p["valid"] == grasps == 4 and not repeated,
                       native_gpu_and_world_leases_closed=True, panels=panels))


def release(native, source, feedback, step, output):
    from scripts.pi0292.durable import write

    physical, dev, replay = map(read, (native, source, feedback))
    model = physical["model_checkpoint_sha256"]
    require(physical["plan"] == dev["plan"] == V62 and physical["source_step"] == dev["source_step"] == replay["step"] == step
            and physical["audit"] == dev["audit"] and physical["native_gpu_and_world_leases_closed"]
            and model == dev["checkpoint"] == replay["checkpoint"], "SOURCE results were scored across different milestones")
    require(replay["roster"]["sha256"] == FEEDBACK_ROSTER and replay["model_forwards"] == len(replay["rows"]) == 24
            and dev["model_forwards"] == 84 and all(d["summary"]["windows"] >= 16 for d in dev["domains"].values())
            and dev["physics_ticks"] == dev["optimizer_updates"] == replay["physics_ticks"] == replay["training_updates"] == 0
            and all(row["noise_sha256"] == row["actual_sampler_sha256"] for row in replay["rows"]),
            "The fixed source and Go2 feedback evidence is incomplete")
    audit = read(pin(physical["audit"]["path"], physical["audit"]["sha256"]))
    require(audit["bank_receipt_sha256"] == BANK_RECEIPT and audit["complete_training_endpoint"] and audit["rng_restored"],
            "The SOURCE optimizer endpoint is missing")
    write(output, dict(plan=PLAN, science_plan=V62, source_step=step, bank_receipt_sha256=BANK_RECEIPT,
                       training_terminal_sha256=audit["training_terminal_sha256"],
                       source_learned_or_underconverged=dev["source_learned"],
                       released=bool(physical["native_pass"] and dev["source_learned"]),
                       n02_valid=physical["n02_valid"], n02_success=physical["n02_success"],
                       place_valid=physical["place_valid"], place_grasp=physical["place_grasp"],
                       new_repeated_pregrasp_stop=physical["new_repeated_pregrasp_stop"],
                       model_checkpoint_sha256=model, native=reference(native), dev=reference(source), feedback=reference(feedback)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest="action", required=True)
    for name, fields in (("stage", ("base", "endpoint", "overlay")),
                         ("settle", ("base", "parent_release", "hold", "place", "output")),
                         ("release", ("native", "source", "feedback", "output"))):
        parser = actions.add_parser(name)
        for field in fields:
            parser.add_argument(field, type=Path)
        parser.add_argument("step", type=int, choices=ENDPOINTS)
        if name == "stage":
            parser.add_argument("--adapt", action="store_true")
    args = vars(cli.parse_args())
    action = args.pop("action")
    {"stage": stage, "settle": settle, "release": release}[action](**args)
