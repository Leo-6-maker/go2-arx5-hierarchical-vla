"""Bind a nonproduction warmup comparison to its original registered inputs."""

import argparse
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write


CANDIDATE = "ac84febc5752f47fddfd364db5396bb4eb1c3363e45ca81a7d6487201e66b808"
BASELINE = "573b0732ff446a1b4bf26af487452caae74ffa4f381c135c4aeb5dfeb2bea80b"
PROFILE = "9c47048fe8a5318a679711a19654be21aba5d1532fa99864ce19c1e43bfd448a"
SLOT = "MEV1-ENG-INTERFACE-03"


def ref(path, expected=None):
    return dict(path=str(path), sha256=sha(path) if expected is None else expected)


def stage(stock, stock_sha, candidate, baseline, profiles, output):
    from go2arx5_vla.world_model.multitask_scope_v6 import read_pin, validate, validate_frozen_assets
    from scripts.sla.core_multienv_geometry import load_population
    from scripts.sla.v64_n2_native import binding, prepare_case
    from scripts.wm02.multitask_stage_cards_v6 import require_unstarted_cases

    spec = read(pin(stock, stock_sha))
    built = read(pin(candidate / "RESULT.json", CANDIDATE))
    binary = pin(candidate / "rl_sim_mujoco", built["new_binary_sha256"])
    require(built["status"] == "COMPILED_AND_PHASE_LOGIC_CHECKED" and not built["native_state_qualified"]
            and built["fixture_sha256"] == PROFILE and not (built["native_starts"] or built["native_integrations"])
            and built["clock_windows_checked"] == 400 and built["same_dynamic_dependency_identity"],
            "Only the identified CPU build may enter this comparison")
    bound, population, rows, worlds = binding(spec["v64_n2"])
    row, = (r for r in rows if r["family"] == "F1" and r["motion"] == "STABLE")
    authority = read_pin(bound["original_authority"])
    slot, = (item for item in load_population(Path(authority["logical"]["path"]))["engineering"]
             if item["id"] == SLOT)
    require(slot["kind"] == "SHORT" and slot["profile"] == "SHORT_FIXED_REFERENCE"
            and slot["max_native_requests"] == slot["parallel_worlds"] == 1 and not slot["training_eligible"]
            and slot["max_native_ticks"] == 10703 and authority["tolerance"] == population["tolerances"],
            "The original one-world nontraining engineering slot and state tolerance are required")
    require_unstarted_cases(spec["registry"], {SLOT})

    closed = read(baseline / "control/LEASE.json")
    monitor = read(pin(baseline / "EXECUTION_SPEC.json", closed["spec_sha256"]))
    evidence = read(pin(baseline / "RESULT.private.json", BASELINE))
    require(closed["status"] == "CLOSED" and evidence["shutdown_clean"] and not evidence["unsafe"]
            and not evidence["technical_failure"] and evidence["actual_native_integrations"] == 10703
            and monitor["panel_entry_id"] == row["n1_comparator"]
            and monitor["site_world_spec"]["fixture"] == row["reference"]
            and row["n1_frozen_verdict"] == "OUTSIDE_FROZEN_STATE_TOLERANCE",
            "The actual closed N1 negative comparator must be preserved")
    fixture, = (p["phase_fixture"] for p in read(pin(profiles, PROFILE))["optional_phase_fixture"]
                if p["comparison"] == "F1_STABLE_N1")
    archived_result = Path(row["archived_comparator"]["path"]).parent / "init.trace"
    pin(archived_result, fixture["trace_sha256"])
    bits = fixture["require_callback"]
    require(len(bits) == 100 and sum(bits) == 35 and all(type(bit) is bool for bit in bits),
            "Use the F1 STABLE archived callback mask")

    _, case = prepare_case(spec,row,worlds[row["world_role"]],bound,{},spec["deadline_epoch"])
    validate(case)
    require(sha(case["native_binary"]) == built["original_binary_sha256"],
            "Baseline and isolated candidate must share the original live simulator ancestry")
    frozen = {}
    for role in ("controller_binary","solver_binary","mobile_safety_library"):
        asset = Path(case[role])
        frozen[role] = ref(pin(asset,case["frozen_files"][str(asset)]),case["frozen_files"][str(asset)])
    alternate = dict(case, native_binary=str(binary),
                     frozen_files={**case["frozen_files"], str(binary):built["new_binary_sha256"]})
    rejected = {}
    for name, check in (("stock_entry",validate), ("stock_binary_identity",validate_frozen_assets)):
        try:
            check(alternate)
        except PermissionError as exc:
            rejected[name] = str(exc)
        else:
            raise RuntimeError("The independent build unexpectedly passed the stock route")

    write(output, dict(schema="V64_PD_PHASE_TRIAL_PREFLIGHT", reference_slot=slot,
            actual_native_requests=0, native_integrations=0, slot_unstarted_at_readback=True,
            simulator_candidate=ref(binary,built["new_binary_sha256"]), original_stock_gate_rejections=rejected,
            native_authority_issued=False, live_permitted=False, compiled_result=ref(candidate/"RESULT.json",CANDIDATE),
            current_stock_preflight=ref(stock,stock_sha),
            reference=row["reference"], archived_state=row["archived_comparator"], original_tolerance=authority["tolerance"],
            unchanged_binaries=frozen, unchanged_mobile_safety=case["mobile_safety_binding"],
            matched_closed_negative=ref(baseline/"RESULT.private.json",BASELINE),
            optional_callback_mask="".join("1" if bit else "0" for bit in bits),
            sensor_fixture=ref(profiles,PROFILE), inherited_goal_start_epoch=authority["goal_start"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stock", "stock_sha", "candidate", "baseline", "profiles", "output"):
        parser.add_argument(name, type=str if name == "stock_sha" else Path)
    stage(**vars(parser.parse_args()))
