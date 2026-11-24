"""Bind four independent provisional VAL cards to frozen teacher evidence and compiled scenes."""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts/sla_hierarchical_v16"))
from demonstrated_grip import pin, read, require, rows

SELECTION = (
    "V17-VAL-F1-HOLD-red",
    "V17-VAL-F1-HOLD-blue",
    "V17-VAL-F1-PLACE-red-right",
    "V17-VAL-F2-HOLD-blue",
)
OLD_RESULT = "d7b749541d0474d343aa80ebfc64829f7df81193517106ea9c7a52506c29cd25"
OLD_PANEL = "62daa720bf9e131106792946ae0dae1022037a3252fe93f17420db305bc94271"
OLD_SCOPE = "89f058d5cd3a69cae8f68eb6a63f63a040f20151d07daf0d88e6a0ec5c3f8ffe"


def audit_effort_stop(case, receipt, terminal):
    import mujoco
    import numpy as np
    from go2arx5_vla.world_model.multitask_state_v6 import packets

    lease = read(case / "control/LEASE.json")
    spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    require(
        lease["status"] == "CLOSED" and terminal["shutdown_clean"],
        "historic actuator stop not terminal",
    )
    stop = terminal["mobile_safety_stop"]
    require(
        stop["reason"] == receipt["first_failure"] == "NATIVE_ACTUATOR_EFFORT"
        and terminal["trace_audit"]["STOP_observed"] is False
        and receipt["post_STOP_ticks"] is None,
        "original exceptional STOP classification differs",
    )
    native = case / "native-01"
    require(
        read(native / "MOBILE_SAFETY_STOP.private.json") == stop,
        "actuator stop receipt differs",
    )
    stop_tick = round(stop["time_s"] / stop["native_dt_s"])
    require(
        abs(stop_tick * stop["native_dt_s"] - stop["time_s"]) < 1e-8,
        "effort stop off native clock",
    )
    qualification = spec["mobile_guard_qualification"]
    guard = read(pin(Path(qualification["path"]), qualification["sha256"]))
    pin(Path(spec["mobile_safety_library"]), guard["library_sha256"])
    binding = pin(Path(spec["mobile_safety_binding"]), guard["binding_sha256"])
    scene = Path(spec["model_path"])
    model = mujoco.MjModel.from_binary_path(
        str(pin(scene, spec["frozen_files"][str(scene)]))
    )
    words = binding.read_text().split()
    require(
        words[0] == "MOBILE_SAFETY_V1" and int(words[9]) == 18 and len(words) == 46,
        "actuator bound table changed",
    )
    joints = {}
    for joint, threshold in zip(words[10::2], words[11::2], strict=True):
        idx = np.flatnonzero(
            (model.actuator_trnid[:, 0] == model.joint(joint).id)
            & (model.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT))
        )
        require(
            len(idx) == 1 and model.actuator_gear[idx[0], 0] == 1.0,
            "actuator names and torque gain disagree",
        )
        joints[joint] = (int(idx[0]), float(threshold))
    terminal_controls = []
    journal = native / "state.bin.journal"
    last = -1
    seen = 0
    after = 0
    for packet in packets(journal):
        tick = packet["step"]
        require(
            (seen == 0 and tick in (0, 1)) or tick == last + 1, "old effort journal gap"
        )
        last = tick
        after += int(tick > stop_tick)
        seen += 1
        if stop_tick - 3 <= tick <= stop_tick:
            terminal_controls.append(
                {
                    "tick": tick,
                    "joint4_command_nm": float(packet["ctrl"][joints["x5_joint4"][0]]),
                    "command_over_limit": [
                        joint
                        for joint, (idx, threshold) in joints.items()
                        if abs(packet["ctrl"][idx]) > threshold
                    ],
                }
            )
    resume = max(
        r["requested_step"] for r in rows(native / "CLOCK_RESUME_REQUESTS.jsonl")
    )
    commands = rows(native / "COMMANDS.private.jsonl")
    ticks = [r["native_step"] for r in commands if r.get("exact_receiver_knot") is True]
    require(
        last
        == stop_tick
        == terminal["actual_native_integrations"]
        == terminal["trace_audit"]["native_step_records"]
        and resume > stop_tick
        and max(ticks) <= stop_tick
        and after == 0,
        "effort archive closure cannot be cross-checked",
    )
    require(
        len(terminal_controls) == 4
        and not any(x["command_over_limit"] for x in terminal_controls[:3])
        and terminal_controls[-1]["command_over_limit"] == ["x5_joint4"],
        "archived actuator overrun cannot be isolated",
    )
    return dict(
        slot=receipt["condition"],
        verdict="PRESERVED_HISTORICAL_EFFORT_STOP",
        stop_tick=stop_tick,
        last_request_tick=resume,
        last_integrated_tick=last,
        last_recorded_exact_consumption_tick=max(ticks),
        recorded_frames_after_effort_stop=after,
        original_trace_has_explicit_STOP=False,
        original_result_sha256=receipt["result"]["sha256"],
        journal_sha256=_sha(journal),
        archived_processes_exited_cleanly=True,
        actual_effort_timeseries_available=False,
        guarded_joint_count=len(joints),
        joint4_limit_nm=joints["x5_joint4"][1],
        final_recorded_command=terminal_controls,
    )


def _sha(path):
    import hashlib

    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run(archive, compiled, static):
    from scripts.wm02.multitask_stage_cards_v6 import language_for, panel_cards

    proof = read(static)
    population = read(pin(compiled, proof["compiled_manifest_sha256"]))
    require(
        proof["status"] == "COMPILED_STATIC_ONLY"
        and not population["native_dispatch_qualified"]
        and proof["native_requests"]
        == proof["model_forwards"]
        == proof["training_updates"]
        == 0,
        "compiled geometry boundary",
    )
    old = archive / "sla-core-hopper-v1-expert-chain-20261121-02"
    original_scope = read(pin(old / "EXECUTION_SPEC.json", OLD_SCOPE))
    reference = read(pin(old / "RESULT.private.json", OLD_RESULT))
    original = {
        r["id"]: r for r in read(pin(old / "REGISTERED_PANEL.private.json", OLD_PANEL))
    }
    require(
        reference["source_sha"] == original_scope["source_sha"]
        and read(old / "control/LEASE.json")["status"] == "CLOSED",
        "historical producer identity or terminal state",
    )
    by_group = defaultdict(Counter)
    positives = {}
    stops = Counter()
    effort_stop = None
    for r in reference["records"]:
        item = original[r["condition"]]
        family, task = item["condition"]["family"], item["instruction"]
        group = f"{family}/{task}"
        by_group[group][f"{r['split']}_closed"] += 1
        result_path = pin(Path(r["result"]["path"]), r["result"]["sha256"])
        terminal = read(result_path)
        require(
            terminal["source_sha"] == reference["source_sha"]
            and terminal["physical_requests"] == r["requests"] == 1
            and terminal["actual_native_integrations"] == r["ticks"]
            and terminal["model_generations"] == terminal["optimizer_updates"] == 0,
            "historical native receipt mismatch",
        )
        if r["condition"] == reference["unreconciled"]:
            effort_stop = audit_effort_stop(result_path.parent, r, terminal)
        if r["hard_STOP"]:
            stops[f"{r['first_failure']}/after={r['post_STOP_ticks']}"] += 1
        if r["admitted"]:
            score = r["score"]
            require(
                terminal["shutdown_clean"]
                and not terminal["unsafe"]
                and r["task_and_mobile_success"]
                and score["task_success"]
                and score["ever_target_grasp"]
                and score["ever_target_lift20mm"]
                and not score["wrong_object_grasp_ever"]
                and not score["wrong_object_contact_ever"],
                "teacher success lost physical provenance",
            )
            by_group[group][f"{r['split']}_success"] += 1
            if group not in positives and r["split"] == "TRAIN":
                positives[group] = dict(
                    id=r["condition"],
                    result_sha256=r["result"]["sha256"],
                    stage_budget_s=original_scope["stage_budget_s"],
                    grasp_step=score["events"]["target_grasp"]["native_step"],
                    lift20mm_step=score["events"]["target_lift20mm"]["native_step"],
                    hold_s=score["longest_grasp_lift_hold_s"],
                    stable_release_s=score["longest_stable_released_retreat_s"],
                )
    require(
        effort_stop is not None,
        "unreconciled legacy actuator effort event must remain visible",
    )
    old_templates = {}
    for family, group in (("F1", "F1/HOLD-red"), ("F2", "F2/HOLD-blue")):
        name = positives[group]["id"]
        spec_path = old / "cases" / name / "EXECUTION_SPEC.json"
        lease = read(spec_path.parent / "control/LEASE.json")
        require(lease["status"] == "CLOSED", "historic teacher case still active")
        old_templates[family] = read(pin(spec_path, lease["spec_sha256"]))
    val = {r["id"]: r for r in population["conditions"]["VAL"]}
    panel = panel_cards(
        population,
        [(dict(val[k], noise_seed=val[k]["seed"]), None) for k in SELECTION],
        old_templates,
        role="EXPERT",
    )
    rows = []
    for item in panel:
        slot = item["id"]
        frozen = proof["slots"][slot]
        group = f"{frozen['family']}/{frozen['task']}"
        pos = positives[group]
        require(
            item["scene"]["sha256"] == frozen["scene_sha256"]
            and item["condition"]["seed"] == frozen["seed"]
            and item["condition"]["reset"] == frozen["reset"]
            and group in positives,
            "heldout card does not bind",
        )
        language = language_for(item["instruction"])
        path = pin(Path(item["scene"]["path"]), frozen["scene_sha256"])
        require(path.stat().st_size == frozen["scene_bytes"], "scene asset damaged")
        rows.append(
            dict(
                slot=slot,
                split="VAL",
                family=frozen["family"],
                task=frozen["task"],
                language=language,
                mode=item["condition"]["mode"],
                seed=frozen["seed"],
                reset=frozen["reset"],
                geometry_sha256=frozen["scene_sha256"],
                prospective_max_ticks=35703
                if frozen["task"].startswith("HOLD")
                else 50703,
                historical_positive=pos,
                historical_group_counts=dict(by_group[group]),
                permit_status="NOT_ALLOCATED",
            )
        )
    require(
        len(rows) == 4 and sum(r["prospective_max_ticks"] for r in rows) == 157812,
        "VAL proposal denominator",
    )
    return dict(
        status="PROSPECTIVE_GEOMETRY_BOUND_VAL_ONLY",
        rows=rows,
        remaining_VAL=sorted(set(val) - set(SELECTION)),
        protected_TEST_count=len(population["conditions"]["TEST"]),
        historical_hard_STOP_counts=dict(stops),
        legacy_named_unreconciled_effort_stop=effort_stop,
        historical_teacher_groups={k: dict(v) for k, v in sorted(by_group.items())},
        registered_legacy_authority_reusable=False,
        placed_test_rows_into_development=0,
        max_candidate_native_requests=4,
        max_candidate_ticks=157812,
        actual_native_requests=0,
        actual_ticks=0,
        gpu_forwards=0,
        training_updates=0,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("archive", "compiled", "static"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    os.umask(0o077)
    print(json.dumps(run(a.archive, a.compiled, a.static), indent=2, allow_nan=False))
