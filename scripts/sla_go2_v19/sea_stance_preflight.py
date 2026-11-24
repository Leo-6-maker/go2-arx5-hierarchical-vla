"""Cold evaluation of the one-time BASE anchor on real paused archive states."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

from sea_fruit_h0_cold import BASE, pin, read, require, sha
from sea_stance_attribution import G01, REPORT
from sea_stance_anchor import PROFILE, SETTLING_TICKS, check_document, reference
from d1_factorial_reference_matrix import held_part
from standing_reach import StateMonitor
from go2arx5_vla.world_model.mobile_wbc_reference_v1 import full_eef_gate
from go2arx5_vla.world_model.multitask_state_v6 import packets
from go2arx5_vla.world_model.native_replay import forbid_integration
from go2arx5_vla.trajectory.so3 import quat_to_matrix
from scripts.pi0292.eval_contract import matrix10

G00_CARRIER = "Z1_REACH_483/G00-A"


def reached_case(site, role):
    if role == "G01":
        published = read(REPORT)["G01"]
        seals = dict(published["evidence_sha256"])
        case = site / G01
        lease = read(pin(case / "control/LEASE.json", seals["LEASE.json"]))
        require(lease["status"] == "CLOSED", "unclosed fruit carrier")
        spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
        result = read(pin(case / "RESULT.private.json", seals["RESULT.private.json"]))
        native = case / result["episode"]
        log = pin(case / "HOLD_CONSUMPTION.private.jsonl", seals["HOLD_CONSUMPTION.private.jsonl"])
        start = published["entry_tick"]
        # End of task atom 0 is after the cutover, so use its preceding exact
        # paused boundary in the original measured atom receipt ledger.
        require(read(case / "AUDIT.json")["task"]["native_step"] == published["terminal_tick"],
                "fruit replay terminal mismatch")
        files = {k: pin(native / k, seals[k]) for k in ("planner.jsonl", "state.bin.journal")}
        state_log = native / "MOBILE_SAMPLES.private.jsonl"
        seals = {**seals, state_log.name: sha(state_log)}
        files[state_log.name] = pin(state_log, seals[state_log.name])
        with log.open() as stream:
            first_task = json.loads(next(stream))
        require(first_task["gripper"]["native_step"]-25 == start, "candidate phase is not the first task pause")
        return spec, start, files, True
    sealed = read(pin(site / "v18/sealed-Z1/members.json",
                      read(BASE / "reports/SLA_GO2_V18/Z1_NATIVE/ARCHIVE.json")["member_manifest_sha256"]))
    case = site / "v18"
    fields = ("EXECUTION_SPEC.json", "REACH_ENTRY.json", "planner.jsonl", "state.bin.journal",
              "MOBILE_SAMPLES.private.jsonl")
    files = {}
    for name in fields:
        key, = [p for p in sealed if p.startswith(f"{G00_CARRIER}/") and p.endswith(f"/{name}")]
        files[name] = pin(case / key, sealed[key])
    return read(files["EXECUTION_SPEC.json"]), read(files["REACH_ENTRY.json"])["tick"], files, False


def evaluate(site, role):
    import mujoco

    spec, start, inputs, fruit = reached_case(site, role)
    with inputs["MOBILE_SAMPLES.private.jsonl"].open() as f:
        sample = next(a for a in map(json.loads, f) if a["native_end_step"] == start)
    with inputs["planner.jsonl"].open() as f:
        original = next(e for e in map(json.loads, f)
                        if e["kind"] == "accept" and e["ref_id"] == sample["installed_reference_sha256"])
    require(sample["solver_exact_installed_targets"] and sample["native_end_step"] == start,
            "entry previously consumed reference absent")
    doc = json.loads(original["raw"])
    old_base, held_ee = held_part(doc, "base"), held_part(doc, "ee")
    np.testing.assert_allclose(matrix10(sample["target_endpoint_EEF10"]), held_ee, rtol=0, atol=2e-7)
    action, base, proof = reference(old_base, held_ee, sample["measured_MPC_mount"])
    actual = np.asarray(sample["actual_TCP"])
    with forbid_integration(mujoco) as calls:
        gate = full_eef_gate(action, actual)
        accepted = check_document(action, base)
        for key, expected in (("ee", held_ee), ("base", base[0])):
            actual_wire = held_part(accepted, key)
            np.testing.assert_allclose(actual_wire, expected, rtol=0, atol=2e-7)
            assert_array_equal([row["position"] for row in accepted[key]["waypoints"]],
                               np.repeat(actual_wire[None, :3, 3], 41, axis=0))

        model_path = spec["model_path"]
        model = mujoco.MjModel.from_binary_path(str(pin(Path(model_path), spec["frozen_files"][model_path])))
        monitor = StateMonitor(model)
        for packet in packets(inputs["state.bin.journal"]):
            if start-SETTLING_TICKS <= packet["step"] <= start:
                monitor.observe(packet)
            if packet["step"] == start:
                break
        require(monitor.stable_now() and monitor.last == start, "archived physical stance not stable")
    require(not calls, "anchor preflight integrated dynamics")
    return dict(archive_entry_tick=start, actual_fruit_scene=fruit, observed_settled_states=len(monitor.stable),
                old_accepted_EEF_preserved=True, constant_base_waypoints=41, original_EEF_gate=gate,
                source_wire_sha256=sample["installed_reference_sha256"],
                journal_sha256=sha(inputs["state.bin.journal"]),
                correction_magnitude_mm=float(np.linalg.norm(proof["base_reference_shift_mm"])),
                **proof)


def run(site):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only anchor preflight")
    rows = {layout: evaluate(site, layout) for layout in ("G00", "G01")}
    current = np.eye(4)
    corrupt = current.copy()
    corrupt[0, 0] = np.nan
    try:
        reference(current, current, corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite native mount passed")
    return dict(status="STANCE_ANCHOR_STATIC_VALIDATED", profile=PROFILE,
                minimum_settling_ticks=SETTLING_TICKS,
                stance_measurement_rebased_after_cutover=False,
                same_native_hold_and_task_thresholds=True, only_once_before_first_teacher_query=True,
                cases=rows, new_integrations=0, new_rendered_frames=0, model_forwards=0,
                training_updates=0, native_dispatch_qualified=False)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("site", type=Path)
    print(json.dumps(run(cli.parse_args().site.resolve()), indent=2))
