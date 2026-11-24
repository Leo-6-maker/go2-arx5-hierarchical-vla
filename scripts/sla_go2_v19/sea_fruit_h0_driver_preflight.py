"""Replay the stationary H0 driver's BASE binding and physical gripper grant path."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sea_fruit_h0_cold import BASE, pin, read, require, sha
from sea_fruit_h0_native import gripper_receipt, grip_at_grant, verify_fixed_base
from sea_fruit_h0_replay import Z1, Z1_OTHER
from d1_factorial_reference_matrix import held_part
from go2arx5_vla.world_model.multitask_teacher_v6 import MultitaskTeacher
from go2arx5_vla.world_model.task_learning_grasp_v1 import FiftyMsProgram
from scripts.pi0291_arx5_data.native_dataset import pose10


def must_reject(check, *values):
    try:
        check(*values)
    except ValueError:
        return
    raise AssertionError(f"missing expected rejection from {check.__name__}")


def archived_bases(site):
    sealed = read(pin(site / "v18/sealed-Z1/members.json",
                      read(BASE / "reports/SLA_GO2_V18/Z1_NATIVE/ARCHIVE.json")["member_manifest_sha256"]))
    rows = {}
    for role, ref in (("G00", Z1), ("G01", Z1_OTHER)):
        key, = [k for k in sealed if k.startswith(ref + "/") and k.endswith("/planner.jsonl")]
        with pin(site / "v18" / key, sealed[key]).open() as f:
            planner = [json.loads(line) for line in f]
        accepted = {e["ref_id"]: e for e in planner if e["kind"] == "accept"}
        installs = [e for e in planner if e["kind"] == "installed"
                    and e.get("ref_id") in accepted and e["bound_start"] >= 5703*.002 - 1e-8]
        require(len(installs) > 1, "actual native installations missing")
        # Verify the same initial stand mount at every publication in each archive.
        mount = held_part(json.loads(accepted[installs[0]["ref_id"]]["raw"]), "base")
        verify_fixed_base(installs, mount)
        changed = deepcopy(installs[0])
        changed["poses"][20][7] += .001
        must_reject(verify_fixed_base, [changed], mount)
        rows[role] = dict(installed_publications=len(installs), ledger_sha256=sealed[key],
                          changed_one_point_1mm_rejected=True)
    return rows


def closed_command():
    # Stationary Cartesian pose isolates the original Teacher's known width trajectory
    # from target pose changes. The original P10 preview and full H40 gate still run.
    anchor = np.eye(4)
    start = 6203
    teacher = MultitaskTeacher(
        {"object_pose": anchor, "tcp": anchor, "base_pose": anchor},
        {"family": "F1", "mode": "M_ASSIST"}, {"direction_world_XY": [1., 0.]}, "HOLD-red")
    teacher.close_start = start
    adapter = SimpleNamespace(mobile_contact_stage="close")
    widths = []
    for block in range(5):
        tick = start + 250 * block
        row = pose10(np.repeat(anchor[None], 40, axis=0), teacher.grip_trajectory(tick))
        plan = FiftyMsProgram(row, anchor, anchor, np.array([0., 0., 0., 0., 0., 0., 1.]))
        for atom in range(10):
            if atom in (0, 7):
                adapter.mobile_installed_step = tick + 25*atom
                adapter.active_reference = {"action_EEF10": plan.preview(atom)}
            widths.append(grip_at_grant(adapter, tick + 25*atom))
    speeds = np.abs(np.diff(widths)) / .05
    require(abs(widths[0]-.042) < 1e-12 and abs(widths[-1]-.002) < 1e-12
            and float(max(speeds)) < .04 and max(speeds[9::10]) < .04, "working slide closure")
    adapter.mobile_contact_stage = "approach"
    must_reject(grip_at_grant, adapter, adapter.mobile_installed_step)
    must_reject(grip_at_grant, adapter, adapter.mobile_installed_step + 1)
    return dict(grants=len(widths), slide_range_mm=[2, 42],
                maximum_slide_speed_mm_s=float(max(speeds))*1000,
                maximum_P10_seam_slide_speed_mm_s=float(max(speeds[9::10]))*1000,
                off_grid_grant_and_closed_approach_rejected=True)


def actual_receiver_ack(replay_root):
    path = replay_root / "CONTACT/SELECTED_WHILE_CLOSING"
    receipt = read(replay_root / "RESULT.json")
    expected = receipt["synthetic_contact_treatments"]["SELECTED_WHILE_CLOSING"]["receiver_log_sha256"]
    with pin(path / "receiver.csv", expected).open() as f:
        epoch = next(f).split(",")[1]
    ack = gripper_receipt(path, epoch, 157, 6428, .041)
    must_reject(gripper_receipt, path, epoch, 157, 6428, .042)
    must_reject(gripper_receipt, path, "unknown", 157, 6428, .041)
    return dict(**ack, receiver_log_sha256=expected,
                wrong_width_and_unknown_epoch_rejected=True,
                finger_state_from_nonintegrating_replay=True)


def run(site, replay_root):
    import mujoco
    from go2arx5_vla.world_model.native_replay import forbid_integration

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU only driver acceptance")
    with forbid_integration(mujoco) as calls:
        data = dict(archived_BASE=archived_bases(site), exact_teacher_P10_grip=closed_command(),
                    compiled_receiver_ACK=actual_receiver_ack(replay_root))
    require(not calls, "driver acceptance attempted integration")
    return dict(status="CPU_H0_DRIVER_SEAMS_VERIFIED", **data, new_native_starts=0,
                new_physics_ticks=0, new_rendered_frames=0, new_model_forwards=0,
                receiver_proof_sha256=sha(replay_root / "RESULT.json"), whole_system_qualified=False)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--site", type=Path, required=True)
    cli.add_argument("--receiver-replay", type=Path, required=True)
    args = cli.parse_args()
    print(json.dumps(run(args.site, args.receiver_replay), indent=2))
