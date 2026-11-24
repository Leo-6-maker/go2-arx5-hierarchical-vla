"""Verify both optical copies and replay two archived native traces on CPU."""

import argparse
import json
from pathlib import Path
import os

import numpy as np
from numpy.testing import assert_array_equal

from sea_fruit_h0_cold import BASE, pin, read, require, sha
from sea_fruit_camera import fruit_body_model
from sea_fruit_h0_score import StationaryHoldScore
from sea_three_view import calibration_sha256, camera_copy
from go2arx5_vla.world_model.multitask_teacher_v6 import MultitaskTeacher
from go2arx5_vla.world_model.multitask_state_v6 import TwoObjectState, packets
from go2arx5_vla.world_model.native_replay import forbid_integration
from scripts.pi0292.eval_contract import matrix10
from scipy.spatial.transform import Rotation

MOBILE = "CHV1-TRAIN-F1-HOLD-red-00"
Z1 = "Z1_REACH_483/G00-A"
Z1_OTHER = "Z1_REACH_483_G01/G01-A"


def replay(scene, geometry, journal, start, end=None):
    import mujoco

    score = StationaryHoldScore(mujoco.MjModel.from_binary_path(str(scene)), geometry)
    for packet in packets(journal):
        if packet["step"] >= start:
            score.update_packet(packet)
        if packet["step"] == end:
            break
    require(score.start == start and (end is None or score.last == end), "archived score coverage")
    return score.result()


def verify_cameras(folder, layouts):
    import mujoco

    evidence = {}
    with forbid_integration(mujoco) as calls:
        for layout in layouts:
            path = pin(folder / layout["layout"] / "SCENE.private.mjb", layout["scene_sha256"])
            original = mujoco.MjModel.from_binary_path(str(path))
            body = fruit_body_model(path)
            auxiliary, camera = camera_copy(path)
            ids = np.flatnonzero(np.arange(original.ncam) != camera)
            for field in ("cam_bodyid", "cam_mode", "cam_pos", "cam_quat", "cam_fovy"):
                for model in (body, auxiliary):
                    assert_array_equal(getattr(original, field)[ids], getattr(model, field)[ids])
            evidence[layout["layout"]] = dict(
                body_xyz_m=body.cam_pos[camera].tolist(),
                body_fovy_deg=float(body.cam_fovy[camera]),
                auxiliary_xyz_m=auxiliary.cam_pos[camera].tolist(),
                auxiliary_fovy_deg=float(auxiliary.cam_fovy[camera]),
                wrist_mount_unchanged=True,
            )
    require(not calls, "optical copies attempted native integration")
    return evidence


def cold_teacher(site, folder, layout, carrier, sealed):
    """A first open approach proposal on archived robot state and static fruit pose."""
    import mujoco

    rel = Path(carrier)
    entry = str(rel / "REACH_ENTRY.json")
    tick = read(pin(site / "v18" / entry, sealed[entry]))["tick"]
    journal, = [k for k in sealed if k.startswith(carrier + "/") and k.endswith("/state.bin.journal")]
    samples, = [k for k in sealed if k.startswith(carrier + "/") and k.endswith("/MOBILE_SAMPLES.private.jsonl")]
    packet = next(x for x in packets(pin(site / "v18" / journal, sealed[journal])) if x["step"] == tick)
    with pin(site / "v18" / samples, sealed[samples]).open() as stream:
        prior = next(a for a in map(json.loads, stream) if a["native_end_step"] == tick)
    require(prior["solver_exact_installed_targets"], "archived approach entry lacks consumption")
    anchor = matrix10(np.array(prior["target_endpoint_EEF10"]))
    mount = np.asarray(prior["actual_mount"])
    base = np.r_[mount[:3, 3], Rotation.from_matrix(mount[:3, :3]).as_quat()]
    scene = pin(folder / layout["layout"] / "SCENE.private.mjb", layout["scene_sha256"])
    model = mujoco.MjModel.from_binary_path(str(scene))
    with forbid_integration(mujoco) as calls:
        edited = dict(packet, qpos=np.r_[packet["qpos"][:27], model.qpos0[27:]],
                      qvel=np.r_[packet["qvel"][:26], np.zeros(12)])
        initial = TwoObjectState(model, "red").update(edited)
        teacher = MultitaskTeacher(initial, {"family": "F1", "mode": "M_ASSIST"},
                                   layout["scoring_geometry"], "HOLD-red")
        plan, evidence = teacher.decide(initial, anchor, base)
        first, second = plan.preview(0), plan.preview(7)
        assert_array_equal(first[7:], second[:33])
        assert_array_equal(first[:, 9], np.repeat(first[0, 9], len(first)))
        require(evidence["stage"] == "PREGRASP" and len(first) == len(second) == 40,
                "expected one original open-gripper H40 at the archived entry")
    require(not calls, "teacher preview attempted integration")
    return dict(archive_entry_tick=tick, stage=evidence["stage"], H40_points=40,
                physical_displacement_mm=float(1000*np.linalg.norm(first[-1,:3]-first[0,:3])),
                gripper_command_fraction=float(first[0, 9]), overlapping_points_exact=33,
                archived_robot_plus_compiled_static_objects=True)


def run(site, archive, folder):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only replay")
    compiled = read(folder / "RESULT.json")
    sealed = read(pin(site / "v18/sealed-Z1/members.json",
                      read(BASE / "reports/SLA_GO2_V18/Z1_NATIVE/ARCHIVE.json")[
                          "member_manifest_sha256"]))
    key = f"{Z1}/EXECUTION_SPEC.json"
    standing = read(pin(site / "v18" / key, sealed[key]))
    start_key = f"{Z1}/REACH_ENTRY.json"
    start = read(pin(site / "v18" / start_key, sealed[start_key]))["tick"]
    journal, = [p for p in sealed if p.startswith(f"{Z1}/") and p.endswith("/state.bin.journal")]
    standing_scene = Path(standing["model_path"])
    stationary = replay(
        pin(standing_scene, standing["frozen_files"][str(standing_scene)]),
        standing["multitask_geometry"], pin(site / "v18" / journal, sealed[journal]), start,
    )

    historical = archive / "sla-core-hopper-v1-expert-chain-20261121-02/cases" / MOBILE
    roster = read(BASE / "reports/SLA_GO2_V18/P1_P2/VAL_CANDIDATES.json")
    ref, = [r["historical_positive"] for r in roster["rows"]
            if r["historical_positive"]["id"] == MOBILE]
    old = read(pin(historical / "RESULT.private.json", ref["result_sha256"]))
    require(old["completed"] and old["shutdown_clean"] and old["detail"]["task_success"],
            "historical mobile positive control incomplete")
    lease = read(historical / "control/LEASE.json")
    require(lease["status"] == "CLOSED", "historical carrier still active")
    spec = read(pin(historical / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    model = Path(spec["model_path"])
    mobile_journal = historical / "native-01/state.bin.journal"
    journal_sha = sha(mobile_journal)
    mobile = replay(
        pin(model, spec["frozen_files"][str(model)]), spec["multitask_geometry"],
        mobile_journal, old["detail"]["stages"][0]["native_step"], old["detail"]["terminal_native_step"],
    )
    pin(mobile_journal, journal_sha)
    require(not stationary["task_success"] and not stationary["ever_target_grasp"]
            and stationary["unsafe_states"] == 0 and not mobile["task_success"]
            and mobile["ever_target_lift20mm"]
            and mobile["longest_grasp_lift_hold_s"] >= 1.
            and mobile["first_stance_or_contact_violation"] is not None,
            "archived controls must distinguish standing and mobile HOLD")
    teachers = {row["layout"]: cold_teacher(site, folder, row, carrier, sealed)
                for row, carrier in zip(compiled["layouts"], (Z1, Z1_OTHER), strict=True)}
    return dict(
        cameras=verify_cameras(folder, compiled["layouts"]),
        initial_teacher_open_approach=teachers,
        auxiliary_calibration_sha256=calibration_sha256(),
        archived_standing_reach=dict(journal_sha256=sealed[journal], **stationary),
        archived_mobile_hold=dict(journal_sha256=journal_sha, **mobile),
        new_integrations=0, new_rendered_frames=0, new_model_generations=0,
        new_updates=0, native_dispatch_qualified=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.site, args.archive, args.compiled), indent=2))
