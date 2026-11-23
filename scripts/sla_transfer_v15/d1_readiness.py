"""Freeze the CPU qualification and eight-card D1 roster before observing a D1 outcome."""

import argparse
import json
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC / "scripts/sla_transfer_v14"))
from b2_owner import baseline, configure, pinned
from identity import read, record, rows, sha

PLAN = "docs/protocols/SLA_MEMORY_V15_D1_FACTORIAL_PREREGISTRATION_20261123.json"
PLAN_PIN = "a4ff35963d273206ceadc9fe612c352650af726a075b9508abffdd6fbc04c465"
B2_PIN = "ed2d54421c6d2d2005f6c1b2f542fcd2bb7970f6a240f4dd196db6558e034ea9"
D0_PIN = "a97a1fc5c8ec72df77363d06c8a6f50322df72c9f9d03f86b8cfe759e4434ab0"


def callback(directory, parts, action, base, old_sequence):
    import numpy as np
    from clock_replay import GuardedManager, episode
    from d1_factorial_reference_matrix import check, fixed, pose7
    from go2arx5_vla.world_model.mobile_wbc_reference_v1 import document
    from ocs2_msgs.msg import MpcObservation
    from shadow import check as check_message
    from shadow import message

    directory.mkdir()
    with episode(directory.name, directory):
        manager = GuardedManager()

        def install(now, seq, payload):
            state = MpcObservation()
            state.time = float(now)
            state.mode = 1
            state.state.value = pose7(parts["measured_ee"]).tolist()
            manager.on_ee_state(state)
            obs = MpcObservation()
            obs.time = float(now)
            obs.state.value = [0.0] * 12
            bm = parts["measured_base"]
            r = bm[:3, :3]
            obs.state.value = [
                *bm[:3, 3],
                np.arctan2(r[1, 0], r[0, 0]),
                np.arcsin(-r[2, 0]),
                np.arctan2(r[2, 1], r[2, 2]),
                *([0.0] * 6),
            ]
            manager.on_observation(obs)
            payload["_pi0294"] = {
                "episode": directory.name,
                "sequence": seq,
                "source_mpc_s": now,
                "ttl_s": 0.4,
                "operation": "install",
            }
            start = len(manager.target_pub)
            manager.on_trajectory_json(message(payload))
            check(len(manager.target_pub) == start + 1, "shadow install rejected")
            return check_message(manager.target_pub[-1], 14)

        old_action, old_base = fixed("OO", parts)
        olddoc = document(old_action, mode="M_ASSIST", base_poses=old_base)
        # Seed the fresh offline episode with the same initialized sequence count and legal clocks.
        for seq in range(old_sequence + 1):
            install(5.006 + 0.35 * seq, seq, olddoc)
        newdoc = document(action, mode="M_ASSIST", base_poses=base)
        from scripts.pi0292.eval_contract import matrix10

        ee = matrix10(action[0])
        for i in range(10):
            target = install(11.406 + 0.1 * i, old_sequence + 1 + i, newdoc)
            expected = np.r_[pose7(ee), pose7(base[0])]
            check(
                target.shape == (41, 14)
                and np.allclose(target, expected, rtol=0, atol=2e-7),
                "incorrect world field",
            )
        check(not manager._r4_mode and not manager.clock_fault, "unexpected HOLD")
        count = len(manager.target_pub)
        manager.on_trajectory_json(message(newdoc))
        check(len(manager.target_pub) == count, "stale replay accepted")
    return {
        "new_installs": 10,
        "points_checked": 410,
        "stale_reinstall_rejected": True,
        "source_clock": "offline fixture",
        "native_acks": 0,
    }


def archive(root, out, public, card):
    import mujoco
    import numpy as np
    from d1_factorial_reference_matrix import (
        CELLS,
        boundary,
        check,
        digest,
        fixed,
        initial_features,
        prior_pd,
    )
    from d1_geometry import sweep
    from go2arx5_vla.world_model.mobile_safety_profile_v1 import binding
    from go2arx5_vla.world_model.mobile_wbc_reference_v1 import full_eef_gate
    from go2arx5_vla.world_model.native_replay import forbid_integration, packets

    from scripts.pi0292.eval_contract import matrix10
    from scripts.pi0294.task_path_model import certify_chunk
    from scripts.pi0294.task_state import ActualState

    group = root / "v14/b2-technical-20261123-01"
    folder = group / card["card"]
    pinned(folder / "EXECUTION_SPEC.json", card["spec_sha256"])
    pinned(folder / "RESULT.private.json", card["result_sha256"])
    result = read(folder / "RESULT.private.json")
    spec = read(folder / "EXECUTION_SPEC.json")
    native = folder / result["episode"]
    check(
        read(folder / "control/LEASE.json")["status"] == "CLOSED",
        "historical lease status",
    )
    for name, pin in card["evidence_sha256"].items():
        pinned(native / name, pin)
    for name, pin in spec["frozen_files"].items():
        pinned(Path(name), pin)
    identity = read(native / "LOADED_EXECUTOR_00.private.json")
    for k, pin in public["frozen_sha256"].items():
        check(identity[k]["executable_sha256"] == pin, "loaded executable " + k)
    source = Path(spec["source_root"])
    parent = spec["trusted_source_parent"]
    pinned(Path(parent["spec_path"]), parent["spec_sha256"])
    for name, pin in spec["source_files"].items():
        pinned(source / name, pin)
    for f in (
        "scripts/pi0294/task_path_model.py",
        "scripts/pi0294/task_state.py",
        "scripts/pi0294/mp_client.py",
        "scripts/pi0294/task_contacts.py",
    ):
        pinned(
            Path(parent["source_root"]) / f,
            read(Path(parent["spec_path"]))["source_files"][f],
        )
    hist = list(rows(native / "MOBILE_SAMPLES.private.jsonl"))
    hs = [r for r in hist if r["native_end_step"] in (5678, 5703)]
    check(len(hs) == 2, "boundary history")
    parts, old, seq = boundary(native, hs[-1])
    old_history = [
        {"end_native_step": r["native_end_step"], "pose": matrix10(r["installed_EEF10"])}
        for r in hs
    ]
    ps = packets(native / "state.bin.journal")
    packet = ps[5702]
    check(
        packet["step"] == 5703
        and abs(packet["time_s"] - 11.406) < 1e-8
        and np.array_equal(packet["qpos"], hs[-1]["qpos"])
        and np.array_equal(packet["qvel"], hs[-1]["qvel"]),
        "state clock/identity",
    )
    pd = prior_pd(native / "init.trace")
    features = initial_features(packet, parts, pd, hs[-1]["installed_EEF10"][9])
    accepted = next(
        json.loads(e["raw"])
        for e in rows(native / "planner.jsonl")
        if e["kind"] == "accept"
        and json.loads(e["raw"])["_pi0294"]["sequence"] == seq + 1
    )
    from d1_factorial_reference_matrix import held_part

    for k, w in (("measured_ee", "ee"), ("measured_base", "base")):
        check(
            np.allclose(held_part(accepted, w), parts[k], atol=1e-7, rtol=0),
            "archived MM reconstruction",
        )
    with forbid_integration(mujoco) as calls:
        model = mujoco.MjModel.from_binary_path(spec["model_path"])
        sample = ActualState(model).update(packet)
        check(
            np.max(abs(sample["tcp"] - parts["measured_ee"])) < 1e-10,
            "native model FK differed",
        )
        guard = binding(spec["native_root"])
        case_results = []
        for cell in CELLS:
            action, base = fixed(cell, parts)
            ee = matrix10(action[0])
            gate = full_eef_gate(action, sample["tcp"])
            try:
                cert = certify_chunk(
                    model,
                    sample,
                    matrix10(action),
                    stage="approach",
                    history=old_history,
                )
                shadow = cert["status"]
            except ValueError as error:
                shadow = str(error)
            cb = callback(out / (card["card"] + "-" + cell), parts, action, base, seq)
            geo = sweep(model, sample, parts, ee, base[0], guard)
            case_results.append(
                {
                    "cell": cell,
                    "reference_content_sha256": digest([ee.tolist(), base[0].tolist()]),
                    "geometry": geo,
                    "callback": cb,
                    "fixed_root_shadow": shadow,
                    "full40_gate": True,
                    "max_anchor_distance_m": gate["max_anchor_distance_m"],
                }
            )
        check(not calls, "unexpected integration")
    private = {
        "prior_reference_wire_sha256": old,
        "parts": {k: v.tolist() for k, v in parts.items()},
        "initial_features": features,
        "last_sequence": seq,
        "source_scene_sha256": spec["frozen_files"][spec["model_path"]],
    }
    private_path = out / (card["card"] + "_anchor.private.json")
    record(private_path, private)
    return features, {
        "source": card["card"],
        "spec_sha256": card["spec_sha256"],
        "native_result_sha256": card["result_sha256"],
        "source_sha": spec["source_sha"],
        "parent_source_sha": parent["source_sha"],
        "scene_sha256": private["source_scene_sha256"],
        "previous_wire_sha256": old,
        "private_anchor_sha256": sha(private_path),
        "source_evidence_sha256": card["evidence_sha256"],
        "cases": case_results,
    }


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("root", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    root = args.root.resolve()
    out = args.output.resolve()
    cfg = baseline(root)
    configure(cfg)
    from d1_factorial_reference_matrix import check, distances, matching_rule

    plan_path = SRC / PLAN
    pinned(plan_path, PLAN_PIN)
    plan = read(plan_path)
    public_path = root / "v14/b2-technical-20261123-01/B2_PUBLIC_READOUT.json"
    pinned(public_path, B2_PIN)
    d0 = root / "v14/B2_STATIC_BIAS_ATTRIBUTION.json"
    pinned(d0, D0_PIN)
    public = read(public_path)
    os.umask(0o077)
    out.mkdir(parents=True, exist_ok=False)
    archived = [archive(root, out, public, c) for c in public["cards"]]
    anchors, origins = zip(*archived, strict=True)
    limits = matching_rule(anchors)
    roster = [
        {"wave": w["name"], "factor": k, "id": w["name"] + "-" + k, "cap": 6203}
        for w in plan["waves"]
        for k in w["cells"]
    ]
    reg = {
        "schema": "D1_FACTORIAL_4X2_REGISTER",
        "plan_sha256": PLAN_PIN,
        "historical_b2_public_sha256": B2_PIN,
        "historical_D0_sha256": D0_PIN,
        "roster": roster,
        "cap_per_wave": 24812,
        "max_ticks": 49624,
        "initialization_ticks": 5703,
        "physics_dt_s": 0.002,
        "prior_difference": distances(*anchors),
        "pairwise_initial_limits": limits,
        "prior_anchor_sha256": [v["private_anchor_sha256"] for v in origins],
        "matching": "Each new reset must lie within every registered feature bound against both B2 anchors and every earlier D1 reset. Three times the archived two-reset difference or the fixed numeric floor, whichever is larger.",
        "source_sha256": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
        "estimands": plan["estimands"],
        "descriptive_contrast_scale_mm": 5,
        "owner": "fresh per physical wave",
        "hardware": False,
        "epoch": "new for every card",
        "scene_sha256": origins[0]["scene_sha256"],
        "seed": "same frozen compiled scene and native reset with no per-card seed override; match recorded initial state",
        "new_gpu_forwards": 0,
        "new_optimizer_updates": 0,
    }
    record(out / "D1_FACTORIAL_4X2_REGISTER.json", reg)
    check(
        len(roster) == 8 and sum(r["cap"] for r in roster) == 49624,
        "factorial population",
    )
    record(
        out / "D1_READINESS_ACTUAL.json",
        {
            "schema": "D1_CPU_READINESS",
            "registration_sha256": sha(out / "D1_FACTORIAL_4X2_REGISTER.json"),
            "anchors": origins,
            "frozen_sha256": public["frozen_sha256"],
            "source_cards": 2,
            "source_cells": 8,
            "new_physical_requests": 0,
            "new_ticks": 0,
            "native_authority": "held only by actual owner after separate lease",
            "dynamic_safety": "requires original live guard",
            "cpu_ready": True,
        },
    )
    print("CPU D1 readiness and registration sealed", flush=True)


if __name__ == "__main__":
    main()
