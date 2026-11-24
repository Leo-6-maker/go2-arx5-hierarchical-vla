"""Compare one proposed update at each real #472 A0 boundary. CPU only."""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from demonstrated_grip import pin, read, require, rows
from known_dynamic_curve import trajectory
from known_dynamic_panel import RATES, KnownCurve
from moving_stop_card import BASE, baseline, configure, record


def preview(nominal, anchor, actual, cfg):
    """Transport the consumed offset once and add one bounded world-frame error correction."""
    from feedback import capped, rotation, rotvec

    goal = nominal[0]
    dx = capped(
        cfg["translation_gain_per_update"] * (goal[:3, 3] - actual[:3, 3]),
        cfg["max_step_mm"] / 1000,
    )
    dr = capped(
        cfg["rotation_gain_per_update"] * rotvec(goal[:3, :3] @ actual[:3, :3].T),
        np.deg2rad(cfg["max_step_deg"]),
    )
    cmd = nominal.copy()
    cmd[:, :3, 3] += anchor[:3, 3] - goal[:3, 3]
    cmd[:, :3, :3] = nominal[:, :3, :3] @ goal[:3, :3].T @ anchor[:3, :3]
    for i in range(1, len(cmd)):
        f = min(1.0, i / 2)
        cmd[i, :3, 3] += f * dx
        cmd[i, :3, :3] = rotation(f * dr) @ cmd[i, :3, :3]
    cmd[-1] = cmd[-2]
    return cmd


def replay(root, result, cfg):
    import mujoco
    from d1_geometry import sweep
    from feedback import ReferenceManager, boundary, rotvec
    from go2arx5_vla.world_model.mobile_safety_profile_v1 import binding
    from native_known import atom_receipt
    from native_servo import measure
    from verify_grip_card import states

    case = root / "v17/known-dynamic-472" / result["card"]
    for path, h in result["sha256"].items():
        pin(case / path, h)
    res = read(case / "RESULT.private.json")
    out = case / res["episode"]
    spec = read(case / "EXECUTION_SPEC.json")
    scene = Path(spec["model_path"])
    model = mujoco.MjModel.from_binary_path(
        str(pin(scene, spec["frozen_files"][str(scene)]))
    )
    guard = binding(spec["native_root"])
    parts = {
        k: np.array(v)
        for k, v in read(case / "INITIAL.private.json")["features"].items()
        if k in ("old_ee", "old_base")
    }
    entry = read(case / "DYNAMIC_ENTRY.json")
    first = entry["tick"]
    atoms = {
        r["native_end_step"]: r for r in rows(out / "MOBILE_SAMPLES.private.jsonl")
    }
    accepted = {
        r["ref_id"]: r for r in rows(out / "planner.jsonl") if r["kind"] == "accept"
    }
    installed = {
        r["ref_id"] for r in rows(out / "planner.jsonl") if r["kind"] == "installed"
    }
    original = rows(case / "DYNAMIC_WIRE.jsonl")
    snapshots = {
        p["step"]: p
        for p in states(out)
        if first <= p["step"] < first + 3500 and (p["step"] - first) % 50 == 0
    }
    require(len(snapshots) == len(original) == 70, "missing decision boundary")
    names = [model.geom(i).name for i in range(model.ngeom)]
    rate = RATES[result["card"].rsplit("_", 1)[0]]
    stats = {}
    for mode in ("A1", "A2"):
        windows = []
        for i, (tick, packet) in enumerate(snapshots.items()):
            KnownCurve.support(names, packet)
            last = atoms[tick]
            wire = last["installed_reference_sha256"]
            require(
                wire in installed
                and wire
                == (entry["entry_wire"] if i == 0 else original[i - 1]["wire"]),
                "unbound prior wire",
            )
            source = round(accepted[wire]["source_mpc_s"] / 0.002)
            sample = {
                "qpos": np.asarray(packet["qpos"]),
                "tcp": np.asarray(last["actual_TCP"]),
                "time_s": packet["time_s"],
                "safety_valid": last["safety_valid"],
                "contact_valid": True,
                "native_grasp": last["native_grasp"],
                "robot_object_contact": False,
            }
            require(
                np.allclose(sample["qpos"], last["qpos"], rtol=0, atol=1e-9),
                "state/atom join",
            )
            receipt = atom_receipt(out, last, sample, source, wire)
            measured = measure(out, tick, sample)
            anchor, actual = boundary(out.name, tick, wire, receipt, measured)
            nominal = trajectory(
                np.array(entry["nominal"]),
                (tick - first) * 0.002 + np.arange(41) * 0.05,
                *rate,
            )
            no_error = preview(nominal, anchor, nominal[0], cfg)
            require(
                np.allclose(
                    no_error[:3], original[i]["installed_0_50_100ms"], rtol=0, atol=2e-7
                ),
                "nominal-only A0 control",
            )
            row = {"tick": tick, "accepted": False, "geometry_poses": 0}
            windows.append(row)
            try:
                if mode == "A1":
                    a = ReferenceManager(out.name, parts["old_ee"], cfg)
                    a.next_tick, a.last_wire, a.anchor = tick, wire, anchor.copy()
                    knots = a.propose("A1", tick, wire, receipt, measured, nominal[0])
                else:
                    knots = preview(nominal, anchor, actual, cfg)
                KnownCurve.admit(knots, receipt, sample)
                distance = (
                    np.linalg.norm(knots[:, :3, 3] - parts["old_ee"][:3, 3], axis=1)
                    * 1000
                )
                angle = np.degrees(
                    [
                        np.linalg.norm(rotvec(x[:3, :3] @ parts["old_ee"][:3, :3].T))
                        for x in knots
                    ]
                )
                require(
                    distance.max() <= cfg["max_translation_from_initial_ref_mm"] + 1e-9
                    and angle.max() <= cfg["max_rotation_from_initial_ref_deg"] + 1e-9,
                    "WORKSPACE_CAP",
                )
                geometry = np.unique(knots.reshape(len(knots), 16), axis=0).reshape(
                    -1, 4, 4
                )
                for pose in geometry:
                    sweep(
                        model,
                        sample,
                        dict(parts, old_ee=anchor),
                        pose,
                        parts["old_base"],
                        guard,
                    )
                    row["geometry_poses"] += 1
                row.update(
                    accepted=True,
                    command_mm=float(
                        1000 * np.linalg.norm(knots[2, :3, 3] - anchor[:3, 3])
                    ),
                    command_deg=float(
                        np.degrees(
                            np.linalg.norm(rotvec(knots[2, :3, :3] @ anchor[:3, :3].T))
                        )
                    ),
                    delta_relative_A0_mm=float(
                        1000 * np.linalg.norm(knots[2, :3, 3] - no_error[2, :3, 3])
                    ),
                    delta_relative_A0_deg=float(
                        np.degrees(
                            np.linalg.norm(
                                rotvec(knots[2, :3, :3] @ no_error[2, :3, :3].T)
                            )
                        )
                    ),
                )
            except (ValueError, RuntimeError, AssertionError) as exc:
                row["failure"] = str(exc)
        stats[mode] = {
            "accepted": sum(r["accepted"] for r in windows),
            "total": len(windows),
            "failures": dict(Counter(r["failure"] for r in windows if "failure" in r)),
            "windows": windows,
        }
    return {"source": result["card"], "rate": list(rate), "modes": stats}


def main():
    import mujoco
    from go2arx5_vla.world_model.native_replay import forbid_integration

    cfgpath = BASE / "reports/SLA_FROZEN_FEEDBACK/F0_REGISTRATION.json"
    cfg = read(
        pin(cfgpath, read(cfgpath.with_name("F0_MODEL.json"))["registration_sha256"])
    )["controller"]
    parent = read(BASE / "reports/SLA_GO2_V17/DYNAMIC_472/ACTUAL.json")
    cases = []
    with forbid_integration(mujoco) as count:
        for r in parent["case_audits"]:
            if r["outcome"] == "DYNAMIC_COMPLETE":
                cases.append(replay(root, r, cfg))
    require(len(cases) == 2, "missing completed A0 donor")
    record(
        args.out,
        {
            "method": "One counterfactual proposal per real A0 boundary, anchored to the actual prior A0 receipt. No shadow A1/A2 outputs fed into later inputs.",
            "controls": "The zero-error A2 proposal must reproduce all three consumed A0 knots for every interval.",
            "native_starts": 0,
            "model_forwards": 0,
            "training_updates": 0,
            "integration_guard": count,
            "cases": cases,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    configure(baseline(root))
    sys.path.insert(0, str(root / "v12/CPU_LIB"))
    main()
