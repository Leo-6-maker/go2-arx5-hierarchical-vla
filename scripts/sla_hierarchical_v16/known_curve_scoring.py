"""Challenge the two registered dynamic references using controls with known motion."""

import json
import sys
from pathlib import Path

import numpy as np
from sealed_panel import configure_site


def shape_score(actual, target):
    # Remove the starting coordinates before demeaning to retain static zeros exactly.
    actual = actual - actual[0]
    target = target - target[0]
    actual -= actual.mean(axis=0)
    target -= target.mean(axis=0)
    energy = float(np.sum(target**2))
    if energy <= 0:
        raise ValueError("reference has no variation for this channel")
    return {
        "relative_shape_score": 1 - float(np.sum((actual - target) ** 2)) / energy,
        "signed_gain": float(np.sum(actual * target)) / energy,
    }


def metrics(measured, reference):
    from scipy.spatial.transform import Rotation

    if measured.shape != reference.shape or reference.shape != (2001, 4, 4):
        raise ValueError("expected synchronized four-second pose traces at 2 ms")

    def angles(p):
        return Rotation.from_matrix(p[:, :3, :3] @ p[0, :3, :3].T).as_rotvec()

    dx = 1000 * np.linalg.norm(measured[:, :3, 3] - reference[:, :3, 3], axis=1)
    dr = np.rad2deg(
        Rotation.from_matrix(
            measured[:, :3, :3] @ reference[:, :3, :3].transpose(0, 2, 1)
        ).magnitude()
    )
    return {
        "peak_mm": float(dx.max()),
        "peak_deg": float(dr.max()),
        "rms_mm": float(np.sqrt(np.mean(dx**2))),
        "rms_deg": float(np.sqrt(np.mean(dr**2))),
        "full_tube": bool(dx.max() <= 4 and dr.max() <= 2),
        "translation": shape_score(measured[:, :3, 3], reference[:, :3, 3]),
        "rotation": shape_score(angles(measured), angles(reference)),
    }


def wire(anchor, linear, angular):
    from known_dynamic_curve import trajectory
    from scipy.spatial.transform import Rotation, Slerp

    knots = np.arange(20, 101) * 0.05
    times = np.arange(500, 2501) * 0.002
    poses = trajectory(anchor, knots, linear, angular)
    result = np.repeat(np.eye(4)[None], len(times), axis=0)
    result[:, :3, 3] = np.column_stack(
        [np.interp(times, knots, x) for x in poses[:, :3, 3].T]
    )
    result[:, :3, :3] = Slerp(knots, Rotation.from_matrix(poses[:, :3, :3]))(
        times
    ).as_matrix()
    return result


def measured_hold(root, archive):
    import hashlib

    import mujoco
    from demonstrated_grip import pin, read, require
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from native_known import tcp
    from verify_grip_card import states

    case = root / "feedback/sla-feedback-f1-a0-20261123-01" / archive["card"]
    result = read(pin(case / "RESULT.private.json", archive["result_sha256"]))
    spec = read(
        pin(
            case / "EXECUTION_SPEC.json",
            read(case / "control/LEASE.json")["spec_sha256"],
        )
    )
    native = case / result["episode"]
    start = archive["ticks"] - 2000
    pin(native / "state.bin.journal", archive["journal_sha256"])

    def lines(name):
        with (native / name).open() as f:
            for line in f:
                yield json.loads(line)

    atoms = [
        a for a in lines("MOBILE_SAMPLES.private.jsonl") if a["native_end_step"] > start
    ]
    require(
        [a["native_start_step"] for a in atoms]
        == list(range(start, archive["ticks"], 25))
        and all(
            a["native_start_step"] + 25 == a["native_end_step"] and a["safety_valid"]
            for a in atoms
        ),
        "hold interval coverage",
    )
    hashes = {a["installed_reference_sha256"] for a in atoms}
    accepted = set()
    installed = {}
    receipt_points = []
    for e in lines("planner.jsonl"):
        key = e.get("ref_id")
        if key not in hashes:
            continue
        if e["kind"] == "accept":
            require(
                hashlib.sha256(e["raw"].encode()).hexdigest() == key,
                "bad accepted bytes",
            )
            accepted.add(key)
        if e["kind"] == "installed":
            p = np.asarray(e["poses"])
            require(
                p.ndim == 2
                and p.shape[1] == 14
                and len(p) == len(e["times"]) > 1
                and np.array_equal(p, np.broadcast_to(p[0], p.shape)),
                "forecast still moving",
            )
            if installed:
                require(
                    np.array_equal(p[0], next(iter(installed.values()))),
                    "target moved between publications",
                )
            installed[key] = p[0]
            receipt_points.append(len(p))
    require(
        set(installed) == accepted == hashes
        and np.allclose(
            list(installed.values()), next(iter(installed.values())), atol=1e-8, rtol=0
        ),
        "installed target changed between holds",
    )
    model = mujoco.MjModel.from_binary_path(
        str(pin(Path(spec["model_path"]), spec["frozen_files"][spec["model_path"]]))
    )
    data = mujoco.MjData(model)
    tip = model.site("x5_ee").id
    with forbid_integration(mujoco) as calls:
        poses = np.array(
            [
                tcp(model, data, tip, s)
                for s in states(native)
                if start <= s["step"] <= archive["ticks"]
            ]
        )
        require(not calls and len(poses) == 2001, "unmeasured hold or integration")
    return poses, {
        "first_tick": start,
        "last_tick": archive["ticks"],
        "verified_atoms": len(atoms),
        "installation_lengths": {
            str(n): receipt_points.count(n) for n in sorted(set(receipt_points))
        },
        "distinct_installs": len(hashes),
        "journal_sha256": archive["journal_sha256"],
    }


def run(root, repo, out):
    from ablation import output
    from demonstrated_grip import pin, read, require
    from scipy.spatial.transform import Rotation

    reports = repo / "reports/SLA_HIERARCHICAL_V16"
    f = read(
        pin(
            root / "feedback/F1_VERIFIED.json",
            read(reports / "V0_EVIDENCE_MATRIX_ACTUAL.json")["source_hashes"]["F1"],
        )
    )["records"]
    speeds = (("SLOW", 0.004, np.deg2rad(0.6)), ("SOURCE_SPEED", 0.01125, 0.045))
    controls = {}
    for key, linear, angular in speeds:
        ref = wire(np.eye(4), linear, angular)
        parked = np.repeat(ref[:1], len(ref), axis=0)
        center = parked.copy()
        center[:, 2, 3] += linear / (2 * np.pi * 1.875 / 4)
        reverse = ref.copy()
        reverse[:, :3, 3] = 2 * ref[:1, :3, 3] - ref[:, :3, 3]
        reverse[:, :3, :3] = Rotation.from_matrix(ref[:, :3, :3]).inv().as_matrix()
        controls[key] = {
            name: metrics(actual, ref)
            for name, actual in (
                ("exact", ref),
                ("parked_at_start", parked),
                ("parked_at_center", center),
                ("sign_reversed", reverse),
            )
        }
    population = {}
    for row in f:
        result = {"original_outcome": row["outcome"], "eligible": row["eligible"]}
        if row["eligible"]:
            poses, ledger = measured_hold(root, row)
            result.update(
                ledger=ledger,
                conditions={
                    key: metrics(poses, wire(poses[0], v, w)) for key, v, w in speeds
                },
            )
        population[row["card"]] = result
    require(
        len(population) == 12
        and sum(bool(r["eligible"]) for r in population.values()) == 4,
        "wrong F1 population",
    )
    output(
        out,
        {
            "controls": controls,
            "archived_stationary_holds": population,
            "new_native_starts": 0,
            "physics_integrations": 0,
            "new_forwards": 0,
        },
    )


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parents[2]
    configure_site(root, repo)
    sys.path.insert(0, str(repo / "scripts/sla_transfer_feedback"))
    run(root, repo, Path(sys.argv[2]).resolve())
