"""Reconstruct all three reference/measurement channels directly from closed #472 native ledgers."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from demonstrated_grip import pin, read, require, rows
from known_curve_scoring import shape_score
from known_dynamic_curve import trajectory
from known_dynamic_panel import CARDS, RATES, KnownCurve
from moving_stop_card import baseline, configure, record, sha


def errors(actual, target):
    from scipy.spatial.transform import Rotation

    dp = (actual[:, :3, 3] - target[:, :3, 3]) * 1000
    dr = (
        Rotation.from_matrix(
            actual[:, :3, :3] @ target[:, :3, :3].transpose(0, 2, 1)
        ).as_rotvec()
        * 180
        / np.pi
    )
    mm = np.linalg.norm(dp, axis=1)
    deg = np.linalg.norm(dr, axis=1)
    return {
        "count": len(mm),
        "mm_rms_max": [float(np.sqrt(np.mean(mm**2))), float(mm.max())],
        "deg_rms_max": [float(np.sqrt(np.mean(deg**2))), float(deg.max())],
        "xyz_rms_mm": np.sqrt(np.mean(dp**2, axis=0)).tolist(),
        "xyz_rotation_vector_rms_deg": np.sqrt(np.mean(dr**2, axis=0)).tolist(),
        "in_tube": float(np.mean((mm <= 4) & (deg <= 2))),
        "last_mm_deg": [float(mm[-1]), float(deg[-1])],
    }


def channel(actual, target):
    from scipy.spatial.transform import Rotation

    moving = slice(500, 2501)
    a, b = actual[moving], target[moving]

    def angles(p):
        return Rotation.from_matrix(p[:, :3, :3] @ p[0, :3, :3].T).as_rotvec()

    def shape(a, b):
        if np.linalg.norm(b - b[0]) < 1e-12:
            return {
                "relative_shape_score": None,
                "signed_gain": None,
                "reason": "zero_reference_variation",
            }
        return shape_score(a, b)

    initial_error_mm = (actual[0, :3, 3] - target[0, :3, 3]) * 1000
    motion_error_mm = (a[:, :3, 3] - b[:, :3, 3]) * 1000
    changes_a = np.diff(a[:, :3, 3], axis=0)
    changes_b = np.diff(b[:, :3, 3], axis=0)
    defined = np.linalg.norm(changes_b, axis=1) > 1e-12
    axes_a = [a[:, 0, 3], a[:, 2, 3], angles(a)[:, 1]]
    axes_b = [b[:, 0, 3], b[:, 2, 3], angles(b)[:, 1]]
    return {
        "entry_error_xyz_mm": initial_error_mm.tolist(),
        "motion_mean_error_xyz_mm": motion_error_mm.mean(axis=0).tolist(),
        "motion_displacement_error_rms_mm_from_entry": float(
            np.sqrt(np.mean(np.sum((motion_error_mm - initial_error_mm) ** 2, axis=1)))
        ),
        "axis_shapes": {
            key: shape(x[:, None], y[:, None])
            for key, x, y in zip(("world_x", "world_z", "world_pitch"), axes_a, axes_b)
        },
        "positive_peak_delay_ms": {
            key: (int(np.argmax(x)) - int(np.argmax(y))) * 2
            if np.ptp(x) > 1e-12 and np.ptp(y) > 1e-12
            else None
            for key, x, y in zip(("world_x", "world_z", "world_pitch"), axes_a, axes_b)
        },
        "opposed_or_stationary_translation_ms": int(
            2 * np.sum(defined & (np.sum(changes_a * changes_b, axis=1) <= 0))
        ),
        "hold": errors(actual[:501], target[:501]),
        "motion": errors(a, b),
        "park": errors(actual[2500:], target[2500:]),
        "final_one_second": errors(actual[-501:], target[-501:]),
        "translation": shape(a[:, :3, 3], b[:, :3, 3]),
        "rotation": shape(angles(a), angles(b)),
        "motion_excursion_mm": float(
            1000 * np.linalg.norm(a[:, :3, 3] - a[0, :3, 3], axis=1).max()
        ),
        "motion_excursion_deg": float(
            np.degrees(np.linalg.norm(angles(a), axis=1).max())
        ),
    }


def score(case):
    import mujoco
    from native_known import tcp
    from scipy.spatial.transform import Rotation, Slerp
    from verify_grip_card import states

    from scripts.pi0292.eval_contract import matrix10

    r = read(case / "RESULT.private.json")
    audit = read(case / "AUDIT.json")
    lease = read(case / "control/LEASE.json")
    spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    require(
        lease["status"] == "CLOSED"
        and lease["owner"] == spec["owner"]
        and Path(spec["registered_root"]) == case,
        "missing closed case identity",
    )
    out = case / r["episode"]
    shape = {
        "card": case.name,
        "outcome": r["outcome"],
        "eligible": r.get("eligible"),
        "violations": r.get("violations"),
        "exception": r.get("exception"),
        "audit": audit,
        "lease": "CLOSED",
        "end": r.get("end"),
        "sha256": {},
    }
    files = [
        case / "EXECUTION_SPEC.json",
        case / "control/LEASE.json",
        case / "RESULT.private.json",
        case / "AUDIT.json",
        case / "NATIVE_REQUEST.json",
        case / "INITIAL.private.json",
        out / "state.bin.journal",
        out / "physics.jsonl",
        out / "init.trace",
        out / "planner.jsonl",
        out / "MOBILE_SAMPLES.private.jsonl",
        case / "student/receiver.csv",
        case / "student/issuer.jsonl",
    ]
    files += [
        case / k
        for k in (
            "SERVO.private.jsonl",
            "DYNAMIC_ENTRY.json",
            "DYNAMIC_WIRE.jsonl",
            "DYNAMIC_TRACK.npz",
        )
        if (case / k).exists()
    ]
    shape["sha256"] = {str(p.relative_to(case)): sha(p) for p in files if p.is_file()}
    receiver = case / "student/receiver.csv"
    stopped = None
    after = 0
    native_steps = 0
    with receiver.open() as f:
        for event in csv.reader(f):
            if "STOP" in event[0] and stopped is None:
                stopped = int(event[2])
            if event[0] == "NATIVE_STEP":
                native_steps += 1
                after += int(stopped is not None and int(event[2]) > stopped)
    shape.update(
        receiver_steps=native_steps,
        first_receiver_stop_tick=stopped,
        step_records_after_receiver_stop=after,
        selected_MRT_policy_id=None,
    )
    if r["outcome"] != "DYNAMIC_COMPLETE":
        return shape
    require(
        audit["clean"] and r["end"] == native_steps == r["entry_tick"] + 3500,
        "incomplete dynamic execution",
    )
    entry = read(case / "DYNAMIC_ENTRY.json")
    start = entry["tick"]
    require(entry["entry_dwell_start"] + 500 == start, "missing entry dwell")
    scene = Path(spec["model_path"])
    model = mujoco.MjModel.from_binary_path(
        str(pin(scene, spec["frozen_files"][str(scene)]))
    )
    data = mujoco.MjData(model)
    names = [model.geom(i).name for i in range(model.ngeom)]
    tool = model.site("x5_ee").id
    actual = []
    for p in states(out):
        if p["step"] >= start:
            require(p["step"] == start + len(actual), "native clock gap")
            KnownCurve.support(names, p)
            actual.append(tcp(model, data, tool, p))
    actual = np.array(actual)
    rate, _ = case.name.rsplit("_", 1)
    nominal = trajectory(
        np.array(entry["nominal"]), np.arange(3501) * 0.002, *RATES[rate]
    )
    updates = rows(case / "DYNAMIC_WIRE.jsonl")
    atoms = [
        a
        for a in rows(out / "MOBILE_SAMPLES.private.jsonl")
        if a["native_start_step"] >= start
    ]
    require(len(atoms) == 140 and len(updates) == 70, "wrong scientific window")
    bywire = {u["wire"]: u for u in updates}
    accepted = set()
    original_wires = {}
    resampled = 0
    installed = set()
    with (out / "planner.jsonl").open() as f:
        for line in f:
            ev = json.loads(line)
            w = ev.get("ref_id")
            if w not in bywire:
                continue
            if ev["kind"] == "accept":
                require(
                    hashlib.sha256(ev["raw"].encode()).hexdigest() == w, "wire payload"
                )
                accepted.add(w)
            elif ev["kind"] == "installed":
                v = np.array(ev["poses"])
                clock = np.asarray(ev["times"])
                if w in original_wires:
                    original_t, original_v = original_wires[w]
                    require(
                        len(clock) == 80
                        and np.all(np.diff(clock) > 0)
                        and abs(ev["bound_start"] - original_t[0]) < 1e-8,
                        "unbound receding echo",
                    )
                    clipped = np.clip(clock, original_t[0], original_t[-1])
                    for offset in (0, 7):
                        xyz = np.column_stack(
                            [
                                np.interp(
                                    clipped, original_t, original_v[:, offset + j]
                                )
                                for j in range(3)
                            ]
                        )
                        rot = Slerp(
                            original_t,
                            Rotation.from_quat(original_v[:, offset + 3 : offset + 7]),
                        )(clipped)
                        require(
                            np.allclose(
                                v[:, offset : offset + 3], xyz, rtol=0, atol=2e-7
                            )
                            and np.allclose(
                                Rotation.from_quat(
                                    v[:, offset + 3 : offset + 7]
                                ).as_matrix(),
                                rot.as_matrix(),
                                rtol=0,
                                atol=2e-7,
                            ),
                            "receding echo changed geometry",
                        )
                    resampled += 1
                    continue
                require(
                    v.shape == (41, 14)
                    and np.allclose(
                        clock,
                        bywire[w]["source_tick"] * 0.002 + np.arange(41) * 0.05,
                        rtol=0,
                        atol=1e-8,
                    ),
                    "first wire clock",
                )
                original_wires[w] = clock, v
                k = np.asarray(bywire[w]["installed_0_50_100ms"])
                require(
                    np.allclose(v[:3, :3], k[:, :3, 3], rtol=0, atol=2e-7)
                    and np.allclose(
                        Rotation.from_quat(v[:3, 3:7]).as_matrix(),
                        k[:, :3, :3],
                        rtol=0,
                        atol=2e-7,
                    ),
                    "installed vs preview",
                )
                installed.add(w)
    require(
        accepted == installed == set(bywire) and len(accepted) == 70,
        "wire admission or install gap",
    )
    issued = [np.asarray(entry["consumed"])]
    for i, a in enumerate(atoms):
        require(
            a["native_start_step"] == start + 25 * i
            and a["native_end_step"] == start + 25 * (i + 1)
            and a["ticks"] == 25
            and a["safety_valid"]
            and a["solver_exact_installed_targets"]
            and a["installed_reference_sha256"] == updates[i // 2]["wire"],
            "unbound atom",
        )
        p, q = (
            matrix10(np.asarray(a[k]))
            for k in ("installed_EEF10", "target_endpoint_EEF10")
        )
        require(
            np.allclose(issued[-1], p, rtol=0, atol=2e-7), "unjoined issued channel"
        )
        u = np.arange(1, 26) / 25
        interp = np.repeat(np.eye(4)[None], 25, axis=0)
        interp[:, :3, :3] = Slerp(
            [0, 1], Rotation.from_matrix(np.stack((p[:3, :3], q[:3, :3])))
        )(u).as_matrix()
        interp[:, :3, 3] = p[:3, 3] + u[:, None] * (q[:3, 3] - p[:3, 3])
        issued.extend(interp)
    issued = np.array(issued)
    with np.load(case / "DYNAMIC_TRACK.npz") as z:
        for key, val in (("actual", actual), ("issued", issued), ("nominal", nominal)):
            require(
                val.shape == (3501, 4, 4)
                and np.allclose(z[key], val, rtol=0, atol=2e-7),
                "independent trace differs",
            )
    from known_curve_scoring import wire

    # #471 fixes the primary clock to the 50 ms nominal knots interpolated at
    # each 2 ms measurement. Keep the continuous formula only for provenance.
    analytic = nominal.copy()
    nominal[500:2501] = wire(np.array(entry["nominal"]), *RATES[rate])
    require(
        np.allclose(nominal[::25], analytic[::25], rtol=0, atol=2e-10),
        "nominal wire knots differ",
    )
    shape.update(
        entry_tick=start,
        primary_nominal_clock="50 ms knots, linear XYZ and shortest SLERP at 2 ms; no lag adjustment",
        dwell_start=entry["entry_dwell_start"],
        accepted_installed_wires=len(accepted),
        verified_receding_echoes=resampled,
        confirmed_atoms=len(atoms),
        verified_four_foot_samples=len(actual),
        achieved_to_nominal=channel(actual, nominal),
        issued_to_nominal=channel(issued, nominal),
        achieved_to_issued=channel(actual, issued),
        constant_to_nominal=channel(np.repeat(actual[:1], 3501, axis=0), nominal),
    )
    return shape


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--card", choices=CARDS)
    a = p.parse_args()
    root = a.root.resolve()
    configure(baseline(root))
    sys.path.insert(0, str(root / "v12/CPU_LIB"))
    campaign = root / "v17/known-dynamic-472"
    if a.card:
        record(a.out, score(campaign / a.card))
        sys.exit(0)
    require((campaign / "PANEL_CLOSE.json").exists(), "panel owner still active")
    cases = [
        score(campaign / card)
        for card in CARDS
        if (campaign / card / "AUDIT.json").is_file()
    ]
    record(
        a.out,
        {
            "registered": 4,
            "issued": sum(
                (campaign / c / "NATIVE_REQUEST.json").exists() for c in CARDS
            ),
            "case_audits": cases,
            "unstarted": [
                c for c in CARDS if not (campaign / c / "NATIVE_REQUEST.json").exists()
            ],
            "native_ticks": sum(c["audit"]["ticks"] for c in cases),
            "model_forwards": 0,
            "training_updates": 0,
        },
    )
