"""Read sealed F0 traces, decompose tracking and test a local lagged response model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from feedback import OPEN, START, ReferenceManager, StopMonitor, require
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned(path, digest):
    require(sha(path) == digest, "EVIDENCE_DIGEST_" + path.name)


def read(path):
    return json.loads(path.read_text())


def rms(v):
    return float(np.sqrt(np.mean(np.sum(v * v, axis=-1))))


def xyz(v):
    return np.asarray(v).tolist()


def d1(root, reference):
    out = []
    for r in reference["records"]:
        card = root / "v15/sla-d1-matched-b2-20261123-01" / r["slot"]
        pinned(card / "RESULT.private.json", r["result_sha256"])
        entry = {
            "slot": r["slot"],
            "eligible": r["eligible"],
            "factor": r["factor"],
            "ticks": r["ticks"],
        }
        if r["eligible"]:
            f = card / "CURVE.private.json"
            pinned(f, r["per_tick_curve_sha256"])
            rows = read(f)
            require(
                [p["elapsed_ms"] for p in rows] == list(range(0, 1001, 2)), "D1_GRID"
            )

            def field(name, rows=rows):
                return np.array([p[name] for p in rows])

            absolute = field("absolute_error_mm")
            servo = field("pd_fk_minus_actual_mm")
            upstream = field("pd_fk_minus_reference_mm")
            inc = field("incremental_error_mm")
            base = field("base_only_delta_mm")
            closure = float(np.max(np.abs(absolute - upstream + servo)))
            require(
                closure < 1e-8 and np.allclose(inc, absolute - absolute[0]),
                "D1_CLOSURE",
            )
            entry.update(
                closure_max_mm=closure,
                actual_minus_command_rms_mm=rms(absolute),
                fk_minus_reference_rms_mm=rms(upstream),
                fk_minus_actual_rms_mm=rms(servo),
                world_base_only_rms_mm=rms(base),
                arm_relative_in_world_rms_mm=rms(inc - base),
                actual_minus_ref_at_0_100_1000_ms_mm=xyz(absolute[[0, 50, 500]]),
                fk_minus_actual_end_mm=xyz(servo[-1]),
            )
        out.append(entry)
    return out


def logdeg(r):
    return Rotation.from_matrix(r).as_rotvec(degrees=True)


def feature(p, r):
    return np.c_[(p - p[0]) * 1000, logdeg(r @ r[0].T)]


def d3(root, reference):
    data = []
    for r in reference["records"]:
        f = root / "v15/d3-verified01" / (r["id"] + "_curve.private.npz")
        pinned(f, r["private_curve_sha256"])
        with np.load(f, allow_pickle=False) as a:
            p, g, q, b = [
                a[k]
                for k in (
                    "actual_xyz_m",
                    "reference_xyz_m",
                    "pd_fk_xyz_m",
                    "base_xyz_m",
                )
            ]
            R, G, B = [
                a[k] for k in ("actual_rotation", "reference_rotation", "base_rotation")
            ]
            require(np.array_equal(a["elapsed_ms"], np.arange(1001) * 2), "D3_GRID")
        absolute = (p - g) * 1000
        low = (q - p) * 1000
        upstream = (q - g) * 1000
        body_reference = b + np.einsum("nij,j->ni", B, B[0].T @ (p[0] - b[0]))
        err = float(np.max(np.abs(absolute - upstream + low)))
        require(
            err < 1e-8 and abs(rms(absolute[1:]) - r["abs_position_RMS_mm"]) < 1e-7,
            "D3_CLOSURE",
        )
        summary = {
            "id": r["id"],
            "B2_match": r["B2_anchor_matched"],
            "family": r["family"],
            "rate": r["rate"],
            "closure_max_mm": err,
            "fk_minus_ref_rms_mm": rms(upstream),
            "fk_minus_actual_rms_mm": rms(low),
            "actual_minus_ref_rms_mm": rms(absolute[1:]),
            "world_base_only_rms_mm": rms((body_reference - p[0]) * 1000),
            "arm_relative_in_world_rms_mm": rms((p - body_reference) * 1000),
            "end_actual_minus_ref_mm": xyz(absolute[-1]),
            "end_fk_minus_actual_mm": xyz(low[-1]),
        }
        u = feature(g, G)
        y = feature(p, R)
        require(np.max(np.abs(u[:, [1, 3, 5]])) < 1e-4, "D3_UNEXPECTED_EXCITATION")
        data.append(
            {
                "summary": summary,
                "u": u[:, [0, 2, 4]],
                "y": y,
                "actual": p[0],
                "rot": R[0],
                "old": g[0],
                "old_R": G[0],
            }
        )
    return data


WEIGHT = np.array([1.0] * 3 + [57 * np.pi / 180] * 3)


def shifted(v, lag):
    return v[np.maximum(0, np.arange(len(v)) - lag)]


def fit(data):
    # All cards have the same 1001 time points. Lag is selected only on fitting cards.
    y = np.concatenate([d["y"] for d in data])
    best = None
    for lag in range(0, 151, 5):
        u = np.concatenate([shifted(d["u"], lag) for d in data])
        coef, _, rank, sv = np.linalg.lstsq(u, y, rcond=None)
        require(rank == 3, "UNIDENTIFIABLE_TRAINING_FOLD")
        score = rms((y - u @ coef) * WEIGHT)
        if best is None or score < best[0]:
            best = (score, lag, coef, sv)
    return best


def model_readout(data, test, label):
    _, lag, coef, sv = fit(data)
    errors = []
    for d in test:
        res = d["y"] - shifted(d["u"], lag) @ coef
        errors.append(
            {
                "id": d["summary"]["id"],
                "rms_mm": rms(res[:, :3]),
                "rms_deg": rms(res[:, 3:]),
            }
        )
    return {
        "fold": label,
        "train": [d["summary"]["id"] for d in data],
        "lag_ms": 2 * lag,
        "singular_values": xyz(sv),
        "matrix_output_by_input": xyz(coef.T),
        "test": errors,
    }


def transform(p, R):
    a = np.eye(4)
    a[:3, 3] = p
    a[:3, :3] = R
    return a


def shadow(d, config, setting, mode, fit_result, hull):
    _, lag, coef, _ = fit_result
    matrix = np.zeros((6, 6))
    matrix[[0, 2, 4]] = coef
    for j in [1, 3, 5]:
        matrix[j, j] = setting
    old = transform(d["old"], d["old_R"])
    actual = transform(d["actual"], d["rot"])
    c = config["controller"]
    g = config["geometry"]
    mgr = ReferenceManager("model", old, c)
    mon = StopMonitor()
    u = np.zeros((1 + 500 * c["timeout_s"], 6))
    # These receipts describe the model execution only. They carry no native ACK authority.
    wire = "model_initial"
    anchor = old.copy()
    previous_atom = old.copy()

    def receipt(tick, source):
        return {
            "episode": "model",
            "start": tick - 25,
            "end": tick,
            "source": source,
            "accepted": wire,
            "installed": wire,
            "consumed": wire,
            "solver_echo": True,
            "end_pose": anchor,
            "start_pose": previous_atom,
            "finger_owner": "model",
            "finger_fraction": OPEN,
        }

    def measurement(tick):
        return {
            "episode": "model",
            "safe": True,
            "contact": False,
            "TCP": actual,
            "tick": tick,
        }

    residual_pos = []
    residual_deg = []
    incremental = []
    outside = None
    box_exit = None
    subspace_exit = None
    stop = "TIMEOUT"
    t = 0

    def observe(ix):
        dp = 1000 * (d["old"] - actual[:3, 3])
        dr = logdeg(d["old_R"] @ actual[:3, :3].T)
        incremental.append(
            [
                float(1000 * np.linalg.norm(actual[:3, 3] - d["actual"])),
                float(np.linalg.norm(logdeg(actual[:3, :3] @ d["rot"].T))),
            ]
        )
        residual_pos.append(float(np.linalg.norm(dp)))
        residual_deg.append(float(np.linalg.norm(dr)))
        return mon.observe(
            ix * 2, dp, dr, g["abs_position_bound_mm"], g["rotation_bound_deg"]
        )

    status = observe(0)
    rec = receipt(START, START - 50)
    lo, hi, equations = hull
    start_error = xyz([residual_pos[0], residual_deg[0]])
    while t < 500 * c["timeout_s"]:
        tick = START + t
        try:
            knots = mgr.propose(
                "A0" if status == "HOLD" else mode,
                tick,
                wire,
                rec,
                measurement(tick),
                old,
            )
        except ValueError as exc:
            stop = str(exc)
            break
        end = knots[2]
        a0 = np.r_[
            (anchor[:3, 3] - d["old"]) * 1000, logdeg(anchor[:3, :3] @ d["old_R"].T)
        ]
        f = np.arange(1, 51) / 50
        rv = logdeg(end[:3, :3] @ anchor[:3, :3].T)
        rotations = (
            Rotation.from_rotvec(f[:, None] * rv, degrees=True).as_matrix()
            @ anchor[:3, :3]
        )
        u[t + 1 : t + 51, :3] = (
            a0[:3] + f[:, None] * (end[:3, 3] - anchor[:3, 3]) * 1000
        )
        u[t + 1 : t + 51, 3:] = logdeg(rotations @ d["old_R"].T)
        idx = np.arange(t + 1, t + 51)
        delta = u[np.maximum(0, idx - lag)] @ matrix
        poses = d["actual"] + delta[:, :3] / 1000
        rots = Rotation.from_rotvec(delta[:, 3:], degrees=True).as_matrix() @ d["rot"]
        for i, k in enumerate(idx):
            t = int(k)
            seen = u[k, [0, 2, 4]]
            if (
                outside is None
                and np.max(equations[:, :3] @ seen + equations[:, 3]) > 1e-5
            ):
                outside = 2 * t
            if box_exit is None and ((seen < lo - 1e-5) | (seen > hi + 1e-5)).any():
                box_exit = 2 * t
            if subspace_exit is None and np.max(np.abs(u[k, [1, 3, 5]])) > 1e-5:
                subspace_exit = 2 * t
            actual = transform(poses[i], rots[i])
            status = observe(t)
            if status not in ("HOLD", "CONTINUE"):
                stop = status
                break
        if status not in ("HOLD", "CONTINUE"):
            break
        previous_atom = knots[1]
        anchor = end
        wire = "model_" + str(t)
        rec = receipt(START + t, tick)
        mgr.consumed(START + t, wire, knots, rec, measurement(START + t))
    return {
        "id": d["summary"]["id"],
        "mode": mode,
        "unexcited_diagonal_assumption": setting,
        "outcome": stop,
        "elapsed_ms": 2 * t,
        "start_mm_deg": start_error,
        "incremental_rms_mm_deg": np.sqrt(
            np.mean(np.square(incremental), axis=0)
        ).tolist(),
        "end_mm_deg": [residual_pos[-1], residual_deg[-1]],
        "absolute_rms_mm_deg": [
            float(np.sqrt(np.mean(np.square(v)))) for v in (residual_pos, residual_deg)
        ],
        "first_observed_subspace_exit_ms": subspace_exit,
        "first_training_hull_exit_projected_ms": outside,
        "first_training_box_exit_projected_ms": box_exit,
        "max_command_excursion_mm": float(np.linalg.norm(u[: t + 1, :3], axis=1).max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("registration", type=Path)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()
    cfg = read(a.registration)
    root = a.root
    p1 = root / "v15/matched-verified01/D1_VERIFIED.json"
    p3 = root / "v15/d3-verified01/D3_VERIFIED.json"
    pinned(p1, cfg["source_D1_sha256"])
    pinned(p3, cfg["source_D3_sha256"])
    static = d1(root, read(p1))
    data = d3(root, read(p3))
    folds = []
    models = []
    for key in ("family", "rate"):
        for value in sorted({d["summary"][key] for d in data}):
            test = [d for d in data if d["summary"][key] == value]
            train = [d for d in data if d["summary"][key] != value]
            folds.append(model_readout(train, test, key + "_" + value))
    for d in data:
        others = [k for k in data if k is not d]
        m = fit(others)
        u = np.concatenate([k["u"] for k in others])
        hull = (np.min(u, axis=0), np.max(u, axis=0), ConvexHull(u).equations)
        models.append(shadow(d, cfg, 0, "A0", m, hull))
        for setting in [0, 0.5, 1]:
            models.append(shadow(d, cfg, setting, "A1", m, hull))
    result = {
        "scope": "OFF_POLICY_MODEL_ONLY",
        "registration_sha256": sha(a.registration),
        "d1": static,
        "d3": [d["summary"] for d in data],
        "held_out_fits": folds,
        "shadow": models,
        "control_sha256": sha(Path(__file__).with_name("feedback.py")),
        "analysis_sha256": sha(Path(__file__)),
        "historical_d3_qualified_mode_pairs": sum(
            x["qualified"] for x in read(p3)["pairs"]
        ),
        "new_native_requests": 0,
        "new_physics_ticks": 0,
        "new_gpu_forwards": 0,
        "new_training_updates": 0,
    }
    with a.output.open("x") as f:
        json.dump(result, f, allow_nan=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
