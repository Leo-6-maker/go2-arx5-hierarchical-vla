"""Audit continuation from the last truly consumed reference on every settled A1 control."""

import json
import sys
from pathlib import Path
import numpy as np
from sealed_panel import configure_site
from known_dynamic_curve import trajectory
from dynamic_margin import clearances


def last(path):
    row = None
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
    if row is None:
        raise ValueError("missing receipt")
    return row


def delta(a, b):
    from scipy.spatial.transform import Rotation

    return dict(
        mm=float(1000 * np.linalg.norm(a[:3, 3] - b[:3, 3])),
        deg=float(
            np.degrees(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude())
        ),
    )


def run(root, repo):
    import mujoco
    from audit_sealed import read, pin
    from scripts.pi0291_arx5_data.native_dataset import matrix_from10, pose10
    from go2arx5_vla.world_model.mobile_wbc_reference_v1 import full_eef_gate

    folder = repo / "reports/SLA_HIERARCHICAL_V16"
    v0 = read(folder / "V0_EVIDENCE_MATRIX_ACTUAL.json")
    pub = root / "feedback/F2_VERIFIED.json"
    pin(pub, v0["source_hashes"]["F2"])
    roster = read(pub)
    d = read(folder / "V3_DYNAMIC_MARGIN_ACTUAL.json")
    raw = {r["card"]: r for r in d["records"]}
    out = []
    for rr in roster["records"]:
        if not rr["eligible"]:
            continue
        card = rr["card"]
        f = root / "feedback/sla-feedback-f2-a1-20261123-01" / card
        result = f / "RESULT.private.json"
        pin(result, rr["result_sha256"])
        r = read(result)
        robot = f / r["episode"]
        trace = f / "TRACK.private.npz"
        ledger = f / "SERVO.private.jsonl"
        pin(trace, rr["curve_sha256"])
        pin(ledger, rr["servo_ledger_sha256"])
        with np.load(trace, allow_pickle=False) as z:
            goal = z["goal"][-1]
            actual = z["actual"][-1]
        final = last(ledger)
        atom = last(robot / "MOBILE_SAMPLES.private.jsonl")
        started = np.asarray(atom["installed_EEF10"])
        ended = np.asarray(atom["target_endpoint_EEF10"])
        wire = atom["installed_reference_sha256"]
        if (
            r["outcome"] != "HOLD_COMPLETE"
            or atom["native_end_step"] != final["end"]
            or final["end"] != r["ticks"]
            or not final["solver_echoes"]
            or not atom["solver_exact_installed_targets"]
            or final["wire"] != wire
            or not np.allclose(final["actual_endpoint"], actual, atol=2e-7, rtol=0)
            or not np.allclose(atom["actual_TCP"], actual, atol=2e-7, rtol=0)
            or started[9] != ended[9]
        ):
            raise ValueError("unjoined actual settled boundary")
        anchor = matrix_from10(ended)
        sc = f / "EXECUTION_SPEC.json"
        pin(sc, read(f / "control/LEASE.json")["spec_sha256"])
        spec = read(sc)
        scene = Path(spec["model_path"])
        pin(scene, spec["frozen_files"][str(scene)])
        model = mujoco.MjModel.from_binary_path(str(scene))
        q = np.asarray(atom["qpos"])
        lo = raw[card]["slide_min_mm"]
        hi = raw[card]["slide_max_mm"]
        qmm = q[25:27] * 1000
        slide = np.maximum(abs(qmm - lo), abs(qmm - hi))
        modes = {}
        for method, origin in (
            ("RESET_NOMINAL", goal),
            ("CONTINUE_CONSUMED_OFFSET", anchor),
        ):
            by_rate = {}
            for k, lin, ang in (
                ("SLOW", 0.004, np.deg2rad(0.6)),
                ("SOURCE_SPEED", 0.01125, 0.045),
            ):
                poses = trajectory(origin, np.arange(181) * 0.05, lin, ang)
                actions = pose10(poses, np.full(len(poses), ended[9]))

                def gate(a):
                    try:
                        full_eef_gate(a, actual)
                    except ValueError as exc:
                        return str(exc)
                    return None

                join = delta(anchor, poses[0])
                seam = gate(np.vstack((started, ended, actions[:38])))
                fail = [
                    i
                    for i in range(71)
                    if gate(actions[i * 2 : i * 2 + 40]) is not None
                ]
                geo = clearances(model, q, origin, (lin, ang), slide)
                by_rate[k] = dict(
                    first_knot_gap=join,
                    optimistic_one_tick_seam_rejection=seam,
                    hypothetical_join_speed_m_s=join["mm"] / 0.05 / 1000,
                    all_previews=71,
                    rejected_preview_indices=fail,
                    minimum_pad_clearance_mm=min(v["net_lower_bound_mm"] for v in geo),
                    pair_clearances=geo,
                    fixed_open_g7_m_s=abs(float(ended[9] - actions[0, 9]))
                    * 0.044
                    / 0.05,
                )
            modes[method] = by_rate
        out.append(
            dict(
                card=card,
                consumed_wire_sha256=wire,
                end_tick=r["ticks"],
                elapsed_to_dwell_start_s=(r["first_qualified_hold_tick"] - 5703)
                * 0.002,
                measured_to_goal=delta(actual, goal),
                command_to_goal=delta(anchor, goal),
                methods=modes,
            )
        )
    if len(out) != 4:
        raise ValueError("incomplete settled population")
    return dict(
        cards=out,
        F2_started=roster["started"],
        F2_screened=roster["screened"],
        settled_other_stratum=0,
        new_native_requests=0,
        new_physics_ticks=0,
        actual_dynamic_trials=0,
    )


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parents[2]
    configure_site(root, repo)
    print(json.dumps(run(root, repo), indent=2, allow_nan=False))
