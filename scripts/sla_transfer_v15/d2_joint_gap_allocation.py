"""Allocate recorded TCP displacement across six arm-joint substitutions by exact order averaging."""

import argparse
import math
from pathlib import Path

from d1_readiness import B2_PIN, baseline, configure, pinned, read, record, sha


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("root", type=Path)
    cli.add_argument("out", type=Path)
    args = cli.parse_args()
    root = args.root.resolve()
    configure(baseline(root))
    import mujoco
    import numpy as np
    from go2arx5_vla.world_model.native_replay import forbid_integration, packets

    base = root / "v14/b2-technical-20261123-01"
    m = base / "B2_PUBLIC_READOUT.json"
    pinned(m, B2_PIN)
    prior = root / "v15/readout01/D2_LOW_LEVEL_TRACKING_AND_POLICY_AGE.json"
    pinned(prior, "1279b591f45208427542cbd94fcaf311fb5d98f49f983a273618d52b6c6f4e40")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    masks = np.array([[bool(i & (1 << j)) for j in range(6)] for i in range(64)])
    coefficients = np.zeros((6, 64))
    for j in range(6):
        for subset in range(64):
            if not masks[subset, j]:
                k = int(np.sum(masks[subset]))
                weight = 1 / (6 * math.comb(5, k))
                coefficients[j, subset | 1 << j] += weight
                coefficients[j, subset] -= weight
    totals = {}
    for info, prev in zip(
        read(m)["cards"], read(prior)["historical_B2_cases"], strict=True
    ):
        name = info["card"]
        path = base / name
        pinned(path / "EXECUTION_SPEC.json", info["spec_sha256"])
        cfg = read(path / "EXECUTION_SPEC.json")
        model_path = Path(cfg["model_path"])
        pinned(model_path, prev["model_sha256"])
        curve_file = prior.parent / (name + "_curve.private.json")
        pinned(curve_file, prev["private_curve_sha256"])
        native = path / read(path / "RESULT.private.json")["episode"]
        state = native / "state.bin.journal"
        pinned(state, info["evidence_sha256"][state.name])
        curve = read(curve_file)
        pack = packets(state)
        values = []
        err = 0
        with forbid_integration(mujoco) as calls:
            model = mujoco.MjModel.from_binary_path(str(model_path))
            data = mujoco.MjData(model)
            site = model.site("x5_ee").id
            qa = model.jnt_qposadr[[model.joint(j).id for j in prev["joints"]]]
            for c in curve:
                st = pack[c["tick"] - 1]
                data.qpos[:] = st["qpos"]
                q = st["qpos"][qa]
                qcmd = np.asarray(c["desired_q_rad"])
                if not np.array_equal(q, c["actual_q_rad"]) or st["step"] != c["tick"]:
                    raise ValueError("state/curve join")
                positions = []
                for variant in np.where(masks, qcmd, q):
                    data.qpos[qa] = variant
                    mujoco.mj_kinematics(model, data)
                    positions.append(data.site_xpos[site].copy())
                xyz = coefficients @ (np.array(positions) * 1000.0)
                native_gap = np.array(c["pd_fk_minus_actual_mm"])
                err = max(err, float(np.max(abs(xyz.sum(axis=0) - native_gap))))
                values.append(xyz)
        if calls or err > 1e-7:
            raise ValueError("integration or additive closure failure")
        values = np.asarray(values)
        private = out / (name + "_xyz.private.json")
        record(private, values.tolist())
        totals[name] = {
            "joints": prev["joints"],
            "native_test_ticks": len(curve) - 1,
            "states_including_boundary": len(curve),
            "max_additive_closure_error_mm": err,
            "mean_test_xyz_mm_by_joint": np.mean(values[1:], axis=0).tolist(),
            "z_mm_by_joint_at_elapsed_ms": {
                str(ms): values[ms // 2, :, 2].tolist()
                for ms in (0, 100, 250, 500, 1000)
                if ms // 2 < len(values)
            },
            "private_curve_sha256": sha(private),
            "native_telemetry_sha256": sha(state),
            "source_curve_sha256": sha(curve_file),
        }
    record(
        out / "D2_JOINT_GAP_ALLOCATION_ACTUAL.json",
        {
            "cases": totals,
            "scope": "instantaneous kinematic accounting, averaged across all 6! orders at fixed actual base and all other actual joints",
            "units": "signed millimetres",
            "new_native_requests": 0,
            "new_ticks": 0,
            "new_model_forwards": 0,
            "new_updates": 0,
        },
    )


if __name__ == "__main__":
    main()
