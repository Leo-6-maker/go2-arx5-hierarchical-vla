"""Check the known path through the existing mobile converter and actual offline receiver."""

import argparse
from pathlib import Path

from d1_readiness import B2_PIN, baseline, configure, pinned, read, record


def replay(folder, anchor, tcp, base, family, rate, mode):
    import hashlib

    import numpy as np
    from clock_replay import GuardedManager, episode
    from d1_factorial_reference_matrix import check
    from d3_known_path import KnownPath
    from go2arx5_vla.world_model.fusion_trajectory_adapter import WORKING_OPEN_FRACTION
    from go2arx5_vla.world_model.mobile_wbc_reference_v1 import document, full_eef_gate
    from shadow import check as check_message
    from shadow import message, observe

    from go2arx5_vla.trajectory.so3 import quat_to_matrix
    from scripts.pi0291_arx5_data.native_dataset import pose10

    path = KnownPath(tcp, family, rate, mode)
    folder.mkdir()
    fields = hashlib.sha256()
    metrics = {}
    base = np.repeat(base[None], 41, axis=0)
    maximum_field_error = 0.0
    with episode(folder.name, folder):
        receiver = GuardedManager()
        for i in range(20):
            now = 11.406 + i * 0.1
            poses = path.window(2 * i)
            action = pose10(poses[:40], np.full(40, WORKING_OPEN_FRACTION))
            full40 = full_eef_gate(action, anchor)
            for k, v in full40.items():
                if k.startswith("maximum_"):
                    metrics[k] = max(metrics.get(k, 0), float(v))
            doc = document(action, mode="M_ASSIST", base_poses=base)
            doc["_pi0294"] = {
                "episode": folder.name,
                "sequence": i,
                "source_mpc_s": now,
                "ttl_s": 0.4,
                "operation": "install",
            }
            # Source pose and receiver state are separate. This CPU check has no moving body.
            observe(receiver, now)
            receiver.on_trajectory_json(message(doc))
            msg = receiver.target_pub[-1]
            val = check_message(msg, 14)
            check(
                len(val) == 41
                and np.allclose(
                    msg.time_trajectory, now + np.arange(41) * 0.05, rtol=0, atol=1e-8
                )
                and np.array_equal(val[:, :3], poses[:, :3, 3].astype(np.float32))
                and np.array_equal(val[:, 7:10], base[:, :3, 3].astype(np.float32))
                and np.allclose(
                    quat_to_matrix(val[:, 3:7]), poses[:, :3, :3], rtol=0, atol=1e-7
                )
                and np.allclose(
                    quat_to_matrix(val[:, 10:14]), base[:, :3, :3], rtol=0, atol=1e-7
                ),
                "known 41-point field",
            )
            maximum_field_error = max(
                maximum_field_error, float(np.max(abs(val[:, :3] - poses[:, :3, 3])))
            )
            for a in (
                msg.time_trajectory,
                val,
                [list(k.value) for k in msg.input_trajectory],
            ):
                fields.update(np.asarray(a, dtype="<f8").tobytes())
        check(
            not receiver._r4_mode and not receiver.clock_fault, "unexpected clock/hold"
        )
    return {
        "fields_sha256": fields.hexdigest(),
        "installs": 20,
        "revisions": path.revisions,
        "max_field_error_m": maximum_field_error,
        "full40_maxima": metrics,
    }


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("root", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    root = args.root.resolve()
    configure(baseline(root))
    import mujoco
    import numpy as np
    from d1_factorial_reference_matrix import check
    from d1_geometry import sweep
    from d3_known_path import DURATIONS, FAMILIES, poses_at
    from go2arx5_vla.world_model.mobile_safety_profile_v1 import binding
    from go2arx5_vla.world_model.native_replay import forbid_integration, packets

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    b2 = root / "v14/b2-technical-20261123-01"
    manifest = b2 / "B2_PUBLIC_READOUT.json"
    pinned(manifest, B2_PIN)
    ready = root / "v15/readiness01"
    reg = ready / "D1_FACTORIAL_4X2_REGISTER.json"
    pinned(reg, "c69cea0c14cf4390d369a5fec9efc44ec29ecfe273df44326f2f39fffb42578a")
    records = []
    for n, card in enumerate(read(manifest)["cards"]):
        case = b2 / card["card"]
        cfg_path = case / "EXECUTION_SPEC.json"
        pinned(cfg_path, card["spec_sha256"])
        cfg = read(cfg_path)
        journal = (
            case / read(case / "RESULT.private.json")["episode"] / "state.bin.journal"
        )
        pinned(journal, card["evidence_sha256"]["state.bin.journal"])
        packet = packets(journal)[5702]
        model_path = Path(cfg["model_path"])
        pinned(model_path, cfg["frozen_files"][str(model_path)])
        model = mujoco.MjModel.from_binary_path(str(model_path))
        a = ready / (card["card"] + "_anchor.private.json")
        pinned(a, read(reg)["prior_anchor_sha256"][n])
        parts = {k: np.asarray(v) for k, v in read(a)["parts"].items()}
        tcp, base = parts["old_ee"], parts["old_base"]
        guard = binding(cfg["native_root"])
        for family in FAMILIES:
            for rate in DURATIONS:
                name = card["card"] + "-" + family + "-" + rate
                geometry = []
                # Include the true joint state and all consecutive segments of the 50 ms reference.
                with forbid_integration(mujoco) as calls:
                    waypoints = poses_at(tcp, family, rate, np.arange(41) * 0.05)
                    old = tcp
                    for pose in waypoints:
                        if geometry and np.array_equal(old, pose):
                            continue
                        geometry.append(
                            sweep(
                                model,
                                packet,
                                dict(parts, old_ee=old),
                                pose,
                                base,
                                guard,
                            )
                        )
                        old = pose
                check(not calls, "unexpected integration")
                modes = [
                    replay(
                        out / (name + "-" + mode),
                        parts["measured_ee"],
                        tcp,
                        base,
                        family,
                        rate,
                        mode,
                    )
                    for mode in ("ONE_SHOT_INTENT", "P10_REVISION")
                ]
                check(
                    modes[0]["fields_sha256"] == modes[1]["fields_sha256"]
                    and [m["revisions"] for m in modes] == [0, 3],
                    "one-shot / P10 inconsistency",
                )
                records.append(
                    {
                        "anchor": card["card"],
                        "family": family,
                        "rate": rate,
                        "duration_s": DURATIONS[rate],
                        "modes": modes,
                        "swept_poses": sum(g["poses_checked"] for g in geometry),
                        "minimum_joint_margin_rad": min(
                            g["minimum_joint_margin_rad"] for g in geometry
                        ),
                        "max_IK_component_residual": max(
                            g["maximum_IK_component_residual"] for g in geometry
                        ),
                    }
                )
    record(
        out / "D3_KNOWN_PATH_PREVIEW.json",
        {
            "records": records,
            "scope": "Offline design check at two historical B2 anchors; no dynamic accuracy measurement",
            "native_requests": 0,
            "physics_ticks": 0,
            "SLA_forwards": 0,
            "optimizer_updates": 0,
        },
    )


if __name__ == "__main__":
    main()
