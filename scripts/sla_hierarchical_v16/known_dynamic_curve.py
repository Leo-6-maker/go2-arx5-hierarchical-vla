"""Original gate and offline scene witnesses for the next known A0/A1 dynamic comparison."""

import json
import sys
from pathlib import Path
import numpy as np
from sealed_panel import configure_site


def trajectory(origin, t, linear, angular):
    from scipy.spatial.transform import Rotation

    u = np.clip((np.asarray(t) - 1) / 4, 0, 1)
    phase = 2 * np.pi * (u**3 * (10 + u * (-15 + 6 * u)))
    max_rate = 2 * np.pi * 1.875 / 4
    out = np.repeat(np.asarray(origin)[None], len(t), axis=0)
    out[:, 0, 3] += linear / max_rate * np.sin(phase)
    out[:, 2, 3] += linear / max_rate * (1 - np.cos(phase))
    out[:, :3, :3] = (
        Rotation.from_rotvec(
            np.outer(angular / max_rate * np.sin(phase), [0.0, 1.0, 0.0])
        ).as_matrix()
        @ origin[:3, :3]
    )
    return out


def run(root, repo):
    import mujoco
    from audit_sealed import pin, read, digest
    from d1_geometry import sweep
    from go2arx5_vla.world_model.native_replay import forbid_integration
    from go2arx5_vla.world_model.mobile_wbc_reference_v1 import full_eef_gate
    from go2arx5_vla.world_model.mobile_safety_profile_v1 import binding
    from scripts.pi0291_arx5_data.native_dataset import pose10

    F = root / "feedback/F1_VERIFIED.json"
    pin(
        F,
        read(repo / "reports/SLA_HIERARCHICAL_V16/V0_EVIDENCE_MATRIX_ACTUAL.json")[
            "source_hashes"
        ]["F1"],
    )
    official = {r["card"]: r for r in read(F)["records"]}
    rows = []
    for card in ("F1-01", "F1-02"):
        case = root / "feedback/sla-feedback-f1-a0-20261123-01" / card
        ip = case / "INITIAL.private.json"
        rp = case / "RESULT.private.json"
        pin(ip, official[card]["initial_sha256"])
        pin(rp, official[card]["result_sha256"])
        sp = case / "EXECUTION_SPEC.json"
        pin(sp, read(case / "control/LEASE.json")["spec_sha256"])
        spec = read(sp)
        scene = Path(spec["model_path"])
        pin(scene, spec["frozen_files"][str(scene)])
        model = mujoco.MjModel.from_binary_path(str(scene))
        guard = binding(spec["native_root"])
        feat = read(ip)["features"]
        parts = {
            k: np.asarray(feat[k])
            for k in ("old_ee", "old_base", "measured_base", "measured_ee")
        }
        log = case / read(rp)["episode"] / "MOBILE_SAMPLES.private.jsonl"
        with log.open() as f:
            q = next(r for l in f if (r := json.loads(l))["native_end_step"] == 5703)
        opened = feat["gripper_command"][0]
        if abs(opened - q["installed_EEF10"][9]) > 1e-12:
            raise ValueError("previous consumed open reference")
        for mode, linear, angular in (
            ("SLOW", 0.004, np.deg2rad(0.6)),
            ("SOURCE_SPEED", 0.01125, 0.045),
        ):
            poses = trajectory(parts["old_ee"], np.arange(181) * 0.05, linear, angular)
            wire = pose10(poses, np.full(len(poses), opened))
            metrics = {}
            reject = []
            for seq in range(71):
                a = wire[seq * 2 : seq * 2 + 40]
                try:
                    m = full_eef_gate(a, parts["measured_ee"])
                    for k, v in m.items():
                        if k.startswith("maximum_"):
                            metrics[k] = max(float(v), metrics.get(k, 0))
                except ValueError as e:
                    reject.append(dict(sequence=seq, error=str(e)))
            previous = parts["old_ee"]
            samples = 0
            minimum = None
            failure = None
            with forbid_integration(mujoco) as calls:
                for i, dest in enumerate(poses[:101]):
                    if i and np.array_equal(dest, previous):
                        continue
                    try:
                        r = sweep(
                            model,
                            q,
                            dict(parts, old_ee=previous),
                            dest,
                            parts["old_base"],
                            guard,
                        )
                    except (ValueError, AssertionError) as e:
                        failure = dict(index=i, elapsed_s=i * 0.05, reason=str(e))
                        break
                    samples += r["poses_checked"]
                    minimum = (
                        r["minimum_joint_margin_rad"]
                        if minimum is None
                        else min(minimum, r["minimum_joint_margin_rad"])
                    )
                    previous = dest
            if calls:
                raise ValueError("offline witness entered dynamics")
            rows.append(
                dict(
                    card=card,
                    mode=mode,
                    compiled_scene_sha256=digest(scene),
                    gate_previews=71,
                    rejections=reject,
                    maximum_fields=metrics,
                    g7_seam_mm_s=0,
                    geometry_checks=samples,
                    minimum_joint_margin_rad=minimum,
                    first_geometry_failure=failure,
                )
            )
    return dict(
        cases=rows,
        physics_integrations=0,
        new_native_requests=0,
        native_dynamic_qualification=False,
    )


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parents[2]
    configure_site(root, repo)
    print(json.dumps(run(root, repo), indent=2, allow_nan=False))
