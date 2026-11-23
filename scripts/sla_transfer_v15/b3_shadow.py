"""Exercise long frozen source intentions through the actual, offline OCS2 receiver callbacks."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from d1_readiness import baseline, configure, pinned, read, record, sha


def mode_run(folder, intent, base, mode):
    import numpy as np
    from clock_replay import GuardedManager, episode
    from d1_factorial_reference_matrix import check
    from ocs2_target_adapter import wire
    from reference_manager_v2 import ReferenceManagerV2
    from shadow import check as check_message
    from shadow import observe
    from std_msgs.msg import String

    from go2arx5_vla.trajectory.so3 import quat_slerp, quat_to_matrix

    folder.mkdir()
    clock = 30.0
    count = math.ceil(intent.execution_times[-1] / 0.1 - 1e-10) + 1
    fields_hash = hashlib.sha256()
    widths = {}
    max_error = 0.0
    rolling_error = 0.0
    last_phase = 0.0
    cursor = 0
    full40_rejected = 0
    from go2arx5_vla.world_model import fusion_trajectory_adapter as limits
    from go2arx5_vla.world_model.mobile_wbc_reference_v1 import _geometry, full_eef_gate

    envelope = {
        "maximum_translation_speed_m_s": limits.MAX_TRANSLATION_SPEED_M_S,
        "maximum_angular_speed_rad_s": limits.MAX_ANGULAR_SPEED_RAD_S,
        "maximum_finger_speed_m_s": limits.MAX_FINGER_SPEED_M_S,
        "maximum_discrete_translation_acceleration_m_s2": limits.MAX_TRANSLATION_ACCELERATION_M_S2,
        "maximum_discrete_angular_acceleration_rad_s2": limits.MAX_ANGULAR_ACCELERATION_RAD_S2,
    }
    violations = []

    from scripts.pi0291_arx5_data.native_dataset import pose10

    manager = ReferenceManagerV2(intent, folder.name, clock)
    with episode(folder.name, folder):
        site = GuardedManager()
        for i in range(count):
            t = clock + i * 0.1
            if mode == "P10" and i and i % 5 == 0:
                manager.revise(intent, t, folder.name)
            key, (poses, grip, phase) = manager.window(t)
            check(phase[0] >= last_phase - 1e-8, "source phase rewind")
            last_phase = phase[0]
            action = pose10(poses[:40], grip[:40])
            try:
                full_eef_gate(action, poses[0])
            except ValueError as error:
                values = _geometry(action, poses[0])
                failures = {
                    k: {"observed": float(values[k]), "limit": float(limit)}
                    for k, limit in envelope.items()
                    if values[k] > limit * (1 + 1e-10)
                }
                check(
                    str(error) == "MOBILE_ORIGINAL_FULL40_EEF_GATE" and failures,
                    "unexplained envelope rejection",
                )
                violations.append({"sequence": i, "failures": failures})
                full40_rejected += 1
            raw = wire(poses, base, folder.name, key[1], t)
            ident = hashlib.sha256(raw.encode()).hexdigest()
            observe(site, t)
            site.on_trajectory_json(String(data=raw))
            msg = site.target_pub[-1]
            values = check_message(msg, 14)
            check(
                len(values) == 41
                and np.allclose(
                    msg.time_trajectory, t + np.arange(41) * 0.05, atol=1e-8, rtol=0
                ),
                "fresh time grid",
            )
            max_error = max(
                max_error,
                float(np.max(abs(values[:, :3] - poses[:, :3, 3]))),
                float(np.max(abs(values[:, 7:10] - base[:, :3, 3]))),
            )
            check(
                np.allclose(
                    quat_to_matrix(values[:, 3:7]), poses[:, :3, :3], atol=7e-7, rtol=0
                )
                and np.allclose(
                    quat_to_matrix(values[:, 10:14]), base[:, :3, :3], atol=7e-7, rtol=0
                ),
                "rotation ABI",
            )
            with (folder / "CALLBACKS.private.jsonl").open() as f:
                f.seek(cursor)
                ev = [json.loads(s) for s in f]
                cursor = f.tell()
            a = next(r for r in ev if r["kind"] == "accept" and r["ref_id"] == ident)
            emitted = next(
                r for r in ev if r["kind"] == "installed" and r["ref_id"] == ident
            )
            manager.acknowledge(key, ident, a, emitted)
            for extra in (False, True):
                if extra:
                    observe(site, t + 0.09)
                    site.publish()
                m = site.target_pub[-1]
                v = check_message(m, 14)
                if extra:
                    age = np.asarray(m.time_trajectory) - t
                    check(
                        len(v) == 80
                        and np.isclose(age[0], 0.09)
                        and np.isclose(age[-1], 2.09),
                        "rolling preview boundaries",
                    )
                    phase_index = np.clip(age / 0.05, 0.0, 40.0)
                    idx = np.minimum(np.floor(phase_index).astype(int), 39)
                    alpha = phase_index - idx
                    for pos, rot in (
                        (slice(0, 3), slice(3, 7)),
                        (slice(7, 10), slice(10, 14)),
                    ):
                        expected = (1 - alpha[:, None]) * values[idx, pos] + alpha[
                            :, None
                        ] * values[idx + 1, pos]
                        rolling_error = max(
                            rolling_error, float(np.max(abs(v[:, pos] - expected)))
                        )
                        orientation = quat_slerp(
                            values[idx, rot], values[idx + 1, rot], alpha
                        )
                        check(
                            np.allclose(
                                quat_to_matrix(orientation),
                                quat_to_matrix(v[:, rot]),
                                rtol=0,
                                atol=1e-6,
                            ),
                            "rolling rotation fields",
                        )
                for array in (
                    m.time_trajectory,
                    v,
                    [list(k.value) for k in m.input_trajectory],
                ):
                    fields_hash.update(np.asarray(array, dtype="<f8").tobytes())
                widths[len(v)] = widths.get(len(v), 0) + 1
            site.target_pub.clear()
        check(
            not site._r4_mode
            and manager.consumed is None
            and abs(last_phase - 49 / 30) < 1e-7
            and rolling_error < 2e-7
            and max_error < 2e-7,
            "incomplete shadow",
        )
        cancel = {
            "_pi0294": {
                "episode": folder.name,
                "sequence": count,
                "source_mpc_s": t + 0.09,
                "ttl_s": 0.4,
                "operation": "cancel",
            }
        }
        site.on_trajectory_json(String(data=json.dumps(cancel)))
        check([m.data for m in site._r4_mode] == ["HOLD"], "terminal HOLD")
    evidence = folder / "FULL40_ENVELOPE.private.json"
    record(evidence, violations)
    return {
        "rejection_evidence_sha256": sha(evidence),
        "maximum_observed_rejected_values": {
            k: {
                "observed": max(
                    f["failures"][k]["observed"]
                    for f in violations
                    if k in f["failures"]
                ),
                "limit": limit,
            }
            for k, limit in envelope.items()
            if any(k in f["failures"] for f in violations)
        },
        "installs": count,
        "windows_by_points": widths,
        "defined_fields_sha256": fields_hash.hexdigest(),
        "revisions": manager.revision,
        "max_pose_translation_field_error_m": max_error,
        "max_rolling_translation_field_error_m": rolling_error,
        "original_full40_envelope_rejected_windows": full40_rejected,
        "source_seconds_represented": float(last_phase),
        "final_phase_complete": True,
        "HOLD_requests": 1,
        "native_consumption_receipts": 0,
    }


def fault_tests(intent):
    from dataclasses import replace

    from d1_factorial_reference_matrix import check
    from reference_manager_v2 import ReferenceManagerV2

    def reject(fn):
        try:
            fn()
        except ValueError:
            return
        raise ValueError("negative test unexpectedly accepted")

    m = ReferenceManagerV2(intent, "new", 10.0)
    raw = intent.source.copy()
    raw[-1, 8] += 0.001
    revised = replace(intent, source=raw)
    m.revise(revised, 10.5, "new")
    past = ReferenceManagerV2(intent, "new", 10.0)
    past.revise(revised, 10.5, "new")
    reject(lambda: past.revise(intent, 10.4, "new"))
    original_value = intent.at([0.0, 0.5, intent.execution_times[-1]])[0]
    revised_value = m.intent.at([0.0, 0.5, intent.execution_times[-1]])[0]
    import numpy as np

    check(
        np.array_equal(original_value[:2], revised_value[:2])
        and not np.allclose(original_value[-1], revised_value[-1]),
        "future-only revision",
    )
    bad = raw.copy()
    bad[0, 8] += 0.01
    reject(lambda: m.revise(replace(intent, source=bad), 10.5, "new"))
    key, _ = m.window(10.5)
    reject(lambda: m.consume(key, "fake", {"kind": "observer_received"}))
    reject(lambda: m.acknowledge(key, "bad", {}, {}))
    check(m.terminal, "unmatched ack not terminal")
    for next_time in (10.0, 9.8, 10.4, 10.5, float("nan")):
        m = ReferenceManagerV2(intent, "e", 10.0)
        m.window(10.0)
        # Test the source clock predicate independent of the additional pending-ACK stop.
        m.pending = None
        reject(lambda t=next_time, node=m: node.window(t))
        check(m.terminal, "clock fault not terminal")
    return [
        "future-only revision",
        "changing consumed geometry rejected",
        "observer receipt rejected",
        "unmatched installation aborted",
        "duplicate/regressed/nonfinite clock rejected",
        "400ms TTL and 500ms transport gap aborted",
    ]


def main():
    import os

    cli = argparse.ArgumentParser()
    cli.add_argument("root", type=Path)
    cli.add_argument("output", type=Path)
    a = cli.parse_args()
    root = a.root.resolve()
    configure(baseline(root))
    out = a.output.resolve()
    os.umask(0o077)
    out.mkdir(exist_ok=False, parents=True)
    import numpy as np
    from d1_factorial_reference_matrix import check
    from gripper_ownership import assess
    from reference_manager_v2 import TaskSpaceIntentV2

    p = root / "v13/run01"
    v = root / "v12"
    s0 = p / "S0_ACTUAL.json"
    pinned(s0, "6917c8ce81dd237c2e54dca491bec3347ae3c5730546aadae69b12ee36ce21b9")
    pinned(
        p / "S2_ACTUAL.json",
        "e53b719ef09742b623a15d337a6ad1fc45aefd33156cee373b1f956a5c4749d2",
    )
    pinned(
        p / "V13_READOUT.json",
        "bc867cc408a70f050b449f138a6273e94bb8ec295cc6e01036d00b8fc7527409",
    )
    proof = read(s0)
    pins = proof["inputs"]
    for f in (
        v / "A2_O2_DENSE_CAPTURE.json",
        v / "A4_P2_ABI.json",
        v / "TARGET_REGISTRATION.json",
    ):
        pinned(f, pins[str(f)])
    captures = read(v / "A2_O2_DENSE_CAPTURE.json")["windows"]
    abi = read(v / "A4_P2_ABI.json")["windows"]
    grips = read(p / "S2_ACTUAL.json")["windows"]
    v13 = read(p / "V13_READOUT.json")
    zero = read(root / "v15/readiness01/T0_STATIC_HOLD_anchor.private.json")
    base = np.repeat(np.asarray(zero["parts"]["measured_base"])[None], 41, axis=0)
    records = []
    tests = []
    for row, chain_info, g in zip(v13["windows"], abi, grips, strict=True):
        key = row["id"]
        check(key == g["id"] == chain_info["window"], "wrong panel join")
        binding = captures[key]
        path = v / "A4_ABI" / (key + ".npz")
        pinned(path, pins[str(path)])
        pinned(path, chain_info["chain_sha256"])
        rawpath = next(
            Path(f)
            for f, h in pins.items()
            if h == chain_info["raw_sha256"] and f.endswith(".npy")
        )
        pinned(rawpath, chain_info["raw_sha256"])
        with np.load(path, allow_pickle=False) as z:
            raw = z["raw16"]
            archived_world = z["world_knots"]
            archived_source_s = z["source_s"]
            check(
                np.array_equal(raw, np.load(rawpath, allow_pickle=False))
                and np.array_equal(raw[:, 8:], z["selected8"]),
                "source ABI",
            )
        duration = row["duration_s"]
        intent = TaskSpaceIntentV2(
            key,
            raw,
            binding["T_W_source_origin"],
            binding["T_link6_TCP"],
            np.linspace(0, duration, 50),
        )
        poses, gripper, phase = intent.at(intent.execution_times)
        check(
            np.allclose(poses, archived_world, atol=2e-6, rtol=0)
            and np.allclose(phase, archived_source_s, atol=1e-12, rtol=0)
            and np.allclose(gripper, raw[:, 15], atol=1e-10, rtol=0)
            and np.allclose(phase, np.arange(50) / 30, atol=1e-12),
            "source knots",
        )
        if not tests:
            tests = fault_tests(intent)
            from ocs2_target_adapter import wire

            try:
                wire(poses[:39], base, "bad_width", 0, 0.0)
            except ValueError:
                tests.append("39-point window rejected by new wire adapter")
            else:
                raise ValueError("accepted truncated installation")
        one = mode_run(out / (key + "-one"), intent, base, "ONE")
        p10 = mode_run(out / (key + "-p10"), intent, base, "P10")
        check(
            one["defined_fields_sha256"] == p10["defined_fields_sha256"],
            "modes differ for identical remaining intention",
        )
        records.append(
            {
                "id": key,
                "duration_s": duration,
                "selected_right_arm": True,
                "raw_sha256": sha(rawpath),
                "chain_sha256": sha(path),
                "capture_binding_sha256": binding["source_sha256"],
                "input_sha256": binding["input_sha256"],
                "rgb_sha256": binding["rgb_sha256"],
                "gripper_ownership": assess(g),
                "source_gripper_crossings_0.5": int(
                    np.sum((gripper[1:] > 0.5) != (gripper[:-1] > 0.5))
                ),
                "one_shot": one,
                "P10": p10,
            }
        )
    record(
        out / "B3_REFERENCE_MANAGER_SHADOW_ACTUAL.json",
        {
            "stage": "CPU ABI shadow only",
            "windows": records,
            "checks": tests,
            "source_revision": proof["weights"],
            "source_code": proof["xpolicy"],
            "source_samples": 50,
            "source_hz": 30,
            "world_frame": "source origin × interpolated link6 × TCP offset",
            "timing": "linear source phase over archived V1.3 diagnostic duration; D3 feasibility unmeasured",
            "base": "historical T0 measured mount as idle ABI fixture",
            "transport_hz": 10,
            "wire_TTL_s": 0.4,
            "new_native_requests": 0,
            "new_integrations": 0,
            "new_model_forwards": 0,
            "new_optimizer_updates": 0,
            "student_command_issues": 0,
            "native_qualification": "NOT_RUN",
            "D3_precision": "UNMEASURED",
            "B4": "BLOCKED",
        },
    )


if __name__ == "__main__":
    main()
