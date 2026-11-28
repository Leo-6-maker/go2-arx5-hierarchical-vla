"""Extend causal A2 context to the two admitted early suffixes without simulation."""

from collections import Counter
import json
from pathlib import Path
import sys
from tempfile import mkdtemp

import numpy as np
from scipy.spatial.transform import Rotation

from m38_gpu_preflight_v1 import pin, read, sha
from v64_adapt_pregrasp_attribution import journal


EARLY = (
    ("v7-d1-early-native-cold-20261128-01", "34e4d7432c4372e3eb8d4ff6b8cf4757c99c409f732d2c848cef7ad117f5a029",
     "V7-D1-EARLY-SEALED-02", "V7-D1-EARLY-NATIVE-01", 100),
    ("v7-d1-early-second-red-cold-20261128-01", "697a31cc889ec6f48e31ede16f25bd94936ffcb26d281bc72d42dcd58fb8e51a",
     "V7-D1-EARLY-SEALED-04", "V7-D1-EARLY-NATIVE-03", 114),
)


def run(work):
    package = Path("/mnt/local/hdd1/UserData/liuyu-nfs/Projects/GR00T-WBC/WM02")
    records, residuals, ticks, phases, elapsed, prior, sources = [], [], [], [], [], [], []
    for cold_name, digest, sealed_name, native_name, expected_windows in EARLY:
        cold = package / cold_name
        manifest = read(pin(cold / "MANIFEST.json", digest))
        entry, = manifest["episodes"]
        seal = read(pin(work / sealed_name / entry["id"] / "ACTUAL.json", entry["source_receipt_sha256"]))
        case = work / native_name / entry["id"]
        result = read(pin(case / "RESULT.private.json", entry["native_result_sha256"]))
        assert result["detail"]["capture"] == seal["capture"]
        assert read(case / "control/LEASE.json")["status"] == "CLOSED"
        lease = read(case / "control/LEASE.json")
        spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
        cap = read(pin(seal["capture"]["path"], seal["capture"]["sha256"]))
        queries = read(pin(cap["labels"]["path"], cap["labels"]["sha256"]))
        pin(seal["actual_native_samples"]["path"], entry["actual_sample_log_sha256"])
        samples = {s["native_end_step"]: s for s in journal(Path(seal["actual_native_samples"]["path"]))}
        with np.load(pin(cold / entry["labels"]["file"], entry["labels"]["sha256"]), allow_pickle=False) as f:
            labels = {key: f[key].copy() for key in ("native_step", "state_eef10", "phase")}
        with np.load(pin(cold / entry["gripper"]["file"], entry["gripper"]["sha256"]), allow_pickle=False) as f:
            previous = f["prior"].copy()
        assert len(queries) == entry["windows"] == expected_windows == len(previous)
        world_from_mount = np.linalg.inv(np.asarray(spec["policy_from_world"])) @ np.asarray(entry["T_policy_from_mount"])
        fixed_from_world = world_from_mount[:3, :3].T
        first = len(ticks)
        handoff = seal["native_teacher_start_tick"]
        for i, q in enumerate(queries):
            tick, recent = q["native_step"], q["recent_reference"]
            sample = samples[tick]
            assert q["query"] == i and tick == labels["native_step"][i] == handoff + 250*i
            assert np.array_equal(q["state_eef10"], labels["state_eef10"][i])
            assert str(labels["phase"][i]) == q["phase"]
            assert sample["native_start_step"] == tick-25 and sample["ticks"] == 25 and sample["safety_valid"]
            assert recent["native_step"] == tick and recent["exact_receiver_knot"]
            assert recent["accepted_wire_sha256"] == sample["installed_reference_sha256"]
            assert abs(recent["receiver_knot_time_s"]-.002*tick) < 1e-8
            assert recent["receiver_observation_time_s"] <= .002*tick+1e-8
            assert recent["receiver_bound_start_s"] <= .002*tick+1e-8
            assert abs(previous[i]-.044*sample["installed_EEF10"][9]) < 1e-12
            actual = np.asarray(q["actual_state"]["tcp"])
            position = fixed_from_world @ (np.asarray(recent["accepted_pose7"][:3])-actual[:3, 3])
            reference_rotation = Rotation.from_quat(recent["accepted_pose7"][3:]).as_matrix()
            rotation = Rotation.from_matrix(fixed_from_world @ reference_rotation @ actual[:3, :3].T @ fixed_from_world.T).as_rotvec()
            residual = np.r_[position, rotation]
            assert np.isfinite(residual).all()
            ticks.append(tick); residuals.append(residual); phases.append(q["phase"])
            prior.append(previous[i]); elapsed.append(.002*(tick-handoff)); sources.append(len(records))
        records.append(dict(episode=entry["id"], reset=entry["reset"], first=first, count=expected_windows,
            cold_manifest_sha256=digest, native_result_sha256=entry["native_result_sha256"],
            query_sha256=cap["labels"]["sha256"], sample_log_sha256=entry["actual_sample_log_sha256"],
            first_teacher_tick=handoff, fixed_world_from_mount=world_from_mount.tolist()))
    assert len(ticks) == 214 and len({r["reset"] for r in records}) == 2
    original = read(pin(work / "v7-full-tracking-context.a9de00e5/ACTUAL.json",
                        "9e15c1ec1b7cd69cbe86fbc8a24b3f5be0e52f1d693d8ac26ba0dbca4d07b70c"))
    assert original["checked_windows"] == 8031
    # Only verify the immutable existing context; do not rescan 96 unchanged episodes.
    pin(original["context"]["path"], original["context"]["sha256"])
    output = Path(mkdtemp(prefix="v71-early-tracking-context.", dir=work))
    residuals, elapsed = np.asarray(residuals), np.asarray(elapsed)
    tiers = np.where(elapsed == 0, "HANDOFF", np.where(elapsed <= 2, "EARLY_0_2S", "LATER"))
    path = output / "CONTEXT.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, episode_index=np.asarray(sources), native_step=np.asarray(ticks),
            accepted_minus_actual_se3_fixed_mount=residuals, phase_audit_only=np.asarray(phases),
            previous_actual_width_m=np.asarray(prior), seconds_since_handoff=elapsed,
            c2_tier_audit_only=tiers, valid=np.ones(214, dtype=bool))
    phase_stats = {}
    for phase, n in Counter(phases).items():
        v = residuals[np.asarray(phases) == phase]
        phase_stats[phase] = dict(windows=n,
            position_norm_mm_p50_p95_max=np.percentile(1000*np.linalg.norm(v[:, :3], axis=1), [50, 95, 100]).tolist(),
            rotation_norm_deg_p50_p95_max=np.percentile(np.degrees(np.linalg.norm(v[:, 3:], axis=1)), [50, 95, 100]).tolist())
    value = dict(status="EARLY214_CAUSAL_ACCEPTED_TRACKING_CONTEXT_PASS", code_sha256=sha(Path(__file__)),
        episodes=records, checked_windows=214, context=dict(path=str(path), sha256=sha(path)),
        original_context_sha256=original["context"]["sha256"], proposed_combined_windows=8245,
        definition="Accepted boundary minus actual TCP: XYZ metres and SO3 log radians in episode-fixed mount",
        causal_previous_25_ticks_exact_wire_verified=True, first_handoff_included=True,
        original_8031_reaudited=False, combined_training_reader_qualified=False,
        controller_selected_reference_directly_observed=False, phase_statistics=phase_stats,
        c2_time_tier_counts=dict(Counter(tiers.tolist())), c2_strata_are_training_changes=False,
        teacher_or_object_metadata_used_as_model_input=False, model_input_changed=False,
        training_accesses=0, new_model_forwards=0, optimizer_updates=0, physics_integrations=0)
    p = output / "ACTUAL.json"
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    print(p, sha(p), flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
