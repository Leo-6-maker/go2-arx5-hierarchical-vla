"""Prepare causal accepted-reference tracking context for the fixed 8031 TRAIN queries."""

from collections import Counter
import io
import json
from pathlib import Path
import sys
import tarfile
from tempfile import mkdtemp

import numpy as np
from scipy.spatial.transform import Rotation

from m38_gpu_preflight_v1 import pin, read, sha
from v64_adapt_pregrasp_attribution import journal


INDEX = "9d9396b503064b9fc94646db1e38c0c3a3ca1a1a24b254310644bd21fca43f56"
REPLAY = "f7f332d0ec16edbd711e838064d6d4a45813260d5807caf5ba09fc13d48f6da4"


def run(work):
    package = Path("/mnt/local/hdd1/UserData/liuyu-nfs/Projects/GR00T-WBC/WM02/slago2-embodiment-a93-r3-hub-20261127-01")
    index = read(pin(package / "index.json", INDEX))
    replay = read(pin(work / "v7-full-physical-label-replay.pmt24d0m/ACTUAL.json", REPLAY))
    assert replay["status"] == "FULL_96_PHYSICAL_LABEL_REPLAY_PASS" and replay["checked_windows"] == 8031
    lookup = {e["episode"]: e for e in index["episodes"]}
    output = Path(mkdtemp(prefix="v7-full-tracking-context.", dir=work))
    episode_id, ticks, residuals, phases, previous_width = [], [], [], [], []
    episodes, errors = [], []
    for number, record in enumerate(replay["episodes"]):
        name = record["episode"]
        first = len(ticks)
        try:
            case = Path(record["canonical_original_case"])
            terminal = read(pin(case / "RESULT.private.json", record["result_sha256"]))
            lease = read(case / "control/LEASE.json")
            assert lease["status"] == "CLOSED"
            spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
            capture = terminal["detail"]["capture"]
            cap = read(pin(capture["path"], capture["sha256"]))
            assert cap["labels"]["sha256"] == record["original_queries_sha256"]
            queries = read(pin(cap["labels"]["path"], record["original_queries_sha256"]))
            samples = {s["native_end_step"]: s for s in journal(case / "native-01/MOBILE_SAMPLES.private.jsonl")}
            entry = lookup[name]
            archive = package / entry["path"]
            # All immutable payload hashes were checked by the immediately
            # preceding full replay; here inspect only its labels member.
            assert archive.stat().st_size == entry["archive_bytes"]
            with tarfile.open(archive, "r") as tar:
                with tar.extractfile("labels.npz") as member, np.load(io.BytesIO(member.read()), allow_pickle=False) as f:
                    labels = {k:f[k] for k in ("native_step", "state_eef10", "prior_actual_width_m", "phase")}
            assert len(queries) == entry["windows"] == len(labels["native_step"])
            world_from_mount = np.linalg.inv(np.asarray(spec["policy_from_world"])) @ np.asarray(entry["T_policy_from_mount"])
            fixed_from_world = world_from_mount[:3,:3].T
            current = []
            for i, q in enumerate(queries):
                tick, recent = q["native_step"], q["recent_reference"]
                before = samples[tick]
                assert tick == labels["native_step"][i] and np.array_equal(q["state_eef10"], labels["state_eef10"][i])
                assert q["phase"] == labels["phase"][i]
                assert before["native_end_step"] == tick and before["native_start_step"] == tick-25
                assert before["ticks"] == 25 and before["safety_valid"]
                assert recent["native_step"] == tick and recent["exact_receiver_knot"]
                assert recent["accepted_wire_sha256"] == before["installed_reference_sha256"]
                assert abs(recent["receiver_knot_time_s"]-.002*tick) < 1e-8
                assert recent["receiver_observation_time_s"] <= .002*tick+1e-8
                assert recent["receiver_bound_start_s"] <= .002*tick+1e-8
                assert abs(labels["prior_actual_width_m"][i]-.044*before["installed_EEF10"][9]) < 1e-12
                actual = np.asarray(q["actual_state"]["tcp"])
                position = fixed_from_world @ (np.asarray(recent["accepted_pose7"][:3])-actual[:3,3])
                reference_rotation = Rotation.from_quat(recent["accepted_pose7"][3:]).as_matrix()
                rotation = Rotation.from_matrix(fixed_from_world @ reference_rotation @ actual[:3,:3].T @ fixed_from_world.T).as_rotvec()
                residual = np.r_[position, rotation]
                assert np.isfinite(residual).all()
                current.append((tick, residual, str(q["phase"]), labels["prior_actual_width_m"][i]))
            for tick, residual, phase, width in current:
                episode_id.append(number)
                ticks.append(tick)
                residuals.append(residual)
                phases.append(phase)
                previous_width.append(width)
            episodes.append(dict(episode=name, first=first, count=len(current), query_sha256=record["original_queries_sha256"],
                                 native_result_sha256=record["result_sha256"], fixed_world_from_mount=world_from_mount.tolist()))
            print(json.dumps(dict(episode=name, windows=len(current), done=len(episodes))),flush=True)
        except Exception as exc:
            errors.append(dict(episode=name, type=type(exc).__name__, message=str(exc)))
            print(json.dumps(errors[-1]),flush=True)
    residuals = np.asarray(residuals)
    arrays = output / "CONTEXT.npz"
    with arrays.open("xb") as stream:
        np.savez_compressed(stream, episode_index=np.asarray(episode_id), native_step=np.asarray(ticks),
            accepted_minus_actual_se3_fixed_mount=residuals, phase_audit_only=np.asarray(phases),
            previous_actual_width_m=np.asarray(previous_width), valid=np.ones(len(ticks), dtype=bool))
    phase_stats = {}
    for phase, count in Counter(phases).items():
        values = residuals[np.asarray(phases) == phase]
        phase_stats[phase] = dict(windows=count, position_norm_mm_p50_p95_max=np.percentile(1000*np.linalg.norm(values[:,:3],axis=1),[50,95,100]).tolist(),
                                 rotation_norm_deg_p50_p95_max=np.percentile(np.degrees(np.linalg.norm(values[:,3:],axis=1)),[50,95,100]).tolist())
    receipt = dict(status="FULL_8031_CAUSAL_ACCEPTED_TRACKING_CONTEXT_PASS" if len(ticks) == 8031 and len(episodes) == 96 and not errors else "INCOMPLETE_OR_FAILED",
        parent_physical_replay_sha256=REPLAY, index_sha256=INDEX, checked_episodes=len(episodes), checked_windows=len(ticks),
        context=dict(path=str(arrays),sha256=sha(arrays)), episodes=episodes, errors=errors, phase_statistics=phase_stats,
        definition="accepted boundary minus actual TCP: XYZ metres and SO3 log radians, in the episode-fixed mount frame",
        causal_source="Exact prior-consumption wire, receiver observation and bound start no later than current query",
        controller_selected_reference_directly_observed=False, model_input_changed=False, training_accesses=0,
        phase_teacher_or_object_metadata_used_as_model_input=False, new_model_forwards=0, new_native_integrations=0, new_optimizer_updates=0)
    path = output / "ACTUAL.json"
    path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    print(path,sha(path),flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]))
