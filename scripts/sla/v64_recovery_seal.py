"""Package a safe Recovery suffix from actual P10 consumption and real RGB."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from v64_recovery_cohorts import roster_ref
from v64_recovery_launcher import ref
from v64_recovery_roster import fixed, sha


RED_CONTROL = "9925170439298abc600b25ebaaacfad331b245a3b42ba845353f6310ef6f42a6"
RED_CORRECTED = "d9402d9586069944cff0977206863ecf2e9dea1a60e361b7dcc5a290d38b85dd"


def read(path):
    return json.loads(path.read_bytes())


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def actual_queries(case, detail, spec):
    authority = detail["capture"]
    capture = fixed(Path(authority["path"]), authority["sha256"])
    labels = capture["labels"]
    queries = fixed(Path(labels["path"]), labels["sha256"])
    need(capture["schema"] == "MFREE_LIVE_FINAL_PLAN_CAPTURE_V1"
         and capture["writer_error"] is None and capture["queries"] == len(queries)
         and len(queries) > 0 and capture["language"] == spec["language"]
         and capture["frozen_scene_sha256"] == spec["frozen_files"][spec["model_path"]]
         and not capture["planned_suffix_is_executed"],
         "The closed capture must bind the same real task, scene and query denominator")
    need(Path(labels["path"]).resolve(strict=True).is_relative_to(case / "native-01"),
         "The original capture must stay within its case")
    return capture, queries


def accepted_rows(case, queries, terminal_tick):
    first_tick = queries[0]["native_step"]
    need(terminal_tick == first_tick + 250 * len(queries), "Require a complete exact stopped-clock suffix")
    sample_log = case / "native-01/MOBILE_SAMPLES.private.jsonl"
    with sample_log.open() as stream:
        samples = [row for row in map(json.loads, stream) if row["native_start_step"] >= first_tick]
    need(len(samples) == 10 * len(queries), "Support every 50ms slot using the real native log")
    world, cursor = [], 0
    for i, query in enumerate(queries):
        tick = first_tick + 250 * i
        need(query["native_step"] == tick and query["query"] == i + queries[0]["query"]
             and query["consumed_mask"] == [True] * 10 + [False] * 30
             and query["actual_elapsed_prefix_ticks"] == 250 and query["partial_interval_ticks"] == 0
             and query["rgb_provenance"] == "CURRENT_OWNED_PAUSED_NATIVE_PACKET"
             and len(query["intervals"]) == 10, "Required actual confirmed P10 and synchronized RGB")
        for atom, interval in enumerate(query["intervals"]):
            sample = samples[cursor]
            boundary = interval["boundary"]
            start = tick + 25 * atom
            need(interval["atom"] == atom and sample["ticks"] == interval["confirmed_ticks"] == 25
                 and sample["native_start_step"] == interval["start_native_step"] == start
                 and sample["native_end_step"] == interval["end_native_step"] == boundary["native_step"] == start + 25
                 and sample["installed_reference_sha256"] == interval["installed_reference_sha256"]
                 and boundary["accepted_wire_sha256"] == interval["installed_reference_sha256"]
                 and boundary["exact_receiver_knot"] and sample["safety_valid"],
                 "The declared Teacher atom must match the accepted wire and real native sample")
            world.append(sample["installed_EEF10"])
            cursor += 1
    return np.asarray(world), ref(sample_log)


def typed_labels(spec, queries, world):
    from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec
    from scripts.pi0291_arx5_data.native_dataset import matrix_from10, pose10
    from v6_reanchored_teacher_labels import reexpress

    width = (world[:, 9] * .044 - .002) / .040
    need(np.isfinite(world).all() and np.all((width >= -1e-10) & (width <= 1. + 1e-10)),
         "Every training row must keep its actual G7 jaw width")
    declared = pose10(np.asarray(spec["policy_from_world"]) @ matrix_from10(world), np.clip(width, 0., 1.))
    frame = queries[0]["episode_fixed_T_policy_from_mount"]
    codec = GripLogitCodec(np.asarray(frame))
    state = np.asarray([row["state_eef10"] for row in queries])
    latent = np.zeros((len(queries), 40, 7), dtype=np.float32)
    valid = np.zeros(latent.shape[:2], dtype=bool)
    for i, q in enumerate(queries):
        need(np.array_equal(q["episode_fixed_T_policy_from_mount"], frame), "Fixed native frame changed")
        rows = declared[10*i:10*i+40]
        n = len(rows)
        _, encoded = codec.encode(state[i], np.vstack((rows, np.tile(rows[-1], (40-n, 1)))))
        latent[i, :n] = encoded[:n]
        valid[i, :n] = True
    raw = dict(action=latent, valid=valid, phase=np.asarray([q["phase"] for q in queries]),
               native_step=np.asarray([q["native_step"] for q in queries], dtype=np.int64))
    corrected, residual, p10 = reexpress(spec, queries, raw)
    need(max(residual) < .001 and max(p10) < .001 and np.array_equal(corrected[..., 6], latent[..., 6])
         and not corrected[~valid].any(), "R4 inverse transport or real action masking failed")
    return dict(raw, action=corrected, state_eef10=state), residual.tolist(), p10.tolist()


def verify_frames(case, capture, queries):
    from go2arx5_vla.sla_adapter.dataset_v1 import images224
    from openpi_client.image_tools import resize_with_pad

    ticks, reads, shards = [], [], []
    for member in capture["frame_shards"]:
        source = Path(member["path"])
        need(source.resolve(strict=True).is_relative_to(case / "native-01")
             and sha(source) == member["sha256"]
             and member["first_query"] == len(ticks), "Original frame shard identity or order changed")
        with np.load(source, allow_pickle=False) as f:
            first, count = len(ticks), member["count"]
            rgb, states, native = (f[key] for key in ("rgb", "state_eef10", "native_step"))
            selected = queries[first:first+count]
            need(rgb.shape == (count, 2, 256, 256, 3) and rgb.dtype == np.uint8
                 and np.array_equal(states, np.asarray([q["state_eef10"] for q in selected]))
                 and np.array_equal(native, np.asarray([q["native_step"] for q in selected])),
                 "RGB, state and native time must come from the same paused query")
            for i in (0, count-1):
                views = images224(rgb[i], resize_with_pad)
                reads.append(dict(native_tick=int(native[i]), images_sha256={
                    key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in views["image"].items()
                    if views["image_mask"][key]}))
            ticks.extend(native.tolist())
        shards.append(member)
    need(ticks == [q["native_step"] for q in queries], "The capture must cover every compiled query")
    return dict(shards=shards, original_reader_boundary_reads=reads)


def inspect(case, output, control=None):
    lease = read(case / "control/LEASE.json")
    spec = fixed(case / "EXECUTION_SPEC.json", lease["spec_sha256"])
    terminal = case / "RESULT.private.json"
    original_sha = sha(terminal)
    result = fixed(terminal, original_sha)
    need(lease["status"] == "CLOSED" and result["completed"] and result["shutdown_clean"]
         and not result["unsafe"] and not result["technical_failure"] and not result["optimizer_updates"]
         and result["detail"]["score"]["task_success"], "Only successful, clean, sealed capture may qualify")
    detail = result["detail"]
    capture, queries = actual_queries(case, detail, spec)
    if control is None:
        binding, _ = roster_ref(spec["v64_recovery_train"])
        card, = (c for c in binding["cases"] if c["id"] == case.name)
        need(spec["collection_split"] == "TRAIN" and spec["evaluation_split"] == "TRAIN_RECOVERY"
             and detail["teacher_takeover"] and not detail["task_success"]
             and detail["recovery_task_success"] and spec["mobile_condition"]["reset"] == card["reset"]
             and len(queries) == detail["completed_teacher_P10"]
             and detail["consumed_teacher_atoms"] == 10 * len(queries)
             and all(q["supervision_role"] == "RECOVERY_TEACHER" for q in queries),
             "The student prefix and partial recovery are ineligible for TRAIN")
        intent = read(case / "native-01/RECOVERY-HANDOFF-INTENT.private.json")
        need(intent["actual_student_queries"] == result["model_generations"]
             and intent["native_step"] == queries[0]["native_step"] and intent["clock_owner_unchanged"],
             "The accepted handoff must precede every labeled Teacher step")
    else:
        need(original_sha == RED_CONTROL and not result["model_generations"],
             "The comparison is the fixed successful red Teacher control")
        entry, = fixed(control, RED_CORRECTED)["episodes"]
        cut = next(i for i, q in enumerate(queries) if q["phase"] == "CARRY")
    world, samples = accepted_rows(case, queries, result["actual_native_integrations"])
    labels, maximum, prefix = typed_labels(spec, queries, world)
    frames = verify_frames(case, capture, queries)
    if control is not None:
        actual = entry["corrected"]
        path = Path(actual["path"])
        need(sha(path) == actual["sha256"], "The independent R4 red control changed")
        with np.load(path, allow_pickle=False) as prior:
            need(all(np.array_equal(labels[k][cut:], prior[k][cut:]) for k in ("action", "valid", "native_step", "phase")),
                 "The CARRY to WITHDRAW suffix must equal the independent full Teacher reader")
    else:
        cut = 0
    output.mkdir(mode=0o700)
    labels_path = output / "LABELS.npz"
    with labels_path.open("xb") as stream:
        np.savez_compressed(stream, **{k: array[cut:] for k, array in labels.items()})
    chosen = queries[cut:]
    receipt = dict(schema="V64_RECOVERY_R4_ACTUAL_CONSUMED_SUFFIX", actual_recovery=control is None,
                   native_result=ref(terminal), capture=detail["capture"], actual_native_samples=samples,
                   scene=spec["frozen_files"][spec["model_path"]], reset=spec["mobile_condition"]["reset"],
                   task=spec["task_instruction_id"], training_windows=len(chosen), labels=ref(labels_path),
                   maximum_roundtrip_mm_deg=maximum, maximum_P10_roundtrip_mm_deg=prefix,
                   native_teacher_start_tick=chosen[0]["native_step"], actual_teacher_atoms=10 * len(chosen),
                   phase_windows=dict(Counter(q["phase"] for q in chosen)), rgb=frames,
                   original_control_npz_compared=control is not None, optimizer_visits=0, new_integrations=0)
    with (output / "ACTUAL.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("case", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--control", type=Path)
    inspect(**vars(cli.parse_args()))
