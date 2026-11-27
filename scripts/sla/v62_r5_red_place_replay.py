"""Bind the new successful red PLACE to a separate paired release diagnostic."""

import argparse
from pathlib import Path

import numpy as np

from go2arx5_vla.sla_adapter.gripper_time_v1 import reference_width
from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec
from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from scripts.sla.export_mfree_c20_v1 import _reference
from v62_r5_teacher_feedback import NOISE, PLAN, receipt


EPISODE = "EF5_DIVERSE_PLACE_001"
RESULT = "ea8d2d7b8b9d5d8ec4ffb0571911917723b4b8dc880818c962e86eace4aa65ae"
QUERIES = "b962ca03f5cababc2d6044180b93edd2ef62468b1bde44f05d4c658c8d84e352"
CAPTURE = "2258e31ab2e33eecea6e01d065bb5a99999f0796b0bf4f9e19ea0284843d0c73"
LABELS = "6c235b50fb7595e27949c5e11d2ed71a3c23b4855ff5cf051155a42da519a5b8"
EXPECTED = {"CARRY": 2, "LOWER": 3, "OPEN": 4, "WITHDRAW": 3}


def build(case, label_manifest, noise, out):
    result_file = pin(case / "RESULT.private.json", RESULT)
    result = read(result_file)
    score = result["detail"]["score"]
    events = score["events"]
    require(case.name == EPISODE and score["task_success"] and score["released"]
            and result["shutdown_clean"] and not result["unsafe"] and not result["technical_failure"]
            and read(case / "control/LEASE.json")["status"] == "CLOSED", "A closed native success is required")
    capture = case / "native-01/LIVE_DATA"
    query_file = pin(capture / "QUERIES.private.json", QUERIES)
    frame_file = pin(capture / "MANIFEST.private.json", CAPTURE)
    queries = read(query_file)
    frames = read(frame_file)
    spec = read(case / "EXECUTION_SPEC.json")
    manifest = read(pin(label_manifest, LABELS))
    label, = manifest["episodes"]
    require(label["episode"] == EPISODE and label["result"] == RESULT
            and manifest["gripper_preserved"] and manifest["mask_preserved"]
            and manifest["optimizer_visits"] == 0 and label["windows"] == len(queries),
            "Use verified corrected labels and original support")
    corrected = label["corrected"]
    with np.load(pin(corrected["path"], corrected["sha256"]), allow_pickle=False) as z:
        support, actions, steps, phases = (z[k] for k in ("valid", "action", "native_step", "phase"))
    carrying = [q for q in queries if q["phase"] == "CARRY" and
                events["target_lift20mm"]["native_step"] <= q["native_step"] < events["correct_region"]["native_step"]]
    require(len(carrying) >= 2, "Need distinct observations while safely carrying")
    chosen = {"CARRY": [carrying[0], carrying[-1]],
              "LOWER": [q for q in queries if q["phase"] == "LOWER"][-3:],
              "OPEN": [q for q in queries if q["phase"] == "OPEN"],
              "WITHDRAW": [q for q in queries if q["phase"] == "WITHDRAW"][:3]}
    require({key: len(rows) for key, rows in chosen.items()} == EXPECTED, "Unexpected physical phase counts")
    noise_rows = sorted((r for r in read(pin(noise, NOISE))["tensors"] if r["label"] == 701),
                        key=lambda r: r["query"])[48:60]
    require(len({r["noise_id"] for r in noise_rows}) == 12, "Need twelve fixed noise samples")
    out.mkdir(mode=0o700)
    samples, shards = [], {}
    for row, tensor in zip((row for rows in chosen.values() for row in rows), noise_rows, strict=True):
        query, tick, phase = (row[k] for k in ("query", "native_step", "phase"))
        require(tick == steps[query] and phase == phases[query] and support[query, :10].all()
                and row["actual_state"]["gripper_state_valid"] and not row["object_phase_memory_in_student_inputs"]
                and row["rgb_provenance"] == "CURRENT_OWNED_PAUSED_NATIVE_PACKET",
                "Need actual causal camera, state, and confirmed P10 support")
        shard = next(s for s in frames["frame_shards"] if s["first_query"] <= query < s["first_query"] + s["count"])
        if shard["path"] not in shards:
            with np.load(pin(shard["path"], shard["sha256"]), allow_pickle=False) as z:
                shards[shard["path"]] = {key: z[key] for key in ("rgb", "state_eef10", "native_step")}
        bank = shards[shard["path"]]
        i = query - shard["first_query"]
        state, rgb = np.asarray(row["state_eef10"]), bank["rgb"][i]
        require(int(bank["native_step"][i]) == tick and np.array_equal(bank["state_eef10"][i], state)
                and rgb.shape == (2, 256, 256, 3) and rgb.dtype == np.uint8, "Recorded sensor bytes differ")
        fixed = np.asarray(row["episode_fixed_T_policy_from_mount"])
        codec = GripLogitCodec(fixed)
        recovered = codec.decode(state, np.pad(actions[query], ((0, 0), (0, 25))))
        width = (2 + 40 * recovered[:, 9])[support[query]]
        require(np.isfinite(width).all(), "Teacher consumption labels must be finite")
        previous = row["previous_confirmed_command"]
        reference_width(previous)
        action_label = dict(previous_consumed_mm=previous["width_m"] * 1e3,
                           measured_open_fraction=float(state[9]),
                           consumed_label_p10_mm=width[:10].tolist(),
                           consumed_label_h40_supported_max_mm=float(width.max()),
                           h40_supported_atoms=len(width),
                           current_TCP_world_m=np.asarray(row["actual_state"]["tcp"])[:3, 3].tolist())
        path = out / f"{len(samples):02d}-INPUT.npz"
        with path.open("xb") as stream:
            np.savez(stream, state_eef10=state, body_rgb=rgb[0], wrist_rgb=rgb[1],
                     T_policy_from_mount=fixed, policy_from_world=np.asarray(spec["policy_from_world"]),
                     accepted_reference_TCP=_reference(row))
        samples.append(dict(id=path.stem, category=phase, query=query, tick=tick, episode=EPISODE,
            language=spec["language"], previous_reference=previous, input=receipt(path), teacher=action_label,
            noise_id=tensor["noise_id"], noise_sha256=tensor["gpu_sha256"]))
    dst = out / "ROSTER.json"
    write(dst, dict(plan=PLAN, diagnostic="NEW_RED_PLACE", rows=samples, events=events,
          provenance=dict(result=receipt(result_file), capture=receipt(frame_file), queries=receipt(query_file),
                          corrected_labels=receipt(label_manifest), noise=receipt(noise)),
          evaluation_only=True, model_forwards=0, native_integrations=0, training_updates=0))
    print(sha(dst), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "label_manifest", "noise", "out"):
        parser.add_argument(name, type=Path)
    build(**vars(parser.parse_args()))
