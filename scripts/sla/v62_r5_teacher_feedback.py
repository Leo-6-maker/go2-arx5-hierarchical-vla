"""Seal genuine held-out teacher observations for a paired R4/R5 replay."""

import argparse
from pathlib import Path

import numpy as np

from go2arx5_vla.sla_adapter.gripper_time_v1 import reference_width
from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from scripts.sla.export_mfree_c20_v1 import _reference


PLAN = "90efbfb8785dbdf7fbb1219fbe179e049ec41e31"
NOISE = "69f4997392ae291bdc50bc2b8699fd0ccac43e9a19b6bbbc7809dee6d92ade0d"
EPISODES = (
    "CHV1-VAL-F1-PLACE-blue-left-01",
    "CHV1-VAL-F1-PLACE-red-right-00",
    "CHV1-VAL-F1-PLACE-red-right-01",
)


def receipt(path):
    return dict(path=str(path), sha256=sha(path))


def choose(queries, events):
    step = lambda label: events[label]["native_step"]
    release = sorted((q for q in queries if q["phase"] in ("LOWER", "OPEN", "WITHDRAW")),
                     key=lambda q: (abs(q["native_step"] - step("released")), q["native_step"]))[:8]
    carry = [q for q in queries if q["phase"] == "CARRY"
             and step("target_lift20mm") <= q["native_step"] < step("correct_region")]
    require(len(release) == 8 and sum(q["phase"] == "OPEN" for q in release) >= 2 and len(carry) >= 8,
            "Need original observed release and hold state populations")
    carrying = [carry[i] for i in np.rint(np.linspace(0, len(carry) - 1, 8)).astype(int)]
    return (("STILL_CARRYING", carrying),
            ("RELEASE_NEIGHBORHOOD", sorted(release, key=lambda q: q["native_step"])))


def build(cases, train, noise, output):
    train_manifest = read(train)
    require(len(train_manifest["episodes"]) == 52, "Compare against the actual R4/R5 target training roster")
    train_scenes = {row["scene"]["sha256"] if isinstance(row["scene"], dict) else row["scene"]
                    for row in train_manifest["episodes"]}
    noise_manifest = read(pin(noise, NOISE))
    tensors = sorted((t for t in noise_manifest["tensors"] if t["label"] == 701), key=lambda t: t["query"])[:48]
    require(len(tensors) == len({t["noise_id"] for t in tensors}) == 48, "Need 48 distinct fixed CUDA noise entries")
    output.mkdir(mode=0o700)
    write(output / "DESIGN.json", dict(plan=PLAN, episodes=EPISODES, windows_per_episode_and_role=8,
          rule_release="Nearest eight LOWER, OPEN or WITHDRAW observations to actual release",
          rule_carry="Eight evenly spaced CARRY observations after actual lift and before entry",
          models=("R4-200", "R5-200"), noise=receipt(noise), training_manifest=receipt(train)))
    samples = []
    for episode in EPISODES:
        case = cases / episode
        native, result = (case / name for name in ("EXECUTION_SPEC.json", "RESULT.private.json"))
        closed = case / "control/LEASE.json"
        spec, actual = read(native), read(result)
        score = actual["detail"]["score"]
        scene = spec["frozen_files"][spec["model_path"]]
        require(read(closed)["status"] == "CLOSED" and actual["physical_requests"] == 1
                and actual["shutdown_clean"] and not actual["unsafe"] and not actual["technical_failure"]
                and score["task_success"] and score["released"] and scene not in train_scenes,
                "Each replay episode must be closed, fully successful and geometry-disjoint from TRAIN")
        observed = case / "native-01/LIVE_DATA"
        query_file, manifest_file = (observed / name for name in ("QUERIES.private.json", "MANIFEST.private.json"))
        captured, queries = read(manifest_file), read(query_file)
        shards = {}
        for role, chosen in choose(queries, score["events"]):
            for row in chosen:
                require(not row["object_phase_memory_in_student_inputs"]
                        and row["rgb_provenance"] == "CURRENT_OWNED_PAUSED_NATIVE_PACKET",
                        "Only causal, real recorded RGB and state are permitted")
                query = row["query"]
                shard = next(d for d in captured["frame_shards"] if d["first_query"] <= query
                             < d["first_query"] + d["count"])
                source = pin(shard["path"], shard["sha256"])
                if source not in shards:
                    with np.load(source, allow_pickle=False) as z:
                        shards[source] = {key: z[key] for key in ("rgb", "state_eef10", "native_step")}
                frames = shards[source]
                i = query - shard["first_query"]
                state = np.asarray(row["state_eef10"])
                rgb = frames["rgb"][i]
                previous = row["previous_confirmed_command"]
                reference_width(previous)
                require(int(frames["native_step"][i]) == row["native_step"]
                        and np.array_equal(state, frames["state_eef10"][i])
                        and rgb.shape == (2, 256, 256, 3) and rgb.dtype == np.uint8,
                        "Both RGB and measured EEF must match the original query")
                path = output / f"{len(samples):02d}-INPUT.npz"
                with path.open("xb") as stream:
                    np.savez(stream, state_eef10=state, body_rgb=rgb[0], wrist_rgb=rgb[1],
                             T_policy_from_mount=np.asarray(row["episode_fixed_T_policy_from_mount"]),
                             policy_from_world=np.asarray(spec["policy_from_world"]),
                             accepted_reference_TCP=_reference(row))
                tensor = tensors[len(samples)]
                samples.append(dict(id=path.stem, episode=episode, scene_sha256=scene,
                    category=role, phase_score_only=row["phase"], query=query, tick=row["native_step"],
                    previous_reference=previous, language=spec["language"], input=receipt(path),
                    teacher_result=receipt(result), teacher_capture=receipt(manifest_file),
                    noise_id=tensor["noise_id"], noise_sha256=tensor["gpu_sha256"]))
    require(len(samples) == 48, "The source population must be closed before either model runs")
    dst = output / "ROSTER.json"
    write(dst, dict(plan=PLAN, design=receipt(output / "DESIGN.json"), rows=samples,
                    model_forwards=0, native_integrations=0, training_updates=0, evaluation_only=True))
    print(sha(dst), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("cases", "train", "noise", "output"):
        parser.add_argument(field, type=Path)
    build(**vars(parser.parse_args()))
