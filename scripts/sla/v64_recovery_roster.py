"""Bind two previous successful TRAIN geometries to fresh Recovery attempts."""

import argparse
import hashlib
import json
from pathlib import Path


PLAN = "531f826b36c1195db8c253ca71497f1aeb569c80"
POPULATION = "7a9e350319006ce628a718f533de2dd520222dc8c3042a7b738697957762418e"
PROTECTED = "a1dfa32f329cb91d474e20488cbeeafa3dc41858f32dacd6cd593dbed61bbb26"
TERMINAL = "ad40b4a7dbbd410061b861f64014f9a33bff35b74a15b0cf786b8d907aa7ec82"
LABELS = "7903047946c60cb538b54e195fb65b1b88a7d79024aef4a766c06613a81dc8b5"
R4_RELEASE = "0896270957c478197e586a71f2d2d3a55464a5bb7436cf3d390fa14804926433"
R4_PROVIDER = "e825a48ed540b5d4d6c3623da24ac56d5aa3b51f2bf3aa488b313f21e044da5e"
ROSTER = (("V64_RECOVERY_TRAIN_01", "EF5_DIVERSE_PLACE_007", "PLACE-blue-left"),
          ("V64_RECOVERY_TRAIN_02", "EF5_DIVERSE_PLACE_008", "PLACE-red-right"))
TICKS, QUERIES = 50_703, 180


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fixed(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Frozen recovery witness changed: {path.name}")
    return json.loads(raw)


def seed(identity, use):
    payload = f"{PLAN}|{identity}|{use}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def bind(population, protected, terminal, labels, release, provider, output):
    closed = fixed(terminal, TERMINAL)
    prior = {r["condition"]: r for r in closed["records"]}
    train = fixed(labels, LABELS)
    known = {r["episode"]: r for r in train["episodes"]}
    source = fixed(population, POPULATION)
    frozen = fixed(protected, PROTECTED)["clusters"]
    r4 = fixed(release, R4_RELEASE)
    package = fixed(provider/"CHECKPOINT.json", R4_PROVIDER)
    if (closed["paid_requests"] != 2 or closed["exit_code"] != 0 or train["train"] != 2
            or not r4["released"] or not r4["native_gpu_and_world_leases_closed"]):
        raise ValueError("Use the two actually admitted TRAIN and released R4 identities")
    guarded = {r["scene"]["sha256"] for r in frozen if r.get("split") != "TRAIN"}
    resets = {r["reset"] for r in frozen} | {r["reset"] for r in source["clusters"]}
    by_scene = {row["id"]: row for row in source["clusters"]}
    by_task = {row["id"]: row for row in source["conditions"]["TRAIN"]}
    cards = []
    for name, original, task in ROSTER:
        scene, instruction, old, row = by_scene[original], by_task[original], prior[original], known[original]
        s = scene["scene"]
        if (scene["split"] != "TRAIN" or instruction["instruction"] != task
                or s["sha256"] in guarded or sha(Path(s["path"])) != row["scene"] or row["scene"] != s["sha256"]
                or old["result"]["sha256"] != row["result"] or not old["admitted"] or not old["task_and_mobile_success"]
                or old["hard_STOP"] or old["technical_failure"] or old["split"] != "TRAIN"):
            raise ValueError("A held-out or incomplete case cannot serve as Recovery TRAIN geometry")
        reset = seed(name, "reset")
        if reset in resets:
            raise ValueError("The prospectively fixed Recovery reset collides with history")
        resets.add(reset)
        cards.append(dict(id=name, original_teacher=original, task=task, scene_sha256=s["sha256"],
            scene_path=s["path"], geometry=scene["scoring_geometry"], reset=reset,
            policy_seed=seed(name, "noise"), family=scene["family"], max_ticks=TICKS, maximum_queries=QUERIES))
    if len({c["scene_sha256"] for c in cards}) != 2 or len({c["policy_seed"] for c in cards}) != 2:
        raise ValueError("The two Recovery conditions must remain independent")
    document = dict(plan=PLAN, split="TRAIN", cases=cards, protected_scene_sha256=sorted(guarded),
        student_provider=dict(path=str(provider), sha256=R4_PROVIDER), checkpoint_sha256=package["checkpoint_sha256"],
        source_teacher_labels_sha256=LABELS, teacher_terminal_sha256=TERMINAL, protected_inputs_sha256=PROTECTED,
        max_native_requests=2, max_native_ticks=2*TICKS, actual_start_count=0,
        actual_native_integrations=0, actual_recovery_admissions=0, exact_runtime_preflight_required=True)
    with output.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("population", "protected", "terminal", "labels", "release", "provider", "output"):
        parser.add_argument(name, type=Path)
    bind(**vars(parser.parse_args()))
