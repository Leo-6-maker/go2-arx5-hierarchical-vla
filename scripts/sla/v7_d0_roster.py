"""Bind two fresh early Recovery TRAIN candidates to proven teacher geometries."""

import argparse
import hashlib
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write


PLAN = "67d15fb8fad688a2bbd82e225111bb49ec34559b"
HISTORY = (
    ("e9b7552d4c10779f4c87a04ef23c252ee2f599a69a24df5d8f94624166ad6cba",
     "c59b8c4118397c31c9d05eb45e7a7749e30dcb7fb6b3dc04d9df97015cf0aba2"),
    ("4018539bb1f083322450b2fd0bdfdc76380f41a91aa926225e47c2a17ed77959",
     "9dd1cf7bca7ea087bebb183c2a2d117fb9dbb8d87b890ca85c3357da1b1c4e2c"),
    ("c8af07809c2e49ce1c2ec3f135e57710339008aaccfc4349e4988a22428639e4",
     "70092525ee39111c7df6e6132b8cb72abbecf2774fbc32e963d0c2d917298862"),
    ("ec398ae3b43222bdfd5c1d075b7827ea017b6cda1ef6401ee9987dbd7be3bb0d",
     "62efea73bda02390ad9a478ddc2e228a648d389d2b2c650a2c65a34610bd2805"),
    ("a140b5a5577807582d2009ce24999821ac9397acf189bc6180a3d81f495834cf",
     "06cbfdc794d9fd05a6336c8e15a4f6017b946e13aaac067625003bad93d7771e"),
)
PROTECTED = "a1dfa32f329cb91d474e20488cbeeafa3dc41858f32dacd6cd593dbed61bbb26"
REPLAY = "e524b5d6d8e239e019dfe575ea848e8ceb88fbc56abce481be7bd4d7dbda81de"
PROVIDER = "e825a48ed540b5d4d6c3623da24ac56d5aa3b51f2bf3aa488b313f21e044da5e"
R4 = "f1f5a994005a9fc831934a156ca823d7e4b67794ce2533d72e087c2aeb29b9cb"
FIRST_EARLY = "4e10cfdd865ade7fe6e360664d2601530ba5737e9160e14d1c9abdde4edfe761"
FIRST_CLOSE = "61051e05a5bbecabd2a0da00b7117d69871c1bd000d7e85f2d0b35bf305d2576"
SEQUENCE_PROOF = "f226da4f79d41c7389fabe202886fa75995b163369a8232b660b9f726ba7ecd2"
TICKS, QUERIES = 50703, 180
ARM = (
    ("V64_RECOVERY_TRAIN_09", "V7_D0_EARLY_TRAIN_11", "LIFT", "PLACE-blue-left"),
    ("V64_RECOVERY_TRAIN_10", "V7_D0_EARLY_TRAIN_12", "PREGRASP", "PLACE-red-right"),
)


def key(case, purpose):
    return int.from_bytes(hashlib.sha256(f"{PLAN}|{case}|{purpose}".encode()).digest()[:4], "little")


def teacher(card, source):
    provenance_path = pin(source["terminal"]["path"], source["terminal"]["sha256"])
    record, = [r for r in read(provenance_path)["records"] if r["condition"] == card["original_teacher"]]
    labels_path = pin(source["labels"]["path"], source["labels"]["sha256"])
    label, = [r for r in read(labels_path)["episodes"] if r["episode"] == card["original_teacher"]]
    require(record["admitted"] and record["task_and_mobile_success"] and record["split"] == "TRAIN"
            and not record["technical_failure"] and not record["hard_STOP"]
            and label["scene_sha256"] == card["scene_sha256"] and label["task"] == card["task"],
            "Require the prior admitted teacher task, without an unsafe terminal")
    pin(Path(labels_path).parent/label["label"],label["sha256"])
    outcome = Path(pin(record["result"]["path"],record["result"]["sha256"]))
    result = read(outcome)
    require(result["shutdown_clean"] and result["detail"]["score"]["stable_release_condition"],
            "Original admitted geometry needs complete teacher release and clean closure")
    lease = read(outcome.parent / "control/LEASE.json")
    spec = outcome.parent / "EXECUTION_SPEC.json"
    require(lease["status"] == "CLOSED" and sha(spec) == lease["spec_sha256"], "Teacher owner must be closed")
    config = read(spec)
    scene = Path(card["scene_path"])
    require(config["collection_split"] == "TRAIN" and config["task_instruction_id"] == card["task"]
            and config["multitask_geometry"] == card["geometry"] and config["model_path"] == str(scene)
            and config["frozen_files"][str(scene)] == card["scene_sha256"] == sha(scene),
            "Recovery must use the original frozen teacher scene and task")
    return dict(spec_sha256=sha(spec), result_sha256=sha(outcome), labels_sha256=sha(labels_path),
                original_teacher=card["original_teacher"])


def followup_history(site, previous):
    first = read(pin(previous, FIRST_EARLY))
    close = read(pin(site / "V7-D0-EARLY-SEALED-01/RESULT.json", FIRST_CLOSE))
    proof = read(pin(site / "v7-d0-sequence-fix.NNgNEL/ACTUAL.json", SEQUENCE_PROOF))
    require(first["plan"] == PLAN and close["complete_reconciliation"]
            and close["denominator"] == close["terminal_cases"] == 2 and close["train_recovery_count"] == 0
            and [row["id"] for row in first["cases"]] == [row["id"] for row in close["cases"]]
            and all(row["state"] == "CLOSED" and not row["recovery_admitted"] for row in close["cases"])
            and not close["cases"][0]["technical_failure"] and close["cases"][1]["technical_failure"],
            "The original early cards must be closed and retained")
    require(proof["frozen_receiver_requires_contiguous"] and not proof["actual_starts"]
            and proof["expected_next_sequence"] == proof["previous_actual_gripper_sequence"] + 1
            and proof["archived_technical_attempt_packet_sequence"] > proof["expected_next_sequence"],
            "Require the exact closed sequence fix witness")
    return first, dict(earlier_early_roster_sha256=FIRST_EARLY, earlier_early_result_sha256=FIRST_CLOSE,
                       sequence_fix_cpu_sha256=SEQUENCE_PROOF, remaining_before=8, remaining_after=6)


def build(site, output, followup=None):
    used_resets, used_noises = set(), set()
    started, admitted = 0, 0
    earlier = []
    for i, (bank, result) in enumerate(HISTORY, 1):
        roster = read(pin(site/f"RECOVERY-TRAIN-ROSTER-{i:02d}.json", bank))
        close = read(pin(site/f"RECOVERY-TRAIN-SEALED-{i:02d}/RESULT.json", result))
        require(close["complete_reconciliation"] and close["denominator"] == close["terminal_cases"] == 2
                and [r["id"] for r in close["cases"]] == [r["id"] for r in roster["cases"]]
                and all(r["state"] == "CLOSED" for r in close["cases"]), "Incomplete earlier denominator")
        started += close["terminal_cases"]
        admitted += close["train_recovery_count"]
        for row in roster["cases"]:
            used_resets.add(row["reset"])
            used_noises.update((row["policy_seed"]+i) % 2**31 for i in range(QUERIES))
        earlier.append(roster)
    require((started,admitted) == (10,4), "Early Recovery cohort budget changed")
    last = earlier[-1]
    protected = read(pin(site/"v63-place-core-code-01/PROTECTED_POPULATION.json",PROTECTED))["clusters"]
    guarded = {r["scene"]["sha256"] for r in protected if r.get("split") != "TRAIN"}
    used_resets.update(r["reset"] for r in protected)
    replay_path = pin(site/"v64-recovery-early-teacher-code-01/ACTUAL-EARLY-TEACHER-GUARDED-CLOCK-01.json",REPLAY)
    replay = read(replay_path)
    require(not any(replay[x] for x in ("native_integrations","model_forwards","new_training_labels")),
            "Keep the archived replay separate from dynamic evidence")
    r4 = last["student_provider"]
    manifest = read(pin(Path(r4["path"])/"CHECKPOINT.json",PROVIDER))
    require(r4["sha256"] == PROVIDER and manifest["training_receipt_sha256"] == R4
            and manifest["checkpoint_sha256"] == last["checkpoint_sha256"], "Require actual R4-200")

    cohort = ARM
    extension = {}
    if followup is not None:
        previous, extension = followup_history(site, followup)
        for row in previous["cases"]:
            used_resets.add(row["reset"])
            used_noises.update((row["policy_seed"] + i) % 2**31 for i in range(QUERIES))
        started += 2
        cohort = tuple((old, f"V7_D0_EARLY_TRAIN_{number}", stage, task)
                       for (old, _, stage, task), number in zip(ARM, (13, 14), strict=True))

    cases = []
    for old,new,stage,task in cohort:
        baseline, = [r for r in last["cases"] if r["id"] == old]
        fixture, = [r for r in replay["cases"] if r["case"] == old]
        require(baseline["task"] == fixture["task"] == task and fixture["candidate"]["stage"] == stage
                and baseline["scene_sha256"] not in guarded, "Choose only the registered TRAIN stage and scene")
        reset, noise = key(new,"reset"), key(new,"noise")
        ids = {(noise+i) % 2**31 for i in range(QUERIES)}
        require(reset not in used_resets and ids.isdisjoint(used_noises), "Prospective random stream is not fresh")
        used_resets.add(reset)
        used_noises.update(ids)
        provenance = teacher(baseline,last["teacher_sources"][old])
        cases.append(dict(baseline,id=new,reset=reset,policy_seed=noise,required_stage=stage,
                          historical_failure=old,teacher_provenance=provenance))
    require(len({r["scene_sha256"] for r in cases}) == 2, "Require two distinct TRAIN scenes")
    write(output,dict(plan=PLAN,split="TRAIN",cases=cases,student_provider=r4,
          checkpoint_sha256=manifest["checkpoint_sha256"],r4_training_receipt_sha256=R4,
          protected_scene_sha256=sorted(guarded),archive_replay_sha256=REPLAY,past_attempts=started,
          past_admissions=admitted,allowed_trigger="EARLY_UNINSTALLED_G7",
          max_new_requests=2,max_ticks=2*TICKS,max_queries=2*QUERIES,
          authorized_native_runtime=False,actual_starts=0,model_forwards=0,actual_integrations=0,actual_admissions=0,
          **extension))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("site","output"):
        cli.add_argument(arg,type=Path)
    cli.add_argument("--followup",type=Path)
    build(**vars(cli.parse_args()))
