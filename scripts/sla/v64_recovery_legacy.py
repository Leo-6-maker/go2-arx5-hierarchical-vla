"""Register new Recovery attempts on two independently qualified legacy TRAIN scenes."""

import argparse
import json
from pathlib import Path

from v64_recovery_roster import PROTECTED, QUERIES, TICKS, fixed, seed, sha


FIFTH_CASES = (
    ("V64_RECOVERY_TRAIN_09", "MEV1-TRAIN-F1-NEAR_NEG-00-PLACE-blue-left", "PLACE-blue-left"),
    ("V64_RECOVERY_TRAIN_10", "MEV1-TRAIN-F1-NEAR_NEG-01-PLACE-red-right", "PLACE-red-right"),
)
FOURTH_RESULT = "62efea73bda02390ad9a478ddc2e228a648d389d2b2c650a2c65a34610bd2805"
TEACHER_TERMINAL = "55bb24644b237fdb7f2118ff9e9809f97a910063cd34ef9e0d31eccbf0967c16"
TEACHER_LABELS = "acdaa698c0c30bc1e68ea4f48556cd0f69c6c434da407c245d10c118a85f4027"
TASK_EVENTS = {"target_grasp", "target_lift20mm", "correct_region", "released", "retreated50mm"}


def reference(path):
    return dict(path=str(path), sha256=sha(path))


def bind(site, teacher_wave, labels, output):
    from v64_recovery_cohorts import read_roster, FOURTH_ROSTER

    prior_roster = site / "RECOVERY-TRAIN-ROSTER-04.json"
    prior_result = site / "RECOVERY-TRAIN-SEALED-04/RESULT.json"
    fourth, _ = read_roster(prior_roster, FOURTH_ROSTER)
    terminal = fixed(prior_result, FOURTH_RESULT)
    if (not terminal["complete_reconciliation"] or not terminal["forward_ledger_reconciled"]
            or terminal["terminal_cases"] != terminal["denominator"] or terminal["denominator"] != 2
            or [c["id"] for c in terminal["cases"]] != [c["id"] for c in fourth["cases"]]
            or any(c["state"] != "CLOSED" for c in terminal["cases"])):
        raise ValueError("Require all previously paid fourth-cohort attempts to be closed")

    protected = fixed(site / "v63-place-core-code-01/PROTECTED_POPULATION.json", PROTECTED)["clusters"]
    prior = [case for wave in range(1, 5) for case in read_roster(site / f"RECOVERY-TRAIN-ROSTER-{wave:02}.json")[0]["cases"]]
    paid = fourth["previous_recovery_native_requests"] + terminal["denominator"]
    if len(prior) != paid or paid + len(FIFTH_CASES) > fourth["max_native_requests_cumulative"]:
        raise ValueError("Paid Recovery history or total request allowance changed")
    used_scenes = set(fourth["protected_scene_sha256"]) | {case["scene_sha256"] for case in prior}
    used_resets = {row["reset"] for row in (*protected, *prior)}

    teacher_terminal = teacher_wave / "RESULT.private.json"
    teacher_records = fixed(teacher_terminal, TEACHER_TERMINAL)
    corrected = fixed(labels, TEACHER_LABELS)
    if teacher_records["exit_code"] or teacher_records["unreconciled"] is not None:
        raise ValueError("Use only the completed legacy Teacher wave")
    cards, sources = [], {}
    for identity, original, task in FIFTH_CASES:
        row, = (item for item in teacher_records["records"] if item["condition"] == original)
        label, = (item for item in corrected["episodes"] if item["episode"] == original)
        case = teacher_wave / "cases" / original
        score = fixed(case / "RESULT.private.json", row["result"]["sha256"])
        lease = json.loads((case / "control/LEASE.json").read_text())
        spec = fixed(case / "EXECUTION_SPEC.json", lease["spec_sha256"])
        scene = Path(spec["model_path"])
        scene_sha = sha(scene)
        reset, noise = seed(identity, "reset"), seed(identity, "noise")
        if (row["split"] != "TRAIN" or not row["admitted"] or not row["task_and_mobile_success"]
                or row["hard_STOP"] or row["technical_failure"] or not score["shutdown_clean"]
                or not score["detail"]["score"]["stable_release_condition"]
                or not TASK_EVENTS <= score["detail"]["score"]["events"].keys()
                or lease["status"] != "CLOSED" or spec["collection_split"] != "TRAIN"
                or spec["task_instruction_id"] != task or label["task"] != task
                or label["scene_sha256"] != scene_sha or spec["frozen_files"][str(scene)] != scene_sha
                or sha(labels.parent / label["label"]) != label["sha256"]
                or scene_sha in used_scenes or reset in used_resets or reset == spec["mobile_condition"]["reset"]):
            raise ValueError("Legacy Recovery requires a distinct TRAIN scene and its successful original Teacher and R4 labels")
        used_scenes.add(scene_sha)
        used_resets.add(reset)
        cards.append(dict(id=identity, original_teacher=original, task=task, scene_sha256=scene_sha,
                          scene_path=str(scene), geometry=spec["multitask_geometry"], reset=reset, policy_seed=noise,
                          family=spec["mobile_condition"]["family"], max_ticks=TICKS, maximum_queries=QUERIES))
        sources[identity] = dict(teacher_wave=str(teacher_wave), terminal=reference(teacher_terminal), labels=reference(labels))

    document = dict(fourth, cases=cards, teacher_sources=sources, previous_recovery_native_requests=paid,
                    fourth_prior_roster=reference(prior_roster), fourth_prior_terminal=reference(prior_result))
    with output.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(sha(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("site", "teacher_wave", "labels", "output"):
        parser.add_argument(argument, type=Path)
    bind(**vars(parser.parse_args()))
