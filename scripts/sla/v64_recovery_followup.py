"""Seal the second independent Recovery pair from two actual TRAIN PLACE waves."""
import argparse
from pathlib import Path

from v64_recovery_roster import PLAN, PROTECTED, QUERIES, TICKS, fixed, seed, sha


FIRST_ROSTER = "e9b7552d4c10779f4c87a04ef23c252ee2f599a69a24df5d8f94624166ad6cba"
FIRST_RESULT = "c59b8c4118397c31c9d05eb45e7a7749e30dcb7fb6b3dc04d9df97015cf0aba2"
FOLLOWUP = (
    ("V64_RECOVERY_TRAIN_03", "EF5_DIVERSE_PLACE_005", "PLACE-blue-left", "03",
     "a6d6a35dd5f9fc15b6f587c17eed3bcb355b044b68776494a915d602546b301e",
     "6c86071bb00828c6a891be72f6be2c48aeea1b3f3d9a9ceb61306a7caa0526f1",
     "955ad71fe4ddeb8e47fc184b2a3c3b8fc49678a8b4d546dfd73f64102d52af69"),
    ("V64_RECOVERY_TRAIN_04", "EF5_DIVERSE_PLACE_010", "PLACE-red-right", "05",
     "664aee826dcb87082ec6b26ef17d700dd07775afff60ce4a090cb12a6eca4d22",
     "270991e71f8cc47171d40f05c31a3d61ad629c4cf9bface9767ca484ced20e2c",
     "d9402d9586069944cff0977206863ecf2e9dea1a60e361b7dcc5a290d38b85dd"),
)
ROSTER = tuple(row[:3] for row in FOLLOWUP)


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def bind(site, output):
    import json

    first_path = site / "RECOVERY-TRAIN-ROSTER-01.json"
    close_path = site / "RECOVERY-TRAIN-SEALED-01/RESULT.json"
    first, close = fixed(first_path, FIRST_ROSTER), fixed(close_path, FIRST_RESULT)
    protected = fixed(site / "v63-place-core-code-01/PROTECTED_POPULATION.json", PROTECTED)["clusters"]
    if (close["plan"] != PLAN or not close["complete_reconciliation"] or close["terminal_cases"] != 2
            or close["denominator"] != 2 or close["train_recovery_count"] != 1
            or first["plan"] != PLAN or first["split"] != "TRAIN"
            or any(row["state"] != "CLOSED" for row in close["cases"])):
        raise ValueError("Require the closed two-card pilot with one real admitted Recovery")
    blocked = {r["scene"]["sha256"] for r in protected if r.get("split") != "TRAIN"}
    taken = blocked | {row["scene_sha256"] for row in first["cases"]}
    used_reset = {row["reset"] for row in protected} | {row["reset"] for row in first["cases"]}
    used_noise = {row["policy_seed"] % 2**31 for row in first["cases"]}
    cards, sources = [], {}
    for identity, origin, instruction, wave, pop_sha, terminal_sha, labels_sha in FOLLOWUP:
        teacher = site / f"DIVERSE-PLACE-TRAIN-{wave}"
        population = site / f"SCENE-FACTORY-DIVERSE-PLACE-{wave}/COMPILED_POPULATION.private.json"
        labels = site / f"DIVERSE-PLACE-TRAIN-{wave}-LABELS-R4/MANIFEST.json"
        terminal = teacher / "RESULT.private.json"
        complete, sealed, scenes = (fixed(path, digest) for path, digest in
                                    ((terminal, terminal_sha), (labels, labels_sha), (population, pop_sha)))
        scene, = (row for row in scenes["clusters"] if row["id"] == origin)
        task, = (row for row in scenes["conditions"]["TRAIN"] if row["id"] == origin)
        record, = (row for row in complete["records"] if row["condition"] == origin)
        label, = (row for row in sealed["episodes"] if row["episode"] == origin)
        native_scene = scene["scene"]
        if (complete["exit_code"] != 0 or complete["unreconciled"] is not None or task["instruction"] != instruction
                or scene["split"] != "TRAIN" or scene["family"] != "F1" or native_scene["sha256"] in taken
                or sha(Path(native_scene["path"])) != native_scene["sha256"] or label["scene"] != native_scene["sha256"]
                or label["result"] != record["result"]["sha256"] or not record["admitted"]
                or not record["task_and_mobile_success"] or record["hard_STOP"] or record["technical_failure"]):
            raise ValueError("The new Recovery condition is not disjoint or lacks actual TRAIN admission")
        result = fixed(Path(record["result"]["path"]), record["result"]["sha256"])
        observed = result["detail"]["score"]["events"]
        if not {"target_grasp","target_lift20mm","correct_region","released","retreated50mm"} <= observed.keys():
            raise ValueError("The new scene requires genuine full teacher PLACE")
        reset, noise = seed(identity, "reset"), seed(identity, "noise")
        used_reset.update(row["reset"] for row in scenes["clusters"])
        if reset in used_reset or noise % 2**31 in used_noise:
            raise ValueError("Fresh reset or realized CUDA sequence overlaps original Recovery")
        taken.add(native_scene["sha256"])
        used_reset.add(reset)
        used_noise.add(noise % 2**31)
        cards.append(dict(id=identity, original_teacher=origin, task=instruction, scene_sha256=native_scene["sha256"],
                          scene_path=native_scene["path"], geometry=scene["scoring_geometry"], reset=reset,
                          policy_seed=noise, family=scene["family"], max_ticks=TICKS, maximum_queries=QUERIES))
        sources[identity] = dict(teacher_wave=str(teacher), terminal=ref(terminal), labels=ref(labels),
                                 compiled_population=ref(population))
    cohort = dict(plan=PLAN, split="TRAIN", cases=cards, protected_scene_sha256=sorted(blocked),
                  student_provider=first["student_provider"], checkpoint_sha256=first["checkpoint_sha256"],
                  prior_roster=ref(first_path), prior_terminal=ref(close_path), teacher_sources=sources,
                  protected_inputs_sha256=PROTECTED, previous_recovery_native_requests=2,
                  max_native_requests=2, max_native_ticks=2*TICKS,
                  max_native_requests_cumulative=20, actual_start_count=0, actual_native_integrations=0,
                  actual_recovery_admissions=0, exact_runtime_preflight_required=True)
    with output.open("x") as stream:
        json.dump(cohort, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("site", type=Path)
    cli.add_argument("output", type=Path)
    bind(**vars(cli.parse_args()))
