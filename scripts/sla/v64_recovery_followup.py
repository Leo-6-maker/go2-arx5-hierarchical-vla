"""Seal the second independent Recovery pair from two actual TRAIN PLACE waves."""
import argparse
from pathlib import Path

from v64_recovery_roster import PLAN, PROTECTED, QUERIES, TICKS, fixed, seed, sha


FIRST_ROSTER = "e9b7552d4c10779f4c87a04ef23c252ee2f599a69a24df5d8f94624166ad6cba"
FIRST_RESULT = "c59b8c4118397c31c9d05eb45e7a7749e30dcb7fb6b3dc04d9df97015cf0aba2"
SECOND_RESULT = "9dd1cf7bca7ea087bebb183c2a2d117fb9dbb8d87b890ca85c3357da1b1c4e2c"
OPEN_REPLAY = "bfda1426f0e91c6cbc8efe81df2ab3c590973152c034582d404a11dff805de49"
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
OPEN_FOLLOWUP = (
    ("V64_RECOVERY_TRAIN_05", "EF5_DIVERSE_PLACE_001", "PLACE-red-right", "01",
     "838cd05519316ee5fc16d97a051d9a707e9203fdf4d452ba842f9885c8da723b",
     "9a81a8fa1f9f8898424df702c29fa44f2722fe82a93f17e5d1bb4ad425103f3d",
     "6c235b50fb7595e27949c5e11d2ed71a3c23b4855ff5cf051155a42da519a5b8"),
    ("V64_RECOVERY_TRAIN_06", "EF5_DIVERSE_PLACE_012", "PLACE-red-right", "06",
     "7a5f0a2466cc139c589112b186bdfcd7f76c2fd6b610fc84b43359ea1a15b001",
     "1d7c9a77f9d6ae3ef1d23dc07825f71ec50e68369c8a896cedf7e670a1cdbe42",
     "1a7ff398ece8bf3ee79f3271b60937aca05361a907032dc4fc16c81614ddd0d9"),
)
OPEN_ROSTER = tuple(row[:3] for row in OPEN_FOLLOWUP)


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def bind(site, output, *, prior_open_replay=None):
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
    followup = FOLLOWUP
    prior_cases = list(first["cases"])
    history = {}
    if prior_open_replay is not None:
        from v64_recovery_cohorts import SECOND_ROSTER

        second_path = site / "RECOVERY-TRAIN-ROSTER-02.json"
        second_close = site / "RECOVERY-TRAIN-SEALED-02/RESULT.json"
        second = fixed(second_path, SECOND_ROSTER)
        sealed = fixed(second_close, SECOND_RESULT)
        replay = fixed(prior_open_replay, OPEN_REPLAY)
        if (sealed["terminal_cases"] != 2 or not sealed["complete_reconciliation"]
                or sealed["denominator"] != 2 or sealed["train_recovery_count"] != 1
                or any(row["state"] != "CLOSED" for row in sealed["cases"])
                or replay["new_native_integrations"] or replay["new_model_forwards"]
                or len(replay["cases"]) != 4 or not any(row["reached"] for row in replay["cases"])):
            raise ValueError("Require the closed second pair and actual supported OPEN replay")
        followup = OPEN_FOLLOWUP
        prior_cases.extend(second["cases"])
        history = dict(handoff_protocol="CARRY_OR_SUPPORTED_OPEN", supported_open_replay=ref(prior_open_replay),
                       second_prior_roster=ref(second_path), second_prior_terminal=ref(second_close))
    taken = blocked | {row["scene_sha256"] for row in prior_cases}
    used_reset = {row["reset"] for row in protected} | {row["reset"] for row in prior_cases}
    used_noise = {row["policy_seed"] % 2**31 for row in prior_cases}
    cards, sources = [], {}
    for identity, origin, instruction, wave, pop_sha, terminal_sha, labels_sha in followup:
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
                  protected_inputs_sha256=PROTECTED, previous_recovery_native_requests=len(prior_cases),
                  max_native_requests=2, max_native_ticks=2*TICKS,
                  max_native_requests_cumulative=20, actual_start_count=0, actual_native_integrations=0,
                  actual_recovery_admissions=0, exact_runtime_preflight_required=True, **history)
    with output.open("x") as stream:
        json.dump(cohort, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("site", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--prior-open-replay", type=Path)
    bind(**vars(cli.parse_args()))
