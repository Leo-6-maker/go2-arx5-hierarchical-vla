"""Stage fixed Scene Factory waves through the qualified single-world Teacher."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


PLAN = "90efbfb8785dbdf7fbb1219fbe179e049ec41e31"
PRIOR_CONFIG = "a07cde3a0b644609b6fed3b67c40c6aa0d66f2691b23121e8303180bcd566d19"
GEOMETRY = "f10b07348ad2746876a3e3ba1399a4b3b950ae5875c965f7c1d909d07b50df39"
TRAIN_GEOMETRY = "d36c9da25c6c927134fe578be1a378c60d3a7851ea9fa2b97e099e77eefe38e4"
F2_GEOMETRY = "0e99e023f4348139b8123d5b1c8b4a7da7f4acf2b48d2312f665dba913eb4621"
F2_REPLAY = "9d0d116a768cfb00f24b42e57e4cb0df1d7ef1e9cf139479ed3b1b6668c697a1"
F1_RED_LEGACY = "f0015701b5070b877dce21d2292cefb41b9aa13d17ba1e9fc669f7eaa6648f92"
DIVERSE_GEOMETRY = "838cd05519316ee5fc16d97a051d9a707e9203fdf4d452ba842f9885c8da723b"
DIVERSE_WAVE2_GEOMETRY = "25c15ab27d2fa528dca803432f7de2f4d0c3a423f13255c427da7099ad9bd6fe"
DIVERSE_WAVE3_GEOMETRY = "a6d6a35dd5f9fc15b6f587c17eed3bcb355b044b68776494a915d602546b301e"
DIVERSE_WAVE4_GEOMETRY = "7a9e350319006ce628a718f533de2dd520222dc8c3042a7b738697957762418e"
DIVERSE_WAVE5_GEOMETRY = "664aee826dcb87082ec6b26ef17d700dd07775afff60ce4a090cb12a6eca4d22"
DIVERSE_WAVE6_GEOMETRY = "7a5f0a2466cc139c589112b186bdfcd7f76c2fd6b610fc84b43359ea1a15b001"
DIVERSE_WAVE2_TERMINAL = "fd4ab7afadf92a45d097e638c23cc488a27d6916c890705be265513c9a62e3e0"
DIVERSE_WAVE3_TERMINAL = "6c86071bb00828c6a891be72f6be2c48aeea1b3f3d9a9ceb61306a7caa0526f1"
DIVERSE_WAVE4_TERMINAL = "ad40b4a7dbbd410061b861f64014f9a33bff35b74a15b0cf786b8d907aa7ec82"
DIVERSE_WAVE5_TERMINAL = "270991e71f8cc47171d40f05c31a3d61ad629c4cf9bface9767ca484ced20e2c"
DIVERSE_WAVE6_TERMINAL = "1d7c9a77f9d6ae3ef1d23dc07825f71ec50e68369c8a896cedf7e670a1cdbe42"
DIVERSE_WAVE6_LABELS = "1a7ff398ece8bf3ee79f3271b60937aca05361a907032dc4fc16c81614ddd0d9"
F2_ACTUAL_PREFIX = "bc56a16a451df2dc89aea9b725eabccc0f5be0b92442a71f69569d0635827aed"
DIVERSE_LEGACY = "05b145f759de77ff7cf00c82db6aee4fdafe21b826c8fa15a410e88c07ebf154"
DIVERSE_LEGACY_LEFT = "414fa12a17d0527c7ac91ff8eec6254b1714e8ecb00a447a0d57c284b1beebeb"
TICKS = 35703
PLACE_TICKS = 50703
IDS = ("EF5_Q_MOBILE_01", "EF5_Q_MOBILE_02")
REACH_IDS = ("EF5_Q_REACH_01", "EF5_Q_REACH_02")
REACH_REPAIR_IDS = ("EF5_Q_REACH_01_R2", "EF5_Q_REACH_02")
REACH_REPAIR_CASE = "b9ffb1cb26774ec7c6e766d82893aec05b2ffc5a0f769934bc1f651932580ea5"
REACH_REPAIR_WAVE = "847c78ede7dcb4b19c6786800007db4f1c0a6933db4c6d9b796a86760812fed8"
REACH_TRAIN_IDS = ("EF5_T_REACH_01", "EF5_T_REACH_02")
REACH_TRAIN_GEOMETRY = "1af08a4eb6fc5fac4a26630a378a5963fd2cd363bc29137ffbfe30508c7ca5f6"
REACH_WAVE2_IDS = ("EF5_T_REACH_03", "EF5_T_REACH_04")
REACH_WAVE2_GEOMETRY = "3de669760f801209edfb8d5f2641d9c2c7c653a00935ed7d51952cca2b4706ca"
REACH_WAVE3_IDS = ("EF5_T_REACH_05", "EF5_T_REACH_06", "EF5_T_REACH_07", "EF5_T_REACH_08")
REACH_WAVE3_GEOMETRY = "00597cce53f233e92d2fffd868768de8648411049fbbdcda7a6064b526cfe994"
REACH_WAVE1_TERMINAL = "64cc057a42622225bce85755605a556586df8c4afe7ec6651edc3322c9f731f2"
REACH_WAVE1_LABELS = "e5c542bc56ccdd5d0bb466110f8b80ea7eeb875a9aa6a8a3eeade895cfb815a8"
REACH_WAVE2_TERMINAL = "5951404110ac002393878b28f0aa878d054abf9436858c6b63e04a6c67e2577b"
REACH_WAVE2_LABELS = "376506d99de68ad342bd5ba5ebd8f40fe31251d5b98696a9fb526b094422fff2"
REACH_TRAIN_QUAL = "a2e61eae7f07ee2f06aab25bded6c59af55a9222bcc5c03f2e2e6344ac804f25"
REACH_TRAIN_READS = ("80dffe79a2c0789665170bd82a58ed1fee5adcfc2e76f6bfc9193380fef415f7",
                     "402f1bab61e1687c93cc38a04e5971352a5bafffa19cc6c2e5d2db344094eb1e")
REACH_TRAIN_SCOPES = ("REACH_TRAIN", "REACH_TRAIN_WAVE2", "REACH_TRAIN_WAVE3")
REACH_PREDECESSORS = {
    "REACH_TRAIN_WAVE2": (REACH_WAVE1_TERMINAL, REACH_WAVE1_LABELS, REACH_TRAIN_IDS),
    "REACH_TRAIN_WAVE3": (REACH_WAVE2_TERMINAL, REACH_WAVE2_LABELS, REACH_WAVE2_IDS),
}
REACH_SCOPES = ("REACH_QUAL", "REACH_QUAL_R2", *REACH_TRAIN_SCOPES)
FIRST_TRAIN = ("EF5_T_MOBILE_01", "EF5_T_MOBILE_05")
F1_RED_TRAIN = ("EF5_T_MOBILE_03", "EF5_T_MOBILE_07")
DIVERSE_PLACE = {
    "EF5_DIVERSE_PLACE_001": ("F1", "PLACE-red-right", "CHV1-TRAIN-F1-PLACE-red-right-02"),
    "EF5_DIVERSE_PLACE_002": ("F2", "PLACE-blue-right", "CHV1-TRAIN-F2-PLACE-blue-right-02"),
}
DIVERSE_PLACE_WAVE2 = {
    "EF5_DIVERSE_PLACE_003": ("F1", "PLACE-red-right", "CHV1-TRAIN-F1-PLACE-red-right-02"),
    "EF5_DIVERSE_PLACE_004": ("F2", "PLACE-red-left", "CHV1-TRAIN-F2-PLACE-red-left-02"),
}
DIVERSE_PLACE_WAVE3 = {
    "EF5_DIVERSE_PLACE_005": ("F1", "PLACE-blue-left", "MEV1-TRAIN-F1-NEAR_NEG-00-PLACE-blue-left"),
    "EF5_DIVERSE_PLACE_006": ("F2", "PLACE-red-left", "CHV1-TRAIN-F2-PLACE-red-left-02"),
}
DIVERSE_PLACE_WAVE4 = {
    "EF5_DIVERSE_PLACE_007": ("F1", "PLACE-blue-left", "MEV1-TRAIN-F1-NEAR_NEG-00-PLACE-blue-left"),
    "EF5_DIVERSE_PLACE_008": ("F1", "PLACE-red-right", "CHV1-TRAIN-F1-PLACE-red-right-02"),
}
DIVERSE_PLACE_WAVE5 = {
    "EF5_DIVERSE_PLACE_009": DIVERSE_PLACE_WAVE4["EF5_DIVERSE_PLACE_007"],
    "EF5_DIVERSE_PLACE_010": DIVERSE_PLACE_WAVE4["EF5_DIVERSE_PLACE_008"],
}
DIVERSE_PLACE_WAVE6 = {
    "EF5_DIVERSE_PLACE_011": DIVERSE_PLACE_WAVE4["EF5_DIVERSE_PLACE_007"],
    "EF5_DIVERSE_PLACE_012": DIVERSE_PLACE_WAVE4["EF5_DIVERSE_PLACE_008"],
}
DIVERSE_PLACE_WAVE7 = {"EF5_DIVERSE_PLACE_013": DIVERSE_PLACE_WAVE6["EF5_DIVERSE_PLACE_012"]}
PLACE_WAVES = {"DIVERSE_PLACE": DIVERSE_PLACE, "DIVERSE_PLACE_WAVE2": DIVERSE_PLACE_WAVE2,
               "DIVERSE_PLACE_WAVE3": DIVERSE_PLACE_WAVE3, "DIVERSE_PLACE_WAVE4": DIVERSE_PLACE_WAVE4,
               "DIVERSE_PLACE_WAVE5": DIVERSE_PLACE_WAVE5, "DIVERSE_PLACE_WAVE6": DIVERSE_PLACE_WAVE6,
               "DIVERSE_PLACE_WAVE7": DIVERSE_PLACE_WAVE7}
SCOPES = {"QUAL": ("QUAL", IDS, GEOMETRY), "REACH_QUAL": ("QUAL", REACH_IDS, GEOMETRY),
          "REACH_QUAL_R2": ("QUAL", REACH_REPAIR_IDS, GEOMETRY),
          "REACH_TRAIN": ("TRAIN", REACH_TRAIN_IDS, REACH_TRAIN_GEOMETRY),
          "REACH_TRAIN_WAVE2": ("TRAIN", REACH_WAVE2_IDS, REACH_WAVE2_GEOMETRY),
          "REACH_TRAIN_WAVE3": ("TRAIN", REACH_WAVE3_IDS, REACH_WAVE3_GEOMETRY),
          "TRAIN": ("TRAIN", FIRST_TRAIN, TRAIN_GEOMETRY),
          "F2_REPAIR": ("QUAL", ("EF5_Q_MOBILE_F2_RED_R2",), F2_GEOMETRY),
          "F1_RED_TRAIN": ("TRAIN", F1_RED_TRAIN, TRAIN_GEOMETRY),
          "DIVERSE_PLACE": ("TRAIN", tuple(DIVERSE_PLACE), DIVERSE_GEOMETRY),
          "DIVERSE_PLACE_WAVE2": ("TRAIN", tuple(DIVERSE_PLACE_WAVE2), DIVERSE_WAVE2_GEOMETRY),
          "DIVERSE_PLACE_WAVE3": ("TRAIN", tuple(DIVERSE_PLACE_WAVE3), DIVERSE_WAVE3_GEOMETRY),
          "DIVERSE_PLACE_WAVE4": ("TRAIN", tuple(DIVERSE_PLACE_WAVE4), DIVERSE_WAVE4_GEOMETRY),
          "DIVERSE_PLACE_WAVE5": ("TRAIN", tuple(DIVERSE_PLACE_WAVE5), DIVERSE_WAVE5_GEOMETRY),
          "DIVERSE_PLACE_WAVE6": ("TRAIN", tuple(DIVERSE_PLACE_WAVE6), DIVERSE_WAVE6_GEOMETRY),
          "DIVERSE_PLACE_WAVE7": ("TRAIN", tuple(DIVERSE_PLACE_WAVE7), None)}


def reach_repair_evidence(roster):
    group_pin = roster["first_reach_group"]
    failure_pin = roster["first_reach_failure"]
    group, failure = pinned(group_pin), pinned(failure_pin)
    record, = group["records"]
    if (group_pin["sha256"] != REACH_REPAIR_WAVE or failure_pin["sha256"] != REACH_REPAIR_CASE
            or group["paid_requests"] != 1 or record["condition"] != REACH_IDS[0]
            or record["result"]["sha256"] != REACH_REPAIR_CASE or not record["technical_failure"]
            or failure["actual_native_integrations"] != 5703 or failure["unsafe"] or not failure["technical_failure"]
            or not failure["shutdown_clean"] or failure["detail"] is not None
            or "ReachTeacher' object has no attribute 'landing'" not in failure["failure"]):
        raise ValueError("Only the preserved first Reach interface failure can precede this repair")


def wave7_evidence(roster):
    previous = pinned(roster["previous_place_result"])
    labels = pinned(roster["previous_place_labels"])
    sweep = pinned(roster["candidate_place_sweep"])
    stop, admitted = previous["records"]
    label, = labels["episodes"]
    if (roster["previous_place_result"]["sha256"] != DIVERSE_WAVE6_TERMINAL
            or roster["previous_place_labels"]["sha256"] != DIVERSE_WAVE6_LABELS
            or previous["paid_requests"] != 2 or previous["admitted_train"] != 1 or previous["exit_code"]
            or stop["condition"] != "EF5_DIVERSE_PLACE_011" or not stop["hard_STOP"] or stop["post_STOP_ticks"]
            or admitted["condition"] != "EF5_DIVERSE_PLACE_012" or not admitted["admitted"]
            or not admitted["task_and_mobile_success"] or admitted["technical_failure"] or admitted["hard_STOP"]
            or label["episode"] != admitted["condition"] or label["result"] != admitted["result"]["sha256"]
            or sweep["original_result_sha256"] != admitted["result"]["sha256"] or sweep["new_integrations"]
            or sweep["candidate"] != "EF5_DIVERSE_PLACE_013" or sweep["samples"] < 1000
            or min(sweep["minimum_clearance_mm"].values()) <= 0):
        raise ValueError("The final PLACE requires the original sealed success, retained STOP and clear fixed replay")
    return sweep["compiled_population_sha256"]


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def plain(path):
    return json.loads(Path(path).read_bytes())


def pinned(ref):
    path = Path(ref["path"])
    if digest(path) != ref["sha256"]:
        raise ValueError(f"Pinned input changed: {path.name}")
    return plain(path)


def ref(path):
    return dict(path=str(path), sha256=digest(path))


def write(path, data):
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def reach_train_evidence(roster):
    terminal = pinned(roster["reach_parent"])
    if (roster["reach_parent"]["sha256"] != REACH_TRAIN_QUAL or terminal["exit_code"] != 0
            or terminal["unreconciled"] is not None or terminal["paid_requests"] != 2
            or terminal["admitted_train"] != 0):
        raise ValueError("Reach TRAIN requires the original closed two-case qualification")
    for row, identity, read, expected in zip(terminal["records"], REACH_REPAIR_IDS, roster["reach_r4_readbacks"],
                                             REACH_TRAIN_READS, strict=True):
        native = pinned(row["result"])
        witness = pinned(read)
        if (row["condition"] != identity or read["sha256"] != expected or row["hard_STOP"]
                or row["technical_failure"] or not native["detail"]["reach_orientation_success"]
                or not native["shutdown_clean"] or witness["original_native_result_sha256"] != row["result"]["sha256"]
                or not witness["qualification_only"] or witness["train_admissions"] or witness["windows"] <= 0):
            raise ValueError("Reach TRAIN requires matching native success and R4 readback")


def reach_predecessor_evidence(roster, variant):
    previous, labels = roster["previous_reach_train"], roster["previous_reach_labels"]
    terminal = pinned(previous)
    batch = pinned(labels)
    terminal_sha, labels_sha, identities = REACH_PREDECESSORS[variant]
    if (previous["sha256"] != terminal_sha or labels["sha256"] != labels_sha
            or terminal["exit_code"] or terminal["unreconciled"] is not None
            or terminal["paid_requests"] != len(identities) or batch["wave"] != previous
            or batch["train"] != len(identities) or batch["rejected"]
            or tuple(entry["episode"]["id"] for entry in batch["episodes"]) != identities):
        raise ValueError("Reach TRAIN must follow the previous sealed native results")


def rows(population, binding):
    roster = pinned(binding)
    variant = roster.get("variant", roster.get("split", "QUAL"))
    split, identities, geometry = SCOPES[variant]
    if variant == "DIVERSE_PLACE_WAVE7":
        geometry = wave7_evidence(roster)
    ticks = PLACE_TICKS if variant in PLACE_WAVES else TICKS
    selected = population["conditions"][split]
    if (population["plan"] != PLAN or population["source_geometry_sha256"] != geometry
            or roster["plan"] != PLAN or roster["physics_requests"] != len(identities)
            or roster["integrations"] != len(identities)*ticks or roster["previously_qualified_worlds"] != 1
            or [x["id"] for x in selected] != list(identities) or set(roster["cards"]) != set(identities)
            or any(x["split"] != split or x["native_max_ticks"] != ticks for x in selected)):
        raise ValueError("Only the registered scene wave may run")
    by_id = {cluster["id"]: cluster["scene"]["sha256"] for cluster in population["clusters"]}
    if by_id != roster["cards"]:
        raise ValueError("Compiled scenes differ from the qualification allocation")
    if variant in REACH_SCOPES:
        if variant == "REACH_QUAL_R2":
            reach_repair_evidence(roster)
        if variant in REACH_TRAIN_SCOPES:
            reach_train_evidence(roster)
        if variant in REACH_PREDECESSORS:
            reach_predecessor_evidence(roster, variant)
        routes = (("F1", "HOLD-red"), ("F2", "HOLD-blue"))
        expected = tuple((name, *routes[i % 2]) for i, name in enumerate(identities))
        if tuple((r["id"], r["family"], r["instruction"]) for r in selected) != expected:
            raise ValueError("Only the preregistered Reach/Orient scenes have this transport route")
    elif variant in PLACE_WAVES:
        history = pinned(roster["qualification"])
        if roster["qualification"]["sha256"] != DIVERSE_LEGACY:
            raise ValueError("Diverse PLACE requires the original consumed Teacher archive")
        approved = {r["episode"]: r for r in history["episodes"]}
        if variant in ("DIVERSE_PLACE_WAVE2", "DIVERSE_PLACE_WAVE3"):
            left = pinned(roster["left_qualification"])
            if roster["left_qualification"]["sha256"] != DIVERSE_LEGACY_LEFT:
                raise ValueError("The second wave requires the original complete F2 red-left witness")
            for admission in left["admissions"]:
                episode = admission["episode"]
                if episode["split"] == "TRAIN" and episode["individual_pass"]:
                    approved[episode["id"]] = admission
        for row in selected:
            family, instruction, witness = PLACE_WAVES[variant][row["id"]]
            proof = approved[witness]
            if ((row["family"], row["instruction"]) != (family, instruction) or proof["task"] != instruction
                    or not all(proof["phase_windows"].get(phase, 0) > 0 for phase in ("CARRY", "LOWER", "OPEN", "WITHDRAW"))):
                raise ValueError("Diverse PLACE requires a complete same-family task and actual release support")
        if variant == "DIVERSE_PLACE_WAVE3":
            terminal = pinned(roster["previous_place_result"])
            failed, = (r for r in terminal["records"] if r["condition"] == "EF5_DIVERSE_PLACE_004")
            if (roster["previous_place_result"]["sha256"] != DIVERSE_WAVE2_TERMINAL
                    or failed["first_failure"] != "ILLEGAL_CONTACT" or failed["post_STOP_ticks"] != 0):
                raise ValueError("The archived F2 STOP must remain complete and in the denominator")
        if variant == "DIVERSE_PLACE_WAVE4":
            terminal = pinned(roster["previous_place_result"])
            passed, failed = terminal["records"]
            prefix = pinned(roster["stop_prefix"])
            if (roster["previous_place_result"]["sha256"] != DIVERSE_WAVE3_TERMINAL
                    or passed["condition"] != "EF5_DIVERSE_PLACE_005" or not passed["task_and_mobile_success"]
                    or failed["condition"] != "EF5_DIVERSE_PLACE_006"
                    or failed["first_failure"] != "ILLEGAL_CONTACT" or failed["post_STOP_ticks"] != 0
                    or roster["stop_prefix"]["sha256"] != F2_ACTUAL_PREFIX
                    or prefix[-1]["condition"] != failed["condition"]
                    or prefix[-1]["terminal_tick"] != 18846):
                raise ValueError("F1-only production requires the closed preceding wave and actual F2 prefix")
        if variant == "DIVERSE_PLACE_WAVE5":
            terminal = pinned(roster["previous_place_result"])
            preceding = terminal["records"]
            if (roster["previous_place_result"]["sha256"] != DIVERSE_WAVE4_TERMINAL
                    or terminal["paid_requests"] != 2 or terminal["exit_code"] != 0
                    or [row["condition"] for row in preceding] != list(DIVERSE_PLACE_WAVE4)
                    or any(not row["task_and_mobile_success"] or row["hard_STOP"] or row["technical_failure"]
                           for row in preceding)):
                raise ValueError("The new F1 layouts require both original successful terminal records")
        if variant == "DIVERSE_PLACE_WAVE6":
            terminal = pinned(roster["previous_place_result"])
            stopped, admitted = terminal["records"]
            if (roster["previous_place_result"]["sha256"] != DIVERSE_WAVE5_TERMINAL
                    or terminal["paid_requests"] != 2 or terminal["exit_code"] != 0
                    or [row["condition"] for row in (stopped, admitted)] != list(DIVERSE_PLACE_WAVE5)
                    or stopped["first_failure"] != "ILLEGAL_CONTACT" or not stopped["hard_STOP"]
                    or stopped["post_STOP_ticks"] != 0 or any(row["technical_failure"] for row in (stopped, admitted))
                    or not admitted["admitted"] or not admitted["task_and_mobile_success"] or admitted["hard_STOP"]):
                raise ValueError("Keep the preceding F1 contact failure and admitted red PLACE")
    elif split == "TRAIN":
        confirmation = pinned(roster["qualification"])
        expected, instruction = IDS[0], "HOLD-blue"
        if variant == "F1_RED_TRAIN":
            if roster["qualification"]["sha256"] != F1_RED_LEGACY or confirmation["paid_requests"] != 1:
                raise ValueError("The closed F1 red mobile-grasp episode is required")
            expected, instruction = "CHV1-TRAIN-F1-HOLD-red-19", "HOLD-red"
        matches = [r for r in confirmation["records"] if r["condition"] == expected]
        if (len(matches) != 1 or not matches[0]["task_and_mobile_success"]
                or matches[0]["hard_STOP"] or matches[0]["technical_failure"]
                or matches[0]["post_STOP_ticks"] not in (None, 0)
                or any((x["family"], x["instruction"]) != ("F1", instruction) for x in selected)):
            raise ValueError("TRAIN requires a closed successful same-family, same-target mobile-grasp qualification")
    if variant == "F2_REPAIR":
        proof = pinned(roster["repair"])
        confirmation = pinned(roster["qualification"])
        previous, = (r for r in confirmation["records"] if r["condition"] == IDS[1])
        if (roster["repair"]["sha256"] != F2_REPLAY or previous["result"]["sha256"] != proof["result_sha256"]
                or not previous["hard_STOP"] or previous["technical_failure"] or previous["post_STOP_ticks"] != 0
                or (selected[0]["family"], selected[0]["instruction"]) != ("F2", "HOLD-red")):
            raise ValueError("The new F2 case is limited to the evidenced distractor displacement")
    return selected


def capacity_file(root):
    if root.name in ("REACH-TRAIN-03", "REACH-TRAIN-OVERLAY-03"):
        return root.parent / "REACH-TRAIN-CAPACITY-03.json"
    if root.name in ("REACH-TRAIN-02", "REACH-TRAIN-OVERLAY-02"):
        return root.parent / "REACH-TRAIN-CAPACITY-02.json"
    if root.name in ("REACH-TRAIN-01", "REACH-TRAIN-OVERLAY-01"):
        return root.parent / "REACH-TRAIN-CAPACITY-01.json"
    if root.name in ("REACH-QUAL-02", "REACH-QUAL-OVERLAY-02"):
        return root.parent / "REACH-QUAL-CAPACITY-02.json"
    if root.name in ("REACH-QUAL-01", "REACH-QUAL-OVERLAY-01"):
        return root.parent / "REACH-QUAL-CAPACITY-01.json"
    if root.name in ("DIVERSE-PLACE-TRAIN-07", "DIVERSE-PLACE-OVERLAY-07"):
        return root.parent / "DIVERSE-PLACE-CAPACITY-07.json"
    if root.name in ("DIVERSE-PLACE-TRAIN-06", "DIVERSE-PLACE-OVERLAY-06"):
        return root.parent / "DIVERSE-PLACE-CAPACITY-06.json"
    if root.name in ("DIVERSE-PLACE-TRAIN-05", "DIVERSE-PLACE-OVERLAY-05"):
        return root.parent / "DIVERSE-PLACE-CAPACITY-05.json"
    if root.name in ("DIVERSE-PLACE-TRAIN-04", "DIVERSE-PLACE-OVERLAY-04"):
        return root.parent / "DIVERSE-PLACE-CAPACITY-04.json"
    if root.name in ("DIVERSE-PLACE-TRAIN-03", "DIVERSE-PLACE-OVERLAY-03"):
        return root.parent / "DIVERSE-PLACE-CAPACITY-03.json"
    if root.name in ("DIVERSE-PLACE-TRAIN-02", "DIVERSE-PLACE-OVERLAY-02"):
        return root.parent / "DIVERSE-PLACE-CAPACITY-02.json"
    name = "DIVERSE-PLACE-CAPACITY.json" if root.name.startswith("DIVERSE-PLACE-") else "QUAL-CAPACITY.json"
    return root.parent / name


def capacity(root):
    planned = plain(capacity_file(root))
    memory = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                      if line.startswith("MemAvailable:"))) * 1024
    return shutil.disk_usage(root).free > planned["upcoming_writes_bytes"] and memory >= 13*1024**3


def build_population(compiled, old, variant):
    clusters = {x["id"]: x for x in compiled["clusters"]}
    historical = {x["scene"]["sha256"] for x in old["clusters"]}
    resets = {x["reset"] for x in old["clusters"]}
    cards = []
    split, identities, geometry = SCOPES[variant]
    place = variant in PLACE_WAVES
    skill = "DIVERSE_PLACE" if place else "REACH_ORIENT" if variant in REACH_SCOPES else "MOBILE_GRASP"
    source_ids = REACH_IDS if variant == "REACH_QUAL_R2" else identities
    selected_clusters = []
    for name, source_id in zip(identities, source_ids, strict=True):
        q, = (x for x in compiled["conditions"][split] if x["id"] == source_id)
        cluster = dict(clusters[source_id], id=name)
        if (q["skill"] != skill
                or q["physics_eligible"] or not q["original_2ms"]
                or cluster["scene"]["sha256"] in historical or cluster["reset"] in resets
                or digest(cluster["scene"]["path"]) != cluster["scene"]["sha256"]):
            raise ValueError("Unexpected mobile pilot or geometry overlaps historical data")
        instruction = q["instruction"] if place else f"HOLD-{q['target']}"
        original = next(x for x in old["conditions"]["TRAIN"]
                        if x["family"] == q["family"] and x["instruction"] == instruction)
        cards.append(dict(original, id=name, condition_id=name, split=split, instruction=instruction,
                          cluster=name, scene_family=name, reset=q["reset"], reset_seed=q["reset"],
                          scene_seed=q["reset"], native_max_ticks=PLACE_TICKS if place else TICKS))
        selected_clusters.append(cluster)
    return dict(plan=PLAN, source_geometry_sha256=geometry, clusters=selected_clusters,
                conditions={split: cards})


def patch(anchor, path, expected, edits):
    if digest(anchor) != expected:
        raise ValueError(f"Qualified launcher bytes changed: {anchor.name}")
    text = anchor.read_text()
    for before, after in edits:
        if text.count(before) != 1:
            raise ValueError(f"Expected one dispatch match for {anchor.name}")
        text = text.replace(before, after)
    text = text.replace("v5_legacy_single", "v62_mobile_qualification")
    compile(text, str(path), "exec")
    path.unlink()
    path.write_text(text)


def stage(prior, compiled, original_output, destination, root, revision, qualification=None, repair=None,
          legacy_red_train=False, place_catalog=None, place_left_proof=None, previous_place_wave=None,
          stop_prefix=None, after_clean_place_wave=False, after_mixed_place_wave=False,
          final_place_labels=None, final_place_sweep=None, reach_qual=False, reach_repair=None,
          reach_train=None, reach_readbacks=(), previous_reach_train=None, previous_reach_labels=None,
          reach_wave3=False):
    if repair and not qualification:
        raise ValueError("Read the original F2 terminal before staging a repair")
    if legacy_red_train and (qualification or repair):
        raise ValueError("F1 red uses only the preserved legacy qualification")
    if place_catalog and (qualification or repair or legacy_red_train):
        raise ValueError("Diverse PLACE requires its own legacy task witness")
    if place_left_proof and not place_catalog:
        raise ValueError("The second PLACE wave also requires the original right-placement witness")
    if after_clean_place_wave and (not (previous_place_wave and place_catalog) or stop_prefix or place_left_proof):
        raise ValueError("The fifth wave uses the sealed two-card F1 terminal")
    if after_mixed_place_wave and (not (previous_place_wave and place_catalog)
            or after_clean_place_wave or stop_prefix or place_left_proof):
        raise ValueError("The sixth wave uses the sealed fifth-wave results")
    final_place = final_place_labels is not None or final_place_sweep is not None
    if reach_train and any((reach_qual, reach_repair, qualification, repair, legacy_red_train, place_catalog,
                            previous_place_wave, final_place)):
        raise ValueError("Use the independent Reach TRAIN route with its qualification receipts")
    if bool(reach_train) != bool(reach_readbacks):
        raise ValueError("Reach TRAIN requires both R4 qualification readbacks")
    reach_successor = previous_reach_train is not None
    if reach_successor != bool(previous_reach_labels) or ((reach_successor or reach_wave3) and not reach_train):
        raise ValueError("Reach successors require the closed prior wave and R4 labels")
    if reach_wave3 and not reach_successor:
        raise ValueError("The final Reach wave must follow the closed second wave")
    if reach_qual and any((qualification, repair, legacy_red_train, place_catalog, previous_place_wave, final_place)):
        raise ValueError("Use the original independent Reach/Orient qualification alone")
    if reach_repair is not None and not reach_qual:
        raise ValueError("A Reach repair must use the Reach qualification route")
    if final_place and (not (final_place_labels and final_place_sweep and place_catalog and previous_place_wave)
                        or any((place_left_proof, stop_prefix, after_clean_place_wave, after_mixed_place_wave))):
        raise ValueError("The final PLACE requires its own fixed replay, prior admission and closed wave")
    if previous_place_wave and (not (place_left_proof or stop_prefix or after_clean_place_wave or after_mixed_place_wave or final_place)
            or plain(previous_place_wave / "control/LEASE.json")["status"] != "CLOSED"):
        raise ValueError("The successor wave requires closed production and its earlier evidence")
    if stop_prefix and (not (place_catalog and previous_place_wave) or place_left_proof):
        raise ValueError("The fourth wave requires the previous terminal and the two actual contact prefixes")
    variant = ("REACH_TRAIN_WAVE3" if reach_wave3 else "REACH_TRAIN_WAVE2" if reach_successor else
               "REACH_TRAIN" if reach_train else
               "REACH_QUAL_R2" if reach_repair else "REACH_QUAL" if reach_qual else
               "DIVERSE_PLACE_WAVE7" if final_place else "DIVERSE_PLACE_WAVE6" if after_mixed_place_wave else
               "DIVERSE_PLACE_WAVE5" if after_clean_place_wave else
               "DIVERSE_PLACE_WAVE4" if stop_prefix else
               "DIVERSE_PLACE_WAVE3" if previous_place_wave else
               "DIVERSE_PLACE_WAVE2" if place_left_proof else "DIVERSE_PLACE" if place_catalog else
               "F1_RED_TRAIN" if legacy_red_train else
               "F2_REPAIR" if repair else "TRAIN" if qualification else "QUAL")
    split, identities, geometry = SCOPES[variant]
    final_refs = {}
    if reach_train:
        parents = (reach_train, *(reach_train / "cases" / name for name in REACH_REPAIR_IDS))
        if any(plain(folder / "control/LEASE.json")["status"] != "CLOSED" for folder in parents):
            raise ValueError("Original Reach qualification leases must be closed")
        final_refs = dict(reach_parent=ref(reach_train / "RESULT.private.json"),
                          reach_r4_readbacks=[ref(path) for path in reach_readbacks])
        reach_train_evidence(final_refs)
        if reach_successor:
            if plain(previous_reach_train / "control/LEASE.json")["status"] != "CLOSED":
                raise ValueError("The prior Reach production lease must be closed")
            final_refs.update(previous_reach_train=ref(previous_reach_train / "RESULT.private.json"),
                              previous_reach_labels=ref(previous_reach_labels))
            reach_predecessor_evidence(final_refs, variant)
    if reach_repair:
        failed = reach_repair / "cases" / REACH_IDS[0]
        if any(plain(folder / "control/LEASE.json")["status"] != "CLOSED" for folder in (reach_repair, failed)):
            raise ValueError("Require a closed original Reach driver and case")
        final_refs = dict(first_reach_group=ref(reach_repair / "RESULT.private.json"),
                          first_reach_failure=ref(failed / "RESULT.private.json"))
        reach_repair_evidence(final_refs)
    if final_place:
        final_refs = dict(previous_place_result=ref(previous_place_wave / "RESULT.private.json"),
                          previous_place_labels=ref(final_place_labels), candidate_place_sweep=ref(final_place_sweep))
        geometry = wave7_evidence(final_refs)
    ticks = PLACE_TICKS if place_catalog else TICKS
    if digest(prior / "CONFIG.json") != PRIOR_CONFIG or digest(compiled) != geometry:
        raise ValueError("The original single world launcher and frozen scene compiler are required")
    history = plain(original_output / "RESULT.private.json")
    if (history["paid_requests"] != 1 or history["exit_code"] != 0
            or plain(original_output / "control/LEASE.json")["status"] != "CLOSED"):
        raise ValueError("Read the real previously successful one-world control first")
    old = plain(prior / "CONFIG.json")
    old_source = Path(old["source_root"])
    source = destination / "source"
    symbols = pinned(old["source_members"])
    population = build_population(plain(compiled), pinned(old["population"]), variant)
    if final_place:
        population["source_geometry_sha256"] = geometry
    destination.mkdir(mode=0o700)
    shutil.copytree(old_source, source, copy_function=os.link,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    scope = "scripts/sla/"
    overlays = {
        "core_hopper_register.py": [
            ("    if config['population']['sha256']!=POP:", "    selection=config.get('v5_legacy_single')\n"
             "    if not selection and config['population']['sha256']!=POP:"),
            ("    pop=read_pin(config['population']); rows=order(pop)\n    selection=config.get('v5_legacy_single')",
             "    pop=read_pin(config['population']); rows=[] if selection else order(pop)"),
            ("compiled_population_sha256=POP,hardware_allowed=False",
             "compiled_population_sha256=config['population']['sha256'],hardware_allowed=False"),
            ("population_sha256=POP,plan=PLAN", "population_sha256=config['population']['sha256'],plan=PLAN"),
        ],
        "core_hopper_expert.py": [
            ("(1, 35703) if binding else (264, 12065592)",
             f"({len(identities)}, {len(identities)*ticks}) if binding else (264, 12065592)"),
            ("    if (ref['sha256'] != POP or authority['compiled_population_sha256'] != POP",
             "    if ((not binding and ref['sha256'] != POP) or authority['compiled_population_sha256'] != ref['sha256']"),
            ("    selected = order(population)\n", "    selected = [] if binding else order(population)\n"),
        ],
        "core_hopper_collection.py": [
            ("scheduled, capacity_for_scope = order(pop), capacity", "scheduled, capacity_for_scope = "
             "([] if selection else order(pop)), capacity"),
            ("if outcome['task_and_mobile_success'] and outcome['completed'] and not outcome['hard_STOP']:",
             "if row['split'] != 'QUAL' and outcome['task_and_mobile_success'] and outcome['completed'] and not outcome['hard_STOP']:"),
        ],
    }
    for name, edits in overlays.items():
        relative = scope + name
        patch(old_source / relative, source / relative, symbols[relative], edits)
        symbols[relative] = digest(source / relative)
    if variant in REACH_SCOPES:
        model_scope = "src/go2arx5_vla/world_model/"
        reach_edits = {
            "multitask_scope_v6.py": [
                ("def episode(adapter):\n",
                 "def episode(adapter):\n"
                 "    binding = adapter.spec.get('v62_mobile_qualification')\n"
                 "    if binding and read_pin(binding).get('variant') in " + repr(REACH_SCOPES) + ":\n"
                 "        from scripts.sla.v64_reach_qual import episode as reach\n"
                 "        return reach(adapter)\n"),
            ],
            "multitask_teacher_episode_v6.py": [
                ("def episode(adapter, *, placement_teacher=None):",
                 "def episode(adapter, *, placement_teacher=None, reach_qualification=None, capture_language=None):"),
                ("    if adapter.spec['scope'] == RELEASE_SCOPE:",
                 "    if reach_qualification is not None:\n"
                 "        teacher, score = reach_qualification(initial)\n"
                 "    if adapter.spec['scope'] == RELEASE_SCOPE:"),
                ("    if placement_teacher is None and teacher.landing is not None:",
                 "    if reach_qualification is None and placement_teacher is None and teacher.landing is not None:"),
                ("writer = LiveWriter(adapter,frame,camera_factory=TwoObjectCameraSession)",
                 "writer = LiveWriter(adapter,frame,camera_factory=TwoObjectCameraSession,language=capture_language)"),
            ],
            "mfree_capture_v1.py": [
                ("def __init__(self, adapter, frame, *, camera_factory=LiveCamera):",
                 "def __init__(self, adapter, frame, *, camera_factory=LiveCamera, language=None):\n"
                 "        self.language = adapter.spec['language'] if language is None else language"),
                ("language=self.adapter.spec['language'],", "language=self.language,"),
            ],
        }
        for name, edits in reach_edits.items():
            relative = model_scope + name
            patch(old_source / relative, source / relative, symbols[relative], edits)
            symbols[relative] = digest(source / relative)
        relative = scope + "v64_reach_qual.py"
        shutil.copy2(Path(__file__).with_name("v64_reach_qual.py"), source / relative)
        symbols[relative] = digest(source / relative)
    relative = scope + Path(__file__).name
    shutil.copy2(__file__, source / relative)
    symbols[relative] = digest(source / relative)
    pop_file = destination / "POPULATION.json"
    registry_file = destination / "ROSTER.json"
    source_file = destination / "SOURCE_MEMBERS.json"
    write(pop_file, population)
    registered = dict(plan=PLAN, previously_qualified_worlds=1, physics_requests=len(identities), integrations=len(identities)*ticks,
                      cards={c["id"]: c["scene"]["sha256"] for c in population["clusters"]})
    if qualification:
        result = qualification / "RESULT.private.json"
        if (plain(qualification / "control/LEASE.json")["status"] != "CLOSED"
                or plain(result)["paid_requests"] != 2 or plain(result)["exit_code"] != 0):
            raise ValueError("TRAIN requires the original mobile qualification wave to close")
        registered.update(split=split, qualification=ref(result))
    if repair:
        registered.update(variant=variant, repair=ref(repair))
    if variant in REACH_SCOPES:
        registered.update(variant=variant, split=split)
    if legacy_red_train:
        registered.update(variant=variant, split=split, qualification=ref(original_output / "RESULT.private.json"))
    if place_catalog:
        registered.update(variant=variant, split=split, qualification=ref(place_catalog))
    if place_left_proof:
        registered["left_qualification"] = ref(place_left_proof)
    if previous_place_wave:
        registered["previous_place_result"] = ref(previous_place_wave / "RESULT.private.json")
    if stop_prefix:
        registered["stop_prefix"] = ref(stop_prefix)
    registered.update(final_refs)
    write(registry_file, registered)
    write(source_file, symbols)
    use = shutil.disk_usage(root.parent)
    demonstrated = int(sum(p.stat().st_blocks*512 for p in original_output.rglob("*") if p.is_file()))
    planned = demonstrated * 3 * max(2, len(identities))
    if place_catalog:
        require_ticks = history["verified_ticks"]
        if not 0 < require_ticks <= TICKS:
            raise ValueError("No valid native byte-rate witness for the next wave")
        planned = (planned * ticks + require_ticks - 1) // require_ticks
    if use.free <= planned:
        raise OSError("Insufficient real capacity for both world outputs and complete terminal seal")
    with tempfile.TemporaryFile(dir=root.parent) as f:
        os.posix_fallocate(f.fileno(), 0, 64*1024**2)
        os.fsync(f.fileno())
    reservation = capacity_file(root)
    if variant in ("QUAL", *REACH_SCOPES) or variant in PLACE_WAVES:
        write(reservation, dict(free_bytes=use.free, previous_trace_bytes=demonstrated,
              upcoming_writes_bytes=planned, writable_probe_bytes=64*1024**2, device=root.parent.stat().st_dev))
    elif plain(reservation)["upcoming_writes_bytes"] != planned:
        raise ValueError("The same native output reservation must cover this two-card TRAIN wave")
    config = dict(old, root=str(root), source_root=str(source), source_sha=revision,
                  population=ref(pop_file), source_members=ref(source_file))
    del config["v5_legacy_single"]
    config["v62_mobile_qualification"] = ref(registry_file)
    write(destination / "CONFIG.json", config)
    sys.path[:0] = [str(source), str(source / "src")]
    from scripts.sla.core_hopper_register import prepare

    prepare(dict(config, root=str(destination / "CPU-PREFLIGHT")), execute=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prior", "compiled", "original_output", "destination", "root"):
        parser.add_argument(name, type=Path)
    parser.add_argument("revision")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--repair", type=Path)
    parser.add_argument("--legacy-red-train", action="store_true")
    parser.add_argument("--place-catalog", type=Path)
    parser.add_argument("--place-left-proof", type=Path)
    parser.add_argument("--previous-place-wave", type=Path)
    parser.add_argument("--stop-prefix", type=Path)
    parser.add_argument("--after-clean-place-wave", action="store_true")
    parser.add_argument("--after-mixed-place-wave", action="store_true")
    parser.add_argument("--final-place-labels", type=Path)
    parser.add_argument("--final-place-sweep", type=Path)
    parser.add_argument("--reach-qual", action="store_true")
    parser.add_argument("--reach-repair", type=Path)
    parser.add_argument("--reach-train", type=Path)
    parser.add_argument("--reach-readbacks", type=Path, nargs=2)
    parser.add_argument("--previous-reach-train", type=Path)
    parser.add_argument("--previous-reach-labels", type=Path)
    parser.add_argument("--reach-wave3", action="store_true")
    stage(**vars(parser.parse_args()))
