"""Bind the four V4 long-run checkpoints to the paired original L1 native panel."""

import argparse
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require, sha
from v4_long_eval_noise import REGISTRY_SHA256, STARTS
from v4_long_export import FORMAL_ROLES, SCHEMA


PLAN = "f72f3d57b470de3e62821cd5e1a33d938a669cb7"
ROLES = {"PARENT": 3800, "M1000": 1000, "T1000": 1000, "M2000": 2000}
MAIN = "502d890d3ff314ac42f4c79c360942b9b626bf1fcc4b6fc488fbab15ba508361"


def manifest(ref, original, role):
    folder = Path(ref["path"])
    values = read(pin(folder / "CHECKPOINT.json", ref["sha256"]))
    require(original["checkpoint_sha256"] == MAIN and role in ROLES, "Wrong frozen MAIN parent for this runtime")
    if role == "PARENT":
        require(values == original and values["phase"] == ROLES[role], "Use the exact original MAIN package")
        return values
    policy_role = "TARGET_CONTROL_1000_V4" if role == "T1000" else FORMAL_ROLES[ROLES[role]]
    require(values["schema"] == SCHEMA and values["formal_inference_eligible"] and values["role"] == policy_role
            and values["phase"] == ROLES[role] and values["previous_checkpoint_sha256"] == MAIN
            and values["checkpoint_sha256"] != MAIN and values["source_codec"] == "DENSE_REFERENCE_G7_V6"
            and values["frozen_wbc"] and values["original_l1"] and not values["hardware_allowed"],
            "Only the fixed complete V4 trained checkpoint is eligible")
    require(all(values[key] == original[key] for key in ("foundation_sha256", "frozen_model",
                "action_sha256", "state_sha256", "tokenizer_sha256", "partition_sha256")),
            "V4 changed the inherited Go2 observation, statistics, or parameter inventory")
    for old, new in zip(original["arrays"], values["arrays"], strict=True):
        require(all(old[key] == new[key] for key in ("file", "key", "shape", "dtype")),
                "V4 learned weights have a changed Go2 provider ABI")
    return values


def schedule(registry, receipt, place_noise, hold_noise, study, v2_registry, qualification, role):
    from m38_a_native_v1 import PLACE_MAX_QUERIES, PLACE_MAX_TICKS, PLACE_MAX_WALL_S
    from v2_m38_confirmation_student import confirmation_schedule

    rows = [row for row in read(pin(registry, REGISTRY_SHA256))["conditions"]
            if row["block"] == "LONG_DEV6" and row["model"] == role]
    require([(r["slot"], r.get("noise_label"), r.get("archived_noise_alias")) for r in rows]
            == [("P01", 701, None), ("P02", 702, None), ("P02", 701, None), ("P01", 702, None),
                ("N02", None, "N02_A"), ("N02", None, "N02_B")],
            "Expected the original finite matched V4 roster")
    bank = read(place_noise)
    hold = read(pin(hold_noise, bank["historical_hold_sha256"]))
    require(bank["stage"] == "LONG_DEV6_PLACE" and bank["registry_sha256"] == REGISTRY_SHA256
            and bank["labels"] == {str(k): v for k, v in STARTS.items()}
            and bank["old_gpu_manifest_sha256"] == hold["old_gpu_manifest_sha256"]
            and len({row["gpu_sha256"] for row in bank["tensors"]}) == 320,
            "Realized paired CUDA inputs changed")
    baseline = confirmation_schedule(v2_registry, study, receipt, hold_noise, qualification)
    archived = {row["noise_label"]: row for row in baseline if row["layout_slot"] == "N02" and row["enabled"]}
    require(set(archived) == {201, 202}, "Both original N02 holds must be independently qualified")
    template, = [row for row in receipt["schedule"] if row["task"] == "PLACE_RED_RIGHT" and row["link"] == "L1"]
    places = [dict(template, id=r["id"], pair_id="LONG_DEV6_PLACE", within_pair_order=i + 1,
                   layout_slot=r["slot"], policy_seed=bank["labels"][str(r["noise_label"])],
                   noise_label=r["noise_label"], max_ticks=PLACE_MAX_TICKS,
                   policy_queries=PLACE_MAX_QUERIES, wall_limit_s=PLACE_MAX_WALL_S)
              for i, r in enumerate(rows[:4])]
    holds = [dict(archived[label], id=r["id"], pair_id="LONG_DEV6_HOLD", within_pair_order=i + 1)
             for i, (label, r) in enumerate(zip((201, 202), rows[4:], strict=True))]
    return places, holds


if __name__ == "__main__":
    from scripts.pi0292.durable import write

    tool = argparse.ArgumentParser(description=__doc__)
    for field in ("registry", "receipt", "place_noise", "hold_noise", "study", "v2_registry", "candidate",
                  "original", "destination"):
        tool.add_argument(field, type=Path)
    tool.add_argument("--role", required=True, choices=ROLES)
    a = tool.parse_args()
    original = read(a.original / "CHECKPOINT.json")
    require(not a.destination.exists(), "Keep the first sealed CPU panel binding")
    ref = dict(path=str(a.candidate), sha256=sha(a.candidate / "CHECKPOINT.json"))
    state = manifest(ref, original, a.role)
    qualified = a.study / "Q_CONFIRM_ACTUAL.json"
    places, holds = schedule(a.registry, read(a.receipt), a.place_noise, a.hold_noise, a.study,
                             a.v2_registry, dict(path=str(qualified), sha256=sha(qualified)), a.role)
    write(a.destination, dict(plan=PLAN, registry_sha256=REGISTRY_SHA256, role=a.role, candidate=ref,
          checkpoint_sha256=state["checkpoint_sha256"], place=places, hold=holds,
          noise={"place": {"path": str(a.place_noise), "sha256": sha(a.place_noise)},
                 "hold": {"path": str(a.hold_noise), "sha256": sha(a.hold_noise)}},
          model_forwards=0, optimizer_updates=0, native_integrations=0))
