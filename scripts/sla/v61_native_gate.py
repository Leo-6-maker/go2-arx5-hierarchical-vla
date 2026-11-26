"""Score the exact independent V6.1 HOLD waves and determine R4 parent eligibility."""

import argparse
from collections import Counter
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require, sha
from v61_hold_pair import PLAN, ROSTER, registered, rows


R4 = "f1f5a994005a9fc831934a156ca823d7e4b67794ce2533d72e087c2aeb29b9cb"
N02 = "db77327c1ff1bd3d1dd759d51afb4b7ada4c5d230b0da937aa893d0d8fbb243d"
PLACE = "ce7077c88ed98d00277340098a3851fff1acfe6e35dc529d0b69e65c96e9806d"
BASELINE = "f42d9b1a50d9652dae3c93934dd86b0ce762208f45fec5d7fab2c6686642aa3c"


def read_precedents(base, material):
    n02 = read(pin(base / "R4-DEV-0200-HOLD-NATIVE-01/INDEPENDENT_PANEL_READBACK.json", N02))
    place = read(pin(base / "R4-DEV-0200-PLACE-NATIVE-01/INDEPENDENT_PANEL_READBACK.json", PLACE))
    baseline = read(pin(material / "N03_MAIN38_MATCHED_READBACK.json", BASELINE))
    require(n02["registered"] == n02["valid"] == n02["full_success"] == 2
            and place["registered"] == place["valid"] == 4
            and n02["model_checkpoint_sha256"] == place["model_checkpoint_sha256"]
            and all(x["gpu_and_native_leases_closed"] for x in (n02, place))
            and not any(x["unsafe"] for x in (*place["arms"], *n02["arms"]))
            and baseline["pair_roster_sha256"] == ROSTER
            and len(baseline["arms"]) == baseline["reused_requests"] == 2
            and all(x["valid_science"] for x in baseline["arms"]),
            "The original closed R4 panel or matched MAIN3.8K archive changed")
    return n02, place, {("N03", int(x["id"].rsplit("_", 1)[1])): x for x in baseline["arms"]}


def read_wave(root, role, roster_ref):
    from m38_a_native_v1 import bind_native, comparison_manifest
    from m38_a_readout_v1 import read_arm

    comparison = read(root / "COMPARISON.json")
    require(comparison["v61_roster"] == roster_ref and comparison["v4_model_role"] == role
            and comparison["v4_block"] == "LONG_HOLD"
            and all(read(path)["status"] == "CLOSED" for path in (root / "control/LEASE.json",
                    root / "worker/control/LEASE.json")), "A paired wave is still open or changed")
    bank = read(pin(registered(roster_ref)["noise"]["path"], comparison["gpu_noise_sha256"]))
    expected = {row["noise_id"]: row["gpu_sha256"] for row in bank["tensors"]}
    roster = rows(roster_ref, role)
    provider = comparison_manifest(comparison)
    require(comparison["schedule"] == [dict(cell, checkpoint=provider["checkpoint_sha256"]) for cell in roster],
            "Unexpected closed-wave condition identities")
    scored = {}
    for cell in roster:
        case = root / cell["id"]
        request = read(case / "COMPARISON_REQUEST.json")
        lease = read(case / "control/LEASE.json")
        require(lease["status"] == "CLOSED" and request["spec_sha256"] == lease["spec_sha256"],
                "Incomplete official native request custody")
        spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
        bind_native(spec)
        result = read_arm(case, expected)[0]
        require(spec["comparison_policy_seed"] == cell["policy_seed"] and result["id"] == cell["id"],
                "Condition or actual seed changed")
        scored[cell["layout_slot"], cell["noise_label"]] = result
    worker = read(root / "worker/RESULT.private.json")
    require(sum(x["forwards"] for x in scored.values()) == worker["completed_generations"],
            "Case counters disagree with the frozen worker")
    return scored


def stop_mechanism(row):
    stop = row.get("actual_stop")
    if not row["unsafe"] or row["grasp"] or not stop:
        return None
    contacts = set()
    for item in stop.get("forbidden_contacts", ()):
        contact_pair = tuple(sorted((item.get("body0") or item.get("geom0", ""),
                                     item.get("body1") or item.get("geom1", ""))))
        contacts.add(contact_pair)
    return stop["reason"], tuple(sorted(contacts))


def settle(base, material, parent, r4, output):
    from scripts.pi0292.durable import write

    n02, place, existing_parent = read_precedents(base, material)
    roster = dict(path=str(material / "PAIRED_HOLD_ROSTER.json"), sha256=ROSTER)
    match_parent = existing_parent | read_wave(parent, "PARENT", roster)
    match_r4 = read_wave(r4, "R4_DEV_0200", roster)
    require(match_parent.keys() == match_r4.keys() and len(match_r4) == 4, "Incomplete paired population")
    paired, failures = [], Counter()
    parent_count = r4_count = valid = 0
    for slot, noise in sorted(match_r4):
        p, c = match_parent[slot, noise], match_r4[slot, noise]
        one_valid = p["valid_science"] and c["valid_science"]
        valid += int(one_valid)
        parent_count += int(one_valid and p["task_success"])
        r4_count += int(one_valid and c["task_success"])
        mechanism = stop_mechanism(c)
        if mechanism is not None and mechanism != stop_mechanism(p):
            failures[mechanism] += 1
        paired.append(dict(slot=slot, noise=noise, valid=one_valid, parent=p, r4=c,
                           paired_regression=one_valid and p["task_success"] and not c["task_success"]))
    repeated = any(count >= 2 for count in failures.values())
    grasps = sum(row["grasp"] for row in place["arms"])
    release = valid == 4 and grasps >= 3 and r4_count >= max(1, parent_count) and not repeated
    write(output, dict(plan=PLAN, r4_receipt_sha256=R4, previous_N02_success=n02["full_success"],
                       place=dict(valid=place["valid"], correct_grasp=grasps),
                       independent_HOLD=dict(paired_valid=valid, r4_full_success=r4_count,
                                             parent_full_success=parent_count, pairs=paired),
                       native_gpu_and_world_leases_closed=True, new_repeated_pregrasp_stop=repeated,
                       source_roster_sha256=ROSTER, reused_original_parent_requests=2, released=release,
                       hold_and_place_readback=[N02, PLACE], matched_n03_parent_readback=BASELINE))
    print(sha(output), release, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("base", "material", "parent", "r4", "output"):
        cli.add_argument(field, type=Path)
    settle(**vars(cli.parse_args()))
