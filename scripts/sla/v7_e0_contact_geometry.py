"""Separate the base, arm and finger geometry of the four archived 3K PLACE stops."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from go2arx5_vla.world_model.native_replay import forbid_integration
from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v64_adapt_pregrasp_attribution import journal
from v65_place_contact_audit import frozen
from v7_place_first_divergence import CELLS, boundaries


PAIR_SCORE = "b66d0983503b63291d9d8e8efc0ab4c1b5518f45adec1ad6c890c8f9ba579ba8"
ARM = tuple(f"x5_joint{i}" for i in range(1, 7))
FINGERS = ("task_left_slide", "task_right_slide")


def motion_indices(model):
    base = model.body("base_link")
    free = model.joint(int(base.jntadr[0]))
    require(int(free.type[0]) == mujoco.mjtJoint.mjJNT_FREE, "Expected the original free base")
    offset = int(free.qposadr[0])
    return slice(offset, offset + 7), [int(model.joint(name).qposadr[0]) for name in ARM], [
        int(model.joint(name).qposadr[0]) for name in FINGERS]


def with_substitution(current, donor, indices):
    values = current.copy()
    values[indices] = donor[indices]
    return values


def sample_geometry(candidate, baseline, received, historical):
    spec, other = frozen(candidate, received), frozen(baseline, historical)
    model_path = Path(spec["model_path"])
    definition = spec["frozen_files"][spec["model_path"]]
    pin(model_path, definition)
    require(other["frozen_files"][other["model_path"]] == definition and all(
        spec[key] == other[key] for key in ("comparison_policy_seed", "task_instruction_id")), "The matched scene or seed changed")
    stop = received["actual_stop"]
    tick = stop["native_tick"]
    require(stop["reason"] == "ILLEGAL_CONTACT" and received["after_native_stop"] == 0 and not received["grasp"]
            and historical["grasp"] and not historical["unsafe"], "Require the closed first pregrasp stop and matched R4")
    hit, = stop["forbidden_contacts"]
    paths = [case / "native-01/physics.jsonl" for case in (candidate, baseline)]
    on_candidate = [r for r in journal(paths[0]) if tick - 125 <= r.get("step", -1) <= tick]
    earlier, actual = on_candidate[0], on_candidate[-1]
    orig = max((r for r in journal(paths[1]) if tick - 2 <= r.get("step", -1) <= tick), key=lambda r: r["step"])
    require(actual["step"] == tick and actual["contact_stop"] and not earlier["illegal_contact"]
            and not orig["illegal_contact"] and earlier["step"] < tick, "Missing exact contact or matched physical sample")

    model = mujoco.MjModel.from_binary_path(str(model_path))
    expected = {(hit["geom0"], hit["body0"]), (hit["geom1"], hit["body1"])}
    matches = set()
    for pair in actual["contacts"]:
        ids = (pair["geom1"], pair["geom2"])
        observed = {(model.geom(i).name, model.body(int(model.geom_bodyid[i])).name) for i in ids}
        if observed == expected and pair["dist"] == hit["distance_m"]:
            matches.add(ids)
    ids, = matches
    p, r4, own_earlier = [np.asarray(row["qpos"]) for row in (actual, orig, earlier)]
    base, arm, fingers = motion_indices(model)
    tests = {"actual": p, "earlier_candidate": own_earlier, "r4": r4,
             "restore_earlier_candidate_base": with_substitution(p, own_earlier, base),
             "substitute_r4_base": with_substitution(p, r4, base),
             "substitute_r4_arm": with_substitution(p, r4, arm),
             "substitute_r4_fingers": with_substitution(p, r4, fingers)}
    data = mujoco.MjData(model)
    distances = {}
    with forbid_integration(mujoco) as effects:
        for name, qpos in tests.items():
            data.qpos[:] = qpos
            mujoco.mj_fwdPosition(model, data)
            distances[name] = 1000 * mujoco.mj_geomDistance(model, data, *ids, .25, None)
    require(not effects, "Static geometry replay must integrate zero steps")

    cand_track, cand_evidence = boundaries(candidate)
    base_track, base_evidence = boundaries(baseline)
    last = max(k for k in cand_track.keys() & base_track.keys() if k < tick)
    before = {key: (1000 * (cand_track[last][key] - base_track[last][key])).tolist()
              for key in ("reference", "tcp", "mount")}
    return dict(cell=candidate.name, terminal_sha256=received["result_sha256"], scene_sha256=definition,
                stop_tick=tick, stop_pair=hit, earlier_tick=earlier["step"], r4_tick=orig["step"],
                clearance_mm=distances, last_shared_boundary=last, candidate_minus_r4_mm=before,
                evidence=[cand_evidence, base_evidence], physics_sha256=[sha(path) for path in paths])


def run(base, site, output):
    pair_score = site / "v65-3k-native-code-02/PLACE-COMPARE-FINAL-01.json"
    pairs = read(pin(pair_score, PAIR_SCORE))["pairs"]
    require(tuple(item["candidate_3k"]["id"].removeprefix("RECOVERY3000_") for item in pairs) == CELLS,
            "Expected the registered four-cell cohort")
    rows = []
    for item in pairs:
        candidate, baseline = item["candidate_3k"], item["baselines"]["R4-200"]
        rows.append(sample_geometry(site / "RECOVERY3000-NATIVE-PLACE-01" / candidate["id"],
                       base / "R4-DEV-0200-PLACE-NATIVE-01" / baseline["id"], candidate, baseline))
    write(output, dict(scope="STATIC_ARCHIVED_CONTACT_GEOMETRY", source_score_sha256=PAIR_SCORE, cases=rows,
                       new_model_forwards=0, new_native_integrations=0))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "site", "output"):
        cli.add_argument(name, type=Path)
    run(**vars(cli.parse_args()))
