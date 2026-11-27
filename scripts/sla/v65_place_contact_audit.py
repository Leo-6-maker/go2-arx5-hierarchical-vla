"""Replay geometry from the exact pregrasp PLACE contact and matched baseline."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from go2arx5_vla.world_model.native_replay import forbid_integration
from v64_adapt_pregrasp_attribution import digest, journal, read, trajectory


def frozen(case, scored):
    path = case / "EXECUTION_SPEC.json"
    terminal = case / "RESULT.private.json"
    lease = read(case / "control/LEASE.json")
    spec = read(path)
    if (lease["status"] != "CLOSED" or lease["spec_sha256"] != digest(path)
            or case.name != scored["id"] or digest(terminal) != scored["result_sha256"]
            or spec["interface_comparison_link"] != "L1"):
        raise ValueError("Require the closed paired PLACE receipts")
    return spec


def inspect(output, compare, candidate, baseline):
    receipt = read(compare)
    pair, = (item for item in receipt["pairs"] if item["candidate_1k"]["id"] == candidate.name)
    stopped, reference = pair["candidate_1k"], pair["baselines"]["R4-200"]
    a, b = (frozen(case, item) for case, item in ((candidate, stopped), (baseline, reference)))
    geometry = Path(a["model_path"])
    fingerprint = digest(geometry)
    if (any(spec["frozen_files"][spec["model_path"]] != fingerprint for spec in (a, b))
            or any(a[key] != b[key] for key in ("comparison_policy_seed", "task_instruction_id"))
            or stopped["actual_stop"]["reason"] != "ILLEGAL_CONTACT" or stopped["after_native_stop"] != 0
            or stopped["grasp"] or not reference["grasp"] or reference["unsafe"]):
        raise ValueError("Expected a matched first pregrasp contact and a valid R4 trajectory")

    hit, = stopped["actual_stop"]["forbidden_contacts"]
    cutoff = stopped["actual_stop"]["native_tick"]
    path_a, path_b = (case / "native-01/physics.jsonl" for case in (candidate, baseline))
    model = mujoco.MjModel.from_binary_path(str(geometry))
    data = mujoco.MjData(model)
    window_a = [r for r in journal(path_a) if cutoff - 125 <= r.get("step", -1) <= cutoff]
    earlier_a, last = window_a[0], window_a[-1]
    if earlier_a["step"] >= cutoff or last["step"] != cutoff or not last["contact_stop"]:
        raise ValueError("The closed case is missing the exact native contact sample")
    latest_b = max((r for r in journal(path_b) if cutoff - 2 <= r.get("step", -1) <= cutoff),
                   key=lambda row: row["step"])
    candidates = []
    for contact in last["contacts"]:
        ids = (contact["geom1"], contact["geom2"])
        names = {(model.geom(i).name, model.body(int(model.geom_bodyid[i])).name) for i in ids}
        if names == {(hit["geom0"], hit["body0"]), (hit["geom1"], hit["body1"])} and contact["dist"] == hit["distance_m"]:
            candidates.append(ids)
    if len(set(candidates)) != 1:
        raise ValueError("Cannot bind the exact reported contact pair")
    first, second = candidates[0]

    base = model.body("base_link")
    joint = model.jnt(int(base.jntadr[0]))
    if joint.type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Expect the original robot free base")
    q0 = int(joint.qposadr[0])
    acquired, archived = np.asarray(last["qpos"]), np.asarray(earlier_a["qpos"])
    prior = acquired.copy()
    prior[q0:q0 + 7] = archived[q0:q0 + 7]
    with forbid_integration(mujoco) as effects:
        clearance = {}
        for name, qpos in (("actual_stop", acquired), ("earlier_candidate", archived),
                           ("stop_with_earlier_base", prior), ("same_time_R4", latest_b["qpos"])):
            data.qpos[:] = qpos
            mujoco.mj_fwdPosition(model, data)
            clearance[name] = 1000 * mujoco.mj_geomDistance(model, data, first, second, .25, None)
    if effects or earlier_a["illegal_contact"] or latest_b["illegal_contact"]:
        raise ValueError("The comparison must preserve the original first contact and use zero integration")

    a_track, a_archive = trajectory(candidate)
    b_track, b_archive = trajectory(baseline)
    before = max(a_track)
    ac, pc = a_track[before], b_track[before]
    result = dict(cell=candidate.name, compare_sha256=digest(compare), scene_sha256=fingerprint, stop_tick=cutoff,
                  earlier_state_tick=earlier_a["step"], actual_R4_tick=latest_b["step"], exact_contact=hit,
                  geometry_clearance_mm=clearance, last_shared_accepted_boundary=before,
                  candidate_minus_R4_mm={key: (1000 * (ac[key] - pc[key])).tolist() for key in ("tcp", "reference", "mount")},
                  archived_control={"candidate": a_archive, "R4": b_archive},
                  new_integrations=0, new_model_forwards=0, controller_selection_directly_observed=False)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("output", "compare", "candidate", "baseline"):
        cli.add_argument(field, type=Path)
    inspect(**vars(cli.parse_args()))
