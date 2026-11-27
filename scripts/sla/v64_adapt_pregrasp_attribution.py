"""Read the matched R4/A250 HOLD prefixes at already-confirmed 50 ms boundaries."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_bytes())


def journal(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def trajectory(case):
    raw = case / "native-01"
    wire = raw / "COMMANDS.private.jsonl"
    samples = raw / "MOBILE_SAMPLES.private.jsonl"
    accepted = {}
    for row in journal(wire):
        if row["phase"] == "mobile-accepted-boundary-050":
            step = row["native_step"]
            if step in accepted or not row["exact_receiver_knot"]:
                raise ValueError("Cannot assign one receiver knot to the confirmed boundary")
            accepted[step] = row
    result = {}
    for row in journal(samples):
        step = row["native_end_step"]
        if step not in accepted:
            continue
        target = accepted[step]
        reference = np.asarray(target["accepted_pose7"][:3])
        if (step in result or row["ticks"] != 25 or step-row["native_start_step"] != 25
                or target["accepted_wire_sha256"] != row["installed_reference_sha256"]
                or np.linalg.norm(reference - row["target_endpoint_EEF10"][:3]) > 1e-6):
            raise ValueError("Confirmed sample disagrees with the exact accepted reference")
        result[step] = dict(tcp=np.asarray(row["actual_TCP"])[:3,3], reference=reference,
                           mount=np.asarray(row["actual_mount"])[:3,3],
                           requested_jaw_fraction=row["installed_EEF10"][9], grasp=row["native_grasp"])
    if not result or any(b-a != 25 for a,b in zip(result, list(result)[1:])):
        raise ValueError("The original task-consumption clock is incomplete")
    return result, {"exact_receiver_log": digest(wire), "native_samples": digest(samples)}


def review(candidate, parent, expected_seed):
    import mujoco

    outcomes = [read(case / "RESULT.private.json") for case in (candidate, parent)]
    specs = [read(case / "EXECUTION_SPEC.json") for case in (candidate, parent)]
    actual, original = outcomes
    if (any(not outcome["shutdown_clean"] or outcome["technical_failure"] for outcome in outcomes)
            or any(read(case / "control/LEASE.json")["status"] != "CLOSED" for case in (candidate, parent))
            or not actual["unsafe"] or original["unsafe"] or not original["metrics"]["task_success"]
            or actual["trace_audit"]["integrations_after_first_STOP"] != 0):
        raise ValueError("Expected a clean matched HOLD success and a closed safety terminal")
    stop, = actual["physics_stops"]
    if stop["reason"] != "ILLEGAL_CONTACT" or stop["steps"] != actual["actual_native_integrations"]:
        raise ValueError("The registered stop mechanism changed")
    geometry = Path(specs[0]["model_path"])
    sig = digest(geometry)
    if any(s["frozen_files"][s["model_path"]] != sig for s in specs):
        raise ValueError("The paired compiled scene changed")
    if any(spec["comparison_policy_seed"] != expected_seed for spec in specs):
        raise ValueError("The paired policy noise seed changed")
    # Read the original compiled geometry only. No simulator advance is involved.
    model = mujoco.MjModel.from_binary_path(str(geometry))
    physics = candidate / "native-01/physics.jsonl"
    snapshots = {row["step"]: row for row in journal(physics)
                 if row["kind"] == "sample" and row["step"] in (stop["steps"]-1, stop["steps"])}
    earlier, terminal = (snapshots[stop["steps"]+delta] for delta in (-1, 0))
    if earlier["illegal_contact"] or not terminal["illegal_contact"] or not terminal["contact_stop"]:
        raise ValueError("The unsafe event must start at the observed STOP")

    def pairs(row):
        return {(r["geom1"], r["geom2"]) for r in row["contacts"]}

    collision = []
    for first, second in sorted(pairs(terminal)-pairs(earlier)):
        names = []
        for index in (first, second):
            geom = model.geom(index)
            names.append({"geom": geom.name, "body": model.body(int(geom.bodyid[0])).name})
        collision.append(names)
    if len(collision) != 1:
        raise ValueError("The original contact transition must add exactly one geometric pair")
    a, a_evidence = trajectory(candidate)
    r4, r4_evidence = trajectory(parent)
    if not set(a) <= set(r4) or any(a[step]["grasp"] or r4[step]["grasp"] for step in a):
        raise ValueError("Both runs must have the same supported pregrasp clock")

    def mm(value):
        return (1000*value).tolist()

    comparisons = []
    for step, sample in a.items():
        previous = r4[step]
        tracking = sample["tcp"]-sample["reference"]
        r4_tracking = previous["tcp"]-previous["reference"]
        comparisons.append(dict(tick=step, selected_controller_reference_observed=False,
            tcp_change_mm=mm(sample["tcp"]-previous["tcp"]),
            accepted_reference_change_mm=mm(sample["reference"]-previous["reference"]),
            a250_minus_accepted_mm=mm(tracking), r4_minus_accepted_mm=mm(r4_tracking),
            tracking_change_mm=mm(tracking-r4_tracking), mount_change_mm=mm(sample["mount"]-previous["mount"]),
            a250_jaw_fraction=sample["requested_jaw_fraction"], r4_jaw_fraction=previous["requested_jaw_fraction"]))
    last = comparisons[-1]
    return dict(condition=candidate.name, policy_seed=expected_seed, scene_sha256=sig,
                stop_tick=stop["steps"], post_STOP_ticks=0, collision=collision,
                unmatched_partial_ticks=stop["steps"]-last["tick"], common_pregrasp_atoms=len(comparisons),
                last_confirmed=last, a250_terminal_sha256=digest(candidate/"RESULT.private.json"),
                r4_terminal_sha256=digest(parent/"RESULT.private.json"), physics_sha256=digest(physics),
                candidate_archive=a_evidence, parent_archive=r4_evidence, rows=comparisons)


def main(candidate, parent, output):
    conditions = (("A", 1349149438), ("B", 1594556684))
    paired = [review(candidate/f"ADAPT0250_N02_{side}", parent/f"R4_DEV_0200_N02_{side}", seed)
              for side, seed in conditions]
    with output.open("x") as stream:
        json.dump(dict(scope="PAIRED_ACCEPTED_REFERENCE_DIAGNOSTIC", native_integrations=0,
                       model_generations=0, admitted_recovery_episodes=0, paired=paired),
                  stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "parent", "output"):
        parser.add_argument(name, type=Path)
    main(**vars(parser.parse_args()))
