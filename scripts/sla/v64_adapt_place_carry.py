"""Compare measured post-lift carry and exactly accepted TCP references."""

import argparse
import json
from pathlib import Path

import numpy as np

from v64_adapt_pregrasp_attribution import digest, journal, read, trajectory


CELLS = ("P01_701", "P01_702", "P02_701", "P02_702")
ATOM = 25
VARIANTS = {"A250": "ADAPT0250", "V6.5-1K": "RECOVERY1000"}


def measured_carry(case, outcome, zone, half_size, lift):
    accepted, witness = trajectory(case)
    samples = {}
    for atom in journal(case / "native-01/MOBILE_SAMPLES.private.jsonl"):
        step = atom["native_end_step"]
        if step in accepted and step >= lift:
            pose = np.asarray(atom["object_pose"])
            if pose.shape != (4, 4) or not np.isfinite(pose).all() or not atom["safety_valid"]:
                raise ValueError("Carry samples must be measured, finite and safe")
            center = pose[:2, 3]
            slack = zone["half"] - abs(center - zone["center"]) - abs(pose[:2, :3]) @ half_size
            samples[step] = dict(accepted[step], object_xy=center, margin=slack.min())
    if not samples or not samples[min(samples)]["grasp"]:
        raise ValueError("A measured post-lift consumed sample with a grasp is required")
    # Limit the analysis to the first uninterrupted measured grasp after lift.
    held = []
    for step, sample in samples.items():
        if not sample["grasp"]:
            break
        if held and step != held[-1][0] + ATOM:
            raise ValueError("The post-lift 50 ms carry clock has a gap")
        held.append((step, sample))
    return held, witness | {"terminal": digest(case / "RESULT.private.json"),
                            "terminal_tick": outcome["actual_native_integrations"]}


def describe(sequence, direction):
    start, end = sequence[0][1], sequence[-1][1]

    def progress(field):
        return float(1000 * ((end[field][:2] - start[field][:2]) @ direction))

    ref = progress("reference")
    actual = progress("tcp")
    start_residual = 1000 * ((start["tcp"][:2] - start["reference"][:2]) @ direction)
    end_residual = 1000 * ((end["tcp"][:2] - end["reference"][:2]) @ direction)
    return dict(start_tick=sequence[0][0], end_tick=sequence[-1][0], samples=len(sequence),
                accepted_reference_progress_mm=ref, actual_tcp_progress_mm=actual,
                tracking_residual_progress_mm=actual - ref, object_progress_mm=progress("object_xy"),
                start_tracking_residual_mm=float(start_residual), end_tracking_residual_mm=float(end_residual),
                initial_containment_margin_mm=1000 * start["margin"],
                best_containment_margin_mm=1000 * max(row["margin"] for _, row in sequence),
                final_containment_margin_mm=1000 * end["margin"],
                min_installed_jaw_fraction=min(row["requested_jaw_fraction"] for _, row in sequence),
                max_installed_jaw_fraction=max(row["requested_jaw_fraction"] for _, row in sequence))


def review(cell, paths, *, candidate_role="A250"):
    specs = [read(case / "EXECUTION_SPEC.json") for case in paths]
    outcomes = [read(case / "RESULT.private.json") for case in paths]
    if (any(read(case / "control/LEASE.json")["status"] != "CLOSED" for case in paths)
            or any(result["technical_failure"] or not result["shutdown_clean"] for result in outcomes)
            or any(specs[0][name] != specs[1][name] for name in (
                "comparison_policy_seed", "task_instruction_id", "multitask_geometry"))
            or len({spec["frozen_files"][spec["model_path"]] for spec in specs}) != 1):
        raise ValueError("The original matched and independently closed PLACE conditions are required")
    identity = dict(cell=cell, policy_seed=specs[0]["comparison_policy_seed"],
                    terminal_sha256=[digest(case / "RESULT.private.json") for case in paths])
    roles = ("R4", candidate_role)
    if any(result["unsafe"] for result in outcomes):
        return dict(identity, status="SAFETY_TERMINAL_RETAINED",
                    unsafe=dict(zip(roles, [r["unsafe"] for r in outcomes], strict=True)))
    timing = []
    for result in outcomes:
        events = result["detail"]["score"]["events"]
        timing.append({name: events.get(name, {}).get("native_step") for name in (
            "target_grasp", "target_lift20mm", "correct_region", "released")})
    identity["events"] = dict(zip(roles, timing, strict=True))
    if not all(row["target_lift20mm"] is not None for row in timing):
        return dict(identity, status="NO_PAIRED_LIFT")

    _, color, destination = specs[0]["task_instruction_id"].split("-")
    geometry = specs[0]["multitask_geometry"]
    definition = geometry["zones"][destination]
    zone = dict(center=np.asarray(definition["center_world_XY"]), half=np.asarray(definition["half_size_XY"]))
    half_size = np.asarray(geometry["object_half_sizes"][color])
    held = [measured_carry(case, result, zone, half_size, events["target_lift20mm"])
            for case, result, events in zip(paths, outcomes, timing, strict=True)]
    # Use one spatial direction and align each trial at its first verified
    # post-lift atom. Natively recorded event ticks remain in the receipt.
    direction = zone["center"] - held[0][0][0][1]["object_xy"]
    distance = np.linalg.norm(direction)
    if distance <= 1e-9:
        raise ValueError("The initial measured object and goal must define a nonzero direction")
    direction /= distance
    common = min(len(sequence) for sequence, _ in held)
    if common < 2:
        raise ValueError("Expected a measurable common post-lift carry prefix")
    return dict(identity, status="PAIRED_CARRY", direction_world_xy=direction.tolist(),
                common_samples=common, role={
                    name: dict(common=describe(sequence[:common], direction),
                               full=describe(sequence, direction), witness=witness)
                    for name, (sequence, witness) in zip(roles, held, strict=True)})


def run(candidate, parent, output, *, candidate_role="A250", cell=None):
    prefix = VARIANTS[candidate_role]
    paired = [review(key, (parent / f"R4_DEV_0200_{key}", candidate / f"{prefix}_{key}"),
                     candidate_role=candidate_role) for key in (CELLS if cell is None else (cell,))]
    with output.open("x") as stream:
        json.dump(dict(schema="V64_PAIRED_POST_LIFT_ACCEPTED_REFERENCE", native_integrations=0,
                       model_generations=0, optimizer_updates=0, controller_selected_reference=False,
                       trials=paired), stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("candidate", "parent", "output"):
        parser.add_argument(argument, type=Path)
    parser.add_argument("--candidate-role", choices=VARIANTS, default="A250")
    parser.add_argument("--cell", choices=CELLS)
    run(**vars(parser.parse_args()))
