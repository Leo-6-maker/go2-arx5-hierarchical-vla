"""Align the R4, 1K and 3K PLACE archives without running policy or physics."""

import argparse
import json
from pathlib import Path

import numpy as np

from v64_adapt_pregrasp_attribution import digest, journal, read


CELLS = ("P01_701", "P02_702", "P02_701", "P01_702")
ROOTS = ("R4_DEV_0200", "RECOVERY1000", "RECOVERY3000")
TICK_DT_S = .002
# Descriptive sensitivity levels. None changes an execution or promotion threshold.
DISTANCE_LEVELS_MM = (.1, 1., 2.)


def queries(case):
    for folder in sorted((case / "native-01/g7-queries").glob("[0-9][0-9][0-9][0-9]")):
        report = folder / "QUERY_ACTUAL.private.json"
        if not report.exists():
            continue
        query = read(report)
        result = read(folder / "MODEL_RETURN.private.json")["response"]
        action = folder / "model-response/ACTION.npz"
        if (query["query"] != int(folder.name) or digest(action) != result["action"]["sha256"]
                or result["action"]["sha256"] != query["action"]["sha256"]):
            raise ValueError(f"Unbound archived action in {case.name}/{folder.name}")
        with np.load(action, allow_pickle=False) as arrays:
            raw = arrays["normalized"]
            decoded = arrays["physical_H40"]
        if raw.shape != (1, 40, 32) or decoded.shape != (40, 10):
            raise ValueError("Full H40 was not preserved")
        meta = read(folder / "INPUT_META.private.json")
        if meta["native_step"] != query["native_start_step"] or query["mock_consumption"]:
            raise ValueError("Archived query is not aligned to real physics")
        with np.load(folder / "INPUT.npz", allow_pickle=False) as inputs:
            state = inputs["state_eef10"]
        yield dict(index=query["query"], tick=query["native_start_step"], raw=raw[0], decoded=decoded,
                   noise=result["comparison_noise_sha256"], observation=meta["array_sha256"], state=state,
                   full_h40=query["full_H40_compiled"], full_p10=query["full_P10"],
                   previous_width_m=query["previous_actual_gripper_command"]["width_m"],
                   seam_mps=query["original_gripper_speed"]["first_seam_speed_m_s"])


def maxima_mm(a, b):
    return float(1000 * np.linalg.norm(a - b, axis=-1).max())


def rotation_degrees(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


def first_crossings(rows, fields, tick):
    return {field: {str(level): next((row[tick] for row in rows if row[field] >= level), None)
                    for level in DISTANCE_LEVELS_MM} for field in fields}


def boundaries(case):
    native = case / "native-01"
    wire, samples = (native / name for name in ("COMMANDS.private.jsonl", "MOBILE_SAMPLES.private.jsonl"))
    accepted = {}
    for row in journal(wire):
        if row["phase"] == "mobile-accepted-boundary-050":
            step = row["native_step"]
            if not row["exact_receiver_knot"] or step in accepted:
                raise ValueError("Cannot establish the original unique accepted receiver knot")
            accepted[step] = row
    result = {}
    for row in journal(samples):
        step = row["native_end_step"]
        if step not in accepted:
            continue
        knot = accepted[step]
        if (step in result or row["ticks"] != 25 or step - row["native_start_step"] != 25
                or knot["accepted_wire_sha256"] != row["installed_reference_sha256"]):
            raise ValueError("Physical sample and accepted receiver do not share the original interval and wire")
        reference = np.asarray(knot["accepted_pose7"][:3])
        commanded = np.asarray(row["target_endpoint_EEF10"][:3])
        result[step] = dict(tcp=np.asarray(row["actual_TCP"])[:3, 3], reference=reference, commanded=commanded,
                            mount=np.asarray(row["actual_mount"])[:3, 3], grasp=row["native_grasp"],
                            same_wire_endpoint_difference_mm=maxima_mm(reference, commanded))
    if list(result) != list(accepted) or any(b - a != 25 for a, b in zip(result, list(result)[1:])):
        raise ValueError("The original accepted/task-consumption timeline is incomplete")
    return result, dict(exact_receiver_log=digest(wire), native_samples=digest(samples))


def compare_pair(parent, candidate, parent_boundaries, candidate_boundaries, ready, stop_tick):
    query_table = []
    for left, right in zip(queries(parent), queries(candidate)):
        if left["tick"] != right["tick"] or left["index"] != right["index"] or left["noise"] != right["noise"]:
            raise ValueError("The original matched noise or query clock diverged")
        if stop_tick is not None and left["tick"] >= stop_tick:
            break
        query_table.append(dict(
            tick=left["tick"], query=left["index"], matching_noise_sha256=left["noise"],
            matching_observation_fields=sorted(k for k, v in left["observation"].items()
                                               if right["observation"].get(k) == v),
            state_xyz_mm=maxima_mm(left["state"][:3], right["state"][:3]),
            normalized_max=float(np.abs(left["raw"] - right["raw"]).max()),
            decoded_h40_xyz_mm=maxima_mm(left["decoded"][:, :3], right["decoded"][:, :3]),
            decoded_p10_xyz_mm=maxima_mm(left["decoded"][:10, :3], right["decoded"][:10, :3]),
            parent_first_xyz_m=left["decoded"][0, :3].tolist(),
            candidate_first_xyz_m=right["decoded"][0, :3].tolist(),
            parent_full_h40=left["full_h40"], candidate_full_h40=right["full_h40"],
            parent_full_p10=left["full_p10"], candidate_full_p10=right["full_p10"],
            parent_previous_width_m=left["previous_width_m"], candidate_previous_width_m=right["previous_width_m"],
            parent_seam_mps=left["seam_mps"], candidate_seam_mps=right["seam_mps"]))
    if not query_table or query_table[0]["tick"] != ready:
        raise ValueError("No paired first inference at the original ready tick")

    original, original_hashes = parent_boundaries
    current, current_hashes = candidate_boundaries
    boundary_table = []
    for tick in sorted(original.keys() & current.keys()):
        if tick <= ready or (stop_tick is not None and tick >= stop_tick):
            continue
        a, b = original[tick], current[tick]
        boundary_table.append(dict(tick=tick, accepted_xyz_mm=maxima_mm(a["reference"], b["reference"]),
                                   tcp_xyz_mm=maxima_mm(a["tcp"], b["tcp"]),
                                   mount_xyz_mm=maxima_mm(a["mount"], b["mount"]),
                                   parent_accepted_z_m=a["reference"][2], candidate_accepted_z_m=b["reference"][2],
                                   parent_tcp_z_m=a["tcp"][2], candidate_tcp_z_m=b["tcp"][2],
                                   parent_mount_z_m=a["mount"][2], candidate_mount_z_m=b["mount"][2],
                                   nominal_target_xyz_mm=maxima_mm(a["commanded"], b["commanded"]),
                                   parent_accepted_vs_nominal_mm=a["same_wire_endpoint_difference_mm"],
                                   candidate_accepted_vs_nominal_mm=b["same_wire_endpoint_difference_mm"],
                                   parent_grasp=a["grasp"], candidate_grasp=b["grasp"]))
    return dict(paired_queries=query_table, common_complete_50ms=boundary_table, hashes=[original_hashes, current_hashes],
                first_query_crossings=first_crossings(query_table, ("decoded_h40_xyz_mm", "decoded_p10_xyz_mm",
                                                                     "state_xyz_mm"), "tick"),
                first_boundary_crossings=first_crossings(boundary_table, ("accepted_xyz_mm", "tcp_xyz_mm",
                                                                          "mount_xyz_mm"), "tick"))


def physics_rows(case, ready, stop):
    path = case / "native-01/physics.jsonl"
    for row in journal(path):
        if row["kind"] == "sample" and ready <= row["step"] <= stop:
            yield row


def physical_pair(parent, candidate, ready, stop):
    fields = ("tcp_xyz_mm", "base_xyz_mm")
    left, right = ({row["step"]: row for row in physics_rows(case, ready, stop)}
                   for case in (parent, candidate))
    if stop not in right or not right[stop]["contact_stop"] or not right[stop]["illegal_contact"]:
        raise ValueError("The native terminal sample does not show the registered contact STOP")
    rows = []
    for tick in sorted(left.keys() & right.keys()):
        a, b = left[tick], right[tick]
        rows.append(dict(tick=tick, tcp_xyz_mm=maxima_mm(np.asarray(a["tcp_xyz"]), np.asarray(b["tcp_xyz"])),
                         base_xyz_mm=maxima_mm(np.asarray(a["qpos"][:3]), np.asarray(b["qpos"][:3])),
                         tcp_rotation_deg=rotation_degrees(np.asarray(a["tcp_R"]).reshape(3, 3),
                                                           np.asarray(b["tcp_R"]).reshape(3, 3)),
                         parent_tcp_m=a["tcp_xyz"], candidate_tcp_m=b["tcp_xyz"],
                         parent_base_m=a["qpos"][:3], candidate_base_m=b["qpos"][:3],
                         parent_qpos=a["qpos"], candidate_qpos=b["qpos"],
                         parent_qvel=a["qvel"], candidate_qvel=b["qvel"]))
    if not rows:
        raise ValueError("No original matched physics samples")
    sampled = [row["tick"] for row in rows]
    terminal = right[stop]
    return dict(first_crossings=first_crossings(rows, fields, "tick"), first_common=rows[0],
                last_common_before_stop=next(r for r in reversed(rows) if r["tick"] < stop),
                at_stop=rows[-1] if sampled[-1] == stop else None, matched_samples=len(rows),
                largest_paired_interval_ticks=max(np.diff(sampled).tolist()),
                terminal_single_arm={k: terminal[k] for k in ("step", "tcp_xyz", "tcp_R", "qpos", "qvel", "contacts")},
                archive_hashes=[digest(case / "native-01/physics.jsonl") for case in (parent, candidate)])


def pair_cases(baseline, early, final, scores, cell):
    paths = [root / f"{model}_{cell}" for root, model in zip((baseline, early, final), ROOTS, strict=True)]
    specs = [read(path / "EXECUTION_SPEC.json") for path in paths]
    results = [read(path / "RESULT.private.json") for path in paths]
    if any(read(path / "control/LEASE.json")["status"] != "CLOSED" or result["technical_failure"]
           or not result["shutdown_clean"] for path, result in zip(paths, results, strict=True)):
        raise ValueError("A source episode is not cleanly closed")
    scenes = {spec["frozen_files"][spec["model_path"]] for spec in specs}
    seeds = {spec["comparison_policy_seed"] for spec in specs}
    if len(scenes) != 1 or len(seeds) != 1:
        raise ValueError("Changed scene or noise seed in a paired case")
    stop = scores[cell]["actual_stop"]
    ready = next(queries(paths[-1]))["tick"]
    stop_tick = stop["native_tick"]
    if results[-1]["actual_native_integrations"] != stop_tick or scores[cell]["after_native_stop"] != 0:
        raise ValueError("Archived 3K physical stop did not match the closed score")
    result_hashes = [digest(path / "RESULT.private.json") for path in paths]
    if result_hashes[-1] != scores[cell]["result_sha256"]:
        raise ValueError("The archived 3K result no longer matches the matched score")
    result = dict(cell=cell, ready_tick=ready, stop_tick=stop_tick, stop=stop, scene_sha256=scenes.pop(),
                  seed=seeds.pop(), result_sha256=result_hashes)
    knots = [boundaries(path) for path in paths]
    result["R4_vs_1K"] = compare_pair(paths[0], paths[1], knots[0], knots[1], ready, stop_tick)
    result["R4_vs_3K"] = compare_pair(paths[0], paths[2], knots[0], knots[2], ready, stop_tick)
    result["1K_vs_3K"] = compare_pair(paths[1], paths[2], knots[1], knots[2], ready, stop_tick)
    result["R4_vs_3K"]["direct_physics_on_native_ticks"] = physical_pair(paths[0], paths[2], ready, stop_tick)
    return result


def main(baseline, early, final, score, output):
    scores = {row["id"].removeprefix("RECOVERY3000_"): row for row in read(score)["arms"]}
    if set(scores) != set(CELLS):
        raise ValueError("Wrong fixed PLACE population")
    results = [pair_cases(baseline, early, final, scores, cell) for cell in CELLS]
    with output.open("x") as stream:
        json.dump(dict(scope="ARCHIVAL_PLACE_DIVERGENCE", physics_dt_s=TICK_DT_S, score_sha256=digest(score),
                       controller_selected_reference="UNKNOWN_IN_DIRECT_ARCHIVE",
                       native_integrations=0, model_forwards=0, cases=results), stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "early", "final", "score", "output"):
        parser.add_argument(name, type=Path)
    main(**vars(parser.parse_args()))
