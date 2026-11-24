"""Score the sealed source factorial against the exact recorded successor states."""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

from sea1b_panel import ANCHORS, PLAN_SHA, recorded, registration, source_inputs
from sea1b_run import run_path
from sea1_archive_readout import TIMES, pose_error, pose_rows, rms, source_score
from source_custody import digest, pin, read, require, write_new
from demonstrated_grip import events
from policy_query import tensor_sha

EFFECTS = {
    "mask": (-1, 1, 0, 0), "state": (-1, 0, 1, 0),
    "joint": (-1, 0, 0, 1), "interaction": (1, -1, -1, 1),
}


def score_cell(output, current, targets):
    right, reference = output[:, 8:], targets[:, 8:]
    predicted, expected = pose_rows(right), pose_rows(reference)
    measured = pose_rows(current[None, 8:])[0]
    result = source_score(predicted, expected, measured)
    for label, n in (("P10", 10), ("FULL50", 50)):
        item = result[label]
        observed_events = events(TIMES[:n], right[:n, 7], step=False)
        target_events = events(TIMES[:n], reference[:n, 7], step=True)
        first_observed = {kind: next((q["s"] for q in observed_events if q["kind"] == kind), None)
                          for kind in ("close", "open")}
        first_target = {kind: next((q["s"] for q in target_events if q["kind"] == kind), None)
                        for kind in ("close", "open")}
        item.update(
            gripper_rms=rms(right[:n, 7]-reference[:n, 7]),
            first_event_errors=[kind for kind in first_target
                                if (first_target[kind] is None) != (first_observed[kind] is None)],
            first_model_grip_events_s=first_observed, first_expert_grip_events_s=first_target,
            model_grip_events=observed_events, expert_grip_events=target_events,
            first_model_grip_event_rows={k: None if t is None else int(np.ceil(30*t-1e-9))
                                         for k, t in first_observed.items()},
            first_expert_grip_event_rows={k: None if t is None else int(np.ceil(30*t-1e-9))
                                          for k, t in first_target.items()},
            model_first_offset_mm=float(1000*np.linalg.norm(predicted[0, :3, 3]-measured[:3, 3])),
            row_end_s=float(TIMES[n-1]), successor_frame_end_s=n/30.,
        )
    return result


def summarize(items, deviations, efforts):
    if any(r["status"] != "RETURNED" for r in items):
        return dict(status="INCONCLUSIVE_MISSING_CELL", effects=deviations)
    reduced = {}
    for label in EFFECTS:
        values = {s: [p["FULL50"][label] for p in deviations if p["anchor"] == s] for s in ANCHORS}
        reduced[label] = dict(
            consistent_increase=[s for s, v in values.items() if min(v) >= 2],
            consistent_decrease=[s for s, v in values.items() if max(v) <= -2],
            opposing_seeds=[s for s, v in values.items() if min(v) < 0 < max(v)],
        )
    base_positive = all(a["beats_hold"] for a in efforts)
    no_wrist_regression = all(not p["new_mask_event_error"] and abs(p["FULL50"]["mask"]) <= 2 for p in deviations)
    joined = any(len(m[k]) >= 3 for m in reduced.values()
                 for k in ("consistent_increase", "consistent_decrease"))
    status = ("MASK_CANDIDATE_FOR_INDEPENDENT_CONFIRMATION" if base_positive and no_wrist_regression
              else "CONSISTENT_MECHANISM_CANDIDATE" if base_positive and joined else "INCONCLUSIVE_OR_WEAK_EFFECT")
    return dict(status=status, effects=deviations, fixed_2mm_screen=reduced,
                full_source_vs_hold=efforts, full_source_beats_hold_all=base_positive)


def read_predictions(root, result, panel, inputs):
    _, fingerprints, _ = inputs
    expected = {x["anchor"]: x for x in fingerprints}
    start, stop = read(result / "START.json"), read(result / "OWNER_EXIT.json")
    require(stop["project_lock_released"] and start["registry_sha256"] == PLAN_SHA
            and stop["physics_ticks"] == stop["training_updates"] == 0, "inference owner still live or bound incorrectly")
    scores, raw = [], {}
    for cell in panel["cells"]:
        slot, anchor, cond, seed = (cell[s] for s in ("id", "anchor", "condition", "seed"))
        row = dict(slot=slot, anchor=anchor, condition=cond, seed=seed)
        entry, done = (result / f"{slot}.{name}.json" for name in ("entered", "returned"))
        row["status"] = "UNSTARTED"
        if entry.exists():
            row["status"] = "TECHNICAL_ENTERED"
        if done.exists():
            e, r = read(entry), read(done)
            identity = expected[anchor]["state_only_input_sha256" if cond in ("S2", "S3") else "original_input_sha256"]
            require((slot, seed, identity) == (e["slot"], e["seed"], e["model_input_sha256"]), "wrong sealed condition")
            with np.load(result / f"noise-{seed[1]}.npz", allow_pickle=False) as z:
                require(tensor_sha(z["noise"]) == e["actual_initial_noise_sha256"]
                        and z["sample_key"].tolist() == e["sample_key"], "actual stored noise changed")
            with np.load(pin(result / f"{slot}.npz", r["output_sha256"]), allow_pickle=False) as z:
                absolute = z["absolute"].copy()
                require(absolute.shape == (50, 16) and z["actions"].shape == (50, 32)
                        and np.isfinite(absolute).all(), "bad sealed return")
                row.update(r, status="RETURNED", decoded_sha256=tensor_sha(absolute),
                           raw_sha256=tensor_sha(z["actions"]), raw_dtype=str(z["actions_dtype"]),
                           sampler_state_sha256=tensor_sha(z["sampler_state"]),
                           actual_initial_noise_sha256=e["actual_initial_noise_sha256"], fields=e["fields"])
            raw[anchor, seed[1], cond] = absolute
        scores.append(row)
    require(len(raw) == stop["returned"]
            and sum(r["status"] != "UNSTARTED" for r in scores) == stop["entered"], "owner counts do not match files")
    return scores, raw, stop


def run(root, output):
    _, panel = registration()
    observations, fingerprints, virtual = source_inputs(root, panel)
    result = run_path(root)
    rows, raw, owner = read_predictions(root, result, panel, (observations, fingerprints, virtual))
    by_slot, contrasts, efforts, labels = {}, [], [], {}
    for a in fingerprints:
        frame = a["frame"]
        with h5py.File(pin(root / "data" / a["episode"], a["source_sha256"])) as h:
            labels[a["anchor"]] = recorded(h, "state", slice(frame+1, frame+51))
            original = recorded(h, "action", slice(frame, frame+50))
            require(np.array_equal(original, labels[a["anchor"]]), "step t label is not exact recorded successor t+1")
    for r in rows:
        if r["status"] != "RETURNED":
            continue
        s, n, c = key = (r["anchor"], r["seed"][1], r["condition"])
        r["metrics"] = score_cell(raw[key], observations[s][0]["state"], labels[s])
        r["left_proposal_sha256"] = tensor_sha(raw[key][:, :8])
        for horizon, size in (("P10", 10), ("FULL50", 50)):
            if (s, n, "S0") in raw:
                r["metrics"][horizon]["deviation_from_S0"] = pose_error(
                    pose_rows(raw[key][:size, 8:]), pose_rows(raw[s, n, "S0"][:size, 8:]))
        by_slot[key] = r
    for s in ANCHORS:
        for n in range(2):
            if (s, n, "S0") in by_slot:
                base = by_slot[s, n, "S0"]["metrics"]["FULL50"]
                efforts.append(dict(anchor=s, seed=[0, n], model_mm=base["model"]["xyz_rms_mm"],
                                    hold_mm=base["hold"]["xyz_rms_mm"],
                                    beats_hold=base["model"]["xyz_rms_mm"] < base["hold"]["xyz_rms_mm"]))
            if not all((s, n, c) in by_slot for c in ("S0", "S1", "S2", "S3")):
                continue
            group = [by_slot[s, n, c] for c in ("S0", "S1", "S2", "S3")]
            require(len({r["actual_initial_noise_sha256"] for r in group}) == 1, "factorial noise changed")
            effects = {}
            for h in ("P10", "FULL50"):
                vals = [r["metrics"][h]["model"]["xyz_rms_mm"] for r in group]
                effects[h] = {k: float(np.dot(vals, v)) for k, v in EFFECTS.items()}
            event_error = any(set(group[1]["metrics"][h]["first_event_errors"])
                              - set(group[0]["metrics"][h]["first_event_errors"]) for h in ("P10", "FULL50"))
            contrasts.append(dict(anchor=s, seed=[0, n], new_mask_event_error=event_error, **effects))
    write_new(output, dict(
        registry_sha256=PLAN_SHA, results=rows, resource_end=owner,
        historical_exact_reuse=0, technical=owner["entered"]-owner["returned"],
        unstarted=len(rows)-owner["entered"], device_input_sha256=digest(result / "DEVICE_INPUTS.json")
        if (result / "DEVICE_INPUTS.json").exists() else None,
        **summarize(rows, contrasts, efforts),
    ))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    os.umask(0o077)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU scoring only")
    run(args.root.resolve(), args.output.resolve())
