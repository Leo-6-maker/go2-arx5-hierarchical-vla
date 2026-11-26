"""Test R5-200 on fixed feedback after actual R4 entry into the PLACE region."""

import argparse
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v4_long_watch import pinned, ref, watch


PANEL = "ce7077c88ed98d00277340098a3851fff1acfe6e35dc529d0b69e65c96e9806d"
R5 = "f63869a7dda6f47a93985becd6811cfb88e80d967cbafb506546b3c24421563c"
CASES = {"R4_DEV_0200_P01_701": 65, "R4_DEV_0200_P02_701": 67}
WINDOWS_PER_CASE = 12
OPEN_MM = 39.6


def cold_roster(base, worker, path):
    from scripts.pi0292.durable import write

    report = base / "R4-DEV-0200-PLACE-NATIVE-01"
    panel = read(pin(report / "INDEPENDENT_PANEL_READBACK.json", PANEL))
    require(panel["gpu_and_native_leases_closed"] and panel["valid"] == 4, "Needs a fully closed R4 panel")
    arms = {row["id"]: row for row in panel["arms"]}
    rows = []
    for case_id, expected in CASES.items():
        case = report / case_id
        require(read(case / "control/LEASE.json")["status"] == "CLOSED", "Original native lease is live")
        result = pin(case / "RESULT.private.json", arms[case_id]["result_sha256"])
        terminal = read(result)["detail"]["score"]
        require(terminal["target_grasp"] and not terminal["released"], "The original case did not fail to release")
        tick = terminal["events"]["correct_region"]["native_step"]
        directory = case / "native-01/g7-queries"
        windows = [f.parent for f in sorted(directory.glob("*/MODEL_RETURN.private.json"))
                   if read(f.parent / "MODEL_REQUEST.private.json")["native_step"] >= tick]
        require(len(windows) == expected, "Original post-entry native population changed")
        positions = np.rint(np.linspace(0, len(windows) - 1, WINDOWS_PER_CASE)).astype(int)
        for position in positions:
            visit = windows[position]
            request, returned = (visit / label for label in
                                 ("MODEL_REQUEST.private.json", "MODEL_RETURN.private.json"))
            request_row = read(request)
            input_row = read(visit / "INPUT_META.private.json")
            record = read(returned)["response"]
            actual = read(visit / "QUERY_ACTUAL.private.json")
            require(request_row["input"] == input_row["input"]
                    and request_row["previous_reference"] == input_row["previous_reference"]
                    and not input_row["teacher_or_object_truth_in_model_input"]
                    and request_row["native_step"] == actual["native_start_step"]
                    and record["action"] == actual["action"], "The original feedback or returned action changed")
            pinned(input_row["input"])
            pinned(record["action"])
            noise = record["comparison_noise_sha256"]
            require(len(noise) == 64, "Missing archived actual CUDA noise")
            rows.append(dict(id=f"{case_id}-{visit.name}", case=case_id, phase="AFTER_CORRECT_REGION",
                             query=int(visit.name), tick=request_row["native_step"], original_first_region_tick=tick,
                             input=input_row["input"], request=ref(request), terminal=ref(result), returned=ref(returned),
                             noise_sha256=noise, noise_id=request_row["noise_id"]))
    require(len(rows) == 2 * WINDOWS_PER_CASE, "Incomplete registered fixed feedback")
    write(path, dict(rows=rows, original_worker=ref(worker), r4_panel_sha256=PANEL,
                     sampling="12 deterministic equally spaced post-entry returns per native case, including both ends",
                     native_integrations=0, model_forwards=0))
    print(sha(path), flush=True)


def width(action):
    with np.load(pinned(action), allow_pickle=False) as packet:
        raw = packet["physical_H40"]
        require(raw.shape == (40, 10) and np.isfinite(raw).all(), "Malformed original H40")
        return 2. + 40. * raw[:, 9]


def evaluate(base, worker, retention, roster, roster_sha, out):
    from scripts.pi0292.durable import write

    audit = pin(retention / "R5_STEP0200_PROVIDER_READBACK.json", R5)
    registered = read(pin(roster, roster_sha))
    require(registered["r4_panel_sha256"] == PANEL and len(registered["rows"]) == 24, "Wrong R5 feedback bank")
    watch(base, worker, roster, retention / "R5_STEP0200_PROVIDER", 200, audit, out,
          execute=True, retained_roster_sha=roster_sha)
    result_path = out / "RESULT.json"
    result = read(result_path)
    require(result["model_forwards"] == 24 and not result["physics_ticks"] and not result["training_updates"],
            "Actual counters differ from the fixed feedback test")
    summary = []
    for row in result["rows"]:
        require(row["actual_sampler_sha256"] == row["noise_sha256"], "Comparison changed its CUDA sample")
        original = width(read(pinned(row["returned"]))["response"]["action"])
        new = width(row["action"])
        metrics = row["metrics"]
        summary.append(dict(id=row["id"], case=row["case"], tick=row["tick"],
                            r4_p10_mm=float(original[:10].max()), r4_h40_mm=float(original.max()),
                            r5_p10_mm=float(new[:10].max()), r5_h40_mm=float(new.max()),
                            r5_h40_pass=metrics["original_envelope"]["original_full_H40_pass"],
                            r5_jaw_report=metrics["jaw"]))
    scored = out / "RELEASE_FEEDBACK_ACTUAL.json"
    write(scored, dict(rows=summary, original_result_sha256=sha(result_path), roster_sha256=roster_sha,
                       open_diagnostic_mm=OPEN_MM, r4_new_model_forwards=0, r5_new_model_forwards=24,
                       physics_ticks=0, training_updates=0))
    print(sha(scored), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "worker", "roster"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--retention", type=Path)
    parser.add_argument("--roster-sha")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.retention is None:
        cold_roster(args.base, args.worker, args.roster)
    else:
        evaluate(**vars(args))
