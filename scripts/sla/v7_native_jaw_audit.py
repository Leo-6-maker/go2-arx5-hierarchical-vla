"""Compare raw G7 proposals and actual prefix consumption in closed native V7-A arms."""

import argparse
from pathlib import Path

import numpy as np

from go2arx5_vla.sla_adapter.dense_g7_v6 import DT, GRIP_LIMIT, label_margin_report, previous_command_context
from go2arx5_vla.world_model.fusion_trajectory_adapter import MECHANICAL_TRAVEL_M, WORKING_CLOSED_M, WORKING_OPEN_M
from m38_gpu_preflight_v1 import pin, read, require
from scripts.pi0292.durable import write
from v7_native_binding import ARMS, PHASES, PLAN


def episode(case, score, checkpoint):
    terminal = read(pin(case / "RESULT.private.json", score["result_sha256"]))
    require(read(case / "control/LEASE.json")["status"] == "CLOSED"
            and terminal["actual_native_integrations"] == score["native_ticks"]
            and terminal["model_generations"] == score["forwards"], "Native card must have its original independent terminal")
    result = []
    repaired = {row["query"]: row for row in score.get("retiming_consumption", ())}
    for path in sorted((case / "native-01/g7-queries").glob("*/QUERY_ACTUAL.private.json")):
        report = read(path)
        req = read(path.parent / "MODEL_REQUEST.private.json")
        returned = read(path.parent / "MODEL_RETURN.private.json")["response"]
        q, tick = report["query"], report["native_start_step"]
        require(q == len(result) == int(path.parent.name) and tick == req["native_step"]
                and returned["comparison_checkpoint_sha256"] == checkpoint
                and returned["request_id"] == req["request_id"] and returned["new_generations_this_request"] == 1
                and returned["action"] == report["action"], "Native feedback or model return is not attributable")
        prior = req["previous_reference"]
        previous = previous_command_context(prior)
        meta = read(path.parent / "INPUT_META.private.json")
        require(meta["native_step"] == tick and meta["previous_reference"] == prior, "Wrong original observation boundary")
        with np.load(pin(meta["input"]["path"], meta["input"]["sha256"]), allow_pickle=False) as observation:
            require(float(observation["previous_width_m"]) == previous, "Wrong previous emitted command")
            measured = MECHANICAL_TRAVEL_M * float(observation["state_eef10"][9])
        with np.load(pin(returned["action"]["path"], returned["action"]["sha256"]), allow_pickle=False) as action:
            raw = action["physical_H40"]
        original = label_margin_report(raw, prior)
        logged = report.get("original_gripper_speed")
        if logged is not None:
            require(logged == original, "Original native G7 ledger changed")
        width = WORKING_CLOSED_M + (WORKING_OPEN_M - WORKING_CLOSED_M) * raw[:, 9]
        speed = np.abs(np.diff(np.r_[previous, width])) / DT
        violations = speed > GRIP_LIMIT + 1e-10
        require(bool(violations.any()) == original["original_execution_speed_exceeded"], "Original G7 threshold differs")
        intervals = report.get("confirmed_intervals", ())
        require(len(intervals) == report["confirmed_atoms"], "Original exact consumption count differs")
        for offset, part in enumerate(intervals):
            witness = part["accepted_boundary"]
            require(part["native_start_step"] == tick + 25 * offset and part["native_end_step"] == tick + 25 * (offset + 1)
                    and witness["exact_receiver_knot"] and witness["accepted_wire_sha256"] == part["wire_sha256"],
                    "An interval lacks its confirmed boundary")
        last = intervals[-1] if intervals else None
        retimed = repaired.get(q)
        result.append(dict(query=q, native_tick=tick, actual_cuda_noise_sha256=returned["comparison_noise_sha256"],
                           past_first_grasp=bool(score.get("grasp_tick") is not None and tick >= score["grasp_tick"]),
                           previous_mm=1000 * previous, measured_mm=1000 * measured, first_seam_mm_s=1000 * float(speed[0]),
                           peak_inside_mm_s=1000 * float(speed[1:].max()), first_seam_over_limit=bool(violations[0]),
                           interior_over_limit=bool(violations[1:].any()), originally_reported_g7=logged is not None,
                           confirmed_atoms=len(intervals), last_confirmed_jaw_mm=None if last is None else 1000 * last["emitted_gripper_width_m"],
                           retimed_verified=retimed, action_sha256=returned["action"]["sha256"]))
    require(len(result) == score["forwards"], "Missing native model returns")
    return result


def summary(rows, cutoff=None):
    completed = [row["confirmed_atoms"] if cutoff is None else
                 min(row["confirmed_atoms"], max(0, (cutoff - row["native_tick"]) // 25)) for row in rows]
    return dict(model_returns=len(rows), first_seam_over_limit=sum(row["first_seam_over_limit"] for row in rows),
                interior_over_limit=sum(row["interior_over_limit"] for row in rows),
                confirmed_50ms_intervals=sum(completed), confirmed_P10s=completed.count(10),
                retimed_with_confirmed_prefix=sum(row["retimed_verified"] is not None and count == row["confirmed_atoms"]
                                                 for row, count in zip(rows, completed, strict=True)),
                p95_first_seam_mm_s=float(np.percentile([row["first_seam_mm_s"] for row in rows], 95)) if rows else None)


def audit(root, phase, digests):
    data = [read(pin(root / f"SCORE-{i}-{phase}.json", digest)) for i, digest in enumerate(digests)]
    require(all(r["plan"] == PLAN and r["phase"] == phase and r["arm"] == arm for r, arm in zip(data, ARMS, strict=True)),
            "Wrong closed registered pair")
    results = []
    for left, right in zip(*(r["arms"] for r in data), strict=True):
        require(left["prior_cell"] == right["prior_cell"], "Original native case identities changed")
        variants = {arm: episode(root / f"NATIVE-{idx}-{phase}" / row["id"], row, score["checkpoint"])
                    for idx, (arm, row, score) in enumerate(zip(ARMS, (left, right), data, strict=True))}
        cutoff = min(left["native_ticks"], right["native_ticks"])
        shared = {arm: [row for row in rows if row["native_tick"] < cutoff] for arm, rows in variants.items()}
        keys = ("native_tick", "query", "actual_cuda_noise_sha256")
        require(all(all(a[k] == b[k] for k in keys) for a, b in zip(*(shared[arm] for arm in ARMS), strict=True)),
                "Shared native clock or actual noise changed")
        results.append(dict(case=left["prior_cell"], common_until_tick=cutoff,
                            same_grasp_phase=sum(a["past_first_grasp"] == b["past_first_grasp"]
                                                 for a, b in zip(*(shared[arm] for arm in ARMS), strict=True)),
                            full={arm: summary(rows) for arm, rows in variants.items()},
                            matched_clock={arm: summary(rows, cutoff) for arm, rows in shared.items()}, rows=variants))
    return dict(plan=PLAN, phase=phase, read_only=True, score_sha256=list(digests), cases=results,
                new_native_integrations=0, new_model_forwards=0, new_optimizer_updates=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("control_score_sha256")
    parser.add_argument("treatment_score_sha256")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write(args.output, audit(args.root, args.phase, (args.control_score_sha256, args.treatment_score_sha256)))
