"""Record the fixed V3 decision for the first B6 100-to-250 continuation."""
import argparse
from pathlib import Path

from m38_gpu_preflight_v1 import read, require, sha
from scripts.pi0292.durable import write

ARMS = ("LAMBDA_ZERO", "SHARED_L2SP")


def decide(root, output):
    require(not output.exists(), "Retain the earlier continuation decision")
    provider = root/"CORE-PROVIDERS/ACTUAL.json"
    panel = root/"TRAIN-PANEL/PANEL.json"
    summary = root/"TRAIN-PANEL-OUTPUT/SUMMARY.json"
    distance = root/"PARAMETER-DISTANCE.json"
    proof, fixed, results, parameters = map(read, (provider, panel, summary, distance))
    require(proof["complete_training_endpoint"] and proof["step"] == parameters["complete_step"] == 100
            and results["panel_sha256"] == sha(panel)
            and fixed["split"] == "TRAIN_DIAGNOSTIC" and fixed["protected_conditions"] == 0
            and fixed["teacher_scene_count"] == 4 and len(fixed["frames"]) == 37
            and results["actual_forwards"] == fixed["max_forwards"] == 222,
            "The registered first endpoint and TRAIN panel changed")
    rows = {(row["model"],row["frame"]):row for row in results["rows"]}
    require(len(rows) == 3*37 and all(row["repeat_max_abs"] == 0 for row in rows.values()),
            "Incomplete or nonrepeatable same-input measurements")
    new_relative_to_control = []
    lift = []
    for frame in fixed["frames"]:
        control, treatment = (rows[arm,frame["id"]] for arm in ARMS)
        if treatment["g7"]["original_execution_speed_exceeded"] and not control["g7"]["original_execution_speed_exceeded"]:
            new_relative_to_control.append(frame["id"])
        if frame["phase"] in ("LIFT", "target_lift20mm"):
            lift.append(dict(frame=frame["id"], control_up_mm=control["p10_up_request_mm"],
                             treatment_up_mm=treatment["p10_up_request_mm"]))
    sources = {arm:read(root/f"{arm}-SOURCE/RESULT.json") for arm in ARMS}
    require(all(s["scope"] == "V7_B6_SOURCE_RETENTION" and s["step"] == 100
                and s["model_forwards"] == 42 and s["physics_ticks"] == s["optimizer_updates"] == 0
                and s["pair_sha256"] == sha(provider) for s in sources.values()), "Incomplete fixed source evaluation")
    reduction = parameters["shared_squared_distance_reduction"]
    permit = (reduction >= .05 and not new_relative_to_control
              and all(row["treatment_up_mm"] >= row["control_up_mm"] for row in lift)
              and not any(s["retention_alert_domains"] for s in sources.values()))
    inputs = (provider,panel,summary,distance,*(root/f"{arm}-SOURCE/RESULT.json" for arm in ARMS))
    write(output,dict(plan="8e6ffa192d3e6c0adce7e2f89ca97e8ea20f461a", scope="V7_B6_SHARED_L2SP_MATCHED_V1",
        source_step=100, proposed_end_step=250, maximum_new_formal_updates=300,
        decision="CONDITIONAL_PAIR_CONTINUE_TO_250" if permit else "NO_EXTENSION",
        relative_control_new_raw_illegal_frames=new_relative_to_control, lift_requests=lift,
        shared_squared_distance_reduction=reduction,
        source_relative_mae={arm:s["relative_mae"] for arm,s in sources.items()},
        inputs=[dict(path=str(p),sha256=sha(p)) for p in inputs],
        original_R4_safe_seam_restored=False, dynamic_L1_and_WBC_tested=False, scientific_promotion=False,
        rationale="REGISTERED_PARAMETER_AND_FUNCTION_DIRECTION_SCREEN_ONLY; RECHECK_SAME_TRAIN_PANEL_AT_250; DO_NOT_CHANGE_LAMBDA",
        new_native_integrations=0,new_optimizer_updates=0))
    print(output,sha(output),flush=True)


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root",type=Path)
    cli.add_argument("output",type=Path)
    decide(**vars(cli.parse_args()))
