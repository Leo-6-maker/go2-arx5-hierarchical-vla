"""Seal two successful-stage R4 TRAIN observations for fixed-state V7 comparison."""

import argparse
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require
from scripts.pi0292.durable import write
from v7_d0_roster import FIRST_CLOSE, PLAN
from v7_e0_matched_inputs import ref
from v7_g7_output_gradient import R4


SCORE = "de6280d6e3dfa4f8ed81939c9862d2555914f0318785aa9ac1fb573f0d54e185"


def freeze(site, output):
    from go2arx5_vla.sla_adapter.dense_g7_v6 import previous_command_context

    close = read(pin(site / "V7-D0-EARLY-SEALED-01/RESULT.json", FIRST_CLOSE))
    case = site / "V7-D0-EARLY-NATIVE-01/V7_D0_EARLY_TRAIN_11"
    row, = (row for row in close["cases"] if row["id"] == case.name)
    result = read(pin(case / "RESULT.private.json", row["result"]["sha256"]))
    lease = read(case / "control/LEASE.json")
    spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    require(lease["status"] == "CLOSED" and spec["collection_split"] == "TRAIN" and result["shutdown_clean"]
            and not result["unsafe"] and not result["technical_failure"], "Require the original safe R4 TRAIN run")
    native = case / "native-01"
    events = read(pin(native / "MULTITASK_SCORE_PROGRESS.private.json", SCORE))["events"]
    queries = [(p, read(p)) for p in sorted((native / "g7-queries").glob("*/QUERY_ACTUAL.private.json"))]
    frames = {}
    for name, stage in (("R4_SAFE_GRASP", "target_grasp"), ("R4_SAFE_LIFT", "target_lift20mm")):
        tick = events[stage]["native_step"]
        path, evidence = next((p, r) for p, r in queries if r["native_start_step"] <= tick < r["terminal_native_step"])
        require(evidence["full_P10"] and evidence["confirmed_atoms"] == 10 and not evidence["unconfirmed_native_ticks"],
                "The successful stage must occur during the full R4 prefix")
        folder = path.parent
        meta, request, response = (read(folder / role) for role in
            ("INPUT_META.private.json", "MODEL_REQUEST.private.json", "MODEL_RETURN.private.json"))
        response = response["response"]
        require(response["comparison_checkpoint_sha256"] == R4 and response["new_generations_this_request"] == 1
                and request["native_step"] == meta["native_step"] == evidence["native_start_step"]
                and request["previous_reference"] == meta["previous_reference"], "The actual R4 input binding changed")
        captured = meta["input"]
        image = pin(Path(captured["path"]), captured["sha256"])
        action = pin(response["action"]["path"], response["action"]["sha256"])
        with np.load(image, allow_pickle=False) as packet, np.load(action, allow_pickle=False) as original:
            require(packet["body_rgb"].shape == packet["wrist_rgb"].shape == (256, 256, 3)
                    and packet["state_eef10"].shape == (10,) and packet["T_policy_from_mount"].shape == (4, 4)
                    and packet["previous_width_m"] == previous_command_context(meta["previous_reference"])
                    and original["physical_H40"].shape == (40, 10), "Incomplete archived R4 observation or action")
        frames[name] = dict(case=case.name, task=spec["task_instruction_id"], language=request["language"],
                            phase=stage, observed_tick=tick, query=evidence["query"], input=captured, baseline=ref(action),
                            observed_noise_id=request["noise_id"], observed_noise_sha256=response["comparison_noise_sha256"],
                            previous_command=meta["previous_reference"], prefix_evidence=ref(path))
    require(len({frame["observed_noise_sha256"] for frame in frames.values()}) == 2, "Expected independent query noise")
    write(output, dict(plan=PLAN, split="TRAIN_DIAGNOSTIC", evidence_scope="FIXED_SUCCESSFUL_STAGE_OBSERVATIONS",
                      frames=frames, authorized_for_training_labels=False, model_forwards=0, native_integrations=0))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("site", type=Path)
    cli.add_argument("output", type=Path)
    freeze(**vars(cli.parse_args()))
