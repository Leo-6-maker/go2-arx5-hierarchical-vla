"""Compare the original H40 gate on true labels before and after coordinate repair."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2arx5_vla.sla_adapter.dense_g7_v6 import label_margin_report
from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec
from go2arx5_vla.world_model.sla_dense_policy_v5 import compile_proposal
from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0291_arx5_data.native_dataset import matrix_from10, pose10
from v6_reference_anchors import original_cases, receiver


def run(native_root, consumed, transformed, output):
    contract = read(transformed / "MANIFEST.json")
    require(contract["source_manifest_sha256"] == sha(consumed / "MANIFEST.json"),
            "Both supervision variants must bind the exact same archive")
    targets = {e["episode"]: e for e in contract["episodes"]}
    program = compile_proposal.__globals__["task"].FiftyMsProgram
    stages = defaultdict(Counter)
    for entry, spec, queries, _ in original_cases(native_root, consumed):
        replacement = targets[entry["episode"]]
        axes = np.linalg.inv(np.asarray(spec["policy_from_world"]))
        codec = GripLogitCodec(np.asarray(queries[0]["episode_fixed_T_policy_from_mount"]))
        with np.load(pin(consumed / entry["labels"], entry["sha256"]), allow_pickle=False) as before, np.load(
                pin(transformed / replacement["labels"], replacement["sha256"]), allow_pickle=False) as after:
            a, b = before["action"], after["action"]
            have = before["valid"].all(axis=1)
            require(np.array_equal(before["valid"], after["valid"]) and np.array_equal(a[..., 6], b[..., 6]),
                    "Changing the action populations or the gripper would invalidate this contrast")
        for index in np.flatnonzero(have):
            q = queries[index]
            stats = stages[q["phase"]]
            stats["full_H40"] += 1
            world_tcp = np.asarray(q["actual_state"]["tcp"])
            other = receiver(q["recent_reference"]["accepted_pose7"])
            base_pose = np.asarray(q["actual_state"]["base_pose"])
            base = np.r_[base_pose[:3, 3], Rotation.from_matrix(base_pose[:3, :3]).as_quat()]
            state = np.asarray(q["state_eef10"])
            jaw_gate = []
            for variant, values in (("original", a[index]), ("corrected", b[index])):
                decoded = codec.decode(state, np.pad(values, ((0, 0), (0, 25))))
                truth = pose10(axes @ matrix_from10(decoded), decoded[:, 9])
                jaw_gate.append(label_margin_report(decoded, q["previous_confirmed_command"])[
                    "original_execution_speed_exceeded"])
                try:
                    program(truth, world_tcp, other, base)
                except ValueError as error:
                    require("ORIGINAL_FULL_PREVIEW_RATE_AND_ACCELERATION_LIMITS" in str(error),
                            "Only the original geometric envelope can reject these inputs")
                    stats[f"{variant}_H40_rejected"] += 1
                else:
                    stats[f"{variant}_H40_passed"] += 1
            require(jaw_gate[0] == jaw_gate[1], "The original physical gripper seam decision changed")
            stats["original_jaw_reject"] += int(jaw_gate[0])
    total = sum(stages.values(), Counter())
    require(total["full_H40"] == contract["full_H40"], "Wrong complete real-support denominator")
    with output.open("x") as stream:
        json.dump(dict(complete_H40=dict(total), phases=dict(stages), transformed_manifest_sha256=sha(
            transformed / "MANIFEST.json"), native_integrations=0, model_forwards=0, optimizer_updates=0),
            stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native_root", "consumed", "transformed", "output"):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
