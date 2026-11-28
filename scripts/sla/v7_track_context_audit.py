"""Check prior accepted pose and measured EEF at independent TRAIN query boundaries."""

import argparse
from pathlib import Path

import numpy as np

from go2arx5_vla.trajectory.so3 import quat_to_matrix
from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from scripts.pi0292.eval_contract import matrix10


TRAIN = (
    ("V7-D0-EARLY-NATIVE-01/V7_D0_EARLY_TRAIN_11", "8dfcb2e6cc2a330e190139fd2c7ebec752c9e0a43984a8e4438870f17a2ca48a", (72, 84)),
    ("RECOVERY-TRAIN-LIVE-05/V64_RECOVERY_TRAIN_09", "669c70da7d2de72f072acde9d3b878f9cc883824761e5c4f9a1d9667f2beeed9", (61,)),
    ("RECOVERY-TRAIN-LIVE-05/V64_RECOVERY_TRAIN_10", "e10c1348f74e976131b9b68c56c751f615acd378a84a3415a706a4d759fa882b", (46,)),
)


def episode(case, expected_terminal, key_queries):
    result = read(pin(case / "RESULT.private.json", expected_terminal))
    lease = read(case / "control/LEASE.json")
    spec = read(pin(case / "EXECUTION_SPEC.json", lease["spec_sha256"]))
    require(lease["status"] == "CLOSED" and spec["collection_split"] == "TRAIN" and spec["evaluation_split"] == "TRAIN_RECOVERY"
            and result["shutdown_clean"] and not result["technical_failure"], "Use the original independent TRAIN archive")
    native = case / "native-01"
    inputs = sorted((native / "g7-queries").glob("*/INPUT_META.private.json"))
    require(len(inputs) == result["model_generations"], "Original query population changed")
    rows = []
    for q, (past, current) in enumerate(zip(inputs, inputs[1:]), start=1):
        meta = read(current)
        earlier = past.parent / "QUERY_ACTUAL.private.json"
        prior = read(earlier)
        atom = prior["confirmed_intervals"][-1]
        witness = atom["accepted_boundary"]
        request = read(current.parent / "MODEL_REQUEST.private.json")
        response = read(current.parent / "MODEL_RETURN.private.json")
        require(meta["query_index"] == q == int(current.parent.name) == prior["query"] + 1
                and not meta["teacher_or_object_truth_in_model_input"] and not meta["controller_selected_reference_known"]
                and "accepted_reference_TCP" in meta["compiler_only_fields"]
                and prior["full_P10"] and prior["confirmed_atoms"] == 10
                and witness["exact_receiver_knot"] and witness["accepted_wire_sha256"] == atom["wire_sha256"]
                and witness["native_step"] == atom["native_end_step"] == meta["native_step"] == request["native_step"]
                and request["input"] == meta["input"] and request["request_id"] == response["response"]["request_id"]
                and meta["input_ready_monotonic"] <= request["created_monotonic"] <= response["wall_end_monotonic"],
                "Ref/witness unavailable at the original query")
        with np.load(pin(meta["input"]["path"], meta["input"]["sha256"]), allow_pickle=False) as archive:
            actual = np.linalg.inv(archive["policy_from_world"]) @ matrix10(archive["state_eef10"])
            reference = archive["accepted_reference_TCP"]
        pose = witness["accepted_pose7"]
        require(np.allclose(reference[:3, 3], pose[:3], atol=2e-7, rtol=0.)
                and np.allclose(reference[:3, :3], quat_to_matrix(pose[3:]), atol=2e-7, rtol=0.),
                "The live pre-inference pose does not match the prior accepted receiver knot")
        delta = 1000 * (reference[:3, 3] - actual[:3, 3])
        rotation = reference[:3, :3] @ actual[:3, :3].T
        rows.append(dict(query=q, tick=meta["native_step"], accepted_minus_measured_xyz_mm=delta.tolist(),
                         accepted_minus_measured_l2_mm=float(np.linalg.norm(delta)),
                         accepted_minus_measured_angle_deg=float(np.rad2deg(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))),
                         observation_sha256=meta["input"]["sha256"], previous_consumption_sha256=sha(earlier),
                         accepted_wire_sha256=atom["wire_sha256"]))
    require(all(any(row["query"] == q for row in rows) for q in key_queries), "Missing fixed diagnostic state")
    return dict(episode=case.name, source_result_sha256=expected_terminal, total_queries=len(inputs),
                matched_prior_accepted_queries=len(rows), first_query_has_no_previous_model_prefix=True,
                key_queries=[row for row in rows if row["query"] in key_queries],
                translation_p95_mm=float(np.percentile([row["accepted_minus_measured_l2_mm"] for row in rows], 95)),
                rotation_p95_deg=float(np.percentile([row["accepted_minus_measured_angle_deg"] for row in rows], 95)),
                rows=rows)


def audit(site):
    cases = [episode(site / path, terminal, queries) for path, terminal, queries in TRAIN]
    return dict(cases=cases, linked_queries=sum(r["matched_prior_accepted_queries"] for r in cases),
                scope="PRE_INFERENCE_ACCEPTED_REFERENCE_DIAGNOSTIC", controller_selected_reference_available=False,
                training_or_deployment_condition_enabled=False, native_integrations=0, model_forwards=0, optimizer_updates=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write(args.output, audit(args.site))
