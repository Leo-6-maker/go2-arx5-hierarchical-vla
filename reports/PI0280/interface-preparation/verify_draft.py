"""Offline metadata check only; no neural, physics, optimizer or remote imports."""
import json
import math
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    evidence = json.loads((root / "SOURCE_REVIEW.json").read_text(encoding="utf-8"))
    draft = json.loads((root / "INTERFACE_DRAFT.json").read_text(encoding="utf-8"))
    assert draft["authorization"] == "NOT_AUTHORIZED" and draft["launch_allowed"] is False
    assert draft["selected_checkpoint"] == draft["deployment"] == "NONE"
    assert all(value == 0 for value in draft["actual_operations"].values())
    assert evidence["metadata_only"] and not evidence["model_instantiation"]
    assert evidence["status"] == ""
    assert evidence["official_head"] == draft["official_source_sha"]
    assert all(value["working_bytes_match_commit"] for value in evidence["files"].values())
    counts = {}
    # ponytail: derive counts from saved headers; never load tensors for a dimension audit.
    for name, tensor in evidence["tensor_headers"].items():
        if ".position_embedding." in name:
            continue
        assert tensor["shape"][0] == 32
        group = name.split(".")[1]
        counts[group] = counts.get(group, 0) + math.prod(tensor["shape"][1:])
    assert counts == draft["parameters"]["per_row"]
    assert sum(counts.values()) == draft["parameters"]["allowed_to_change_scalars"] == 10180740
    positions = math.prod(evidence["tensor_headers"]["action_head.position_embedding.weight"]["shape"])
    assert sum(counts.values()) * 32 + positions == draft["parameters"]["stock_tune_projector_scalars_including_shared_positions"]
    budget = draft["future_budget_proposed_not_authorized"]
    assert budget["first_hypothesis_max_updates_including_canary"] * budget["batch_size"] == budget["sampled_chunks_max"]
    assert budget["max_policy_calls_two_tasks"] == (4 + 4 + 10) * 2 * budget["episode_max_policy_queries"]
    assert budget["max_policy_calls_four_tasks"] == budget["max_policy_calls_two_tasks"] * 2
    assert draft["candidate"]["projector_index"] == 10 and not draft["candidate"]["source_row_copy"]
    print("PASS: saved source/headers, row counts, proposed caps and zero-execution boundary")


if __name__ == "__main__":
    main()
