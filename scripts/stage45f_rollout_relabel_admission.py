#!/usr/bin/env python3
"""Preparation-only admission validator for rollout-state relabel proposals.

This tool validates provenance for a possible future DAgger/on-policy relabel
stage.  Policy/WBC may be the source of visited states, but every supervision
label must come from an independent simulator expert/oracle.  The tool never
runs a simulator, model, WBC, or training job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


class RolloutRelabelAdmissionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RolloutRelabelAdmissionError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def validate(document: Any) -> dict[str, Any]:
    require(isinstance(document, Mapping), "proposal must be object")
    require(document.get("schema") == "STAGE45F_ROLLOUT_RELABEL_PROPOSAL_V1", "proposal schema drift")
    require(document.get("source_split") == "TRAIN", "visited states must come from TRAIN-only collection")
    require(document.get("validation_e_read") is False, "Validation-E read indicated")
    require(document.get("holdout_f_read") is False, "Holdout-F read indicated")
    generator = document.get("state_generator")
    require(isinstance(generator, Mapping), "state_generator missing")
    require(generator.get("type") in {"GROOT_POLICY", "GROOT_PLUS_WBC"}, "unsupported visited-state generator")
    for key in ("checkpoint_tree_sha256", "recipe_identity", "sampler_identity"):
        require(isinstance(generator.get(key), str) and generator[key], f"state_generator.{key} missing")
    if generator.get("type") == "GROOT_PLUS_WBC":
        require(isinstance(generator.get("wbc_identity"), str) and generator["wbc_identity"], "WBC identity required")

    labeler = document.get("labeler")
    require(isinstance(labeler, Mapping), "labeler missing")
    require(labeler.get("type") in {"SIMULATOR_EXPERT", "ORACLE"}, "labels must come from independent simulator expert/oracle")
    require(labeler.get("is_same_model_as_state_generator") is False, "GR00T self-labeling is forbidden")
    for key in ("implementation_identity", "environment_identity", "label_policy_identity"):
        require(isinstance(labeler.get(key), str) and labeler[key], f"labeler.{key} missing")

    samples = document.get("samples")
    require(isinstance(samples, list) and samples, "samples must be non-empty")
    state_ids: set[str] = set()
    label_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        require(isinstance(sample, Mapping), f"sample[{index}] must be object")
        state_id = sample.get("visited_state_id")
        label_id = sample.get("oracle_label_id")
        source_episode = sample.get("source_episode_id")
        require(isinstance(state_id, str) and state_id, f"sample[{index}].visited_state_id missing")
        require(state_id not in state_ids, f"duplicate visited_state_id: {state_id}")
        state_ids.add(state_id)
        require(isinstance(label_id, str) and label_id, f"sample[{index}].oracle_label_id missing")
        require(label_id not in label_ids, f"duplicate oracle_label_id: {label_id}")
        label_ids.add(label_id)
        require(isinstance(source_episode, str) and source_episode, f"sample[{index}].source_episode_id missing")
        require(sample.get("label_generated_from_same_policy_output") is False, f"sample[{index}] self-label flag must be false")
        require(sample.get("validation_or_holdout_origin") is False, f"sample[{index}] comes from locked split")
        normalized.append({
            "visited_state_id": state_id,
            "oracle_label_id": label_id,
            "source_episode_id": source_episode,
            "state_artifact_sha256": sample.get("state_artifact_sha256"),
            "label_artifact_sha256": sample.get("label_artifact_sha256"),
        })
        for key in ("state_artifact_sha256", "label_artifact_sha256"):
            require(isinstance(sample.get(key), str) and len(sample[key]) == 64, f"sample[{index}].{key} invalid")

    return {
        "schema": document["schema"],
        "source_split": "TRAIN",
        "state_generator": dict(generator),
        "labeler": dict(labeler),
        "samples": normalized,
        "validation_e_read": False,
        "holdout_f_read": False,
    }


def audit(value: Mapping[str, Any]) -> dict[str, Any]:
    generator = value["state_generator"]
    labeler = value["labeler"]
    independence = bool(
        labeler["is_same_model_as_state_generator"] is False
        and labeler["type"] in {"SIMULATOR_EXPERT", "ORACLE"}
    )
    return {
        "schema": "STAGE45F_ROLLOUT_RELABEL_ADMISSION_V1",
        "status": "PASS_PROVENANCE_ONLY" if independence else "BLOCKED",
        "decision_state": "PI_DECISION_REQUIRED",
        "sample_count": len(value["samples"]),
        "state_generator": generator,
        "labeler": labeler,
        "independent_supervision": independence,
        "groot_self_labeling": False,
        "state_and_label_ids_unique": True,
        "proposal_sha256": canonical_sha(value),
        "simulation_executed": False,
        "wbc_executed": False,
        "model_forward": False,
        "training": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "execution_authorized": False,
        "required_future_gate": [
            "explicit PI authorization of state-collection environment and policy checkpoint",
            "explicit PI authorization of independent oracle/expert implementation",
            "frozen collection budget before first rollout",
            "frozen relabel schema before first label generation",
            "separate evidence roots for visited state and oracle label artifacts",
        ],
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    value = validate(read_json(args.input))
    print(json.dumps(audit(value), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RolloutRelabelAdmissionError as exc:
        print(f"STAGE45F_ROLLOUT_RELABEL_ADMISSION_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
