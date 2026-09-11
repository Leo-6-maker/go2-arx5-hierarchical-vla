#!/usr/bin/env python3
"""Preparation-only checkpoint/step-tier selection ledger.

Large campaigns often produce several predeclared step tiers. This ledger makes
selection generation explicit so a later choice cannot silently become 'best of
many checkpoints' without provenance. It does not define a metric threshold or
select a winner automatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


class CheckpointSelectionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointSelectionError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(value: Any, field: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value), f"{field} must be lowercase sha256")
    return value


def build(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "request must be object")
    require(value.get("schema") == "STAGE45F_CHECKPOINT_SELECTION_REQUEST_V1", "schema drift")
    require(value.get("selection_split") == "TRAIN24", "checkpoint selection is TRAIN24-only at this stage")
    require(value.get("validation_e_read") is False and value.get("holdout_f_read") is False, "locked split read indicated")
    require(value.get("automatic_best_checkpoint_selection") is False, "automatic best-checkpoint selection is forbidden")
    require(value.get("pi_selected") is True, "explicit PI selection is required")

    candidates = value.get("candidate_checkpoints")
    require(isinstance(candidates, list) and candidates, "candidate_checkpoints missing")
    normalized = []
    labels: set[str] = set()
    trees: set[str] = set()
    for index, item in enumerate(candidates):
        require(isinstance(item, Mapping), f"candidate[{index}] must be object")
        label = item.get("label")
        require(isinstance(label, str) and label and label not in labels, f"candidate[{index}] label invalid/duplicate")
        labels.add(label)
        tree = _sha(item.get("checkpoint_tree_sha256"), f"candidate[{index}].checkpoint_tree_sha256")
        require(tree not in trees, f"candidate[{index}] duplicates checkpoint tree")
        trees.add(tree)
        step = item.get("optimizer_step")
        seed = item.get("training_seed")
        require(isinstance(step, int) and not isinstance(step, bool) and step > 0, f"candidate[{index}].optimizer_step invalid")
        require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0, f"candidate[{index}].training_seed invalid")
        packet = _sha(item.get("train24_evidence_packet_sha256"), f"candidate[{index}].train24_evidence_packet_sha256")
        normalized.append({
            "label": label,
            "checkpoint_tree_sha256": tree,
            "optimizer_step": step,
            "training_seed": seed,
            "train24_evidence_packet_sha256": packet,
        })

    selected_label = value.get("selected_label")
    require(isinstance(selected_label, str) and selected_label in labels, "selected_label is not a registered candidate")
    rationale = value.get("selection_rationale")
    require(isinstance(rationale, str) and rationale.strip(), "selection rationale missing")
    selected = next(item for item in normalized if item["label"] == selected_label)
    generation = value.get("selection_generation")
    require(isinstance(generation, int) and not isinstance(generation, bool) and generation >= 1, "selection_generation invalid")
    prior = value.get("prior_selection_ledger_sha256")
    if generation == 1:
        require(prior in {None, ""}, "generation 1 must not claim prior selection ledger")
    else:
        _sha(prior, "prior_selection_ledger_sha256")

    return {
        "schema": "STAGE45F_CHECKPOINT_SELECTION_LEDGER_V1",
        "status": "FROZEN_PI_SELECTION_ON_TRAIN24",
        "selection_generation": generation,
        "prior_selection_ledger_sha256": prior or None,
        "candidate_checkpoints": normalized,
        "selected_checkpoint": selected,
        "selection_rationale": rationale.strip(),
        "automatic_best_checkpoint_selection": False,
        "posthoc_unregistered_checkpoint_allowed": False,
        "metric_threshold_defined_by_tool": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "validation_execution_authorized": False,
        "training_authorized": False,
        "claim_boundary": "The ledger records a PI choice among pre-registered TRAIN24 candidates. It does not establish generalization, define a success threshold, or authorize Validation-E/training.",
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(build(read_json(args.input)), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CheckpointSelectionError as exc:
        print(f"STAGE45F_CHECKPOINT_SELECTION_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
