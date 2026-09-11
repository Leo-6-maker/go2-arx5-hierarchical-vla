#!/usr/bin/env python3
"""Preparation-only recipe freeze ledger for a future first Validation-E read.

The ledger binds one PI-selected candidate to exact model/data/normalization/
loss/sampler identities before any Validation-E access.  It cannot open a
validation path and cannot authorize the read; a later PI gate must do that.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


class RecipeFreezeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RecipeFreezeError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def validate(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "freeze request must be object")
    require(value.get("schema") == "STAGE45F_RECIPE_FREEZE_REQUEST_V1", "freeze request schema drift")
    require(value.get("selection_source_split") == "TRAIN24", "recipe must be selected from TRAIN-only evidence")
    require(value.get("validation_e_read") is False and value.get("holdout_f_read") is False, "locked split already read")
    require(value.get("pi_selected") is True, "recipe freeze requires explicit PI selection")
    candidate = value.get("candidate")
    require(isinstance(candidate, Mapping), "candidate block missing")
    required = (
        "candidate_label",
        "checkpoint_tree_sha256",
        "training_recipe_identity",
        "dataset_tree_sha256",
        "normalization_identity",
        "loss_identity",
        "sampler_identity",
        "decoder_identity",
        "metric_identity",
    )
    for field in required:
        require(isinstance(candidate.get(field), str) and candidate[field], f"candidate.{field} missing")
    seeds = value.get("training_seeds_supporting_selection")
    require(isinstance(seeds, list) and seeds, "training_seeds_supporting_selection missing")
    require(all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds), "training seed invalid")
    require(len(set(seeds)) == len(seeds), "training seeds duplicate")
    decision_packet_sha = value.get("train_decision_packet_sha256")
    require(isinstance(decision_packet_sha, str) and len(decision_packet_sha) == 64, "TRAIN decision packet SHA invalid")
    return {
        "schema": value["schema"],
        "pi_selected": True,
        "selection_source_split": "TRAIN24",
        "candidate": dict(candidate),
        "training_seeds_supporting_selection": list(seeds),
        "train_decision_packet_sha256": decision_packet_sha,
        "validation_e_read": False,
        "holdout_f_read": False,
    }


def freeze(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "candidate": value["candidate"],
        "training_seeds_supporting_selection": value["training_seeds_supporting_selection"],
        "train_decision_packet_sha256": value["train_decision_packet_sha256"],
    }
    return {
        "schema": "STAGE45F_RECIPE_FREEZE_LEDGER_V1",
        "status": "FROZEN_FOR_FUTURE_PI_VALIDATION_GATE",
        "decision_state": "PI_VALIDATION_GATE_REQUIRED",
        "frozen_payload": payload,
        "frozen_payload_sha256": canonical_sha(payload),
        "selection_source_split": "TRAIN24",
        "validation_e_read": False,
        "holdout_f_read": False,
        "validation_e_execute_supported": False,
        "holdout_f_execute_supported": False,
        "post_validation_tuning_under_same_selection_decision_allowed": False,
        "automatic_winner": False,
        "execution_authorized": False,
        "required_future_gate": [
            "exact Validation-E split identity without opening Holdout-F",
            "explicit one-candidate PI authorization",
            "fresh immutable output root and raw-before-derived evidence contract",
            "no recipe/checkpoint/statistics/loss/sampler mutation after freeze",
        ],
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(freeze(validate(read_json(args.input))), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RecipeFreezeError as exc:
        print(f"STAGE45F_RECIPE_FREEZE_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
