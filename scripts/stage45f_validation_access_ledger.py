#!/usr/bin/env python3
"""Preparation-only ledger for future Validation-E access generations.

The ledger makes repeated validation use explicit.  It never opens a dataset
path and cannot authorize a read.  A later PI gate must bind an exact frozen
recipe and exact Validation-E split identity before execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


class ValidationLedgerError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationLedgerError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def build(value: Any) -> dict[str, Any]:
    require(isinstance(value, Mapping), "request must be object")
    require(value.get("schema") == "STAGE45F_VALIDATION_ACCESS_LEDGER_REQUEST_V1", "request schema drift")
    require(value.get("validation_e_currently_read") is False, "this prep request must precede the next Validation-E read")
    require(value.get("holdout_f_read") is False, "Holdout-F must remain unread")
    freeze_sha = value.get("recipe_freeze_ledger_sha256")
    require(isinstance(freeze_sha, str) and len(freeze_sha) == 64, "recipe freeze SHA invalid")
    prior = value.get("prior_validation_accesses", [])
    require(isinstance(prior, list), "prior_validation_accesses must be list")
    normalized_prior: list[dict[str, Any]] = []
    generations: list[int] = []
    for index, record in enumerate(prior):
        require(isinstance(record, Mapping), f"prior access {index} must be object")
        generation = record.get("generation")
        require(isinstance(generation, int) and generation >= 1, f"prior access {index} generation invalid")
        require(generation not in generations, f"duplicate validation generation {generation}")
        generations.append(generation)
        for field in ("recipe_freeze_ledger_sha256", "validation_e_evidence_sha256", "pi_gate"):
            require(isinstance(record.get(field), str) and record[field], f"prior access {index}.{field} missing")
        require(record.get("holdout_f_read") is False, f"prior access {index} indicates Holdout-F read")
        normalized_prior.append(dict(record))
    if generations:
        require(sorted(generations) == list(range(1, max(generations) + 1)), "validation generations must be contiguous")
    next_generation = max(generations, default=0) + 1
    first_unseen_claim_allowed = len(prior) == 0
    return {
        "schema": "STAGE45F_VALIDATION_ACCESS_LEDGER_V1",
        "status": "LOCKED_PENDING_EXPLICIT_PI_GATE",
        "decision_state": "PI_VALIDATION_GATE_REQUIRED",
        "recipe_freeze_ledger_sha256": freeze_sha,
        "prior_validation_accesses": normalized_prior,
        "next_validation_generation": next_generation,
        "first_unseen_validation_claim_allowed_if_future_read_occurs": first_unseen_claim_allowed,
        "future_read_after_prior_access_must_not_be_called_first_unseen": not first_unseen_claim_allowed,
        "validation_e_read_now": False,
        "holdout_f_read": False,
        "execute_supported": False,
        "ledger_payload_sha256": canonical_sha({
            "recipe_freeze_ledger_sha256": freeze_sha,
            "prior_validation_accesses": normalized_prior,
            "next_validation_generation": next_generation,
        }),
        "required_future_gate": [
            "explicit PI authorization for exactly one frozen recipe",
            "exact Validation-E split identity",
            "fresh immutable evidence root",
            "raw-before-derived evaluator contract",
            "record resulting evidence SHA as this generation before any later Validation-E request",
        ],
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
    except ValidationLedgerError as exc:
        print(f"STAGE45F_VALIDATION_ACCESS_LEDGER_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
