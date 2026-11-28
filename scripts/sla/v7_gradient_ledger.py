"""Read measured V7-A gradient magnitudes from both sealed formal runs."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


ARMS = ("CONTROL_NEUTRAL", "TREATMENT_PREVIOUS_EXECUTED_GRIPPER_COMMAND")
WEIGHTS = {"target": 96 / 128, "robodojo": 16 / 128, "libero": 16 / 128, "auxiliary": 0.375}


def read_sealed_ledgers(root):
    terminal = json.loads((root / "TERMINAL.json").read_bytes())
    if terminal["formal_updates"] != 1000 or terminal["status"] != "FORMAL_V7_PAIR_COMPLETE":
        raise ValueError("Expected a completed V7-A pair")
    for arm in ARMS:
        receipt = (root / arm / "TERMINAL.json").read_bytes()
        if hashlib.sha256(receipt).hexdigest() != terminal["arms"][arm]["terminal_sha256"]:
            raise ValueError(f"Arm terminal SHA-256 changed for {arm}")
        data = (root / arm / "UPDATES.jsonl").read_bytes()
        if hashlib.sha256(data).hexdigest() != json.loads(receipt)["optimizer_ledger_sha256"]:
            raise ValueError(f"Optimizer ledger SHA-256 changed for {arm}")
        rows = [json.loads(line) for line in data.splitlines()]
        if [row["step"] for row in rows] != list(range(1, 501)):
            raise ValueError(f"Incomplete step sequence for {arm}")
        yield arm, rows


def summarize(rows):
    magnitudes = {term: [] for term in WEIGHTS}
    norms = []
    cross_terms = []
    aggregate_cosines = []
    for row in rows:
        weighted = {term: row["components"][term]["gradient_l2"] * scale for term, scale in WEIGHTS.items()}
        joint = row["global_gradient_norm"]
        if not all(math.isfinite(value) and value >= 0 for value in [joint, *weighted.values()]):
            raise ValueError("Invalid measured gradient norm")
        for term, norm in weighted.items():
            magnitudes[term].append(norm)
        parts = list(weighted.values())
        cross = joint**2 - sum(norm**2 for norm in parts)
        available = 2 * sum(left * right for i, left in enumerate(parts) for right in parts[i + 1:])
        if available <= 0 or abs(cross) > available * 1.0001:
            raise ValueError("Logged joint and component norms cannot come from the same weighted sum")
        norms.append(joint)
        cross_terms.append(cross)
        aggregate_cosines.append(cross / available)
    return dict(
        updates=len(rows), weighted_component_l2_median={term: median(values) for term, values in magnitudes.items()},
        joint_l2_median=median(norms), joint_l2_max=max(norms),
        clips=sum(norm > 1.0 for norm in norms), minimum_clip_scale=1 / max(1, max(norms)),
        net_alignment_positive=sum(cross > 0 for cross in cross_terms),
        aggregate_pairwise_cosine_median=median(aggregate_cosines),
    )


def main(root, output):
    ledgers = dict(read_sealed_ledgers(root))
    arms = {arm: summarize(rows) for arm, rows in ledgers.items()}
    payload = dict(scope="ACTUAL_FULL_MODEL_NORMS_FROM_SEALED_FORMAL_LEDGERS", weights=WEIGHTS,
                   global_clip=1.0, arms=arms, gradient_replays=0, optimizer_updates=0)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("output", type=Path)
    main(**vars(cli.parse_args()))
