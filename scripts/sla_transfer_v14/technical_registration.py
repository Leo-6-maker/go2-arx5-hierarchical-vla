"""Derive the two proposed, contact-free B2 technical cards from an existing native timing card."""

import json
import sys
from hashlib import sha256
from pathlib import Path

from identity import read, record, rows, sha


def main(root, out):
    prior = read(root / "v13/run01/S0_ACTUAL.json")
    specs = [Path(k) for k in prior["inputs"] if k.endswith("/EXECUTION_SPEC.json")]
    starts = []
    for spec in specs:
        assert sha(spec) == prior["inputs"][str(spec)]
        trace = spec.parent / "native-01/planner.jsonl"
        first = next(r for r in rows(trace) if r["kind"] == "accept")
        assert sha256(first["raw"].encode()).hexdigest() == first["ref_id"]
        meta = json.loads(first["raw"])["_pi0294"]
        starts.append((meta["episode"], meta["sequence"], meta["source_mpc_s"]))
    binding = read(specs[0])["mobile_interface_qualification"]
    result = Path(binding["path"])
    assert sha(result) == binding["sha256"]
    spec = read(result.parent / "EXECUTION_SPEC.json")
    evidence = read(result)
    measured = list(rows(result.parent / "native-01/MOBILE_SAMPLES.private.jsonl"))
    atom = measured[0]["ticks"]
    assert all(
        r["ticks"] == atom and r["native_end_step"] - r["native_start_step"] == atom
        for r in measured
    )
    first_tick = measured[0]["native_start_step"]
    start = (
        evidence["actual_native_integrations"]
        - evidence["metrics"]["task_integrations"]
    )
    assert start == 5703 and first_tick == 2503 and start - first_tick == 128 * atom
    dt = spec["physics_dt_s"]
    durations = {"T0_STATIC_HOLD": 1.0, "T1_CLOCK_FAULT_STOP": 2.0}
    cards = []
    for name, seconds in durations.items():
        post_ticks = round(seconds / dt)
        assert post_ticks % atom == 0
        cards.append(
            {
                "id": name,
                "initialization_tick_cap": start,
                "after_initialization_tick_cap": post_ticks,
                "total_tick_cap": start + post_ticks,
                "physics_dt_s": dt,
                "wall_limit_s": spec["wall_limit_s"],
                "reference": "world TCP from post-initialization measured pose, constant orientation, neutral held base reference",
                "registration_only": True,
            }
        )
    record(
        out / "B2_TECHNICAL_REGISTRATION.json",
        {
            "card_source_sha256": sha(result),
            "site_sample_sha256": sha(
                result.parent / "native-01/MOBILE_SAMPLES.private.jsonl"
            ),
            "prior_first_install_diversity": len(set(starts)),
            "historical_cards": len(starts),
            "first_sequence": starts[0][1],
            "first_mpc_s": starts[0][2],
            "new_epoch_required": True,
            "startup_native_ticks": first_tick,
            "stabilization_atoms": (start - first_tick) // atom,
            "atom_ticks": atom,
            "cards": cards,
            "total_tick_cap": sum(c["total_tick_cap"] for c in cards),
            "max_requests": 2,
            "T0_acceptance": "New owner, frozen receiver and binary hashes, known zero-motion world reference, target echo, measured TCP, no forbidden contact; physics paused and sealed before exit",
            "T1_acceptance": "Only after T0 success. At a confirmed paused atom boundary, inject a clock-fault into the isolated receiver test callback; record HOLD request and independent receiver/controller acknowledgments separately. No further integration permits. Read stop boundary and unchanged final native tick from independent native instrumentation.",
            "fail_conditions": [
                "unmatched ownership or session",
                "failed hardware/scene separation",
                "unverified capture of stop boundary",
                "unknown install/consume result",
                "contact guard",
                "solver or mode failure",
                "post-stop tick",
                "time or tick limit",
            ],
            "scope": "Candidate ceilings only, no new owner or lease created. No manipulation, no GPU. Science pairs remain separate.",
        },
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
