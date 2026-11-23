"""Check archived initial-state compatibility across every feature against the original bounds."""

import argparse
from itertools import combinations
from pathlib import Path

from d1_readiness import baseline, configure, pinned, read, record, rows, sha


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("root", type=Path)
    cli.add_argument("output", type=Path)
    a = cli.parse_args()
    root = a.root.resolve()
    configure(baseline(root))
    from d1_factorial_reference_matrix import (
        boundary,
        distances,
        initial_features,
        prior_pd,
    )
    from d1_gait_replay import population
    from go2arx5_vla.world_model.native_replay import packets

    reg = root / "v15/readiness01/D1_FACTORIAL_4X2_REGISTER.json"
    pinned(reg, "c69cea0c14cf4390d369a5fec9efc44ec29ecfe273df44326f2f39fffb42578a")
    bounds = read(reg)["pairwise_initial_limits"]
    states = {}
    groups = {}
    unavailable = []
    for name, group, folder, pin in population(root):
        if group in ("M0 stopped early", "withheld"):
            unavailable.append(name)
            continue
        path = folder / "init.trace"
        pinned(path, pin)
        row = next(
            x
            for x in rows(folder / "MOBILE_SAMPLES.private.jsonl")
            if x["native_end_step"] == 5703
        )
        parts, _, _ = boundary(folder, row)
        packet = packets(folder / "state.bin.journal")[5702]
        if packet["step"] != 5703:
            raise ValueError("not at initial gate")
        states[name] = initial_features(
            packet, parts, prior_pd(path), row["installed_EEF10"][9]
        )
        groups[name] = group
    comparisons = []
    for x, y in combinations(states, 2):
        actual = distances(states[x], states[y])
        violations = {
            k: {"observed": v, "limit": bounds[k]}
            for k, v in actual.items()
            if v > bounds[k]
        }
        comparisons.append(
            {"pair": [x, y], "compatible": not violations, "violations": violations}
        )
    record(
        a.output,
        {
            "registration_sha256": sha(reg),
            "scope": "post-hoc comparison of every pair of retained 5703-tick boundaries using all unchanged D1 features; no eligibility or treatment assignment in this audit",
            "groups": groups,
            "unavailable": unavailable,
            "pairs": comparisons,
            "new_native_requests": 0,
            "new_physics_ticks": 0,
            "new_model_forwards": 0,
            "new_updates": 0,
        },
    )


if __name__ == "__main__":
    main()
