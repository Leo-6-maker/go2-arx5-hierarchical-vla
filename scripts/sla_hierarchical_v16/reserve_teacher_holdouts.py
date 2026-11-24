"""Reserve new validation/test placements using only historical geometry and identities."""

import argparse
import hashlib
from itertools import product
from pathlib import Path

import numpy as np
from demonstrated_grip import pin, read, require
from visual_lift_panel import write_new

BASE = "4185a00c6ea6f868394ed8e550c9a720b2a0f3490c0765b8014df3c14c3d5ef9"
CH = "d92fe34f122637bb6dc29bb93d9f7a3d60464f65c5474b00962432bbc07b4914"
OLD = "297b21cc67baa4b2cbdd78af68d244b550337b4cb9213fd27b26818e3958bec0"


def load_ref(ref):
    return read(pin(Path(ref["path"]), ref["sha256"]))


def key_positions(card):
    return (
        np.asarray([card["object_positions_world"][c][:2] for c in ("red", "blue")])
        * 1000
    )


def generate(archive):
    base = archive / "sla-goal-v2-train-hold4-cold-20261120-01"
    source = read(pin(base / "COLD_INPUT.private.json", BASE))
    proto = source["prototype_cards"]
    inv = load_ref(source["split_inventory"])
    refs = [
        *inv["original_populations"],
        inv["new_evaluation_population"],
        *[v for k, v in source.items() if k.endswith("_population")],
    ]
    refs += [
        {
            "path": str(base / "population/COMPILED_POPULATION.private.json"),
            "sha256": OLD,
        },
        {
            "path": str(
                archive
                / "sla-core-hopper-v1-geometries-20261121-01/population/COMPILED_POPULATION.private.json"
            ),
            "sha256": CH,
        },
    ]
    banks = {r["sha256"]: load_ref(r) for r in refs}
    used = {f: [] for f in proto}
    seeds = set()
    for bank in banks.values():
        for cluster in bank["clusters"]:
            if cluster["family"] in used:
                used[cluster["family"]].append(key_positions(cluster["card"]))
            for k in ("scene_seed", "reset", "seed", "reset_seed"):
                if k in cluster:
                    seeds.add(cluster[k])
        for roster in bank.get("conditions", {}).values():
            for row in roster:
                seeds.update(
                    row[k]
                    for k in ("seed", "reset", "scene_seed", "reset_seed")
                    if k in row
                )
    for cat, offset_key in (
        (load_ref(inv["C20_scene_catalog"]), "xy_mm"),
        (load_ref(inv["V52_split"])["layouts"], "xy_offset_mm"),
    ):
        for row in cat:
            f = row["family"]
            used[f].append(key_positions(proto[f]) + np.asarray(row[offset_key]))
    pools = {f: np.stack(v) for f, v in used.items()}
    roster = []
    tasks = [
        "HOLD-red",
        "HOLD-blue",
        "PLACE-red-right",
        "PLACE-blue-right",
        "PLACE-red-left",
        "PLACE-blue-left",
    ]
    for split, size in (("VAL", 4), ("TEST", 6)):
        for f in ("F1", "F2"):
            for task in tasks[:size]:
                name = f"V17-{split}-{f}-{task}"
                grid = np.asarray(list(product(range(-12, 13, 2), repeat=2)))
                shifts = np.stack([grid, -grid], axis=1)
                candidates = key_positions(proto[f])[None] + shifts
                # Changing the relative two-cube vector separates layout from a
                # nearly identical global shift. Compare with every prior layout.
                relative = candidates[:, 1] - candidates[:, 0]
                old = pools[f][:, 1] - pools[f][:, 0]
                dist = np.linalg.norm(relative[:, None] - old[None], axis=-1).min(
                    axis=1
                )
                choices = np.flatnonzero(
                    np.isclose(dist, dist.max(), rtol=0, atol=1e-8)
                )
                i = min(
                    choices,
                    key=lambda j: hashlib.sha256(
                        f"{name}/{grid[j].tolist()}".encode()
                    ).digest(),
                )
                xy = candidates[i]
                require(
                    dist[i] >= 8.0 - 1e-8, "insufficient two-object layout separation"
                )
                pools[f] = np.concatenate((pools[f], xy[None]))
                ids = {}
                for tag in ("reset", "seed"):
                    n = int.from_bytes(
                        hashlib.sha256(f"{name}/{tag}".encode()).digest()[:4], "big"
                    )
                    require(n not in seeds, "reused seed")
                    seeds.add(n)
                    ids[tag] = n
                roster.append(
                    {
                        "episode": name,
                        "split": split,
                        "layout": name + "-layout",
                        "family": f,
                        "task": task,
                        "red_blue_offsets_mm": shifts[i].tolist(),
                        "positions_world_m": {
                            c: [
                                *list(xy[j] / 1000),
                                proto[f]["object_positions_world"][c][2],
                            ]
                            for j, c in enumerate(("red", "blue"))
                        },
                        "nearest_previous_relative_vector_mm": round(float(dist[i]), 6),
                        **ids,
                    }
                )
    return {
        "registry": "V17_TEACHER_HOLDOUT_V2",
        "status": "reserved_numeric_conditions_only",
        "population_sources": sorted(banks),
        "C20_geometry_sha256": inv["C20_scene_catalog"]["sha256"],
        "V52_geometry_sha256": inv["V52_split"]["sha256"],
        "historical_geometry_counts": {f: len(v) for f, v in used.items()},
        "enrollment": "one teacher attempt per fixed row after allocation; retain every failed or unused row",
        "test_access": "isolated TEST labels; one frozen version after VAL selection",
        "initial_opening": "original task initialization only; no opening generalization tested",
        "roster": roster,
        "collected_VAL": 0,
        "collected_TEST": 0,
        "optimizer_updates": 0,
        "native_starts": 0,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    write_new(args.out, generate(args.archive))
