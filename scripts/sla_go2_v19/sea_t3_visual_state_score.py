"""Score the fixed T3 factorial returns without altering its population."""

import argparse
import os
from pathlib import Path

import numpy as np

from sea_t3_visual_state import CONDITIONS, NEW_CAP, PANEL, PAIRS, T2_PANEL, code_binding
from sea_target_t2_score import HOLD_CLOSE
from source_custody import SEEDS, digest, pin, read, require, write_new
from demonstrated_grip import events
from policy_query import tensor_sha


def measure(path, raw_sha, absolute_sha, aperture):
    with np.load(path, allow_pickle=False) as z:
        raw, absolute = z["actions"], z["absolute"]
        require(raw.shape == (50, 32) and absolute.shape == (50, 16)
                and np.isfinite(raw).all() and np.isfinite(absolute).all()
                and tensor_sha(raw) == raw_sha and tensor_sha(absolute) == absolute_sha,
                "T3 official tensor boundary differs")
        grip = absolute[:, 15]
        changes = events(np.arange(51)/30., np.r_[aperture, grip], step=True)
        low, high = float(grip.min()), float(grip.max())
    return dict(
        query_aperture=aperture, min_predicted_aperture=low, max_predicted_aperture=high,
        closing_margin=HOLD_CLOSE-low, min_minus_query_aperture=low-aperture,
        closes=any(e["kind"] == "close" for e in changes),
        threshold_events=changes, raw_sha256=raw_sha, decoded_sha256=absolute_sha,
    )


def summarise(rows):
    complete = {(a["anchor"], tuple(a["seed"]), a["condition"]): a for a in rows if a["status"] == "RETURNED"}
    groups = []
    for pair in PAIRS:
        for seed in SEEDS:
            keys = [(pair, tuple(seed), k) for k in CONDITIONS]
            if not all(k in complete for k in keys):
                continue
            results = dict(zip(CONDITIONS, (complete[k] for k in keys), strict=True))
            g = {k: r["closing_margin"] for k, r in results.items()}
            groups.append(dict(
                pair=pair, seed=seed, closes={k: r["closes"] for k, r in results.items()},
                right_state_effect_with_go2_images=g["GS"]-g["GG"],
                right_state_effect_with_source_images=g["SS"]-g["SG"],
                camera_effect_with_go2_right_state=g["SG"]-g["GG"],
                camera_effect_with_source_right_state=g["SS"]-g["GS"],
                interaction=g["SS"]-g["SG"]-g["GS"]+g["GG"],
            ))
    totals = {}
    for condition in CONDITIONS:
        values = [a["closes"] for a in complete.values() if a["condition"] == condition]
        totals[condition] = dict(returned=len(values), closes=sum(values), expected=4)
    return groups, totals


def score(root, cold_path, output):
    cold = read(cold_path)
    dest, old = (root / "v19" / panel for panel in (PANEL, T2_PANEL))
    start, owner = (read(dest / name) for name in ("START.json", "OWNER_EXIT.json"))
    require(start["panel"] == PANEL and start["cold_sha256"] == digest(cold_path) and cold["code"] == code_binding()
            and owner["project_lock_released"] and not Path("/proc", str(start["owner_pid"])).exists()
            and owner["physics_ticks"] == owner["optimizer_updates"] == 0
            and 0 <= owner["returned"] <= owner["entered"] <= NEW_CAP, "closed owner or fixed proof differs")
    reused_path = dest / "REUSE.json"
    reuse = read(reused_path) if reused_path.exists() else {}
    require(not reuse or reuse == cold["reuse_cpu_verified"], "unexpected exact reuse manifest")
    records = []
    for cell in cold["cells"]:
        name, condition = cell["id"], cell["condition"]
        entered, returned = (dest / f"{name}.{phase}.json" for phase in ("entered", "returned"))
        if name in reuse:
            require(condition == "GG" and not entered.exists() and not returned.exists(), "GG reused and re-run")
            item = reuse[name]
            path = pin(old / (item["archived_slot"] + ".npz"), item["output_sha256"])
            kind = "EXACT_T2_REUSE"
        elif returned.is_file():
            e = read(entered)
            require(all(e[key] == val for key, val in cell.items()), "returned sampler or roster identity changed")
            seed = cell["seed"][1]
            with np.load(dest / f"noise-{seed}.npz", allow_pickle=False) as z:
                require(tensor_sha(z["noise"]) == cell["initial_noise_sha256"]
                        and z["sample_key"].tolist() == e["sample_key"], "actual T3 noise differs")
            item = read(returned)
            path = pin(dest / (name + ".npz"), item["file_sha256"])
            kind = "NEW"
        else:
            status = "TECHNICAL_ENTERED" if entered.is_file() else (
                "GG_REUSE_UNVERIFIED" if condition == "GG" else "UNSTARTED")
            records.append(dict(**cell, status=status))
            continue
        aperture = cold["evidence"][cell["anchor"]][condition]["right_query_aperture"]
        records.append(dict(**cell, status="RETURNED", origin=kind, artifact_sha256=digest(path),
                            **measure(path, item["raw_sha256"], item["decoded_sha256"], aperture)))
    counts = dict(
        new_returned=sum(r.get("origin") == "NEW" for r in records),
        exact_reused=sum(r.get("origin") == "EXACT_T2_REUSE" for r in records),
        technical_entered=sum(r["status"] == "TECHNICAL_ENTERED" for r in records),
        unstarted=sum(r["status"] == "UNSTARTED" for r in records),
        gg_reuse_unverified=sum(r["status"] == "GG_REUSE_UNVERIFIED" for r in records),
    )
    require(counts["new_returned"] == owner["returned"]
            and counts["technical_entered"] + counts["new_returned"] == owner["entered"]
            and sum(counts.values()) == 16, "T3 actual denominator differs")
    comparisons, totals = summarise(records)
    write_new(output, dict(
        status="T3_FACTORIAL_CLOSED", panel=PANEL, counts=counts, by_condition=totals,
        complete_pair_seed_comparisons=comparisons, rows=records, owner=owner,
        allocation_sha256=start["allocation_sha256"], cold_sha256=start["cold_sha256"],
        physical_gate="UNSCORED_MATCHED_SOURCE_CODEC_MISSING",
        native_starts=0, physics_ticks=0, optimizer_updates=0,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("site", "cold", "output"):
        parser.add_argument(arg, type=Path)
    a = parser.parse_args()
    os.umask(0o077)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU offline score")
    score(a.site.resolve(), a.cold, a.output)
