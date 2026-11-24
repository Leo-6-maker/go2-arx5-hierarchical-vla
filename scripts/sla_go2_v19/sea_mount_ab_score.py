"""Read the V1/V2 paired gripper forecasts from complete or interrupted runs."""

import argparse
import os
from pathlib import Path

import numpy as np

from sea_mount_ab import CAP, CONDITIONS, LAYOUTS, PANEL, binding
from sea_target_t2 import PANEL as T2_PANEL
from sea_t3_visual_state_score import measure
from source_custody import SEEDS, digest, pin, read, require, write_new
from policy_query import tensor_sha


def run(root, cold, output):
    proof = read(cold)
    site = root / "v19" / PANEL
    start, owner = (read(site / name) for name in ("START.json", "OWNER_EXIT.json"))
    require(start["panel"] == PANEL and start["cold_sha256"] == digest(cold) and proof["code"] == binding()
            and owner["project_lock_released"] and not Path("/proc", str(start["owner_pid"])).exists()
            and owner["physics_ticks"] == owner["optimizer_updates"] == 0
            and 0 <= owner["returned"] <= owner["entered"] <= CAP, "open owner or changed proof")
    manifest = site / "REUSE.json"
    reuse = read(manifest) if manifest.exists() else {}
    require(not reuse or reuse == proof["cpu_reuse_candidates"], "unexpected historical reuse")
    records = []
    for cell in proof["cells"]:
        name, cond = cell["id"], cell["condition"]
        enter, returned = (site / f"{name}.{suffix}.json" for suffix in ("entered", "returned"))
        if name in reuse:
            require(cond == "GG" and not enter.exists() and not returned.exists(), "V1 also sampled")
            cached = reuse[name]
            artifact = pin(root / "v19" / T2_PANEL / (cached["archived_slot"]+".npz"), cached["output_sha256"])
            source = "EXACT_V1_REUSE"
        elif returned.exists():
            e = read(enter)
            require(cond == "AUX2" and all(e[k] == v for k, v in cell.items()), "unexpected fresh cell")
            with np.load(site / f"noise-{cell['seed'][1]}.npz", allow_pickle=False) as z:
                require(tensor_sha(z["noise"]) == cell["initial_noise_sha256"]
                        and z["sample_key"].tolist() == e["sample_key"], "wrong actual paired sample")
            cached = read(returned)
            artifact = pin(site / (name+".npz"), cached["file_sha256"])
            source = "NEW_V2"
        else:
            status = "TECHNICAL_ENTERED" if enter.exists() else ("V1_REUSE_UNVERIFIED" if cond == "GG" else "UNSTARTED")
            records.append(dict(**cell, status=status))
            continue
        amounts = measure(artifact, cached["raw_sha256"], cached["decoded_sha256"],
                          proof["query_aperture"][cell["anchor"]])
        records.append(dict(**cell, **amounts, status="RETURNED", source=source,
                            artifact_sha256=digest(artifact)))
    by = {(r["anchor"], tuple(r["seed"]), r["condition"]): r for r in records if r["status"] == "RETURNED"}
    pairs = []
    for layout in LAYOUTS:
        for seed in SEEDS:
            v1, v2 = (by.get((layout, tuple(seed), c)) for c in CONDITIONS)
            if v1 and v2:
                pairs.append(dict(anchor=layout, seed=seed, v1_closes=v1["closes"], v2_closes=v2["closes"],
                                  change_in_closing_margin=v2["closing_margin"]-v1["closing_margin"]))
    counts = dict(
        v1_reused=sum(r.get("source") == "EXACT_V1_REUSE" for r in records),
        v2_returned=sum(r.get("source") == "NEW_V2" for r in records),
        technical_entered=sum(r["status"] == "TECHNICAL_ENTERED" for r in records),
        unstarted=sum(r["status"] == "UNSTARTED" for r in records),
        v1_reuse_unverified=sum(r["status"] == "V1_REUSE_UNVERIFIED" for r in records),
    )
    require(sum(counts.values()) == 8 and counts["v2_returned"] == owner["returned"]
            and counts["v2_returned"]+counts["technical_entered"] == owner["entered"], "paired denominator differs")
    write_new(output, dict(status="AUX_V2_PAIRED_FROZEN_CLOSED", panel=PANEL, counts=counts, pairs=pairs,
                           rows=records, owner=owner, cold_sha256=start["cold_sha256"],
                           allocation_sha256=start["allocation_sha256"], native_starts=0, physics_ticks=0,
                           new_rgb_renders=0, optimizer_updates=0, physical_source_tool_codec=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "cold", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    os.umask(0o077)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU score only")
    run(a.root.resolve(), a.cold, a.output)
