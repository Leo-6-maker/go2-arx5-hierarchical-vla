"""Bind the ten V6.5 comparisons to closed, originally registered L1 inputs."""

import argparse
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write


PLAN = "d7f4c1357fd9a48a6f3612c52c63172195a1da60"
STUDIES = (
    ("N02_HOLD", "ADAPT0250-NATIVE-HOLD-01", "5d02fddf7b8bc14fe759f8b5efc8e2b094fff1cc3b084746eebafa8995da3cf9", "LONG_HOLD"),
    ("PLACE", "ADAPT0250-NATIVE-PLACE-01", "da7fe283b4c0efe101dae94dc80513c53742d156b120b290d1fcc1413f6c472b", "LONG_PLACE"),
    ("INDEPENDENT_HOLD", "HOLD-R4_DEV_0200", "3ae6430f1aab24d81c5a4b37064f28db4ff8d03519a792384d0b0cf5cc2eedbe", "LONG_HOLD"),
)
HOLD_ROSTER = "c2af53f5b70340d1543082adbdbb769ea8e0e66231ac63d56059485136669605"
NOISE = {"LONG_HOLD": "fd542cdfcba6d3c205b6449168e1cc665a7168fdb81e64458f08b6471cb90fb3",
         "LONG_PLACE": "69f4997392ae291bdc50bc2b8699fd0ccac43e9a19b6bbbc7809dee6d92ade0d"}
CELLS = {
    "N02_HOLD": [("N02", 201), ("N02", 202)],
    "PLACE": [("P01", 701), ("P02", 702), ("P02", 701), ("P01", 702)],
    "INDEPENDENT_HOLD": [("N03", 201), ("N03", 202), ("VAL01", 201), ("VAL01", 202)],
}


def reference(path):
    return dict(path=str(path), sha256=sha(path))


def bind_site(base, paired_hold, output):
    paired = read(pin(paired_hold, HOLD_ROSTER))
    pairs = {(row["slot"], row["noise"]): row for row in paired["rows"]}
    require(paired["link"] == "L1" and paired["interval_s"] == .002 and set(pairs) == set(CELLS["INDEPENDENT_HOLD"])
            and paired["noise"]["sha256"] == NOISE["LONG_HOLD"], "Independent HOLD identity drift")
    groups = {}
    for cohort, name, expected, phase in STUDIES:
        root = base / name
        comparison = read(pin(root / "COMPARISON.json", expected))
        rows = comparison["schedule"]
        require(comparison["v4_block"] == phase and comparison["gpu_noise_sha256"] == NOISE[phase]
                and read(root / "control/LEASE.json")["status"] == "CLOSED"
                and read(root / "worker/control/LEASE.json")["status"] == "CLOSED"
                and [(e["layout_slot"], e["noise_label"]) for e in rows] == CELLS[cohort], "The frozen matched roster changed")
        panel_ref = comparison["v4_panel"]
        panel = read(pin(panel_ref["path"], panel_ref["sha256"]))
        noise = panel["noise"][phase.removeprefix("LONG_").lower()]
        require(noise["sha256"] == NOISE[phase], "Wrong original CUDA noise bank")
        pin(noise["path"], noise["sha256"])
        group = []
        for row in rows:
            case = root / row["id"]
            contract, result = (case / m for m in ("EXECUTION_SPEC.json", "RESULT.private.json"))
            lease = read(case / "control/LEASE.json")
            old = read(pin(contract, lease["spec_sha256"]))
            closed = read(result)
            require(lease["status"] == "CLOSED" and closed["physical_requests"] == 1 and closed["shutdown_clean"]
                    and not closed["technical_failure"] and old["interface_comparison_link"] == "L1"
                    and old["comparison_policy_seed"] == row["policy_seed"] and old["max_ticks"] == row["max_ticks"]
                    and old["policy_queries"] == row["policy_queries"], "Measured prior card cannot anchor a matched comparison")
            scene = old["frozen_files"][old["model_path"]]
            pin(old["model_path"], scene)
            slot, noise_label = row["layout_slot"], row["noise_label"]
            if cohort == "INDEPENDENT_HOLD":
                require(scene == paired["slots"][slot]["scene"]["sha256"]
                        and row["policy_seed"] == pairs[slot, noise_label]["policy_seed"], "Independent geometry or seed changed")
            group.append(dict(slot=slot, noise_label=noise_label, seed=row["policy_seed"], task=old["task_instruction_id"],
                              prior_comparison=reference(result), original_contract=reference(contract), scene_sha256=scene,
                              max_ticks=row["max_ticks"], max_queries=row["policy_queries"], original_id=row["id"]))
        groups[cohort] = dict(archive_sha256=expected, original_noise=noise, cells=group)
    count = [row for g in groups.values() for row in g["cells"]]
    require(len(count) == 10 and len({(r["scene_sha256"], r["seed"]) for r in count}) == 10,
            "Duplicate evaluation scene/noise identity")
    write(output, dict(plan=PLAN, formal_step=1000, groups=groups, original_hold_roster_sha256=HOLD_ROSTER,
                       caps=dict(physical_requests=len(count), integrations=sum(r["max_ticks"] for r in count),
                                 model_queries=sum(r["max_queries"] for r in count)),
                       launched_requests=0, physics_ticks=0, model_queries=0, stage="ARCHIVED_MATCHED_INPUTS_VERIFIED"))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "paired_hold", "output"):
        cli.add_argument(key, type=Path)
    args = cli.parse_args()
    require(not args.output.exists(), "Preserve existing matched protocol evidence")
    bind_site(**vars(args))
