"""Build one TRAIN-normalized source reader from the three verified Ville cohorts."""

import argparse
from collections import Counter
import os
from pathlib import Path

import numpy as np

from v6_robodojo_intake import digest, read
from v6_ville_pilot import need, write
from v6_ville_reader import preflight


def build(catalog, output):
    binding = read(catalog)
    need(binding["core_catalog_count"] == 337 and binding["unique_new_single"] == 30,
         "Expected the Core-verified 337-episode source version")
    return seal_cohorts(binding["single_arm_cohorts"], output, {"TRAIN": 30, "SOURCE_DEV": 6})


def seal_cohorts(datasets, output, expected_counts, **reader_contract):
    output.mkdir(mode=0o700)
    raw = output / "raw"
    pooled, seen = [], set()
    fit = {"state": [], "actions": []}
    combined = {}
    for cohort in datasets:
        table = Path(cohort["manifest"]["path"])
        need(digest(table) == cohort["manifest"]["sha256"], "A sealed Ville cohort changed")
        manifest = read(table)
        signature = {key: manifest[key] for key in (
            "schema", "license", "revision", "native_hz", "horizon", "task",
            "label_kind", "target_command", "camera_mask")}
        if not combined:
            combined = signature
        need(combined == signature, "Different source task or codec requires a separate pool")
        for episode in manifest["episodes"]:
            identity = episode["episode"]
            need(identity not in seen, "A Ville episode appeared in multiple cohorts")
            seen.add(identity)
            arrays = episode["arrays"]
            source = table.parent / arrays["file"]
            need(digest(source) == arrays["sha256"], "Sealed source data changed")
            os.link(source, output / arrays["file"])
            for view in episode["views"].values():
                if view["enabled"]:
                    member = Path(view["member"])
                    source = Path(manifest["video_root"]) / member
                    need(digest(source) == view["sha256"], "Sealed source video changed")
                    target = raw / member
                    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    os.link(source, target)
            if episode["split"] == "TRAIN":
                with np.load(output / arrays["file"], allow_pickle=False) as data:
                    for name in fit:
                        fit[name].append(data[name].reshape(-1, 7))
            pooled.append(episode)
    counts = Counter(row["split"] for row in pooled)
    need(counts == expected_counts, "Source split population changed")
    statistics = dict(split="TRAIN", schema=combined["schema"])
    for key, parts in fit.items():
        values = np.concatenate(parts)
        q = np.quantile(values, [.01, .25, .75, .99], axis=0)
        statistics[key] = dict(q01=q[0].tolist(), q99=q[3].tolist(), IQR=(q[2] - q[1]).tolist(), samples=len(values))
    statfile = output / "STATISTICS.json"
    write(statfile, statistics)
    manifest = output / "MANIFEST.json"
    write(manifest, dict(combined, episodes=pooled, video_root="raw", cohort_manifests=datasets,
                         statistics=dict(file=statfile.name, sha256=digest(statfile))))
    preflight(output, digest(manifest), raw, output / "CORE_READER.json", **reader_contract)
    print(digest(manifest), digest(output / "CORE_READER.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("output", type=Path)
    build(**vars(parser.parse_args()))
