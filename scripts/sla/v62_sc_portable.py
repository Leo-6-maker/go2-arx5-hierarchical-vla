"""Build three portable H30 source pools with TRAIN-only normalization per task."""

import argparse
from pathlib import Path

from v61_ville_dataset import seal_cohorts
from v62_sc_source import CAMERAS, HORIZON, NATIVE_HZ, SCHEMA, SOURCE_TASKS, checked
from v6_robodojo_intake import digest, read
from v6_ville_pilot import need, write


CATALOGUE = "165f388dabc4f6d5ecd147f709c1749ee726daa820a828dc560ef985f936877f"
TASK_COUNTS = (("peg", 14, 4), ("ball", 24, 6), ("stack", 13, 4))


def build(catalogue, output):
    registry = read(checked(catalogue, CATALOGUE))
    selected = {key: [] for key in SOURCE_TASKS}
    for version in registry["versions"]:
        if version["domain"] != "sc_vla":
            continue
        source, ready = version["manifest"], version["core_ready"]
        manifest = read(checked(Path(source["path"]), source["sha256"]))
        receipt = read(checked(Path(ready["path"]), ready["sha256"]))
        need(receipt["manifest_sha256"] == source["sha256"] and receipt["by_split"]["TRAIN"]["episodes"] == version["episodes"],
             "A source cohort lacks its real Corelab reader")
        selected[manifest["task"]].append(version)
    output.mkdir(mode=0o700)
    records = []
    for task, (name, train, dev) in zip(SOURCE_TASKS, TASK_COUNTS, strict=True):
        path = output / name
        seal_cohorts(selected[task], path, {"TRAIN": train, "SOURCE_DEV": dev}, schema=SCHEMA,
                     native_hz=NATIVE_HZ, horizon=HORIZON, cameras=CAMERAS)
        receipt = read(path / "CORE_READER.json")
        records.append(dict(task=task, component=name, manifest_sha256=digest(path / "MANIFEST.json"),
                            reader_sha256=digest(path / "CORE_READER.json"), by_split=receipt["by_split"]))
    write(output / "ACTUAL.json", dict(catalogue_sha256=CATALOGUE, distinct_source_rows=65,
                                       scopes=records, hopper_reader=False, optimizer_visits=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalogue", type=Path)
    parser.add_argument("output", type=Path)
    build(**vars(parser.parse_args()))
