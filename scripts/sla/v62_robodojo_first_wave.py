"""Pin, read, and seal a small disjoint RoboDojo wave for the V6.2 source expansion."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from v6_robodojo_intake import MAX_MEMBER, R200_CORE_SHA, TASKS, digest, population, rank, read, wave
from v6_robodojo_pilot import REPO, REV
from v6_robodojo_seal import seal
from v6_ville_pilot import need, write


PREVIOUS_ROSTER = "476dbe0e03ed309170ba76a333f19ff1b1fc18a7e2ac3bafff4bbff1696ec496"
PREVIOUS_EXTENSION = "fb33c0da28af0359b3b157d9288cd2b6fb3ac6c59d12d20f2e859a33f5a01317"
SEED = "SLA-V62-RD-FIRST12-20261127"


def prepare(old_roster, baseline, output):
    need(digest(old_roster) == PREVIOUS_ROSTER and digest(baseline) == R200_CORE_SHA,
         "Expected both historical split registrations")
    old = read(old_roster)
    protected = set(row["raw_sha256"] for row in read(baseline)["episodes"])
    protected.update(row["sha256"] for row in old["protected_test_members"] + old["candidates"])
    need(old["revision"] == REV and old["repo"] == REPO and old["license"] == "Apache-2.0",
         "Official RoboDojo lineage changed")
    with ThreadPoolExecutor(max_workers=3) as pool:
        episodes = list(pool.map(population, TASKS))
    groups = [rank([r for r in rows if r["sha256"] not in protected and 0 < r["bytes"] <= MAX_MEMBER], SEED)[:2]
              for rows in episodes]
    candidates = [row for group in zip(*groups, strict=True) for row in group]
    need(len(candidates) == len({row["sha256"] for row in candidates}) == 12,
         "Expected two new episodes per source task before reading labels")
    write(output, dict(repo=REPO, revision=REV, license="Apache-2.0", baseline_sha256=R200_CORE_SHA,
                       protected_test_members=old["protected_test_members"], prior_roster_sha256=PREVIOUS_ROSTER,
                       seed=SEED, candidates=candidates, raw_bytes=sum(row["bytes"] for row in candidates),
                       admitted=0))


def run(old_roster, baseline, stats, preceding, output):
    need(digest(preceding) == PREVIOUS_EXTENSION, "Expected the sealed 107-episode V6 expansion")
    output.mkdir(mode=0o700)
    registered = output / "ROSTER.json"
    prepare(old_roster, baseline, registered)
    receipt_sha = digest(registered)
    wave(registered, receipt_sha, 0, 12, output / "RAW-WAVE", actual_capacity=True)
    result = output / "RAW-WAVE/ACTUAL.json"
    seal(registered, receipt_sha, baseline, stats, output / "READY-SOURCE",
         [(result, digest(result))], preceding=(preceding, PREVIOUS_EXTENSION))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("old_roster", "baseline", "stats", "preceding", "output"):
        parser.add_argument(field, type=Path)
    run(**vars(parser.parse_args()))
