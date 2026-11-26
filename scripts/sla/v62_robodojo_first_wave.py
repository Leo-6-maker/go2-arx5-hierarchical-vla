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
FIRST_ROSTER = "7c8490d8792d19933faed138cd7936a643e38a89bc070552dfe97187af7ff5d7"
FIRST_RELEASE = "b0e872a084a859ec3ad65221a6ab764bead57f23d2837374fce4042146ec9b0b"
SECOND_SEED = "SLA-V62-RD-NEXT18-20261127"
SECOND_ROSTER = "78eb7f0d979440296d633740d6d205798fdb349a1d8e9981f8a5d077a7b618a5"
SECOND_RELEASE = "86531eb99e58c43e8a4505f46ed5da9ab30306601d30efeba2de68ab8825c236"
THIRD_SEED = "SLA-V62-RD-NEXT24-20261127"


def prepare(old_roster, baseline, output, previous_waves=()):
    need(digest(old_roster) == PREVIOUS_ROSTER and digest(baseline) == R200_CORE_SHA,
         "Expected both historical split registrations")
    old = read(old_roster)
    protected = set(row["raw_sha256"] for row in read(baseline)["episodes"])
    protected.update(row["sha256"] for row in old["protected_test_members"] + old["candidates"])
    for previous, expected in zip(previous_waves, (FIRST_ROSTER, SECOND_ROSTER)):
        roster = previous / "ROSTER.json"
        need(digest(roster) == expected, "A preceding V6.2 sample allocation changed")
        protected.update(row["sha256"] for row in read(roster)["candidates"])
    per_task = 2 + len(previous_waves)
    seed = (SEED, SECOND_SEED, THIRD_SEED)[len(previous_waves)]
    need(old["revision"] == REV and old["repo"] == REPO and old["license"] == "Apache-2.0",
         "Official RoboDojo lineage changed")
    with ThreadPoolExecutor(max_workers=3) as pool:
        episodes = list(pool.map(population, TASKS))
    groups = [rank([r for r in rows if r["sha256"] not in protected and 0 < r["bytes"] <= MAX_MEMBER], seed)[:per_task]
              for rows in episodes]
    candidates = [row for group in zip(*groups, strict=True) for row in group]
    need(len(candidates) == len({row["sha256"] for row in candidates}) == per_task*len(TASKS),
         "Expected the exact balanced population before reading labels")
    registration = dict(repo=REPO, revision=REV, license="Apache-2.0", baseline_sha256=R200_CORE_SHA,
                        protected_test_members=old["protected_test_members"], prior_roster_sha256=PREVIOUS_ROSTER,
                        seed=seed, candidates=candidates, raw_bytes=sum(row["bytes"] for row in candidates),
                        previous_v62_roster_sha256=FIRST_ROSTER if previous_waves else None, admitted=0)
    if len(previous_waves) == 2:
        registration["previous_v62_roster_sha256"] = [FIRST_ROSTER, SECOND_ROSTER]
    write(output, registration)


def run(old_roster, baseline, stats, preceding, output, previous_wave=None, second_wave=None):
    need(digest(preceding) == PREVIOUS_EXTENSION, "Expected the sealed 107-episode V6 expansion")
    need(second_wave is None or previous_wave is not None, "Lock both preceding V6.2 releases in order")
    previous_waves = tuple(path for path in (previous_wave, second_wave) if path is not None)
    old_releases = [(preceding, PREVIOUS_EXTENSION)]
    for previous, expected in zip(previous_waves, (FIRST_RELEASE, SECOND_RELEASE)):
        release = previous / "READY-SOURCE/MANIFEST.private.json"
        need(digest(release) == expected, "A preceding verified V6.2 source release changed")
        old_releases.append((release, expected))
    output.mkdir(mode=0o700)
    registered = output / "ROSTER.json"
    prepare(old_roster, baseline, registered, previous_waves)
    receipt_sha = digest(registered)
    wave(registered, receipt_sha, 0, len(read(registered)["candidates"]), output / "RAW-WAVE", actual_capacity=True)
    result = output / "RAW-WAVE/ACTUAL.json"
    seal(registered, receipt_sha, baseline, stats, output / "READY-SOURCE",
         [(result, digest(result))], preceding=old_releases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("old_roster", "baseline", "stats", "preceding", "output"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--previous-wave", type=Path)
    parser.add_argument("--second-wave", type=Path)
    run(**vars(parser.parse_args()))
