"""Bind continued Recovery work to the two closed, independently registered scopes."""
from pathlib import Path

from v64_recovery_followup import FIRST_ROSTER, ROSTER as SECOND_CASES
from v64_recovery_roster import PLAN, QUERIES, ROSTER as FIRST_CASES, TICKS, fixed, sha


SECOND_ROSTER = "4018539bb1f083322450b2fd0bdfdc76380f41a91aa926225e47c2a17ed77959"
FIRST_NOISE = "8ddc137196a1b035d9b835b9397bef51eff86d6037ee55de47c06131dce6e89e"
SECOND_NOISE = "ccd74ac999aa16fc29cad8a7e57d6fc097207c8cbcc1274bf77d736dba0f3cf6"
NOISE = {FIRST_ROSTER: FIRST_NOISE, SECOND_ROSTER: SECOND_NOISE}
CASES = {FIRST_ROSTER: FIRST_CASES, SECOND_ROSTER: SECOND_CASES}


def read_roster(path, expected=None):
    digest = sha(path)
    if digest not in CASES or (expected is not None and digest != expected):
        raise ValueError("Unknown or changed Recovery TRAIN roster")
    manifest = fixed(path, digest)
    signature = tuple((c["id"], c["original_teacher"], c["task"]) for c in manifest["cases"])
    if (manifest["plan"] != PLAN or manifest["split"] != "TRAIN" or signature != CASES[digest]
            or manifest["max_native_requests"] != 2 or manifest["max_native_ticks"] != 2*TICKS
            or not all(row["max_ticks"] == TICKS and row["maximum_queries"] == QUERIES for row in manifest["cases"])):
        raise ValueError("The fixed Recovery identities, task family or execution bounds changed")
    return manifest, digest


def roster_ref(reference):
    return read_roster(Path(reference["path"]), reference["sha256"])
