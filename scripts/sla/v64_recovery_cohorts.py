"""Bind continued Recovery work to independently registered scopes."""
from pathlib import Path

from v64_recovery_followup import (
    FIRST_ROSTER,
    FOURTH_ROSTER as FOURTH_CASES,
    OPEN_ROSTER as OPEN_CASES,
    ROSTER as SECOND_CASES,
)
from v64_recovery_roster import PLAN, QUERIES, ROSTER as FIRST_CASES, TICKS, fixed, sha


SECOND_ROSTER = "4018539bb1f083322450b2fd0bdfdc76380f41a91aa926225e47c2a17ed77959"
OPEN_ROSTER = "c8af07809c2e49ce1c2ec3f135e57710339008aaccfc4349e4988a22428639e4"
FOURTH_ROSTER = "ec398ae3b43222bdfd5c1d075b7827ea017b6cda1ef6401ee9987dbd7be3bb0d"
FIRST_NOISE = "8ddc137196a1b035d9b835b9397bef51eff86d6037ee55de47c06131dce6e89e"
SECOND_NOISE = "ccd74ac999aa16fc29cad8a7e57d6fc097207c8cbcc1274bf77d736dba0f3cf6"
OPEN_NOISE = "fe1dcf4529d77a822da1ef220bab0c04d41f1d306863b8ca2182f7dcbeccb30a"
NOISE = {FIRST_ROSTER: FIRST_NOISE, SECOND_ROSTER: SECOND_NOISE, OPEN_ROSTER: OPEN_NOISE}
CASES = {FIRST_ROSTER: FIRST_CASES, SECOND_ROSTER: SECOND_CASES, OPEN_ROSTER: OPEN_CASES, FOURTH_ROSTER: FOURTH_CASES}


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
