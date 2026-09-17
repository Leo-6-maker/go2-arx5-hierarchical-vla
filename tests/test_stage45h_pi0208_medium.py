import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "stage45h_pi0208_medium.py"
SPEC = importlib.util.spec_from_file_location("stage45h_pi0208_medium", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_cohorts_are_frozen_and_disjoint():
    manifest = MODULE.cohort_manifest()
    assert manifest["collection_count"] == 128
    assert manifest["evaluation_count"] == 16
    assert len(set(manifest["collection_condition_order"])) == 128
    assert not set(manifest["collection_condition_order"]).intersection(manifest["evaluation_condition_order"])
    assert all(delta[2] == 0.0 for delta in manifest["collection_deltas_m"].values())


def test_matched_schedule_replaces_exactly_400_positions():
    rows = [f"row-{index:03d}" for index in range(512)]
    schedule = MODULE.make_schedule(rows, 7101)
    assert len(schedule["MCTRL"]) == 2000
    assert len(schedule["MSUP"]) == 2000
    assert sum(slot["source"] == "FULL24" for slot in schedule["MCTRL"]) == 2000
    assert sum(slot["source"] == "STATE_SUPPORT_MEDIUM" for slot in schedule["MSUP"]) == 1600
    support = [slot["support_row_id"] for slot in schedule["MSUP"] if slot["source"] == "STATE_SUPPORT_MEDIUM"]
    assert len(support) == 400
    assert len(set(support)) == 400
    for control, supervised in zip(schedule["MCTRL"], schedule["MSUP"]):
        if supervised["source"] == "FULL24":
            assert control["full24_slot"] == supervised["full24_slot"]

