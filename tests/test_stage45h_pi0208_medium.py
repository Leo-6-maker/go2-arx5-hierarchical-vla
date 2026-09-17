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
    assert manifest["collection_generation_rule"].endswith("phase pi/32; one pre-S0 root-state translation only")
    collection_points = {tuple(delta) for delta in manifest["collection_deltas_m"].values()}
    evaluation_points = {tuple(delta) for delta in manifest["evaluation_deltas_m"].values()}
    assert not collection_points.intersection(evaluation_points)


def test_medium_collection_binds_s25_base():
    assert MODULE.BASE_S25_LABEL == "S25@80"
    assert MODULE.BASE_S25_GLOBAL_STEP == 80
    assert MODULE.BASE_S25_TREE_SHA256 == "93b8cd12dd1aa34ae1be91aa0e39baeacae0e189c72ea83d48a764a93c796492"
    assert MODULE.BASE_S25_TREE_PREFIX == "pi0205_S25/checkpoint-80"


def test_matched_schedule_replaces_exactly_400_positions():
    rows = [f"row-{index:03d}" for index in range(512)]
    schedule = MODULE.make_schedule(rows, 7101)
    assert len(schedule["MCTRL"]) == 2000
    assert len(schedule["MSUP"]) == 2000
    assert sum(slot["source"] == "FULL24" for slot in schedule["MCTRL"]) == 2000
    assert sum(slot["source"] == "FULL24" for slot in schedule["MSUP"]) == 1600
    assert sum(slot["source"] == "STATE_SUPPORT_MEDIUM" for slot in schedule["MSUP"]) == 400
    support = [slot["support_row_id"] for slot in schedule["MSUP"] if slot["source"] == "STATE_SUPPORT_MEDIUM"]
    assert len(support) == 400
    assert len(set(support)) == 400
    assert schedule["support_pool_count"] == 512
    assert set(support).issubset(set(rows))
    assert set(support) != set(rows[:400])
    assert schedule["selected_support_row_ids_sha256"] == MODULE.canonical_sha(support)
    for control, supervised in zip(schedule["MCTRL"], schedule["MSUP"]):
        if supervised["source"] == "FULL24":
            assert control["full24_slot"] == supervised["full24_slot"]
