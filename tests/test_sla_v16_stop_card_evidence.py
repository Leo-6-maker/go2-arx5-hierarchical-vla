from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts/sla_hierarchical_v16")
)
from stop_card_evidence import terminal_evidence


def evidence():
    trace = [{"kind": 12, "token": k, "ticket": k - 1} for k in range(1, 6440)] + [
        {
            "kind": 13,
            "ticket": 6439,
            "token": 6439,
            "sim_time_s": 12.878,
            "values": [1.0, 157.0],
            "monotonic_ns": 9_000_000_000,
        }
    ]
    audit = {
        "status": "RECORDED_TRACE_CHECKS_PASS",
        "native_step_records": 6439,
        "integrations_after_first_STOP": 0,
        "writer": {"events": 6440, "written": 6440, "dropped": 0, "stop_latched": True},
    }
    return dict(
        trace=trace,
        audit=audit,
        last_grant={"requested_step": 6453, "monotonic_s": 8.0},
        installs=[{"monotonic_s": 7.0}],
        exits=[
            {"name": s, "exit_code": 0}
            for s in ("sim", "planner", "controller", "solver")
        ],
        absent=True,
    )


def test_joined_event_and_process_terminal():
    assert terminal_evidence(**evidence())["after_stop_ticks"] == 0


@pytest.mark.parametrize(
    "fault",
    [
        "receiver_only",
        "duplicate",
        "wrong_time",
        "different_command",
        "step_after",
        "incomplete_state",
        "late_grant",
        "new_install",
        "solver_alive",
        "orphan",
        "lost_trace",
    ],
)
def test_individual_missing_evidence_rejects(fault):
    v = evidence()
    if fault == "receiver_only":
        v["trace"] = []
    if fault == "duplicate":
        v["trace"] += deepcopy(v["trace"])
    if fault == "wrong_time":
        v["trace"][-1]["sim_time_s"] += 0.002
    if fault == "different_command":
        v["trace"][-1]["values"] = [1.0, 156.0]
    if fault == "step_after":
        v["audit"]["integrations_after_first_STOP"] = 1
    if fault == "incomplete_state":
        v["audit"]["native_step_records"] -= 1
    if fault == "late_grant":
        v["last_grant"]["monotonic_s"] = 9.0
    if fault == "new_install":
        v["installs"].append({"monotonic_s": 9.0})
    if fault == "solver_alive":
        v["exits"][-1]["exit_code"] = None
    if fault == "orphan":
        v["absent"] = False
    if fault == "lost_trace":
        v["audit"]["writer"]["dropped"] = 1
    with pytest.raises(ValueError):
        terminal_evidence(**v)


def test_distinct_grounded_feet_at_native_contact_time():
    from fault_reference import floor_feet

    names = ("floor", "FL", "FR", "RL", "RR", "task_cube_geom", "task_table")
    packet = {
        "step": 6439,
        "contact_time_s": 6438 * 0.002,
        "contacts": [
            {"geom0": a, "geom1": b}
            for a, b in ((0, 1), (1, 0), (2, 0), (3, 0), (5, 6))
        ],
    }
    assert floor_feet(names, packet) == ("FL", "FR", "RL")
    with pytest.raises(ValueError, match="clock"):
        floor_feet(names, {**packet, "contact_time_s": 6439 * 0.002})
    with pytest.raises(ValueError, match="geometry"):
        floor_feet(names, {**packet, "contacts": [{"geom0": -1, "geom1": 0}]})
