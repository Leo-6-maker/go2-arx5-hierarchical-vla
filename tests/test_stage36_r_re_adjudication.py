from __future__ import annotations

from scripts import stage36_r_re_adjudication as stage36_r


def test_stage36_r_soft_joint_margin_is_not_a_hard_failure() -> None:
    row = {
        "reference_complete": True,
        "joint_limit_violation": True,
        "joint_limit_termination": False,
        "termination_code": "REFERENCE_COMPLETE",
    }
    flags = stage36_r.hard_flags_independent(row)
    assert not any(flags.values())
    assert stage36_r.true_termination_code([flags], True) == "REFERENCE_COMPLETE"


def test_stage36_r_true_joint_limit_termination_is_a_hard_failure() -> None:
    row = {
        "reference_complete": False,
        "joint_limit_violation": False,
        "joint_limit_termination": True,
        "termination_code": "JOINT_LIMIT",
    }
    flags = stage36_r.hard_flags_independent(row)
    assert flags["joint_limit_termination"] is True
    assert stage36_r.true_termination_code([flags], False) == "JOINT_LIMIT"


def test_stage36_r_scope_constants_are_frozen() -> None:
    assert stage36_r.PI_GATE_ID == "PI_GATE_STATE_0025"
    assert stage36_r.EXPECTED_ORIGINAL_ROW_COUNT == 32
    assert len(stage36_r.EXPECTED_ROW_IDS) == 32
    assert stage36_r.STAGE == "Stage36-R"
