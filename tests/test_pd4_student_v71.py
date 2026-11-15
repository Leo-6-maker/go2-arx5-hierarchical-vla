import pytest
from go2arx5_vla.world_model import task_learning_grasp_v1 as task
from go2arx5_vla.world_model import pd4_student_v71 as p


def test_PD4_has_distinct_typed_500_identity_and_keeps_raw40_P10_clock():
    c=task.protocol(task.SLA_PD4_ROLE)
    assert task.native_body_profile(task.SLA_PD4_ROLE)is p
    assert task.student_role(task.SLA_PD4_ROLE)and task.sla_role(task.SLA_PD4_ROLE)and task.dense_sla_role(task.SLA_PD4_ROLE)
    assert (c['prefix_atoms'],c['policy_queries'],c['final_hold_atoms'])==(10,60,40)
    assert c['complete_H40_gate_before_every_prefix'] and c['full_H40_preview']
    assert p.ENDPOINT==21703 and c['stage']==p.PROTOCOL


def test_legacy3K_policy_cannot_be_called_a_PD4_500_candidate():
    with pytest.raises(PermissionError,match='PD4_ONLY'):
        p.policy_contract({'arm':'N20_FM3K_RAW_P10','package':{},'Native_body_qualification':{}},expected_global_step=500)
    with pytest.raises(PermissionError,match='PD4_ONLY'):
        p.policy_contract({'arm':'PD4_FINAL_PLAN','package':{},'qualification':{}},expected_global_step=3000)
