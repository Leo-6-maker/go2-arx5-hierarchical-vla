import pytest
from scripts.wm02.sla_n20_endpoint_followup_v5 import all_entry_rejections
from go2arx5_vla.world_model import task_learning_grasp_v1 as task


def test_only_a_full_all_entry_rejected_train8_defers_the_independent_population():
    rows=[dict(technical_valid=True,policy_proposal_rejected=True,actual_native_integrations=task.B)for _ in range(8)]
    assert all_entry_rejections(rows)
    assert not all_entry_rejections(rows[:7])
    rows[0]['actual_native_integrations']+=25
    assert not all_entry_rejections(rows)
    rows[0].update(actual_native_integrations=task.B,technical_valid=False)
    assert not all_entry_rejections(rows)
