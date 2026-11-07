"""Mechanical versus semantic admission; synthetic only, no native PASS."""
import numpy as np
import pytest
from go2arx5_vla.world_model import fusion_bounded_reference_tracking_v1 as mechanical
from go2arx5_vla.world_model import fusion_graded_short_skill_v1 as strict
from go2arx5_vla.world_model.fusion_trajectory_adapter import TrajectoryAdaptation
from test_fusion_graded_short_skill_v1 import proposal


def test_bounded_away_command_can_be_examined_without_reclassifying_task_failure():
    program,cube=proposal('away');before=program.evidence()
    value=mechanical.screen(program,cube)
    assert value['eligible_for_original_native_certificate']
    assert not value['strict_short_proposal_screen']['qualified']
    assert not strict.tracking_diagnostic_screen(program,cube)['eligible_for_original_native_certificate']
    assert not value['strict_short_task_success'] and not value['native_execution_authorized']
    assert not value['native_safety_established'] and program.evidence()==before


@pytest.mark.parametrize('change',['late-close','large-middle','near-cube'])
def test_entire_mechanical_path_keeps_open_bounded_noncontact_domain(change):
    program,cube=proposal(change)
    assert not mechanical.screen(program,cube)['eligible_for_original_native_certificate']


def test_late_rotation_outside_domain_is_not_hidden_by_small_translation():
    program,cube=proposal();raw=np.array(program.decoded_H40)
    angle=.11;c,s=np.cos(angle),np.sin(angle)
    raw[-1,3:9]=[c,-s,0.,s,c,0.]
    rotated=TrajectoryAdaptation(raw,program.query_TCP_source,program.T_source_from_target,
        program.decision_TCP_target,program.initial_reference_TCP_target,program.base_pose_hold,1.)
    value=mechanical.screen(rotated,cube)
    assert not value['predicates']['orientation'] and not value['eligible_for_original_native_certificate']
