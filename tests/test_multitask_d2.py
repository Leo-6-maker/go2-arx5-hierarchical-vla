import numpy as np

from go2arx5_vla.world_model.mobile_continuous_teacher_v2 import rotation_increment
from go2arx5_vla.world_model.multitask_placement_geometry_d2 import landing_tcp, remaining_placement, transform
from scripts.wm02.multitask_d2_campaign import matrix_still_possible


def test_touchdown_uses_measured_se3_and_rotated_support_footprint():
    attachment = transform(rotation_increment([.1,.2,0]), [.04,0,-.015])
    rotation = rotation_increment([.1,-.22,-.35])
    half = np.array([.02,.025,.021])
    tcp = landing_tcp(attachment,[.4,.12],rotation,half,.61)
    lowered = tcp@attachment
    assert np.allclose(lowered[:2,3],[.4,.12])
    assert np.isclose(lowered[2,3]-abs(lowered[2,:3])@half,.61)
    initial=tcp.copy()
    initial[2,3]+=.04
    knots=remaining_placement(initial,initial@attachment,tcp,np.full(2,.020),np.zeros(2),
                              supported=False,withdrawal=[0,0,.06])
    assert np.allclose(knots[1].object_pose,lowered)
    assert np.allclose(knots[3].object_pose,lowered)
    assert np.isclose(np.linalg.norm(knots[3].tcp[:3,3]-tcp[:3,3]),.06)


def test_futility_keeps_in_flight_and_excludes_non_g7_success():
    cards = [dict(panel_entry_id=str(k),mobile_condition={'family':'F1'},
                  task_instruction_id='PLACE-red-left') for k in range(4)]
    complete = {'task_and_mobile_success':True,'G7_pass':True}
    rejected = {'task_and_mobile_success':True,'G7_pass':False}
    assert matrix_still_possible(cards,{'0':rejected,'1':rejected})
    assert not matrix_still_possible(cards,{'0':rejected,'1':rejected,'2':rejected})
    assert matrix_still_possible(cards,{'0':rejected,'1':rejected,'2':complete})
