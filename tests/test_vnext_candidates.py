from copy import deepcopy
import pytest
from scripts.pi0294.multitask import vnext_candidates as v
from scripts.pi0294.multitask.libero_book_teacher import route_card,RELEASE_REVISION
from scripts.pi0294.multitask.cabinet_braked_turn import cards


def test_fixed_twelve_do_not_claim_data_or_reuse_old_position_geometries():
    p=v.specification();rows=p['candidates']
    assert len(rows)==len({r['id']for r in rows})==12
    assert all(r['split']=='TRAIN'for r in rows)
    assert sum(r['kind']=='CONTACT_EVENT_RECOVERY'for r in rows)==4
    assert p['maximum_attempts_per_candidate']==1 and p['replacement_seed_after_rejection']is False
    assert not p['added_to_shared59_D2_or_PB'] and p['qualified_demonstrations']==0
    assert all(not r['student_prefix_is_expert']and not r['DEV_or_TEST_states']and not r['native_adapter_bound']
               for r in rows if r['kind']=='CONTACT_EVENT_RECOVERY')


def test_geometry_retains_original_task_goal_safety_and_grasp_semantics():
    original=route_card(RELEASE_REVISION);door=cards()['right']
    for i in range(1,5):
        b=v.book_geometry_card(i);d=v.door_geometry_card(i)
        assert b['book']==original['book']and b['book_route']==original['book_route']
        assert b['task_duration_s']==d['task_duration_s']==32
        assert b['warmup_s']==d['warmup_s']==5
        for k in ('target_angle_rad','angle_limits_rad','object_to_tcp_grasp','approach_direction_object'):
            assert d['door'][k]==door['door'][k]
        assert d['turn_reference_limits']==door['turn_reference_limits']
    with pytest.raises(ValueError):v.book_geometry_card(5)
    with pytest.raises(ValueError):v.door_geometry_card(0)
