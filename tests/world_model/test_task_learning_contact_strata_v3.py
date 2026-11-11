import pytest

from go2arx5_vla.world_model.task_learning_contact_strata_v3 import episode_strata,active_sides


def test_ordered_one_sided_bilateral_and_support_absence_use_transition_duration():
    table='task_table / task_cube_geom';left='task_left_pad / task_cube_geom';right='task_right_pad / task_cube_geom'
    row=dict(episode='e',trace_sha256='a',transitions=[dict(native_step=5704,pairs=[table]),
        dict(native_step=15250,pairs=[right,table]),dict(native_step=15253,pairs=[left,right,table]),
        dict(native_step=15997,pairs=[left,right])],contact_sample_counts={left:1000000})
    result=episode_strata(row)
    assert result['first_pad_asymmetry_s']==.006
    assert result['support_contact_disappearance_events']==1
    assert result['maximum_recorded_uninterrupted_bilateral_without_support_s']==(21205-15997)*.002
    assert result['events'][0]['after_pad_state']=='RIGHT_ONLY'
    assert result['events'][1]['after_pad_state']=='BILATERAL'
    assert result['native_grasp_or_hold_certificate']is False


def test_unknown_geometry_is_not_relabelled_as_a_pad():
    with pytest.raises(ValueError,match='Unknown contact geometry'):
        active_sides(['x5_link6 / task_cube_geom'])
