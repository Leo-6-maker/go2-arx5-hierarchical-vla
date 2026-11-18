import numpy as np

from go2arx5_vla.world_model.mobile_continuous_teacher_v2 import rotation_increment
from go2arx5_vla.world_model.multitask_contact_geometry_d3 import (
    AttachmentWitness, SUPPORT_CONFIRM_FRAMES, SupportClock, relative_margin,
)
from go2arx5_vla.world_model.multitask_placement_geometry_d2 import transform


def test_held_relative_margin_is_invariant_to_shared_world_motion_and_detects_slip():
    first = AttachmentWitness(transform(np.eye(3), [.1, .2, .3]),
                              transform(np.eye(3), [.3, .5, .7]),
                              transform(np.eye(3), [.4, .7, .8]), np.array([.02, .02]))
    rigid = transform(rotation_increment(np.array([.13, -.09, .04])), [-.11, .02, .05])
    follow = AttachmentWitness(rigid @ first.mount, rigid @ first.tcp,
                               rigid @ first.object_pose, first.fingers)
    expected = first.tcp
    unchanged = first.relative_residual(follow, expected, .08)
    assert np.allclose(unchanged, [0., 0., 0.], atol=1e-12)

    moved = follow.object_pose.copy()
    moved[2, 3] += .003
    slipping = AttachmentWitness(follow.mount, follow.tcp, moved, np.array([.019, .021]))
    increased = first.relative_residual(slipping, expected, .08)
    assert np.allclose(increased, [0., .003, .001], atol=1e-12)
    assert np.isclose(relative_margin(.004, [increased], .08)
                      - relative_margin(.004, [unchanged], .08), .004)


def test_support_requires_contiguous_actual_contact_and_resets_on_loss():
    clock = SupportClock('red', {'object_half_sizes': {'red': [.02, .02, .02]},
                                 'table_top_world_z': .60})
    obj = {'pose': transform(np.eye(3), [.4, .1, .62]), 'table_contact': True,
           'linear_velocity': np.zeros(3)}
    for tick in range(SUPPORT_CONFIRM_FRAMES - 1):
        clock.observe({'native_step': tick, 'objects': {'red': obj}})
    assert not clock.confirmed(tick)
    clock.observe({'native_step': tick + 2, 'objects': {'red': obj}})
    assert clock.streak == 1
    for t in range(tick + 3, tick + SUPPORT_CONFIRM_FRAMES + 2):
        clock.observe({'native_step': t, 'objects': {'red': obj}})
    assert clock.confirmed(t)
    clock.observe({'native_step': t + 1, 'objects': {'red': dict(obj, table_contact=False)}})
    assert not clock.confirmed(t + 1)
