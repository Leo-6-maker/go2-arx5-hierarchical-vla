import json
import numpy as np
import pytest
from .task_path_reference import PersistentPathReference
from .task_gripper_contract import teacher_physical_fraction
from .task_path_planning import grasp_precontact_candidates, push_precontact_candidates


def path():
    poses = np.repeat(np.eye(4)[None], 21, axis=0)
    poses[:, 0, 3] = np.arange(21)*.001
    return poses


def reference():
    return PersistentPathReference(episode_id='CPU_FIXTURE_NOT_NATIVE',
        poses_world=path(), opening=teacher_physical_fraction(np.ones(21)), initial_native_step=2500)


def observe(r, index, actual=None, safe=True):
    return r.observe(native_step=2500+25*index, raw_time_s=5+index*.05,
        actual_tcp=path()[min(index, 20)] if actual is None else actual, native_safe=safe)


def test_prefix_installed_once_preserves_world_geometry_after_actual_perturbation():
    r = reference()
    observe(r, 0)
    first = r.chunk()
    with pytest.raises(ValueError, match='ONE_NORMAL'):
        r.chunk()
    r.acknowledge_install(ref_id=first['ref_id'], native_step=2500)
    for i in range(1, 8):
        observe(r, i)
        with pytest.raises(ValueError, match='ONE_NORMAL'):
            r.chunk()
    perturbed = path()[8].copy()
    perturbed[0, 3] -= .01
    observe(r, 8, perturbed)
    second = r.chunk()
    np.testing.assert_array_equal(second['action_EEF10'][0], first['action_EEF10'][8])
    assert second['action_EEF10'][0, 0] != perturbed[0, 3]
    assert len(r.intervals) == 8
    assert all(x['selected_MPC_policy_consumption'] == 'UNKNOWN' for x in r.intervals)
    assert first['BASE_supervision'] is False
    assert 'base' not in json.loads(first['wire'])


def test_wrong_ack_and_unacknowledged_advance_never_create_labels():
    r = reference()
    observe(r, 0)
    proposal = r.chunk()
    with pytest.raises(ValueError, match='INSTALL_ACK'):
        r.acknowledge_install(ref_id='another-reference', native_step=2500)
    with pytest.raises(PermissionError, match='NO_INSTALLED_PREFIX'):
        observe(r, 1)
    assert r.intervals == []
    with pytest.raises(PermissionError):
        r.acknowledge_install(ref_id=proposal['ref_id'], native_step=2500)


def test_missing_next_chunk_stops_instead_of_consuming_replaced_suffix():
    r = reference()
    observe(r, 0)
    chunk = r.chunk()
    r.acknowledge_install(ref_id=chunk['ref_id'], native_step=2500)
    for i in range(1, 9):
        observe(r, i)
    with pytest.raises(PermissionError, match='NO_INSTALLED_PREFIX'):
        observe(r, 9)
    assert len(r.intervals) == 8


def test_path_end_holds_but_does_not_transition_on_time():
    r = reference()
    for i in range(25):
        status = observe(r, i)
        if i % 8 == 0:
            c = r.chunk()
            r.acknowledge_install(ref_id=c['ref_id'], native_step=2500+25*i)
    assert status['path_end_time_reached']
    assert status['task_success'] is False
    assert status['stage_transition'] == 'REQUIRES_ACTUAL_GEOMETRY_NOT_PATH_TIMER'
    np.testing.assert_array_equal(c['action_EEF10'][:, :3], np.tile(path()[-1, :3, 3], (40, 1)))


def test_native_safety_and_wrong_start_or_clock_reject():
    r = reference()
    bad = path()[0].copy()
    bad[0, 3] = .1
    with pytest.raises(ValueError, match='REPLAN_REQUIRED'):
        observe(r, 0, bad)
    observe(r, 0)
    with pytest.raises(ValueError, match='EXACT_NATIVE_20HZ'):
        observe(r, 2)
    with pytest.raises(PermissionError, match='NATIVE_SAFETY'):
        observe(r, 1, safe=False)
    with pytest.raises(PermissionError):
        r.chunk()


def test_task_candidates_transform_with_tabletop_geometry_and_keep_all_outcomes():
    original = grasp_precontact_candidates(np.eye(4))
    world = np.eye(4)
    world[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    world[:3, 3] = [.2, -.1, .46]
    moved = grasp_precontact_candidates(world)
    assert [r['id'] for r in moved] == ['VERTICAL', 'TILT_NEG30', 'TILT_POS30']
    for a, b in zip(original, moved):
        np.testing.assert_allclose(b['goal_world'], world@a['goal_world'], atol=1e-14)
        assert not b['launch_allowed']
    tipped = np.eye(4)
    tipped[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    with pytest.raises(ValueError, match='UPRIGHT_TABLETOP'):
        grasp_precontact_candidates(tipped)


def test_push_candidates_follow_direction_and_measure_pad_support():
    obj = np.eye(4)
    obj[2, 3] = .46
    pads = [dict(center_tcp=[.01, sign*.02, -.01], axes_tcp=np.eye(3), half_size=[.01, .003, .02])
        for sign in (-1, 1)]
    initial = push_precontact_candidates(obj, [0, .16], pads)
    world = np.eye(4)
    world[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    world[:2, 3] = [.2, -.1]
    changed_goal = (world@np.array([0, .16, .46, 1]))[:2]
    changed = push_precontact_candidates(world@obj, changed_goal, pads)
    for a, b in zip(initial, changed):
        np.testing.assert_allclose(b['goal_world'], world@a['goal_world'], atol=1e-14)
        assert not b['launch_allowed']
        assert a['goal_world'][2, 3]+a['bottom_offset_m'] == pytest.approx(.524)
    with pytest.raises(ValueError, match='TASK_DISTANCE'):
        push_precontact_candidates(obj, [0, 0], pads)
    with pytest.raises(ValueError, match='TWO_MEASURED'):
        push_precontact_candidates(obj, [0, .16], pads[:1])
