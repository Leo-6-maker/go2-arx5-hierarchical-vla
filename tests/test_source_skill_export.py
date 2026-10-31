"""Command provenance and input isolation at the new-skill export boundary."""
import copy
import hashlib
import json

import numpy as np
import pytest

from scripts.pi0294.multitask.source_export import packet, source_identity
from scripts.pi0292.train_adapter import NativeDataset


def records():
    rows = []
    pose = np.eye(4).tolist()
    card = {'instruction': 'Put the book in the left compartment.'}
    for i in range(641):
        state = {'tcp': pose, 'gripper_fraction': .5}
        p = dict(instruction=card['instruction'], actual_tcp=pose, actual_gripper=.5)
        p.update({n: dict(file=f'{i:04}_{n}.png', sha256='0' * 64, bytes=1) for n in ('body_rgb', 'wrist_rgb')})
        rows.append(dict(kind='observation', value=dict(index=i, native_step=2500 + 25 * i,
            time_s=5 + i / 20, policy_obs=p, actual_state=state, privileged_teacher_stage='hidden',
            calibration={n: dict(timestamp_s=5 + i / 20) for n in ('body_rgb', 'wrist_rgb')})))
        if i == 640:
            break
        if i % 8 == 0:
            a = np.tile([0., 0., 0., 1., 0., 0., 0., 1., 0., .5], (40, 1))
            # Deliberately different unexecuted suffixes must never be labels.
            a[8:, 0] = 99
            identity = dict(index=i, source_time_s=5 + i / 20, rows10=a.tolist())
            ref_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            rows.append(dict(kind='planned_H40', value=dict(ref_id=ref_id, observation_index=i,
                source_time_s=5 + i / 20, rows10=a.tolist(), committed_prefix_count=0,
                role='UNEXECUTED_REFERENCE_EVIDENCE_NOT_TRAINING_LABEL')))
        v = dict(ref_id=ref_id, observation_index=i, prefix_index=i % 8,
            source_time_s=5 + i / 20, action_EEF10=a[i % 8].tolist(),
            start_native_step=2500 + i * 25, completed=False, training_admitted=False)
        rows.append(dict(kind='submitted_interval', value=dict(v, expected_end_native_step=2525 + i * 25)))
        rows.append(dict(kind='completed_interval', value=dict(v, completed=True, end_native_step=2525 + i * 25)))
    return rows, card


def test_only_executed_prefixes_and_actual_state_are_exported():
    rows, card = records()
    a = packet(rows, card)
    assert a['committed_absolute10'].shape == (640, 10)
    assert not a['committed_absolute10'][:, 0].any()
    assert a['valid_window'].sum() == 601
    assert a['actual_state10'].shape == (641, 10)
    assert 'privileged_teacher_stage' not in a
    for r in rows:
        if r['kind'] == 'observation':
            r['value']['actual_state']['tcp'] = copy.deepcopy(r['value']['actual_state']['tcp'])
            r['value']['actual_state']['tcp'][0][3] = .123
            r['value']['policy_obs']['actual_tcp'] = copy.deepcopy(r['value']['actual_state']['tcp'])
    b = packet(rows, card)
    np.testing.assert_array_equal(a['committed_absolute10'], b['committed_absolute10'])
    assert np.all(b['actual_state10'][:, 0] == .123)


@pytest.mark.parametrize('defect', ['mismatch', 'incomplete', 'policy_privilege', 'bad_clock', 'missing_observation'])
def test_provenance_failures_are_rejected(defect):
    rows, card = records()
    if defect == 'mismatch':
        next(r for r in rows if r['kind'] == 'completed_interval')['value']['action_EEF10'][0] = 9
    elif defect == 'incomplete':
        rows.pop(-2)
    elif defect == 'policy_privilege':
        rows[0]['value']['policy_obs']['object_pose'] = np.eye(4).tolist()
    elif defect == 'bad_clock':
        rows[0]['value']['time_s'] += .05
    else:
        rows.pop()
    with pytest.raises(ValueError):
        packet(rows, card)


def test_candidate_cannot_enter_optimizer(tmp_path):
    (tmp_path / 'DATASET_MANIFEST.json').write_text(json.dumps({'training_admitted': False}))
    with pytest.raises(PermissionError, match='SEPARATE_TRAINING_ADMISSION'):
        NativeDataset(tmp_path, 'unused', 'unused', for_optimization=True)


def test_book_and_custom_cabinet_require_their_own_native_goal():
    book = {'task': 'BOOK', 'book_seed_index': 5}
    door = dict(task='DOOR', source_port='ROBOCASA_SINGLE_CABINET_ARX5_SOURCE_PORT_V1',
        port_revision='ROBOCASA_ARX5_PORT08_EDGE_HINGE_BRAKED_TURN',
        cabinet={'mechanical_variant': 'OUTER_PANEL_EDGE_AXIS_EXACT_QUARTER_TURN_V1'},
        original_benchmark_robot_or_seed=False)
    cabinet_audit = dict(schema='NIGHT_SOURCE_CABINET_INDEPENDENT_RAW_AUDIT_V1', native_ROBOCASA_goal_final=True)
    assert source_identity(book, {'native_LIBERO_goal_final': True})['episode_id'] == 'BOOK_SOURCE_SEED005'
    cabinet = source_identity(door, cabinet_audit)
    assert cabinet['task'] == 'DOOR' and 'CUSTOM' in cabinet['family']
    with pytest.raises(PermissionError):
        source_identity(book, cabinet_audit)
    with pytest.raises(PermissionError):
        source_identity(door, {'native_LIBERO_goal_final': True})
    for key, value in [('original_benchmark_robot_or_seed', True), ('port_revision', 'unqualified-port')]:
        with pytest.raises(PermissionError):
            source_identity(dict(door, **{key: value}), cabinet_audit)
