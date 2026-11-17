import hashlib
import json

import pytest

from scripts.sla.robodojo_source_audit_v1 import RAW_CAP, selection


def population():
    return dict(repository='RoboDojo-Benchmark/RoboDojo', revision='a'*40, tasks=[
        dict(task=task, episodes=[dict(path='data/RoboDojo_real/%s/arx_x5/data/episode_%07d.hdf5' % (task, i),
            oid='git-'+str(i), size=100, lfs=dict(oid=hashlib.sha256((task+str(i)).encode()).hexdigest(), size=100))
            for i in range(100)]) for task in ('cover_blocks', 'insert_tubes', 'make_bread',
                'make_food', 'pack_and_pour_fruit', 'store_in_safe')])


def test_first12_are_in_the_same_frozen60_and_no_float_revision_or_data_filter():
    inventory = population()
    result = selection(inventory)
    assert result['candidate_count'] == 60 and result['first_count'] == 12
    assert len(set(result['first_paths'])) == 12
    assert set(result['first_paths']) <= {r['path'] for r in result['candidates']}
    assert result == selection(dict(inventory, tasks=list(reversed(inventory['tasks']))))
    for task in inventory['tasks']:
        chosen = [r for r in result['candidates'] if r['task'] == task['task']]
        assert [sum(r['proposed_split'] == split for r in chosen)
            for split in ('TRAIN', 'SOURCE_DEV', 'SOURCE_TEST')] == [6, 2, 2]
        assert sum(r['path'] in result['first_paths'] for r in chosen) == 2
    assert not result['source_semantics_qualified']
    assert all('PROVISIONAL' in r['split_status'] for r in result['candidates'])


def test_unpinned_file_wrong_robot_and_predeclared_raw_overflow_are_rejected():
    data = population()
    # Change every candidate so fixed selection cannot silently avoid the bad field.
    for r in data['tasks'][0]['episodes']:
        r.pop('lfs')
    with pytest.raises(ValueError, match='LFS'):
        selection(data)
    data = population()
    for r in data['tasks'][0]['episodes']:
        r['path'] = r['path'].replace('/arx_x5/', '/piper/')
    with pytest.raises(ValueError, match='REAL_ARX'):
        selection(data)
    data = population()
    for task in data['tasks']:
        for r in task['episodes']:
            r['size'] = r['lfs']['size'] = RAW_CAP // 59
    with pytest.raises(ValueError, match='BUDGET'):
        selection(data)


def test_profile_does_not_turn_different_actions_into_ack_or_native_clock(tmp_path):
    h5py = pytest.importorskip('h5py')
    np = pytest.importorskip('numpy')
    from scripts.sla.robodojo_source_audit_v1 import profile
    path = tmp_path/'episode.hdf5'
    with h5py.File(path, 'w') as f:
        for group in ('state', 'action'):
            for arm in ('left', 'right'):
                pose = np.zeros((100, 7)); pose[:, 3] = 1
                if group == 'action':
                    pose[:, 0] = .01
                f[group+'/'+arm+'_ee_poses'] = pose
                f[group+'/'+arm+'_ee_joint_states'] = np.linspace(0, 1, 100)
        f['instruction'] = 'one recorded instruction'
        f.attrs['labels'] = np.array([b'a', b'b'])
    result = profile(path)
    assert not result['arms']['left']['action_equal_state']
    assert result['frequency'] is None
    assert result['command_provenance'].startswith('UNKNOWN')
    assert result['command_consumption_evidence'].startswith('UNKNOWN')
    assert not result['previous_actual_emitted_context_created']
    assert not result['Go2_P10_evidence_created']
    assert not result['canonical_qualified']
    json.dumps(result, allow_nan=False)
