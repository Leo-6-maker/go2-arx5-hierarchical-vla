import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

from . import task_path_entry as entry


def approval():
    return (Path(__file__).resolve().parents[2]/
        'reports/PI0294/design/parallel-science-20261022/GO2_ARX5_S1_HUMAN_APPROVAL_20261029.json').read_bytes()


def test_direct_approval_has_one_finite_window_and_no_implicit_extension():
    raw = approval()
    value = json.loads(raw)
    begin = datetime.fromisoformat(value['adoption_recorded_at_UTC']).timestamp()
    end = datetime.fromisoformat(value['window_end_UTC']).timestamp()
    assert entry.authority(raw, now_epoch=begin)['approved_scope_only'] is True
    assert end-begin == pytest.approx(24*3600)
    for at in (begin-.001, end, end+3600):
        with pytest.raises(PermissionError, match='FINITE_WINDOW'):
            entry.authority(raw, now_epoch=at)
    changed = dict(value, window_end_UTC='2026-11-07T23:59:00+08:00')
    with pytest.raises(PermissionError, match='EXACT_HUMAN_APPROVAL'):
        entry.authority(json.dumps(changed).encode(), now_epoch=begin)


def test_same_registry_snapshot_requires_exactly_one_correct_run_role():
    root = '/node/runs/PI0294/20261029T113139Z-standing-task-path-s1'
    row = dict(role='S1_SCOPE__example', path=root, run_uuid=Path(root).name,
        executor_node=entry.HOST)
    raw = json.dumps(dict(entries=[dict(role='OLD_CLOSED', path='/old'), row])).encode()
    assert entry.registered_snapshot(raw, expected_sha256=entry.sha(raw),
        run_root=root, role=row['role']) == row
    for rows in ([row, row], [dict(row, path='/another')], [dict(row, executor_node='login')]):
        wrong = json.dumps(dict(entries=rows)).encode()
        with pytest.raises(PermissionError, match='UNIQUE_ORIGINAL_REGISTRY'):
            entry.registered_snapshot(wrong, expected_sha256=entry.sha(wrong),
                run_root=root, role=row['role'])


def test_original_doctor_layout_uses_run_capsule_and_never_old_source(monkeypatch):
    # Runtime pins are exercised against the actual deployed files separately.
    # These small fixtures isolate the relocation contract without vendor data.
    files = {'enter.sh': b'original entry', 'doctor.py': b'original doctor',
        'ENVIRONMENTS.json': json.dumps(dict(profiles=[dict(name=entry.PROFILE,
            hostnames=[entry.HOST], python='/node/env/bin/python', source='/nfs/old/source')])).encode()}
    monkeypatch.setattr(entry, 'ENTRY_FILES', {k:hashlib.sha256(v).hexdigest() for k, v in files.items()})
    raw = approval()
    at = datetime.fromisoformat(json.loads(raw)['adoption_recorded_at_UTC']).timestamp()
    root = '/node/runs/PI0294/20261029T113139Z-standing-task-path-s1'
    plan = entry.prepare_standard_entry(raw, files, run_root=root, source_sha='a'*40, now_epoch=at)
    assert plan['entry_cwd'] == root+'/entry/code/vla'
    assert plan['entry_path'] == root+'/entry/infrastructure/enter.sh'
    assert not plan['launch_allowed'] and not plan['resources_reserved']
    assert files['doctor.py'] == b'original doctor'
    with pytest.raises(PermissionError, match='ENTRY_MEMBER_CHANGED'):
        entry.prepare_standard_entry(raw, dict(files, **{'doctor.py': b'changed'}),
            run_root=root, source_sha='a'*40, now_epoch=at)
    with pytest.raises(PermissionError, match='RUN_LAYOUT'):
        entry.prepare_standard_entry(raw, files, run_root='/node/runs/PI0294/old-task',
            source_sha='a'*40, now_epoch=at)


def test_mount_parser_decodes_escaped_names_and_uses_longest_real_mount():
    mounts = entry._mounts(b'1 0 0:1 / / rw - ext4 /dev/root rw\n'
        b'2 1 0:2 / /node/remote\\040data rw - nfs4 server:/export rw\n')
    assert entry._filesystem(Path('/node/remote data/code'), mounts) == 'nfs4'
    assert entry._filesystem(Path('/node/remote database/code'), mounts) == 'ext4'


def test_resolved_network_symlink_is_not_mistaken_for_local_cache(tmp_path, monkeypatch):
    remote = tmp_path/'network'
    remote.mkdir()
    link = tmp_path/'local-cache'
    link.symlink_to(remote, target_is_directory=True)
    mounts = [(Path('/'), 'ext4'), (remote, 'nfs4')]
    monkeypatch.setattr(entry, '_mounts', lambda _: mounts)
    paths = {role: [str(tmp_path)] for role in entry.PATH_ROLES}
    paths['cache'] = [str(link)]
    with pytest.raises(PermissionError, match='NONLOCAL_OR_VOLATILE_RUNTIME_PATH_cache'):
        entry.node_local_paths(paths)
    assert link.is_symlink() and remote.is_dir()


def test_partial_path_inventory_and_volatile_control_cannot_pass(tmp_path, monkeypatch):
    with pytest.raises(PermissionError, match='COMPLETE_RUNTIME'):
        entry.node_local_paths({'cache': [str(tmp_path)]})
    paths = {role: [str(tmp_path)] for role in entry.PATH_ROLES}
    monkeypatch.setattr(entry, '_mounts', lambda _: [(Path('/'), 'tmpfs')])
    with pytest.raises(PermissionError, match='VOLATILE_RUNTIME_PATH'):
        entry.node_local_paths(paths)
