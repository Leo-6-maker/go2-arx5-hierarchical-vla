"""Real temporary Git objects/worktrees; no model, site or user repo writes."""
import base64
from copy import deepcopy
import os
from pathlib import Path
import subprocess

import pytest

from scripts.pi0294.pv1 import local_runtime_inputs as package
from scripts.pi0294.pv1 import pinned_sparse_source as sparse


@pytest.fixture
def original(tmp_path, monkeypatch):
    root = tmp_path / 'original'
    root.mkdir()
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], env=env)
    git('init', '--quiet', '--template=')
    included = {'gr00t/__init__.py': b'', 'gr00t/example.py': b'value = 1\n',
                'LICENSE': b'fixture', 'pyproject.toml': b'fixture metadata\n'}
    for name, body in {**included, 'docs/README.md': b'omitted',
                       '.gitattributes': b'*.py text\n'}.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
        if name == 'gr00t/example.py':
            p.chmod(0o755)
    git('add', '.')
    git('-c', 'user.name=CPU fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '--quiet', '-m', 'explicit CPU fixture')
    commit = git('rev-parse', 'HEAD').decode().strip()
    trees = {git('rev-parse', 'HEAD^{tree}').decode().strip()}
    for row in git('ls-tree', '-r', '-d', '-z', commit).split(b'\0'):
        if row:
            trees.add(row.split(b'\t')[0].split()[2].decode())
    objects = {}
    for pin in [commit, *trees]:
        kind = 'commit' if pin == commit else 'tree'
        objects[pin] = dict(type=kind, data_base64=base64.b64encode(
            git('cat-file', kind, pin)).decode())
    inventory = {name: dict(git_blob_sha=sparse.object_id('blob', body),
                            bytes=len(body), mode='100755' if name == 'gr00t/example.py'
                            else '100644') for name, body in included.items()}
    # Only this isolated fixture's expected commit replaces the production pin.
    monkeypatch.setattr(package, 'OFFICIAL_COMMIT', commit)
    return root, commit, objects, inventory, included, env


def test_real_sparse_git_preserves_original_checks_and_detects_runtime_edit(original, tmp_path):
    _, commit, objects, inventory, included, env = original
    destination = tmp_path / 'local'
    destination.mkdir()
    for name, body in included.items():
        p = destination / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
        p.chmod(0o400)
    receipt = sparse.materialize(destination, objects, inventory, expected_commit=commit)
    assert receipt['included_files'] == 4 and receipt['omitted_files'] == 2
    assert receipt['included_flags_all_H'] and receipt['native_git_diff_clean']
    assert (destination / 'gr00t/example.py').stat().st_mode & 0o111 == 0o100
    assert receipt['network_fetches'] == 0 and receipt['runtime_admitted'] is False
    assert not (destination / 'docs').exists()
    changed = destination / 'gr00t/example.py'
    changed.chmod(0o600)
    changed.write_bytes(b'value = 2\n')
    assert subprocess.run(['git', '-C', str(destination), 'diff', '--quiet', 'HEAD', '--'],
                          env=env).returncode == 1
    with pytest.raises(PermissionError):
        sparse.materialize(destination, objects, inventory, expected_commit=commit)


@pytest.mark.parametrize('change', ['object_hash', 'missing_tree', 'path_substitution'])
def test_unproven_commit_tree_or_path_is_rejected(original, change):
    _, commit, objects, inventory, _, _ = original
    objects, inventory = deepcopy(objects), deepcopy(inventory)
    if change == 'object_hash':
        objects[commit]['data_base64'] = base64.b64encode(b'tampered commit').decode()
    elif change == 'missing_tree':
        pin = next(k for k, v in objects.items() if v['type'] == 'tree')
        del objects[pin]
    else:
        inventory['gr00t/uncommitted.py'] = inventory.pop('gr00t/example.py')
    with pytest.raises(PermissionError):
        sparse.plan(objects, inventory, expected_commit=commit)


def test_binary_tree_truncation_duplicate_or_escape_is_rejected():
    pin = bytes.fromhex('1' * 40)
    for body in [b'100644 file\0' + pin[:-1], b'100644 ../escape\0' + pin,
                 (b'100644 same\0' + pin) * 2]:
        with pytest.raises(PermissionError):
            sparse.tree_entries(body)
