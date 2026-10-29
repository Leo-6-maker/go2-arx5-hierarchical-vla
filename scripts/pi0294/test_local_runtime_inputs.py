"""Reject changed/truncated/unrequested pinned package bytes before staging."""
import hashlib

import pytest

from scripts.pi0294.pv1.local_runtime_inputs import (
    OFFICIAL_COMMIT, byte_witness, decode_git_batch, package_inventory)


def fixture():
    members = {'gr00t/__init__.py': b'', 'gr00t/example.py': b'original bytes\n'}
    inventory, batch = {}, b''
    for name, body in members.items():
        blob = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
        inventory[name] = dict(git_blob_sha=blob, bytes=len(body), mode='100644')
        batch += (blob + ' blob ' + str(len(body)) + '\n').encode() + body + b'\n'
    return members, inventory, batch


def test_actual_bytes_and_zero_length_member_preserved_without_admission():
    members, inventory, batch = fixture()
    actual = decode_git_batch(inventory, batch, official_commit=OFFICIAL_COMMIT)
    assert actual == members
    witness = byte_witness(actual, inventory, official_commit=OFFICIAL_COMMIT)
    assert witness['total_bytes'] == sum(map(len, members.values()))
    assert all(witness[k] is False for k in ('environment_installed', 'package_imported',
        'model_loaded', 'runtime_admitted', 'launch_allowed'))


@pytest.mark.parametrize('change', ['content', 'header', 'truncate', 'trailing'])
def test_wrong_batch_is_rejected(change):
    _, inventory, batch = fixture()
    changed = dict(content=batch.replace(b'original bytes', b'changed! bytes'),
                   header=batch.replace(b' blob ', b' tree ', 1),
                   truncate=batch[:-1], trailing=batch+b'unrequested')[change]
    with pytest.raises(PermissionError):
        decode_git_batch(inventory, changed, official_commit=OFFICIAL_COMMIT)


@pytest.mark.parametrize('name', ['/gr00t/escape.py', 'gr00t/../escape.py',
                                  '.git/config', 'notebook.ipynb'])
def test_unsafe_or_private_checkout_member_is_rejected(name):
    _, inventory, _ = fixture()
    inventory[name] = inventory['gr00t/example.py'].copy()
    with pytest.raises(PermissionError):
        package_inventory(inventory, official_commit=OFFICIAL_COMMIT)


def test_new_revision_symlink_or_missing_member_is_rejected():
    members, inventory, batch = fixture()
    with pytest.raises(PermissionError):
        decode_git_batch(inventory, batch, official_commit='0'*40)
    inventory['gr00t/example.py']['mode'] = '120000'
    with pytest.raises(PermissionError):
        package_inventory(inventory, official_commit=OFFICIAL_COMMIT)
    _, inventory, _ = fixture()
    members.pop('gr00t/example.py')
    with pytest.raises(PermissionError):
        byte_witness(members, inventory, official_commit=OFFICIAL_COMMIT)
