"""Cold input bytes for the existing Core evaluator; no runtime admission.

The management owner supplies an exact pinned Git inventory. This module
validates its batch object response without importing GR00T or invoking Git.
Staging/registration, leases, resource guards and launches remain with the
existing once-only owner. No original checkout or environment is changed.
"""
import hashlib
import re
from pathlib import PurePosixPath

OFFICIAL_COMMIT = '51d4c89f72fda44cbf77285c6a8114b52676b8a1'
MAX_FILES = 2048
MAX_BYTES = 16 * 1024 ** 2


def need(ok, reason):
    if not ok:
        raise PermissionError('COLD_RUNTIME_INPUT_' + reason)


def package_inventory(inventory, *, official_commit):
    """Bound the exact original Python package and its two root metadata files."""
    need(official_commit == OFFICIAL_COMMIT, 'ORIGINAL_OFFICIAL_COMMIT')
    need(type(inventory) is dict and 0 < len(inventory) <= MAX_FILES,
         'FINITE_EXPLICIT_PACKAGE_INVENTORY')
    total = 0
    for name, item in inventory.items():
        need(type(name) is str, 'EXPLICIT_MEMBER_NAME')
        path = PurePosixPath(name)
        need(not path.is_absolute() and str(path) == name and '..' not in path.parts
             and (name.startswith('gr00t/') or name in ('LICENSE', 'pyproject.toml')),
             'ORIGINAL_PACKAGE_ONLY_NO_PRIVATE_CHECKOUT_CONTENT')
        need(type(item) is dict and set(item) == {'git_blob_sha', 'bytes', 'mode'}
             and type(item['git_blob_sha']) is str
             and re.fullmatch('[0-9a-f]{40}', item['git_blob_sha']) is not None
             and type(item['bytes']) is int and item['bytes'] >= 0
             and item['mode'] in ('100644', '100755'), 'REGULAR_PINNED_GIT_BLOB')
        total += item['bytes']
    need('gr00t/__init__.py' in inventory and total <= MAX_BYTES,
         'COMPLETE_PACKAGE_ROOT_AND_FINITE_TOTAL_BYTES')
    return total


def decode_git_batch(inventory, batch_bytes, *, official_commit):
    """Verify every byte returned by one `git cat-file --batch` operation.

    The order is the inventory's insertion order. Headers, declared sizes,
    delimiters, Git blob hashes and the complete response length all matter.
    A working-tree file is never substituted for its pinned upstream object.
    """
    total = package_inventory(inventory, official_commit=official_commit)
    need(type(batch_bytes) is bytes and len(batch_bytes) <= total + MAX_FILES * 96,
         'BOUNDED_EXACT_BATCH_RESPONSE')
    offset, result = 0, {}
    for name, pin in inventory.items():
        end = batch_bytes.find(b'\n', offset, offset + 96)
        need(end >= 0, 'COMPLETE_BLOB_HEADER')
        expected = (pin['git_blob_sha'] + ' blob ' + str(pin['bytes'])).encode()
        need(batch_bytes[offset:end] == expected, 'EXACT_BLOB_HEADER')
        start = end + 1
        body = batch_bytes[start:start + pin['bytes']]
        offset = start + pin['bytes']
        need(len(body) == pin['bytes'] and batch_bytes[offset:offset + 1] == b'\n',
             'EXACT_BLOB_BYTES_AND_DELIMITER')
        offset += 1
        digest = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
        need(digest == pin['git_blob_sha'], 'ACTUAL_PINNED_GIT_OBJECT_BYTES')
        result[name] = body
    need(offset == len(batch_bytes), 'NO_TRAILING_OR_UNREQUESTED_CONTENT')
    return result


def byte_witness(members, inventory, *, official_commit):
    """An unwritten content index, explicitly weaker than runtime qualification."""
    total = package_inventory(inventory, official_commit=official_commit)
    need(set(members) == set(inventory) and all(type(b) is bytes for b in members.values()),
         'EVERY_ORIGINAL_MEMBER_PRESENT')
    pins = {}
    for name, body in members.items():
        original = inventory[name]
        blob = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
        need(len(body) == original['bytes'] and blob == original['git_blob_sha'],
             'UNCHANGED_ORIGINAL_MEMBER')
        pins[name] = dict(bytes=len(body), git_blob_sha=blob,
                          sha256=hashlib.sha256(body).hexdigest())
    return dict(schema='ORIGINAL_GR00T_NODE_LOCAL_INPUT_BYTES_V1',
                official_commit=official_commit, files=pins, total_bytes=total,
                environment_installed=False, package_imported=False,
                model_loaded=False, runtime_admitted=False, launch_allowed=False)
