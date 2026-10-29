"""A genuine sparse Git view of already-staged, fixed upstream code bytes.

This preserves the original official_import Git checks and frozen numerical
files. It does not fetch, install an environment, register or launch a job.
The existing management Journal must own any materialization operation.
"""
import base64
import hashlib
import os
from pathlib import Path
import re
import stat
import subprocess
import zlib

from .local_runtime_inputs import OFFICIAL_COMMIT, package_inventory


def need(ok, why):
    if not ok:
        raise PermissionError('PINNED_SPARSE_SOURCE_' + why)


def object_id(kind, body):
    return hashlib.sha1(kind.encode() + b' ' + str(len(body)).encode() + b'\0' + body).hexdigest()


def tree_entries(body):
    """Parse original binary Git tree bytes, including opaque gitlink entries."""
    result, offset = {}, 0
    while offset < len(body):
        end = body.find(b'\0', offset)
        need(end >= 0 and end + 21 <= len(body), 'COMPLETE_TREE_ENTRY')
        header = body[offset:end].split(b' ', 1)
        need(len(header) == 2, 'EXPLICIT_MODE_AND_NAME')
        mode, name = header[0].decode('ascii'), header[1].decode('utf-8')
        need(mode in ('40000', '100644', '100755', '120000', '160000')
             and name not in ('', '.', '..', '.git') and '/' not in name
             and name not in result, 'SAFE_UNIQUE_GIT_TREE_NAME')
        result[name] = dict(mode=mode, git_blob_sha=body[end + 1:end + 21].hex())
        offset = end + 21
    return result


def verify_metadata(objects, *, expected_commit):
    """Recompute the commit and every tree SHA, then walk its complete tree."""
    need(type(objects) is dict and 0 < len(objects) <= 2048
         and re.fullmatch('[0-9a-f]{40}', expected_commit) is not None,
         'BOUNDED_EXACT_COMMIT_PROOF')
    decoded, total = {}, 0
    for pin, value in objects.items():
        need(re.fullmatch('[0-9a-f]{40}', pin) is not None
             and set(value) == {'type', 'data_base64'}
             and value['type'] in ('commit', 'tree'), 'ONLY_COMMIT_AND_TREE_OBJECTS')
        body = base64.b64decode(value['data_base64'], validate=True)
        total += len(body)
        need(total <= 1024 ** 2 and object_id(value['type'], body) == pin,
             'ACTUAL_ORIGINAL_OBJECT_HASH')
        decoded[pin] = value['type'], body
    need(expected_commit in decoded and decoded[expected_commit][0] == 'commit',
         'EXACT_ORIGINAL_COMMIT_OBJECT')
    first = decoded[expected_commit][1].split(b'\n', 1)[0]
    need(re.fullmatch(b'tree [0-9a-f]{40}', first) is not None, 'COMMIT_TREE_EDGE')
    root = first[5:].decode()
    files, visited = {}, {expected_commit}

    def walk(pin, prefix, depth):
        need(depth <= 32 and pin in decoded and decoded[pin][0] == 'tree',
             'COMPLETE_BOUNDED_TREE_CLOSURE')
        visited.add(pin)
        for name, value in tree_entries(decoded[pin][1]).items():
            path = prefix + name
            if value['mode'] == '40000':
                walk(value['git_blob_sha'], path + '/', depth + 1)
            else:
                need(len(files) < 8192, 'BOUNDED_TRACKED_FILE_COUNT')
                files[path] = value

    walk(root, '', 0)
    need(set(decoded) == visited, 'NO_UNRELATED_OR_UNREACHABLE_METADATA')
    return dict(root_tree=root, files=files, decoded=decoded)


def plan(objects, inventory, *, expected_commit=OFFICIAL_COMMIT):
    package_inventory(inventory, official_commit=expected_commit)
    proof = verify_metadata(objects, expected_commit=expected_commit)
    selected = {n: v for n, v in proof['files'].items()
                if n.startswith('gr00t/') or n in ('LICENSE', 'pyproject.toml')}
    need(set(selected) == set(inventory) and all(
        v['git_blob_sha'] == inventory[n]['git_blob_sha']
        and v['mode'] == inventory[n]['mode'] for n, v in selected.items()),
        'ALL_PACKAGE_PATHS_BOUND_TO_ORIGINAL_COMMIT_TREE')
    return dict(commit=expected_commit, selected=selected,
                omitted=sorted(set(proof['files']) - set(selected)), **proof)


def materialize(source, objects, inventory, *, expected_commit=OFFICIAL_COMMIT):
    """Add only fresh Git metadata to the existing verified package directory.

    The worktree is sparse: only the complete gr00t package and its two root
    metadata files are present. Omitted documentation, datasets and gitlinks
    stay omitted. Included files never get skip-worktree or assume-unchanged.
    """
    p = plan(objects, inventory, expected_commit=expected_commit)
    source = Path(source)
    need(source.is_absolute() and source.is_dir() and not source.is_symlink()
         and source.resolve() == source and not (source / '.git').exists()
         and not (source / '.git').is_symlink(),
         'ONE_FRESH_DECLARED_PACKAGE_VIEW')
    bodies = {}
    for name, pin in inventory.items():
        path = source / name
        need(not any((source / Path(*Path(name).parts[:i])).is_symlink()
                     for i in range(1, len(Path(name).parts) + 1)), 'NO_INPUT_SYMLINK')
        s = path.lstat()
        need(stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid()
             and not s.st_mode & 0o222 and s.st_size == pin['bytes'],
             'EXISTING_IMMUTABLE_OWN_PACKAGE_BYTES')
        body = path.read_bytes()
        need(object_id('blob', body) == pin['git_blob_sha'], 'UNCHANGED_INCLUDED_BLOB')
        bodies[pin['git_blob_sha']] = ('blob', body)

    def git(*args, input=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
        env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
                   GIT_CONFIG_SYSTEM=os.devnull)
        q = subprocess.run(['git', '-C', str(source), *args], input=input,
                           capture_output=True, timeout=8, env=env)
        need(q.returncode == 0, 'NATIVE_GIT_OPERATION_FAILED:' + args[0])
        return q.stdout

    git('init', '--quiet', '--template=', '--object-format=sha1')
    for name, pin in inventory.items():
        # The earlier byte-only stage made every member read-only. Restore
        # only the executable bit recorded in the original committed tree.
        (source / name).chmod(0o500 if pin['mode'] == '100755' else 0o400)
    all_objects = {**p['decoded'], **bodies}
    for pin, (kind, body) in all_objects.items():
        path = source / '.git/objects' / pin[:2] / pin[2:]
        path.parent.mkdir(mode=0o700, exist_ok=True)
        wire = zlib.compress(kind.encode() + b' ' + str(len(body)).encode() + b'\0' + body)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
        with os.fdopen(fd, 'wb') as f:
            f.write(wire)
            f.flush()
            os.fsync(f.fileno())
    git('update-ref', '--no-deref', 'HEAD', expected_commit)
    git('read-tree', expected_commit)
    if p['omitted']:
        git('update-index', '--skip-worktree', '-z', '--stdin',
            input=b''.join(n.encode() + b'\0' for n in p['omitted']))
    git('diff', '--quiet', 'HEAD', '--')
    flags = git('ls-files', '-v', '-z').split(b'\0')
    actual = {row[2:].decode(): row[:1].decode() for row in flags if row}
    need(set(actual) == set(p['files'])
         and all(actual[n] == 'H' for n in p['selected'])
         and all(actual[n] == 'S' for n in p['omitted']),
         'ONLY_OMITTED_FILES_SKIPPED_ALL_RUNTIME_FILES_CHECKED')
    need(git('rev-parse', 'HEAD').decode().strip() == expected_commit
         and not git('remote'), 'EXACT_DETACHED_COMMIT_WITH_NO_REMOTE')
    # All newly created metadata belongs to this fresh view. Make its result
    # durable without touching the original source checkout or its settings.
    paths = list((source / '.git').rglob('*'))
    need(len(paths) <= 8192 and not any(p.is_symlink() for p in paths),
         'BOUNDED_OWN_GIT_METADATA_NO_SYMLINK')
    for path in paths:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    for path in (source / '.git', source):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return dict(schema='ORIGINAL_GR00T_GENUINE_SPARSE_GIT_VIEW_V1',
                commit=expected_commit, root_tree=p['root_tree'],
                included_files=len(inventory), omitted_files=len(p['omitted']),
                original_objects=len(all_objects), included_flags_all_H=True,
                omitted_flags_all_S=True, native_git_diff_clean=True,
                network_fetches=0, environment_installs=0, runtime_admitted=False)
