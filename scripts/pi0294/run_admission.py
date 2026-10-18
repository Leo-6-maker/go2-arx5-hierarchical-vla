"""Fixed replacement authority, registered output and sealed-source admission."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import time

RUN = '20261013T142457Z-localpc-pi0294'
AUTHORITY = '09e68e41752383fcb1087a23c52fe487c9a14a89'
T0 = 1791901497
CUTOFF = 1791939600
DEADLINE = 1791943200
GO_COMMENT = 6148228083


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def contained(path, parent):
    path, parent = Path(path), Path(parent)
    if not path.resolve().is_relative_to(parent.resolve()):
        raise PermissionError('REGISTERED_PATH_ESCAPE')
    return path


def validate(root, *, profile, manifest_path=None, now=None, closeout=False,
             hostname=None, uid=None, verify_source=True):
    if (Path(root)/'PV1_STAGE_BINDING.json').is_file():
        if (any(v is not None for v in (hostname,uid,manifest_path,now)) or not verify_source
                or (Path(root)/'MP1_STAGE_BINDING.json').exists()):
            raise PermissionError('PV1_NATIVE_IDENTITY_NO_OLD_SCOPE_OR_INJECTED_HOST')
        from .pv1.runtime import validate_runtime
        return validate_runtime(root,profile,closeout=closeout)
    if (Path(root)/'MP1_STAGE_BINDING.json').is_file():
        from .mp1.science import validate_runtime
        if any(v is not None for v in (hostname,uid,manifest_path)) or not verify_source:
            raise PermissionError('MP1_NATIVE_SCIENTIFIC_IDENTITY_ONLY')
        return validate_runtime(root,profile,now=now,closeout=closeout)
    if Path(root).name=='20261016T150228Z-localpc-e1':
        from .e1.scope import validate as validate_e1
        if any(v is not None for v in (hostname,uid,manifest_path)):
            raise PermissionError('E1_NATIVE_IDENTITY_ONLY_NO_INJECTED_HOST')
        return validate_e1(root,profile,now=now,verify_source=verify_source)
    from .e0_scope import RUN as E0_RUN, validate_entry as validate_e0
    if Path(root).name == E0_RUN:
        return validate_e0(root, profile=profile, manifest_path=manifest_path, now=now,
            closeout=closeout, hostname=hostname, uid=uid, verify_source=verify_source)
    from .r4 import RUN as R4_RUN, validate_entry as validate_r4
    if Path(root).name == R4_RUN:
        return validate_r4(root,profile=profile,manifest_path=manifest_path,now=now,
            closeout=closeout,hostname=hostname,uid=uid,verify_source=verify_source)
    from .continuation import RUN as CONTINUATION_RUN, validate_entry
    if Path(root).name == CONTINUATION_RUN:
        return validate_entry(root, profile=profile, manifest_path=manifest_path,
            now=now, closeout=closeout, hostname=hostname, uid=uid, verify_source=verify_source)
    root = Path(root)
    manifest_path = contained(manifest_path or root / 'RUN_MANIFEST.json', root)
    manifest = read(manifest_path)
    now = time.time() if now is None else now
    hostname = socket.gethostname() if hostname is None else hostname
    uid = os.getuid() if uid is None else uid
    if (manifest.get('schema') != 'RUN_MANIFEST_V1' or manifest.get('run_uuid') != RUN
            or manifest.get('pi') != 'PI0294' or manifest.get('owner') != 'EXISTING_LOCAL_PC'
            or root.name != RUN):
        raise PermissionError('EXACT_PI0294_OWNER_AND_RUN_REQUIRED')
    authority, limits = manifest['authority'], manifest['limits']
    if (authority.get('protocol_ref') != AUTHORITY or authority.get('go_comment') != GO_COMMENT
            or limits.get('T0_epoch') != T0 or limits.get('new_work_cutoff_epoch') != CUTOFF
            or limits.get('deadline_epoch') != DEADLINE or limits.get('reset_on_resume') is not False):
        raise PermissionError('FIXED_USER_AUTHORITY_AND_ORIGINAL_CLOCK_REQUIRED')
    if now < T0 or now >= (DEADLINE if closeout else CUTOFF):
        raise PermissionError('PI0294_TIME_WINDOW_CLOSED')
    execution = manifest['execution']
    if (execution['profile'] != profile or execution['uid'] != uid
            or hostname not in execution['hostnames']):
        raise PermissionError('HOST_PROFILE_AND_OWNER_REQUIRED')
    binding = read(contained(root / 'OUTPUT_BINDING.private.json', root))
    if sha(root / 'OUTPUT_BINDING.private.json') != execution['output_binding_sha256']:
        raise PermissionError('OUTPUT_BINDING_CHANGED')
    if (Path(binding['run_root']).resolve() != root.resolve()
            or binding['role'] != execution['private_path_role']):
        raise PermissionError('EXACT_REGISTERED_NODE_LOCAL_RUN_REQUIRED')
    parent = Path(binding['registered_parent'])
    contained(root, parent)
    if root.is_symlink() or parent.is_symlink() or root.stat().st_dev != parent.stat().st_dev:
        raise PermissionError('NO_CROSS_DEVICE_OR_SYMLINK_RUN_ROOT')
    registry = read(binding['registry'])
    if not any(row.get('role') == binding['role'] and
               Path(row['path']).resolve() == root.resolve() and
               row.get('executor_node') in (hostname, 'hopper') for row in registry['entries']):
        raise PermissionError('LIVE_HOST_REGISTRY_ROLE_REQUIRED')
    lease = read(binding['lease'])
    if (lease.get('run_uuid') != RUN or lease.get('owner') != 'EXISTING_LOCAL_PC'
            or lease.get('deadline_epoch') != DEADLINE or lease.get('status') != 'ACTIVE'
            or lease.get('protocol_ref') != AUTHORITY):
        raise PermissionError('LIVE_EXACT_LEASE_REQUIRED')
    if verify_source:
        seal_path = contained(root / manifest['code'].get('source_seal_relative', 'SOURCE_SEAL.json'), root)
        if sha(seal_path) != manifest['code']['source_seal_sha256']:
            raise PermissionError('SOURCE_SEAL_CHANGED')
        seal = read(seal_path)
        if seal['source_sha'] != manifest['code']['vla_commit']:
            raise PermissionError('FIXED_CODE_SHA_CHANGED')
        code = contained(root / manifest['code'].get('capsule_relative', 'code'), root)
        if code.is_symlink():
            raise PermissionError('NO_MUTABLE_SOURCE_SYMLINK')
        for name, digest in seal['files'].items():
            member = contained(code / name, code)
            if member.is_symlink() or sha(member) != digest:
                raise PermissionError('SEALED_SOURCE_MEMBER_CHANGED:' + name)
    return manifest
