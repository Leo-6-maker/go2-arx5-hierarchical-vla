"""Stage immutable N1 consumers before an explicit, cold manifest advancement."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re

from scripts.pi0292.durable import OwnerLock, read
from scripts.pi0294.native_dataset import sha
from . import n1_boundary as boundary
from . import n1_limits as n1


def put(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise PermissionError('NO_COLD_ARTIFACT_SYMLINK')
    if path.exists():
        if path.stat().st_mode & 0o222 or path.read_bytes() != raw:
            raise PermissionError('PRESERVE_UNKNOWN_OR_DIFFERENT_COLD_ARTIFACT')
        return
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)


def jput(path, value):
    put(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def assert_no_consumers(root):
    root = Path(root)
    for folder in ['worker_processes', 'aux_worker_processes']:
        for path in (root / folder).glob('*.json'):
            value = read(path)
            stat = Path('/proc', str(value['pid']), 'stat')
            if stat.exists():
                fields = stat.read_text().split()
                if fields[2] != 'Z' and fields[21] == str(value['proc_start']):
                    raise PermissionError('OLD_NATIVE_CONSUMER_STILL_LIVE')
    for name in ['SOURCE_PHYSICS_STOP.json', 'SOURCE_SAFETY_STOP_FROM_CORE.json']:
        if (root / name).exists():
            raise PermissionError('NO_COLD_SAFETY_STOP_BYPASS')
    core = root / 'RESOURCE_LEDGER.json'
    if core.exists():
        value = read(core)
        if value.get('reservations') or value.get('CPU_reserved_s', 0):
            raise PermissionError('OLD_CORE_RESERVATION_STILL_ACTIVE')
    hopper = root / 'HOPPER_RESOURCE_LEDGER.json'
    if hopper.exists():
        value = read(hopper)
        if value.get('GPU_reserved_s', 0) or value.get('CPU_reserved_s', 0) or value.get('pending_updates', 0):
            raise PermissionError('OLD_HOPPER_RESERVATION_STILL_ACTIVE')


def stage(root, number, source, contents, *, prior_ledger, native_terminals,
          outcomes, effects, processes, authority_files, fixture_plan,endpoint_probe_files=None):
    root = Path(root)
    names = boundary.paths(number)
    if not re.fullmatch('[0-9a-f]{40}', source) or (root / 'pbs' / f'alloc-{number:02}').exists():
        raise PermissionError('FUTURE_UNSUBMITTED_COLD_SOURCE_ONLY')
    assert_no_consumers(root)
    original = (root / 'RUN_MANIFEST.json').read_bytes()
    old = json.loads(original)
    if len(prior_ledger['PBS_submissions']) != number - 1:
        raise PermissionError('CONTIGUOUS_NATIVE_PRIOR_HISTORY')
    code = root / 'capsules' / 'N1' / f'alloc-{number:02}' / source / 'code'
    members = {}
    for name, raw in sorted(contents.items()):
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.as_posix() != name:
            raise PermissionError('EXACT_REPOSITORY_CAPSULE_MEMBER')
        if not (code / path).resolve().is_relative_to(root.resolve()):
            raise PermissionError('CAPSULE_MEMBER_MUST_REMAIN_IN_OWNED_ROOT')
        put(code / path, raw)
        members[name] = hashlib.sha256(raw).hexdigest()
    learning = old['code']['unchanged_learning_files']
    for name, digest in dict(old['code']['unchanged_numerical_files'], **learning).items():
        if members.get(name) != digest:
            raise PermissionError('ORIGINAL_NUMERICAL_AND_LEARNING_BYTES_REQUIRED')
    seal = code.parent / 'SOURCE_SEAL.json'
    jput(seal, dict(source_sha=source, authority_release=old['authority']['protocol_ref'], files=members))
    if set(authority_files) != set(n1.FILES):
        raise PermissionError('COMPLETE_N1_AUTHORITY_FILES_REQUIRED')
    for name, raw in authority_files.items():
        put(root / name, raw)
    jput(root / boundary.TRANSPORT_PLAN, fixture_plan)
    put(root / names[1], original)
    for name, value in zip(names[2:], [prior_ledger, outcomes, native_terminals, effects, processes]):
        jput(root / name, value)
    stops = {p.name: sha(p) for p in root.glob('RELAY_ENGINEERING_STOP*.json')}
    value = dict(schema=boundary.SCHEMA, authority_release=n1.RELEASE,
        parent_authority_release=n1.PARENT_AUTHORITY, run_uuid=root.name,
        parent_run_uuid=root.parent.parent.name, original_T0_epoch=n1.ORIGINAL_T0,
        target_allocation_number=number, effective_source_PBS_cap=12, new_source_sha=source,
        prior_manifest_sha256=sha(root / names[1]), adoption_delta_sha256=sha(root / n1.DELTA),
        new_attempt_rights=False, old_STOP_removed=False, preserved_STOP_hashes=stops)
    extra=[]
    if endpoint_probe_files is not None:
        from . import endpoint_anchor_ni as ni
        if number!=10 or set(endpoint_probe_files)!=set(ni.FILES):
            raise PermissionError('ONLY_COMPLETE_NI16_COLD_ALLOCATION10_FILE_BUNDLE')
        for name,raw in endpoint_probe_files.items():put(root/name,raw)
        value['endpoint_probe_binding_relative']=ni.PLAN
        extra=ni.FILES
    jput(root / names[0], value)
    proposed = deepcopy(old)
    proposed['code'].update(vla_commit=source, capsule_relative=str(code.relative_to(root)),
        source_seal_relative=str(seal.relative_to(root)), source_seal_sha256=sha(seal),
        unchanged_learning_files=learning)
    proposed['execution'].update(cold_engineering_binding_relative=names[0],
        n1_authority_binding=dict(release=n1.RELEASE, adoption_delta_sha256=sha(root / n1.DELTA)))
    for name in names + n1.FILES + list(stops) + [boundary.TRANSPORT_PLAN]+extra:
        proposed['execution']['sealed_inputs'][name] = sha(root / name)
    boundary.validate(root, proposed_manifest=proposed)
    pinned = code.parent / 'WORKER_RUN_MANIFEST.json'
    jput(pinned, proposed)
    if (root / 'RUN_MANIFEST.json').read_bytes() != original:
        raise PermissionError('STAGING_MUST_NOT_ADVANCE_CURRENT_POINTER')
    return dict(status='STAGED_COLD_N1_CONSUMER_NOT_ACTIVATED', number=number, source_sha=source,
        old_manifest_sha256=hashlib.sha256(original).hexdigest(), proposed_manifest_sha256=sha(pinned),
        worker_manifest_relative=str(pinned.relative_to(root)), source_members=len(members),
        new_PBS=0, new_models=0, new_physics=0)


def activate(root, expected):
    root = Path(root)
    number = expected['number']
    boundary.paths(number)
    pinned = root / expected['worker_manifest_relative']
    if pinned.is_symlink() or not pinned.resolve().is_relative_to(root.resolve()):
        raise PermissionError('EXACT_OWNED_PINNED_MANIFEST')
    lock = OwnerLock(root / 'COLD_SOURCE_BOUNDARY_OWNER.lock')
    try:
        assert_no_consumers(root)
        if ((root / 'pbs' / f'alloc-{number:02}').exists()
                or sha(root / 'RUN_MANIFEST.json') != expected['old_manifest_sha256']
                or sha(pinned) != expected['proposed_manifest_sha256']):
            raise PermissionError('EXACT_UNCHANGED_COLD_ACTIVATION_INTENT')
        value = read(pinned)
        if value['code']['vla_commit'] != expected['source_sha']:
            raise PermissionError('EXACT_TESTED_NEW_SOURCE')
        boundary.validate(root, proposed_manifest=value)
        pending = root / f'RUN_MANIFEST.n1-{number:02}.{os.getpid()}.pending'
        put(pending, pinned.read_bytes())
        os.replace(pending, root / 'RUN_MANIFEST.json')
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        boundary.validate(root)
        if sha(root / 'RUN_MANIFEST.json') != expected['proposed_manifest_sha256']:
            raise PermissionError('ACTUAL_COLD_POINTER_READBACK_FAILED')
    finally:
        lock.close()
    return dict(status='ACTUAL_COLD_N1_MANIFEST_ADVANCED', number=number,
        source_sha=expected['source_sha'], manifest_sha256=sha(root / 'RUN_MANIFEST.json'),
        previous_manifest_sha256=expected['old_manifest_sha256'], new_PBS=0, new_models=0, new_physics=0)
