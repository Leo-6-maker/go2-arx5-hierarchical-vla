"""Explicitly renewed task-data window, retaining the original R4 ledger.

The historical R4 runtime and lease are immutable. This child has its own source
capsule, manifest and lease, and spends the parent's cumulative ledger.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import socket
import time

RUN = '20261015T061000Z-task-data-first-continuation'
ADOPTION_SHA256 = '48dbb18904ee9b1886ca6d7f2d30022efbedb5af0cfcd14592df82c2d64be8b2'
T0, CUTOFF, DEADLINE = 1792044600, 1792084200, 1792087800


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def within(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if not path.is_relative_to(parent):
        raise PermissionError('TASK_CAPTURE_PATH_ESCAPE')
    return path


def validate_entry(root, *, profile='4080-simulation', now=None, closeout=False, read_only=False):
    root = Path(root).resolve()
    now = time.time() if now is None else now
    authority_path = root/'WINDOW_ADOPTION.json'
    if sha(authority_path) != ADOPTION_SHA256:
        raise PermissionError('EXACT_EXPLICIT_WINDOW_ADOPTION_REQUIRED')
    a, m = read(authority_path), read(root/'RUN_MANIFEST.json')
    if (root.name != RUN or m['run_uuid'] != RUN or m['owner'] != 'EXISTING_LOCAL_PC'
            or not a['authorizes_execution'] or a['counter_reset'] or a['old_lease_reopened']
            or (a['T0_epoch'], a['new_work_cutoff_epoch'], a['deadline_epoch']) != (T0,CUTOFF,DEADLINE)
            or not T0 <= now < (DEADLINE if closeout else CUTOFF)
            or m['scope'] != 'GR00T_TASK_CAPTURE_FROZEN_WBC_V1'
            or m['authority']['adoption_sha256'] != ADOPTION_SHA256
            or m['limits'] != a['cumulative_caps_unchanged']):
        raise PermissionError('TASK_CAPTURE_AUTHORITY_OR_WINDOW')
    execution = m['execution']
    if (execution['profile'] != profile or socket.gethostname() not in execution['hostnames']
            or os.getuid() != execution['uid']):
        raise PermissionError('TASK_CAPTURE_HOST_PROFILE_UID')
    if profile.startswith('4080-'):
        if (os.environ.get('GR00T_WBC_PYTHON') != execution['profile_python']
                or sha(execution['entry_path']) != execution['entry_sha256']):
            raise PermissionError('DEPLOYED_ENTRY_PROFILE_REQUIRED')
    bpath = root/'OUTPUT_BINDING.private.json'
    if sha(bpath) != execution['output_binding_sha256']:
        raise PermissionError('TASK_CAPTURE_OUTPUT_BINDING_CHANGED')
    b = read(bpath)
    if Path(b['run_root']).resolve() != root or b['run_uuid'] != RUN:
        raise PermissionError('TASK_CAPTURE_OUTPUT_IDENTITY')
    within(root, b['registered_parent'])
    registry, lease = read(b['registry']), read(b['lease'])
    if not any(x['role'] == b['role'] and Path(x['path']).resolve() == root for x in registry['entries']):
        raise PermissionError('TASK_CAPTURE_UNREGISTERED_OUTPUT')
    if (lease['status'] != 'ACTIVE' or lease['run_uuid'] != RUN
            or lease['deadline_epoch'] != DEADLINE or lease['owner'] != m['owner']
            or lease['adoption_sha256'] != ADOPTION_SHA256):
        raise PermissionError('TASK_CAPTURE_OWNER_LEASE')
    parent = Path(b['parent_root']).resolve()
    if parent.name != a['parent_run'] or sha(parent/'RUN_MANIFEST.json') != b['parent_manifest_sha256']:
        raise PermissionError('CUMULATIVE_PARENT_IDENTITY_CHANGED')
    code = within(root/m['code']['capsule_relative'], root)
    sealpath = within(root/m['code']['source_seal_relative'], root)
    if sha(sealpath) != m['code']['source_seal_sha256']:
        raise PermissionError('TASK_CAPTURE_SOURCE_SEAL_CHANGED')
    seal = read(sealpath)
    if (seal['source_sha'] != m['code']['vla_commit']
            or Path(__file__).resolve() != code/'scripts/pi0294/task_capture_runtime.py'):
        raise PermissionError('TASK_CAPTURE_SEALED_SOURCE_REQUIRED')
    for name, digest in seal['files'].items():
        if sha(within(code/name, code)) != digest:
            raise PermissionError('TASK_CAPTURE_SOURCE_MEMBER_CHANGED:'+name)
    if (root/'NATIVE_SAFETY_FAIL.json').exists() and not read_only:
        raise PermissionError('TASK_CAPTURE_NATIVE_SAFETY_FAIL_PRESERVED')
    return m, parent


@contextlib.contextmanager
def cpu_stage(root, stage, *, cores=1, cpu_s_cap=600):
    """Engineering CPU only; physical/model dispatch needs separate tickets."""
    import resource
    from scripts.pi0292.durable import OwnerLock, write
    root = Path(root)
    manifest, parent = validate_entry(root, read_only=True)
    started, wall = time.process_time(), time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with_lock = OwnerLock(parent/'R4_RESOURCE_OWNER.lock')
    ledger_path = parent/'R4_RESOURCE_LEDGER.json'
    ledger = read(ledger_path)
    active = ledger['CPU_reservations']
    key = RUN+'/'+stage
    if key in active or any(x['stage'] == key for x in ledger['CPU_stages']):
        raise FileExistsError('TASK_STAGE_ALREADY_ATTEMPTED')
    used = {c for x in active.values() for c in x['cpus']}
    affinity = sorted(os.sched_getaffinity(0))
    cpus = [c for c in affinity if c not in used][-16:][:cores]
    if (not 1 <= cores <= 4 or len(cpus) != cores or len(used)+cores > 16
            or ledger['Core_CPU_s']+sum(x['cpu_s_cap'] for x in active.values())+cpu_s_cap > 96*3600):
        raise PermissionError('CUMULATIVE_CPU_CAP')
    ticket = dict(stage=key, kind='TASK_ENGINEERING_CPU', run_uuid=RUN,
        pid=os.getpid(), proc_start=Path('/proc/self/stat').read_text().split()[21],
        cpus=cpus, cores=cores, cpu_s_cap=cpu_s_cap, started_epoch=time.time())
    active[key] = ticket
    write(ledger_path, ledger)
    with_lock.close()
    os.sched_setaffinity(0, set(cpus))
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    limit = int(time.process_time()+cpu_s_cap)
    resource.setrlimit(resource.RLIMIT_CPU, (min(limit,hard) if hard != resource.RLIM_INFINITY else limit,hard))
    write(root/'evidence'/f'{stage}-CPU_TICKET.json', ticket)
    status = 'FAILED_PRESERVED'
    try:
        yield manifest
        status = 'COMPLETE'
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cost = time.process_time()-started+after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
        owner = OwnerLock(parent/'R4_RESOURCE_OWNER.lock')
        ledger = read(ledger_path)
        ledger['CPU_reservations'].pop(key)
        ledger['Core_CPU_s'] += cost
        receipt = dict(ticket, status=status, finished_epoch=time.time(), actual_CPU_s=cost,
            service_s=time.monotonic()-wall, new_physics=0, new_neural_queries=0)
        ledger['CPU_stages'].append(receipt)
        write(ledger_path, ledger)
        write(root/'evidence'/f'{stage}-CPU_RECEIPT.json', receipt)
        owner.close()
