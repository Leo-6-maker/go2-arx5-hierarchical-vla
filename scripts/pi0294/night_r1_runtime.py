"""R1 successor admission. Closed parent runs and original night totals are immutable."""
import contextlib
import os
from pathlib import Path
import resource
import socket
import time

from .task_capture_runtime import read, sha, within
from scripts.pi0292.durable import OwnerLock, write

RELEASE = 'a9c8e29c2ceba50d06fe4d3b705f9661d7bb2b16'
CUTOFF = 1792112400
DEADLINE = 1792116000
FOLDER = 'factory'
VERSION = 'TF_NIGHT_R1_V1'


def resource_lock(root):
    """Serialize short ledger mutations; a parallel CPU ticket is not a second owner."""
    end=time.monotonic()+5
    while True:
        try:return OwnerLock(Path(root)/'R1_RESOURCE_OWNER.lock')
        except BlockingIOError:
            if time.monotonic()>=end:raise
            time.sleep(.01)


def validate_entry(root, *, read_only=False, closeout=False, physical_asset=None):
    root = Path(root).resolve()
    a, m = read(root/'R1_ADOPTION.json'), read(root/'RUN_MANIFEST.json')
    now = time.time()
    if (not a['authorizes_execution'] or a['adopted_release'] != RELEASE
            or a['run_uuid'] != root.name or m['run_uuid'] != root.name
            or m['scope'] != 'PI0294_TF_NIGHT_R1_20261016'
            or m['authority']['adoption_sha256'] != sha(root/'R1_ADOPTION.json')
            or a['T0_epoch'] != 1792081407.219026
            or a['new_work_cutoff_epoch'] != CUTOFF or a['deadline_epoch'] != DEADLINE
            or not a['R1_adopted_epoch'] <= now < (DEADLINE if read_only or closeout else CUTOFF)):
        raise PermissionError('R1_EXPLICIT_AUTHORITY_OR_INHERITED_WINDOW')
    e, b = m['execution'], read(root/'OUTPUT_BINDING.private.json')
    if (socket.gethostname() not in e['hostnames'] or os.getuid() != e['uid']
            or os.environ.get('GR00T_WBC_PYTHON') != e['profile_python']
            or sha(e['entry_path']) != e['entry_sha256']
            or sha(root/'OUTPUT_BINDING.private.json') != e['output_binding_sha256']):
        raise PermissionError('R1_HOST_PROFILE_OUTPUT_BINDING')
    within(root, b['registered_parent'])
    registry, lease = read(b['registry']), read(b['lease'])
    if not any(x['role'] == b['role'] and Path(x['path']).resolve() == root for x in registry['entries']):
        raise PermissionError('R1_OUTPUT_NOT_REGISTERED')
    if (lease['run_uuid'] != root.name or lease['owner'] != m['owner']
            or lease['adoption_sha256'] != sha(root/'R1_ADOPTION.json')
            or lease['status'] != 'ACTIVE' or lease['deadline_epoch'] != DEADLINE):
        raise PermissionError('R1_OWNER_LEASE')
    code = within(root/m['code']['capsule_relative'], root)
    sealpath = within(root/m['code']['source_seal_relative'], root)
    seal = read(sealpath)
    if (sha(sealpath) != m['code']['source_seal_sha256'] or seal['source_sha'] != m['code']['vla_commit']
            or Path(__file__).resolve() != code/'scripts/pi0294/night_r1_runtime.py'):
        raise PermissionError('R1_IMMUTABLE_SOURCE')
    for path, digest in seal['files'].items():
        if sha(within(code/path, code)) != digest:
            raise PermissionError('R1_SOURCE_CHANGED:'+path)
    old = Path(b['prior_night_run'])
    for path, digest in read(root/'PRIOR_NIGHT_EVIDENCE.json')['members'].items():
        if sha(old/path) != digest:
            raise PermissionError('R1_PARENT_CHANGED:'+path)
    ledger = read(root/'R1_RESOURCE_LEDGER.json')
    if (ledger['parent_CPU_upper_s'] != a['parent_CPU_conservative_charged_upper_s']
            or ledger['parent_native_safety_failures'] != 1
            or a['effective_R1_CPU_cap_s'] > min(32*3600, 64*3600-ledger['parent_CPU_upper_s'])):
        raise PermissionError('R1_PARENT_BALANCE_OR_SAFETY_RESET')
    if not read_only and not closeout:
        if (root/'R1_SHARED_SAFETY_STOP.json').exists() or (root/'GOAL_CLOSED.json').exists():
            raise PermissionError('R1_TERMINAL_PRESERVED')
        if ledger['native_safety_failures'] + ledger['parent_native_safety_failures'] >= 2:
            raise PermissionError('R1_SECOND_NIGHT_SAFETY_FAILURE')
        if physical_asset:
            diff = read(root/'evidence/STARTUP_LIFECYCLE_DIFF.json')
            if not diff.get('allowlisted_change_supported_by_evidence'):
                raise PermissionError('R1_STARTUP_NO_EVIDENCE_BOUND_WRAPPER_CHANGE')
            if physical_asset != 'STARTUP':
                q = read(root/'evidence/STARTUP_QUALIFICATION.json')
                if q.get('status') != 'PASS_TWO_SAFE_COMPLETE_STARTS':
                    raise PermissionError('R1_TWO_STARTUPS_REQUIRED')
    return m, root


@contextlib.contextmanager
def cpu_scope(root, stage, *, cpu_s_cap=1800, cores=1, closeout=False, include_children=True):
    root = Path(root); m, _ = validate_entry(root, closeout=closeout)
    t0, wall = time.process_time(), time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    lock = resource_lock(root)
    path = root/'R1_RESOURCE_LEDGER.json'; v = read(path)
    if stage in v['CPU_reservations'] or any(x['stage'] == stage for x in v['CPU_stages']):
        lock.close(); raise FileExistsError('R1_FIRST_STAGE_PRESERVED')
    used = {c for t in v['CPU_reservations'].values() for c in t['cpus']}
    cpus = [c for c in sorted(os.sched_getaffinity(0)) if c not in used][-8:][:cores]
    limit = read(root/'R1_ADOPTION.json')['effective_R1_CPU_cap_s']
    reserve = 0 if closeout else 6*3600
    if (not 1 <= cores <= 4 or len(cpus) != cores or len(used)+cores > 8
            or v['Core_CPU_s']+sum(x['cpu_s_cap'] for x in v['CPU_reservations'].values())+cpu_s_cap+reserve > limit):
        lock.close(); raise PermissionError('R1_CPU_BALANCE_AND_CLOSEOUT_RESERVE')
    ticket = dict(stage=stage,pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],
                  cpus=cpus,cores=cores,cpu_s_cap=cpu_s_cap,started_epoch=time.time())
    v['CPU_reservations'][stage]=ticket;write(path,v);lock.close()
    affinity=os.sched_getaffinity(0);os.sched_setaffinity(0,cpus);status='INCOMPLETE'
    try:
        yield m
        status='COMPLETE'
    finally:
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        actual=time.process_time()-t0
        if include_children:actual+=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
        charge=max(actual,cores*(time.monotonic()-wall))
        proof=dict(ticket,status=status,actual_CPU_s=actual,conservative_CPU_charged_s=charge)
        lock=resource_lock(root);v=read(path);v['CPU_reservations'].pop(stage)
        v['Core_CPU_s']+=charge;v['CPU_stages'].append(proof);write(path,v);lock.close()
        write(root/'evidence'/(stage+'-CPU_RECEIPT.json'),proof);os.sched_setaffinity(0,affinity)
