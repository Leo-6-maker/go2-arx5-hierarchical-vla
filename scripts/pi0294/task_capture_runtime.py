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


def validate_entry(root, *, profile='4080-simulation', now=None, closeout=False, read_only=False, physical_asset=None):
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
    if not read_only:
        if physical_asset is not None:
            require_task_asset(root,m,physical_asset)
        elif (root/'NATIVE_SAFETY_FAIL.json').exists():
            raise PermissionError('TASK_CAPTURE_NATIVE_SAFETY_FAIL_PRESERVED')
    return m, parent


def require_task_asset(root,manifest,version,*,_adopted_teacher=False):
    """V1 stays stopped. One separately sealed task-gripper reset contract V2."""
    root=Path(root)
    if version=='FULLBODY_MANISKILL_SEARCH_AND_ASSISTED_SEED_V2':
        from .mps2_runtime import require_adoption
        require_adoption(root,manifest)
        return
    if version=='FULLBODY_MANISKILL_COMPONENT_SEED_V1':
        from .mp_runtime import require_adoption
        require_adoption(root,manifest)
        return
    if version=='FULLBODY_COMPONENT_PROGRESS_QUALIFICATION_V2':
        from .component_qualification import require_adoption
        require_adoption(root,manifest)
        return
    if version=='FULLBODY_PERSISTENT_TEACHER_QUALIFICATION_V1':
        from .teacher_qualification import require_adoption
        require_adoption(root,manifest)
        return
    if version!='FULLBODY_TASK_OVERLAY_V2_WORKING_STROKE':
        raise PermissionError('UNKNOWN_NATIVE_TASK_ASSET')
    if (root/'NATIVE_SAFETY_FAIL_V2.json').exists():
        raise PermissionError('TASK_V2_NATIVE_SAFETY_FAIL_PRESERVED')
    if _adopted_teacher:
        from .teacher_qualification import AUTHORITY_SHA256,OLD_TERMINAL_SHA256
        if (sha(root/'teacher-qualification-v1/HUMAN_ADOPTION.json')!=AUTHORITY_SHA256
                or sha(root/'TASK_PILOT_CLOSED.json')!=OLD_TERMINAL_SHA256):
            raise PermissionError('NEW_TEACHER_EXPLICIT_ADOPTION_NOT_BOUND')
    if (root/'TASK_PILOT_CLOSED.json').exists() and not _adopted_teacher:
        raise PermissionError('FIXED_TASK_PILOTS_TERMINAL_SUCCESS_DATA_NOT_ADMITTED')
    legacy=root/'NATIVE_SAFETY_FAIL.json'
    if sha(legacy)!='175ae9cdab3d108e4826968c5ffb3724a34ec069a3ed1f909cf82fafb6a39a71':
        raise PermissionError('ORIGINAL_TASK_V1_FAILURE_IDENTITY_REQUIRED')
    p=read(root/'evidence/TASK_V2_ENGINEERING_ACCEPTANCE.json')
    if (p['status']!='PASS_CPU_RESET_COMMAND_AND_FROZEN_CORE_CONTRACT_ONLY'
            or p['version']!=version or p['physical_qualification'] is not False
            or p['original_failure_sha256']!=sha(legacy)
            or p['mechanical_travel_m']!=.044 or p['working_margin_m']!=.04*.05
            or p['WBC_changes']!=0 or p['safety_threshold_changes']!=0):
        raise PermissionError('TASK_V2_EXACT_ENGINEERING_RECEIPT_REQUIRED')
    code=root/manifest['code']['capsule_relative']
    expected={'task_gripper_contract.py','task_plant.py','task_actuation.cpp','task_native_worker.py'}
    if set(p['source_files'])!=expected:raise PermissionError('TASK_V2_SOURCE_SET')
    timing_path=root/'evidence/TASK_TIMING_ENGINEERING_ACCEPTANCE.json'
    timing=None
    if timing_path.exists():
        timing=read(timing_path)
        amendment=root/'evidence/TASK_TIMING_ENGINEERING_ACCEPTANCE_R2.json'
        if amendment.exists():
            amended=read(amendment)
            if amended['previous_timing_acceptance_sha256']!=sha(timing_path):raise PermissionError('TIMING_AMENDMENT_PARENT_CHANGED')
            for key in ('clock_build_receipt','task_build_receipt','base_acceptance_sha256'):
                if amended[key]!=timing[key]:raise PermissionError('TIMING_AMENDMENT_BINARY_OR_PLANT_CHANGED')
            timing=amended
        r3=root/'evidence/TASK_TIMING_ENGINEERING_ACCEPTANCE_R3.json'
        if r3.exists():
            amended=read(r3)
            if amended['previous_timing_acceptance_sha256']!=sha(amendment):raise PermissionError('TIMING_R3_PARENT_CHANGED')
            for key in ('clock_build_receipt','task_build_receipt','base_acceptance_sha256'):
                if amended[key]!=timing[key]:raise PermissionError('TIMING_R3_BINARY_OR_PLANT_CHANGED')
            timing=amended
        if (timing['status']!='PASS_CPU_EXACT_STEP_PAUSE_ONLY' or timing['physical_qualification'] is not False
                or timing['base_acceptance_sha256']!=sha(root/'evidence/TASK_V2_ENGINEERING_ACCEPTANCE.json')
                or timing['safety_threshold_changes'] or timing['controller_math_changes']
                or timing['source_age_limit_s']!=.025):raise PermissionError('TASK_TIMING_SOURCE_CONTRACT')
        for name,digest in timing['source_files'].items():
            if sha(code/'scripts/pi0294'/name)!=digest:raise PermissionError('TASK_TIMING_SOURCE_CHANGED:'+name)
        for key in ('clock_build_receipt','task_build_receipt','CPU_receipt'):
            row=timing[key]
            if sha(within(root/row['relative'],root))!=row['sha256']:raise PermissionError('TASK_TIMING_RECEIPT_CHANGED')
    for name,digest in p['source_files'].items():
        if timing and name in ('task_actuation.cpp','task_native_worker.py'):digest=timing['source_files'][name]
        if sha(code/'scripts/pi0294'/name)!=digest:raise PermissionError('TASK_V2_SOURCE_CHANGED:'+name)
    for key in ('plant_cpu_receipt','build_receipt','contract_cpu_receipt'):
        row=p[key]
        if sha(within(root/row['relative'],root))!=row['sha256']:raise PermissionError('TASK_V2_MEMBER_CHANGED')


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
