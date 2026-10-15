"""Independent, explicitly adopted overnight scope. Never reopen an old scope."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import time

from .task_capture_runtime import read, sha, within
from scripts.pi0292.durable import OwnerLock, write

FOLDER = 'factory'
VERSION = 'TF_NIGHT_V1'
CUTOFF = 1792112400
DEADLINE = 1792116000
RELEASE = '354e45abe2aa11107ffd99339c34f1ab9f58f1ea'


def validate_entry(root, *, read_only=False, closeout=False, physical_asset=None):
    root = Path(root).resolve()
    a = read(root/'NIGHT_ADOPTION.json'); m = read(root/'RUN_MANIFEST.json')
    now = time.time()
    if (not a['authorizes_execution'] or a['adopted_release'] != RELEASE
            or a['run_uuid'] != root.name or m['run_uuid'] != root.name
            or m['scope'] != 'PI0294_TF_NIGHT_20261016'
            or m['authority']['adoption_sha256'] != sha(root/'NIGHT_ADOPTION.json')
            or a['new_work_cutoff_epoch'] != CUTOFF or a['deadline_epoch'] != DEADLINE
            or not a['T0_epoch'] <= now < (DEADLINE if closeout or read_only else CUTOFF)):
        raise PermissionError('NIGHT_EXPLICIT_AUTHORITY_OR_DEADLINE')
    e = m['execution']; b = read(root/'OUTPUT_BINDING.private.json')
    if (socket.gethostname() not in e['hostnames'] or os.getuid() != e['uid']
            or os.environ.get('GR00T_WBC_PYTHON') != e['profile_python']
            or sha(e['entry_path']) != e['entry_sha256']
            or sha(root/'OUTPUT_BINDING.private.json') != e['output_binding_sha256']):
        raise PermissionError('NIGHT_HOST_PROFILE_OR_OUTPUT')
    within(root, b['registered_parent'])
    registry, lease = read(b['registry']), read(b['lease'])
    if not any(x['role'] == b['role'] and Path(x['path']).resolve() == root for x in registry['entries']):
        raise PermissionError('NIGHT_OUTPUT_NOT_REGISTERED')
    if (lease['run_uuid'] != root.name or lease['owner'] != m['owner']
            or lease['adoption_sha256'] != sha(root/'NIGHT_ADOPTION.json')
            or lease['status'] != 'ACTIVE' or lease['deadline_epoch'] != DEADLINE):
        raise PermissionError('NIGHT_OWNER_LEASE')
    code = within(root/m['code']['capsule_relative'], root)
    sealpath = within(root/m['code']['source_seal_relative'], root)
    seal = read(sealpath)
    if (sha(sealpath) != m['code']['source_seal_sha256'] or seal['source_sha'] != m['code']['vla_commit']
            or Path(__file__).resolve() != code/'scripts/pi0294/night_runtime.py'):
        raise PermissionError('NIGHT_IMMUTABLE_SOURCE')
    for name, digest in seal['files'].items():
        if sha(within(code/name, code)) != digest:
            raise PermissionError('NIGHT_SOURCE_CHANGED:'+name)
    old = Path(b['prior_run'])
    for rel, digest in read(root/'PRIOR_EVIDENCE.json')['members'].items():
        if sha(old/rel) != digest:
            raise PermissionError('NIGHT_PRIOR_EVIDENCE_CHANGED:'+rel)
    if not read_only:
        ledger = read(root/'NIGHT_RESOURCE_LEDGER.json')
        if (root/'GOAL_CLOSED.json').exists() or ledger['native_safety_failures'] >= 2:
            raise PermissionError('NIGHT_TERMINAL_OR_SECOND_SAFETY_FAILURE')
        if physical_asset and physical_asset != VERSION:
            raise PermissionError('NIGHT_UNKNOWN_TEACHER_VERSION')
    return m, root


def require_adoption(root, manifest=None, *, read_only=False):
    m, _ = validate_entry(root, read_only=read_only)
    return read(Path(root)/'NIGHT_ADOPTION.json'), read(Path(root)/FOLDER/'SERIES_LEDGER.json')


def cards(root):
    return read(Path(root)/FOLDER/'CARDS.json')['cards']


def require_card(root, manifest, episode):
    validate_entry(root, physical_asset=VERSION)
    card = next(c for c in cards(root) if c['id'] == episode)
    release = read(Path(root)/FOLDER/'TEACHER_RELEASE.json')
    if (release['version'] != card['teacher_version'] or release['source_sha'] != manifest['code']['vla_commit']
            or release['CPU_admission'] != 'PASS' or release.get('quarantined', False)):
        raise PermissionError('NIGHT_ACTUAL_CHANGED_CONTRACT_ADMISSION_REQUIRED')
    if card['role'] == 'PRODUCTION' and not release['tasks'][card['task']]['repeatability_PASS']:
        raise PermissionError('NIGHT_TWO_OF_THREE_REPEATABILITY_REQUIRED')
    return card


@contextlib.contextmanager
def cpu_scope(root, stage, *, cpu_s_cap=1800, cores=1, closeout=False):
    root = Path(root); m, _ = validate_entry(root, read_only=closeout)
    t0, wall = time.process_time(), time.monotonic(); before = resource.getrusage(resource.RUSAGE_CHILDREN)
    lock = OwnerLock(root/'NIGHT_RESOURCE_OWNER.lock'); file = root/'NIGHT_RESOURCE_LEDGER.json'; v = read(file)
    if stage in v['CPU_reservations'] or any(x['stage'] == stage for x in v['CPU_stages']):
        lock.close(); raise FileExistsError('NIGHT_FIRST_STAGE_PRESERVED')
    used = {c for x in v['CPU_reservations'].values() for c in x['cpus']}
    cpus = [c for c in sorted(os.sched_getaffinity(0)) if c not in used][-12:][:cores]
    reserve = 0 if closeout else 6*3600
    if (not 1 <= cores <= 4 or len(cpus) != cores or len(used)+cores > 12
            or v['Core_CPU_s']+sum(x['cpu_s_cap'] for x in v['CPU_reservations'].values())+cpu_s_cap+reserve > 64*3600):
        lock.close(); raise PermissionError('NIGHT_CPU_CLOSEOUT_RESERVE')
    ticket = dict(stage=stage,pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],cpus=cpus,cores=cores,cpu_s_cap=cpu_s_cap,started_epoch=time.time())
    v['CPU_reservations'][stage] = ticket; write(file,v); lock.close()
    affinity = os.sched_getaffinity(0); os.sched_setaffinity(0,cpus)
    status = 'INCOMPLETE'
    try:
        yield m
        status = 'COMPLETE'
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        actual = time.process_time()-t0+after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
        charge = max(actual, cores*(time.monotonic()-wall))
        proof = dict(ticket,status=status,actual_CPU_s=actual,conservative_CPU_charged_s=charge)
        lock = OwnerLock(root/'NIGHT_RESOURCE_OWNER.lock');v=read(file);v['CPU_reservations'].pop(stage);v['Core_CPU_s']+=charge;v['CPU_stages'].append(proof);write(file,v);lock.close()
        write(root/'evidence'/(stage+'-CPU_RECEIPT.json'),proof);os.sched_setaffinity(0,affinity)


def condition(root, name):
    return read(Path(root)/FOLDER/'CONDITIONS.json')['conditions'][name]


def reserve_bundle(root, name, index, identity):
    root=Path(root);validate_entry(root);c=condition(root,name)
    if not 0<=index<c['decision_slots']:raise PermissionError('NIGHT_CONDITION_BUNDLE_CAP')
    lock=OwnerLock(root/'NIGHT_RESOURCE_OWNER.lock')
    try:
        file=root/FOLDER/'SEARCH_LEDGER.json';v=read(file);key=name+'/'+str(index)
        if key in v['bundles']:
            if v['bundles'][key]['input_sha256']!=identity:raise ValueError('NIGHT_FIRST_INPUT_CHANGED')
        else:
            if len(v['bundles'])>=2000:raise PermissionError('NIGHT_BUNDLE_CAP')
            v['bundles'][key]=dict(input_sha256=identity,slots={},started_epoch=time.time());write(file,v)
    finally:lock.close()
    return c


def reserve_internal(root,key,kind,slot):
    root=Path(root);c=condition(root,key.split('/')[0]);lock=OwnerLock(root/'NIGHT_RESOURCE_OWNER.lock')
    try:
        if time.time()>=CUTOFF:raise PermissionError('NIGHT_WORK_CUTOFF')
        file=root/FOLDER/'SEARCH_LEDGER.json';v=read(file);b=v['bundles'][key]
        if kind+'/'+str(slot) in b['slots']:raise PermissionError('NIGHT_INTERNAL_SLOT_NO_REDRAW')
        limits={'IK':c.get('max_new_goal_IK_calls',12),'RRT':8,'DENSE':c.get('max_dense_windows',4)}
        costs={'IK':{'goal_IK_calls':1,'CLIK_reserved':16},'RRT':{'RRT_calls':1},'DENSE':{'dense_reserved_points':1001,'local_IK_reserved':5005}}
        caps={'goal_IK_calls':24000,'CLIK_reserved':384000,'RRT_calls':8000,'dense_reserved_points':8000000,'local_IK_reserved':40000000}
        if not 0<=slot<limits[kind]:raise PermissionError('NIGHT_BUNDLE_'+kind+'_CAP')
        for k,n in costs[kind].items():
            if v['counters'].get(k,0)+n>caps[k]:raise PermissionError('NIGHT_GLOBAL_'+k)
        for k,n in costs[kind].items():v['counters'][k]=v['counters'].get(k,0)+n
        b['slots'][kind+'/'+str(slot)]=dict(state='RESERVED_NO_REDRAW',capacity=costs[kind],reserved_epoch=time.time());write(file,v)
    finally:lock.close()


def finish_internal(root,key,kind,slot,actual):
    root=Path(root);lock=OwnerLock(root/'NIGHT_RESOURCE_OWNER.lock')
    try:
        file=root/FOLDER/'SEARCH_LEDGER.json';v=read(file);r=v['bundles'][key]['slots'][kind+'/'+str(slot)]
        if r['state']=='COMPLETE':
            if r['actual']!=actual:raise ValueError('NIGHT_FIRST_INTERNAL_RESULT_CHANGED')
        else:r.update(state='COMPLETE',actual=actual,finished_epoch=time.time());write(file,v)
    finally:lock.close()
