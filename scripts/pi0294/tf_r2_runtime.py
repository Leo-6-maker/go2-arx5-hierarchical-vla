"""Explicit TF-R2 successor; historical R1 authority and capsules stay immutable."""
import contextlib,os,resource,socket,time
from pathlib import Path
from .task_capture_runtime import read,sha,within
from scripts.pi0292.durable import OwnerLock,write
RELEASE='13c72eca98d682280fa67ec316b909f1aa1fdd5b'
RUN='20261016T032943Z-tf-r2-closure-to-dataset'
T0=1792121383.494512
CUTOFF=1792146583.494512
DEADLINE=1792150183.494512
FOLDER='factory'
VERSION='TF_R2_V1'


def resource_lock(root):
    end=time.monotonic()+5
    while True:
        try:return OwnerLock(Path(root)/'R2_RESOURCE_OWNER.lock')
        except BlockingIOError:
            if time.monotonic()>=end:raise
            time.sleep(.01)


def validate_entry(root,*,read_only=False,closeout=False,physical_asset=None):
    root=Path(root).resolve();a=read(root/'R2_ADOPTION_AND_BUDGET.json');m=read(root/'RUN_MANIFEST.json')
    if (root.name!=RUN or m['run_uuid']!=RUN or not a['authorizes_execution'] or a['adopted_release']!=RELEASE
        or (a['T0_epoch'],a['new_work_cutoff_epoch'],a['deadline_epoch'])!=(T0,CUTOFF,DEADLINE)
        or not T0<=time.time()<(DEADLINE if read_only or closeout else CUTOFF)
        or m['authority']['adoption_sha256']!=sha(root/'R2_ADOPTION_AND_BUDGET.json')
        or a['resource_caps_override']['cumulative_CPU_starts_integrations_RGB_search_caps_removed'] is not True):
        raise PermissionError('R2_AUTHORITY_WINDOW')
    e=m['execution'];b=read(root/'OUTPUT_BINDING.private.json')
    if (socket.gethostname() not in e['hostnames'] or os.getuid()!=e['uid']
        or os.environ.get('GR00T_WBC_PYTHON')!=e['profile_python'] or sha(e['entry_path'])!=e['entry_sha256']
        or sha(root/'OUTPUT_BINDING.private.json')!=e['output_binding_sha256']):raise PermissionError('R2_HOST_PROFILE')
    within(root,b['registered_parent']);reg=read(b['registry']);lease=read(b['lease'])
    if (not any(x['role']==b['role'] and Path(x['path']).resolve()==root for x in reg['entries'])
        or lease['status']!='ACTIVE' or lease['run_uuid']!=RUN or lease['owner']!=m['owner']
        or lease['deadline_epoch']!=DEADLINE or lease['adoption_sha256']!=sha(root/'R2_ADOPTION_AND_BUDGET.json')):raise PermissionError('R2_REGISTRY_LEASE')
    code=within(root/m['code']['capsule_relative'],root);sealpath=within(root/m['code']['source_seal_relative'],root);seal=read(sealpath)
    if (sha(sealpath)!=m['code']['source_seal_sha256'] or seal['source_sha']!=m['code']['vla_commit']
        or Path(__file__).resolve()!=code/'scripts/pi0294/tf_r2_runtime.py'):raise PermissionError('R2_CAPSULE_IDENTITY')
    for n,h in seal['files'].items():
        if sha(within(code/n,code))!=h:raise PermissionError('R2_SOURCE_CHANGED:'+n)
    if not read_only and not closeout:
        if (root/'TASK_PHYSICS_STOP.json').exists():raise PermissionError('R2_ACTUAL_SAFETY_STOP_PRESERVED')
        if physical_asset:
            q=read(root/'evidence/STARTUP_QUALIFICATION.json')
            if q['status']!='PASS_TWO_SAFE_COMPLETE_STARTS' or any(r['module_sha256']!=sha(code/'scripts/pi0294/night_r1_startup.py') for r in q['rows']):raise PermissionError('R2_QUALIFIED_STARTUP_IDENTITY')
    return m,root


@contextlib.contextmanager
def cpu_scope(root, stage, *, cpu_s_cap=1800, cores=1, closeout=False, include_children=True):
    root = Path(root)
    if not closeout and time.time()>=CUTOFF:raise PermissionError('R1_CPU_NEW_WORK_CUTOFF')
    # A safety stop closes physics; the adopted R1 explicitly keeps bounded
    # CPU reference/data engineering and evidence closeout available.
    m, _ = validate_entry(root, read_only=True, closeout=closeout)
    t0, wall = time.process_time(), time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    lock = resource_lock(root)
    path = root/'R2_RESOURCE_LEDGER.json'; v = read(path)
    if stage in v['CPU_reservations'] or any(x['stage'] == stage for x in v['CPU_stages']):
        lock.close(); raise FileExistsError('R1_FIRST_STAGE_PRESERVED')
    used = {c for t in v['CPU_reservations'].values() for c in t['cpus']}
    cpus = [c for c in sorted(os.sched_getaffinity(0)) if c not in used][-8:][:cores]
    limit = None
    reserve = 0 if closeout else 6*3600
    if (not 1 <= cores <= 4 or len(cpus) != cores or len(used)+cores > 8
           ):
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
