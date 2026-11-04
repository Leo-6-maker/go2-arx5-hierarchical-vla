"""Finite two-slot native batch under one existing physical owner, no queue."""
from pathlib import Path
import fcntl,json,os,subprocess,time
from contextlib import contextmanager
from .storage import digest,write_json


def validate_layout(batch, children):
    if len(children)!=2 or len(batch['slots'])!=2:raise PermissionError('EXACT_TWO_SLOT_PILOT_REQUIRED')
    domains=[];cpus=set();roots=set()
    for slot,child in zip(batch['slots'],children):
        root=Path(child['registered_root']).resolve()
        if (root.parent!=Path(batch['campaign_root'])/'runs' or str(root)in roots
                or child.get('parallel_batch_spec')!=str(Path(batch['registered_root'])/'EXECUTION_SPEC.json')
                or child['campaign_root']!=batch['campaign_root']
                or child['physical_owner_lock']!=batch['physical_owner_lock']
                or child['source_sha']!=batch['source_sha']
                or child['ROS_DOMAIN_ID']not in (220,221)
                or not set(child['cpus'])or cpus.intersection(child['cpus'])
                or child['phase']not in ('cold-1','g2-standing-hold-short-v2')
                or child['max_ticks']not in (50,2528)
                or digest(slot['spec'])!=slot['spec_sha256']):
            raise PermissionError('ISOLATED_PINNED_FINITE_TARGET_SLOTS_REQUIRED')
        domains.append(child['ROS_DOMAIN_ID']);cpus.update(child['cpus']);roots.add(str(root))
    if len(set(domains))!=2 or not cpus<=set(batch['cpus']):raise PermissionError('DISTINCT_DOMAINS_AND_AVAILABLE_CPUS_REQUIRED')


def process_start(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]


def child_environment(child,fds):
    env=dict(os.environ,**child['native_environment'])
    env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=child['source_root']+'/src:'+child['source_root']+':'+env.get('PYTHONPATH',''),WM02_AUTO1_OWNER_FDS=json.dumps(fds),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    return env


def verify_inherited_owner(fd,path):
    """An inherited locked open-file description, not a permission boolean."""
    expected=os.stat(path);actual=os.fstat(fd)
    if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):raise PermissionError('OWNER_FD_PATH_MISMATCH')
    # An independent open must contend. The inherited description must succeed.
    with Path(path).open('a')as probe:
        try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:raise PermissionError('CANONICAL_OWNER_LOCK_NOT_HELD')
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)


@contextmanager
def inherited_batch_owner(spec_path,spec):
    batch_path=Path(spec['parallel_batch_spec']);batch=json.loads(batch_path.read_text())
    root=Path(batch['registered_root']);lease=json.loads(Path(batch['lease']).read_text());started=json.loads((root/'batch-01/INHERITED_OWNER.private.json').read_text())
    from .auto1_native import validate_scope
    campaign=Path(spec['campaign_root']);validate_scope(batch,json.loads((campaign/'control/LEASE.json').read_text()),json.loads(Path(spec['registry']).read_text()),host=os.uname().nodename,uid=os.getuid(),now=time.time())
    if (lease['status']!='ACTIVE'or lease['spec_sha256']!=digest(batch_path)
            or started['batch_spec_sha256']!=digest(batch_path)
            or started['pid']!=os.getppid()or started['process_start']!=process_start(os.getppid())
            or Path(f'/proc/{os.getppid()}').stat().st_uid!=os.getuid()):raise PermissionError('ACTUAL_LIVE_REGISTERED_BATCH_PARENT_REQUIRED')
    children=[json.loads(Path(s['spec']).read_text())for s in batch['slots']];validate_layout(batch,children)
    if not any(s['spec']==str(spec_path)and s['spec_sha256']==digest(spec_path)for s in batch['slots']):raise PermissionError('CHILD_NOT_PINNED_IN_BATCH')
    binding=json.loads(Path(spec['campaign_binding']).read_text())
    if binding.get('max_target_instances')!=2 or binding.get('owner_model')!='ONE_PARENT_INHERITED_CANONICAL_LOCKS':raise PermissionError('REGISTERED_TWO_INSTANCE_SCOPE_REQUIRED')
    fds=json.loads(os.environ.get('WM02_AUTO1_OWNER_FDS','[]'))
    if len(fds)!=2 or fds!=started['fds']:raise PermissionError('TRUSTED_INHERITED_OWNER_FDS_REQUIRED')
    for fd,path in zip(fds,(campaign/'OWNER.lock',spec['physical_owner_lock'])):verify_inherited_owner(fd,path)
    try:yield
    finally:
        for fd in fds:os.close(fd)


def run_batch(spec_path,output,run_id):
    from .auto1_native import validate_scope
    from .runtime import admitted_run
    from scripts.pi0292.durable import OwnerLock,write
    spec=json.loads(Path(spec_path).read_text());campaign=Path(spec['campaign_root'])
    validate_scope(spec,json.loads((campaign/'control/LEASE.json').read_text()),json.loads(Path(spec['registry']).read_text()),host=os.uname().nodename,uid=os.getuid(),now=time.time())
    children=[json.loads(Path(s['spec']).read_text())for s in spec['slots']];validate_layout(spec,children)
    with admitted_run(spec_path,run_id,'wm02-auto1-target-batch',output,physics=True):
        campaign_owner=OwnerLock(campaign/'OWNER.lock')
        try:
            physical_owner=OwnerLock(spec['physical_owner_lock'])
            try:
                fds=[campaign_owner.handle.fileno(),physical_owner.handle.fileno()]
                write_json(Path(output)/'INHERITED_OWNER.private.json',dict(pid=os.getpid(),process_start=process_start(os.getpid()),fds=fds,batch_spec_sha256=digest(spec_path)))
                jobs=[];handles=[];start=time.monotonic()
                try:
                    for slot,child in zip(spec['slots'],children):
                        env=child_environment(child,fds)
                        argv=[child['python'],'-B','-m','scripts.wm02.commission_target','--spec',slot['spec'],'--output',str(Path(child['registered_root'])/'native-01'),'--run-id','native-01']
                        h=(Path(output)/(Path(child['registered_root']).name+'.log')).open('x');handles.append(h)
                        jobs.append((child,subprocess.Popen(argv,cwd=child['source_root'],env=env,stdout=h,stderr=subprocess.STDOUT,pass_fds=fds)))
                    while any(p.poll()is None for _,p in jobs):
                        if time.time()>=spec['deadline_epoch']or(campaign/'STOP.json').exists()or(Path(spec['registered_root'])/'STOP.json').exists():
                            for child,p in jobs:
                                if p.poll()is None:write(Path(child['registered_root'])/'STOP.json',dict(reason='BATCH_FINITE_STOP'))
                        time.sleep(.05)
                    rows=[]
                    for child,p in jobs:
                        result=Path(child['registered_root'])/'native-01/RESULT.json';value=json.loads(result.read_text())if result.exists()else{}
                        lease_path=Path(child['lease']);lease=json.loads(lease_path.read_text());lease.update(status='CLOSED',closed_epoch=time.time(),worker_exit_code=p.returncode);write(lease_path,lease)
                        rows.append(dict(run=Path(child['registered_root']).name,exit_code=p.returncode,status=value.get('status'),ticks=value.get('actual_target_integrations',0),shutdown_clean=value.get('shutdown_clean',False),ROS_DOMAIN_ID=child['ROS_DOMAIN_ID'],cpus=child['cpus']))
                    result=dict(status='TWO_ENVIRONMENT_NATIVE_PILOT_PASSED'if all(r['exit_code']==0 and r['status']=='NATIVE_STAGE_PASSED'and r['shutdown_clean']for r in rows)else'TWO_ENVIRONMENT_NATIVE_PILOT_FAILED_PRESERVED',children=rows,wall_s=time.monotonic()-start,physical_owners=1,WM_updates=0,Actor_updates=0,learning_atoms=0)
                    write_json(Path(output)/'RESULT.json',result);return result
                finally:
                    for child,p in jobs:
                        if p.poll()is None:
                            write(Path(child['registered_root'])/'STOP.json',dict(reason='BATCH_EXCEPTION_CLOSEOUT'))
                            try:p.wait(timeout=45)
                            except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30)
                    for h in handles:h.close()
                    for child in children:
                        owned=next((p for c,p in jobs if c is child),None)
                        if owned is None or owned.poll()is not None:
                            path=Path(child['lease']);lease=json.loads(path.read_text())
                            if lease['status']=='ACTIVE':
                                lease.update(status='CLOSED',closed_epoch=time.time(),worker_exit_code=None if owned is None else owned.returncode,started=owned is not None);write(path,lease)
            finally:physical_owner.close()
        finally:campaign_owner.close()
