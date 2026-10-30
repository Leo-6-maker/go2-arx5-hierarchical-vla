"""Existing registered owner's finite predeclared source sequence and closeout.

One fixed task pair or both pairs, no retries or external-site runtime calls. Each child
and the separate raw auditor use the same standard entry and one night account.
"""
import hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
from scripts.pi0294.dec1 import source_cpu_once_io as once
from . import night_scope as scope

raw=lambda v:(json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
sha=lambda b:hashlib.sha256(b).hexdigest()


def put(path,value):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,'wb')as f:f.write(raw(value));f.flush();os.fsync(f.fileno())


def reserve_audit(root,side,task_kind):
    """Charge the separate auditor before its process, on the same account."""
    if side not in('left','right')or task_kind not in('BOOK','ROBOCASA_CABINET'):
        raise ValueError('FIXED_SOURCE_AUDITOR_IDENTITY')
    path=Path(root)/'control/NIGHT_SOURCE_ACCOUNT.json';before=path.read_bytes();v=json.loads(before)
    if not v['native_adopted']:raise PermissionError('ORIGINAL_NATIVE_SOURCE_ACCOUNT_REQUIRED')
    op=('BOOK_'if task_kind=='BOOK'else'CABINET_')+side.upper()+'_INDEPENDENT_RAW_AUDIT'
    if task_kind=='ROBOCASA_CABINET':
        from .cabinet_repair02 import checked_card,attempt_operation
        fixed=json.loads((Path(root)/'FIXED_TASK_CARDS.json').read_bytes())
        card=(fixed[task_kind]if task_kind in fixed else fixed)[side]
        if checked_card(card,side):op=attempt_operation(card,side)+'_INDEPENDENT_RAW_AUDIT'
    w=scope.reserve(v,operation=op,kind='PREPARATION',upper=dict(Core_CPU_s=40),now=time.time())
    key='NIGHT30-'+op
    return once.commit(path,expected_before_sha256=sha(before),proposed_bytes=raw(w),request_id=key,
        intent_path=path.with_name(key+'.INTENT.json'),receipt_path=path.with_name(key+'.RECEIPT.json'),
        lock_path=path.with_name('NIGHT_SOURCE_ACCOUNT.owner.lock'))


def planned_tasks(task_kind):
    if task_kind=='BOOK':return(('BOOK','left'),('BOOK','right'))
    if task_kind=='ROBOCASA_CABINET':return(('ROBOCASA_CABINET','right'),('ROBOCASA_CABINET','left'))
    if task_kind=='ALL_PINNED':return planned_tasks('BOOK')+planned_tasks('ROBOCASA_CABINET')
    raise ValueError('FIXED_SOURCE_CHAIN_KIND')


def checked_plan(root,task_kind):
    """Reject incomplete mixed-task staging before the first child effect."""
    plan=planned_tasks(task_kind);root=Path(root)
    fixed=json.loads((root/'FIXED_TASK_CARDS.json').read_bytes())
    from .libero_book_cards import cards as book_cards
    for kind,side in plan:
        selected=(fixed[kind]if kind in fixed else fixed)[side]
        if kind=='ROBOCASA_CABINET':
            from .cabinet_repair02 import checked_card
            checked_card(selected,side)
        elif selected!=book_cards()[side]:raise PermissionError('ONE_SEALED_PREDECLARED_TASK_CARD_PER_CHILD')
    if any(kind=='ROBOCASA_CABINET'for kind,_ in plan):
        assets=json.loads((root/'ASSET_BINDINGS.private.json').read_bytes())
        definitions=assets.get('robocasa_definition_members',{});textures=assets.get('robocasa_texture_members',{})
        if (set(definitions)!={'cabinet_single','slab','bar_handle'}or set(textures)!={'wood','metal'}
                or any(key not in assets['files']for key in[*definitions.values(),*textures.values()])):
            raise PermissionError('ALL_MIXED_TASK_ASSET_BINDINGS_BEFORE_FIRST_CHILD')
    return plan


def main(root,*,task_kind='BOOK'):
    planned_tasks(task_kind)
    root=Path(root);start=time.monotonic()
    from .night_scope import for_root
    scope=for_root(root)
    if os.getuid()!=2009 or os.uname().nodename!='corelab-4080s':raise PermissionError('EXISTING_CORE_OWNER_ONLY')
    scope.authority((root/'NIGHT_HUMAN_APPROVAL.json').read_bytes(),now=time.time())
    plan=checked_plan(root,task_kind)
    entry=root/'entry/infrastructure/enter.sh'
    unit_facts=dict(pid=os.getpid(),process_start_ticks=Path('/proc/self/stat').read_text().split()[21],
        cgroup=Path('/proc/self/cgroup').read_text(),uid=os.getuid(),host=os.uname().nodename,
        pc_or_external_site_runtime_dependency=False,predeclared_tasks=[dict(task_kind=k,side=s)for k,s in plan],task_kind=task_kind,automatic_retry=False,
        deadline_epoch=scope.DEADLINE,observed_epoch=time.time())
    put(root/'control/ACTUAL_DETACHED_OWNER_STARTED.json',unit_facts)
    results=[]
    for kind,side in plan:
        worker='night_book_worker'if kind=='BOOK'else'night_cabinet_worker'
        auditor='night_book_audit'if kind=='BOOK'else'night_cabinet_audit'
        prefix='LIBERO_'if kind=='BOOK'else'ROBOCASA_'
        if time.time()>=scope.STOP_NEW or (root/'STOP.json').exists():break
        command=['/bin/bash',str(entry),'4080-simulation','python','-B','-m',
            'scripts.pi0294.multitask.'+worker,str(root),side]
        with (root/'logs'/(prefix+side+'.log')).open('xb')as log:
            try:
                child=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=min(600,scope.DEADLINE-time.time()-30))
                status=child.returncode
            except subprocess.TimeoutExpired:
                status='WALL_TIMEOUT_NO_RETRY'
        folder=root/'outputs'/(prefix+side.upper()+'_FIRST01')
        if kind=='ROBOCASA_CABINET':
            from .cabinet_repair02 import attempt_operation
            fixed=json.loads((root/'FIXED_TASK_CARDS.json').read_bytes())
            card=(fixed[kind]if kind in fixed else fixed)[side]
            folder=root/'outputs'/attempt_operation(card,side)
        row=dict(task_kind=kind,side=side,OS_exit_status=status,raw_terminal_exists=(folder/'TERMINAL.json').exists())
        if row['raw_terminal_exists']:
            # A distinct process independently reads the actual saved raw bytes.
            audit_code='import sys,json,resource;resource.setrlimit(resource.RLIMIT_CPU,(30,30));from pathlib import Path;from scripts.pi0294.multitask.'+auditor+' import audit;print(json.dumps(audit(Path(sys.argv[1]))))'
            try:
                reserve_audit(root,side,kind)
                q=subprocess.run(['/bin/bash',str(entry),'4080-simulation','python','-B','-c',audit_code,str(folder)],
                    stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=30)
                row['audit_OS_exit_status']=q.returncode
                if q.returncode==0:
                    value=json.loads(q.stdout);put(folder/'INDEPENDENT_RAW_AUDIT.json',value)
                    row.update(independent_audit_sha256=sha(raw(value)),task_success=value['task_success'],
                        qualified_source_pilot=value['qualified_source_pilot'])
                else:row['audit_error']=q.stderr[-1200:]
            except Exception as exc:
                row.update(audit_OS_exit_status='FAILED_NO_RETRY',audit_error=type(exc).__name__+': '+str(exc))
        results.append(row)
        # Same engineering fault is never blindly submitted to the right task.
        if status!=0 or row.get('audit_OS_exit_status')!=0:break
    lease_path=root/'control/NATIVE_SOURCE_LEASE.json';b=lease_path.read_bytes();lease=json.loads(b)
    if lease['status']!='ACTIVE' or lease['run_uuid']!=root.name:raise PermissionError('EXACT_OWN_SOURCE_LEASE_CLOSE')
    lease.update(status='CLOSED',terminal_reason='FINITE_PREDECLARED_SOURCE_TASKS_TERMINAL_NO_RETRY',closed_epoch=time.time(),all_children_terminal=True)
    key='NIGHT30-FINITE-SOURCE-CHAIN-CLOSE'
    receipt=once.commit(lease_path,expected_before_sha256=sha(b),proposed_bytes=raw(lease),request_id=key,
        intent_path=lease_path.with_name(key+'.INTENT.json'),receipt_path=lease_path.with_name(key+'.RECEIPT.json'),
        lock_path=lease_path.with_name('NATIVE_SOURCE_LEASE.owner.lock'))
    usage=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
    value=dict(status='FINITE_SOURCE_CHAIN_TERMINAL',results=results,closed_lease_sha256=receipt['after_sha256'],
        qualified_source_pilots=sum(r.get('qualified_source_pilot',False)for r in results),
        CPU_s=usage.ru_utime+usage.ru_stime+children.ru_utime+children.ru_stime,wall_s=time.monotonic()-start,
        pc_or_external_site_runtime_dependency=False,automatic_retry=False,new_model_queries_updates_PBS=0,observed_epoch=time.time())
    put(root/'outputs/SOURCE_CHAIN_TERMINAL.json',value);print(json.dumps(value))


if __name__=='__main__':main(sys.argv[1])
